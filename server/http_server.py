"""mikeos-storyteller-cloud HTTP server.

The MikeOS storyteller cloud: it turns an ORDERED list of POIs (the route.pois
MikeGuide produces along a route) into a duration-matched, spoken-word ROAD-TRIP
STORY generated on the FREE GPU (qwen3:8b), and keeps a per-user library of the
stories it has told.

Pacing: target spoken length = minutes * ~150 words/min, distributed across the
POIs proportional to each POI's `dwell` weight (a landmark you linger at gets a
longer segment), with a small intro + outro. One narrated segment per POI, in
drive-past order, each with an est_seconds ~= words/150*60.

Never-trust-the-GPU: if the GPU returns empty/unparseable, it retries once then
falls back to a deterministic template so the endpoint ALWAYS returns usable
segments (marked model='fallback'). It never returns an empty segments array
with 200.

Identity: apps present X-API-KEY: <hive agent key>, resolved to a user_id via
mikeoscomputers (the IdP). Every story is scoped to that user_id. /api/health is
keyless.

Endpoints:
  GET  /api/health              -> {status, db, version}                       (no auth)
  POST /api/story               -> generate (or cache-hit) a story
  GET  /api/story/{id}          -> one story (scoped to the user)
  GET  /api/stories?limit=20    -> the user's story library
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from pydantic import BaseModel, Field

from server import citystory, gpu, story, tts, worker
from server.identity import resolve_agent_key
from server.storage.postgres_manager import get_postgres_manager

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

VERSION = "1.3.0"

# Strong references to fire-and-forget background tasks (road-trip audio prewarm).
# Without this, asyncio may GC a task mid-flight (it only keeps a weak ref), silently
# cancelling the prewarm. We discard each task from the set when it finishes.
_bg_tasks: set = set()


def _spawn_bg(coro) -> None:
    task = asyncio.ensure_future(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class POIIn(BaseModel):
    name: str
    cat: Optional[str] = None
    order: Optional[int] = None
    dwell: Optional[float] = None
    blurb: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


class StoryIn(BaseModel):
    trip_id: Optional[str] = None
    dest: Optional[str] = None
    minutes: int = Field(..., ge=1, le=180)
    pois: List[POIIn]


# --- Async city-story engine (MikeStoryteller app) ---
class CityStoryRequest(BaseModel):
    city: str = Field(..., min_length=1)
    lat: Optional[float] = None
    lon: Optional[float] = None
    theme: str = Field(...)
    minutes: int = Field(...)


class BaselineRequest(BaseModel):
    """Kick a ~10-story baseline for a city.

    The MikeStoryteller app sends the CURRENT city it wants a library for as
    `city` (its whole model is "keep ~10 ready for the city Mike is in now").
    `home_city` / `current_city` are accepted for back-compat and to let the app
    ALSO seed a lighter set for a second city in one call. At least one of
    `city` / `home_city` / `current_city` must be present (validated in the
    handler so a missing field is a clean 422 with a useful message)."""
    city: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    home_city: Optional[str] = None
    home_lat: Optional[float] = None
    home_lon: Optional[float] = None
    current_city: Optional[str] = None
    current_lat: Optional[float] = None
    current_lon: Optional[float] = None


async def _segments_with_audio(
    db, story_id: str, segments: List[Dict[str, Any]], lang: str
) -> List[Dict[str, Any]]:
    """Attach a relative `audio_url` (+ best-effort `audio_ready`) to each road-trip
    segment so the app downloads narration exactly like a city-story chapter.

    A segment is ALWAYS fetchable via its audio_url (the proxy synthesizes on
    demand); `audio_ready` just reports whether the tts cache is already warm.
    """
    audio_map: Dict[int, Dict[str, Any]] = {}
    if tts.is_configured():
        try:
            audio_map = await db.get_segment_audio_map(story_id, lang)
        except Exception:  # noqa: BLE001 - reporting only, never fail the response
            audio_map = {}
    out: List[Dict[str, Any]] = []
    for seg in (segments or []):
        order = seg.get("order")
        a = audio_map.get(order) if order is not None else None
        enriched = dict(seg)
        enriched["audio_url"] = (
            f"/api/story/{story_id}/segment/{order}/audio" if order is not None else None
        )
        enriched["audio_ready"] = bool(a and a.get("status") == "ready")
        if a and a.get("duration_sec"):
            enriched["audio_duration_sec"] = a["duration_sec"]
        out.append(enriched)
    return out


def _validate_theme_minutes(theme: str, minutes: int) -> tuple[str, int]:
    theme = (theme or "").strip().lower()
    if theme not in citystory.VALID_THEMES:
        raise HTTPException(
            status_code=422,
            detail=f"theme must be one of {list(citystory.VALID_THEMES)}",
        )
    if int(minutes) not in citystory.VALID_MINUTES:
        raise HTTPException(
            status_code=422,
            detail=f"minutes must be one of {list(citystory.VALID_MINUTES)}",
        )
    return theme, int(minutes)


# ---------------------------------------------------------------------------
# Auth dependency: resolve X-API-KEY -> user_id
# ---------------------------------------------------------------------------
async def current_user_id(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-KEY"),
) -> str:
    """Resolve the caller's hive agent key to a user_id, or 401."""
    user_id = await resolve_agent_key(x_api_key)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-KEY")
    return user_id


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting mikeos-storyteller-cloud...")
    try:
        db = await get_postgres_manager()
        if await db.ping():
            logger.info("Database connection established")
        else:
            logger.error("Database connection failed!")
        await db.run_migrations()
    except Exception as e:
        logger.error("Startup error: %s", e)
        raise
    if not gpu.is_configured():
        logger.warning(
            "OLLAMA_GPU_URL is not set — stories will use the deterministic fallback "
            "template until it is configured."
        )
    # Start the SINGLE background story-generation worker (serialized generation).
    stop_event = asyncio.Event()
    worker_task = asyncio.create_task(worker.run_worker(stop_event))
    app.state.worker_stop = stop_event
    app.state.worker_task = worker_task
    logger.info("Background city-story worker started")

    yield

    logger.info("Shutting down mikeos-storyteller-cloud...")
    try:
        stop_event.set()
        await asyncio.wait_for(worker_task, timeout=10.0)
    except Exception as e:
        logger.error("Worker shutdown error: %s", e)
    try:
        db = await get_postgres_manager()
        await db.close()
    except Exception as e:
        logger.error("Shutdown error: %s", e)


app = FastAPI(
    title="mikeos-storyteller-cloud",
    description="Duration-matched, spoken-word road-trip stories from route POIs, on the free GPU, for MikeOS.",
    version=VERSION,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Root / health
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "service": "mikeos-storyteller-cloud",
        "health": "/api/health",
        "endpoints": {
            "make_story": "POST /api/story",
            "get_story": "GET /api/story/{id}",
            "list_stories": "GET /api/stories?limit=20",
            "request_city_story": "POST /api/stories/request",
            "story_status": "GET /api/stories/status/{job_id}",
            "city_story": "GET /api/story/{story_id}",
            "chapter_audio": "GET /api/story/{story_id}/chapter/{index}/audio",
            "segment_audio": "GET /api/story/{story_id}/segment/{order}/audio",
            "city_library": "GET /api/city-stories?city=&theme=&minutes=",
            "baseline": "POST /api/stories/baseline",
        },
    }


@app.get("/api/health")
async def health():
    db_ok = False
    try:
        db = await get_postgres_manager()
        db_ok = await db.ping()
    except Exception as e:
        logger.error("health db check failed: %s", e)
    return {
        "status": "healthy" if db_ok else "degraded",
        "db": "up" if db_ok else "down",
        "gpu": "configured" if gpu.is_configured() else "unset",
        "version": VERSION,
    }


# ---------------------------------------------------------------------------
# Stories
# ---------------------------------------------------------------------------
@app.post("/api/story")
async def make_story(
    body: StoryIn,
    user_id: str = Depends(current_user_id),
) -> Dict[str, Any]:
    """Generate (or cache-hit) a duration-matched road-trip story.

    1. Cache: if trip_id is given and a story for (user_id, trip_id, minutes)
       exists, return it with cached=true.
    2. Pace + generate on the GPU (dwell-weighted word budget), chunked so no
       single call is huge.
    3. Persist and return the story with cached=false. Segments are NEVER empty
       with a 200 (falls back to a deterministic template, model='fallback').
    """
    if not body.pois:
        raise HTTPException(status_code=422, detail="pois must be a non-empty list")

    db = await get_postgres_manager()

    # 1. Cache check.
    if body.trip_id:
        cached = await db.find_cached_story(user_id, body.trip_id, body.minutes)
        if cached is not None:
            logger.info(
                "cache hit for trip_id=%s minutes=%d -> story %s",
                body.trip_id,
                body.minutes,
                cached["id"],
            )
            return {
                "story_id": cached["id"],
                "trip_id": cached["trip_id"],
                "dest": cached["dest"],
                "minutes": cached["minutes"],
                "poi_count": cached["poi_count"],
                "total_words": cached["total_words"],
                "model": cached["model"],
                "audio_available": tts.is_configured(),
                "segments": await _segments_with_audio(
                    db, cached["id"], cached["segments"], "en"
                ),
                "cached": True,
            }

    # 2. Generate.
    pois = [p.model_dump(exclude_none=False) for p in body.pois]
    logger.info(
        "generating story: %d POIs, %d min target (~%d words), trip_id=%s",
        len(pois),
        body.minutes,
        body.minutes * story.WORDS_PER_MIN,
        body.trip_id,
    )
    result = await story.generate_story(pois, body.minutes, dest=body.dest)
    segments = result["segments"]

    # Never trust a 200 with nothing in it.
    if not segments:
        raise HTTPException(status_code=502, detail="story generation produced no segments")

    # 3. Persist.
    saved = await db.create_story(
        user_id,
        trip_id=body.trip_id,
        dest=body.dest,
        minutes=body.minutes,
        poi_count=len(pois),
        segments=segments,
        total_words=result["total_words"],
        model=result["model"],
    )
    if not saved or not saved.get("id"):
        raise HTTPException(status_code=500, detail="failed to persist story")

    logger.info(
        "story %s stored: %d segments, %d words, model=%s",
        saved["id"],
        len(segments),
        result["total_words"],
        result["model"],
    )
    # Pre-warm the tts.osmike.com cache for each segment in the BACKGROUND so the
    # app's first audio fetch is instant. A bonus AFTER the story is saved — a tts
    # hiccup never delays or fails this response (the proxy still synths on demand).
    if tts.is_configured():
        _spawn_bg(worker.prewarm_roadtrip_audio(saved["id"], user_id))
    return {
        "story_id": saved["id"],
        "trip_id": saved["trip_id"],
        "dest": saved["dest"],
        "minutes": saved["minutes"],
        "poi_count": saved["poi_count"],
        "total_words": saved["total_words"],
        "model": saved["model"],
        "audio_available": tts.is_configured(),
        "segments": await _segments_with_audio(db, saved["id"], saved["segments"], "en"),
        "cached": False,
    }


@app.get("/api/story/{story_id}")
async def get_story(
    story_id: str,
    user_id: str = Depends(current_user_id),
) -> Dict[str, Any]:
    """Full story by id.

    First tries the SHARED city_stories cache (the MikeStoryteller contract:
    a flat {story_id, city, theme, minutes, title, total_words, sources,
    chapters:[...]} shape). If not a city story, falls back to the user-scoped
    road-trip story ({"story": row}).
    """
    db = await get_postgres_manager()
    city = await db.get_city_story(story_id)
    if city is not None:
        lang = tts.normalize_lang(city.get("lang"))
        # Per-chapter audio state (pre-warmed by the worker). A chapter is
        # ALWAYS fetchable via its audio_url (the proxy synths on demand); the
        # audio_ready flag just tells the app the tts cache is already warm.
        audio_map = await db.get_chapter_audio_map(city["id"], lang)
        chapters = []
        for ch in (city["chapters"] or []):
            idx = ch.get("index")
            a = audio_map.get(idx) if idx is not None else None
            enriched = dict(ch)
            enriched["audio_url"] = (
                f"/api/story/{city['id']}/chapter/{idx}/audio" if idx is not None else None
            )
            enriched["audio_ready"] = bool(a and a.get("status") == "ready")
            if a and a.get("duration_sec"):
                enriched["audio_duration_sec"] = a["duration_sec"]
            chapters.append(enriched)
        return {
            "story_id": city["id"],
            "city": city["city"],
            "theme": city["theme"],
            "minutes": city["minutes"],
            "title": city["title"],
            "total_words": city["total_words"],
            "sources": city["sources"],
            "source_book": city["source_book"],
            "model": city["model"],
            "lang": lang,
            "audio_available": tts.is_configured(),
            "created_at": city["created_at"],
            "chapters": chapters,
        }
    row = await db.get_story(user_id, story_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Story not found")
    return {"story": row}


@app.get("/api/story/{story_id}/chapter/{index}/audio")
async def get_chapter_audio(
    story_id: str,
    index: int,
    user_id: str = Depends(current_user_id),
):
    """Stream a chapter's spoken MP3 (the MikeStoryteller app audio contract).

    Auth via X-API-KEY (same resolve as every other endpoint — the app NEVER
    holds the TTS token). Looks up the chapter text, POSTs it to tts.osmike.com
    with the SERVER-SIDE TTS_TOKEN (cache hit -> fast), and returns the MP3 with
    `Content-Type: audio/mpeg`. On a tts failure returns a clean 502/503 — never
    a fake 200 with an empty body.
    """
    if not tts.is_configured():
        raise HTTPException(status_code=503, detail="audio (TTS) is not configured on this server")

    db = await get_postgres_manager()
    city = await db.get_city_story(story_id)
    if city is None:
        raise HTTPException(status_code=404, detail="Story not found")

    chapters = city.get("chapters") or []
    ch = next((c for c in chapters if c.get("index") == index), None)
    if ch is None and 0 <= index < len(chapters):
        ch = chapters[index]  # tolerate stories whose chapters lack an index
    if ch is None:
        raise HTTPException(status_code=404, detail="Chapter not found")

    text = (ch.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=404, detail="Chapter has no text")

    lang = tts.normalize_lang(city.get("lang"))
    try:
        data, cache_hit = await tts.synthesize(text, lang)
    except tts.TTSError as e:
        logger.warning("chapter audio %s/%d tts failed: %s", story_id, index, e)
        # Record the failure state (best-effort) and return a clean 502.
        try:
            await db.upsert_chapter_audio(
                story_id=story_id, chapter_index=index, lang=lang,
                status="failed", error=str(e)[:500],
            )
        except Exception:  # noqa: BLE001
            pass
        raise HTTPException(status_code=502, detail="text-to-speech synthesis failed")

    # Record ready state so status/get-story can report audio_ready (best-effort).
    try:
        await db.upsert_chapter_audio(
            story_id=story_id, chapter_index=index, lang=lang, status="ready",
            byte_len=len(data), duration_sec=tts.estimate_duration_sec(len(data)),
        )
    except Exception:  # noqa: BLE001
        pass

    return Response(
        content=data,
        media_type="audio/mpeg",
        headers={
            "Content-Length": str(len(data)),
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=86400",
            "X-TTS-Cache": "hit" if cache_hit else "miss",
        },
    )


@app.get("/api/story/{story_id}/segment/{order}/audio")
async def get_segment_audio(
    story_id: str,
    order: int,
    user_id: str = Depends(current_user_id),
):
    """Stream a road-trip SEGMENT's spoken MP3 (the app's road-trip audio contract).

    The road-trip analogue of the chapter-audio proxy. Road-trip stories are
    user-scoped (`stories` table), so this is looked up per user_id, its segment
    matched by `order`, its text POSTed to tts.osmike.com with the SERVER-SIDE
    TTS_TOKEN (cache hit -> fast), and returned as audio/mpeg. The app NEVER holds
    the TTS token. On a tts failure returns a clean 502/503 — never a fake 200.
    """
    if not tts.is_configured():
        raise HTTPException(status_code=503, detail="audio (TTS) is not configured on this server")

    db = await get_postgres_manager()
    story = await db.get_story(user_id, story_id)
    if story is None:
        raise HTTPException(status_code=404, detail="Story not found")

    segments = story.get("segments") or []
    seg = next((s for s in segments if s.get("order") == order), None)
    if seg is None and 0 <= order < len(segments):
        seg = segments[order]  # tolerate segments whose order is positional
    if seg is None:
        raise HTTPException(status_code=404, detail="Segment not found")

    text = (seg.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=404, detail="Segment has no text")

    lang = tts.normalize_lang(story.get("lang"))
    try:
        data, cache_hit = await tts.synthesize(text, lang)
    except tts.TTSError as e:
        logger.warning("segment audio %s/%d tts failed: %s", story_id, order, e)
        try:
            await db.upsert_segment_audio(
                story_id=story_id, segment_order=order, lang=lang,
                status="failed", error=str(e)[:500],
            )
        except Exception:  # noqa: BLE001
            pass
        raise HTTPException(status_code=502, detail="text-to-speech synthesis failed")

    try:
        await db.upsert_segment_audio(
            story_id=story_id, segment_order=order, lang=lang, status="ready",
            byte_len=len(data), duration_sec=tts.estimate_duration_sec(len(data)),
        )
    except Exception:  # noqa: BLE001
        pass

    return Response(
        content=data,
        media_type="audio/mpeg",
        headers={
            "Content-Length": str(len(data)),
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=86400",
            "X-TTS-Cache": "hit" if cache_hit else "miss",
        },
    )


@app.get("/api/stories")
async def list_stories(
    limit: int = Query(default=20, ge=1, le=200),
    user_id: str = Depends(current_user_id),
) -> Dict[str, Any]:
    """The user's road-trip story library (lightweight rows), newest first."""
    db = await get_postgres_manager()
    stories = await db.list_stories(user_id, limit=limit)
    return {"stories": stories, "count": len(stories)}


# ---------------------------------------------------------------------------
# Async DEEP CITY-STORY engine (MikeStoryteller app contract)
# ---------------------------------------------------------------------------
@app.post("/api/stories/request")
async def request_city_story(
    body: CityStoryRequest,
    user_id: str = Depends(current_user_id),
) -> Dict[str, Any]:
    """Request a deep city story.

    Cache hit (a READY shared story for city_key/theme/minutes exists) -> return
    {status:"ready", story_id} immediately, no GPU. Else create/find a queued
    job and return {status:"queued", job_id}. Generation is async + background.
    """
    theme, minutes = _validate_theme_minutes(body.theme, body.minutes)
    ck = citystory.city_key(body.city)
    db = await get_postgres_manager()

    ready = await db.find_ready_city_story(ck, theme, minutes)
    if ready is not None:
        return {"status": "ready", "story_id": ready["id"]}

    existing = await db.find_open_job(ck, theme, minutes)
    if existing is not None:
        return {"status": existing["status"], "job_id": existing["id"]}

    job = await db.create_job(
        user_id=user_id,
        city=body.city.strip(),
        city_key=ck,
        theme=theme,
        minutes=minutes,
        lat=body.lat,
        lon=body.lon,
    )
    if not job or not job.get("id"):
        raise HTTPException(status_code=500, detail="failed to enqueue story job")
    return {"status": "queued", "job_id": job["id"]}


@app.get("/api/stories/status/{job_id}")
async def story_status(
    job_id: str,
    user_id: str = Depends(current_user_id),
) -> Dict[str, Any]:
    """Poll a job: {status, story_id?, error?}."""
    db = await get_postgres_manager()
    job = await db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    out: Dict[str, Any] = {"status": job["status"]}
    if job.get("story_id"):
        out["story_id"] = job["story_id"]
    if job.get("error"):
        out["error"] = job["error"]
    return out


@app.get("/api/city-stories")
async def city_story_library(
    city: Optional[str] = Query(default=None),
    theme: Optional[str] = Query(default=None),
    minutes: Optional[int] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    user_id: str = Depends(current_user_id),
) -> Dict[str, Any]:
    """READY shared city stories matching any subset of filters (library feed)."""
    ck = citystory.city_key(city) if city else None
    theme_norm = (theme or "").strip().lower() or None
    db = await get_postgres_manager()
    rows = await db.list_city_stories(
        city_key=ck, theme=theme_norm, minutes=minutes, limit=limit
    )
    return {"stories": rows, "count": len(rows)}


@app.post("/api/stories/baseline")
async def baseline(
    body: BaselineRequest,
    user_id: str = Depends(current_user_id),
) -> Dict[str, Any]:
    """Enqueue a ~10-story baseline so a city's library fills over time.

    The PRIMARY city (the app's `city`, i.e. the city Mike is in now — falling
    back to `home_city`/`current_city`) gets the full varied ~10-job plan. If a
    DISTINCT second city is also supplied, it gets a lighter starter plan in the
    same call. Idempotent: a (city_key,theme,minutes) that is already ready OR
    has an open job is NOT re-enqueued, so re-running creates 0 duplicates.
    """
    # The primary city to baseline: `city` (the app's contract) first, then the
    # legacy home/current fields. A request with none of them is a clean 422.
    primary_city = (body.city or body.home_city or body.current_city or "").strip()
    if not primary_city:
        raise HTTPException(
            status_code=422,
            detail="baseline needs a city (send {\"city\": \"<name>\"})",
        )
    primary_lat = body.lat if body.city else (body.home_lat if body.home_city else body.current_lat)
    primary_lon = body.lon if body.city else (body.home_lon if body.home_city else body.current_lon)

    db = await get_postgres_manager()

    # Full varied ~10-job plan for the PRIMARY city (matches the app's target of
    # ~10 ready stories per city: themes across mixed lengths).
    primary_plan = [
        ("complete", 20),
        ("history", 30),
        ("history", 10),
        ("people", 20),
        ("people", 10),
        ("architecture", 10),
        ("art", 10),
        ("nature", 10),
        ("food", 10),
        ("food", 5),
    ]
    # A lighter starter plan for a DISTINCT second city, if the app named one.
    secondary_plan = [
        ("complete", 10),
        ("history", 20),
        ("food", 5),
    ]

    targets: List[Dict[str, Any]] = []
    for theme, minutes in primary_plan:
        targets.append({
            "city": primary_city, "lat": primary_lat, "lon": primary_lon,
            "theme": theme, "minutes": minutes,
        })
    # A second, DISTINCT city: prefer current_city, else home_city, whichever
    # differs from the primary. (Lets one baseline call warm home + current.)
    second_city = None
    second_lat = second_lon = None
    for cand, clat, clon in (
        (body.current_city, body.current_lat, body.current_lon),
        (body.home_city, body.home_lat, body.home_lon),
    ):
        if cand and citystory.city_key(cand) != citystory.city_key(primary_city):
            second_city, second_lat, second_lon = cand, clat, clon
            break
    if second_city:
        for theme, minutes in secondary_plan:
            targets.append({
                "city": second_city, "lat": second_lat, "lon": second_lon,
                "theme": theme, "minutes": minutes,
            })

    jobs: List[Dict[str, Any]] = []
    for t in targets:
        ck = citystory.city_key(t["city"])
        theme, minutes = t["theme"], t["minutes"]
        ready = await db.find_ready_city_story(ck, theme, minutes)
        if ready is not None:
            jobs.append({"status": "ready", "story_id": ready["id"],
                         "city": t["city"], "theme": theme, "minutes": minutes})
            continue
        existing = await db.find_open_job(ck, theme, minutes)
        if existing is not None:
            jobs.append({"status": existing["status"], "job_id": existing["id"],
                         "city": t["city"], "theme": theme, "minutes": minutes})
            continue
        job = await db.create_job(
            user_id=user_id, city=t["city"].strip(), city_key=ck,
            theme=theme, minutes=minutes, lat=t["lat"], lon=t["lon"],
        )
        jobs.append({"status": "queued", "job_id": job["id"],
                     "city": t["city"], "theme": theme, "minutes": minutes})

    return {"enqueued": sum(1 for j in jobs if j["status"] == "queued"), "jobs": jobs}
