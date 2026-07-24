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
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from server import gpu, story
from server.identity import resolve_agent_key
from server.storage.postgres_manager import get_postgres_manager

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

VERSION = "1.0.0"


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
    yield
    logger.info("Shutting down mikeos-storyteller-cloud...")
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
                "segments": cached["segments"],
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
    return {
        "story_id": saved["id"],
        "trip_id": saved["trip_id"],
        "dest": saved["dest"],
        "minutes": saved["minutes"],
        "poi_count": saved["poi_count"],
        "total_words": saved["total_words"],
        "model": saved["model"],
        "segments": saved["segments"],
        "cached": False,
    }


@app.get("/api/story/{story_id}")
async def get_story(
    story_id: str,
    user_id: str = Depends(current_user_id),
) -> Dict[str, Any]:
    """One story (scoped to the user)."""
    db = await get_postgres_manager()
    row = await db.get_story(user_id, story_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Story not found")
    return {"story": row}


@app.get("/api/stories")
async def list_stories(
    limit: int = Query(default=20, ge=1, le=200),
    user_id: str = Depends(current_user_id),
) -> Dict[str, Any]:
    """The user's story library (lightweight rows), newest first."""
    db = await get_postgres_manager()
    stories = await db.list_stories(user_id, limit=limit)
    return {"stories": stories, "count": len(stories)}
