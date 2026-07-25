"""Background story-generation worker for mikeos-storyteller-cloud.

A single asyncio task started on app lifespan. It polls story_jobs for the
oldest `queued` job, claims it (`generating`), runs the DEEP city-story pipeline
(server/citystory.py), and marks it `ready` (linking the shared city_stories
row) or `failed`.

Generation is SERIALIZED — exactly one story is generated at a time. The free
GPU must never be flooded (we just fixed a 503-storm), and deep stories are slow
by design; we have time. `claim_next_job` uses FOR UPDATE SKIP LOCKED so this is
safe even if more than one worker ever ran.

Cache-aware: before generating, it re-checks the shared cache (another job may
have produced the same city/theme/length while this one waited) and, on
success, resolves ALL other open jobs for the same key to the same story
(dedupe — one generation serves everyone).
"""
import asyncio
import logging

from server import citystory, tts
from server.storage.postgres_manager import get_postgres_manager

logger = logging.getLogger(__name__)

POLL_INTERVAL_SEC = 5.0


async def _prewarm_audio(story_id: str) -> None:
    """PRE-WARM the tts.osmike.com cache for every chapter of a ready story.

    A BONUS pass, run AFTER the story is already marked ready — a tts hiccup must
    NEVER fail (or block) the story. Serialized per-chapter (respect the box).
    Records per-chapter synth state in chapter_audio so status can report it.
    Never-trust-200 is enforced inside tts.synthesize (audio/mpeg + real bytes).
    """
    if not tts.is_configured():
        logger.info("audio prewarm skipped for %s: TTS_TOKEN not configured", story_id)
        return
    db = await get_postgres_manager()
    story = await db.get_city_story(story_id)
    if story is None:
        return
    lang = tts.normalize_lang(story.get("lang"))
    chapters = story.get("chapters") or []
    ok = 0
    for ch in chapters:
        idx = ch.get("index")
        text = (ch.get("text") or "").strip()
        if idx is None or not text:
            continue
        try:
            data, cache_hit = await tts.synthesize(text, lang)
            await db.upsert_chapter_audio(
                story_id=story_id,
                chapter_index=int(idx),
                lang=lang,
                status="ready",
                byte_len=len(data),
                duration_sec=tts.estimate_duration_sec(len(data)),
            )
            ok += 1
            logger.info(
                "audio prewarm %s ch%s: %d bytes (cache=%s)",
                story_id, idx, len(data), "hit" if cache_hit else "miss",
            )
        except Exception as e:  # noqa: BLE001 - audio is a bonus, never fatal
            logger.warning("audio prewarm %s ch%s FAILED: %s", story_id, idx, e)
            try:
                await db.upsert_chapter_audio(
                    story_id=story_id, chapter_index=int(idx), lang=lang,
                    status="failed", error=str(e)[:500],
                )
            except Exception:  # noqa: BLE001
                pass
    logger.info("audio prewarm %s done: %d/%d chapters ready", story_id, ok, len(chapters))


async def prewarm_roadtrip_audio(story_id: str, user_id: str) -> None:
    """PRE-WARM the tts.osmike.com cache for every segment of a road-trip story.

    The road-trip analogue of [_prewarm_audio]: a bonus pass run in the
    background AFTER `POST /api/story` has already returned, so a tts hiccup
    never delays (or fails) the story. Road-trip stories live in the per-user
    `stories` table (segments, not chapters), so it reads via get_story and
    records per-segment synth state in `segment_audio`. Serialized per segment
    (respect the box). Never raises.
    """
    if not tts.is_configured():
        logger.info("roadtrip audio prewarm skipped for %s: TTS_TOKEN not configured", story_id)
        return
    db = await get_postgres_manager()
    story = await db.get_story(user_id, story_id)
    if story is None:
        return
    lang = tts.normalize_lang(story.get("lang"))
    segments = story.get("segments") or []
    ok = 0
    for seg in segments:
        order = seg.get("order")
        text = (seg.get("text") or "").strip()
        if order is None or not text:
            continue
        try:
            data, cache_hit = await tts.synthesize(text, lang)
            await db.upsert_segment_audio(
                story_id=story_id,
                segment_order=int(order),
                lang=lang,
                status="ready",
                byte_len=len(data),
                duration_sec=tts.estimate_duration_sec(len(data)),
            )
            ok += 1
            logger.info(
                "roadtrip audio prewarm %s seg%s: %d bytes (cache=%s)",
                story_id, order, len(data), "hit" if cache_hit else "miss",
            )
        except Exception as e:  # noqa: BLE001 - audio is a bonus, never fatal
            logger.warning("roadtrip audio prewarm %s seg%s FAILED: %s", story_id, order, e)
            try:
                await db.upsert_segment_audio(
                    story_id=story_id, segment_order=int(order), lang=lang,
                    status="failed", error=str(e)[:500],
                )
            except Exception:  # noqa: BLE001
                pass
    logger.info("roadtrip audio prewarm %s done: %d/%d segments ready", story_id, ok, len(segments))


async def _process_one() -> bool:
    """Claim + process one queued job. Returns True if a job was handled."""
    db = await get_postgres_manager()
    job = await db.claim_next_job()
    if job is None:
        return False

    job_id = job["id"]
    city = job["city"]
    theme = job["theme"]
    minutes = job["minutes"]
    ck = job["city_key"]
    logger.info("worker: claimed job %s -> %s/%s/%dmin", job_id, city, theme, minutes)

    try:
        # Cache re-check: another job may have already produced this exact story.
        cached = await db.find_ready_city_story(ck, theme, minutes)
        if cached is not None:
            logger.info("worker: cache filled while queued -> job %s ready (%s)", job_id, cached["id"])
            await db.finish_job(job_id, status="ready", story_id=cached["id"])
            await db.resolve_ready_jobs_for_story(ck, theme, minutes, cached["id"])
            # Pre-warm audio (bonus, after ready). A tts hiccup must not fail here.
            await _prewarm_audio(cached["id"])
            return True

        result = await citystory.generate_city_story(city, theme, minutes)

        saved = await db.create_city_story(
            city=city,
            city_key=ck,
            theme=theme,
            minutes=minutes,
            title=result["title"],
            chapters=result["chapters"],
            total_words=result["total_words"],
            sources=result["sources"],
            source_book=result["source_book"],
            model=result["model"],
            lang=result.get("lang", "en"),
        )
        if not saved or not saved.get("id"):
            raise RuntimeError("failed to persist city story")

        await db.finish_job(job_id, status="ready", story_id=saved["id"])
        # One generation serves everyone who asked for this key.
        await db.resolve_ready_jobs_for_story(ck, theme, minutes, saved["id"])
        logger.info(
            "worker: job %s READY -> story %s (%d chapters, %d words, model=%s)",
            job_id, saved["id"], len(result["chapters"]),
            result["total_words"], result["model"],
        )
        # Pre-warm the TTS cache for each chapter now the story is ready. This is
        # a BONUS layer AFTER ready — a tts failure never fails the story.
        await _prewarm_audio(saved["id"])
        return True
    except Exception as e:  # noqa: BLE001
        logger.exception("worker: job %s FAILED: %s", job_id, e)
        try:
            await db.finish_job(job_id, status="failed", error=str(e)[:1000])
        except Exception:  # noqa: BLE001
            logger.error("worker: could not mark job %s failed", job_id)
        return True


async def run_worker(stop_event: asyncio.Event) -> None:
    """The worker loop. Runs until `stop_event` is set. Serialized generation."""
    logger.info("worker: started (poll=%.1fs, serialized generation)", POLL_INTERVAL_SEC)
    while not stop_event.is_set():
        try:
            handled = await _process_one()
        except Exception as e:  # noqa: BLE001 - never let the loop die
            logger.exception("worker: unexpected error in loop: %s", e)
            handled = False
        if handled:
            # Immediately look for the next job (still one-at-a-time).
            continue
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=POLL_INTERVAL_SEC)
        except asyncio.TimeoutError:
            pass
    logger.info("worker: stopped")
