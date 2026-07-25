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

from server import citystory
from server.storage.postgres_manager import get_postgres_manager

logger = logging.getLogger(__name__)

POLL_INTERVAL_SEC = 5.0


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
