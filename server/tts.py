"""Server-side TTS client for mikeos-storyteller-cloud.

Turns a chapter's text into a spoken MP3 via the LIVE mikeos-tts service
(tts.osmike.com). The TTS bearer token lives SERVER-SIDE ONLY (TTS_TOKEN env,
like WIKI_TOKEN) — the Android app NEVER holds it; it gets audio through this
cloud's own X-API-KEY-authenticated proxy endpoint.

tts.osmike.com caches by sha256(text+voice) on the Hetzner box (unlimited disk),
so a repeat POST of the same chapter text is a near-instant cache hit
(`X-TTS-Cache: hit`). We do NOT store the MP3 bytes ourselves — the proxy just
re-fetches (cache hit) on demand. The worker PRE-WARMS this cache after a story
is ready so the app's first fetch is instant.

Never-trust-200: `synthesize` verifies the response is `audio/mpeg` and the body
is a real MP3 (> MIN_MP3_BYTES), else it raises TTSError. The proxy translates a
TTSError into a clean 502/503 — never a fake 200 with an empty body.
"""
import logging
import os
from typing import Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# The live TTS service. Token is server-side only (never client-facing / never git).
TTS_URL = os.environ.get("TTS_URL", "https://tts.osmike.com").rstrip("/")
TTS_TOKEN = os.environ.get("TTS_TOKEN", "")

# Supported narration languages (mikeos-tts: en|fr|sv; defaults en).
VALID_LANGS = ("en", "fr", "sv")

# tts.osmike.com hard cap per request (it sentence-splits + concatenates
# internally, so a full chapter fits in ONE request).
MAX_TTS_CHARS = 20000

# Never-trust-200: a valid MP3 for a real chapter is comfortably over this.
MIN_MP3_BYTES = 1024

# Approx MP3 bitrate mikeos-tts emits (96 kbps CBR) -> duration estimate from size.
_MP3_BITS_PER_SEC = 96000


class TTSError(RuntimeError):
    """Raised when TTS synthesis fails or returns a non-audio / empty body."""


def is_configured() -> bool:
    """True when the server-side TTS token is set (so audio is available)."""
    return bool(TTS_TOKEN)


def normalize_lang(lang: Optional[str]) -> str:
    lang = (lang or "en").strip().lower()
    return lang if lang in VALID_LANGS else "en"


def estimate_duration_sec(byte_len: int) -> int:
    """Rough spoken duration from MP3 size (96 kbps CBR)."""
    if byte_len <= 0:
        return 0
    return int(round(byte_len * 8 / _MP3_BITS_PER_SEC))


async def synthesize(
    text: str,
    lang: str = "en",
    *,
    voice: Optional[str] = None,
    timeout: float = 120.0,
) -> Tuple[bytes, bool]:
    """POST `text` to tts.osmike.com/api/tts -> (mp3_bytes, cache_hit).

    Server-side TTS_TOKEN auth. Verifies content-type is audio/mpeg and the body
    is a real MP3 (never-trust-200). Raises TTSError on any failure.
    """
    if not is_configured():
        raise TTSError("TTS_TOKEN is not configured on the server")
    body = (text or "").strip()
    if not body:
        raise TTSError("empty text")
    if len(body) > MAX_TTS_CHARS:
        # tts caps at 20k; trim on a whitespace boundary so we still say most of it.
        body = body[:MAX_TTS_CHARS].rsplit(" ", 1)[0]

    payload = {"text": body, "lang": normalize_lang(lang)}
    if voice:
        payload["voice"] = voice

    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(
                f"{TTS_URL}/api/tts",
                json=payload,
                headers={
                    "Authorization": f"Bearer {TTS_TOKEN}",
                    "Content-Type": "application/json",
                },
            )
    except httpx.HTTPError as e:
        raise TTSError(f"tts request failed: {e}") from e

    if r.status_code != 200:
        raise TTSError(f"tts returned HTTP {r.status_code}: {r.text[:200]}")

    ctype = (r.headers.get("content-type") or "").split(";")[0].strip().lower()
    if ctype != "audio/mpeg":
        raise TTSError(f"tts returned non-audio content-type {ctype!r}")

    data = r.content
    if not data or len(data) < MIN_MP3_BYTES:
        raise TTSError(f"tts returned too-small body ({len(data) if data else 0} bytes)")

    cache_hit = (r.headers.get("x-tts-cache") or "").strip().lower() == "hit"
    return data, cache_hit
