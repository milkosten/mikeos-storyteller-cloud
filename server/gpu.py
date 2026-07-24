"""FREE-GPU (Ollama) client for mikeos-storyteller-cloud.

The story text is generated ONLY on the free shared GPU (Ollama qwen3:8b at
Kittelfjäll). NO paid API is ever used — that is a hard house rule.

Adapted from the reference client in mikeos-photos-cloud
(server/analysis/vision.py): `_ollama_endpoint()` parses
`OLLAMA_GPU_URL=ollama://user:pass@host:port` into a base URL + HTTP-Basic
header, and `_ollama_chat()` makes one `/api/chat` call. The public proxy uses
a self-signed cert, so TLS is not verified (it is Basic-auth protected).

qwen3 returns an EMPTY body unless `think:false` is set, so we always send it.
`format` (an Ollama JSON-schema grammar) is used to force structured segment
output.
"""
import base64
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# The free GPU URL, e.g. ollama://user:pass@host:port . When unset, GPU
# generation is unavailable and the caller falls back to a deterministic
# template (never a paid API).
OLLAMA_GPU_URL = os.environ.get("OLLAMA_GPU_URL", "")

# Text model for story generation (house rule: qwen3:8b, ~2s / 100 words).
OLLAMA_TEXT_MODEL = os.environ.get("OLLAMA_TEXT_MODEL", "qwen3:8b")


def is_configured() -> bool:
    """True when a free-GPU backend is available."""
    return bool(OLLAMA_GPU_URL)


def _ollama_endpoint() -> tuple[str, Dict[str, str]]:
    """Parse OLLAMA_GPU_URL (ollama://user:pass@host:port) into (base_url, headers)."""
    raw = OLLAMA_GPU_URL
    scheme = "https"
    rest = raw
    if "://" in raw:
        s, rest = raw.split("://", 1)
        s = s.lower()
        scheme = "http" if s in ("ollama+http", "http") else "https"
    headers: Dict[str, str] = {}
    if "@" in rest:
        creds, rest = rest.rsplit("@", 1)
        headers["Authorization"] = "Basic " + base64.b64encode(creds.encode()).decode()
    return f"{scheme}://{rest.rstrip('/')}", headers


async def ollama_chat(
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    schema: Optional[Dict[str, Any]] = None,
    num_predict: int = 4096,
    temperature: float = 0.7,
    timeout: float = 300.0,
) -> str:
    """One Ollama /api/chat call. Returns the message content (may be empty).

    `schema` (when given) constrains the output to that JSON shape. `think:false`
    is mandatory for qwen3 (it otherwise returns an empty body). TLS is
    self-signed on the public proxy -> not verified (Basic-auth protected).
    """
    base, headers = _ollama_endpoint()
    body: Dict[str, Any] = {
        "model": model or OLLAMA_TEXT_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "num_ctx": 8192,
            "num_predict": num_predict,
            "temperature": temperature,
        },
    }
    if schema is not None:
        body["format"] = schema  # structured output (Ollama JSON-schema grammar)
    async with httpx.AsyncClient(verify=False, timeout=timeout) as c:
        r = await c.post(
            f"{base}/api/chat",
            json=body,
            headers={**headers, "Content-Type": "application/json"},
        )
        r.raise_for_status()
        return (r.json().get("message") or {}).get("content", "")
