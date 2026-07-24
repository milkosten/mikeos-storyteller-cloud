"""Agent-key -> user_id resolution against mikeoscomputers (the IdP).

Apps authenticate with their hive agent key, sent as the X-API-KEY header. We
resolve it via
  GET {MIKEOSCOMPUTERS_URL}/api/mikeos/agents/resolve/{agent_key}
    -> {valid: bool, user_id: str, ...}
and scope ALL kitchen/shopping data per user_id. This is the same trust pattern
the hive and mikeos-oauth use.

Successful resolutions are cached briefly to avoid a network hop per request.
"""
import os
import time
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

MIKEOSCOMPUTERS_URL = os.environ.get(
    "MIKEOSCOMPUTERS_URL",
    "https://mikeoscomputers-production.up.railway.app",
)

_RESOLVE_CACHE_TTL = 300  # seconds
_resolve_cache: dict[str, tuple[str, float]] = {}  # agent_key -> (user_id, expires)


async def resolve_agent_key(agent_key: Optional[str]) -> Optional[str]:
    """Resolve an agent (hive) key to its user_id. Returns None if invalid.

    Network / IdP errors fail closed (return None -> caller responds 401).
    """
    if not agent_key:
        return None

    cached = _resolve_cache.get(agent_key)
    if cached and cached[1] > time.monotonic():
        return cached[0]

    base = MIKEOSCOMPUTERS_URL.rstrip("/")
    url = f"{base}/api/mikeos/agents/resolve/{agent_key}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code == 404:
                return None
            if resp.status_code != 200:
                logger.warning("IdP resolve returned HTTP %s", resp.status_code)
                return None
            data = resp.json()
    except Exception as e:
        logger.error("Error resolving agent key against IdP: %s", e)
        return None

    if not data or not data.get("valid"):
        return None
    user_id = data.get("user_id")
    if not user_id:
        logger.warning("IdP resolve returned valid=true without user_id")
        return None

    _resolve_cache[agent_key] = (str(user_id), time.monotonic() + _RESOLVE_CACHE_TTL)
    return str(user_id)
