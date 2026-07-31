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
import jwt
from jwt import PyJWKClient

logger = logging.getLogger(__name__)

MIKEOSCOMPUTERS_URL = os.environ.get(
    "MIKEOSCOMPUTERS_URL",
    "https://account.osmike.com",
)

ISSUER = os.environ.get("ACCOUNT_OSMIKE_ISSUER", "https://account.osmike.com")
JWKS_URL = os.environ.get("ACCOUNT_OSMIKE_JWKS_URL", f"{ISSUER}/oauth/jwks.json")
# Tolerant aud: device tokens carry aud="storyteller", web tokens aud="account.osmike.com".
_AUDIENCES = [a.strip() for a in os.environ.get(
    "OAUTH_AUDIENCE", "storyteller,account.osmike.com").split(",") if a.strip()]
_jwks_client = PyJWKClient(JWKS_URL, cache_keys=True, lifespan=3600)


def _verify_bearer(token: str):
    """Validate an account.osmike.com RS256 JWT locally via JWKS. None if invalid."""
    try:
        key = _jwks_client.get_signing_key_from_jwt(token).key
        return jwt.decode(
            token, key, algorithms=["RS256"], issuer=ISSUER, audience=_AUDIENCES,
            options={"require": ["exp", "iss", "sub"]},
        )
    except Exception as e:
        logger.warning("Bearer rejected: %s", e)
        return None


async def authenticate(authorization: Optional[str], x_api_key: Optional[str]) -> Optional[str]:
    """Return user_id or None. OAuth Bearer first, then legacy X-API-KEY."""
    if authorization and authorization.lower().startswith("bearer "):
        claims = _verify_bearer(authorization[7:].strip())
        # A present-but-invalid Bearer MUST 401 (do NOT silently fall through).
        return str(claims["sub"]) if claims else None
    return await resolve_agent_key(x_api_key)

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
