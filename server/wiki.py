"""Local self-hosted Wikipedia (Kiwix) client — wiki.osmike.com.

Best-effort, NEVER raises: every public coroutine returns None (or {}) on any
miss/failure so callers can transparently fall back to the public Wikipedia API.

Config (env):
  WIKI_URL     e.g. https://wiki.osmike.com   (unset -> disabled, returns None)
  WIKI_TOKEN   bearer token sent on every request
  WIKI_BOOK_EN / WIKI_BOOK_FR / WIKI_BOOK_SV  (optional) date-less book names as
               they appear in the OPDS catalog. The actual Kiwix URL slug is
               date-stamped (e.g. wikipedia_en_all_maxi_2026-02); we resolve the
               real slug from /catalog/v2/entries once and cache it.

API used (Kiwix-serve):
  GET /suggest?content=<slug>&term=<q>   -> JSON [{value,label,kind,path}]
  GET /content/<slug>/A/<path>           -> article HTML (302 -> canonical; follow)
"""
from __future__ import annotations

import asyncio
import html
import logging
import os
import re
from typing import Dict, Optional

import httpx

logger = logging.getLogger("storyteller.wiki")

WIKI_URL = (os.getenv("WIKI_URL") or "").rstrip("/")
WIKI_TOKEN = os.getenv("WIKI_TOKEN") or ""

# Date-less catalog names per language (override via env if the fleet changes).
_BOOK_NAMES = {
    "en": os.getenv("WIKI_BOOK_EN") or "wikipedia_en_all",
    "fr": os.getenv("WIKI_BOOK_FR") or "wikipedia_fr_all",
    "sv": os.getenv("WIKI_BOOK_SV") or "wikipedia_sv_all",
}

# Resolved date-stamped URL slug per catalog name, cached after first lookup.
_slug_cache: Dict[str, Optional[str]] = {}
_slug_lock = asyncio.Lock()

_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NL_RE = re.compile(r"\n{3,}")


def is_configured() -> bool:
    return bool(WIKI_URL and WIKI_TOKEN)


def _headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {WIKI_TOKEN}"}


def _book_name(lang: str) -> str:
    return _BOOK_NAMES.get((lang or "en").lower()[:2], _BOOK_NAMES["en"])


async def _resolve_slug(client: httpx.AsyncClient, book_name: str) -> Optional[str]:
    """Map a date-less catalog name -> the date-stamped /content URL slug.

    Kiwix serves books under a date-stamped slug (wikipedia_en_all_maxi_2026-02)
    while the OPDS <name> is date-less. We read /catalog/v2/entries once and pull
    the slug out of each entry's /content/<slug> link, keyed by <name>.
    """
    if book_name in _slug_cache:
        return _slug_cache[book_name]
    async with _slug_lock:
        if book_name in _slug_cache:
            return _slug_cache[book_name]
        try:
            r = await client.get(
                f"{WIKI_URL}/catalog/v2/entries", headers=_headers(), timeout=15
            )
            if r.status_code != 200:
                return None
            xml = r.text
            for m in re.finditer(r"<entry>.*?</entry>", xml, re.S):
                entry = m.group(0)
                nm = re.search(r"<name>([^<]*)</name>", entry)
                if not nm or nm.group(1).strip() != book_name:
                    continue
                link = re.search(r'href="(/content/[^"/]+)"', entry)
                if link:
                    slug = link.group(1).rsplit("/", 1)[-1]
                    _slug_cache[book_name] = slug
                    logger.info("wiki: resolved book %s -> %s", book_name, slug)
                    return slug
            # Fall back to the name itself (works if the deployment isn't date-stamped).
            _slug_cache[book_name] = book_name
            return book_name
        except Exception as e:  # pragma: no cover - best-effort
            logger.warning("wiki: catalog lookup failed for %s: %s", book_name, e)
            return None


def _strip_html(raw: str, max_chars: int) -> str:
    txt = _STYLE_RE.sub(" ", raw)
    txt = _TAG_RE.sub(" ", txt)
    txt = html.unescape(txt)
    txt = _WS_RE.sub(" ", txt)
    txt = _NL_RE.sub("\n\n", txt)
    txt = "\n".join(line.strip() for line in txt.splitlines())
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
    return txt[:max_chars]


async def _suggest_path(
    client: httpx.AsyncClient, slug: str, query: str
) -> Optional[str]:
    try:
        r = await client.get(
            f"{WIKI_URL}/suggest",
            params={"content": slug, "term": query},
            headers=_headers(),
            timeout=15,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        paths = [
            html.unescape(str(it["path"]))
            for it in (data if isinstance(data, list) else [])
            if it.get("kind") == "path" and it.get("path")
        ]
        if not paths:
            return None
        q = query.strip()
        q_us = q.replace(" ", "_")
        # Kiwix's suggest is a prefix match, so the top hit for e.g. "Nice" can be an
        # unrelated derived article ("NICE" the institute). Prefer an exact, then a
        # case-insensitive, article-name match (ignoring _/space); else the first hit.
        # Case-sensitive exact first — Kiwix paths are case-sensitive ("Nice" != "NICE").
        for p in paths:
            if p == q_us or p.replace("_", " ") == q:
                return p
        ql = q.lower()
        ql_us = q_us.lower()
        for p in paths:
            if p.lower() == ql_us or p.lower().replace("_", " ") == ql:
                return p
        return paths[0]
    except Exception as e:  # pragma: no cover
        logger.warning("wiki: suggest failed for %r: %s", query, e)
        return None


async def article(
    query: str, lang: str = "en", max_chars: int = 6000
) -> Optional[Dict[str, str]]:
    """Return {"title","extract","source"} for `query` from the local wiki, or None.

    Best-effort: any failure / miss returns None so callers can fall back.
    """
    if not is_configured() or not query or not query.strip():
        return None
    try:
        async with httpx.AsyncClient(follow_redirects=True, verify=True) as client:
            slug = await _resolve_slug(client, _book_name(lang))
            if not slug:
                return None
            path = await _suggest_path(client, slug, query.strip())
            if not path:
                return None
            r = await client.get(
                f"{WIKI_URL}/content/{slug}/A/{path}", headers=_headers(), timeout=25
            )
            if r.status_code != 200:
                return None
            text = _strip_html(r.text, max_chars)
            if not text or len(text) < 60:
                return None
            source = f"{WIKI_URL}/content/{slug}/A/{path}"
            return {"title": path.replace("_", " "), "extract": text, "source": source}
    except Exception as e:  # pragma: no cover
        logger.warning("wiki: article fetch failed for %r: %s", query, e)
        return None


async def article_text(
    query: str, lang: str = "en", max_chars: int = 6000
) -> Optional[str]:
    """Plain-text article body for `query`, or None. Best-effort, never raises."""
    art = await article(query, lang=lang, max_chars=max_chars)
    return art["extract"] if art else None
