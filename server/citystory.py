"""DEEP, fact-grounded CITY-STORY engine for mikeos-storyteller-cloud.

This is the backend brain for the MikeStoryteller app. It does NOT do the
"summarize this city" slop — a single prompt yields generic filler. Instead it
runs a multi-source, outline->chapter pipeline grounded in the self-hosted
encyclopedia (wiki.osmike.com), so every chapter is bound to REAL article text
and asked for concrete detail (dates, names, numbers, events, anecdotes).

Pipeline
--------
1. GATHER (server/wiki.py, best-effort): fetch the city article, then
   theme-relevant EXTRA articles (2-6 sources total, not one):
     - people:       notable people resolved from the city article + a
                     "<city> people" search; fetch each person's article.
     - history:      the city + key events / landmarks named in it.
     - art / architecture / nature / food / complete: museums, artists,
       buildings, parks, dishes discovered from the city article + searches.
2. OUTLINE (GPU qwen3:8b, think:false): from the gathered facts + theme, a
   CHAPTER OUTLINE sized to the target length (chapters scale with minutes:
   5min ~= 2, 45min ~= 8-10), each chapter bound to a specific source.
3. CHAPTERS (GPU, per chapter): expand each chapter GROUNDED in its source
   text, prompted for concrete detail and EXPLICITLY forbidden vague filler /
   "nestled in the heart of" slop. A per-chapter word target paces it.
4. STITCH: warm narrator intro chapter + body chapters + gentle outro chapter.
   est_seconds per chapter = words / 150 * 60.

Never-trust-the-GPU: an empty/garbage pass retries once, then falls back to a
simpler-but-STILL-grounded generation. We never store an empty story.
"""
import asyncio
import json
import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from server import gpu
from server import wiki as local_wiki

logger = logging.getLogger(__name__)

WORDS_PER_MIN = 150

# Valid inputs (the app contract).
VALID_MINUTES = (5, 10, 20, 30, 45)
VALID_THEMES = ("people", "history", "art", "architecture", "nature", "food", "complete")

# How many chapters to aim for at each length (intro/outro count toward this).
def _chapter_count(minutes: int) -> int:
    return {5: 3, 10: 4, 20: 6, 30: 8, 45: 10}.get(int(minutes), 5)


# Per-source article text cap fed to the GPU. Kept modest so several sources fit
# the 8k context without truncation.
SOURCE_CHARS = 3800
CITY_CHARS = 5000            # the primary city article gets a bigger slice
MAX_SOURCES = 6              # 2-6 sources per story
MIN_SOURCES = 1

# Theme -> extra search terms appended to "<city> ..." when hunting sources.
_THEME_SEARCHES: Dict[str, List[str]] = {
    "people": ["people", "history"],
    "history": ["history", "castle", "cathedral"],
    "art": ["museum", "art"],
    "architecture": ["cathedral", "castle", "architecture"],
    "nature": ["park", "nature", "beach"],
    "food": ["cuisine", "wine", "food"],
    "complete": ["history", "museum", "park"],
}


# ---------------------------------------------------------------------------
# Normalization: the cache key.
# ---------------------------------------------------------------------------
def city_key(city: str) -> str:
    """Normalized cache key: lowercased, accent-stripped, collapsed whitespace."""
    s = unicodedata.normalize("NFKD", (city or "").strip())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "unknown"


def _est_seconds(text: str) -> int:
    return int(round(len((text or "").split()) / WORDS_PER_MIN * 60))


def _words(text: str) -> int:
    return len((text or "").split())


# ---------------------------------------------------------------------------
# Lenient JSON (shared shape with story.py, kept local so citystory is standalone).
# ---------------------------------------------------------------------------
def _lenient_json(text: str) -> Optional[Any]:
    s = (text or "").strip()
    if not s:
        return None
    if s.startswith("```"):
        s = s.strip("`")
        if s[:4].lower() == "json":
            s = s[4:]
        s = s.strip()
    start = min(
        [i for i in (s.find("{"), s.find("[")) if i >= 0] or [-1]
    )
    if start < 0:
        return None
    dec = json.JSONDecoder(strict=False)
    try:
        obj, _ = dec.raw_decode(s[start:])
        return obj
    except Exception:  # noqa: BLE001
        pass
    frag = s[start:]
    closed = _close_json(frag)
    for candidate in (frag, closed):
        if candidate is None:
            continue
        try:
            return json.loads(candidate, strict=False)
        except Exception:  # noqa: BLE001
            continue
    return None


def _close_json(s: str) -> Optional[str]:
    stack: List[str] = []
    in_str = esc = False
    for ch in s:
        if esc:
            esc = False
        elif ch == "\\" and in_str:
            esc = True
        elif ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]" and stack:
                stack.pop()
    if not in_str and not stack:
        return None
    out = s
    if in_str:
        out += '"'
    out = re.sub(r"[,:]\s*$", "", out.rstrip())
    out = re.sub(r',\s*"[^"]*"\s*:?\s*$', "", out)
    for closer in reversed(stack):
        out += closer
    return out


# ---------------------------------------------------------------------------
# Stage 1: GATHER sources from wiki.osmike.com.
# ---------------------------------------------------------------------------
# People/proper-name-ish extraction from the city article text: find capitalized
# multi-word names to feed back as person-article lookups (people theme).
_NAME_RE = re.compile(
    r"\b([A-Z][a-zà-öø-ÿ'’]+(?:\s+(?:de|van|von|der|of|the|di|le|la)\s+|\s+)"
    r"[A-Z][a-zà-öø-ÿ'’]+(?:\s+[A-Z][a-zà-öø-ÿ'’]+)?)"
)
# Terms that are clearly not a person's name even if capitalized.
_NAME_STOP = {
    "The City", "World War", "Middle Ages", "United States", "New York",
    "Second World", "First World", "Roman Empire", "Cote D", "Côte D",
}


def _candidate_names(text: str, limit: int = 24) -> List[str]:
    seen: List[str] = []
    for m in _NAME_RE.finditer(text or ""):
        name = re.sub(r"\s+", " ", m.group(1)).strip()
        if len(name) < 5 or name in _NAME_STOP:
            continue
        if any(w in name for w in ("City", "War", "Ages", "Empire", "Sea", "River")):
            continue
        if name not in seen:
            seen.append(name)
        if len(seen) >= limit:
            break
    return seen


class Source:
    __slots__ = ("title", "text", "url")

    def __init__(self, title: str, text: str, url: str):
        self.title = title
        self.text = text
        self.url = url


async def _fetch_article(query: str, max_chars: int) -> Optional[Source]:
    try:
        art = await local_wiki.article(query, lang="en", max_chars=max_chars)
    except Exception:  # noqa: BLE001
        art = None
    if not art or len(art.get("extract") or "") < 200:
        return None
    return Source(art["title"], art["extract"], art["source"])


async def gather_sources(city: str, theme: str) -> Tuple[List[Source], Optional[str]]:
    """Gather 2-6 grounding articles for (city, theme). Returns (sources, book_slug).

    Best-effort: the city article is the anchor; theme-relevant extras are added
    up to MAX_SOURCES. If the wiki is unreachable, returns ([], None) and the
    caller degrades to a minimal (still non-empty) story.
    """
    sources: List[Source] = []
    seen_urls = set()

    def _add(src: Optional[Source]) -> bool:
        if not src or src.url in seen_urls:
            return False
        seen_urls.add(src.url)
        sources.append(src)
        return True

    # --- Anchor: the city article. ---
    city_src = await _fetch_article(city, CITY_CHARS)
    _add(city_src)
    city_text = city_src.text if city_src else ""

    # --- Theme-relevant extra searches. ---
    search_terms: List[str] = []
    for suffix in _THEME_SEARCHES.get(theme, _THEME_SEARCHES["complete"]):
        search_terms.append(f"{city} {suffix}")

    # For people, also mine candidate names out of the city article itself.
    person_queries: List[str] = []
    if theme == "people" and city_text:
        person_queries = _candidate_names(city_text, limit=20)

    async def _search_paths(term: str) -> List[str]:
        try:
            hits = await local_wiki.search(term, lang="en", limit=6)
        except Exception:  # noqa: BLE001
            hits = []
        return [h["title"] for h in hits]

    # Run searches concurrently, then fetch top candidates serially-ish until full.
    search_results = await asyncio.gather(
        *[_search_paths(t) for t in search_terms]
    )
    candidate_titles: List[str] = []
    for res in search_results:
        for t in res:
            if t not in candidate_titles:
                candidate_titles.append(t)

    # People theme: prioritise person names discovered in the article, then search hits.
    if theme == "people":
        ordered = person_queries + candidate_titles
    else:
        ordered = candidate_titles

    # Fetch extra sources concurrently in small batches until we hit MAX_SOURCES.
    ck = city_key(city)
    for title in ordered:
        if len(sources) >= MAX_SOURCES:
            break
        # Skip the city page itself / near-duplicates.
        if city_key(title) == ck:
            continue
        src = await _fetch_article(title, SOURCE_CHARS)
        _add(src)

    book_slug = None
    if sources:
        # Derive the book slug from the first source URL (/content/<slug>/A/...).
        m = re.search(r"/content/([^/]+)/A/", sources[0].url)
        if m:
            book_slug = m.group(1)

    logger.info(
        "citystory: gathered %d sources for %s/%s: %s",
        len(sources), city, theme, [s.title for s in sources],
    )
    return sources, book_slug


# ---------------------------------------------------------------------------
# Stage 2: OUTLINE.
# ---------------------------------------------------------------------------
_OUTLINE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "focus": {"type": "string"},
                    "source_index": {"type": "integer"},
                    "facts": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "focus", "source_index"],
            },
        },
    },
    "required": ["title", "chapters"],
}

_THEME_ANGLE = {
    "people": "the notable PEOPLE of the city — their lives, achievements, dates, and connection to the place",
    "history": "the HISTORY of the city — its founding, key eras, wars, rulers, and turning points, in order",
    "art": "the ART and artists of the city — its museums, works, movements, and creators",
    "architecture": "the ARCHITECTURE of the city — its landmark buildings, styles, builders, and eras",
    "nature": "the NATURE and landscape around the city — its geography, coast, parks, wildlife, climate",
    "food": "the FOOD and drink of the city — its dishes, produce, wine, markets, and culinary traditions",
    "complete": "a COMPLETE portrait of the city — history, people, culture, landmarks, and daily life",
}


def _sources_block(sources: List[Source], per_src_chars: int) -> str:
    lines = []
    for i, s in enumerate(sources):
        lines.append(f"[SOURCE {i}] {s.title}\n{s.text[:per_src_chars]}")
    return "\n\n".join(lines)


async def _outline(
    city: str, theme: str, minutes: int, sources: List[Source]
) -> Optional[Dict[str, Any]]:
    n_chapters = _chapter_count(minutes)
    angle = _THEME_ANGLE.get(theme, _THEME_ANGLE["complete"])
    # Give the outline a trimmed view of each source (facts, not whole bodies).
    src_block = _sources_block(sources, per_src_chars=1600)

    prompt = (
        f"You are the editor planning a spoken audio story about {city}, focused on {angle}.\n"
        f"Below are numbered SOURCE excerpts from an encyclopedia. Using ONLY facts present in "
        f"them, produce a chapter outline of EXACTLY {n_chapters} chapters that a narrator will "
        f"expand. The chapters should flow in a sensible order and TOGETHER cover the most "
        f"concrete, interesting material — real dates, names, events, numbers.\n\n"
        f"For each chapter give: a short title; a 'focus' (one sentence on what it covers); "
        f"'source_index' (the SOURCE number it draws most from); and 'facts' (2-5 specific facts "
        f"from the sources that chapter MUST include — real dates/names/events, not vague ideas).\n"
        f"Do NOT invent facts not in the sources. Do NOT write the narration yet — only the plan.\n\n"
        f"SOURCES:\n{src_block}\n\n"
        f'Return JSON: {{"title": "<story title>", "chapters": [{{"title","focus","source_index","facts":[...]}}]}}'
    )
    for attempt in range(2):
        try:
            content = await gpu.ollama_chat(
                [{"role": "user", "content": prompt}],
                schema=_OUTLINE_SCHEMA,
                num_predict=2048,
                temperature=0.5,
            )
            data = _lenient_json(content)
            if isinstance(data, dict) and isinstance(data.get("chapters"), list) and data["chapters"]:
                return data
            logger.warning("outline attempt %d: no usable chapters (len=%d)", attempt + 1, len(content or ""))
        except Exception as e:  # noqa: BLE001
            logger.warning("outline attempt %d failed: %s", attempt + 1, e)
    return None


def _fallback_outline(city: str, theme: str, minutes: int, sources: List[Source]) -> Dict[str, Any]:
    """Deterministic, still-grounded outline: one chapter per source (capped)."""
    n = max(2, min(_chapter_count(minutes), max(len(sources), 2)))
    chapters = []
    for i in range(n):
        si = i % max(len(sources), 1)
        title = sources[si].title if sources else city
        chapters.append({
            "title": title,
            "focus": f"What the sources record about {title}.",
            "source_index": si,
            "facts": [],
        })
    return {"title": f"{city}: A {theme.title()} Story", "chapters": chapters}


# ---------------------------------------------------------------------------
# Stage 3: CHAPTERS.
# ---------------------------------------------------------------------------
_CHAPTER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}

_NO_SLOP = (
    "Write like a knowledgeable documentary narrator, NOT a travel brochure. "
    "Use CONCRETE specifics from the source — real dates, names, numbers, events, "
    "and anecdotes. FORBIDDEN: vague filler such as 'nestled in the heart of', "
    "'a hidden gem', 'rich tapestry', 'timeless charm', 'must-see', 'stunning', "
    "'breathtaking', 'boasts', or any sentence that could describe any city. "
    "Every few sentences must contain a fact a listener could not have guessed. "
    "Do NOT invent facts that are not in the source text."
)


async def _chapter(
    ch: Dict[str, Any],
    city: str,
    theme: str,
    source: Optional[Source],
    word_target: int,
    is_intro: bool,
    is_outro: bool,
) -> Optional[str]:
    facts = ch.get("facts") or []
    fact_block = ""
    if facts:
        fact_block = "This chapter MUST weave in these facts from the source:\n- " + "\n- ".join(
            str(f) for f in facts
        ) + "\n"
    src_block = ""
    if source:
        src_block = f'SOURCE TEXT ("{source.title}"):\n{source.text[:SOURCE_CHARS]}\n\n'

    role = ""
    if is_intro:
        role = ("This is the OPENING chapter: give a brief, warm welcome that immediately lands "
                "the listener in a specific real detail about the city — no generic throat-clearing.\n")
    elif is_outro:
        role = ("This is the CLOSING chapter: tie the threads together and end on one memorable, "
                "specific fact or image — a gentle sign-off, not a summary of platitudes.\n")

    prompt = (
        f"You are narrating an audio story about {city}. Write ONE chapter titled "
        f'"{ch.get("title")}". Focus: {ch.get("focus")}.\n'
        f"{role}"
        f"Target about {word_target} words of flowing spoken narration (no headings, no lists — "
        f"continuous prose meant to be read aloud).\n"
        f"{_NO_SLOP}\n\n"
        f"{fact_block}"
        f"{src_block}"
        f'Return JSON: {{"text": "<the chapter narration>"}}'
    )
    num_predict = min(4096, max(768, int(word_target * 2.2) + 256))
    for attempt in range(2):
        try:
            content = await gpu.ollama_chat(
                [{"role": "user", "content": prompt}],
                schema=_CHAPTER_SCHEMA,
                num_predict=num_predict,
                temperature=0.6,
            )
            data = _lenient_json(content)
            if isinstance(data, dict):
                text = (data.get("text") or "").strip()
                if _words(text) >= 30:
                    return text
            logger.warning("chapter %r attempt %d: too short/empty", ch.get("title"), attempt + 1)
        except Exception as e:  # noqa: BLE001
            logger.warning("chapter %r attempt %d failed: %s", ch.get("title"), attempt + 1, e)
    return None


def _fallback_chapter(ch: Dict[str, Any], source: Optional[Source], word_target: int) -> str:
    """Still-grounded fallback: lead with the source's own opening sentences."""
    title = ch.get("title") or "This chapter"
    base = ""
    if source:
        # Take the first ~word_target words of real source text as the grounding body.
        words = source.text.split()
        base = " ".join(words[: max(60, word_target)])
    if not base:
        base = f"The sources record little more about {title}."
    return base


# ---------------------------------------------------------------------------
# Public entry point: generate a full deep city story.
# ---------------------------------------------------------------------------
async def generate_city_story(
    city: str, theme: str, minutes: int
) -> Dict[str, Any]:
    """Generate a deep, fact-grounded city story.

    Returns {title, chapters:[{index,title,text,est_seconds,source_url}],
    total_words, sources:[urls], source_book, model}. NEVER returns empty
    chapters (raises RuntimeError only if it truly cannot ground anything AND
    the GPU is unavailable, so the worker marks the job failed rather than
    storing junk).
    """
    theme = theme if theme in VALID_THEMES else "complete"
    minutes = int(minutes) if int(minutes) in VALID_MINUTES else 20

    sources, book_slug = await gather_sources(city, theme)
    source_urls = [s.url for s in sources]

    used_fallback = False

    # --- Outline ---
    outline = None
    if gpu.is_configured() and sources:
        outline = await _outline(city, theme, minutes, sources)
    if outline is None:
        used_fallback = True
        outline = _fallback_outline(city, theme, minutes, sources)

    title = (outline.get("title") or f"{city}: A {theme.title()} Story").strip()
    raw_chapters = outline.get("chapters") or []
    if not raw_chapters:
        raw_chapters = _fallback_outline(city, theme, minutes, sources)["chapters"]

    # --- Word budget across chapters (intro/outro a touch lighter) ---
    total_words = minutes * WORDS_PER_MIN
    n = len(raw_chapters)
    weights = []
    for i in range(n):
        if i == 0 or i == n - 1:
            weights.append(0.7)   # intro/outro slightly shorter
        else:
            weights.append(1.0)
    wsum = sum(weights) or float(n)
    word_targets = [max(60, int(round(total_words * (w / wsum)))) for w in weights]

    # --- Chapters (serial per story; the WORKER serializes across stories) ---
    chapters: List[Dict[str, Any]] = []
    for i, ch in enumerate(raw_chapters):
        si = ch.get("source_index")
        source = None
        if isinstance(si, int) and 0 <= si < len(sources):
            source = sources[si]
        elif sources:
            source = sources[i % len(sources)]

        text = None
        if gpu.is_configured():
            text = await _chapter(
                ch, city, theme, source, word_targets[i],
                is_intro=(i == 0), is_outro=(i == n - 1),
            )
        if not text:
            used_fallback = True
            text = _fallback_chapter(ch, source, word_targets[i])

        chapters.append({
            "index": i,
            "title": (ch.get("title") or f"Chapter {i + 1}").strip(),
            "text": text.strip(),
            "est_seconds": _est_seconds(text),
            "source_url": source.url if source else None,
        })

    # Never store an empty story.
    chapters = [c for c in chapters if c["text"] and _words(c["text"]) >= 20]
    if not chapters:
        raise RuntimeError(f"city story for {city}/{theme}/{minutes} produced no usable chapters")

    computed_words = sum(_words(c["text"]) for c in chapters)
    model = "fallback" if used_fallback else gpu.OLLAMA_TEXT_MODEL

    logger.info(
        "citystory: built %s/%s/%dmin -> %d chapters, %d words, %d sources, model=%s",
        city, theme, minutes, len(chapters), computed_words, len(source_urls), model,
    )
    return {
        "title": title,
        "chapters": chapters,
        "total_words": computed_words,
        "sources": source_urls,
        "source_book": book_slug,
        "model": model,
    }
