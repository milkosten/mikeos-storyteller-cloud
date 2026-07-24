"""Story generation for mikeos-storyteller-cloud.

Turns an ORDERED list of POIs (MikeGuide's route.pois) into a duration-matched,
spoken-word road-trip story on the FREE GPU (qwen3:8b).

Pacing
------
Target spoken length = `minutes` at ~150 words/min -> total_words = minutes*150.
A small intro + outro reserve is taken off the top, and the remainder is
distributed across the POIs PROPORTIONAL to each POI's `dwell` weight (a
dwell=3.0 landmark gets ~3x the words of a dwell=1.0 one). Every POI gets at
least a small floor so nothing is silent.

Generation
----------
The POIs are generated in CHUNKS (a few POIs per GPU call) so no single call is
huge — a 30-min story is ~4500 words. Each chunk is asked, with an Ollama JSON
schema, for one narrated segment per POI: {order, poi_name, text, est_seconds}.
The narrator is warm and flowing, grounds the text in the POI names/types/blurbs
and their DRIVE ORDER, and adds only light, plausible colour (NO invented dates
or quotes).

Never-trust-the-GPU
-------------------
If the GPU returns empty / unparseable for a chunk, we retry once, then fall
back to a deterministic template for that chunk so the endpoint ALWAYS returns
usable segments (marked model='fallback'). We never return an empty segments
array with 200.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional

from server import gpu

logger = logging.getLogger(__name__)

WORDS_PER_MIN = 150
# POIs per GPU call. Keeps each generation small so a long story stays fast and
# no single call risks a truncated body.
CHUNK_SIZE = 4
# Reserve a little of the budget for a warm opener + closer woven into the first
# and last segments (kept small so the per-POI pacing stays honest).
INTRO_OUTRO_FRACTION = 0.08
MIN_POI_WORDS = 25


# ---------------------------------------------------------------------------
# Pacing: dwell-weighted per-POI word budget.
# ---------------------------------------------------------------------------
def word_budget(pois: List[Dict[str, Any]], total_words: int) -> List[int]:
    """Split `total_words` across POIs proportional to dwell, with a floor.

    Returns a list of integer word budgets aligned to `pois`. Dwell defaults to
    1.0 when missing / non-positive.
    """
    n = len(pois)
    if n == 0:
        return []
    reserve = int(round(total_words * INTRO_OUTRO_FRACTION))
    body_words = max(total_words - reserve, n * MIN_POI_WORDS)

    weights = []
    for p in pois:
        try:
            w = float(p.get("dwell") or 0.0)
        except (TypeError, ValueError):
            w = 0.0
        weights.append(w if w > 0 else 1.0)
    wsum = sum(weights) or float(n)

    budgets = [max(MIN_POI_WORDS, int(round(body_words * (w / wsum)))) for w in weights]
    # Fold the intro/outro reserve into the first and last segments.
    if reserve > 0:
        budgets[0] += reserve - reserve // 2
        budgets[-1] += reserve // 2
    return budgets


def _est_seconds(text: str) -> int:
    words = len((text or "").split())
    return int(round(words / WORDS_PER_MIN * 60))


# ---------------------------------------------------------------------------
# GPU generation.
# ---------------------------------------------------------------------------
_SEGMENTS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "order": {"type": "integer"},
                    "poi_name": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["order", "poi_name", "text"],
            },
        }
    },
    "required": ["segments"],
}


def _poi_line(p: Dict[str, Any], budget: int) -> str:
    name = p.get("name") or "an unnamed spot"
    cat = p.get("cat") or "place"
    blurb = (p.get("blurb") or "").strip()
    order = p.get("order")
    line = f"- order {order}: \"{name}\" (a {cat}); target ~{budget} words"
    if blurb:
        line += f"; known detail: {blurb}"
    return line


async def _generate_chunk(
    chunk: List[Dict[str, Any]],
    budgets: List[int],
    dest: Optional[str],
    is_first: bool,
    is_last: bool,
) -> Optional[List[Dict[str, Any]]]:
    """Generate segments for one chunk of POIs on the GPU. None on failure."""
    dest_line = f" toward {dest}" if dest else ""
    intro_hint = (
        "This is the OPENING of the story: begin with a brief, warm welcome to the drive "
        "before you reach the first place.\n"
        if is_first
        else ""
    )
    outro_hint = (
        "This is the END of the story: after the last place, close with a short, gentle sign-off.\n"
        if is_last
        else ""
    )
    poi_block = "\n".join(_poi_line(p, b) for p, b in zip(chunk, budgets))

    prompt = (
        "You are a warm, engaging road-trip narrator riding along with the driver"
        f"{dest_line}. Tell a FLOWING spoken-word story about the places passed, IN THE "
        "ORDER GIVEN. Write one segment per place below, connecting them so it reads as one "
        "continuous tale (not a list). Ground every segment in the place's name, its type, and "
        "any known detail given. Light, plausible historical or cultural colour is welcome, but "
        "invent NO specific facts — no made-up dates, statistics, or quotations. Hit each "
        "place's target word count reasonably closely; longer targets deserve richer, longer "
        "segments.\n"
        f"{intro_hint}{outro_hint}"
        "\nPlaces, in drive-past order:\n"
        f"{poi_block}\n\n"
        "Return JSON: an object with a \"segments\" array; each item is "
        "{order, poi_name, text} where order and poi_name match the place above and text is the "
        "spoken narration for it."
    )

    total_budget = sum(budgets)
    # ~1.5 tokens/word plus headroom for JSON overhead.
    num_predict = min(8192, max(1024, int(total_budget * 2.0) + 512))

    for attempt in range(2):
        try:
            content = await gpu.ollama_chat(
                [{"role": "user", "content": prompt}],
                schema=_SEGMENTS_SCHEMA,
                num_predict=num_predict,
            )
            segs = _parse_segments(content, chunk)
            if segs:
                return segs
            logger.warning(
                "GPU chunk returned no usable segments (attempt %d, len=%d)",
                attempt + 1,
                len(content or ""),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("GPU chunk call failed (attempt %d): %s", attempt + 1, e)
    return None


def _parse_segments(
    content: str, chunk: List[Dict[str, Any]]
) -> Optional[List[Dict[str, Any]]]:
    """Parse the model reply into aligned segments for the chunk, or None."""
    data = _lenient_json(content)
    if not isinstance(data, dict):
        return None
    raw = data.get("segments")
    if not isinstance(raw, list) or not raw:
        return None

    # Index the model's segments by order, then by poi_name, for robust matching.
    by_order: Dict[Any, Dict[str, Any]] = {}
    by_name: Dict[str, Dict[str, Any]] = {}
    for s in raw:
        if not isinstance(s, dict):
            continue
        txt = (s.get("text") or "").strip()
        if not txt:
            continue
        if s.get("order") is not None:
            by_order.setdefault(s.get("order"), s)
        if s.get("poi_name"):
            by_name.setdefault(str(s.get("poi_name")).strip().lower(), s)

    out: List[Dict[str, Any]] = []
    for p in chunk:
        order = p.get("order")
        name = p.get("name") or ""
        match = by_order.get(order) or by_name.get(str(name).strip().lower())
        if match is None:
            return None  # incomplete chunk -> treat as failure so we retry/fallback
        text = (match.get("text") or "").strip()
        if not text:
            return None
        out.append(
            {
                "order": order,
                "poi_name": name or match.get("poi_name") or "",
                "text": text,
                "est_seconds": _est_seconds(text),
            }
        )
    return out


def _lenient_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from a model reply that may carry code fences or prose."""
    s = (text or "").strip()
    if not s:
        return None
    if s.startswith("```"):
        s = s.strip("`")
        if s[:4].lower() == "json":
            s = s[4:]
        s = s.strip()
    start = s.find("{")
    if start < 0:
        return None
    dec = json.JSONDecoder(strict=False)
    try:
        obj, _ = dec.raw_decode(s[start:])
        if isinstance(obj, dict):
            return obj
    except Exception:  # noqa: BLE001
        pass
    # Salvage a possibly-truncated object.
    frag = s[start:]
    closed = _close_json(frag)
    for candidate in (frag, closed):
        if candidate is None:
            continue
        try:
            obj = json.loads(candidate, strict=False)
            if isinstance(obj, dict):
                return obj
        except Exception:  # noqa: BLE001
            continue
    return None


def _close_json(s: str) -> Optional[str]:
    """Close an unterminated string + open brackets in a truncated JSON object."""
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
# Deterministic fallback (never a paid API; always usable segments).
# ---------------------------------------------------------------------------
def _fallback_segment(p: Dict[str, Any], budget: int) -> Dict[str, Any]:
    name = p.get("name") or "an unnamed spot"
    cat = p.get("cat") or "place"
    blurb = (p.get("blurb") or "").strip()
    parts = [f"As you pass {name}, a {cat}, take a moment to notice it going by."]
    if blurb:
        parts.append(blurb if blurb.endswith(".") else blurb + ".")
    parts.append(
        f"It is one of the places along today's route — {name} slips past the window as the "
        "road carries you onward to the next."
    )
    text = " ".join(parts)
    return {
        "order": p.get("order"),
        "poi_name": name,
        "text": text,
        "est_seconds": _est_seconds(text),
    }


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
async def generate_story(
    pois: List[Dict[str, Any]],
    minutes: int,
    dest: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a dwell-weighted, duration-matched story.

    Returns {segments:[{order,poi_name,text,est_seconds}], total_words, model}.
    `model` is 'qwen3:8b' if the whole story came from the GPU, else 'fallback'
    (used when the GPU is unavailable or failed for one or more chunks). Segments
    are always non-empty when pois is non-empty.
    """
    # Order the POIs by their drive-past order (stable; missing order -> keep index).
    ordered = sorted(
        list(enumerate(pois)),
        key=lambda t: (t[1].get("order") if t[1].get("order") is not None else t[0]),
    )
    pois = [p for _, p in ordered]

    total_words = int(minutes) * WORDS_PER_MIN
    budgets = word_budget(pois, total_words)

    segments: List[Dict[str, Any]] = []
    used_fallback = False
    n = len(pois)

    if not gpu.is_configured():
        logger.warning("OLLAMA_GPU_URL unset — using deterministic fallback for all POIs")
        used_fallback = True
        segments = [_fallback_segment(p, b) for p, b in zip(pois, budgets)]
    else:
        for start in range(0, n, CHUNK_SIZE):
            chunk = pois[start : start + CHUNK_SIZE]
            chunk_budgets = budgets[start : start + CHUNK_SIZE]
            is_first = start == 0
            is_last = (start + CHUNK_SIZE) >= n
            logger.info(
                "generating chunk %d-%d of %d POIs (~%d words)",
                start,
                start + len(chunk) - 1,
                n,
                sum(chunk_budgets),
            )
            segs = await _generate_chunk(chunk, chunk_budgets, dest, is_first, is_last)
            if segs is None:
                logger.warning("chunk %d failed after retry — deterministic fallback", start)
                used_fallback = True
                segs = [_fallback_segment(p, b) for p, b in zip(chunk, chunk_budgets)]
            segments.extend(segs)

    segments.sort(
        key=lambda s: (s.get("order") if s.get("order") is not None else 0)
    )
    computed_words = sum(len((s.get("text") or "").split()) for s in segments)
    return {
        "segments": segments,
        "total_words": computed_words,
        "model": "fallback" if used_fallback else gpu.OLLAMA_TEXT_MODEL,
    }
