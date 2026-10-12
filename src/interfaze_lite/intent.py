"""Whether a request asks for a translation or a forecast, read off its instruction.

Lite has no router to pick tools as interfaze does, and offered on every text request the
model "translated" its own English answer into English, 53 times in 1,400 extraction requests.
An instruction sits at a message's start or end, and passages mention languages all the time,
so only the ends are read: across 5,000 extraction prompts that offered translation to 88,
where reading the whole message offered it to 1,031.
"""

from __future__ import annotations

import re
from functools import lru_cache

try:
    from .translation import languages
except ImportError:  # flat, from a checkout
    from translation import languages  # type: ignore

WINDOW = 300

_FORECAST = re.compile(
    r"\b(?:forecast|predict|projection|extrapolat)"
    r"|\b(?:expect|next|coming|upcoming|future)\b.{0,40}"
    r"\b(?:days?|weeks?|months?|quarters?|years?|periods?|steps?|values?|points?)\b",
    re.I | re.S,
)


@lru_cache(maxsize=1)
def _translation() -> re.Pattern:
    names = sorted({entry["name"] for entry in languages().values()}, key=len, reverse=True)
    names_pattern = "|".join(re.escape(n) for n in names)
    return re.compile(
        rf"\btranslat|\b(?:in|into|to)\s+(?:{names_pattern})\b"
        r"|\bhow (?:do|would|can) (?:you|i|we) say\b",
        re.I,
    )


def instruction(text: str) -> str:
    """The ends of a message, where its instruction is."""
    if len(text) <= 2 * WINDOW:
        return text
    return text[:WINDOW] + "\n" + text[-WINDOW:]


def wants_translation(text: str) -> bool:
    return bool(_translation().search(instruction(text)))


def wants_forecast(text: str) -> bool:
    return bool(_FORECAST.search(instruction(text)))
