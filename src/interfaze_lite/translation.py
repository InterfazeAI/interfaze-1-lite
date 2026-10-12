"""Translation: interfaze's prompts, chunking and language list, served by the brain.

Ported from interfaze's `helpers/translation.ts` and the `translate` tool (`tools/index.ts`,
`tools/helpers.ts`). Only the model differs: interfaze calls a hosted LLM, lite its own brain.
Pure: the tool supplies the brain.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

MAX_CHARS_PER_CHUNK = 5000
BATCH_SIZE = 100
MAX_OUTPUT_TOKENS = 5000

SCHEMA = {
    "name": "translation",
    "schema": {
        "type": "object",
        "description": "An object output of the translation",
        "properties": {"translated_text": {"type": "string", "description": "The translated text"}},
        "required": ["translated_text"],
        "additionalProperties": False,
    },
}

_ONE_WORD = (
    "You are a professional dictionary. Don't translate acronyms, abbreviations, "
    "brand names and proper nouns."
)
_SENTENCE = (
    "You are a professional translation engine. Maintain the original meaning, tone "
    "and structure of the text. Text may contain multiple languages. Translate all "
    "of them to the target language. Don't translate acronyms, abbreviations, brand "
    "names and proper nouns."
)


@lru_cache(maxsize=1)
def languages() -> dict[str, dict]:
    """Code -> {code, name, native, rtl?, writing_system}: interfaze's `data/languages.json`."""
    path = Path(__file__).with_name("data") / "languages.json"
    return {entry["code"]: entry for entry in json.loads(path.read_text(encoding="utf-8"))}


def unsupported(code: str) -> str:
    """The error interfaze returns for a target code outside its list."""
    return (
        f'"{code}" is not a supported language code. Use a valid ISO 639-1 two-letter '
        "code (e.g., 'es', 'fr', 'ja', 'zh')."
    )


def prompts(text: str, target_language: str, current_language: str | None = None) -> tuple[str, str]:
    """(system, user) for one chunk, worded exactly as interfaze's `runTranslate` words them."""
    langs = languages()
    current = langs.get(current_language or "")
    current_key = f"{current['name']} ({current['code']})" if current else None
    target = langs.get(target_language)
    target_key = f"{target['name']} ({target['code']})" if target else None
    rtl = bool(target and target.get("rtl") == 1)

    one_word = len(text.split(" ")) == 1
    system = (_ONE_WORD if one_word else _SENTENCE) + " Translate numerical value to the target language."
    user = "Translate the following word" if one_word else "Translate the following text"
    if current_key:
        user += ' from "' + current_key + '"'
    user += ' to "' + str(target_key) + '"'
    if rtl:
        user += " in right to left (RTL) direction"
    # The template literal's line break and indent are part of interfaze's prompt.
    user += "\n    \n\nOriginal Text: " + text + "\nTranslated Text:"
    return system, user


def split_into_chunks(text: str, max_len: int = MAX_CHARS_PER_CHUNK) -> tuple[list[str], list[str]]:
    """Chunks of at most `max_len`, cut at a paragraph, sentence or word where possible.

    Also returns each cut's separator: translations come back trimmed, and joining on "" fused words.
    """
    if len(text) <= max_len:
        return [text], []
    chunks: list[str] = []
    separators: list[str] = []
    remaining = text
    while len(remaining) > max_len:
        cut, used = -1, ""
        for sep in ("\n\n", "\n", ". ", "? ", "! ", " "):
            # JS lastIndexOf(sep, maxLen - sep.length): the last match that ends by max_len.
            idx = remaining.rfind(sep, 0, max_len)
            if idx > max_len * 0.5:
                cut, used = idx, sep
                break
        if cut <= 0:
            cut, used = max_len, ""
        chunks.append(remaining[:cut])
        separators.append(used)
        remaining = remaining[cut + len(used) :]
    if remaining:
        chunks.append(remaining)
    return chunks, separators


def stitch(translated: list[str], separators: list[str]) -> str:
    """One item's translated chunks, re-joined on the separators they were cut at."""
    out = translated[0]
    for separator, piece in zip(separators, translated[1:], strict=False):
        out += (separator or " ") + piece
    return out
