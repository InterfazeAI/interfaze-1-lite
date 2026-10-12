"""Translation, served by the brain itself.

The tool interfaze exposes (`tools/index.ts` `translate`), with the same parameters,
validation, chunking and result -- but the model it calls is this deployment's own
brain rather than a hosted one. The brain calls this tool, and the tool calls the
brain back with interfaze's translation prompts and a one-field schema.
"""

from __future__ import annotations

import asyncio
import json

from .. import translation
from .base import Tool, ToolContext, ToolResult

# The brain batches concurrent requests; a long document's chunks are sent together,
# bounded so one translation cannot occupy every sequence slot.
_CONCURRENCY = 8
_ATTEMPTS = 3


async def _translate_chunk(
    chunk: str, target: str, current: str | None, ctx: ToolContext, gate: asyncio.Semaphore
) -> str:
    from ..brain import Sampling

    system, user = translation.prompts(chunk, target, current)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last: Exception | None = None
    async with gate:
        for _ in range(_ATTEMPTS):
            try:
                reply = await ctx.structured(
                    messages,
                    translation.SCHEMA,
                    Sampling(max_tokens=translation.MAX_OUTPUT_TOKENS, temperature=0),
                )
                if ctx.usage is not None:
                    ctx.usage.prompt_tokens += reply.prompt_tokens
                    ctx.usage.completion_tokens += reply.completion_tokens
                return str(json.loads(reply.content)["translated_text"]).strip()
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                last = exc
    raise RuntimeError(f"translation did not return a translated_text: {last}") from last


async def _run_translate(args: dict, ctx: ToolContext) -> ToolResult:
    text = args.get("text")
    target = str(args.get("target_language") or "").strip()
    current = str(args.get("current_language") or "").strip() or None

    if not text or (isinstance(text, list) and not all(isinstance(t, str) and t for t in text)):
        return ToolResult(
            model_facing={
                "error": "Text is required",
                "message": "Pass the text to translate as a string or a list of strings.",
            }
        )
    if current and current == target:
        # interfaze's own validation rule. It is also what stops the model "translating"
        # a passage into the language it is already in.
        return ToolResult(
            model_facing={
                "error": "Source and target language cannot be the same",
                "message": (
                    "The text is already in the target language, so there is nothing to "
                    "translate. Do not call translate again; answer the user directly."
                ),
            }
        )
    if target not in translation.languages():
        return ToolResult(model_facing={"error": translation.unsupported(target)})

    try:
        was_array = isinstance(text, list)
        items = list(text) if was_array else [str(text)]
        flat: list[str] = []
        per_item: list[int] = []
        separators: list[list[str]] = []
        for item in items:
            chunks, seps = translation.split_into_chunks(item, translation.MAX_CHARS_PER_CHUNK)
            per_item.append(len(chunks))
            separators.append(seps)
            flat.extend(chunks)

        gate = asyncio.Semaphore(_CONCURRENCY)
        translated: list[str] = []
        for start in range(0, len(flat), translation.BATCH_SIZE):
            batch = flat[start : start + translation.BATCH_SIZE]
            translated += await asyncio.gather(
                *(_translate_chunk(c, target, current, ctx, gate) for c in batch)
            )

        out: list[str] = []
        cursor = 0
        for count, seps in zip(per_item, separators, strict=True):
            out.append(translation.stitch(translated[cursor : cursor + count], seps))
            cursor += count

        return ToolResult(
            model_facing={
                "translated_text": out if was_array else out[0],
                "source_language": current or "auto-detected",
                "target_language": target,
                "batch_size": len(items),
                "chunks_processed": len(flat),
            }
        )
    except Exception as exc:
        return ToolResult(
            model_facing={
                "error": str(exc) or type(exc).__name__,
                "message": "Translation failed. Inspect the error before retrying — do not repeat "
                "the same call unchanged.",
            }
        )


TRANSLATE = Tool(
    name="translate",
    description=(
        "Translate text from one language to another. Supports 160+ languages. Accepts "
        "either a single string of any length or an array of strings for batch translation "
        "— long inputs are automatically split and batched internally, so you do not need "
        "to chunk. Use only when the user asks for text to be translated into another "
        "language; never to read, answer or reason about text that is merely written in "
        "another language."
    ),
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ],
                "description": (
                    "The text to translate. Pass a single string of any length, or an "
                    "array of strings for batch translation. Long text is split and "
                    "batched internally — pass the full text as-is."
                ),
            },
            "target_language": {
                "type": "string",
                "description": (
                    "The ISO 639-1 two-letter target language code (e.g., 'es' for "
                    "Spanish, 'fr' for French, 'ja' for Japanese, 'ar' for Arabic, "
                    "'zh' for Chinese)."
                ),
            },
            "current_language": {
                "type": "string",
                "description": (
                    "The ISO 639-1 two-letter code of the language the text is "
                    "written in now. Required: when it is the same as "
                    "target_language there is nothing to translate."
                ),
            },
        },
        "required": ["text", "target_language", "current_language"],
        "additionalProperties": False,
    },
    execute=_run_translate,
)
