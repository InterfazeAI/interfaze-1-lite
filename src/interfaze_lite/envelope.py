"""Assemble OpenAI-compatible chat.completion payloads with the interfaze extensions.

Interfaze returns a standard OpenAI envelope plus one non-standard top-level field,
`precontext`, carrying the structured results of every internal tool that ran. That
split matters: `choices[].message.content` is prose for a human, `precontext` is the
machine payload (OCR sections with bounds, transcript chunks, detected objects).

Two rules are easy to break in a rewrite and both break real clients:

1. On a tool-call turn, `content` MUST be null. Populating it -- even with a helpful
   summary of the tool results -- makes OpenAI-compatible agent runtimes treat the turn
   as a final answer and abandon the loop.

2. Streaming cannot carry top-level fields, because clients drop unknown keys on
   chunks. Interfaze therefore smuggles precontext through the content stream wrapped
   in literal `<precontext> ... </precontext>` sentinel tags. Clients parse those out,
   so the format is reproduced verbatim rather than improved.

`vcache` and the `cached_tokens` usage details are carried as interfaze carries them,
always present, though this service has no vector cache and they are always false and
0: a caller written against interfaze reads them, and a missing key is not an answer.
Token counts come from the real vLLM response rather than billing math.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

FinishReason = Literal["stop", "length", "tool_calls", "content_filter"]

DEFAULT_MODEL_NAME = "interfaze-lite"

PRECONTEXT_OPEN = "<precontext>"
PRECONTEXT_CLOSE = "</precontext>"


def new_request_id() -> str:
    return f"req-{uuid.uuid4()}"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # already-serialised JSON, per the OpenAI spec

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Counted inside completion_tokens, never in addition to them -- reporting the sum
    # of the two as the total would bill thinking twice.
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            # Always present, even at zero: an absent key reads as a client-side mapping bug.
            "completion_tokens_details": {"reasoning_tokens": self.reasoning_tokens, "cached_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0},
        }


@dataclass
class PrecontextItem:
    """One internal tool's full result, as the caller receives it."""

    name: str
    result: Any

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "result": self.result}


@dataclass
class Completion:
    request_id: str
    model: str = DEFAULT_MODEL_NAME
    created: int = field(default_factory=lambda: int(time.time()))
    usage: Usage = field(default_factory=Usage)
    precontext: list[PrecontextItem] = field(default_factory=list)
    reasoning: str | None = None
    debug: dict[str, Any] | None = None

    def _base(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.request_id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
        }
        if self.debug is not None:
            out["debug"] = self.debug
        if self.reasoning:
            out["reasoning"] = self.reasoning
        out["usage"] = self.usage.as_dict()
        # ALWAYS present, even as []: a typed SDK reports a missing attribute, not an empty one.
        out["precontext"] = [p.as_dict() for p in self.precontext]
        out["vcache"] = False
        return out

    def message(self, content: str, finish_reason: FinishReason = "stop") -> dict[str, Any]:
        return {
            **self._base(),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
        }

    def tool_calls(self, calls: list[ToolCall], context: list[dict] | None = None) -> dict[str, Any]:
        """A tool-call turn. `content` is null -- see the module docstring.

        `context` is the request's own tool calls and results before it handed these back,
        as messages a caller can send again with its answers (`interfaze_context`).
        """
        return {
            **self._base(),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [c.as_dict() for c in calls],
                        **({"interfaze_context": context} if context else {}),
                    },
                    "logprobs": None,
                    "finish_reason": "tool_calls",
                }
            ],
        }


def _chunk(
    request_id: str,
    model: str,
    created: int,
    delta: dict[str, Any],
    finish_reason: FinishReason | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        out["usage"] = usage
    return out


def sse(payload: dict[str, Any] | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def precontext_delta(completion: Completion) -> dict[str, Any]:
    """The sentinel-tagged precontext chunk.

    Reproduces interfaze byte-for-byte, including the spaces inside the tags:
    `<precontext> {...} </precontext>`. Clients grep for these, so the spacing is part
    of the contract. It lands in `content`, so send it only to a caller that asked for it.
    """
    body = json.dumps([p.as_dict() for p in completion.precontext], separators=(",", ":"), ensure_ascii=False)
    return _chunk(
        completion.request_id,
        completion.model,
        completion.created,
        {"content": f"{PRECONTEXT_OPEN} {body} {PRECONTEXT_CLOSE}"},
    )


def tool_call_deltas(completion: Completion, calls: list[ToolCall]) -> Iterator[dict[str, Any]]:
    """A tool-call turn, chunked. `index` reassembles split arguments; required even for whole calls."""
    for index, call in enumerate(calls):
        yield _chunk(
            completion.request_id,
            completion.model,
            completion.created,
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                ],
            },
        )


def stream(
    completion: Completion, content_parts: list[str], finish_reason: FinishReason = "stop"
) -> Iterator[str]:
    """Emitting the role in its own leading chunk with empty content is what the OpenAI
    SDKs expect; some clients use it to open the message before any text arrives.
    """
    yield sse(
        _chunk(
            completion.request_id, completion.model, completion.created, {"role": "assistant", "content": ""}
        )
    )

    yield sse(precontext_delta(completion))

    for part in content_parts:
        if part:
            yield sse(_chunk(completion.request_id, completion.model, completion.created, {"content": part}))

    yield sse(
        _chunk(
            completion.request_id,
            completion.model,
            completion.created,
            {},
            finish_reason=finish_reason,
            usage=completion.usage.as_dict(),
        )
    )
    yield sse("[DONE]")


def parse_precontext(stream_text: str) -> list[dict[str, Any]] | None:
    """Recover precontext from a reassembled content stream.

    Inverse of `precontext_delta`. Exists so tests can assert round-tripping without
    reimplementing the parse, and so callers have a reference for how to strip the
    sentinel block out of user-visible text.
    """
    start = stream_text.find(PRECONTEXT_OPEN)
    if start == -1:
        return None
    end = stream_text.find(PRECONTEXT_CLOSE, start)
    if end == -1:
        return None
    body = stream_text[start + len(PRECONTEXT_OPEN) : end].strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def strip_precontext(stream_text: str) -> str:
    start = stream_text.find(PRECONTEXT_OPEN)
    if start == -1:
        return stream_text
    end = stream_text.find(PRECONTEXT_CLOSE, start)
    if end == -1:
        return stream_text
    return (stream_text[:start] + stream_text[end + len(PRECONTEXT_CLOSE) :]).lstrip()
