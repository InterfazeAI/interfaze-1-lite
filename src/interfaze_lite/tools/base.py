"""Shared plumbing: the tool context, its result type, and the HTTP helper."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import Settings
from ..contracts import InputFetchError, normalise_error
from ..filerefs import FileRefs


@dataclass
class ToolContext:
    refs: FileRefs
    http: httpx.AsyncClient
    settings: Settings
    results: dict[str, ToolResult] = field(default_factory=dict)
    # Memo key -> how many times executing it has raised this request.
    failures: dict[str, int] = field(default_factory=dict)
    # A backend's answer this request, by what was asked of it, for a tool asked again
    # with different options: one OCR pass serves markdown and line boxes alike.
    cache: dict[str, Any] = field(default_factory=dict)
    # Set by the orchestrator; object and GUI detection ground through the brain rather
    # than a dedicated grounding model.
    ground: Callable[..., Awaitable[dict]] | None = None
    # The brain's schema-constrained completion, for tools the brain serves itself
    # (translation). Returns a BrainReply.
    structured: Callable[..., Awaitable[Any]] | None = None
    # The request's token count. A tool that runs the brain adds what it spent, so the
    # caller is billed for it as interfaze bills its translation calls.
    usage: Any = None
    # The text tools (translate, forecast) this request is offered -- see intent.py.
    text_tools: tuple[str, ...] = ()
    # The latest user message's text. Forecast reads a series written into it, which
    # the model would otherwise copy into its tool call row by row.
    prompt: str = ""
    # The caller asked for zero data retention: nothing of theirs is written to the logs.
    zdr: bool = False
    # The caller's response schema asks for coordinates. Trusted over the model, which often
    # forgets `return_bounds` and then answers a layout question with no geometry.
    wants_geometry: bool = False


@dataclass
class ToolResult:
    model_facing: dict[str, Any]
    full: dict[str, Any] | None = None

    @property
    def precontext(self) -> dict[str, Any]:
        return self.full if self.full is not None else self.model_facing


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[[dict, ToolContext], Awaitable[ToolResult]]

    def as_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                # Load-bearing: under tool_choice="auto" vLLM grammar-constrains arguments only
                # for strict tools; otherwise the JSON is free-form (structural_tag_registry.py:115-118).
                "strict": True,
            },
        }


_MAX_TOOL_ATTEMPTS = 2


def _memo_key(tool: Tool, args: dict) -> str:
    defaults = {
        name: spec["default"]
        for name, spec in (tool.parameters.get("properties") or {}).items()
        if "default" in spec
    }
    effective = {**defaults, **{k: v for k, v in args.items() if v is not None}}
    return tool.name + "|" + json.dumps(effective, sort_keys=True, default=str)


log = logging.getLogger("interfaze.tools")

# What each internal service does, in the words a caller would use.
_CAPABILITIES = {
    "/transcribe": "speech to text",
    "/diarize": "speaker detection",
    "/ocr": "OCR",
    "/segment": "segmentation",
    "/forecast": "forecasting",
    "/guard": "the guardrail check",
}


async def _post(ctx: ToolContext, base: str, path: str, payload: dict) -> dict:
    resp = await ctx.http.post(f"{base.rstrip('/')}{path}", json=payload, timeout=ctx.settings.tool_timeout_s)
    if resp.status_code >= 400:
        detail = (
            resp.json().get("detail", resp.text)
            if resp.headers.get("content-type", "").startswith("application/json")
            else resp.text
        )
        if isinstance(detail, dict) and "input_fetch" in detail:
            raise InputFetchError(detail["input_fetch"])
        if resp.status_code >= 500:
            # A backend failure, reported in general terms. The detail goes to the logs: passed
            # on, a CUDA out-of-memory message with the card's allocation figures reached the landing.
            log.error("%s returned %s: %s", path, resp.status_code, str(detail)[:2000])
            raise RuntimeError(
                f"Something went wrong running {_CAPABILITIES.get(path, 'this tool')}. Please try again."
            )
        # A 4xx describes the caller's input, so it is passed on -- restated in terms of
        # capabilities rather than the components behind them.
        raise RuntimeError(
            f"{path} returned {resp.status_code}: "
            f"{normalise_error(str(detail), ctx.settings.component_names)}"
        )
    return resp.json()
