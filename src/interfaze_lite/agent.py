"""The tool loop the transformers model runs for `chat()`, free of torch so it can be tested.

The service runs the same loop over vLLM (app.py): the brain sees the caller's files by
reference, calls tools, reads what they return -- and, once a tool has read an image, the
image itself -- then answers. Both brief the brain with the same prompts.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Sequence
from typing import Any

try:  # the interfaze_lite package, and the HF repo as trust_remote_code imports it
    from .prompts import NUDGE, SYSTEM_PROMPT
except ImportError:  # the files run flat, from a checkout
    from prompts import NUDGE, SYSTEM_PROMPT  # type: ignore

_FILE = {"type": "string", "description": "The file's reference id from All File References, e.g. ref-0."}

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "ocr",
            "description": "Read an image, PDF or Word document the user supplied: its full text, "
            "with every line's position.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_ref_id": _FILE,
                    "page_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "First and last page to read, 1-based, for a long PDF.",
                    },
                },
                "required": ["file_ref_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stt",
            "description": "Transcribe an audio file, optionally labelling who said what.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_ref_id": _FILE,
                    "by_speaker": {"type": "boolean", "description": "Split the transcript by speaker."},
                },
                "required": ["file_ref_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "object_detection",
            "description": "Find objects in a photo: a box and label for each.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_ref_id": _FILE,
                    "prompts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What to find, as simple noun phrases, e.g. ['person', 'car'].",
                    },
                },
                "required": ["file_ref_id", "prompts"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gui_detection",
            "description": "Find interactive elements in a UI screenshot -- buttons, inputs, links -- "
            "each named by its visible label.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_ref_id": _FILE,
                    "prompts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Elements to find; leave out for every interactive element.",
                    },
                },
                "required": ["file_ref_id"],
            },
        },
    },
]

# The model reads this much of a tool result; the caller gets all of it.
MAX_CONTEXT_CHARS = 120_000

_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_FUNCTION = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
_PARAMETER = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.S)


def tool_calls(text: str) -> list[dict[str, Any]]:
    """The tool calls in a brain turn, in either form the chat template produces.

    `<function=ocr><parameter=file_ref_id>ref-0</parameter></function>` is what this
    brain's template emits; `{"name": ..., "arguments": {...}}` is the older JSON form.
    Both sit inside <tool_call> tags. A parameter that reads as JSON -- a list, a number
    -- is taken as that value.
    """
    calls = []
    for body in _TOOL_CALL.findall(text or ""):
        function = _FUNCTION.search(body)
        if function:
            arguments = {}
            for name, value in _PARAMETER.findall(function.group(2)):
                value = value.strip()
                try:
                    arguments[name] = json.loads(value)
                except json.JSONDecodeError:
                    arguments[name] = value
            calls.append({"name": function.group(1), "arguments": arguments})
            continue
        try:
            obj = json.loads(body.strip())
            arguments = obj.get("arguments") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            calls.append({"name": obj["name"], "arguments": arguments})
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            continue
    return calls


def visible(text: str) -> str:
    """A turn's answer, without any tool-call markup or thinking."""
    text = _TOOL_CALL.sub("", text or "")
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def run(
    messages: list[dict],
    files: Sequence[str],
    *,
    generate: Callable[..., str],
    tools: dict[str, Callable[[str, dict], tuple[dict, dict]]],
    show_images: Callable[[list[dict], dict[str, str]], list[dict]],
    max_tool_steps: int = 6,
) -> dict[str, Any]:
    """Answer `messages`, calling `tools` on `files` as the brain asks.

    `generate(convo, tools=..., keep_markup=...)` is one brain turn. Each tool takes the
    file and the call's arguments and returns what the caller gets and what the model
    reads. Returns `{"content", "precontext": [{name, result}]}`.
    """
    refs = {f"ref-{i}": str(f) for i, f in enumerate(files)}
    system = SYSTEM_PROMPT
    if refs:
        system += "\n\nAll File References:\n" + "\n".join(
            f"- {ref}: {os.path.basename(path.split('?')[0]) or path}" for ref, path in refs.items()
        )
    convo = [{"role": "system", "content": system}, *messages]
    precontext: list[dict[str, Any]] = []
    used = nudged = shown = False

    for _ in range(max_tool_steps):
        reply = generate(convo, tools=TOOLS if refs else None, keep_markup=True)
        calls = tool_calls(reply)
        if not calls:
            # Attached but unread: an answer by eye invents the text. One reminder, as in the service.
            if refs and not used and not nudged:
                nudged = True
                convo = [*convo, {"role": "user", "content": NUDGE}]
                continue
            return {"content": visible(reply), "precontext": precontext}

        convo.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in calls
                ],
            }
        )
        for call in calls:
            full, seen = _dispatch(call, refs, tools)
            used = True  # A failed call too: it goes back to be corrected, not nudged, as in the service.
            if "error" not in seen:
                precontext.append({"name": call["name"], "result": full})
            convo.append({"role": "tool", "content": json.dumps(seen, ensure_ascii=False)})
        if precontext and not shown:
            convo, shown = show_images(convo, refs), True

    return {"content": visible(generate(convo, tools=None, keep_markup=False)), "precontext": precontext}


def _dispatch(call: dict, refs: dict[str, str], tools: dict) -> tuple[dict, dict]:
    tool = tools.get(call["name"])
    if tool is None:
        error = {"error": f"unknown tool {call['name']!r}; available: {sorted(tools)}"}
        return error, error
    ref = str(call["arguments"].get("file_ref_id") or "")
    # Only the caller's own files: a made-up URL is refused, never fetched.
    path = refs.get(ref) or next((p for p in refs.values() if p == ref), None)
    if path is None:
        error = {"error": f"unknown file reference {ref!r}; use one of {sorted(refs)}"}
        return error, error
    try:
        full, seen = tool(path, call["arguments"])
    except Exception as exc:
        error = {"error": f"{call['name']} failed: {type(exc).__name__}: {exc}"}
        return error, error
    text = seen.get("extracted_text")
    if isinstance(text, str) and len(text) > MAX_CONTEXT_CHARS:
        seen = {**seen, "extracted_text": text[:MAX_CONTEXT_CHARS], "truncated": True}
    return full, seen
