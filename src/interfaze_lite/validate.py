"""Request validation for /v1/chat/completions: a 4xx naming the bad field, before any GPU time is spent."""

from __future__ import annotations

import base64
import binascii
import hmac
import re
from typing import Any

TASKS = ("ocr", "stt", "object_detection", "gui_detection", "translate", "forecast")

# Both names are published: transcription is routed as `speech_to_text` but reports itself as `stt`.
TASK_ALIASES = {"speech_to_text": "stt"}

_TASK_TAG = re.compile(r"<task>(.*?)</task>", re.IGNORECASE | re.DOTALL)

# `developer` is OpenAI's newer `system`, sent by the SDKs for recent models; collapsed as interfaze does.
ROLE_ALIASES = {"developer": "system"}
ROLES = ("system", "user", "assistant", "tool", *ROLE_ALIASES)


def normalise_roles(messages: list[dict]) -> list[dict]:
    """Collapse role aliases first: the task tag and system prompt are found by `role == "system"`."""
    if not any(m.get("role") in ROLE_ALIASES for m in messages if isinstance(m, dict)):
        return messages
    return [
        {**m, "role": ROLE_ALIASES[m["role"]]} if isinstance(m, dict) and m.get("role") in ROLE_ALIASES else m
        for m in messages
    ]


class RequestError(Exception):
    """A bad request. `status` is the HTTP code to answer with."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _texts(message: dict) -> list[str]:
    content = message.get("content")
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
    return []


def extract_task(messages: list[dict]) -> str | None:
    """Pull the single `<task>name</task>` directive from the system turns; the SDK's `task=` lowers to it."""
    found: list[str] = []
    for message in messages:
        if message.get("role") != "system":
            continue
        for text in _texts(message):
            found.extend(_TASK_TAG.findall(text))

    if not found:
        return None
    if len(found) > 1:
        raise RequestError("only one task may be specified per request")

    names = [n.strip() for n in found[0].split(",") if n.strip()]
    if len(names) > 1:
        raise RequestError(
            f"only one task may be specified per request, got {len(names)}: " + ", ".join(names)
        )
    if not names:
        # An empty `<task></task>` routes nothing, as if omitted; rejecting it 400'd empty template slots.
        return None

    task = names[0].lower()
    if task not in TASKS and task not in TASK_ALIASES:
        raise RequestError(f"invalid task '{task}'. Supported tasks: " + ", ".join((*TASKS, *TASK_ALIASES)))
    return task


def resolve_task(task: str) -> str:
    return TASK_ALIASES.get(task, task)


def strip_task_tags(messages: list[dict]) -> list[dict]:
    """Remove the directive once it has been read, so it never reaches the model."""
    out: list[dict] = []
    for message in messages:
        if message.get("role") != "system":
            out.append(message)
            continue
        content = message.get("content")
        if isinstance(content, str):
            out.append({**message, "content": _TASK_TAG.sub("", content).strip()})
        elif isinstance(content, list):
            out.append(
                {
                    **message,
                    "content": [
                        {**p, "text": _TASK_TAG.sub("", p["text"]).strip()}
                        if isinstance(p, dict) and p.get("type") == "text" and "text" in p
                        else p
                        for p in content
                    ],
                }
            )
        else:
            out.append(message)
    return out


def _check_data_uri(url: str) -> None:
    """Reject an undecodable data: URI here; downstream it surfaced as a misleading unreadable-image error."""
    if not url.startswith("data:"):
        return
    header, separator, payload = url.partition(",")
    if not separator:
        raise RequestError("malformed data URI: missing ',' separator")
    if "base64" not in header:
        return
    payload = payload.strip()
    if not payload:
        raise RequestError("malformed data URI: empty base64 payload")
    try:
        base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise RequestError("invalid base64 payload in data URI") from None


def validate_messages(messages: Any) -> None:
    if not isinstance(messages, list) or not messages:
        raise RequestError("`messages` must be a non-empty array")

    has_content = False
    for message in messages:
        if not isinstance(message, dict):
            raise RequestError("each message must be an object")
        if message.get("role") not in ROLES:
            raise RequestError(f"unsupported message role: {message.get('role')!r}")

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            has_content = True
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    raise RequestError("each content part must be an object")
                kind = part.get("type")
                if kind == "text" and str(part.get("text", "")).strip():
                    has_content = True
                elif kind in ("image_url", "input_image"):
                    has_content = True
                    url = (
                        (part.get("image_url") or {}).get("url")
                        if kind == "image_url"
                        else part.get("image_url")
                    )
                    if isinstance(url, str):
                        # Anything else is a path on this server, which the tools would read.
                        if not url.startswith(("http://", "https://", "data:")):
                            raise RequestError("image_url must be an http(s) URL or a data: URI")
                        _check_data_uri(url)
                elif kind in ("input_audio", "audio_url", "file", "file_url", "video_url"):
                    has_content = True
        if message.get("tool_calls"):
            has_content = True

    if not has_content:
        raise RequestError("request has no text content: every message is empty")


def check_api_key(
    header: str | None,
    expected: str | None,
    admin_header: str | None = None,
    admin_expected: str | None = None,
) -> None:
    """Enforce a shared key only when one is configured; Settings.api_key says why it ships open.

    `header` is the Authorization header, matched against API_KEY with or without a
    `Bearer ` prefix; `admin_header` is `x-api-admin-key`, matched against ADMIN_KEY.
    With either key set, a request must carry one that matches.
    """
    if not expected and not admin_expected:
        return
    token = ""
    if header and header.lower().startswith("bearer "):
        token = header[7:].strip()
    elif header:
        token = header.strip()
    if expected and token and hmac.compare_digest(token.encode(), expected.encode()):
        return
    admin = (admin_header or "").strip()
    if admin_expected and admin and hmac.compare_digest(admin.encode(), admin_expected.encode()):
        return
    raise RequestError("invalid API key", status=401)


# OpenAI's documented ranges; past them the inference server clamped silently or failed with a 500.
_RANGES: dict[str, tuple[float, float]] = {
    "temperature": (0.0, 2.0),
    "top_p": (0.0, 1.0),
    "frequency_penalty": (-2.0, 2.0),
    "presence_penalty": (-2.0, 2.0),
}


def validate_sampling(body: dict, *, max_output_tokens: int) -> None:
    """Reject sampling parameters the server cannot honour, before generating."""
    for name, (low, high) in _RANGES.items():
        value = body.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RequestError(f"{name} must be a number")
        if not low <= float(value) <= high:
            raise RequestError(f"{name} must be between {low} and {high}, got {value}")

    tokens = body.get("max_tokens")
    if body.get("max_completion_tokens") is not None:
        tokens = body["max_completion_tokens"]
    if tokens is None:
        return
    if isinstance(tokens, bool) or not isinstance(tokens, int):
        raise RequestError("max_tokens must be an integer")
    if tokens < 1:
        raise RequestError(f"max_tokens must be at least 1, got {tokens}")
    if tokens > max_output_tokens:
        raise RequestError(f"max_tokens must be at most {max_output_tokens}, got {tokens}")
