"""Client for the vLLM-served brain, and the multi-stage loop around it.

Interfaze runs four brain stages: route, call tools, synthesise, shape to schema.
Lite collapses the first two. Routing in interfaze picks *which provider* to use --
gpt-5 versus gemini-flash versus glm -- and that decision does not exist here, because
there is one model. What remains of stage one is tool selection, which the tool-calling
call already does.

Three stages, and each uses a different vLLM mode for a specific reason:

  select_tools        streamed for a streamed request, tools passed, the caller's tool_choice
  synthesise[_stream] NO tools passed
  structured          response_format json_schema against the caller's schema

Tool turns stream too: the raw <tool_call> XML vLLM once leaked into streamed content is gone on 0.27.1.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import logging
import mimetypes
import pathlib
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field, replace
from typing import Any

import httpx

from . import grounding
from .config import Settings
from .contracts import InputFetchError
from .envelope import ToolCall

log = logging.getLogger("interfaze.brain")


class ContextLengthError(ValueError):
    """The request is longer than the model's context window."""


class BrainError(RuntimeError):
    pass


# The template raises on "high", which every SDK sends; xhigh answers empty about one call in
# six, so upstream guidance resolves high to medium.
_EFFORT = {"low": "low", "medium": "medium", "high": "medium", "xhigh": "xhigh"}


def thinking_kwargs(think: bool, effort: str | None = None) -> dict[str, Any]:
    """Chat-template switches for a thinking turn.

    The checkpoint never closes the `<think>` block its template opens, so vLLM routes it all to
    content and `reasoning` comes back empty -- upstream; tool parser, effort and temperature ruled out.
    """
    if not think:
        return {"enable_thinking": False}
    return {
        "enable_thinking": True,
        "reasoning_effort": _EFFORT.get((effort or "").lower(), "medium"),  # Not the template's xhigh.
    }


STRUCTURED_MAX_TEMPERATURE = 0.3  # Caps a structured request's tool turns and schema fill, whatever asked.


@dataclass
class Sampling:
    """What the caller asked for, kept apart from the defaults so its `max_tokens` truncates with "length"."""

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None

    def apply(self, payload: dict[str, Any], default_max: int) -> dict[str, Any]:
        payload["max_tokens"] = self.max_tokens or default_max
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        return payload


@dataclass
class BrainReply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    reasoning: str = ""
    reasoning_tokens: int = 0


class BrainClient:
    def __init__(self, http: httpx.AsyncClient, settings: Settings):
        self.http = http
        self.settings = settings

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = await self.http.post(
            self.settings.brain_chat_url,
            json={"model": self.settings.brain_served_name, **payload},
            timeout=self.settings.request_timeout_s,
        )
        if resp.status_code == 400 and "maximum context length" in resp.text:
            # Too long for the model: the caller's error, which a 502 told them to retry.
            try:
                message = resp.json().get("error", {}).get("message") or resp.text
            except ValueError:
                message = resp.text
            raise ContextLengthError(message[:800])
        if resp.status_code >= 400:
            raise BrainError(f"brain returned {resp.status_code}: {resp.text[:800]}")
        return resp.json()

    async def _events(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """A streamed completion, one parsed server-sent event at a time.

        Consume it under `contextlib.aclosing`: closing it closes the connection, which
        is how a caller that stops early makes vLLM abort the generation.
        """
        async with self.http.stream(
            "POST",
            self.settings.brain_chat_url,
            json={"model": self.settings.brain_served_name, **payload, "stream": True},
            timeout=self.settings.request_timeout_s,
        ) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode(errors="replace")[:800]
                raise BrainError(f"brain returned {resp.status_code}: {body}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue

    @staticmethod
    def _parse(body: dict[str, Any]) -> BrainReply:
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = body.get("usage") or {}

        calls = [
            ToolCall(
                id=tc.get("id") or f"call_{i}",
                name=(tc.get("function") or {}).get("name", ""),
                arguments=(tc.get("function") or {}).get("arguments") or "{}",
            )
            for i, tc in enumerate(message.get("tool_calls") or [])
        ]
        # --reasoning-parser qwen3 splits thinking out; empty content falls back to it rather than "".
        reasoning = message.get("reasoning_content") or ""
        content = message.get("content") or ""
        if not content.strip() and reasoning.strip():
            content, reasoning = reasoning, ""
        return BrainReply(
            content=content,
            reasoning=reasoning,
            tool_calls=calls,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason") or "stop",
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
        )

    async def select_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        sampling: Sampling | None = None,
        *,
        tool_choice: str | dict = "auto",
        on_text: Callable[[str], None] | None = None,
    ) -> BrainReply:
        """One turn of the tool loop: it answers, or it asks for tools.

        With `on_text` the turn streams, each piece handed over as generated, then reassembled for `_parse`.
        """
        payload = (sampling or Sampling()).apply(
            {
                "messages": messages,
                "temperature": 0,
                # Thinking is on by default; choosing a tool is not a reasoning task.
                "chat_template_kwargs": {"enable_thinking": False},
                **({"tools": tools, "tool_choice": tool_choice} if tools else {}),
            },
            self.settings.max_new_tokens,
        )
        if on_text is None:
            return self._parse(await self._post({**payload, "stream": False}))

        content, reasoning, finish, usage = [], [], None, {}
        calls: dict[int, dict] = {}
        async with contextlib.aclosing(
            self._events({**payload, "stream_options": {"include_usage": True}})
        ) as events:
            async for chunk in events:
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        content.append(delta["content"])
                        on_text(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning.append(delta["reasoning_content"])
                    for tc in delta.get("tool_calls") or []:
                        slot = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                        slot["id"] = tc.get("id") or slot["id"]
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["args"] += fn.get("arguments") or ""
                    finish = choice.get("finish_reason") or finish
        message = {
            "content": "".join(content),
            "reasoning_content": "".join(reasoning),
            "tool_calls": [
                {"id": c["id"], "function": {"name": c["name"], "arguments": c["args"]}}
                for _, c in sorted(calls.items())
            ],
        }
        return self._parse({"choices": [{"message": message, "finish_reason": finish}], "usage": usage})

    async def synthesise(
        self,
        messages: list[dict],
        *,
        think: bool = False,
        effort: str | None = None,
        sampling: Sampling | None = None,
    ) -> BrainReply:
        """Answer, thinking only when asked: unasked, its tokens are latency, often more than the answer."""
        sampling = sampling or Sampling()
        payload = sampling.apply(
            {
                "messages": messages,
                "stream": False,
                "temperature": 0,
                "chat_template_kwargs": thinking_kwargs(think, effort),
            },
            self.settings.max_new_tokens,
        )
        return self._parse(await self._post(payload))

    async def synthesise_stream(
        self,
        messages: list[dict],
        *,
        think: bool = False,
        effort: str | None = None,
        sampling: Sampling | None = None,
        on_finish: Callable[[str], None] | None = None,
        on_usage: Callable[[dict], None] | None = None,
    ) -> AsyncIterator[str]:
        """Stream the final answer, with no tools passed.

        Thinking is re-wrapped in <think> tags: the side channel the interfaze SDKs strip into `.reasoning`.
        """
        sampling = sampling or Sampling()
        payload = sampling.apply(
            {
                "messages": messages,
                "temperature": 0,
                "chat_template_kwargs": thinking_kwargs(think, effort),
                # Without it a streamed turn reports zero tokens.
                "stream_options": {"include_usage": True},
            },
            self.settings.max_new_tokens,
        )
        thinking = False
        async with contextlib.aclosing(self._events(payload)) as events:
            async for chunk in events:
                choice = (chunk.get("choices") or [{}])[0]
                # Only stated here; assuming "stop" hid truncation from a caller who set max_tokens.
                if choice.get("finish_reason") and on_finish:
                    on_finish(choice["finish_reason"])
                if chunk.get("usage") and on_usage is not None:
                    on_usage(chunk["usage"])
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content"):
                    if not thinking:
                        thinking = True
                        yield "<think>"
                    yield delta["reasoning_content"]
                if delta.get("content"):
                    if thinking:
                        thinking = False
                        yield "</think>"
                    yield delta["content"]
        # An unclosed <think> makes the SDK's side-channel filter swallow everything after it.
        if thinking:
            yield "</think>"

    async def structured(
        self, messages: list[dict], schema: dict, sampling: Sampling | None = None
    ) -> BrainReply:
        """Emit an object conforming to `schema`, by guided decoding: valid by construction, <6% latency."""
        sampling = sampling or Sampling()
        if sampling.temperature is not None and sampling.temperature > STRUCTURED_MAX_TEMPERATURE:
            # Sampling only loses or invents: at 1, a paper's section list came back empty on 2 runs of 11.
            sampling = replace(sampling, temperature=STRUCTURED_MAX_TEMPERATURE)
        return self._parse(
            await self._post(
                sampling.apply(
                    {
                        "messages": messages,
                        "stream": False,
                        "temperature": 0,
                        "response_format": {"type": "json_schema", "json_schema": schema},
                        # No frequency_penalty: JSON is all repeats, and it kept a receipt inside
                        # a string on 12 of 18 runs, 0 without. max_structured_tokens bounds a loop.
                        # With thinking on, --reasoning-parser qwen3 empties `content`.
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    self.settings.max_structured_tokens,
                )
            )
        )

    async def _bounded_image(
        self, image_url: str, budget: int, *, inline: bool = False, floor: int = 0
    ) -> tuple[str, tuple[int, int] | None]:
        """The image within `budget` pixels (scaled up to it under `floor`), and its size for the 0-1000 grid.

        Capped here because some processors ignore vLLM's `--mm-processor-kwargs max_pixels`;
        why each budget is what it is sits with `Settings.ground_max_pixels` and `object_ground_max_pixels`.
        """
        if budget <= 0:
            return image_url, None

        try:
            raw = await self._image_bytes(image_url)
            # Decode and resize are CPU-bound: hundreds of milliseconds on a large page.
            return await asyncio.to_thread(_shrink_to_budget, raw, budget, image_url, inline, floor)
        except InputFetchError:
            raise
        except Exception as exc:
            # A resize failure must not cost the caller their detection.
            log.warning("could not bound grounding image (%s); sending as-is", exc)
            return image_url, None

    async def _image_bytes(self, image_url: str) -> bytes:
        """The image itself, from a data URI, a URL, or a materialised upload's path."""
        if image_url.startswith("data:"):
            return base64.b64decode(image_url.split(",", 1)[1])
        if not image_url.startswith(("http://", "https://")):
            return await asyncio.to_thread(pathlib.Path(image_url).read_bytes)
        try:
            # The shared async client: a blocking fetch parks the event loop and every request on it.
            # A browser User-Agent, because several CDNs answer "Python-urllib" and friends with 403.
            response = await self.http.get(
                image_url,
                timeout=60,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/125.0 Safari/537.36"
                    ),
                    "Accept": "image/*,*/*",
                },
            )
        except httpx.RequestError as exc:
            raise InputFetchError(f"could not fetch {image_url}: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            raise InputFetchError(f"could not fetch {image_url}: HTTP {response.status_code}")
        return response.content

    async def viewable(
        self, image_url: str, budget: int, *, inline: bool = False, floor: int = 0
    ) -> tuple[str, tuple[int, int] | None]:
        """An image as the vision tower should get it: reachable, and within `budget` pixels.

        vLLM fetches image_url itself, so a local path (a materialised data: URI) is inlined;
        `inline` inlines every image that decodes, so vLLM never fetches on its own.
        """
        if not image_url.startswith(("http://", "https://", "data:")):
            # Off the loop: a materialised upload can be tens of megabytes.
            image_url = await asyncio.to_thread(_inline_local_file, image_url)
        return await self._bounded_image(image_url, budget, inline=inline, floor=floor)

    async def ground(self, image_url: str, prompts: list[str], *, domain: str = "ui") -> dict[str, Any]:
        """Locate things in an image with the brain itself, comparable on ScreenSpot-v2 to a dedicated model.

        `domain` picks the wording: asked for elements "in this UI screenshot", a photograph often got [].
        """
        budget = self.settings.ground_max_pixels if domain == "ui" else self.settings.object_ground_max_pixels
        # A small photo is scaled up: at 0.33 MP the crowd demo's cameras boxed ~5% high, at 1.3 on target.
        viewed, size = await self.viewable(image_url, budget, floor=budget // 2 if domain == "object" else 0)
        label = ", ".join(prompts)
        question = grounding.prompt(prompts, domain)

        reply, looped = await self._ground_reply(viewed, question, grounding.max_tokens(domain))
        elements = grounding.parse(reply, label, looped=looped)
        if domain == "ui" and size:
            # The original, not the shrunk copy: quarters are cut from it and margins trimmed against it.
            try:
                raw = await self._image_bytes(image_url)
            except InputFetchError as exc:
                # Read once already; failing a second time must not cost the detection.
                log.warning("could not read the screenshot again (%s); boxes as grounded", exc)
                raw = None
            if raw is not None and looped:
                # Crowded enough to loop: grounded again in quarters, which find what the loop crowded out.
                log.info("grounding %r looped on %dx%d; grounding its quarters", label, *size)
                crops = await asyncio.to_thread(_tile_uris, raw, budget)
                replies = await asyncio.gather(
                    *(self._ground_reply(uri, question, grounding.max_tokens(domain)) for uri, _ in crops)
                )
                elements = grounding.suppress_duplicates(
                    [
                        e
                        for (_, tile), (text, tile_looped) in zip(crops, replies, strict=True)
                        for e in grounding.from_tile(
                            grounding.parse(text, label, looped=tile_looped), tile, *size
                        )
                    ]
                )
            if raw is not None:
                elements = await asyncio.to_thread(_trimmed, elements, raw, budget)
        # Size travels with the boxes: via /segment, a busy moment returned grid coordinates as pixels.
        return {"gui_elements": elements, **({"width": size[0], "height": size[1]} if size else {})}

    async def _ground_reply(self, image_url: str, question: str, max_tokens: int) -> tuple[str, bool]:
        """The grounding reply, and whether it looped; streamed so the watch stops a loop before the cap."""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": question},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
            # Grounding emits a JSON box list; chain of thought only delays it.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        watch, finish = grounding.LoopWatch(), None
        async with contextlib.aclosing(self._events(payload)) as events:
            async for chunk in events:
                for choice in chunk.get("choices") or []:
                    finish = choice.get("finish_reason") or finish
                    if watch.feed((choice.get("delta") or {}).get("content") or ""):
                        return watch.raw, True
        return watch.raw, finish == "length"

    _IMAGE_SAFETY_SCHEMA = {
        "name": "image_safety",
        "schema": {
            "type": "object",
            "properties": {
                "adult_score": {
                    "type": "number",
                    "description": (
                        "0 to 1: how likely the image is sexually explicit -- nudity (exposed "
                        "genitals, female nipples, buttocks) or sexual activity."
                    ),
                },
                "racy_score": {
                    "type": "number",
                    "description": (
                        "0 to 1: how likely the image is sexually suggestive without being "
                        "explicit -- lingerie, revealing clothing in a sexualised context, "
                        "provocative poses, partial nudity."
                    ),
                },
                "gore_score": {
                    "type": "number",
                    "description": (
                        "0 to 1: how likely the image is gory -- blood, open wounds, graphic "
                        "injury, mutilation, dead bodies."
                    ),
                },
            },
            "required": ["adult_score", "racy_score", "gore_score"],
            "additionalProperties": False,
        },
    }

    async def image_safety(self, image_url: str) -> dict[str, float]:
        """Adult, racy and gore scores for one image, from the brain; interfaze uses a hosted classifier."""
        image_url, size = await self.viewable(image_url, self.settings.answer_image_max_pixels, inline=True)
        if size is None:
            raise InputFetchError("could not read the image for the safety check")
        reply = await self.structured(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {
                            "type": "text",
                            "text": (
                                "Classify this image for content moderation. Score each category "
                                "independently from 0 (certainly not) to 1 (certainly)."
                            ),
                        },
                    ],
                }
            ],
            self._IMAGE_SAFETY_SCHEMA,
            Sampling(max_tokens=200, temperature=0),
        )
        scores = json.loads(reply.content)

        def clamp(value) -> float:
            try:
                return min(1.0, max(0.0, float(value)))
            except (TypeError, ValueError):
                return 0.0

        return {
            "adult": clamp(scores.get("adult_score")),
            "racy": clamp(scores.get("racy_score")),
            "gore": clamp(scores.get("gore_score")),
            "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens,
        }

    async def healthy(self) -> bool:
        try:
            resp = await self.http.get(f"{self.settings.brain_url.rstrip('/')}/health", timeout=5)
            return resp.status_code < 400
        except httpx.HTTPError:
            return False


def _shrink_to_budget(
    raw: bytes, budget: int, original: str, inline: bool = False, floor: int = 0
) -> tuple[str, tuple[int, int]]:
    from PIL import Image

    image = Image.open(io.BytesIO(raw))
    pixels = image.width * image.height
    size = (image.width, image.height)
    if 0 < pixels < floor:
        scale = (budget / pixels) ** 0.5
        image = image.convert("RGB").resize(
            (int(image.width * scale), int(image.height * scale)), Image.LANCZOS
        )
        log.info("grounding: %dx%d scaled up to %dx%d", *size, image.width, image.height)
        return _png_uri(image), size
    if pixels <= budget:
        log.info("grounding: %dx%d within budget", image.width, image.height)
        if inline and not original.startswith("data:"):
            image.load()  # decodes, so a file vLLM cannot read is caught here
            mime = Image.MIME.get(image.format or "", "image/png")
            return f"data:{mime};base64," + base64.b64encode(raw).decode(), size
        return original, size

    image = _within(image.convert("RGB"), budget)
    log.info("grounding: %d px -> %dx%d", pixels, image.width, image.height)
    return _png_uri(image), size


def _tile_uris(raw: bytes, budget: int) -> list[tuple[str, tuple[int, int, int, int]]]:
    """The image's quarters (grounding.tiles), each a data URI within `budget` pixels."""
    from PIL import Image

    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return [
        (_png_uri(_within(image.crop(tile), budget)), tile)
        for tile in grounding.tiles(image.width, image.height)
    ]


def _trimmed(elements: list[dict], raw: bytes, budget: int) -> list[dict]:
    from PIL import Image

    try:
        return grounding.trim_to_content(
            elements, _within(Image.open(io.BytesIO(raw)).convert("RGB"), budget)
        )
    except Exception as exc:
        # Trimming is a refinement; failing at it must not cost the boxes themselves.
        log.warning("could not trim the boxes (%s); returning them as grounded", exc)
        return elements


def _within(image, budget: int):
    pixels = image.width * image.height
    if pixels <= budget:
        return image
    scale = (budget / pixels) ** 0.5
    return image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))


def _png_uri(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _inline_local_file(path: str) -> str:
    """A local path as a data URI, for a consumer that cannot reach this filesystem."""
    blob = pathlib.Path(path).read_bytes()
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(blob).decode()
