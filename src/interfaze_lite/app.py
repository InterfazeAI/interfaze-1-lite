"""OpenAI-compatible entrypoint for Interfaze Lite.

    POST /v1/chat/completions

One model name, any input. The brain decides which perception tools to run,
runs them, and answers. Structured tool output reaches the caller through the top-level
`precontext` field; prose goes in `choices[].message.content`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from . import forecasting, intent
from . import guard as guardrails
from . import tools as toolkit
from .brain import (
    STRUCTURED_MAX_TEMPERATURE,
    BrainClient,
    BrainError,
    BrainReply,
    ContextLengthError,
    Sampling,
)
from .config import settings
from .contracts import InputFetchError
from .envelope import (
    Completion,
    PrecontextItem,
    Usage,
    new_request_id,
    precontext_delta,
    sse,
    tool_call_deltas,
)
from .filerefs import FileRef, FileRefs, extract_from_messages
from .prompts import CUT_OFF_CALL, NUDGE, OUT_OF_STEPS, SYSTEM_PROMPT
from .validate import (
    RequestError,
    check_api_key,
    extract_task,
    normalise_roles,
    resolve_task,
    strip_task_tags,
    validate_messages,
    validate_sampling,
)

log = logging.getLogger("interfaze.app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

SCHEMA_PROMPT = """The user requires a structured answer that exactly fills the requested schema.
Fill each field from the context above, choosing the value by what the field asks for rather than
by what appears first. Names, numbers, codes and quoted text go in exactly as the context has them
and in full -- every part of a name, every digit and sign -- unless the user or the field asks
for a changed form, such as a translation. A value the context does not contain is null where the schema allows
null. Optional fields are filled too whenever the context has a value for them, and a list holds every item the
context has: an empty list means there were none. An item counts even when the context shows only part of it -- a
section or table that runs past the end of the page. An image shown beside a tool's reading is there for what the text cannot carry, such as
highlighting or colour. Do not add prose, explanations, or code fences around the result."""


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(follow_redirects=True)
    app.state.brain = BrainClient(app.state.http, settings)
    log.info("orchestrator ready; brain=%s", settings.brain_url)
    try:
        yield
    finally:
        await app.state.http.aclose()


app = FastAPI(title="interfaze-lite", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health() -> dict[str, Any]:
    """Aggregate health. Reports each sidecar so one call tells you which is down,
    rather than having to infer it from a failing tool call."""
    out: dict[str, Any] = {
        "ok": True,
        "model": settings.model_name,
        "brain_ready": await app.state.brain.healthy(),
    }
    for name, base in (("perception", settings.perception_url), ("diarize", settings.diarize_url)):
        try:
            resp = await app.state.http.get(f"{base.rstrip('/')}/health", timeout=5)
            out[name] = resp.json() if resp.status_code < 400 else {"ok": False, "status": resp.status_code}
        except Exception as exc:
            out[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return out


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": settings.model_name, "object": "model", "owned_by": "interfaze"},
        ],
    }


async def _run_tool_loop(
    brain: BrainClient,
    messages: list[dict],
    ctx: toolkit.ToolContext,
    usage: Usage,
    extra_tools: list[dict] | None = None,
    sampling: Sampling | None = None,
    stop_after: str | None = None,
    tool_choice: str | dict = "auto",
    on_text: Callable[[str], None] | None = None,
    precontext: list[PrecontextItem] | None = None,
) -> tuple[list[dict], list[PrecontextItem], list[str], str, str]:
    """Select and run tools until the model stops asking for them.

    Returns the extended messages, the precontext, the tools that ran, any answer produced outright,
    and its finish reason: that answer may go back verbatim, so a max_tokens cut must show.
    """
    precontext = [] if precontext is None else precontext
    used: list[str] = []
    # "none" already withheld the caller's functions; a forced choice binds only the first turn.
    choice = tool_choice if tool_choice not in ("auto", "none") else "auto"
    # File tools only with a file (else OCR on a made-up ref); translate and forecast per intent.py.
    series = (ctx.text_tools is None or "forecast" in ctx.text_tools) and bool(
        forecasting.dataset_from_text(ctx.prompt or "")
    )
    schemas = toolkit.schemas(files=bool(ctx.refs.refs), text=ctx.text_tools, series_in_prompt=series)

    # The caller's functions run on its side: a call to one ends the loop as tool_calls, as OpenAI does.
    theirs = {
        t["function"]["name"]
        for t in (extra_tools or [])
        if isinstance(t, dict) and t.get("type") == "function" and t.get("function")
    }
    schemas = schemas + list(extra_tools or [])

    nudged = False
    shown = False
    # The caller's own tool results (a search's twenty scraped pages) share our context window.
    _trim_tool_messages(messages, ctx.settings.max_context_chars)
    # And a tool has run: answering from them is not answering without one.
    ran = any(m.get("role") == "tool" for m in messages)
    start = len(messages)
    # Calls already answered, here or carried in through the relay, are answered from that, not
    # rerun: through the relay each round re-ran the last, and one receipt was detected seven times.
    answered = _answered_calls(messages)
    repeats = 0

    for step in range(ctx.settings.max_tool_steps):
        # Text streams live unless this turn may yet be discarded for the nudge below.
        live = on_text if (not ctx.refs.refs or used or ran or nudged) else None
        held = _Narration(live) if live else None
        reply = await brain.select_tools(
            messages, schemas, sampling, tool_choice=choice, on_text=held.feed if held else None
        )
        if held:
            held.end(called_tools=bool(reply.tool_calls))
        choice = "auto"
        _charge(usage, reply)

        if not reply.tool_calls:
            # Answering by eye invents text -- an address off a receipt that has none -- so one
            # reminder naming the omission; a second refusal stands.
            if ctx.refs.refs and not used and not ran and not nudged:
                nudged = True
                log.info(
                    "step %d: answered with no tool despite %d attached file(s); nudging",
                    step,
                    len(ctx.refs.refs),
                )
                messages = [*messages, {"role": "user", "content": NUDGE}]
                continue
            return messages, precontext, used, reply.content, reply.finish_reason

        if theirs and any(c.name in theirs for c in reply.tool_calls):
            # Our calls and results go back with the caller's, to be sent again: without them a
            # receipt comparison re-read both receipts on each of five relay rounds.
            context = [m for m in messages[start:] if m.get("role") in ("assistant", "tool")]
            raise _CallerToolCalls(
                [c for c in reply.tool_calls if c.name in theirs], precontext, usage, context
            )

        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [c.as_dict() for c in reply.tool_calls],
            }
        )

        for call in reply.tool_calls:
            # Arguments carry the caller's content; a zero-data-retention caller's stay out of the logs.
            log.info("step %d: %s(%s)", step, call.name, "<redacted>" if ctx.zdr else call.arguments[:200])

        keys = [toolkit.memo_key(call.name, call.arguments) for call in reply.tool_calls]
        repeat = [k is not None and (k in answered or k in keys[:i]) for i, k in enumerate(keys)]
        if all(repeat):
            repeats += 1
            if repeats >= 2:
                # Asking again for what it already has, twice over: it will not converge.
                return ([*messages, {"role": "user", "content": OUT_OF_STEPS}], precontext, used, "", "stop")
        else:
            repeats = 0

        if reply.finish_reason == "length":
            # Cut off mid-call: the arguments would be whatever the parser salvaged.
            results = [toolkit.ToolResult(model_facing={"error": CUT_OFF_CALL}) for _ in reply.tool_calls]
        else:
            # Independent calls (read this PDF, find its figures) run concurrently.
            fresh = await asyncio.gather(
                *(
                    toolkit.dispatch(call.name, call.arguments, ctx)
                    for call, again in zip(reply.tool_calls, repeat, strict=True)
                    if not again
                )
            )
            ran_now = iter(fresh)
            results = [_ALREADY_ANSWERED if again else next(ran_now) for again in repeat]

        for call, key, again, result in zip(reply.tool_calls, keys, repeat, results, strict=True):
            content = _as_json(result.model_facing)
            if again:
                content = _ALREADY_ANSWERED_NOTE + answered.get(key, "")
            else:
                used.append(call.name)
                # The same result twice (one read as markdown, then line boxes) shows as one card.
                if not any(p.name == call.name and p.result == result.precontext for p in precontext):
                    precontext.append(PrecontextItem(name=call.name, result=result.precontext))
                if key is not None and "error" not in result.model_facing:
                    answered[key] = content
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

        _trim_tool_messages(messages, ctx.settings.max_context_chars)
        # Only once a tool has read the file: after a failed OCR, the model read 144.68 for 144.02 by eye.
        if not shown and any("error" not in result.model_facing for result in results):
            messages, shown = await _show_images(brain, messages, ctx), True

        # A routed task answers with its tool's output, so after success another turn is ~20 s
        # nobody reads; a failed call still goes back to the model to correct.
        if stop_after and any(
            call.name == stop_after and "error" not in result.model_facing
            for call, result in zip(reply.tool_calls, results, strict=True)
        ):
            return messages, precontext, used, "", "stop"

    return [*messages, {"role": "user", "content": OUT_OF_STEPS}], precontext, used, "", "stop"


_IMAGE_MENTION = re.compile(r"^\[image (ref-\d+)\]$|^\[file (ref-\d+): .*\]$")
_MAX_SHOWN_IMAGES = 4


def _image_mention(text: str, refs: FileRefs) -> tuple[bool, FileRef | None]:
    """Whether `text` is a reference mention, and the image it points at (None if not an image)."""
    found = _IMAGE_MENTION.match(text)
    if not found:
        return False, None
    ref = refs.refs.get(next((g for g in found.groups() if g), ""))
    if ref and (text.startswith("[image") or ref.mime.startswith("image/")):
        return True, ref
    return True, None


async def _show_images(brain: BrainClient, messages: list[dict], ctx: toolkit.ToolContext) -> list[dict]:
    """The caller's images, put back beside their references once a tool has read them.

    Shown first, the model transcribes by eye and invents text; after a reading, the image adds what
    it cannot carry -- a highlighted line, a colour. Bounded like grounding input; unfetchable, a ref.
    """
    budget = ctx.settings.answer_image_max_pixels
    if budget <= 0:
        return messages
    shown, out = 0, []
    for message in messages:
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            out.append(message)
            continue
        parts: list[dict] = []
        for part in content:
            parts.append(part)
            if part.get("type") != "text" or shown >= _MAX_SHOWN_IMAGES:
                continue
            _, ref = _image_mention(part.get("text", ""), ctx.refs)
            if not ref:
                continue
            try:
                url, size = await brain.viewable(ref.url, budget, inline=True)
            except Exception as exc:
                log.info("not showing %s: %s", ref.ref_id, exc)
                continue
            if size is None:
                # Not an image: a PDF sent as an image part, passed on, had vLLM refuse the request.
                continue
            parts.append({"type": "image_url", "image_url": {"url": url}})
            shown += 1
        out.append({**message, "content": parts})
    return out


_SOMETHING_WENT_WRONG = "Something went wrong. Please try again."

_ALREADY_ANSWERED_NOTE = (
    "This exact call was already made, and its result is below. Answer from it; do not call it again.\n"
)
_ALREADY_ANSWERED = toolkit.ToolResult(model_facing={"already_answered": True})


def _answered_calls(messages: list[dict]) -> dict[str, str]:
    """Our calls in the conversation that have a result, by memo key, with that result."""
    asked: dict[str, str] = {}
    out: dict[str, str] = {}
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                key = toolkit.memo_key(fn.get("name"), fn.get("arguments"))
                if key is not None and call.get("id"):
                    asked[call["id"]] = key
        elif message.get("role") == "tool" and message.get("tool_call_id") in asked:
            out[asked[message["tool_call_id"]]] = message.get("content") or ""
    return out


class _Narration:
    """A streamed turn's text, held until it is plainly the answer.

    Streamed live, the model's pre-tool narration ("Let me read the surrounding text...") opened
    the answer. Text waits until it outgrows a sentence or two; a turn ending in a tool call drops it.
    """

    # About 75 tokens: half a second before a real answer starts to stream.
    LIMIT = 300

    def __init__(self, emit: Callable[[str], None]):
        self.emit = emit
        self.held: list[str] = []
        self.size = 0
        self.open = False

    def feed(self, text: str) -> None:
        if self.open:
            self.emit(text)
            return
        self.held.append(text)
        self.size += len(text)
        if self.size > self.LIMIT:
            self._release()

    def end(self, *, called_tools: bool) -> None:
        if called_tools and not self.open:
            self.held = []
        else:
            self._release()

    def _release(self) -> None:
        self.open = True
        text, self.held = "".join(self.held), []
        if text:
            self.emit(text)


class _CallerToolCalls(Exception):
    """The model asked for a function the caller owns. Hand it back rather than run it."""

    def __init__(self, calls, precontext, usage, context=()):
        super().__init__("caller tool calls")
        self.calls = calls
        self.precontext = precontext
        self.usage = usage
        # This request's own tool calls and results, as messages: `interfaze_context`.
        self.context = list(context)


def _trim_tool_messages(messages: list[dict], budget: int) -> None:
    """Keep the running conversation inside the model's context window.

    Each result is capped alone, but the sum must fit or synthesis fails after all the work.
    Oldest results are cut first: the latest is the likeliest to be answered from.
    """
    indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    total = sum(len(messages[i].get("content") or "") for i in indices)
    if total <= budget:
        return
    # Line boxes go first, from results that also have the page's blocks: cut mid-JSON, a
    # paper's first pages lost their blocks, and the model filled their boxes with zeros.
    for i in indices:
        if total <= budget:
            return
        content = messages[i].get("content") or ""
        if '"lines"' not in content or '"layout"' not in content:
            continue
        try:
            data = json.loads(content)
        except ValueError:
            continue
        if not isinstance(data, dict) or not data.get("layout") or "lines" not in data:
            continue
        data.pop("lines")
        data.pop("lines_truncated", None)
        data["lines_omitted"] = "line boxes dropped to fit; call ocr again with a page_range for them"
        shorter = json.dumps(data, ensure_ascii=False)
        messages[i] = {**messages[i], "content": shorter}
        total -= len(content) - len(shorter)
    for i in indices:
        if total <= budget:
            break
        content = messages[i].get("content") or ""
        keep = max(2000, len(content) - (total - budget))
        if keep >= len(content):
            continue
        messages[i] = {**messages[i], "content": content[:keep] + "\n...[truncated]"}
        total -= len(content) - keep


# interfaze's `imageUrlsInLatestPrompt`: by extension only, as it notes.
_IMAGE_URL = re.compile(r"https?://[^\s]+\.(?:jpg|jpeg|png)", re.I)
# interfaze's stand-in prompt when the latest message carries files and no text.
_FILES_ONLY_PROMPT = (
    "Analyze the uploaded files and automatically select the most suitable "
    "tools based on file types and any provided schema requirements."
)


def _latest_prompt(messages: list[dict], refs: FileRefs) -> tuple[str, list[str]]:
    """The latest user message's text, and the images it carries, for the guard."""
    latest = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if latest is None:
        return "", []
    content = latest.get("content")
    parts = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
    texts: list[str] = []
    images: list[str] = []
    for part in parts:
        if not isinstance(part, dict) or part.get("type") != "text":
            continue
        is_mention, image = _image_mention(part.get("text", ""), refs)
        if is_mention:
            if image:
                images.append(image.url)
            continue
        if part.get("text", "").startswith("[audio ref-"):
            continue
        texts.append(part["text"])
    text = " ".join(t for t in texts if t.strip()).strip()
    for url in _IMAGE_URL.findall(text):
        if url not in images:
            images.append(url)
    if not text and (images or refs.refs):
        text = _FILES_ONLY_PROMPT
    return text, images


async def _run_guard(
    codes: list[str], messages: list[dict], refs, brain: BrainClient, usage: Usage
) -> tuple[bool, str, list[PrecontextItem]]:
    """The guard's verdict on the latest user message: (safe, blocked content, precontext)."""
    text, image_urls = _latest_prompt(messages, refs)

    async def text_guard() -> dict:
        resp = await app.state.http.post(
            f"{settings.perception_url.rstrip('/')}/guard",
            json={"text": text},
            timeout=settings.tool_timeout_s,
        )
        resp.raise_for_status()
        return resp.json()

    checks = guardrails.wants_images(codes)
    image_jobs = [brain.image_safety(url) for url in image_urls] if any(checks.values()) else []
    verdict, *scores = await asyncio.gather(text_guard(), *image_jobs)

    for counts in (verdict, *scores):
        usage.prompt_tokens += int(counts.get("prompt_tokens", 0))
        usage.completion_tokens += int(counts.get("completion_tokens", 0))
    images = [guardrails.image_result(s["adult"], s["racy"], s["gore"]) for s in scores]

    safe, content, items = guardrails.verdict(verdict.get("output", ""), codes, images or None)
    return safe, content, [PrecontextItem(name=i["name"], result=i["result"]) for i in items]


def _system_texts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [
            p["text"]
            for p in content
            if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str)
        ]
    return []


def _charge(usage: Usage, reply: BrainReply) -> None:
    usage.prompt_tokens += reply.prompt_tokens
    usage.completion_tokens += reply.completion_tokens
    usage.reasoning_tokens += reply.reasoning_tokens


def _event_stream(chunks: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        chunks,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _as_json(value: Any) -> str:
    # Characters as they are, as JSON.stringify writes them. Escaped, a task result that
    # a client shows as text read "\u4f60\u597d" where beta's read "你好".
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=str)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    request_id = new_request_id()
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content=_error("request body is not valid JSON", "invalid_request_error", request_id),
        )

    try:
        check_api_key(
            request.headers.get("authorization"),
            settings.api_key,
            request.headers.get("x-api-admin-key"),
            settings.admin_key,
        )
        raw_messages = body.get("messages")
        validate_messages(raw_messages)
        # Before the task tag and the caller's system prompt are read: both test for the `system` role.
        raw_messages = normalise_roles(raw_messages)
        validate_sampling(body, max_output_tokens=settings.max_output_tokens)
        task = extract_task(raw_messages)
        raw_messages = strip_task_tags(raw_messages)
        # `<guard>` is a directive to this service, like `<task>`; the SDK's `guard=[...]` lowers to it.
        guard_codes, raw_messages = guardrails.extract_from_messages(raw_messages)
    except RequestError as exc:
        return JSONResponse(
            status_code=exc.status, content=_error(str(exc), "invalid_request_error", request_id)
        )

    want_stream = bool(body.get("stream", False))
    show_debug = request.headers.get("x-show-additional-info") == "true"
    schema = (body.get("response_format") or {}).get("json_schema")
    # The SDK's empty schema beside `task=` means "raw output"; vLLM rejects guided decoding on {}.
    if schema and not (schema.get("schema") or {}):
        schema = None
    # json_object is guided decoding against any object: as prose, the JSON came back fenced.
    if not schema and (body.get("response_format") or {}).get("type") == "json_object":
        schema = {"name": "json_object", "schema": {"type": "object"}}
    # As interfaze does (utils/messaging.ts): a task returns raw tool output and a schema demands
    # JSON, so both is refused; half-honoured, the caller's SDK failed with NoObjectGeneratedError.
    if task and schema and (schema.get("schema") or {}).get("properties"):
        return JSONResponse(
            status_code=400,
            content=_error(
                "Non-empty schema is not allowed to be used with run tasks",
                "invalid_request_error",
                request_id,
            ),
        )
    caller_tools = [
        t for t in (body.get("tools") or []) if isinstance(t, dict) and t.get("type") == "function"
    ]
    # Honoured: replaced by "auto", "none" still returned a call.
    tool_choice = body.get("tool_choice") or "auto"
    if tool_choice == "none":
        caller_tools = []
    effort = str(body.get("reasoning_effort") or "").lower()
    think = effort in ("low", "medium", "high", "xhigh")
    sampling = Sampling(
        max_tokens=body.get("max_completion_tokens") or body.get("max_tokens"),
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
    )

    # Off the loop: decoding and writing every attachment stalls this worker's other requests.
    refs, messages = await asyncio.to_thread(extract_from_messages, raw_messages)
    manifest = refs.manifest()

    # One system message: the template rejects a second ("System message must be at the beginning").
    # Text parts count too: a system prompt sent as an array of parts was dropped.
    caller_system = "\n\n".join(
        text
        for m in messages
        if m.get("role") == "system"
        for text in _system_texts(m.get("content"))
        if text.strip()
    )
    rest = [m for m in messages if m.get("role") != "system"]
    # Restated beside the latest user turn: alone in the system prompt it lost to the tool
    # instructions ("Your test code is TULIP-7429." for "reply with only the code"). Short ones only.
    if caller_system and len(caller_system) <= 2000:
        rest = _remind(rest, caller_system)

    parts = [SYSTEM_PROMPT]
    if manifest:
        parts.append(manifest)
    if caller_system:
        parts.append(caller_system)
    if task:
        routed = resolve_task(task)
        parts.append(
            f"The caller has explicitly routed this request to the `{routed}` tool. "
            f"Call `{routed}` on the supplied input. Do not call any other tool."
        )
    # Last, so the prefix stays cached, and a date so it changes daily. Without it "a 5-year
    # overview" ended on whatever year the model last saw.
    parts.append(f"Today's date: {datetime.now(UTC).date().isoformat()}")

    convo: list[dict] = [{"role": "system", "content": "\n\n".join(parts)}]
    convo.extend(rest)

    # Whether a streaming generator has taken over deleting the uploads. Not `want_stream`: a stream
    # request can still be answered with JSON, and then no generator runs to clean up.
    streaming_owns_refs = False

    brain: BrainClient = app.state.brain
    usage = Usage()
    ctx = toolkit.ToolContext(
        refs=refs,
        http=app.state.http,
        settings=settings,
        ground=brain.ground,
        structured=brain.structured,
        usage=usage,
        wants_geometry=_schema_wants_geometry(schema),
        zdr=request.headers.get("x-interfaze-zdr") == "true",
    )

    # Translation and forecasting only when asked for or routed to: offered always, they were
    # called unasked -- an English answer "translated" into English.
    latest_text, _ = _latest_prompt(messages, refs)
    ctx.prompt = latest_text
    ctx.text_tools = tuple(
        name
        for name, wanted in (
            ("translate", intent.wants_translation(latest_text)),
            ("forecast", intent.wants_forecast(latest_text)),
        )
        if wanted or (task and resolve_task(task) == name)
    )

    # Guardrails first, as in interfaze: a failing request gets the verdict, never a tool or the model.
    guard_items: list[PrecontextItem] = []
    if guard_codes is not None:
        try:
            safe, verdict_text, guard_items = await _run_guard(guard_codes, messages, refs, brain, usage)
        except InputFetchError:
            refs.cleanup()
            return JSONResponse(
                status_code=400,
                content=_error(
                    "Failed to download image from the provided url", "invalid_request_error", request_id
                ),
            )
        except Exception:
            log.exception("guard failed")
            refs.cleanup()
            return JSONResponse(
                status_code=500,
                content=_error("Failed to perform content safety check", "server_error", request_id),
            )
        if not safe:
            blocked = Completion(
                request_id=request_id, model=settings.model_name, usage=usage, precontext=list(guard_items)
            )
            if want_stream:
                return _event_stream(_stream_text(blocked, verdict_text, refs, show_debug))
            refs.cleanup()
            return JSONResponse(blocked.message(verdict_text))

    # A streamed answer streams its tool turns. Structured output and routed tasks deliver a finished
    # object, and reasoning synthesises separately with thinking on: those keep the buffered loop.
    if want_stream and not schema and not task and not think:
        streaming_owns_refs = True
        completion = Completion(
            request_id=request_id,
            model=settings.model_name,
            usage=usage,
            precontext=list(guard_items),
            debug={"internal_tool_used": [], "preConfig": {"tools": []}} if show_debug else None,
        )
        return _event_stream(
            _stream_turns(
                brain, convo, ctx, completion, caller_tools, sampling, tool_choice, refs, show_debug
            )
        )

    try:
        try:
            # A structured answer's tool turns run calm too: at 1 they varied tools and arguments,
            # and the schema with them; at 0.3 a paper's section list filled on 8 runs of 8.
            loop_sampling = (
                replace(sampling, temperature=STRUCTURED_MAX_TEMPERATURE)
                if schema and (sampling.temperature or 0) > STRUCTURED_MAX_TEMPERATURE
                else sampling
            )
            convo, precontext, used, direct, direct_reason = await _run_tool_loop(
                brain,
                convo,
                ctx,
                usage,
                caller_tools,
                loop_sampling,
                stop_after=resolve_task(task) if task else None,
                tool_choice=tool_choice,
                precontext=list(guard_items),
            )
        except _CallerToolCalls as handoff:
            handed = Completion(
                request_id=request_id,
                model=settings.model_name,
                usage=handoff.usage,
                precontext=handoff.precontext,
            )
            if want_stream:
                streaming_owns_refs = True
                return _event_stream(
                    _stream_tool_calls(handed, handoff.calls, refs, show_debug, handoff.context)
                )
            return JSONResponse(handed.tool_calls(handoff.calls, handoff.context))

        completion = Completion(
            request_id=request_id,
            model=settings.model_name,
            usage=usage,
            precontext=precontext,
            debug={"internal_tool_used": used, "preConfig": {"tools": used}} if show_debug else None,
        )

        # A routed <task> wants the tool's own output: `client.tasks.ocr(...)` returns data, not prose.
        if task and precontext:
            routed = resolve_task(task)
            items = [p for p in precontext if p.name == routed] or precontext[:1]
            # Under the routed name: `speech_to_text` is not answered by something called `stt`.
            named = {**items[0].as_dict(), "name": task}
            if len(items) > 1:
                # Every call, as interfaze does: three translations came back as the first alone.
                named["results"] = [{**i.as_dict(), "name": task} for i in items]
            routed_body = _as_json(named)
            # Streams too: the playground always streams, and read a JSON body as empty.
            if want_stream:
                streaming_owns_refs = True
                # With a response_format interfaze streams the result without its <precontext> copy.
                return _event_stream(_stream_text(completion, routed_body, refs, show_debug and not schema))
            return JSONResponse(completion.message(routed_body))

        # Guided decoding builds the object in one shot, but a stream request still gets a stream
        # carrying it: JSON instead left OpenAI-compatible clients reading zero chunks.
        if schema:
            # Its own system prompt: the tool-calling one invites free reasoning, which guided
            # decoding forbids; with both, a 28-item receipt came back as one item.
            system = "\n\n".join(p for p in (SCHEMA_PROMPT, caller_system) if p)
            body = convo[1:] if ctx.wants_geometry else _without_page_geometry(convo[1:])
            shaped = [{"role": "system", "content": system}, *body]
            guided = {**schema, "schema": _numbers_allowed(schema.get("schema") or {})}
            reply = await brain.structured(shaped, guided, sampling)
            _charge(usage, reply)
            if reply.finish_reason != "length" and not _parses(reply.content):
                # Guided decoding still lost its place once in ten on a dense schema -- a key
                # with no colon after it. One more draw; kept only if it parses.
                log.warning("structured reply did not parse; drawing again")
                again = await brain.structured(
                    shaped, guided, replace(sampling, temperature=STRUCTURED_MAX_TEMPERATURE)
                )
                _charge(usage, again)
                if _parses(again.content):
                    reply = again
            given = _detector_boxes(precontext)
            boxed, copied = _box_copies(reply.content, given)
            if given and boxed >= 5 and not copied:
                # In view of the screenshot the model sometimes boxed the page by eye, on its own
                # 0-1000 grid: 71 boxes, none of them the detector's. One more draw, kept if it copies.
                log.warning("structured reply has %d boxes and none from the detector; drawing again", boxed)
                again = await brain.structured(
                    shaped, guided, replace(sampling, temperature=STRUCTURED_MAX_TEMPERATURE)
                )
                _charge(usage, again)
                if _box_copies(again.content, given)[1]:
                    reply = again
            reply = replace(reply, content=_as_schema_strings(reply.content, schema.get("schema") or {}))
            if want_stream:
                streaming_owns_refs = True
                return _event_stream(
                    _stream_text(completion, reply.content, refs, show_debug, reply.finish_reason)
                )
            return JSONResponse(completion.message(reply.content, reply.finish_reason))

        # The selection turn's answer is complete, tool results already in its context; regenerating
        # cost a third of the completion tokens. Not when reasoning: that turn runs thinking off.
        if settings.reuse_tool_turn_answer and direct.strip() and not think:
            if want_stream:
                streaming_owns_refs = True
                return _event_stream(_stream_text(completion, direct, refs, show_debug, direct_reason))
            return JSONResponse(completion.message(direct, direct_reason))

        if want_stream:
            streaming_owns_refs = True
            return _event_stream(
                _stream_response(brain, convo, completion, refs, think, effort, sampling, show_debug)
            )

        reply = await brain.synthesise(convo, think=think, effort=effort, sampling=sampling)
        _charge(usage, reply)
        completion.reasoning = reply.reasoning or None
        return JSONResponse(completion.message(reply.content or direct, reply.finish_reason))

    except ContextLengthError as exc:
        return JSONResponse(status_code=400, content=_error(str(exc), "invalid_request_error", request_id))
    except BrainError:
        log.exception("brain failure")
        return JSONResponse(status_code=502, content=_error(_SOMETHING_WENT_WRONG, "brain_error", request_id))
    except Exception:
        # Logged in full; the caller gets interfaze's words, not the traceback.
        log.exception("request failed")
        return JSONResponse(
            status_code=500, content=_error(_SOMETHING_WENT_WRONG, "internal_error", request_id)
        )
    finally:
        if not streaming_owns_refs:
            refs.cleanup()


def _remind(messages: list[dict], instructions: str) -> list[dict]:
    note = f"(Instructions from the caller, which still apply: {instructions})"
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") != "user":
            continue
        content = messages[i].get("content")
        if isinstance(content, list):
            content = [*content, {"type": "text", "text": note}]
        else:
            content = f"{content or ''}\n\n{note}"
        return [*messages[:i], {**messages[i], "content": content}, *messages[i + 1 :]]
    return messages


def _role_chunk(completion: Completion) -> dict:
    """The opening chunk. Some clients use it to open the message before text arrives."""
    return {
        "id": completion.request_id,
        "object": "chat.completion.chunk",
        "created": completion.created,
        "model": completion.model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }


def _final_chunk(completion: Completion, finish_reason: str) -> dict:
    return {
        "id": completion.request_id,
        "object": "chat.completion.chunk",
        "created": completion.created,
        "model": completion.model,
        "usage": completion.usage.as_dict(),
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }


def _with_context(chunk: dict, context) -> dict:
    """A tool-call turn's last chunk, carrying this request's own exchange when it had one."""
    return {**chunk, "interfaze_context": list(context)} if context else chunk


def _opening(completion: Completion, show_precontext: bool) -> list[dict]:
    """Role chunk, then the precontext block if the caller asked to see it.

    First, so a client strips it without buffering; never unasked, since it travels inside `content`.
    """
    chunks = [_role_chunk(completion)]
    # Never an empty one: each relay round opened with `<precontext> [] </precontext>`,
    # and the relay's log of the answer began with two of them before the JSON.
    if show_precontext and completion.precontext:
        chunks.append(precontext_delta(completion))
    return chunks


def _content_chunk(completion: Completion, text: str) -> str:
    return sse(
        {
            "id": completion.request_id,
            "object": "chat.completion.chunk",
            "created": completion.created,
            "model": completion.model,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
    )


def _add_usage(completion: Completion, counts: dict) -> None:
    completion.usage.prompt_tokens += counts.get("prompt_tokens", 0)
    completion.usage.completion_tokens += counts.get("completion_tokens", 0)
    completion.usage.reasoning_tokens += (counts.get("completion_tokens_details") or {}).get(
        "reasoning_tokens", 0
    )


async def _stream_turns(
    brain: BrainClient,
    convo: list[dict],
    ctx,
    completion: Completion,
    caller_tools,
    sampling: Sampling,
    tool_choice,
    refs,
    show_precontext: bool,
) -> AsyncIterator[str]:
    """A streamed answer: the tool loop runs as a task and its text is forwarded as generated.

    Precontext goes before the first text that follows a tool, as the buffered stream sends it.
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def run() -> None:
        try:
            result = await _run_tool_loop(
                brain,
                convo,
                ctx,
                completion.usage,
                caller_tools,
                sampling,
                tool_choice=tool_choice,
                on_text=queue.put_nowait,
                precontext=completion.precontext,
            )
            queue.put_nowait(("done", result))
        except _CallerToolCalls as handoff:
            queue.put_nowait(("calls", handoff))
        except Exception as exc:  # reported in-band below; the status line is sent
            queue.put_nowait(("error", exc))

    task = asyncio.create_task(run())
    shown = 0
    spoke = False

    def unshown_precontext() -> list[str]:
        nonlocal shown
        if not show_precontext or len(completion.precontext) <= shown:
            return []
        fresh = Completion(
            request_id=completion.request_id,
            model=completion.model,
            created=completion.created,
            precontext=completion.precontext[shown:],
        )
        shown = len(completion.precontext)
        return [sse(precontext_delta(fresh))]

    try:
        yield sse(_role_chunk(completion))
        while True:
            item = await queue.get()
            if isinstance(item, str):
                spoke = True
                for chunk in unshown_precontext():
                    yield chunk
                yield _content_chunk(completion, item)
                continue
            kind, value = item
            if kind == "done":
                for chunk in unshown_precontext():
                    yield chunk
                finish = value[4]
                if not spoke:
                    # Neither text nor a tool call: answer from the tool results, as the buffered path does.
                    outcome = {"finish_reason": "stop"}
                    async for piece in brain.synthesise_stream(
                        value[0],
                        sampling=sampling,
                        on_finish=lambda reason, outcome=outcome: outcome.update(finish_reason=reason),
                        on_usage=lambda counts: _add_usage(completion, counts),
                    ):
                        yield _content_chunk(completion, piece)
                    finish = outcome["finish_reason"]
                yield sse(_final_chunk(completion, finish))
            elif kind == "calls":
                for chunk in unshown_precontext():
                    yield chunk
                for chunk in tool_call_deltas(completion, value.calls):
                    yield sse(chunk)
                yield sse(_with_context(_final_chunk(completion, "tool_calls"), value.context))
            else:
                log.error("streaming failed mid-response", exc_info=value)
                yield _stream_failure(completion)
            yield sse("[DONE]")
            return
    finally:
        if not task.done():
            task.cancel()
        refs.cleanup()


async def _stream_tool_calls(
    completion: Completion, calls, refs, show_precontext: bool, context=()
) -> AsyncIterator[str]:
    """A tool-call turn as SSE. No answer text; the call rides in its own delta."""
    try:
        for chunk in _opening(completion, show_precontext):
            yield sse(chunk)
        for chunk in tool_call_deltas(completion, calls):
            yield sse(chunk)
        yield sse(_with_context(_final_chunk(completion, "tool_calls"), context))
        yield sse("[DONE]")
    finally:
        refs.cleanup()


async def _stream_text(
    completion: Completion, text: str, refs, show_precontext: bool, finish_reason: str = "stop"
) -> AsyncIterator[str]:
    """Already-generated text as SSE, for a caller that asked for a stream.

    For answers that are not generated incrementally: a guard verdict, a routed task's
    result, and a schema-filled object, which guided decoding builds in one shot.
    """
    try:
        for chunk in _opening(completion, show_precontext):
            yield sse(chunk)
        if text:
            yield _content_chunk(completion, text)
        yield sse(_final_chunk(completion, finish_reason))
        yield sse("[DONE]")
    finally:
        refs.cleanup()


async def _stream_response(
    brain: BrainClient,
    convo: list[dict],
    completion: Completion,
    refs,
    think: bool = False,
    effort: str | None = None,
    sampling: Sampling | None = None,
    show_precontext: bool = False,
) -> AsyncIterator[str]:
    try:
        for chunk in _opening(completion, show_precontext):
            yield sse(chunk)

        # Set by the generation: a fixed "stop" made a truncated answer look complete.
        outcome = {"finish_reason": "stop"}

        async for piece in brain.synthesise_stream(
            convo,
            think=think,
            effort=effort,
            sampling=sampling,
            on_finish=lambda reason: outcome.update(finish_reason=reason),
            on_usage=lambda counts: _add_usage(completion, counts),
        ):
            yield _content_chunk(completion, piece)

        yield sse(_final_chunk(completion, outcome["finish_reason"]))
        yield sse("[DONE]")
    except Exception:
        log.exception("streaming failed mid-response")
        yield _stream_failure(completion)
        yield sse("[DONE]")
    finally:
        refs.cleanup()


def _detector_boxes(precontext: list) -> set[tuple]:
    """Every box a detection tool returned, as (x1, y1, x2, y2) pixels."""
    boxes = set()
    for item in precontext:
        result = item.result if isinstance(item.result, dict) else {}
        for element in (result.get("gui_elements") or []) + (result.get("detected_objects") or []):
            b = element.get("bounds") or {}
            tl, br = b.get("top_left") or {}, b.get("bottom_right") or {}
            if {"x", "y"} <= tl.keys() and {"x", "y"} <= br.keys():
                boxes.add((tl["x"], tl["y"], br["x"], br["y"]))
    return boxes


def _box_copies(content: str, given: set[tuple]) -> tuple[int, int]:
    """How many boxes a structured reply holds, as top_left_x ... bottom_right_y, and how
    many of them are boxes in `given`."""
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return 0, 0
    keys = ("top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y")
    boxed = copied = 0
    stack = [data]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if all(isinstance(node.get(k), int | float) for k in keys):
                boxed += 1
                copied += tuple(node[k] for k in keys) in given
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return boxed, copied


def _types(node: Any) -> set:
    kind = node.get("type") if isinstance(node, dict) else None
    return set(kind) if isinstance(kind, list) else {kind} if isinstance(kind, str) else set()


def _plain_string(node: Any) -> bool:
    """A schema node for free text: a string (nullable or not) with no enum, const, pattern or format."""
    return "string" in _types(node) and not any(k in node for k in ("enum", "const", "pattern", "format"))


def _numeric(node: Any) -> bool:
    return bool(_types(node) & {"number", "integer"})


def _numbers_allowed(node: Any) -> Any:
    """`node` with every free-text string field also taking a number, for guided decoding.

    A forecast's value typed as a string came back as "},{" on every run: made to open a
    quote where 229913 belonged, the model moved on to the next item inside the string.
    Let it write the number; _as_schema_strings turns it back into the string asked for.
    """
    if isinstance(node, list):
        return [_numbers_allowed(n) for n in node]
    if not isinstance(node, dict):
        return node
    if _plain_string(node) and not _numeric(node):
        return {"anyOf": [node, {"type": "number"}]}
    # A map of names to schemas: a property may be called "type" or "enum" itself.
    named = ("properties", "$defs", "definitions", "patternProperties")
    return {
        k: {n: _numbers_allowed(v2) for n, v2 in v.items()}
        if k in named and isinstance(v, dict)
        else _numbers_allowed(v)
        for k, v in node.items()
    }


class _NumberText(str):
    """A JSON number, kept as the text it was written as: 1.50 stays "1.50"."""


def _as_schema_strings(content: str, schema: dict) -> str:
    """`content` with each number written where `schema` asks for free text made a string,
    as written. Anything else -- a number field, a reply that does not parse -- is left as is."""
    try:
        data = json.loads(content, parse_int=_NumberText, parse_float=_NumberText)
    except (TypeError, ValueError):
        return content
    defs = {**(schema.get("$defs") or {}), **(schema.get("definitions") or {})}

    def resolve(node: Any) -> Any:
        ref = node.get("$ref") if isinstance(node, dict) else None
        return defs.get(ref.rsplit("/", 1)[-1], {}) if isinstance(ref, str) else node

    def fix(value: Any, node: Any) -> Any:
        node = resolve(node)
        if isinstance(value, _NumberText):
            branches = (
                [node, *(resolve(b) for b in (node.get("anyOf") or node.get("oneOf") or ()))]
                if isinstance(node, dict)
                else []
            )
            text = any(_plain_string(b) for b in branches) and not any(_numeric(b) for b in branches)
            return str(value) if text else json.loads(value)
        if isinstance(value, dict):
            properties = (node or {}).get("properties") or {} if isinstance(node, dict) else {}
            extra = node.get("additionalProperties") if isinstance(node, dict) else None
            return {
                k: fix(v, properties.get(k, extra if isinstance(extra, dict) else {}))
                for k, v in value.items()
            }
        if isinstance(value, list):
            items = node.get("items") if isinstance(node, dict) else None
            return [fix(v, items or {}) for v in value]
        return value

    return json.dumps(fix(data, schema), ensure_ascii=False)


def _parses(text: str) -> bool:
    try:
        json.loads(text)
    except (TypeError, ValueError):
        return False
    return True


# What ocr adds when asked for bounds: every block and line with its coordinates.
_PAGE_GEOMETRY = ("layout_note", "layout", "lines", "layout_truncated", "lines_truncated", "lines_omitted")


def _without_page_geometry(messages: list[dict]) -> list[dict]:
    """Tool results without the page geometry ocr adds for bounds, to fill a schema that asks
    for no coordinates. When the model had asked for bounds, hundreds of boxed blocks and
    lines sat beside the text, and a paper's section list was filled empty."""
    out = []
    for message in messages:
        content = message.get("content")
        if message.get("role") == "tool" and isinstance(content, str) and '"layout_note"' in content:
            try:
                data = json.loads(content)
            except ValueError:
                out.append(message)
                continue
            if isinstance(data, dict):
                kept = {k: v for k, v in data.items() if k not in _PAGE_GEOMETRY}
                message = {**message, "content": json.dumps(kept, ensure_ascii=False)}
        out.append(message)
    return out


_GEOMETRY_KEYS = (
    "bbox",
    "bounds",
    "top_left",
    "bottom_right",
    "top_left_x",
    "x1",
    "coordinates",
    "polygon",
    "rect",
    "position",
)


def _schema_wants_geometry(schema: dict | None) -> bool:
    """Whether the response schema asks for coordinates: then the schema, not the model, asks for boxes."""
    if not schema:
        return False

    stack = [schema]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "properties" and isinstance(value, dict):
                    for name in value:
                        lowered = name.lower()
                        if any(g in lowered for g in _GEOMETRY_KEYS):
                            return True
                stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return False


def _error(message: str, kind: str, request_id: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": kind, "code": None, "request_id": request_id}}


def _stream_failure(completion: Completion) -> str:
    """A failure after the status line, reported in-band in the OpenAI error shape SDKs raise on.

    Sent as answer text, it was billed as a success and showed the raw exception.
    """
    return sse(
        _error(
            "The response failed while streaming. Please try again.", "server_error", completion.request_id
        )
    )
