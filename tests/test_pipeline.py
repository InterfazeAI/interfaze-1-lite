"""End-to-end orchestrator tests."""

from __future__ import annotations

import asyncio
import json
import types
from datetime import UTC

import httpx
import pytest
from fastapi.testclient import TestClient

from interfaze_lite import app as app_module
from interfaze_lite.brain import BrainClient
from interfaze_lite.config import settings

OCR_FIXTURE = {
    "text": "Invoice total 42.00",
    "context": None,
    "sections": [
        {
            "text": "Invoice total 42.00",
            "lines": [
                {
                    "text": "Invoice total 42.00",
                    "bounds": {
                        "top_left": {"x": 10, "y": 10},
                        "top_right": {"x": 300, "y": 10},
                        "bottom_right": {"x": 300, "y": 40},
                        "bottom_left": {"x": 10, "y": 40},
                        "width": 290,
                        "height": 30,
                    },
                    "average_confidence": 0.94,
                    "words": [],
                }
            ],
        }
    ],
    "has_text": True,
    "width": 800,
    "height": 600,
    "total_pages": 1,
}


def brain_body(content="", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["content"] = None
        message["tool_calls"] = tool_calls
    return {
        "choices": [
            {"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


def tool_call(name: str, arguments: dict | str, call_id: str = "c1") -> dict:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def user_image_message(text: str, url: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": url}},
        ],
    }


def post_image(client: TestClient, text: str, url: str = "https://x/y.png") -> httpx.Response:
    return client.post("/v1/chat/completions", json={"messages": [user_image_message(text, url)]})


def image_urls(brain_payload: dict) -> list[str]:
    """Every image URL in the messages of a request the brain received."""
    return [
        part["image_url"]["url"]
        for m in brain_payload["messages"]
        if isinstance(m.get("content"), list)
        for part in m["content"]
        if part.get("type") == "image_url"
    ]


def user_image_only(url: str) -> dict:
    return {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}


def user_audio_message(text: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "input_audio", "input_audio": {"data": "data:audio/wav;base64,AAAA", "format": "wav"}},
        ],
    }


def json_schema_format(name: str, schema: dict) -> dict:
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}


def stream_raw(client: TestClient, body: dict, headers: dict | None = None) -> str:
    with client.stream("POST", "/v1/chat/completions", json=body, headers=headers) as resp:
        return "".join(resp.iter_text())


def sse_chunks(raw: str) -> list[dict]:
    return [
        json.loads(line[6:])
        for line in raw.splitlines()
        if line.startswith("data: ") and line[6:].strip() != "[DONE]"
    ]


def streamed_text(raw: str) -> str:
    return "".join(chunk["choices"][0]["delta"].get("content") or "" for chunk in sse_chunks(raw))


def _png(width: int, height: int, element: tuple[int, int, int, int] | None = None) -> bytes:
    import io

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), "white")
    if element:
        ImageDraw.Draw(image).rectangle([element[0], element[1], element[2] - 1, element[3] - 1], fill="navy")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# A real image: grounding sizes and trims boxes against its pixels, so the element sits
# under the Fake grounder's [100, 100, 300, 400] grid box.
_PNG_800x600 = _png(800, 600, element=(80, 60, 240, 240))


class Fake:
    """Scripted brain + perception. Records every call for assertions."""

    def __init__(self, brain_turns, ocr=OCR_FIXTURE):
        self.brain_turns = list(brain_turns)
        self.ocr = ocr
        self.calls: list[str] = []
        self.tool_payloads: list[dict] = []
        self.stream_payloads: list[dict] = []
        self.segment_fails = False
        self.segment_payloads: list[dict] = []
        self.transcribe_payloads: list[dict] = []
        self.guard_output = "safe"
        self.guard_payloads: list[dict] = []
        self.structured_reply = '{"a":1}'

    def tool_messages(self) -> list[dict]:
        return [m for p in self.tool_payloads for m in p["messages"] if m.get("role") == "tool"]

    def last_structured_payload(self) -> dict:
        """The last brain request that carried a response_format (the guided-decoding call)."""
        return [p for p in self.tool_payloads if p.get("response_format")][-1]

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append(path)

        if path.endswith("/v1/chat/completions"):
            payload = json.loads(request.content)
            # Grounding: an image and no tool results. Answer turns carry the caller's
            # image too, once a tool has read it. It streams, as brain.ground asks.
            if any(
                isinstance(m.get("content"), list)
                and any(part.get("type") == "image_url" for part in m["content"])
                for m in payload.get("messages", [])
            ) and not any(m.get("role") == "tool" for m in payload.get("messages", [])):
                reply = '[{"bbox_2d": [100, 100, 300, 400], "label": "dog"}]'
                if not payload.get("stream"):
                    return httpx.Response(200, json=brain_body(reply))
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    text=(
                        "data: "
                        + json.dumps({"choices": [{"index": 0, "delta": {"content": reply}}]})
                        + "\n\ndata: "
                        + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                        + "\n\ndata: [DONE]\n\n"
                    ),
                )
            if payload.get("stream"):
                self.stream_payloads.append(payload)
                turn = self.brain_turns.pop(0)
                message = turn["choices"][0]["message"]
                text = message.get("content") or ""
                reason = message.get("reasoning_content") or ""
                deltas = [{"reasoning_content": reason[i : i + 5]} for i in range(0, len(reason), 5)]
                deltas += [{"content": text[i : i + 5]} for i in range(0, len(text), 5)]
                # Tool calls stream as vLLM sends them: their own delta, then a finish.
                calls = message.get("tool_calls") or []
                if calls:
                    deltas.append({"tool_calls": [{**c, "index": i} for i, c in enumerate(calls)]})
                chunks = (
                    "".join(
                        "data: " + json.dumps({"choices": [{"index": 0, "delta": d}]}) + "\n\n"
                        for d in deltas
                    )
                    + "data: "
                    + json.dumps(
                        {
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": "tool_calls" if calls else "stop"}
                            ]
                        }
                    )
                    + "\n\n"
                    + "data: [DONE]\n\n"
                )
                return httpx.Response(200, text=chunks, headers={"content-type": "text/event-stream"})
            if payload.get("response_format"):
                self.tool_payloads.append(payload)
                return httpx.Response(200, json=brain_body(self.structured_reply))
            self.tool_payloads.append(payload)
            turn = self.brain_turns.pop(0)
            return turn if isinstance(turn, httpx.Response) else httpx.Response(200, json=turn)

        if path.endswith("/guard"):
            self.guard_payloads.append(json.loads(request.content))
            return httpx.Response(
                200, json={"output": self.guard_output, "prompt_tokens": 10, "completion_tokens": 2}
            )
        if path.endswith("/ocr"):
            if isinstance(self.ocr, httpx.Response):
                return self.ocr
            return httpx.Response(200, json=self.ocr)
        if path.endswith("/segment"):
            self.segment_payloads.append(json.loads(request.content))
            if self.segment_fails:
                return httpx.Response(500, json={"detail": "sam2 exploded"})
            body = json.loads(request.content)
            boxes = body.get("boxes") or []
            scale = (
                (
                    lambda b: [
                        int(b[0] / 1000 * 800),
                        int(b[1] / 1000 * 600),
                        int(b[2] / 1000 * 800),
                        int(b[3] / 1000 * 600),
                    ]
                )
                if body.get("normalized")
                else (lambda b: [int(v) for v in b])
            )
            return httpx.Response(
                200,
                json={
                    "width": 800,
                    "height": 600,
                    "boxes": [scale(b) for b in boxes],
                    "masks": ["MASKDATA"] * len(boxes),
                },
            )
        if path.endswith("/diarize"):
            return httpx.Response(
                200,
                json={
                    "turns": [
                        {"speaker": "spk_0", "start": 0.0, "end": 2.0},
                        {"speaker": "spk_1", "start": 2.0, "end": 4.0},
                    ]
                },
            )
        if path.endswith("/transcribe"):
            self.transcribe_payloads.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "text": "hello there friend",
                    "language_detected": {"code": "en", "confidence": 1.0},
                    "chunks": [
                        {"text": "hello", "timestamp": [0.1, 0.9]},
                        {"text": "there", "timestamp": [1.0, 1.8]},
                        {"text": "friend", "timestamp": [2.5, 3.2]},
                    ],
                },
            )
        if request.method == "GET":
            if "bing.com" in str(request.url):
                return httpx.Response(200, headers={"content-type": "text/html"}, text="<html>")
            return httpx.Response(200, headers={"content-type": "image/png"}, content=_PNG_800x600)
        if path.endswith("/health"):
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404, json={"detail": f"unrouted {path}"})


@pytest.fixture
def client_for():
    def build(fake: Fake) -> TestClient:
        client = TestClient(app_module.app)
        client.__enter__()
        mock = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
        app_module.app.state.http = mock
        app_module.app.state.brain = BrainClient(mock, settings)
        return client

    return build


def test_tool_loop_runs_ocr_and_answers(client_for):
    fake = Fake(
        [
            brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"})]),
            brain_body(),  # second loop turn: no more tools
            brain_body("The invoice total is 42.00."),  # synthesis
        ]
    )
    client = client_for(fake)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "interfaze-lite",
            "messages": [user_image_message("what is the total?", "https://x.test/invoice.png")],
        },
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "The invoice total is 42.00."
    assert "/ocr" in fake.calls


def test_image_is_shown_to_the_model_only_after_a_tool_read_it(client_for):
    """A highlight is in the picture, not the OCR text; shown it before a tool ran, the model invented."""
    fake = Fake(
        [
            brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"})]),
            brain_body("GALE is highlighted."),
        ]
    )
    post_image(client_for(fake), "which item is highlighted?", "https://x.test/receipt.png")

    first, second = fake.tool_payloads[:2]
    assert image_urls(first) == []
    [shown] = image_urls(second)
    assert shown.startswith("data:image/")  # inlined, so vLLM never fetches it itself


def test_the_image_is_not_shown_after_a_failed_read(client_for):
    """With OCR down, the model read the image by eye and answered 144.68 for 144.02."""
    fake = Fake(
        [
            brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"})]),
            brain_body("The OCR tool is unavailable, so I cannot read the total."),
        ],
        ocr=httpx.Response(500, json={"detail": "ocr failed: perception down"}),
    )
    post_image(client_for(fake), "What is the total?", "https://x.test/receipt.png")
    second = fake.tool_payloads[1]
    assert not image_urls(second)


def test_a_pdf_sent_as_an_image_part_is_not_shown_as_one(client_for):
    """Clients send PDFs as image_url parts. vLLM cannot load one, and refused the request."""
    import base64

    pdf = "data:application/pdf;base64," + base64.b64encode(b"%PDF-1.4 not an image").decode()
    fake = Fake(
        [
            brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"})]),
            brain_body("Page 1 is an invoice."),
        ]
    )
    resp = post_image(client_for(fake), "what is on each page?", pdf)
    assert resp.status_code == 200, resp.text
    assert not image_urls(fake.tool_payloads[1])


def test_the_model_sees_where_each_page_starts(client_for):
    """Run together, a form's two pages read as one, and the model asked for page 2 again."""
    two_pages = {
        **OCR_FIXTURE,
        "text": "Form 1040-NR\n\nSchedule A",
        "total_pages": 2,
        "page_texts": ["Form 1040-NR", "Schedule A"],
        "pages_processed": [1, 2],
    }
    fake = Fake(
        [
            brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"})]),
            brain_body("done"),
        ],
        ocr=two_pages,
    )
    body = (
        client_for(fake)
        .post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all text"},
                            {
                                "type": "file",
                                "file": {"file_data": "https://x.test/f.pdf", "filename": "f.pdf"},
                            },
                        ],
                    }
                ]
            },
        )
        .json()
    )
    seen = json.loads(
        [m for m in fake.tool_payloads[-1]["messages"] if m.get("role") == "tool"][0]["content"]
    )
    assert seen["extracted_text"] == "--- page 1 ---\nForm 1040-NR\n\n--- page 2 ---\nSchedule A"
    assert seen["pages_read"] == [1, 2]
    assert body["precontext"][0]["result"]["extracted_text"] == "Form 1040-NR\n\nSchedule A"


def test_precontext_carries_full_sections_but_model_sees_less(client_for):
    """The two-payload split: bounds reach the caller, not the model's context."""
    fake = Fake(
        [
            brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"})]),
            brain_body(),
            brain_body("done"),
        ]
    )
    client = client_for(fake)
    body = client.post(
        "/v1/chat/completions",
        json={
            "messages": [user_image_only("https://x.test/a.png")],
        },
    ).json()

    [item] = body["precontext"]
    assert item["name"] == "ocr"
    assert item["result"]["sections"][0]["lines"][0]["average_confidence"] == 0.94

    tool_msgs = fake.tool_messages()
    assert tool_msgs, "expected a tool result message"
    returned = json.loads(tool_msgs[0]["content"])
    assert "extracted_text" in returned
    assert "sections" not in returned


def test_tools_are_advertised_with_strict_true(client_for):
    fake = Fake([brain_body(), brain_body("hi")])
    client = client_for(fake)
    post_image(client, "read this")

    tools = fake.tool_payloads[0]["tools"]
    assert {t["function"]["name"] for t in tools} == {"ocr", "stt", "object_detection", "gui_detection"}
    # Without strict:true vLLM silently skips grammar constraints under tool_choice=auto.
    assert all(t["function"]["strict"] for t in tools)
    assert fake.tool_payloads[0]["tool_choice"] == "auto"


def test_a_streamed_answer_arrives_as_it_is_generated(client_for):
    """Token by token, not one block at the end: the answer turn itself is streamed."""
    answer = ", ".join(str(n) for n in range(1, 200))
    fake = Fake([brain_body(answer)])
    client = client_for(fake)
    raw = stream_raw(client, {"stream": True, "messages": [{"role": "user", "content": "count to 199"}]})
    pieces = [
        json.loads(line[6:])["choices"][0]["delta"].get("content")
        for line in raw.splitlines()
        if line.startswith("data: ") and line[6:].strip() != "[DONE]"
    ]
    pieces = [p for p in pieces if p]
    assert len(pieces) > 1 and "".join(pieces) == answer
    assert fake.stream_payloads, "the answer turn must stream"


def test_a_tool_result_the_caller_sent_is_kept_inside_the_context_window(client_for):
    """A caller's own function returned 500k characters: the brain sees it trimmed."""
    fake = Fake([brain_body("Here is the summary.")])
    client_for(fake).post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "search the web for X"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("web_search", {"query": "X"})],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "x" * 500_000},
            ],
            "tools": [
                {"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}
            ],
        },
    )
    sent = fake.tool_payloads[0]["messages"]
    tool = next(m for m in sent if m.get("role") == "tool")
    assert len(tool["content"]) <= settings.max_context_chars + 100


def test_urls_in_a_tool_result_are_not_files_and_answering_from_it_is_not_nudged(client_for):
    """Registered as files, search-result URLs nudged the model into OCRing a page instead of answering."""
    fake = Fake([brain_body("Hopfield and Hinton.")])
    out = (
        client_for(fake)
        .post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Who won the 2024 Nobel Prize in Physics? Check https://www.nobelprize.org",
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call("web_search", {"query": "2024 Nobel physics"})],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "c1",
                        "content": json.dumps(
                            [
                                {
                                    "title": "Press release",
                                    "url": "https://www.nobelprize.org/prizes/physics/2024/press-release/",
                                }
                            ]
                        ),
                    },
                ],
                "tools": [
                    {"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}
                ],
            },
        )
        .json()
    )
    assert out["choices"][0]["message"]["content"] == "Hopfield and Hinton."
    assert len(fake.tool_payloads) == 1, "answered once, not nudged into a second turn"
    tools = {t["function"]["name"] for t in fake.tool_payloads[0].get("tools") or []}
    # The URL the user wrote is a file a tool may read; the one in the results is not.
    from interfaze_lite.filerefs import extract_from_messages

    refs, _ = extract_from_messages(
        [
            {"role": "user", "content": "Check https://www.nobelprize.org"},
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "see https://www.nobelprize.org/prizes/physics/2024/",
            },
        ]
    )
    assert [r.url for r in refs.refs.values()] == ["https://www.nobelprize.org"]
    assert "web_search" in tools


def _ocr_then_their_search():
    ocr = tool_call("ocr", {"file_ref_id": "ref-0"})
    search = tool_call("web_search", {"query": "store hours"}, "c2")
    request = {
        "messages": [user_image_message("When does this store close?", "https://x.test/a.png")],
        "tools": [{"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}],
    }
    return Fake([brain_body(tool_calls=[ocr]), brain_body(tool_calls=[search])]), request


class TestHandbackContext:
    """Through the relay, the next request lost its own OCR and read both receipts again."""

    def test_json(self, client_for):
        fake, request = _ocr_then_their_search()
        message = client_for(fake).post("/v1/chat/completions", json=request).json()["choices"][0]["message"]
        assert [c["function"]["name"] for c in message["tool_calls"]] == ["web_search"]
        context = message["interfaze_context"]
        assert [m["role"] for m in context] == ["assistant", "tool"]
        assert context[0]["tool_calls"][0]["function"]["name"] == "ocr"
        assert context[1]["tool_call_id"] == "c1" and context[1]["content"]

    def test_stream(self, client_for):
        fake, request = _ocr_then_their_search()
        raw = stream_raw(client_for(fake), {**request, "stream": True})
        chunks = sse_chunks(raw)
        last = next(c for c in chunks if (c.get("choices") or [{}])[0].get("finish_reason") == "tool_calls")
        assert [m["role"] for m in last["interfaze_context"]] == ["assistant", "tool"]

    def test_nothing_to_carry_adds_nothing(self, client_for):
        search = tool_call("web_search", "{}", "c2")
        fake = Fake([brain_body(tool_calls=[search])])
        message = (
            client_for(fake)
            .post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "search it"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": "web_search", "parameters": {"type": "object"}},
                        }
                    ],
                },
            )
            .json()["choices"][0]["message"]
        )
        assert "interfaze_context" not in message


def test_what_the_model_says_before_a_tool_call_is_not_streamed(client_for):
    """The landing's answer opened "Let me read the surrounding text...", which the
    buffered response never shows. Only the answer itself is streamed."""
    ocr = tool_call("ocr", {"file_ref_id": "ref-0"})
    narrated = brain_body(
        tool_calls=[
            {
                **ocr,
                "id": "c2",
                "function": {
                    "name": "ocr",
                    "arguments": json.dumps({"file_ref_id": "ref-0", "return_markdown": True}),
                },
            }
        ]
    )
    narrated["choices"][0]["message"]["content"] = "Let me read it again as markdown to be sure."
    fake = Fake([brain_body(tool_calls=[ocr]), narrated, brain_body("The total is 42.")])
    raw = stream_raw(
        client_for(fake),
        {
            "stream": True,
            "messages": [user_image_message("what is the total?", "https://x.test/a.png")],
        },
    )
    text = streamed_text(raw)
    assert text == "The total is 42."


def test_a_stream_that_fails_midway_ends_with_an_error_chunk(client_for):
    """In the OpenAI error shape, not as answer text carrying the raw exception."""
    fake = Fake([])  # the brain call fails: there is no turn to answer with
    raw = stream_raw(client_for(fake), {"stream": True, "messages": [{"role": "user", "content": "hi"}]})
    events = [line[6:] for line in raw.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    error = json.loads(events[-2])["error"]
    assert error["type"] == "server_error" and error["request_id"]
    assert "pop from empty list" not in raw


def test_streaming_emits_precontext_before_content(client_for):
    fake = Fake(
        [
            brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"})]),
            brain_body(),
            brain_body("The total is 42."),
        ]
    )
    client = client_for(fake)
    raw = stream_raw(
        client,
        {
            "stream": True,
            "messages": [user_image_only("https://x.test/a.png")],
        },
        headers={"x-show-additional-info": "true"},
    )

    assert raw.rstrip().endswith("data: [DONE]")
    text = streamed_text(raw)
    assert text.index("<precontext>") < text.index("The total")

    from interfaze_lite.envelope import parse_precontext, strip_precontext

    assert parse_precontext(text)[0]["name"] == "ocr"
    assert strip_precontext(text) == "The total is 42."


def test_no_precontext_block_when_no_tool_ran(client_for):
    """Each relay round opened with `<precontext> [] </precontext>`, and the relay's log of
    a web search answer began with two of them before its JSON."""
    for body in (
        {},
        {
            "response_format": json_schema_format(
                "a", {"type": "object", "properties": {"a": {"type": "integer"}}}
            )
        },
    ):
        fake = Fake([brain_body("Hello there."), brain_body("Hello there.")])
        raw = stream_raw(
            client_for(fake),
            {
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
                **body,
            },
            headers={"x-show-additional-info": "true"},
        )
        assert raw.rstrip().endswith("data: [DONE]")
        assert "precontext" not in raw, body


def test_speaker_attribution_runs_through_the_real_fusion(client_for):
    fake = Fake(
        [
            brain_body(tool_calls=[tool_call("stt", {"file_ref_id": "ref-0", "split_by_speaker": True})]),
            brain_body(),
            brain_body("two speakers."),
        ]
    )
    client = client_for(fake)
    body = client.post(
        "/v1/chat/completions",
        json={
            "messages": [user_audio_message("who said what?")],
        },
    ).json()

    chunks = body["precontext"][0]["result"]["chunks"]
    # hello+there fall in spk_0's [0,2]; friend at [2.5,3.2] falls in spk_1's [2,4].
    assert [c["speaker"] for c in chunks] == ["spk_0", "spk_1"]
    assert chunks[0]["text"] == "hello there"
    assert "/diarize" in fake.calls


def test_unknown_tool_is_reported_to_the_model_not_raised(client_for):
    fake = Fake(
        [
            brain_body(tool_calls=[tool_call("nonexistent", "{}")]),
            brain_body(),
            brain_body("sorry"),
        ]
    )
    client = client_for(fake)
    resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "go"}]})
    assert resp.status_code == 200
    assert "error" in resp.json()["precontext"][0]["result"]


def test_bad_file_ref_comes_back_as_a_retryable_tool_error(client_for):
    fake = Fake(
        [
            brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "invoice.pdf"})]),
            brain_body(),
            brain_body("could not read it"),
        ]
    )
    client = client_for(fake)
    body = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "read invoice.pdf"}]}
    ).json()
    result = body["precontext"][0]["result"]
    assert "not a file reference" in result["error"]
    assert result["message"].startswith("Retry")


def test_tool_loop_is_capped(client_for, monkeypatch):
    """A model that never stops calling tools must not run until the request times out."""
    monkeypatch.setattr(settings, "max_tool_steps", 3)
    calls = [
        brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0", "page_range": [i, i]}, f"c{i}")])
        for i in (1, 2, 3)
    ]
    fake = Fake([*calls, brain_body("stopped")])
    client = client_for(fake)
    body = client.post(
        "/v1/chat/completions",
        json={
            "messages": [user_image_only("https://x.test/a.png")],
        },
    ).json()
    assert fake.calls.count("/ocr") == 3
    # Out of steps, the model is told to answer: offered no tools, it reached for one,
    # and the stripped call left an empty answer.
    last = fake.tool_payloads[-1]["messages"][-1]
    assert last["role"] == "user" and last["content"].startswith("No more tool calls are possible")
    assert body["choices"][0]["message"]["content"] == "stopped"


def test_the_same_call_again_is_answered_from_its_result_not_run_again(client_for):
    """A receipt was detected seven times over with the same prompts, to the step cap."""
    call = brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"})])
    fake = Fake([call, call, call, brain_body("stopped")])
    body = (
        client_for(fake)
        .post(
            "/v1/chat/completions",
            json={
                "messages": [user_image_only("https://x.test/a.png")],
            },
        )
        .json()
    )
    assert [p["name"] for p in body["precontext"]] == ["ocr"]
    assert fake.calls.count("/ocr") == 1
    last = fake.tool_payloads[-1]["messages"][-1]
    assert last["role"] == "user" and last["content"].startswith("No more tool calls are possible")
    assert body["choices"][0]["message"]["content"] == "stopped"


def test_a_call_answered_in_an_earlier_round_is_not_run_again(client_for):
    """Through the relay, each round re-ran the OCR the last round already carried back."""
    ocr = tool_call("ocr", {"file_ref_id": "ref-0"}, "o1")
    fake = Fake([brain_body(tool_calls=[{**ocr, "id": "o2"}]), brain_body("Done.")])
    body = (
        client_for(fake)
        .post(
            "/v1/chat/completions",
            json={
                "messages": [
                    user_image_message("read it", "https://x.test/a.png"),
                    {"role": "assistant", "content": None, "tool_calls": [ocr]},
                    {
                        "role": "tool",
                        "tool_call_id": "o1",
                        "content": json.dumps({"extracted_text": "TOTAL 244.02"}),
                    },
                ]
            },
        )
        .json()
    )
    assert "/ocr" not in fake.calls
    assert body["precontext"] == []
    assert body["choices"][0]["message"]["content"] == "Done."


def test_forecast_is_offered_without_dataset_when_the_series_is_in_the_message(client_for):
    """Offered `dataset`, the model copied a year of rows into the call one by one."""
    series = "Forecast the next 3 days:\ndate,sales\n" + "\n".join(
        f"2024-06-{d:02d},{1000 + d * 10}" for d in range(1, 15)
    )
    fake = Fake([brain_body("ok")])
    client_for(fake).post("/v1/chat/completions", json={"messages": [{"role": "user", "content": series}]})
    forecast = next(t for t in fake.tool_payloads[0]["tools"] if t["function"]["name"] == "forecast")
    assert "dataset" not in forecast["function"]["parameters"]["properties"]
    assert "steps" in forecast["function"]["parameters"]["properties"]


def test_an_internal_failure_reaches_neither_the_model_nor_the_caller(client_for):
    """A CUDA out-of-memory message, allocation figures and all, reached the landing."""
    oom = httpx.Response(
        500,
        json={
            "detail": "ocr failed: OutOfMemoryError: CUDA out of memory. "
            "Tried to allocate 1.34 GiB. GPU 0 has a total capacity of 79.18 GiB"
        },
    )
    ocr = tool_call("ocr", {"file_ref_id": "ref-0"})
    fake = Fake([brain_body(tool_calls=[ocr]), brain_body("Sorry, that failed.")], ocr=oom)
    body = post_image(client_for(fake), "read it", "https://x.test/a.png").json()
    told = json.dumps(fake.tool_payloads[-1]["messages"]) + json.dumps(body)
    assert "CUDA" not in told and "GiB" not in told
    assert "Something went wrong running OCR" in told


def test_a_task_that_ran_its_tool_three_times_answers_with_all_three(client_for, monkeypatch):
    """A translation into French, Spanish and Japanese came back as the French alone."""
    import interfaze_lite.tools as tk

    async def translated(args, ctx):
        return tk.ToolResult(
            model_facing={
                "translated_text": f"[{args['target_language']}]",
                "target_language": args["target_language"],
            }
        )

    monkeypatch.setattr(tk.REGISTRY["translate"], "execute", translated)
    calls = [
        tool_call(
            "translate", {"text": "Good morning", "target_language": lang, "current_language": "en"}, f"t{i}"
        )
        for i, lang in enumerate(("fr", "es", "ja"))
    ]
    fake = Fake([brain_body(tool_calls=calls)])
    body = (
        client_for(fake)
        .post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "<task>translate</task>"},
                    {
                        "role": "user",
                        "content": "Translate 'Good morning' into French, Spanish and Japanese.",
                    },
                ]
            },
        )
        .json()
    )
    content = json.loads(body["choices"][0]["message"]["content"])
    assert content["name"] == "translate" and "result" in content
    assert [r["name"] for r in content["results"]] == ["translate"] * 3
    assert [r["result"]["target_language"] for r in content["results"]] == ["fr", "es", "ja"]


def test_a_structured_task_streams_its_result_alone_as_interfaze_does(client_for):
    """The playground's run-task cards send a response_format; interfaze streams no <precontext> then."""
    det = tool_call("object_detection", {"file_ref_id": "ref-0", "prompts": ["dog"]}, "d1")
    request = {
        "stream": True,
        "messages": [
            {"role": "system", "content": "<task>object_detection</task>"},
            user_image_message("find the dog", "https://x.test/a.png"),
        ],
    }
    texts = {}
    for label, extra in (
        ("plain", {}),
        (
            "structured",
            {
                "response_format": json_schema_format(
                    "response", {"type": "object", "properties": {}, "additionalProperties": True}
                )
            },
        ),
    ):
        fake = Fake([brain_body(tool_calls=[det])])
        raw = stream_raw(client_for(fake), {**request, **extra}, headers={"x-show-additional-info": "true"})
        texts[label] = "".join(
            json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
            for line in raw.splitlines()
            if line.startswith("data: ")
            and line[6:].strip() != "[DONE]"
            and json.loads(line[6:]).get("choices")
        )
    assert "<precontext>" in texts["plain"]
    assert "<precontext>" not in texts["structured"]
    assert json.loads(texts["structured"])["name"] == "object_detection"


def test_a_tool_call_cut_off_at_the_token_limit_is_not_run(client_for):
    """Its arguments are whatever was written before the cut: a copied table arrived empty."""
    cut = brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"})])
    cut["choices"][0]["finish_reason"] = "length"
    fake = Fake([cut, brain_body("done")])
    body = (
        client_for(fake)
        .post(
            "/v1/chat/completions",
            json={
                "messages": [user_image_only("https://x.test/a.png")],
            },
        )
        .json()
    )
    assert not any(c.endswith("/ocr") for c in fake.calls)
    assert "cut off at the output token limit" in body["precontext"][0]["result"]["error"]


def test_health_and_models():
    client = TestClient(app_module.app)
    with client:
        assert client.get("/v1/models").json()["data"][0]["id"] == "interfaze-lite"


class TestChatTemplateConstraints:
    def test_exactly_one_system_message(self, client_for):
        """Qwen rejects a second system message with 'System message must be at the
        beginning', so the file manifest has to ride inside the first one."""
        fake = Fake([brain_body(), brain_body("hi")])
        client = client_for(fake)
        post_image(client, "read it", "https://x.test/a.png")

        msgs = fake.tool_payloads[0]["messages"]
        systems = [m for m in msgs if m["role"] == "system"]
        assert len(systems) == 1
        assert msgs[0]["role"] == "system"
        # ...and the manifest must still be in there, or the model cannot resolve ref-0.
        assert "ref-0" in systems[0]["content"]

    def test_routing_turns_disable_thinking(self, client_for):
        fake = Fake([brain_body(), brain_body("hi")])
        client = client_for(fake)
        client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        kwargs = fake.tool_payloads[0].get("chat_template_kwargs") or {}
        assert kwargs.get("enable_thinking") is False


class TestTextAuthority:
    """extracted_text comes from the document VLM: re-joined sections carried detector text and disagreed."""

    def test_extracted_text_comes_from_the_vlm_field(self, client_for):
        fixture = {
            **OCR_FIXTURE,
            "text": "AUTHORITATIVE VLM TEXT",
            "sections": [{"text": "something else entirely", "lines": []}],
        }
        fake = Fake(
            [
                brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"})]),
                brain_body(),
                brain_body("done"),
            ],
            ocr=fixture,
        )
        client = client_for(fake)
        body = client.post(
            "/v1/chat/completions",
            json={
                "messages": [user_image_only("https://x.test/a.png")],
            },
        ).json()

        result = body["precontext"][0]["result"]
        assert result["extracted_text"] == "AUTHORITATIVE VLM TEXT"

        # ...and the model-facing payload shows the same string, not a different one.
        tool_msgs = fake.tool_messages()
        assert json.loads(tool_msgs[0]["content"])["extracted_text"] == "AUTHORITATIVE VLM TEXT"


class TestBrainDetection:
    """Detection grounds with the brain and segments with SAM 2.1; `perception.segment` says why."""

    def test_boxes_from_brain_masks_from_segment(self, client_for, monkeypatch):
        fake = Fake(
            [
                brain_body(
                    tool_calls=[tool_call("object_detection", {"file_ref_id": "ref-0", "prompts": ["dog"]})]
                ),
                brain_body(),
                brain_body("one dog"),
            ]
        )
        client = client_for(fake)
        body = client.post(
            "/v1/chat/completions",
            json={
                "messages": [user_image_only("https://x.test/a.png")],
            },
        ).json()

        objects = body["precontext"][0]["result"]["detected_objects"]
        assert objects and objects[0]["label"] == "dog"
        # the 0-1000 grid rescaled to the image's own 800x600 pixels
        assert objects[0]["bounds"]["top_left"] == {"x": 80, "y": 60}
        assert objects[0]["mask"] == "MASKDATA"
        assert "/segment" in fake.calls

    def test_segmentation_failure_still_returns_boxes(self, client_for, monkeypatch):
        """Losing masks must not cost the caller their detections."""
        fake = Fake(
            [
                brain_body(
                    tool_calls=[tool_call("object_detection", {"file_ref_id": "ref-0", "prompts": ["dog"]})]
                ),
                brain_body(),
                brain_body("one dog"),
            ]
        )
        fake.segment_fails = True
        client = client_for(fake)
        body = client.post(
            "/v1/chat/completions",
            json={
                "messages": [user_image_only("https://x.test/a.png")],
            },
        ).json()
        objects = body["precontext"][0]["result"]["detected_objects"]
        assert objects and objects[0]["label"] == "dog"
        assert not objects[0].get("mask")


class TestSchemaPrompt:
    def test_schema_instruction_folded_into_the_single_system_message(self, client_for):
        fake = Fake([brain_body(), brain_body('{"a":1}')])
        client = client_for(fake)
        client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "extract"}],
                "response_format": json_schema_format(
                    "x", {"type": "object", "properties": {"a": {"type": "number"}}}
                ),
            },
        )
        msgs = fake.tool_payloads[-1]["messages"]
        systems = [m for m in msgs if m["role"] == "system"]
        assert len(systems) == 1
        assert msgs[0]["role"] == "system"

    def test_structured_call_drops_the_tool_prompt_but_keeps_the_callers(self, client_for):
        # The tool-calling prompt asks for tool calls and free reasoning, which guided
        # decoding forbids; with it in context a receipt came back as '}, {' strings.
        fake = Fake([brain_body(), brain_body('{"a":1}')])
        client = client_for(fake)
        client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "Prices in USD."},
                    {"role": "user", "content": "extract"},
                ],
                "response_format": json_schema_format(
                    "x", {"type": "object", "properties": {"a": {"type": "number"}}}
                ),
            },
        )
        schema_call = fake.last_structured_payload()
        system = schema_call["messages"][0]["content"]
        assert "structured answer" in system
        assert "Prices in USD." in system
        assert "Your tools" not in system
        assert any(str(m.get("content")).startswith("extract") for m in schema_call["messages"])


class TestToolMemoisation:
    def test_identical_call_runs_once(self, client_for):
        call = tool_call("ocr", {"file_ref_id": "ref-0"})
        again = tool_call("ocr", {"file_ref_id": "ref-0"}, "c2")
        fake = Fake(
            [brain_body(tool_calls=[call]), brain_body(tool_calls=[again]), brain_body(), brain_body("done")]
        )
        client = client_for(fake)
        client.post(
            "/v1/chat/completions",
            json={"messages": [user_image_only("https://x.test/a.png")]},
        )
        assert fake.calls.count("/ocr") == 1

    def test_explicit_default_matches_omitted_default(self, client_for):
        call = tool_call("ocr", {"file_ref_id": "ref-0"})
        again = tool_call("ocr", {"file_ref_id": "ref-0", "return_markdown": False}, "c2")
        fake = Fake(
            [brain_body(tool_calls=[call]), brain_body(tool_calls=[again]), brain_body(), brain_body("done")]
        )
        client = client_for(fake)
        client.post(
            "/v1/chat/completions",
            json={"messages": [user_image_only("https://x.test/a.png")]},
        )
        assert fake.calls.count("/ocr") == 1

    def test_another_option_on_the_same_pages_reads_the_document_once(self, client_for):
        """Asked for line boxes after the markdown, a receipt was read twice, ~20 s each."""
        call = tool_call("ocr", {"file_ref_id": "ref-0", "return_markdown": True})
        other = tool_call("ocr", {"file_ref_id": "ref-0", "return_bounds": True}, "c2")
        fake = Fake(
            [brain_body(tool_calls=[call]), brain_body(tool_calls=[other]), brain_body(), brain_body("done")]
        )
        body = (
            client_for(fake)
            .post(
                "/v1/chat/completions",
                json={"messages": [user_image_only("https://x.test/a.png")]},
            )
            .json()
        )
        assert fake.calls.count("/ocr") == 1
        assert [p["name"] for p in body["precontext"]] == ["ocr"]

    def test_other_pages_still_run(self, client_for):
        call = tool_call("ocr", {"file_ref_id": "ref-0", "page_range": [1, 1]})
        other = tool_call("ocr", {"file_ref_id": "ref-0", "page_range": [2, 2]}, "c2")
        fake = Fake(
            [brain_body(tool_calls=[call]), brain_body(tool_calls=[other]), brain_body(), brain_body("done")]
        )
        client_for(fake).post(
            "/v1/chat/completions",
            json={"messages": [user_image_only("https://x.test/a.png")]},
        )
        assert fake.calls.count("/ocr") == 2


def test_a_file_with_no_text_is_said_to_have_none(client_for):
    """interfaze answers "I did not find any text in this image"; the model is told so."""
    empty = {**OCR_FIXTURE, "text": "", "context": "", "sections": [], "has_text": False}
    fake = Fake(
        [
            brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"})]),
            brain_body("There is no text in this image."),
        ],
        ocr=empty,
    )
    post_image(client_for(fake), "Extract all text", "https://x.test/giraffe.jpg")
    seen = json.loads(fake.tool_messages()[0]["content"])
    assert seen["extracted_text"] == "" and "No text" in seen["note"]


class TestReadsKept:
    """A document read lately is not read again: relay rounds and landing demos re-read
    the same receipt, 7-28 s each."""

    URL = "https://x.test/receipt.png"

    def _backend(self, monkeypatch, *replies):
        from interfaze_lite.tools import ocr

        calls = []

        async def post(ctx, base, path, payload):
            calls.append(payload)
            await asyncio.sleep(0.01)
            reply = replies[min(len(calls), len(replies)) - 1]
            if isinstance(reply, Exception):
                raise reply
            return reply

        monkeypatch.setattr(ocr, "_post", post)
        return ocr, calls

    def _ctx(self, zdr=False):
        return types.SimpleNamespace(zdr=zdr, settings=types.SimpleNamespace(perception_url="http://p"))

    def test_a_later_request_is_answered_from_the_earlier_read(self, monkeypatch):
        ocr, calls = self._backend(monkeypatch, OCR_FIXTURE)
        first = asyncio.run(ocr._read(self.URL, None, self._ctx()))
        again = asyncio.run(ocr._read(self.URL, None, self._ctx()))
        assert len(calls) == 1 and again == first

    def test_identical_reads_at_once_share_one(self, monkeypatch):
        ocr, calls = self._backend(monkeypatch, OCR_FIXTURE)

        async def both():
            return await asyncio.gather(*(ocr._read(self.URL, None, self._ctx()) for _ in range(3)))

        assert len({json.dumps(r, sort_keys=True) for r in asyncio.run(both())}) == 1
        assert len(calls) == 1

    def test_other_pages_are_read(self, monkeypatch):
        ocr, calls = self._backend(monkeypatch, OCR_FIXTURE)
        asyncio.run(ocr._read(self.URL, [1, 1], self._ctx()))
        asyncio.run(ocr._read(self.URL, [2, 2], self._ctx()))
        assert len(calls) == 2

    def test_a_zero_data_retention_read_is_not_kept(self, monkeypatch):
        ocr, calls = self._backend(monkeypatch, OCR_FIXTURE)
        asyncio.run(ocr._read(self.URL, None, self._ctx(zdr=True)))
        asyncio.run(ocr._read(self.URL, None, self._ctx()))
        assert len(calls) == 2 and len(ocr._reads) == 1

    def test_a_failed_read_is_tried_again(self, monkeypatch):
        ocr, calls = self._backend(monkeypatch, RuntimeError("Something went wrong"), OCR_FIXTURE)
        with pytest.raises(RuntimeError):
            asyncio.run(ocr._read(self.URL, None, self._ctx()))
        assert asyncio.run(ocr._read(self.URL, None, self._ctx())) == OCR_FIXTURE
        assert len(calls) == 2

    def test_a_read_that_found_nothing_is_tried_again(self, monkeypatch):
        empty = {**OCR_FIXTURE, "text": "", "has_text": False}
        ocr, calls = self._backend(monkeypatch, empty, OCR_FIXTURE)
        asyncio.run(ocr._read(self.URL, None, self._ctx()))
        asyncio.run(ocr._read(self.URL, None, self._ctx()))
        assert len(calls) == 2

    def test_an_hour_old_read_is_read_again(self, monkeypatch):
        ocr, calls = self._backend(monkeypatch, OCR_FIXTURE)
        asyncio.run(ocr._read(self.URL, None, self._ctx()))
        for key, (at, data) in list(ocr._reads.items()):
            ocr._reads[key] = (at - ocr._READ_TTL_S - 1, data)
        asyncio.run(ocr._read(self.URL, None, self._ctx()))
        assert len(calls) == 2


class TestContextCap:
    def test_long_ocr_text_is_capped_for_the_model_not_the_caller(self, client_for, monkeypatch):
        monkeypatch.setattr(settings, "max_context_chars", 100)
        long_text = "x" * 5000
        fake = Fake(
            [
                brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"})]),
                brain_body(),
                brain_body("done"),
            ],
            ocr={**OCR_FIXTURE, "text": long_text},
        )
        client = client_for(fake)
        body = client.post(
            "/v1/chat/completions",
            json={
                "messages": [user_image_only("https://x.test/a.png")],
            },
        ).json()

        caller = body["precontext"][0]["result"]["extracted_text"]
        assert len(caller) == 5000

        tool_msgs = fake.tool_messages()
        seen = json.loads(tool_msgs[0]["content"])
        assert len(seen["extracted_text"]) == 100
        assert seen["truncated"] is True

    def test_short_text_is_untouched(self, client_for, monkeypatch):
        monkeypatch.setattr(settings, "max_context_chars", 100_000)
        fake = Fake(
            [
                brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"})]),
                brain_body(),
                brain_body("done"),
            ]
        )
        client = client_for(fake)
        client.post(
            "/v1/chat/completions",
            json={
                "messages": [user_image_only("https://x.test/a.png")],
            },
        )
        tool_msgs = fake.tool_messages()
        assert "truncated" not in json.loads(tool_msgs[0]["content"])


class TestNoImprovisedBrowsing:
    def test_html_url_is_refused_not_ocrd(self, client_for):
        fake = Fake(
            [
                brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "https://www.bing.com/search?q=x"})]),
                brain_body(),
                brain_body("I cannot search the web."),
            ]
        )
        client = client_for(fake)
        body = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "read https://www.bing.com/search?q=x"}]},
        ).json()
        result = body["precontext"][0]["result"]
        assert "not a document" in result["error"]
        assert "/ocr" not in fake.calls

    def test_a_url_the_request_never_carried_is_not_fetched(self, client_for):
        # The model invented a URL on an image request; only the request's own files resolve.
        fake = Fake(
            [
                brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "https://elsewhere.example/a.png"})]),
                brain_body(),
                brain_body("done"),
            ]
        )
        body = post_image(client_for(fake), "read").json()
        assert "not a file in this request" in body["precontext"][0]["result"]["error"]
        assert "/ocr" not in fake.calls

    def test_text_only_requests_are_offered_no_file_tools(self, client_for):
        fake = Fake([brain_body("positive")])
        client_for(fake).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Classify the sentiment: I love it."}]},
        )
        assert "tools" not in fake.tool_payloads[0]

    @pytest.mark.parametrize(
        "content, offered",
        [
            ("Translate 'good morning' into French.", {"translate"}),
            ("Hello, how are you? — in Spanish please", {"translate"}),
            ("Forecast the next 4 weeks of this: date,value\n2024-01-01,1", {"forecast"}),
            ("Weekly sales: [...]. What can we expect over the next month?", {"forecast"}),
            # A passage that mentions a language mid-way is material, not an instruction.
            (
                "Context: "
                + "x " * 200
                + "The novel was known in English as The Trial. "
                + "x " * 200
                + "Question: Who wrote it? Respond with JSON.",
                set(),
            ),
        ],
    )
    def test_text_tools_are_offered_when_the_instruction_asks(self, client_for, content, offered):
        """An English answer was "translated" into English when translate was always on offer."""
        fake = Fake([brain_body("ok")])
        client_for(fake).post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": content}]}
        )
        assert {t["function"]["name"] for t in fake.tool_payloads[0].get("tools", [])} == offered

    def test_a_routed_task_is_offered_its_tool(self, client_for):
        fake = Fake([brain_body("ok")])
        client_for(fake).post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "<task>translate</task>"},
                    {"role": "user", "content": "Bonjour"},
                ]
            },
        )
        assert [t["function"]["name"] for t in fake.tool_payloads[0]["tools"]] == ["translate"]

    def test_capabilities_and_limits_are_stated(self, client_for):
        fake = Fake([brain_body(), brain_body("hi")])
        client = client_for(fake)
        client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        system = fake.tool_payloads[0]["messages"][0]["content"]
        flat = " ".join(system.split())
        # Web search, scraping and code are the caller's to offer, not the brain's to claim.
        assert "running code are not tools of yours" in flat
        assert "when the request offers functions for them, use them" in flat.lower()
        assert "when it offers none, say you cannot do it" in flat.lower()
        assert "never decline something a tool can do" in flat.lower()
        # Called when the task needs them, not every one that might apply: a question about
        # a team name printed in an image went to web search.
        assert "call the ones the task needs" in flat.lower()
        assert "never for what an attached image, document or recording contains" in flat
        for tool in ("ocr", "stt", "object_detection", "gui_detection"):
            assert tool in flat


class TestDirectAnswer:
    def test_answer_from_the_tool_selection_turn_is_kept(self, client_for):
        fake = Fake([brain_body("I cannot search the web.")])
        client = client_for(fake)
        body = client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "search for X"}]}
        ).json()
        assert body["choices"][0]["message"]["content"] == "I cannot search the web."
        assert body["precontext"] == []

    def test_never_returns_empty_when_the_model_spoke(self, client_for):
        fake = Fake([brain_body("refusal text"), brain_body("")])
        client = client_for(fake)
        body = client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "search for X"}]}
        ).json()
        assert body["choices"][0]["message"]["content"].strip()


class TestCallerSystemMessage:
    def test_caller_system_is_merged_not_appended(self, client_for):
        fake = Fake([brain_body(), brain_body("ok")])
        client = client_for(fake)
        client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "You are a pirate."},
                    {"role": "user", "content": "hi"},
                ]
            },
        )
        msgs = fake.tool_payloads[0]["messages"]
        systems = [m for m in msgs if m["role"] == "system"]
        assert len(systems) == 1
        assert msgs[0]["role"] == "system"
        assert "You are a pirate." in systems[0]["content"]
        assert "Interfaze" in systems[0]["content"]

    def test_multiple_caller_system_messages_all_merge(self, client_for):
        fake = Fake([brain_body(), brain_body("ok")])
        client = client_for(fake)
        client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "Rule one."},
                    {"role": "user", "content": "hi"},
                    {"role": "system", "content": "Rule two."},
                ]
            },
        )
        msgs = fake.tool_payloads[0]["messages"]
        assert len([m for m in msgs if m["role"] == "system"]) == 1
        assert "Rule one." in msgs[0]["content"] and "Rule two." in msgs[0]["content"]


class TestStructuredNotEmpty:
    def test_structured_call_disables_thinking(self, client_for):
        fake = Fake([brain_body(), brain_body('{"a":1}')])
        client = client_for(fake)
        client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "extract"}],
                "response_format": json_schema_format("x", {"type": "object"}),
            },
        )
        schema_call = fake.last_structured_payload()
        assert schema_call["chat_template_kwargs"]["enable_thinking"] is False

    def test_reasoning_content_used_when_content_empty(self, client_for):
        body = brain_body("")
        body["choices"][0]["message"]["reasoning_content"] = "the actual answer"
        fake = Fake([body])
        client = client_for(fake)
        out = client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        ).json()
        assert out["choices"][0]["message"]["content"] == "the actual answer"


class TestRequestValidation:
    def test_empty_message_is_400(self, client_for):
        r = client_for(Fake([])).post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": ""}]}
        )
        assert r.status_code == 400
        assert "no text" in r.json()["error"]["message"]

    def test_bad_base64_is_400(self, client_for):
        r = post_image(
            client_for(Fake([])), "what is this", "data:image/jpeg;base64,@@@@not-valid-base64@@@@===="
        )
        assert r.status_code == 400
        assert "base64" in r.json()["error"]["message"]

    def test_local_path_as_image_url_is_400(self, client_for):
        r = post_image(client_for(Fake([])), "what is this", "/etc/passwd")
        assert r.status_code == 400
        assert "http(s) URL or a data: URI" in r.json()["error"]["message"]

    def test_multiple_tasks_is_400(self, client_for):
        r = client_for(Fake([])).post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "<task>ocr, web_search</task>"},
                    {"role": "user", "content": "hi"},
                ]
            },
        )
        assert r.status_code == 400
        assert "only one task" in r.json()["error"]["message"]

    def test_invalid_task_is_400(self, client_for):
        r = client_for(Fake([])).post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "<task>foobar_tool</task>"},
                    {"role": "user", "content": "hi"},
                ]
            },
        )
        assert r.status_code == 400
        assert "invalid task" in r.json()["error"]["message"]

    def test_valid_task_is_accepted(self, client_for):
        fake = Fake([brain_body("done")])
        r = client_for(fake).post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "<task>ocr</task>"},
                    {"role": "user", "content": "read it"},
                ]
            },
        )
        assert r.status_code == 200
        system = fake.tool_payloads[0]["messages"][0]["content"]
        assert "<task>" not in system
        assert "routed this request to the `ocr` tool" in system

    def test_invalid_key_is_401(self, client_for, monkeypatch):
        monkeypatch.setattr(settings, "api_key", "sk_real")
        r = client_for(Fake([])).post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer sk_wrong"},
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 401

    def test_correct_key_passes(self, client_for, monkeypatch):
        monkeypatch.setattr(settings, "api_key", "sk_real")
        r = client_for(Fake([brain_body("hi")])).post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer sk_real"},
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200

    def test_open_by_default(self, client_for):
        r = client_for(Fake([brain_body("hi")])).post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert r.status_code == 200

    def test_admin_key_header_passes(self, client_for, monkeypatch):
        # How the interfaze API calls the hosted deployment, as it calls every model service.
        monkeypatch.setattr(settings, "admin_key", "admin_real")
        r = client_for(Fake([brain_body("hi")])).post(
            "/v1/chat/completions",
            headers={"x-api-admin-key": "admin_real"},
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200

    def test_admin_key_required_once_set(self, client_for, monkeypatch):
        monkeypatch.setattr(settings, "admin_key", "admin_real")
        for headers in ({}, {"x-api-admin-key": "admin_wrong"}, {"authorization": "Bearer admin_real"}):
            r = client_for(Fake([])).post(
                "/v1/chat/completions",
                headers=headers,
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            assert r.status_code == 401, headers

    def test_either_key_passes_when_both_are_set(self, client_for, monkeypatch):
        monkeypatch.setattr(settings, "api_key", "sk_real")
        monkeypatch.setattr(settings, "admin_key", "admin_real")
        for headers in ({"authorization": "Bearer sk_real"}, {"x-api-admin-key": "admin_real"}):
            r = client_for(Fake([brain_body("hi")])).post(
                "/v1/chat/completions",
                headers=headers,
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            assert r.status_code == 200, headers


class TestTaskRouting:
    def test_task_returns_raw_tool_output(self, client_for):
        fake = Fake(
            [
                brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"})]),
                brain_body("prose"),
            ]
        )
        r = client_for(fake).post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "<task>ocr</task>"},
                    user_image_message("read", "https://x/y.png"),
                ]
            },
        )
        body = r.json()
        assert json.loads(body["choices"][0]["message"]["content"])["name"] == "ocr"
        assert body["precontext"][0]["name"] == "ocr"

    def test_empty_task_schema_does_not_trigger_guided_decoding(self, client_for):
        fake = Fake([brain_body("plain answer")])
        r = client_for(fake).post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": json_schema_format("empty_schema", {}),
            },
        )
        assert r.status_code == 200
        assert not any(p.get("response_format") for p in fake.tool_payloads)
        assert r.json()["choices"][0]["message"]["content"] == "plain answer"


class TestReasoning:
    def test_reasoning_effort_surfaces_reasoning(self, client_for):
        body = brain_body("1175")
        body["choices"][0]["message"]["reasoning_content"] = "25*47 = 1175"
        fake = Fake([brain_body("route"), body])
        out = (
            client_for(fake)
            .post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "25*47?"}], "reasoning_effort": "high"},
            )
            .json()
        )
        assert out["reasoning"] == "25*47 = 1175"
        assert out["choices"][0]["message"]["content"] == "1175"
        assert fake.tool_payloads[-1]["chat_template_kwargs"]["enable_thinking"] is True

    def test_thinking_off_by_default(self, client_for):
        fake = Fake([brain_body("hi")])
        client_for(fake).post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        assert fake.tool_payloads[-1]["chat_template_kwargs"]["enable_thinking"] is False

    def test_stream_wraps_reasoning_in_think_tags(self, client_for):
        body = brain_body("haiku here")
        body["choices"][0]["message"]["reasoning_content"] = "pondering"
        fake = Fake([brain_body("route"), body])
        raw = stream_raw(
            client_for(fake),
            {
                "messages": [{"role": "user", "content": "haiku"}],
                "reasoning_effort": "high",
                "stream": True,
            },
        )
        text = "".join(
            json.loads(ln[6:])["choices"][0]["delta"].get("content", "")
            for ln in raw.splitlines()
            if ln.startswith("data: ") and ln[6:] != "[DONE]" and json.loads(ln[6:]).get("choices")
        )
        assert "<think>pondering</think>" in text
        assert text.endswith("haiku here")


class TestCallerTools:
    def test_caller_function_is_handed_back(self, client_for):
        fake = Fake([brain_body(tool_calls=[tool_call("get_horoscope", '{"sign":"Taurus"}')])])
        out = (
            client_for(fake)
            .post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "horoscope for Taurus"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_horoscope",
                                "description": "get a horoscope",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"sign": {"type": "string"}},
                                    "required": ["sign"],
                                },
                            },
                        }
                    ],
                },
            )
            .json()
        )
        message = out["choices"][0]["message"]
        assert out["choices"][0]["finish_reason"] == "tool_calls"
        assert message["tool_calls"][0]["function"]["name"] == "get_horoscope"

    def test_caller_tools_reach_the_brain(self, client_for):
        fake = Fake([brain_body("no thanks")])
        client_for(fake).post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "get_horoscope", "parameters": {"type": "object"}},
                    }
                ],
            },
        )
        names = [t["function"]["name"] for t in fake.tool_payloads[0]["tools"]]
        # No file in the request, so only the caller's own function is on offer.
        assert names == ["get_horoscope"]


class TestContextBudget:
    def test_tool_results_are_trimmed_to_budget(self, monkeypatch):
        from interfaze_lite.app import _trim_tool_messages

        messages = [
            {"role": "system", "content": "s"},
            {"role": "tool", "content": "A" * 100_000},
            {"role": "tool", "content": "B" * 100_000},
        ]
        _trim_tool_messages(messages, 120_000)
        total = sum(len(m["content"]) for m in messages if m["role"] == "tool")
        assert total <= 120_000 + 32
        assert messages[2]["content"] == "B" * 100_000


class TestEffortMapping:
    def test_high_maps_to_a_level_the_template_accepts(self):
        """`high` is not a level the template takes; `brain._EFFORT` says why it maps to medium."""
        from interfaze_lite.brain import thinking_kwargs

        assert thinking_kwargs(True, "high") == {"enable_thinking": True, "reasoning_effort": "medium"}

    def test_low_and_medium_pass_through(self):
        from interfaze_lite.brain import thinking_kwargs

        assert thinking_kwargs(True, "low")["reasoning_effort"] == "low"
        assert thinking_kwargs(True, "medium")["reasoning_effort"] == "medium"

    def test_unknown_effort_never_reaches_the_template(self):
        from interfaze_lite.brain import thinking_kwargs

        assert thinking_kwargs(True, "banana")["reasoning_effort"] == "medium"

    def test_thinking_off_sends_no_effort(self):
        from interfaze_lite.brain import thinking_kwargs

        assert thinking_kwargs(False, "high") == {"enable_thinking": False}

    def test_effort_reaches_the_brain(self, client_for):
        fake = Fake([brain_body("route"), brain_body("42")])
        client_for(fake).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "high"},
        )
        assert fake.tool_payloads[-1]["chat_template_kwargs"] == {
            "enable_thinking": True,
            "reasoning_effort": "medium",
        }


class TestStructuredTokenBudget:
    def test_structured_gets_the_larger_budget(self, client_for):
        fake = Fake([brain_body(), brain_body('{"a":1}')])
        client_for(fake).post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "extract"}],
                "response_format": json_schema_format(
                    "x", {"type": "object", "properties": {"a": {"type": "integer"}}}
                ),
            },
        )
        schema_call = fake.last_structured_payload()
        assert schema_call["max_tokens"] == settings.max_structured_tokens
        assert settings.max_structured_tokens > settings.max_new_tokens


class TestRepeatedToolFailure:
    def test_identical_failing_call_is_not_retried_forever(self, client_for):
        from interfaze_lite import tools as tk

        calls = {"n": 0}

        async def boom(args, ctx):
            calls["n"] += 1
            raise RuntimeError("CUDA out of memory")

        original = tk.REGISTRY["ocr"].execute
        tk.REGISTRY["ocr"].execute = boom
        try:
            fake = Fake(
                [
                    brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"}, f"c{i}")])
                    for i in range(6)
                ]
                + [brain_body("unavailable")]
            )
            post_image(client_for(fake), "read")
        finally:
            tk.REGISTRY["ocr"].execute = original

        assert calls["n"] == 2, f"executed {calls['n']} times; expected 2"


class TestDiarizeLocalCopy:
    def test_http_url_is_downloaded_to_a_path(self, tmp_path, monkeypatch):
        import io as _io

        from interfaze_lite.services import diarize as dz

        seen = {}

        def fake_urlopen(req, timeout=0):
            seen["ua"] = req.get_header("User-agent")
            return _io.BytesIO(b"RIFFDATA")

        monkeypatch.setattr(dz.urllib.request, "urlopen", fake_urlopen)
        with dz._local_copy("https://example.com/audio/clip.mp3") as path:
            assert not path.startswith("http")
            assert path.endswith(".mp3")
            assert open(path, "rb").read() == b"RIFFDATA"
        assert "Python-urllib" not in (seen["ua"] or "")
        assert "Mozilla" in seen["ua"]
        assert not __import__("os").path.exists(path)

    def test_local_path_is_passed_through_untouched(self):
        from interfaze_lite.services import diarize as dz

        with dz._local_copy("/tmp/already-local.wav") as path:
            assert path == "/tmp/already-local.wav"


class TestDiarizeBatches:
    """A 95-minute recording ran out of memory at pyannote's default batch of 32."""

    def _torch(self, monkeypatch):
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(
            OutOfMemoryError=type("OutOfMemoryError", (RuntimeError,), {}),
            is_available=lambda: False,
            empty_cache=lambda: None,
        )
        monkeypatch.setitem(__import__("sys").modules, "torch", torch)

    class Pipe:
        def __init__(self, fits):
            self.segmentation_batch_size = self.embedding_batch_size = 32
            self.fits, self.tried = fits, []

        def __call__(self, path, **hints):
            self.tried.append(self.segmentation_batch_size)
            if self.segmentation_batch_size > self.fits:
                raise MemoryError("batch_size ( 32) is probably too large.")
            return "turns"

    def test_halved_until_it_fits_then_put_back(self, monkeypatch):
        from interfaze_lite.services import diarize as dz

        self._torch(monkeypatch)
        pipe = self.Pipe(fits=8)
        assert dz._apply(pipe, "/tmp/a.wav", {}) == "turns"
        assert pipe.tried == [32, 16, 8]
        assert pipe.segmentation_batch_size == pipe.embedding_batch_size == 32

    def test_one_that_never_fits_still_fails(self, monkeypatch):
        from interfaze_lite.services import diarize as dz

        self._torch(monkeypatch)
        pipe = self.Pipe(fits=0)
        with pytest.raises(MemoryError):
            dz._apply(pipe, "/tmp/a.wav", {})
        assert pipe.tried == [32, 16, 8, 4, 2, 1]
        assert pipe.segmentation_batch_size == 32


class TestLayoutView:
    def test_plain_extraction_gets_no_layout(self, client_for):
        fake = Fake(
            [
                brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"})]),
                brain_body("done"),
            ]
        )
        post_image(client_for(fake), "read the text")
        tool_msg = [m for m in fake.tool_payloads[-1]["messages"] if m.get("role") == "tool"][0]
        assert "layout" not in json.loads(tool_msg["content"])

    def test_model_sees_block_geometry(self, client_for):
        fake = Fake(
            [
                brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0", "return_bounds": True})]),
                brain_body("done"),
            ]
        )
        client = client_for(fake)
        post_image(client, "layout")
        tool_msg = [m for m in fake.tool_payloads[-1]["messages"] if m.get("role") == "tool"][0]
        payload = json.loads(tool_msg["content"])
        assert payload["lines"][0]["bbox"] == [10, 10, 300, 40]
        assert payload["lines"][0]["page"] == 1
        assert payload["lines"][0]["text"] == "Invoice total 42.00"

    def test_layout_is_capped(self):
        from interfaze_lite.tools.ocr import _layout_view

        line = {"text": "x", "bounds": {"top_left": {"x": 0, "y": 0}, "bottom_right": {"x": 1, "y": 1}}}
        out = _layout_view([{"page": 1, "lines": [line] * 900}], [], 400)
        assert len(out["lines"]) == 400
        assert "900" in out["lines_truncated"]

    def test_no_geometry_means_no_layout_key(self):
        from interfaze_lite.tools.ocr import _layout_view

        assert _layout_view([{"page": 1, "lines": [{"text": "x"}]}], [], 400) == {}

    def test_typed_blocks_come_with_lines_while_they_fit(self):
        """Lines alone got a box per line of guessed type; blocks alone buried a receipt's total in one."""
        from interfaze_lite.tools.ocr import _layout_view

        def bounds(x1, y1, x2, y2):
            return {"top_left": {"x": x1, "y": y1}, "bottom_right": {"x": x2, "y": y2}}

        lines = [{"text": f"line {i}", "bounds": bounds(0, i * 20, 100, i * 20 + 15)} for i in range(5)]
        sections = [{"page": 1, "lines": lines}, {"page": 2, "lines": lines[:2]}]
        layout = [
            {"page": 1, "type": "doc_title", "text": "Title", "bounds": bounds(0, 0, 100, 15)},
            {"page": 1, "type": "image", "text": "", "bounds": bounds(0, 30, 100, 90)},
            {"page": 1, "type": "text", "text": "word " * 200, "bounds": bounds(0, 20, 100, 95)},
        ]
        out = _layout_view(sections, layout, 400)
        assert [b["type"] for b in out["layout"]] == ["title", "image", "paragraph"]
        assert out["layout"][0]["bbox"] == [0, 0, 100, 15]
        assert len(out["layout"][2]["text"]) < 260  # extracted_text carries the full text
        assert [(ln["page"], ln["text"]) for ln in out["lines"]][-2:] == [(2, "line 0"), (2, "line 1")]

        crowded = _layout_view(sections, layout, 6)
        assert "lines" not in crowded and "7 lines" in crowded["lines_omitted"]


class TestGroundingLabels:
    def _prompt_for(self, domain):
        seen = {}

        def handler(request):
            seen["text"] = json.loads(request.content)["messages"][0]["content"][1]["text"]
            return httpx.Response(200, json=brain_body("[]"))

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        brain = BrainClient(http, settings)
        asyncio.run(brain.ground("https://x/y.png", ["a field"], domain=domain))
        return seen["text"]

    def test_ui_asks_interfazes_grounding_question(self):
        """Word for word what interfaze's grounding service asks this same model."""
        from interfaze_lite import grounding

        assert self._prompt_for("ui") == grounding.UI_INSTRUCTION.format(phrase="a field")
        assert "visible text" not in self._prompt_for("ui")

    def test_object_domain_keeps_category_wording(self):
        text = self._prompt_for("object")
        assert "visible text" not in text
        assert "kind of object" in text


class TestSchemaGeometryDetection:
    def test_bbox_schema_detected(self):
        from interfaze_lite.app import _schema_wants_geometry

        assert _schema_wants_geometry(
            {
                "schema": {
                    "type": "object",
                    "properties": {
                        "words": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"text": {"type": "string"}, "top_left_x": {"type": "number"}},
                            },
                        }
                    },
                }
            }
        )

    def test_plain_schema_not_detected(self):
        from interfaze_lite.app import _schema_wants_geometry

        assert not _schema_wants_geometry(
            {
                "schema": {
                    "type": "object",
                    "properties": {"total_cost": {"type": "string"}, "tax": {"type": "string"}},
                }
            }
        )

    def test_no_schema_is_false(self):
        from interfaze_lite.app import _schema_wants_geometry

        assert not _schema_wants_geometry(None)

    def test_geometry_schema_forces_layout(self, client_for):
        fake = Fake(
            [
                brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"})]),
                brain_body('{"ok":1}'),
            ]
        )
        client_for(fake).post(
            "/v1/chat/completions",
            json={
                "messages": [user_image_message("layout", "https://x/y.png")],
                "response_format": json_schema_format(
                    "layout", {"type": "object", "properties": {"bbox": {"type": "array"}}}
                ),
            },
        )
        tool_msg = fake.tool_messages()[0]
        assert "lines" in json.loads(tool_msg["content"])


class TestStructuredDegeneracy:
    def test_structured_output_carries_no_token_penalty(self, client_for):
        # JSON is made of repeats; a penalty on them derailed receipts under load.
        fake = Fake([brain_body(), brain_body('{"a":1}')])
        client_for(fake).post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "extract"}],
                "response_format": json_schema_format(
                    "x", {"type": "object", "properties": {"a": {"type": "integer"}}}
                ),
            },
        )
        call = fake.last_structured_payload()
        assert "frequency_penalty" not in call and "repetition_penalty" not in call

    def test_a_string_field_holding_a_number_is_written_as_one_and_returned_as_text(self, client_for):
        """Made to quote the forecast's 229913, the model wrote "},{" in the string every run. It
        may write the number; the caller still gets its schema's string, digit for digit."""
        schema = {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string"},
                            "value": {"type": "string"},
                            "price": {"type": "string"},
                            "unit": {"type": "string", "enum": ["mins", "hours"]},
                            "count": {"type": "integer"},
                        },
                    },
                }
            },
        }
        fake = Fake([brain_body()])
        fake.structured_reply = (
            '{"rows": [{"date": "2009-01-01", "value": 229913, "price": 1.50, "unit": "mins", "count": 3}]}'
        )
        out = (
            client_for(fake)
            .post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "x"}],
                    "response_format": json_schema_format("x", schema),
                },
            )
            .json()
        )
        item = fake.last_structured_payload()["response_format"]["json_schema"]["schema"]["properties"][
            "rows"
        ]["items"]
        assert item["properties"]["value"] == {"anyOf": [{"type": "string"}, {"type": "number"}]}
        assert item["properties"]["unit"] == {"type": "string", "enum": ["mins", "hours"]}
        assert item["properties"]["count"] == {"type": "integer"}
        [row] = json.loads(out["choices"][0]["message"]["content"])["rows"]
        assert row == {"date": "2009-01-01", "value": "229913", "price": "1.50", "unit": "mins", "count": 3}

    def test_a_nullable_string_takes_a_number_and_a_number_field_keeps_it(self):
        """zod's .nullable() is type ["string", "null"]: still free text. A field that already
        accepts a number gets the number."""
        from interfaze_lite.app import _as_schema_strings, _numbers_allowed

        schema = {
            "type": "object",
            "properties": {
                "maybe": {"type": ["string", "null"]},
                "either": {"anyOf": [{"type": "string"}, {"type": "number"}]},
                "both": {"type": ["string", "number"]},
            },
        }
        widened = _numbers_allowed(schema)["properties"]
        assert widened["maybe"] == {"anyOf": [{"type": ["string", "null"]}, {"type": "number"}]}
        assert widened["both"] == {"type": ["string", "number"]}
        out = json.loads(_as_schema_strings('{"maybe": 42, "either": 7, "both": 1.5}', schema))
        assert out == {"maybe": "42", "either": 7, "both": 1.5}

    def test_a_property_named_type_is_a_property(self):
        """A document schema's element has a field called "type"; read as JSON Schema's own
        keyword, it took the request down with a 500."""
        from interfaze_lite.app import _as_schema_strings, _numbers_allowed

        element = {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["heading", "table"]},
                "content": {"type": "string"},
                "top_left_x": {"type": "number"},
            },
        }
        props = _numbers_allowed({"type": "array", "items": element})["items"]["properties"]
        assert props["type"] == {"type": "string", "enum": ["heading", "table"]}
        assert props["content"] == {"anyOf": [{"type": "string"}, {"type": "number"}]}
        out = json.loads(
            _as_schema_strings(
                '[{"type": "table", "content": 12, "top_left_x": 3}]', {"type": "array", "items": element}
            )
        )
        assert out == [{"type": "table", "content": "12", "top_left_x": 3}]


class TestToolTrim:
    def test_line_boxes_go_before_any_page_is_cut(self):
        """Ten pages read one at a time, each with its blocks and ~170 lines, ran past the
        budget; cut mid-JSON, the first pages' blocks were lost and filled with zeros."""
        from interfaze_lite.app import _trim_tool_messages

        def page(n):
            return json.dumps(
                {
                    "extracted_text": f"page {n}",
                    "layout_note": "x",
                    "layout": [{"page": n, "type": "paragraph", "text": "t", "bbox": [1, 2, 3, 4]}],
                    "lines": [{"page": n, "text": "line " * 20, "bbox": [1, 2, 3, 4]}] * 50,
                }
            )

        messages = [{"role": "tool", "content": page(n)} for n in range(1, 4)]
        _trim_tool_messages(messages, sum(len(m["content"]) for m in messages) // 2)
        pages = [json.loads(m["content"]) for m in messages]
        assert [p["layout"][0]["page"] for p in pages] == [1, 2, 3]
        assert ["lines" in p for p in pages] == [False, False, True]  # the oldest give theirs up first


class TestNoToolNudge:
    def test_answering_without_a_tool_gets_nudged(self, client_for):
        fake = Fake(
            [
                brain_body("The store is at 123 Main St."),
                brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"})]),
                brain_body("The store is in Greenwood."),
                brain_body("The store is in Greenwood."),
            ]
        )
        out = post_image(client_for(fake), "Where is this store located?").json()
        assert [p["name"] for p in out["precontext"]] == ["ocr"]
        nudges = [
            m
            for p in fake.tool_payloads
            for m in p["messages"]
            if m.get("role") == "user" and "without using a tool" in str(m.get("content"))
        ]
        assert nudges, "expected a nudge turn"

    def test_nudge_happens_at_most_once(self, client_for):
        fake = Fake([brain_body("no tool"), brain_body("still no tool")])
        out = post_image(client_for(fake), "hi").json()
        assert out["choices"][0]["message"]["content"] == "still no tool"

    def test_no_file_means_no_nudge(self, client_for):
        fake = Fake([brain_body("Paris")])
        out = (
            client_for(fake)
            .post(
                "/v1/chat/completions", json={"messages": [{"role": "user", "content": "capital of France?"}]}
            )
            .json()
        )
        assert out["choices"][0]["message"]["content"] == "Paris"
        assert len(fake.tool_payloads) == 1


class TestParallelTools:
    def test_multiple_tool_calls_run_concurrently(self, client_for):
        from interfaze_lite import tools as tk

        order = []

        async def slow(args, ctx):
            order.append("start")
            await asyncio.sleep(0.15)
            order.append("end")
            return tk.ToolResult(model_facing={"ok": True})

        original = tk.REGISTRY["ocr"].execute
        tk.REGISTRY["ocr"].execute = slow
        try:
            fake = Fake(
                [
                    brain_body(
                        tool_calls=[
                            tool_call("ocr", {"file_ref_id": "ref-0"}),
                            tool_call("ocr", {"file_ref_id": "ref-0", "return_markdown": True}, "c2"),
                        ]
                    ),
                    brain_body("done"),
                    brain_body("done"),
                ]
            )
            post_image(client_for(fake), "read")
        finally:
            tk.REGISTRY["ocr"].execute = original

        assert order == ["start", "start", "end", "end"], f"ran serially: {order}"


class TestDetectionMasks:
    def _run(self, client_for, args):
        fake = Fake(
            [
                brain_body(tool_calls=[tool_call("object_detection", args)]),
                brain_body("found them"),
                brain_body("found them"),
            ]
        )
        post_image(client_for(fake), "detect")
        return fake

    def test_masks_are_not_requested_by_default(self, client_for):
        fake = self._run(client_for, {"file_ref_id": "ref-0", "prompts": ["dog"]})
        assert fake.segment_payloads, "segment should still run to rescale the boxes"
        assert fake.segment_payloads[-1]["return_masks"] is False

    def test_masks_are_requested_when_asked_for(self, client_for):
        fake = self._run(client_for, {"file_ref_id": "ref-0", "prompts": ["dog"], "return_masks": True})
        assert fake.segment_payloads[-1]["return_masks"] is True


class TestGroundingImageBudget:
    def _uri(self, w, h):
        import base64
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (w, h), (120, 120, 120)).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def _pixels(self, uri):
        import base64
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
        return img.width * img.height

    def test_oversized_image_is_shrunk_to_budget(self):
        brain = BrainClient(None, settings)
        out, size = asyncio.run(brain._bounded_image(self._uri(2400, 1600), settings.ground_max_pixels))
        assert self._pixels(out) <= settings.ground_max_pixels

    def test_small_image_is_passed_through_untouched(self):
        brain = BrainClient(None, settings)
        uri = self._uri(640, 480)
        out, size = asyncio.run(brain._bounded_image(uri, settings.ground_max_pixels))
        assert out is uri

    def test_resize_never_blocks_the_event_loop(self):
        """A blocking fetch in an async def stalls every other in-flight request (modal `max_inputs=6`)."""
        import inspect

        source = inspect.getsource(BrainClient._bounded_image) + inspect.getsource(BrainClient._image_bytes)
        assert inspect.iscoroutinefunction(BrainClient._bounded_image)
        assert "urlopen(" not in source, "synchronous fetch is back"
        assert "await self.http.get" in source, "should use the shared async client"
        assert "to_thread" in source, "CPU-bound resize should be off the loop"


class TestCoordinateContract:
    """Both detectors must report boxes in the caller's pixel frame, not the grid."""

    def _boxes(self, client_for, tool, args):
        fake = Fake(
            [
                brain_body(tool_calls=[tool_call(tool, args)]),
                brain_body("done"),
                brain_body("done"),
            ]
        )
        out = post_image(client_for(fake), "find it").json()
        result = out["precontext"][0]["result"]
        return result.get("detected_objects") or result.get("gui_elements") or []

    def test_gui_detection_returns_pixels(self, client_for):
        # The Fake grounder's [100,100,300,400] grid box on the 800x600 image is 80,60 -> 240,240.
        boxes = self._boxes(
            client_for, "gui_detection", {"file_ref_id": "ref-0", "prompts": ["login button"]}
        )
        assert boxes, "expected an element"
        tl = boxes[0]["bounds"]["top_left"]
        assert tl["x"] == 80 and tl["y"] == 60, boxes[0]["bounds"]

    def test_object_detection_returns_pixels(self, client_for):
        boxes = self._boxes(client_for, "object_detection", {"file_ref_id": "ref-0", "prompts": ["dog"]})
        assert boxes
        tl = boxes[0]["bounds"]["top_left"]
        assert tl["x"] == 80 and tl["y"] == 60, boxes[0]["bounds"]


class TestTempFileCleanup:
    """Uploads are deleted however the request ends; `stream:true` answered with JSON once leaked them."""

    def _leaked(self, client_for, payload, turns):
        import interfaze_lite.filerefs as fr

        created: list = []
        original = fr.FileRefs.add_data_uri

        def spy(self, uri, filename=""):
            ref = original(self, uri, filename)
            created.append(self._temp_paths[-1])
            return ref

        fr.FileRefs.add_data_uri = spy
        try:
            client_for(Fake(turns)).post("/v1/chat/completions", json=payload)
        finally:
            fr.FileRefs.add_data_uri = original
        assert created, "expected a materialised upload"
        return [p for p in created if p.exists()]

    def _payload(self, **extra):
        tiny = (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )
        return {"messages": [user_image_message("what is this", tiny)], **extra}

    def test_stream_with_structured_output_cleans_up(self, client_for):
        payload = self._payload(
            stream=True,
            response_format=json_schema_format(
                "x", {"type": "object", "properties": {"a": {"type": "integer"}}}
            ),
        )
        assert self._leaked(client_for, payload, [brain_body(), brain_body('{"a":1}')]) == []

    def test_non_stream_cleans_up(self, client_for):
        assert self._leaked(client_for, self._payload(), [brain_body("hi")]) == []


class TestDetectionShapesMatchInterfaze:
    """Detections come back key for key as interfaze returns them."""

    def _result(self, client_for, tool, segment_fails=False):
        fake = Fake(
            [
                brain_body(tool_calls=[tool_call(tool, {"file_ref_id": "ref-0", "prompts": ["thing"]})]),
                brain_body("done"),
                brain_body("done"),
            ]
        )
        fake.segment_fails = segment_fails
        out = post_image(client_for(fake), "find it").json()
        res = out["precontext"][0]["result"]
        items = res.get("detected_objects") or res.get("gui_elements") or []
        return items[0] if items else None

    def test_gui_elements_are_type_and_bounds(self, client_for):
        # helpers/detection/detection.ts: {type: o.label, bounds: o.bounds}, where the
        # grounding service labels each box with the phrase that found it.
        item = self._result(client_for, "gui_detection")
        assert set(item) == {"type", "bounds"}, item
        assert item["type"] == "thing"

    def test_the_brain_sees_what_each_gui_box_is(self, client_for):
        """Unnamed boxes left the brain pairing "Health AI" with whichever box came next,
        or grounding again itself on its 0-1000 grid. The caller's shape is unchanged."""
        fake = Fake(
            [
                brain_body(
                    tool_calls=[tool_call("gui_detection", {"file_ref_id": "ref-0", "all_elements": True})]
                ),
                brain_body("done"),
            ]
        )
        out = post_image(client_for(fake), "buttons?").json()
        [seen] = [json.loads(m["content"]) for m in fake.tool_messages()]
        assert [e["label"] for e in seen["gui_elements"]] == ["dog"]
        assert set(out["precontext"][0]["result"]["gui_elements"][0]) == {"type", "bounds"}

    def test_the_brain_sees_the_text_inside_each_gui_box(self, client_for):
        """Named by eye, "Add to cart" landed on the Google logo and "Buy Now" on the star
        rating. The words read inside each box name it; the caller's shape is unchanged."""
        line = {
            "text": "Add to cart",
            "bounds": {"top_left": {"x": 100, "y": 100}, "bottom_right": {"x": 200, "y": 130}},
        }
        far = {
            "text": "Footer",
            "bounds": {"top_left": {"x": 500, "y": 500}, "bottom_right": {"x": 600, "y": 530}},
        }
        ocr = {**OCR_FIXTURE, "sections": [{"text": "Add to cart Footer", "lines": [line, far]}]}
        fake = Fake(
            [
                brain_body(
                    tool_calls=[tool_call("gui_detection", {"file_ref_id": "ref-0", "all_elements": True})]
                ),
                brain_body("done"),
            ],
            ocr=ocr,
        )
        out = post_image(client_for(fake), "buttons?").json()
        [seen] = [json.loads(m["content"]) for m in fake.tool_messages()]
        assert [e.get("text") for e in seen["gui_elements"]] == ["Add to cart"]
        assert set(out["precontext"][0]["result"]["gui_elements"][0]) == {"type", "bounds"}

    def test_a_gui_box_the_size_of_a_text_line_lands_on_it_and_the_lines_go_along(self, client_for):
        """Amazon's nav links came back half a word to the right ("Health AI" over "lth AI
        Sa"). A box about one line's size that mostly covers it is that line; and the page's
        lines go to the brain for any text it is asked to box."""
        link = {
            "text": "Health AI",
            "bounds": {"top_left": {"x": 110, "y": 70}, "bottom_right": {"x": 250, "y": 250}},
        }
        ocr = {**OCR_FIXTURE, "sections": [{"text": "Health AI", "lines": [link]}]}
        fake = Fake(
            [
                brain_body(
                    tool_calls=[tool_call("gui_detection", {"file_ref_id": "ref-0", "all_elements": True})]
                ),
                brain_body("done"),
            ],
            ocr=ocr,
        )
        out = post_image(client_for(fake), "links?").json()
        [element] = out["precontext"][0]["result"]["gui_elements"]
        assert (element["bounds"]["top_left"], element["bounds"]["bottom_right"]) == (
            {"x": 110, "y": 70},
            {"x": 250, "y": 250},
        )
        [seen] = [json.loads(m["content"]) for m in fake.tool_messages()]
        assert seen["page_text"] == [{"text": "Health AI", "bbox": [110, 70, 250, 250]}]

    def test_a_gui_label_that_only_repeats_the_query_is_not_handed_on(self):
        """Grounded for "buttons", every box came back labelled "button"; told what each box
        was, the brain named all 105 of them "button"."""
        from interfaze_lite.tools.detection import _naming_label

        assert _naming_label("Add to cart", "buttons") == "Add to cart"
        assert _naming_label("Rufus", "interactive element (buttons, links, inputs, icons)") == "Rufus"
        for generic, phrase in (
            ("button", "buttons"),
            ("Buttons", "buttons"),
            ("text input", "text input"),
            ("icon", "interactive element (buttons, links, inputs, icons)"),
            ("link", "x"),
            ("", "x"),
        ):
            assert _naming_label(generic, phrase) is None, generic

    def test_each_gui_phrase_is_grounded_on_its_own_and_types_its_boxes(self, client_for):
        fake = Fake(
            [
                brain_body(
                    tool_calls=[
                        tool_call(
                            "gui_detection",
                            {"file_ref_id": "ref-0", "prompts": ["login button", "search box"]},
                        )
                    ]
                ),
                brain_body("done"),
            ]
        )
        out = post_image(client_for(fake), "find them").json()
        types = [e["type"] for e in out["precontext"][0]["result"]["gui_elements"]]
        assert sorted(types) == ["login button", "search box"]

    def test_a_gui_request_is_grounded_on_what_it_names(self):
        """ "Detect all buttons" is grounded as "buttons", as interfaze's caller sends it."""
        from interfaze_lite.tools.detection import _target

        assert _target("Detect all buttons") == "buttons"
        assert _target("Show me all of the icons") == "icons"
        # An instruction that names one element is the target itself, head noun and all.
        assert _target("click the search button") == "click the search button"
        assert _target("check the letters") == "check the letters"

    def test_boxes_are_pixels_even_when_segmentation_fails(self, client_for):
        # Under load /segment failed and the grid came back as though it were pixels.
        item = self._result(client_for, "object_detection", segment_fails=True)
        assert item["bounds"]["top_left"] == {"x": 80, "y": 60}, item

    def test_gui_detection_needs_no_segmentation_call(self, client_for):
        fake = Fake(
            [
                brain_body(
                    tool_calls=[tool_call("gui_detection", {"file_ref_id": "ref-0", "prompts": ["button"]})]
                ),
                brain_body("done"),
            ]
        )
        post_image(client_for(fake), "find the button")
        assert "/segment" not in fake.calls

    def test_objects_carry_no_extra_keys(self, client_for):
        for fails in (False, True):
            item = self._result(client_for, "object_detection", segment_fails=fails)
            assert item and set(item) <= {"label", "bounds", "polygon", "mask"}, item


class TestOcrShapeMatchesInterfaze:
    def test_image_result_is_interfaze_keys_only(self, client_for):
        fake = Fake(
            [
                brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"})]),
                brain_body("done"),
                brain_body("done"),
            ]
        )
        fake.ocr = {
            **OCR_FIXTURE,
            "total_pages": 1,
            "layout": [
                {
                    "type": "title",
                    "score": 0.99,
                    "text": "Invoice",
                    "bounds": {"top_left": {"x": 1, "y": 2}, "bottom_right": {"x": 3, "y": 4}},
                }
            ],
            "layout_status": "ok",
        }
        out = post_image(client_for(fake), "read").json()
        result = out["precontext"][0]["result"]
        # Interfaze sends no layout blocks and no total_pages for an image.
        assert set(result) <= {
            "extracted_text",
            "sections",
            "width",
            "height",
            "should_not_return_to_user",
        }, result


class TestRunTaskWithSchema:
    def test_non_empty_schema_with_a_task_is_refused_as_interfaze_does(self, client_for):
        resp = client_for(Fake([brain_body("x")])).post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "<task>ocr</task>"},
                    {"role": "user", "content": "read"},
                ],
                "response_format": json_schema_format(
                    "r", {"type": "object", "properties": {"extracted_text": {"type": "string"}}}
                ),
            },
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["message"] == "Non-empty schema is not allowed to be used with run tasks"

    def test_the_sdks_empty_task_schema_is_still_accepted(self, client_for):
        fake = Fake([brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"})])])
        resp = client_for(fake).post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "<task>ocr</task>"},
                    user_image_message("read", "https://x/y.png"),
                ],
                "response_format": json_schema_format("empty_schema", {}),
            },
        )
        assert resp.status_code == 200


def test_a_task_result_keeps_its_characters_unescaped(client_for):
    """As beta's JSON.stringify writes it: a client showing the result reads "你好", not "\\u4f60"."""
    text = "你好 café"
    ocr = {**OCR_FIXTURE, "text": text}
    fake = Fake(
        [brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"})])],
        ocr=ocr,
    )
    body = (
        client_for(fake)
        .post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "<task>ocr</task>"},
                    user_image_message("read", "https://x/y.png"),
                ]
            },
        )
        .json()
    )
    content = body["choices"][0]["message"]["content"]
    assert text in content
    assert json.loads(content)["result"]["extracted_text"] == text


def test_streaming_omits_precontext_unless_asked(client_for):
    """`<precontext>` rides inside content, so unasked it would splice tool JSON into every caller's prose."""
    fake = Fake(
        [
            brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"})]),
            brain_body(),
            brain_body("The total is 42."),
        ]
    )
    client = client_for(fake)
    raw = stream_raw(
        client,
        {
            "stream": True,
            "messages": [user_image_only("https://x.test/a.png")],
        },
    )

    text = streamed_text(raw)
    assert "<precontext>" not in text
    assert text == "The total is 42."


class TestCallerChosenTransport:
    """A `stream: true` request gets SSE even for a caller tool call or structured output, once JSON."""

    def test_caller_tool_call_streams(self, client_for):
        weather = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
        fake = Fake([brain_body(tool_calls=[tool_call("get_weather", '{"city":"Paris"}')])])
        client = client_for(fake)
        raw = stream_raw(
            client,
            {
                "stream": True,
                "tools": [weather],
                "messages": [{"role": "user", "content": "weather in Paris?"}],
            },
        )

        assert raw.rstrip().endswith("data: [DONE]")
        chunks = sse_chunks(raw)
        names = [
            tc["function"]["name"]
            for c in chunks
            for tc in (c["choices"][0]["delta"].get("tool_calls") or [])
        ]
        assert names == ["get_weather"]
        assert [c["choices"][0]["finish_reason"] for c in chunks][-1] == "tool_calls"
        # Content must stay empty, or an agent runtime treats the turn as the answer.
        assert "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks) == ""

    def test_structured_output_streams(self, client_for):
        fake = Fake([brain_body('{"a":1}')])
        client = client_for(fake)
        raw = stream_raw(
            client,
            {
                "stream": True,
                "response_format": {
                    "json_schema": {
                        "name": "s",
                        "schema": {"type": "object", "properties": {"text": {"type": "string"}}},
                    }
                },
                "messages": [{"role": "user", "content": "say hello"}],
            },
        )

        assert raw.rstrip().endswith("data: [DONE]")
        chunks = sse_chunks(raw)
        assert chunks, "structured output produced no chunks"
        body = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
        # The double's guided-decoding branch answers with its own fixture; what this
        # asserts is that a schema request still arrives as chunks, not as a JSON body.
        assert json.loads(body) == {"a": 1}


class TestSamplingIsHonoured:
    """The caller's sampling parameters must reach the model, and be checked first."""

    @pytest.mark.parametrize(
        "payload,reason",
        [
            ({"temperature": 5.0}, "above the range"),
            ({"temperature": -1}, "below the range"),
            ({"top_p": 2.0}, "above the range"),
            ({"max_tokens": 1_000_000}, "beyond what the server can generate"),
            ({"max_tokens": 0}, "not a usable budget"),
        ],
    )
    def test_out_of_range_is_rejected(self, client_for, payload, reason):
        resp = client_for(Fake([brain_body("hi")])).post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], **payload}
        )
        assert resp.status_code == 400, f"{payload} is {reason}"

    def test_the_advertised_ceiling_is_accepted(self, client_for):
        """32000 is the SDKs' default max_tokens; `Settings.max_output_tokens` says why it is accepted."""
        fake = Fake([brain_body("hi")])
        resp = client_for(fake).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32000},
        )
        assert resp.status_code == 200
        assert fake.tool_payloads[-1]["max_tokens"] == 32000

    def test_max_tokens_reaches_the_model(self, client_for):
        fake = Fake([brain_body("hi")])
        client_for(fake).post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32}
        )
        assert fake.tool_payloads[-1]["max_tokens"] == 32

    def test_finish_reason_is_relayed(self, client_for):
        body = brain_body("truncated")
        body["choices"][0]["finish_reason"] = "length"
        out = (
            client_for(Fake([body]))
            .post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16},
            )
            .json()
        )
        assert out["choices"][0]["finish_reason"] == "length"


class TestRoutedTaskNaming:
    """Transcription routes as `speech_to_text` and reports as `stt` in precontext; shipped docs use both."""

    def test_alias_routes_and_echoes_the_requested_name(self, client_for):
        fake = Fake(
            [
                brain_body(tool_calls=[tool_call("stt", {"file_ref_id": "ref-0"})]),
                brain_body(),
                brain_body("done"),
            ]
        )
        out = (
            client_for(fake)
            .post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": "<task>speech_to_text</task>"},
                        user_audio_message("transcribe"),
                    ],
                },
            )
            .json()
        )
        routed = json.loads(out["choices"][0]["message"]["content"])
        assert routed["name"] == "speech_to_text"
        assert "result" in routed

    def test_precontext_still_calls_it_stt(self, client_for):
        fake = Fake(
            [
                brain_body(tool_calls=[tool_call("stt", {"file_ref_id": "ref-0"})]),
                brain_body(),
                brain_body("done"),
            ]
        )
        out = (
            client_for(fake)
            .post(
                "/v1/chat/completions",
                json={
                    "messages": [user_audio_message("transcribe")],
                },
            )
            .json()
        )
        assert [p["name"] for p in out["precontext"]] == ["stt"]

    def test_routed_tool_success_ends_the_loop(self, client_for):
        # The answer is the tool's own output; another turn is a generation nobody reads.
        fake = Fake(
            [
                brain_body(tool_calls=[tool_call("stt", {"file_ref_id": "ref-0"})]),
                brain_body("restating the transcript"),
            ]
        )
        client_for(fake).post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "<task>speech_to_text</task>"},
                    user_audio_message("transcribe"),
                ],
            },
        )
        assert len(fake.tool_payloads) == 1
        assert len(fake.brain_turns) == 1

    def test_unknown_task_is_rejected(self, client_for):
        resp = client_for(Fake([brain_body("hi")])).post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "<task>web_search</task>"},
                    {"role": "user", "content": "hi"},
                ]
            },
        )
        assert resp.status_code == 400


def test_routed_task_streams_when_the_caller_streams(client_for):
    """A routed task honours `stream`: sent as JSON, every streamed playground run-task came back blank."""
    fake = Fake(
        [
            brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"})]),
            brain_body(),
            brain_body("done"),
        ]
    )
    client = client_for(fake)
    raw = stream_raw(
        client,
        {
            "stream": True,
            "messages": [
                {"role": "system", "content": "<task>ocr</task>"},
                user_image_message("read", "https://x.test/a.png"),
            ],
        },
    )

    assert raw.rstrip().endswith("data: [DONE]")
    body = streamed_text(raw)
    routed = json.loads(body)
    assert routed["name"] == "ocr"
    assert "result" in routed


def test_a_routed_task_answered_without_its_tool_still_streams(client_for):
    """A routed task whose tool never ran falls to the reused answer, which came back as JSON to a stream."""
    fake = Fake([brain_body("Bonjour")])
    raw = stream_raw(
        client_for(fake),
        {
            "stream": True,
            "messages": [
                {"role": "system", "content": "<task>translate</task>"},
                {"role": "user", "content": "Say hello in French."},
            ],
        },
    )

    assert raw.rstrip().endswith("data: [DONE]")
    assert streamed_text(raw) == "Bonjour"


class TestToolTurnAnswerIsReused:
    """A tool-using request generates its answer once (`Settings.reuse_tool_turn_answer`)."""

    @staticmethod
    def _turns():
        return [
            brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"})]),
            brain_body("The total is 42."),  # selection turn answers outright
            brain_body("The total is 42."),  # only reached if it regenerates
        ]

    @staticmethod
    def _image_request(**extra):
        return {
            "messages": [user_image_message("what is the total?", "https://x.test/a.png")],
            **extra,
        }

    def test_no_second_generation_when_a_tool_ran(self, client_for):
        fake = Fake(self._turns())
        body = client_for(fake).post("/v1/chat/completions", json=self._image_request()).json()
        assert body["choices"][0]["message"]["content"] == "The total is 42."
        assert [p["name"] for p in body["precontext"]] == ["ocr"]
        # Two selection turns and nothing more: a third call is the regeneration.
        assert len(fake.tool_payloads) == 2, "answer was generated twice"

    def test_streaming_reuses_it_too(self, client_for):
        fake = Fake(self._turns())
        client = client_for(fake)
        raw = stream_raw(client, self._image_request(stream=True))
        text = streamed_text(raw)
        assert text == "The total is 42."
        # Two generations, the tool call then the streamed answer. The playground always
        # streams, so this is where every request regenerated.
        assert len(fake.tool_payloads) + len(fake.stream_payloads) == 2
        assert len(fake.brain_turns) == 1

    def test_reasoning_requests_still_generate_afresh(self, client_for):
        """The selection turn runs with thinking off, so its answer cannot stand in."""
        fake = Fake(self._turns())
        client_for(fake).post("/v1/chat/completions", json=self._image_request(reasoning_effort="medium"))
        assert len(fake.tool_payloads) == 3, "reasoning must not reuse the tool turn"


class TestDetectionOutlines:
    """Shape travels as a polygon, small enough for the model; the megabyte-per-object bitmap is opt-in."""

    def _detect(self, client_for, **args):
        fake = Fake(
            [
                brain_body(
                    tool_calls=[
                        tool_call("object_detection", {"file_ref_id": "ref-0", "prompts": ["dog"], **args})
                    ]
                ),
                brain_body("done"),
            ]
        )
        body = post_image(client_for(fake), "find the dog", "https://x.test/a.png").json()
        return body, fake

    def test_the_segment_call_asks_for_outlines(self, client_for):
        _, fake = self._detect(client_for)
        assert fake.segment_payloads, "no segmentation call was made"
        assert fake.segment_payloads[-1]["return_outlines"] is True

    def test_masks_stay_off_by_default(self, client_for):
        _, fake = self._detect(client_for)
        assert fake.segment_payloads[-1]["return_masks"] is False


class TestTranscriptChunks:
    """Timing comes back as interfaze returns it: segments by default, words on request."""

    def _stt(self, client_for, args):
        fake = Fake(
            [
                brain_body(tool_calls=[tool_call("stt", {"file_ref_id": "ref-0", **args})]),
                brain_body("done"),
            ]
        )
        body = (
            client_for(fake)
            .post(
                "/v1/chat/completions",
                json={
                    "messages": [user_audio_message("transcribe")],
                },
            )
            .json()
        )
        return fake, body["precontext"][0]["result"]

    def test_plain_transcription_still_returns_timed_chunks(self, client_for):
        fake, result = self._stt(client_for, {})
        assert fake.transcribe_payloads[0]["word_timestamps"] is False
        assert result["chunks"] and all("timestamp" in c for c in result["chunks"])

    def test_words_are_asked_for_only_when_wanted(self, client_for):
        fake, _ = self._stt(client_for, {"word_timestamps": True})
        assert fake.transcribe_payloads[0]["word_timestamps"] is True

    def test_speaker_split_asks_for_words_to_join_on(self, client_for):
        fake, _ = self._stt(client_for, {"split_by_speaker": True})
        assert fake.transcribe_payloads[0]["words_for_speakers"] is True
        # Exact word timing only when the caller asked for it: on a long recording it took
        # past what a relayed request may wait.
        assert fake.transcribe_payloads[0]["word_timestamps"] is False

    def test_asked_for_word_timestamps_beside_speakers(self, client_for):
        fake, _ = self._stt(client_for, {"split_by_speaker": True, "word_timestamps": True})
        assert fake.transcribe_payloads[0]["word_timestamps"] is True


class TestCallerControls:
    WEATHER = {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}

    def test_tool_choice_none_withholds_the_callers_functions(self, client_for):
        fake = Fake([brain_body("It is sunny.")])
        body = (
            client_for(fake)
            .post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Weather in Paris? Use get_weather."}],
                    "tools": [self.WEATHER],
                    "tool_choice": "none",
                },
            )
            .json()
        )
        offered = {t["function"]["name"] for t in fake.tool_payloads[0].get("tools", [])}
        assert "get_weather" not in offered
        assert body["choices"][0]["message"]["content"] == "It is sunny."

    def test_a_forced_choice_binds_the_first_turn(self, client_for):
        forced = {"type": "function", "function": {"name": "get_weather"}}
        fake = Fake([brain_body(tool_calls=[tool_call("get_weather", "{}")])])
        client_for(fake).post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [self.WEATHER],
                "tool_choice": forced,
            },
        )
        assert fake.tool_payloads[0]["tool_choice"] == forced

    def test_json_object_is_guided_decoding(self, client_for):
        fake = Fake([brain_body("{}")])
        body = (
            client_for(fake)
            .post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Return a JSON object with answer 42."}],
                    "response_format": {"type": "json_object"},
                },
            )
            .json()
        )
        call = fake.last_structured_payload()
        assert call["response_format"]["json_schema"]["schema"] == {"type": "object"}
        assert json.loads(body["choices"][0]["message"]["content"]) == {"a": 1}

    def test_the_callers_instructions_are_restated_at_the_latest_turn(self, client_for):
        fake = Fake([brain_body("TULIP-7429")])
        client_for(fake).post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "Reply with only the code."},
                    {"role": "user", "content": "Remember TULIP-7429."},
                    {"role": "assistant", "content": "Understood."},
                    {"role": "user", "content": "What is my code?"},
                ]
            },
        )
        msgs = fake.tool_payloads[0]["messages"]
        assert msgs[-1]["content"].startswith("What is my code?")
        assert "Reply with only the code." in msgs[-1]["content"]
        assert "Reply with only the code." not in msgs[1]["content"]


class TestUnreachableFiles:
    def test_a_file_that_cannot_be_fetched_is_reported_as_such(self, client_for):
        fake = Fake(
            [
                brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0"})]),
                brain_body("That image could not be fetched (404)."),
            ]
        )
        fake.ocr = httpx.Response(
            422, json={"detail": {"input_fetch": "could not fetch https://x/y.png: HTTP 404"}}
        )
        body = post_image(client_for(fake), "read").json()
        result = body["precontext"][0]["result"]
        assert "HTTP 404" in result["error"]
        assert "could not be fetched" in result["message"] and "unavailable" not in result["message"]

    def test_fetch_url_names_the_status(self, monkeypatch):
        import httpx as real

        from interfaze_lite.contracts import InputFetchError, fetch_url

        req = real.Request("GET", "https://x/y.png")
        monkeypatch.setattr(real, "get", lambda *_args, **_kwargs: real.Response(404, request=req))
        with pytest.raises(InputFetchError, match="HTTP 404"):
            fetch_url("https://x/y.png", timeout=5)


class TestGuardDirectiveAnywhereInTheSystemTurns:
    """`<guard>` was read from one string system prompt only; anywhere else it reached the
    model unenforced."""

    def test_array_form_system_content_is_enforced(self, client_for):
        fake = Fake([])
        fake.guard_output = "unsafe\nS1"
        body = (
            client_for(fake)
            .post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": [{"type": "text", "text": "<guard>S1, S10</guard>"}]},
                        {"role": "user", "content": "How to kill a human?"},
                    ]
                },
            )
            .json()
        )
        assert body["choices"][0]["message"]["content"] == "unsafe S1"
        assert body["precontext"] == [{"name": "text_guardrail_classifier", "result": ["S1"]}]
        assert fake.tool_payloads == []

    def test_a_second_system_message_is_enforced(self, client_for):
        fake = Fake([])
        fake.guard_output = "unsafe\nS10"
        body = (
            client_for(fake)
            .post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": "Be brief."},
                        {"role": "system", "content": "<guard>S1, S10</guard>"},
                        {"role": "user", "content": "hateful request"},
                    ]
                },
            )
            .json()
        )
        assert body["choices"][0]["message"]["content"] == "unsafe S10"

    def test_the_directive_never_reaches_the_model_and_array_instructions_do(self, client_for):
        fake = Fake([brain_body("Bonjour.")])
        client_for(fake).post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "Reply in French. <guard>S1</guard>"}],
                    },
                    {"role": "user", "content": "Say hello."},
                ]
            },
        )
        system = fake.tool_payloads[0]["messages"][0]["content"]
        assert "Reply in French." in system and "<guard>" not in system
        assert fake.guard_payloads == [{"text": "Say hello."}]


def test_a_zero_data_retention_callers_tool_arguments_stay_out_of_the_logs(client_for, caplog):
    call = tool_call("translate", {"text": "secret plans", "target_language": "fr", "current_language": "en"})
    fake = Fake([brain_body(tool_calls=[call]), brain_body("done")])
    with caplog.at_level("INFO", logger="interfaze.app"):
        client_for(fake).post(
            "/v1/chat/completions",
            headers={"x-interfaze-zdr": "true"},
            json={"messages": [{"role": "user", "content": "Translate 'secret plans' into French."}]},
        )
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "translate(<redacted>)" in logged and "secret plans" not in logged


def test_input_longer_than_the_context_window_is_the_callers_400(client_for):
    too_long = httpx.Response(
        400,
        json={
            "error": {
                "message": (
                    "This model's maximum context length is 131072 tokens. However, you requested 200000 tokens."
                )
            }
        },
    )
    fake = Fake([too_long])
    response = client_for(fake).post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "long " * 10}]}
    )
    assert response.status_code == 400
    assert "maximum context length" in response.json()["error"]["message"]


def test_the_brain_is_told_todays_date_last(client_for):
    """Without it, "a 5-year overview" ended on whatever year the model last saw. Last, so
    the prompt before it stays the same across requests and cached."""
    from datetime import datetime

    fake = Fake([brain_body("hi")])
    client_for(fake).post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    system = fake.tool_payloads[0]["messages"][0]["content"]
    assert system.endswith(f"Today's date: {datetime.now(UTC).date().isoformat()}")


def test_a_schema_is_filled_at_low_temperature_whatever_the_caller_asked(client_for):
    """At the playground's temperature of 1, a paper's section list came back empty on 2
    runs of 11. A lower temperature the caller sets is kept."""
    schema = json_schema_format("a", {"type": "object", "properties": {"a": {"type": "integer"}}})
    for asked, sent in ((1, 0.3), (0.3, 0.3), (0, 0)):
        fake = Fake([brain_body("ok")])
        client_for(fake).post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": asked,
                "response_format": schema,
            },
        )
        filled = [p for p in fake.tool_payloads if p.get("response_format")]
        assert filled and filled[-1]["temperature"] == sent, (asked, filled[-1].get("temperature"))


def test_a_schema_without_coordinates_is_filled_from_text_without_the_page_geometry(client_for):
    """Asked for bounds, ocr's hundreds of boxed blocks and lines sat beside the text, and a
    paper's section list was filled empty. A schema that asks for coordinates keeps them."""
    plain = json_schema_format("a", {"type": "object", "properties": {"total": {"type": "string"}}})
    boxed = json_schema_format(
        "b",
        {
            "type": "object",
            "properties": {"total": {"type": "string"}, "bbox": {"type": "array"}},
        },
    )
    for schema, keeps in ((plain, False), (boxed, True)):
        fake = Fake(
            [
                brain_body(tool_calls=[tool_call("ocr", {"file_ref_id": "ref-0", "return_bounds": True})]),
                brain_body(""),
            ]
        )
        client_for(fake).post(
            "/v1/chat/completions",
            json={
                "response_format": schema,
                "messages": [user_image_message("What is the total?", "https://x.test/r.png")],
            },
        )
        filled = fake.last_structured_payload()
        tool = json.loads(next(m for m in filled["messages"] if m.get("role") == "tool")["content"])
        assert tool["extracted_text"] == "Invoice total 42.00"
        assert ("lines" in tool) is keeps, schema["json_schema"]["name"]


def test_a_structured_answers_tool_turns_run_calm_and_others_keep_their_temperature(client_for):
    """At temperature 1 a structured request's tool turns varied which tools ran with what
    arguments, and the schema filled differently from each; their text is discarded."""
    schema = json_schema_format("a", {"type": "object", "properties": {"a": {"type": "integer"}}})
    for body, expected in (({"response_format": schema}, 0.3), ({}, 1)):
        fake = Fake([brain_body("ok"), brain_body("ok")])
        client_for(fake).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "temperature": 1, **body},
        )
        turns = [p for p in fake.tool_payloads if not p.get("response_format")]
        assert turns and turns[0]["temperature"] == expected, body


def test_boxes_none_of_which_the_detector_gave_are_drawn_again(client_for, monkeypatch):
    """With the screenshot in view the model sometimes boxed the page by eye, on its own
    0-1000 grid: 71 boxes, not one of them a detector box. One more draw, kept if it copies."""
    original = Fake.handler
    drawn = []

    def handler(self, request):
        payload = json.loads(request.content) if request.url.path.endswith("/v1/chat/completions") else {}
        if payload.get("response_format"):
            self.tool_payloads.append(payload)
            if not drawn:
                items = [
                    {
                        "name": f"b{i}",
                        "top_left_x": 25 + 50 * i,
                        "top_left_y": 95,
                        "bottom_right_x": 55 + 50 * i,
                        "bottom_right_y": 115,
                    }
                    for i in range(6)
                ]
            else:
                tool = json.loads(next(m for m in payload["messages"] if m.get("role") == "tool")["content"])
                b = tool["gui_elements"][0]["bounds"]
                items = [
                    {
                        "name": "dog",
                        "top_left_x": b["top_left"]["x"],
                        "top_left_y": b["top_left"]["y"],
                        "bottom_right_x": b["bottom_right"]["x"],
                        "bottom_right_y": b["bottom_right"]["y"],
                    }
                ]
            drawn.append(payload.get("temperature"))
            return httpx.Response(200, json=brain_body(json.dumps({"buttons": items})))
        return original(self, request)

    monkeypatch.setattr(Fake, "handler", handler)
    box = {"type": "number"}
    item = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "top_left_x": box,
            "top_left_y": box,
            "bottom_right_x": box,
            "bottom_right_y": box,
        },
    }
    schema = json_schema_format(
        "a", {"type": "object", "properties": {"buttons": {"type": "array", "items": item}}}
    )
    fake = Fake(
        [
            brain_body(
                tool_calls=[tool_call("gui_detection", {"file_ref_id": "ref-0", "all_elements": True})]
            ),
            brain_body(""),
        ]
    )
    body = (
        client_for(fake)
        .post(
            "/v1/chat/completions",
            json={"messages": [user_image_message("buttons?", "https://x/y.png")], "response_format": schema},
        )
        .json()
    )
    [button] = json.loads(body["choices"][0]["message"]["content"])["buttons"]
    assert button["name"] == "dog"
    assert len(drawn) == 2 and drawn[1] == 0.3


def test_a_structured_reply_that_does_not_parse_is_drawn_again(client_for, monkeypatch):
    """Guided decoding lost its place once in ten on a dense schema: a key with no colon."""
    replies = iter(['{"title" "Interfaze"}', '{"title": "Interfaze"}'])
    original = Fake.handler

    def handler(self, request):
        payload = json.loads(request.content) if request.url.path.endswith("/v1/chat/completions") else {}
        if payload.get("response_format"):
            self.tool_payloads.append(payload)
            return httpx.Response(200, json=brain_body(next(replies)))
        return original(self, request)

    monkeypatch.setattr(Fake, "handler", handler)
    schema = json_schema_format("a", {"type": "object", "properties": {"title": {"type": "string"}}})
    fake = Fake([brain_body("ok")])
    body = (
        client_for(fake)
        .post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "response_format": schema},
        )
        .json()
    )
    assert json.loads(body["choices"][0]["message"]["content"]) == {"title": "Interfaze"}
    assert len([p for p in fake.tool_payloads if p.get("response_format")]) == 2
