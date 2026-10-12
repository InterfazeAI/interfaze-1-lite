"""Wire-format tests: the OpenAI envelope plus interfaze's precontext extension."""

import json

from interfaze_lite.envelope import (
    Completion,
    PrecontextItem,
    ToolCall,
    Usage,
    parse_precontext,
    precontext_delta,
    stream,
    strip_precontext,
)


def make(**kw) -> Completion:
    return Completion(request_id="req-test", created=1700000000, **kw)


def content_deltas(events: list[str]) -> list[str]:
    """Each SSE chunk's content delta ("" where it has none), the final [DONE] left off."""
    return [json.loads(e[len("data: ") :])["choices"][0]["delta"].get("content") or "" for e in events[:-1]]


class TestNonStreaming:
    def test_basic_shape_matches_openai(self):
        body = make(usage=Usage(10, 5)).message("hello")
        assert body["object"] == "chat.completion"
        assert body["model"] == "interfaze-lite"
        assert body["usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            # Present even at zero (Usage.as_dict).
            "completion_tokens_details": {"reasoning_tokens": 0, "cached_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0},
        }
        choice = body["choices"][0]
        assert choice["message"] == {"role": "assistant", "content": "hello"}
        assert choice["logprobs"] is None
        assert choice["finish_reason"] == "stop"

    def test_tool_call_turn_has_null_content(self):
        """Populating content on a tool-call turn breaks OpenAI-compatible agent loops."""
        body = make().tool_calls([ToolCall("call_1", "ocr", '{"file_ref_id":"ref-0"}')])
        msg = body["choices"][0]["message"]
        assert msg["content"] is None
        assert body["choices"][0]["finish_reason"] == "tool_calls"
        assert msg["tool_calls"][0]["type"] == "function"
        assert msg["tool_calls"][0]["function"]["name"] == "ocr"
        assert isinstance(msg["tool_calls"][0]["function"]["arguments"], str)

    def test_precontext_carries_structured_results(self):
        body = make(precontext=[PrecontextItem("ocr", {"sections": [{"text": "hi"}]})]).message("hi")
        assert body["precontext"] == [{"name": "ocr", "result": {"sections": [{"text": "hi"}]}}]

    def test_precontext_key_always_present_even_when_empty(self):
        """A stable key: callers never branch, and typed SDKs see [] rather than missing."""
        body = make().message("hello")
        assert body["precontext"] == []

    def test_interfaze_cache_fields_are_always_present(self):
        """A caller written against interfaze reads these, so lite sends them as false and 0."""
        body = make(precontext=[PrecontextItem("ocr", {})]).message("x")
        assert body["vcache"] is False
        assert body["usage"]["prompt_tokens_details"] == {"cached_tokens": 0}
        assert body["usage"]["completion_tokens_details"]["cached_tokens"] == 0

    def test_optional_fields_appear_only_when_set(self):
        plain = make().message("x")
        assert "reasoning" not in plain and "debug" not in plain
        rich = make(reasoning="thought", debug={"preConfig": {}}).message("x")
        assert rich["reasoning"] == "thought"
        assert rich["debug"] == {"preConfig": {}}


class TestStreaming:
    def test_sequence_order_and_termination(self):
        events = list(stream(make(usage=Usage(3, 4)), ["Hel", "lo"]))
        assert events[-1] == "data: [DONE]\n\n"
        payloads = [json.loads(e[len("data: ") :]) for e in events[:-1]]
        assert payloads[0]["choices"][0]["delta"]["role"] == "assistant"
        # Assert the sequence, not indices: content follows the precontext block.
        contents = [p["choices"][0]["delta"].get("content") or "" for p in payloads]
        assert [c for c in contents if c and "<precontext>" not in c] == ["Hel", "lo"]
        final = payloads[-1]
        assert final["choices"][0]["finish_reason"] == "stop"
        assert final["usage"]["total_tokens"] == 7
        assert all(p["object"] == "chat.completion.chunk" for p in payloads)

    def test_precontext_rides_as_a_sentinel_tagged_content_delta(self):
        """Precontext rides the content stream in sentinel tags, byte-compatible with interfaze."""
        c = make(precontext=[PrecontextItem("ocr", {"sections": []})])
        events = list(stream(c, ["done"]))
        text = "".join(content_deltas(events))
        assert text.startswith("<precontext> ")
        assert " </precontext>" in text
        assert parse_precontext(text) == [{"name": "ocr", "result": {"sections": []}}]
        assert strip_precontext(text) == "done"

    def test_precontext_block_present_even_with_no_tools(self):
        events = list(stream(make(), ["hi"]))
        text = "".join(content_deltas(events))
        assert parse_precontext(text) == []
        assert strip_precontext(text) == "hi"

    def test_precontext_delta_is_emitted_when_empty(self):
        chunk = precontext_delta(make())
        assert "<precontext> [] </precontext>" in chunk["choices"][0]["delta"]["content"]

    def test_empty_content_parts_are_skipped(self):
        events = list(stream(make(), ["", "a", ""]))
        real = [c for c in content_deltas(events) if c and "<precontext>" not in c]
        assert real == ["a"]


class TestPrecontextParsing:
    def test_returns_none_without_sentinels(self):
        assert parse_precontext("just prose") is None

    def test_returns_none_on_unterminated_block(self):
        assert parse_precontext("<precontext> {broken") is None

    def test_strip_is_a_noop_without_sentinels(self):
        assert strip_precontext("just prose") == "just prose"
