"""The transformers model's chat loop and grounding reader, without torch."""

import json

from interfaze_lite import agent, grounding, prompts


class TestToolCalls:
    def test_reads_the_function_form_this_brain_emits(self):
        text = (
            "<tool_call>\n<function=object_detection>\n<parameter=file_ref_id>\nref-0\n"
            '</parameter>\n<parameter=prompts>\n["sword", "fan"]\n</parameter>\n'
            "</function>\n</tool_call>"
        )
        assert agent.tool_calls(text) == [
            {"name": "object_detection", "arguments": {"file_ref_id": "ref-0", "prompts": ["sword", "fan"]}}
        ]

    def test_reads_the_json_form(self):
        text = '<tool_call>\n{"name": "ocr", "arguments": {"file_ref_id": "ref-0"}}\n</tool_call>'
        assert agent.tool_calls(text) == [{"name": "ocr", "arguments": {"file_ref_id": "ref-0"}}]

    def test_prose_has_none_and_shows_without_markup(self):
        assert agent.tool_calls("The total is 144.02.") == []
        assert agent.visible("<think>sum it</think>The total is 144.02.") == "The total is 144.02."


def _run(replies, tools, files=("receipt.jpg",)):
    turns, shown = [], []

    def generate(convo, tools=None, keep_markup=False):
        turns.append([dict(m) for m in convo])
        return replies.pop(0)

    def show_images(convo, refs):
        shown.append(dict(refs))
        return convo

    out = agent.run(
        [{"role": "user", "content": "What is the total?"}],
        list(files),
        generate=generate,
        tools=tools,
        show_images=show_images,
    )
    return out, turns, shown


OCR_CALL = '<tool_call>\n{"name": "ocr", "arguments": {"file_ref_id": "ref-0"}}\n</tool_call>'


class TestChatLoop:
    def test_runs_a_tool_and_answers_from_it(self):
        full = {"extracted_text": "TOTAL 144.02", "sections": [{"lines": []}], "width": 720, "height": 960}
        seen = {"extracted_text": "TOTAL 144.02", "width": 720, "height": 960}
        out, turns, shown = _run(
            [OCR_CALL, "The total is 144.02."], {"ocr": lambda _path, _args: (full, seen)}
        )
        assert out == {"content": "The total is 144.02.", "precontext": [{"name": "ocr", "result": full}]}
        assert "All File References:\n- ref-0: receipt.jpg" in turns[0][0]["content"]
        assert json.loads(turns[1][-1]["content"]) == seen
        assert shown == [{"ref-0": "receipt.jpg"}]  # the image, once a tool read it

    def test_a_file_answered_by_eye_gets_one_reminder(self):
        out, turns, _ = _run(
            ["It says $1000.", OCR_CALL, "The total is 144.02."],
            {"ocr": lambda _path, _args: ({"extracted_text": "x"}, {"extracted_text": "x"})},
        )
        assert turns[1][-1] == {"role": "user", "content": prompts.NUDGE}
        assert out["content"] == "The total is 144.02."

    def test_a_made_up_reference_is_refused_not_fetched(self):
        call = '<tool_call>\n{"name": "ocr", "arguments": {"file_ref_id": "https://evil.example/x.png"}}\n</tool_call>'
        fetched = []
        out, turns, _ = _run(
            [call, "I could not read it."], {"ocr": lambda path, _args: fetched.append(path) or ({}, {})}
        )
        assert fetched == [] and out["precontext"] == []
        assert "unknown file reference" in json.loads(turns[1][-1]["content"])["error"]

    def test_the_model_reads_a_bounded_text_the_caller_gets_whole(self):
        long = "x" * (agent.MAX_CONTEXT_CHARS + 10)
        out, turns, _ = _run(
            [OCR_CALL, "done"],
            {"ocr": lambda _path, _args: ({"extracted_text": long}, {"extracted_text": long})},
        )
        read = json.loads(turns[1][-1]["content"])
        assert len(read["extracted_text"]) == agent.MAX_CONTEXT_CHARS and read["truncated"]
        assert out["precontext"][0]["result"]["extracted_text"] == long

    def test_no_file_means_no_tools_offered(self):
        offered = []

        def generate(convo, tools=None, keep_markup=False):
            offered.append(tools)
            return "Hello."

        out = agent.run(
            [{"role": "user", "content": "hi"}], [], generate=generate, tools={}, show_images=lambda c, _r: c
        )
        assert out["content"] == "Hello." and offered == [None]


class TestGroundingReader:
    def test_boxes_come_back_on_the_grid_without_repeats(self):
        raw = (
            '[{"bbox_2d": [100, 200, 300, 400], "label": "sword"},'
            ' {"bbox_2d": [101, 201, 301, 401], "label": "sword"},'
            ' {"bbox_2d": [500, 500, 600, 600]}]'
        )
        found = grounding.parse(raw, "sword, fan")
        assert [e["label"] for e in found] == ["sword", "sword, fan"]
        assert found[0]["bounds"]["top_left"] == {"x": 100, "y": 200}

    def test_boxes_land_on_the_image_in_pixels(self):
        [box] = grounding.to_pixels(
            grounding.parse('[{"bbox_2d": [100, 200, 300, 400], "label": "x"}]', "x"), 1440, 1030
        )
        assert box["bounds"]["top_left"] == {"x": 144, "y": 206}
        assert box["bounds"]["bottom_right"] == {"x": 432, "y": 412}
        assert "coordinate_space" not in box

    def test_the_two_domains_are_asked_differently(self):
        assert "UI screenshot" in grounding.prompt(["save button"], "ui")
        assert "Order the array" in grounding.prompt(["sword"], "object")
        assert "Order the array" not in grounding.prompt(["save button"], "ui")
