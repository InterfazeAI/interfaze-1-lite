import asyncio
import base64
import json
from collections.abc import Callable
from datetime import datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest

from interfaze_lite import forecasting, guard, tools, translation, validate
from interfaze_lite.envelope import Usage
from interfaze_lite.filerefs import FileRefs


class TestTranslationPrompts:
    def test_sentence_prompt_is_interfazes(self):
        system, user = translation.prompts("Good morning everyone", "fr")
        assert system == (
            "You are a professional translation engine. Maintain the original meaning, tone and "
            "structure of the text. Text may contain multiple languages. Translate all of them to "
            "the target language. Don't translate acronyms, abbreviations, brand names and proper "
            "nouns. Translate numerical value to the target language."
        )
        assert user == (
            'Translate the following text to "French (fr)"'
            "\n    \n\nOriginal Text: Good morning everyone\nTranslated Text:"
        )

    def test_one_word_uses_the_dictionary_prompt(self):
        system, user = translation.prompts("Hello", "es", "en")
        assert system.startswith("You are a professional dictionary.")
        assert user.startswith('Translate the following word from "English (en)" to "Spanish (es)"')

    def test_rtl_target_says_so(self):
        _, user = translation.prompts("Good morning", "ar")
        assert 'to "Arabic (ar)" in right to left (RTL) direction' in user

    def test_language_list_is_interfazes(self):
        langs = translation.languages()
        assert len(langs) == 163
        assert langs["zh"]["name"] and "xx" not in langs
        assert translation.unsupported("xx").startswith('"xx" is not a supported language code.')


class TestChunking:
    def test_short_text_is_one_chunk(self):
        assert translation.split_into_chunks("hi there", 5000) == (["hi there"], [])

    def test_long_text_splits_at_a_sentence_and_rejoins(self):
        text = ("First sentence here. " * 400).strip()
        chunks, seps = translation.split_into_chunks(text, 5000)
        assert len(chunks) > 1 and all(len(c) <= 5000 for c in chunks)
        assert seps[0] == ". "
        assert translation.stitch(chunks, seps) == text

    def test_prefers_paragraph_breaks(self):
        text = "a" * 3000 + "\n\n" + "b" * 3000
        chunks, seps = translation.split_into_chunks(text, 5000)
        assert chunks == ["a" * 3000, "b" * 3000] and seps == ["\n\n"]

    def test_no_separator_cuts_hard_and_rejoins_with_space(self):
        chunks, seps = translation.split_into_chunks("x" * 12000, 5000)
        assert [len(c) for c in chunks] == [5000, 5000, 2000] and seps == ["", ""]
        assert translation.stitch(["A", "B"], [""]) == "A B"


def _ctx(
    refs=None,
    structured=None,
    perception: Callable[[httpx.Request], httpx.Response] | None = None,
    prompt="",
):
    async def post_handler(request: httpx.Request) -> httpx.Response:
        assert perception is not None
        return perception(request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(post_handler) if perception is not None else None)

    settings = SimpleNamespace(
        perception_url="http://perception",
        tool_timeout_s=30,
        component_names=(),
        max_context_chars=100_000,
    )

    return tools.ToolContext(
        refs=refs or FileRefs(),
        http=http,
        settings=settings,
        structured=structured,
        usage=Usage(),
        prompt=prompt,
    )


class TestTranslateTool:
    def test_calls_the_brain_with_interfazes_prompt_and_returns_its_shape(self):
        seen = []

        async def structured(messages, schema, sampling):
            seen.append((messages, schema, sampling))
            return SimpleNamespace(
                content=json.dumps({"translated_text": " Bonjour "}), prompt_tokens=40, completion_tokens=5
            )

        ctx = _ctx(structured=structured)
        result = asyncio.run(
            tools.dispatch("translate", json.dumps({"text": "Good morning", "target_language": "fr"}), ctx)
        )
        assert result.model_facing == {
            "translated_text": "Bonjour",
            "source_language": "auto-detected",
            "target_language": "fr",
            "batch_size": 1,
            "chunks_processed": 1,
        }
        messages, schema, sampling = seen[0]
        assert messages[1]["content"].startswith('Translate the following text to "French (fr)"')
        assert schema["schema"]["required"] == ["translated_text"]
        assert sampling.temperature == 0 and sampling.max_tokens == 5000
        assert (ctx.usage.prompt_tokens, ctx.usage.completion_tokens) == (40, 5)

    def test_array_input_returns_an_array(self):
        async def structured(messages, schema, sampling):
            word = messages[1]["content"].split("Original Text: ")[1].split("\n")[0]
            return SimpleNamespace(
                content=json.dumps({"translated_text": word.upper()}), prompt_tokens=1, completion_tokens=1
            )

        result = asyncio.run(
            tools.dispatch(
                "translate",
                json.dumps({"text": ["one two", "three"], "target_language": "de"}),
                _ctx(structured=structured),
            )
        )
        assert result.model_facing["translated_text"] == ["ONE TWO", "THREE"]
        assert result.model_facing["batch_size"] == 2

    def test_unknown_language_and_same_language_are_refused(self):
        result = asyncio.run(
            tools.dispatch("translate", json.dumps({"text": "hi", "target_language": "xx"}), _ctx())
        )
        assert result.model_facing["error"].startswith('"xx" is not a supported language code.')
        result = asyncio.run(
            tools.dispatch(
                "translate",
                json.dumps({"text": "hi", "target_language": "fr", "current_language": "fr"}),
                _ctx(),
            )
        )
        assert result.model_facing["error"] == "Source and target language cannot be the same"


class TestDatasetParsing:
    def test_two_column_csv_with_header(self):
        rows = forecasting.parse_csv("date,sales\n2024-01-01,10\n2024-01-02,12\n")
        assert forecasting.rows_to_dataset(rows) == [
            {"date": "2024-01-01", "value": 10.0},
            {"date": "2024-01-02", "value": 12.0},
        ]

    def test_quoted_fields_bom_and_grouped_numbers(self):
        rows = forecasting.parse_csv('﻿date,"revenue"\r\n2024-01-01,"1,234"\r\n')
        assert forecasting.rows_to_dataset(rows) == [{"date": "2024-01-01", "value": 1234.0}]

    def test_one_numeric_column_among_several_is_taken(self):
        rows = forecasting.parse_csv("date,region,sales\n2024-01-01,west,5\n2024-01-02,east,7\n")
        assert [p["value"] for p in forecasting.rows_to_dataset(rows)] == [5.0, 7.0]

    def test_ambiguous_columns_ask_for_a_hint(self):
        rows = forecasting.parse_csv("date,open,close\n2024-01-01,1,2\n")
        with pytest.raises(forecasting.DatasetError, match="Specify date_column and value_column"):
            forecasting.rows_to_dataset(rows)
        assert forecasting.rows_to_dataset(rows, value_column="close")[0]["value"] == 2.0

    def test_unknown_hint_lists_the_columns(self):
        rows = forecasting.parse_csv("date,a,b\n2024-01-01,1,2\n")
        with pytest.raises(forecasting.DatasetError, match="Available columns: date, a, b"):
            forecasting.rows_to_dataset(rows, value_column="missing")

    def test_json_shapes(self):
        assert forecasting.json_to_dataset([{"ds": "2024-01-01", "y": 3}]) == [
            {"date": "2024-01-01", "value": 3.0}
        ]
        assert forecasting.json_to_dataset({"data": [["2024-01-01", "4"]]}) == [
            {"date": "2024-01-01", "value": 4.0}
        ]
        assert forecasting.parse_dataset('[{"date":"2024-01-01","value":1}]')[0]["value"] == 1.0


class TestSeriesInText:
    def test_the_table_is_found_and_a_sentence_before_it_is_not_its_header(self):
        text = (
            "Forecast this, please: next 3 days\ndate,value\n"
            + "\n".join(f"2024-01-0{d},{d}" for d in range(1, 6))
            + "\nthanks"
        )
        assert forecasting.dataset_from_text(text) == [
            {"date": f"2024-01-0{d}", "value": float(d)} for d in range(1, 6)
        ]

    def test_no_series_is_empty(self):
        assert forecasting.dataset_from_text("no data here, sorry") == []
        assert forecasting.dataset_from_text("date,value\n2024-01-01,1") == []


class TestSeries:
    def test_adds_a_time_sorts_and_sums_duplicates(self):
        data = [
            {"date": d, "value": v}
            for d, v in [
                ("2024-01-03", 3),
                ("2024-01-01", 1),
                ("2024-01-02", 2),
                ("2024-01-02", 5),
                ("2024-01-04", 4),
                ("2024-01-05", 5),
            ]
        ]
        y = forecasting.series(data)
        assert list(y) == [
            "2024-01-01 00:00:00",
            "2024-01-02 00:00:00",
            "2024-01-03 00:00:00",
            "2024-01-04 00:00:00",
            "2024-01-05 00:00:00",
        ]
        assert y["2024-01-02 00:00:00"] == 7.0

    def test_five_unique_dates_minimum(self):
        with pytest.raises(forecasting.DatasetError, match="At least 5 unique dates are required"):
            forecasting.series([{"date": "2024-01-01", "value": 1}] * 6)

    def test_step_and_future_timestamps(self):
        dts = [datetime(2024, 1, 1) + timedelta(weeks=i) for i in range(6)]
        step = forecasting.infer_step(dts)
        assert step == timedelta(days=7)
        assert forecasting.future_timestamps(dts[-1], 2, step) == [
            "2024-02-12 00:00:00",
            "2024-02-19 00:00:00",
        ]

    def test_private_addresses_are_refused(self):
        for url in (
            "http://127.0.0.1/a.csv",
            "http://169.254.169.254/x",
            "file:///etc/passwd",
            "http://localhost/a.csv",
            "http://10.0.0.1/a.csv",
        ):
            with pytest.raises(forecasting.DatasetError):
                forecasting.assert_fetchable_url(url)
        forecasting.assert_fetchable_url("https://example.com/data.csv")


class TestForecastTool:
    SERIES = [
        {"date": f"2024-01-{d:02d}", "value": v}
        for d, v in zip(
            range(1, 13), [412, 387, 524, 461, 398, 542, 475, 401, 558, 489, 419, 571], strict=True
        )
    ]

    def _perception(self, captured):
        def handler(request):
            captured.append(json.loads(request.content))
            return httpx.Response(
                200, json={"timestamp": ["2024-01-13 00:00:00", "2024-01-14 00:00:00"], "value": [500, 510]}
            )

        return handler

    def test_inline_dataset_goes_to_timesfm_as_interfaze_sends_it(self):
        captured = []
        result = asyncio.run(
            tools.dispatch(
                "forecast",
                json.dumps({"steps": 2, "dataset": self.SERIES}),
                _ctx(perception=self._perception(captured)),
            )
        )
        assert captured[0]["fh"] == 2
        assert list(captured[0]["y"])[0] == "2024-01-01 00:00:00"
        assert result.model_facing == {
            "predictions": [
                {"date": "2024-01-13 00:00:00", "value": 500},
                {"date": "2024-01-14 00:00:00", "value": 510},
            ]
        }

    def test_csv_file_reference(self, tmp_path):
        path = tmp_path / "sales.csv"
        path.write_text("date,value\n" + "\n".join(f"{p['date']},{p['value']}" for p in self.SERIES))
        refs = FileRefs()
        refs.add(str(path), filename="sales.csv", mime="text/csv")
        captured = []
        result = asyncio.run(
            tools.dispatch(
                "forecast",
                json.dumps({"steps": 2, "file_ref_id": "ref-0"}),
                _ctx(refs=refs, perception=self._perception(captured)),
            )
        )
        assert "predictions" in result.model_facing and len(captured[0]["y"]) == 12

    def test_a_series_written_in_the_message_is_read_from_it(self):
        """Copied into the call, 365 rows ran past the tool turn's token limit and arrived empty."""
        prompt = "Forecast the next 2 days, please:\ndate,value\n" + "\n".join(
            f"{p['date']},{p['value']}" for p in self.SERIES
        )
        captured = []
        result = asyncio.run(
            tools.dispatch(
                "forecast",
                json.dumps({"steps": 2}),
                _ctx(perception=self._perception(captured), prompt=prompt),
            )
        )
        assert "predictions" in result.model_facing and len(captured[0]["y"]) == 12

    def test_too_few_and_too_many_points(self):
        few = asyncio.run(
            tools.dispatch("forecast", json.dumps({"steps": 2, "dataset": self.SERIES[:3]}), _ctx())
        )
        assert few.model_facing["error"] == "Need at least 5 historical data points to forecast."
        many = [{"date": f"2024-01-01 {h:02d}:{m:02d}:00", "value": 1} for h in range(24) for m in range(60)][
            :1001
        ]
        over = asyncio.run(tools.dispatch("forecast", json.dumps({"steps": 2, "dataset": many}), _ctx()))
        assert over.model_facing["forecast_row_limit_exceeded"] is True


class TestGuardDirective:
    def test_codes_and_the_directive_is_removed(self):
        codes, rest = guard.extract("<guard>S1, s2 ,S10</guard>\nBe brief.")
        assert codes == ["S1", "S2", "S10"] and rest == "Be brief."

    def test_all_expands_to_every_code(self):
        codes, _ = guard.extract("<guard>ALL</guard>")
        assert codes == list(guard.ALL_CODES) and len(codes) == 17

    def test_absent_or_empty_is_no_guard(self):
        assert guard.extract("Be brief.") == (None, "Be brief.")
        assert guard.extract("<guard> </guard>")[0] is None


class TestGuardVerdict:
    def test_safe(self):
        safe, text, pre = guard.verdict("safe", ["S1", "S2"])
        assert safe and pre == [{"name": "text_guardrail_classifier", "result": "safe"}]

    def test_unsafe_in_a_requested_category_blocks(self):
        safe, text, pre = guard.verdict("unsafe\nS1", ["S1", "S2", "S3", "S10"])
        assert not safe and text == "unsafe S1"
        assert pre == [{"name": "text_guardrail_classifier", "result": ["S1"]}]

    def test_unsafe_outside_the_requested_categories_passes(self):
        safe, text, pre = guard.verdict("unsafe\nS6", ["S1", "S2"])
        assert safe and pre[0]["result"] == "safe"

    def test_several_codes(self):
        safe, text, _ = guard.verdict("unsafe\nS1,S10", ["S1", "S10"])
        assert text == "unsafe S1 S10"

    def test_images_add_their_codes(self):
        gore = guard.image_result(adult=0.02, racy=0.1, gore=0.9)
        safe, text, pre = guard.verdict("safe", ["S1_IMAGE"], [gore])
        assert not safe and text == "unsafe S1_IMAGE"
        assert pre[1] == {"name": "image_guardrail_classifier", "result": gore}
        safe, text, _ = guard.verdict(
            "unsafe\nS1", ["S1", "S12_IMAGE", "S15_IMAGE"], [guard.image_result(0.95, 0.9, 0.0)]
        )
        assert text == "unsafe S1 S12_IMAGE S15_IMAGE"

    def test_image_result_matches_interfazes_derivation(self):
        r = guard.image_result(adult=0.6, racy=0.3, gore=0.1)
        assert (r["nudity"], r["gore"], r["nsfw"]) == (True, False, True)
        assert r["nsfw_score"] == pytest.approx(guard.weighed_average([0.6, 0.1, 0.3]))
        assert guard.weighed_average([0, 0, 0]) == 0.0


class TestClassifierVerdict:
    def test_safe(self):
        assert guard.classifier_verdict("Safety: Safe\nCategories: None", 0.01, 0.5) == "safe"

    def test_unsafe_names_every_code_its_categories_cover(self):
        answer = "Safety: Unsafe\nCategories: Violent, Unethical Acts"
        assert guard.classifier_verdict(answer, 0.99, 0.5) == "unsafe\nS1,S9,S5,S10"

    def test_unsafe_below_the_threshold_is_safe(self):
        assert guard.classifier_verdict("Safety: Unsafe\nCategories: PII", 0.4, 0.5) == "safe"

    def test_controversial_is_safe(self):
        assert guard.classifier_verdict("Safety: Controversial\nCategories: Violent", 0.3, 0.5) == "safe"

    def test_controversial_self_harm_is_s11(self):
        answer = "Safety: Controversial\nCategories: Suicide & Self-Harm"
        assert guard.classifier_verdict(answer, 0.05, 0.5) == "unsafe\nS11"

    def test_a_category_without_a_code_blocks_no_requested_code(self):
        text = guard.classifier_verdict("Safety: Unsafe\nCategories: Jailbreak", 0.99, 0.5)
        assert text == "unsafe"
        assert guard.verdict(text, ["S1", "S2", "S3", "S10", "S11"])[0]

    def test_the_landings_bomb_preset_blocks_violence(self):
        text = guard.classifier_verdict("Safety: Unsafe\nCategories: Violent", 1.0, 0.5)
        safe, content, _ = guard.verdict(text, ["S1", "S2", "S3", "S10", "S11"])
        assert not safe and content == "unsafe S1"


def test_new_tasks_are_routable():
    msgs = [{"role": "system", "content": "<task>translate</task>"}, {"role": "user", "content": "x"}]
    assert validate.extract_task(msgs) == "translate"
    msgs[0]["content"] = "<task>forecast</task>"
    assert validate.extract_task(msgs) == "forecast"


def test_text_tools_are_offered_without_files():
    names = [s["function"]["name"] for s in tools.schemas(files=False)]
    assert names == ["translate", "forecast"]
    assert {"ocr", "stt", "translate", "forecast"} <= {
        s["function"]["name"] for s in tools.schemas(files=True)
    }


def test_ocr_returns_a_text_file_as_it_is():
    refs = FileRefs()
    csv = "id,name\nITEM-001,notebook\nITEM-002,measuring cups\n"
    refs.add_data_uri("data:text/csv;base64," + base64.b64encode(csv.encode()).decode(), filename="items.csv")

    def perception(request):
        raise AssertionError("a text file needs no document reader")

    try:
        result = asyncio.run(
            tools.dispatch(
                "ocr", json.dumps({"file_ref_id": "ref-0"}), _ctx(refs=refs, perception=perception)
            )
        )
    finally:
        refs.cleanup()
    assert result.model_facing == {"extracted_text": csv}
    assert result.precontext == {"extracted_text": csv, "sections": [], "width": None, "height": None}
