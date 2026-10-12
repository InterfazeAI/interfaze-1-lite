"""Contract tests. No GPU, no network: pure functions over fixtures, plus source checks on shipped modules."""

import ast
import pathlib

import pytest

from interfaze_lite.contracts import (
    Bounds,
    OCRLine,
    OCRSection,
    OCRWord,
    Point,
    SpeakerTurn,
    VLMRegion,
    Word,
    attach_text_to_layout,
    attribute_speakers,
    backfill_sections,
    collapse_to_page,
    document_kind,
    drop_unread,
    enhance_scan,
    estimate_words,
    group_by_speaker,
    iou,
    merge_lines_into_sections,
    normalise_error,
    pdf_render_dpi,
    read_layout,
    rescale_page,
    restore_marks,
    speech_windows,
    split_long_chunks,
    split_words,
    stitch,
    to_bounds,
)
from interfaze_lite.filerefs import extract_from_messages
from interfaze_lite.grounding import cut_ladders, suppress_duplicates
from interfaze_lite.validate import RequestError, extract_task, normalise_roles, validate_messages

MODELING = pathlib.Path("src/interfaze_lite/modeling_interfaze_lite.py")

IMG_W, IMG_H = 1000, 800


def box(x1, y1, x2, y2):
    return to_bounds([x1, y1, x2, y2], src="xyxy", img_w=IMG_W, img_h=IMG_H)


class TestToBounds:
    def test_xyxy_produces_four_corners(self):
        b = to_bounds([10, 20, 110, 70], src="xyxy", img_w=IMG_W, img_h=IMG_H)
        assert b.top_left == Point(10, 20)
        assert b.top_right == Point(110, 20)
        assert b.bottom_right == Point(110, 70)
        assert b.bottom_left == Point(10, 70)
        assert (b.width, b.height) == (100, 50)

    def test_norm1000_rescales_to_pixels(self):
        # Qwen-VL grounding grid. Half-width, quarter-height.
        b = to_bounds([0, 0, 500, 250], src="norm1000", img_w=1000, img_h=800)
        assert (b.width, b.height) == (500, 200)

    def test_norm01_rescales_to_pixels(self):
        b = to_bounds([0.1, 0.5, 0.2, 0.75], src="norm01", img_w=1000, img_h=800)
        assert (b.x1, b.y1, b.x2, b.y2) == (100, 400, 200, 600)

    def test_cxcywh_centres_the_box(self):
        b = to_bounds([100, 100, 40, 20], src="cxcywh", img_w=IMG_W, img_h=IMG_H)
        assert (b.x1, b.y1, b.x2, b.y2) == (80, 90, 120, 110)

    def test_quad_takes_the_enclosing_axis_aligned_box(self):
        # A line-detector quad, rotated slightly.
        b = to_bounds([[10, 22], [110, 20], [112, 70], [12, 72]], src="quad", img_w=IMG_W, img_h=IMG_H)
        assert (b.x1, b.y1, b.x2, b.y2) == (10, 20, 112, 72)

    def test_reversed_corners_still_yield_a_positive_box(self):
        b = to_bounds([110, 70, 10, 20], src="xyxy", img_w=IMG_W, img_h=IMG_H)
        assert (b.width, b.height) == (100, 50)

    def test_clamps_to_image(self):
        b = to_bounds([-50, -50, 2000, 2000], src="xyxy", img_w=IMG_W, img_h=IMG_H)
        assert (b.x1, b.y1, b.x2, b.y2) == (0, 0, IMG_W, IMG_H)

    def test_width_always_equals_corner_difference(self):
        # Rounding each corner separately can disagree with width by a pixel.
        b = to_bounds([10.6, 20.4, 110.4, 70.6], src="xyxy", img_w=IMG_W, img_h=IMG_H)
        assert b.width == b.x2 - b.x1
        assert b.height == b.y2 - b.y1

    def test_rejects_nan(self):
        with pytest.raises(ValueError):
            to_bounds([float("nan"), 0, 10, 10], src="xyxy", img_w=IMG_W, img_h=IMG_H)

    def test_rejects_wrong_arity(self):
        with pytest.raises(ValueError):
            to_bounds([1, 2, 3], src="xyxy", img_w=IMG_W, img_h=IMG_H)


class TestOCRComposition:
    def test_same_row_lines_merge_into_one_section(self):
        left = OCRLine("left column", box(10, 100, 200, 120), 0.9)
        right = OCRLine("right column", box(400, 100, 600, 120), 0.9)
        sections = merge_lines_into_sections([left, right])
        assert len(sections) == 1
        assert sections[0].text == "left column right column"

    def test_different_rows_stay_separate(self):
        top = OCRLine("first", box(10, 100, 200, 120), 0.9)
        bottom = OCRLine("second", box(10, 300, 200, 320), 0.9)
        assert len(merge_lines_into_sections([top, bottom])) == 2

    def test_differing_height_blocks_the_merge(self):
        # Same vertical band, but a heading is taller than body text.
        heading = OCRLine("HEADING", box(10, 100, 200, 140), 0.9)
        body = OCRLine("body", box(400, 100, 600, 120), 0.9)
        assert len(merge_lines_into_sections([heading, body])) == 2

    def test_merged_section_orders_lines_left_to_right(self):
        right = OCRLine("second", box(400, 100, 600, 120), 0.9)
        left = OCRLine("first", box(10, 100, 200, 120), 0.9)
        sections = merge_lines_into_sections([right, left])
        assert sections[0].text == "first second"


class TestStitch:
    def test_region_text_wins_detector_geometry_survives(self):
        region = VLMRegion(text="# Invoice", bounds=box(0, 0, 300, 50), category="title")
        line = OCRLine(
            "Invoice", box(10, 10, 200, 40), 0.95, words=[OCRWord("Invoice", box(10, 10, 200, 40), 0.95)]
        )
        sections = stitch([region], [line])
        assert len(sections) == 1
        assert sections[0].text == "# Invoice"
        assert sections[0].lines[0].average_confidence == 0.95
        assert sections[0].lines[0].bounds.x1 == 10

    def test_every_line_survives_and_takes_the_readers_words(self):
        """Detector owns boxes, reader owns words; matched by position, 62 of a form's 98 lines vanished."""
        region = VLMRegion(text="Invoice total 42.00", bounds=box(0, 0, 10, 10))
        matched = OCRLine("Invoicetotal 42.00", box(10, 10, 200, 40), 0.9)  # spaces dropped
        orphan = OCRLine("footnote", box(10, 700, 200, 730), 0.6)  # not in the reader
        [section] = stitch([region], [matched, orphan])
        assert [ln.text for ln in section.lines] == ["Invoice total 42.00", "footnote"]
        assert [ln.bounds.y1 for ln in section.lines] == [10, 700]
        assert section.text == "Invoice total 42.00"

    def test_a_number_inside_a_longer_one_is_not_a_match(self):
        """ "150.02" occurs inside the barcode 005150025499; it is not that barcode."""
        region = VLMRegion(text="PEANUT BUTTR 005150025499 F 5.44 CASH TEND 150.02", bounds=box(0, 0, 10, 10))
        [section] = stitch([region], [OCRLine("150.02", box(10, 10, 200, 40), 0.9)])
        assert section.lines[0].text == "150.02"

    def test_a_label_the_reader_has_once_goes_to_one_line(self):
        region = VLMRegion(text="Dependent 1 First name", bounds=box(0, 0, 10, 10))
        lines = [
            OCRLine("Dependent 1", box(10, 10, 100, 40), 0.9),
            OCRLine("Dependent 2", box(110, 10, 200, 40), 0.9),
        ]
        [section] = stitch([region], lines)
        assert [ln.text for ln in section.lines] == ["Dependent 1", "Dependent 2"]

    def test_another_case_is_another_occurrence(self):
        region = VLMRegion(text="Dr. Aaron Patel, MD", bounds=box(0, 0, 10, 10))
        [section] = stitch([region], [OCRLine("DR. AARON PATEL", box(10, 10, 200, 40), 0.9)])
        assert section.lines[0].text == "DR. AARON PATEL"  # the stamp, not the provider line

    def test_a_word_broken_across_lines_is_split_where_the_line_breaks(self):
        region = VLMRegion(
            text="leads a suite of deterministic developer-task benchmarks", bounds=box(0, 0, 10, 10)
        )
        lines = [
            OCRLine("leads a suite of de-", box(10, 10, 300, 40), 0.9),
            OCRLine("terministic developer-task benchmarks", box(10, 50, 300, 80), 0.9),
        ]
        [section] = stitch([region], lines)
        assert [ln.text for ln in section.lines] == [
            "leads a suite of de-",
            "terministic developer-task benchmarks",
        ]

    def test_reader_text_no_line_found_is_still_the_pages_text(self):
        region = VLMRegion(text="orphan heading", bounds=box(0, 0, 50, 20))
        far = OCRLine("elsewhere", box(900, 700, 990, 740), 0.9)
        [section] = stitch([region], [far])
        assert section.text == "orphan heading"
        assert [ln.text for ln in section.lines] == ["elsewhere"]

    def test_lines_are_found_by_what_they_say_not_where_the_reader_put_them(self):
        """With the first reader's x halved, matching on boxes lost a two-column page's right column."""
        left = VLMRegion(text="Many production workloads are not open ended", bounds=box(75, 100, 290, 160))
        right = VLMRegion(text="A common remedy is to route between models", bounds=box(290, 100, 500, 160))
        lines = [
            OCRLine("Many production workloads", box(150, 100, 430, 125), 0.9),
            OCRLine("are not open ended", box(150, 130, 430, 155), 0.9),
            OCRLine("A common remedy is", box(430, 100, 850, 125), 0.9),
            OCRLine("toroute between models", box(430, 130, 850, 155), 0.9),
        ]
        [section] = stitch([left, right], lines)
        assert [ln.text for ln in section.lines] == [
            "Many production workloads",
            "A common remedy is",
            "are not open ended",
            "to route between models",
        ]
        assert section.lines[1].bounds.x1 == 430  # the detector's box, not the reader's

    def test_each_line_gets_its_own_stretch_of_the_paragraph(self):
        """Every line carrying the whole paragraph multiplied the page's text by its lines."""
        region = VLMRegion(
            text="Interfaze fuses task-specific encoders into a decoder "
            "through a shared embedding space, so one pass does it.",
            bounds=box(0, 0, 400, 100),
        )
        lines = [
            OCRLine("Interfazefuses task-specific encoders", box(0, 0, 400, 30), 0.9),
            OCRLine("into adecoder through a shared", box(0, 35, 400, 65), 0.9),
            OCRLine("embedding space, so one pass does it.", box(0, 70, 400, 100), 0.9),
        ]
        [section] = stitch([region], lines)
        assert [ln.text for ln in section.lines] == [
            "Interfaze fuses task-specific encoders",
            "into a decoder through a shared",
            "embedding space, so one pass does it.",
        ]

    def test_a_capital_dotted_i_does_not_shift_the_text(self):
        """'İ'.lower() is two characters, so lowercasing shifted positions: a Turkish page hit IndexError."""
        region = VLMRegion(text="İSTANBUL İZMİR İÇ TİCARET Toplam 2024 1.250.000", bounds=box(0, 0, 10, 10))
        [section] = stitch([region], [OCRLine("Toplam 2024 1.250.000", box(10, 10, 200, 40), 0.9)])
        assert section.lines[0].text == "Toplam 2024 1.250.000"


class TestIou:
    def test_iou_is_zero_for_disjoint_boxes(self):
        assert iou(box(0, 0, 10, 10), box(100, 100, 110, 110)) == 0.0


class TestUnreadLines:
    """Unread detector lines count only when sure: a dog photo's fur and fence read as "LE" and "3213"."""

    def test_an_unsure_line_the_reader_did_not_read_is_dropped(self):
        region = VLMRegion(text="DRIVER LICENSE", bounds=box(0, 0, 10, 10))
        read = OCRLine("DRIVER LICENSE", box(10, 10, 200, 40), 0.3)
        sure = OCRLine("Ivan M", box(10, 300, 200, 340), 0.97)
        noise = OCRLine("B", box(10, 500, 30, 520), 0.22)
        assert drop_unread([region], [read, sure, noise], 0.6, 0.9) == [read, sure]

    def test_with_no_reader_text_only_sure_lines_remain(self):
        lines = [
            OCRLine("LE", box(0, 0, 50, 50), 0.35),
            OCRLine("II", box(0, 0, 300, 190), 0.73),
            OCRLine("STOP", box(0, 60, 50, 90), 0.93),
        ]
        assert drop_unread([], lines, 0.6, 0.9) == lines[2:]


class TestRestoreMarks:
    """Quote marks the reader dropped come back from a sure detector line, only between the same letters."""

    MD = '# Licence\n\nHGT 5-02" WGT 185 lb'

    def _restore(self, detector_text, confidence=0.95, markdown=MD):
        region = VLMRegion(text='HGT 5-02" WGT 185 lb', bounds=box(0, 0, 10, 10))
        line = OCRLine(detector_text, box(10, 10, 200, 40), confidence)
        [region], markdown = restore_marks([region], markdown, [line])
        return region.text, markdown

    def test_a_dropped_apostrophe_comes_back_everywhere(self):
        text, markdown = self._restore("HGT 5'-02\"")
        assert text == "HGT 5'-02\" WGT 185 lb"
        assert markdown == "# Licence\n\nHGT 5'-02\" WGT 185 lb"
        [section] = stitch(
            [VLMRegion(text=text, bounds=box(0, 0, 10, 10))],
            [OCRLine("HGT 5'-02\"", box(10, 10, 200, 40), 0.95)],
        )
        assert section.lines[0].text == "HGT 5'-02\""

    def test_the_readers_spacing_is_kept(self):
        assert self._restore("HGT5'-02\"")[0] == "HGT 5'-02\" WGT 185 lb"

    def test_other_marks_the_detector_saw_are_not_added(self):
        assert self._restore('HGT 5.-02"')[0] == 'HGT 5-02" WGT 185 lb'

    def test_different_letters_or_case_change_nothing(self):
        assert self._restore("HGT 5'-03\"")[0] == 'HGT 5-02" WGT 185 lb'
        assert self._restore("hgt 5'-02\"")[0] == 'HGT 5-02" WGT 185 lb'

    def test_no_mark_goes_inside_a_word_the_reader_read_whole(self):
        region = VLMRegion(text="This also contains the library", bounds=box(0, 0, 10, 10))
        line = OCRLine("This also con'tains the library", box(10, 10, 200, 40), 0.97)
        assert restore_marks([region], "", [line])[0][0].text == "This also contains the library"

    def test_an_unsure_detector_changes_nothing(self):
        assert self._restore("HGT 5'-02\"", confidence=0.6)[0] == 'HGT 5-02" WGT 185 lb'

    def test_markdown_saying_it_twice_is_left_alone(self):
        twice = 'HGT 5-02" WGT 185 lb\n\nHGT 5-02" WGT 185 lb'
        text, markdown = self._restore("HGT 5'-02\"", markdown=twice)
        assert text == "HGT 5'-02\" WGT 185 lb" and markdown == twice


class TestPdfRenderDpi:
    BUDGET, SMALL = 4_194_304, 2_097_152

    def dpi(self, width_pt, height_pt):
        return pdf_render_dpi(width_pt, height_pt, dpi=200, max_pixels=self.BUDGET, small_pixels=self.SMALL)

    def test_a_standard_page_keeps_the_default(self):
        assert self.dpi(595, 842) == 200  # A4
        assert self.dpi(612, 792) == 200  # Letter

    def test_a_small_page_is_read_at_the_pixel_budget(self):
        """At 200 DPI a tiny-print page got 11% of its scan's pixels; the reader made text up (results.md)."""
        dpi = self.dpi(311.04, 204.36)
        assert 560 <= dpi <= 600
        assert (311.04 / 72 * dpi) * (204.36 / 72 * dpi) <= self.BUDGET

    def test_an_oversized_page_is_rendered_down_to_the_budget(self):
        dpi = self.dpi(1200, 1800)
        assert dpi < 200
        assert (1200 / 72 * dpi) * (1800 / 72 * dpi) <= self.BUDGET

    def test_never_below_72_dpi(self):
        assert self.dpi(20000, 30000) == 72

    def test_an_empty_page_keeps_the_default(self):
        assert self.dpi(0, 0) == 200


class TestRescalePage:
    def test_every_box_moves_into_the_callers_frame(self):
        b = {
            "top_left": {"x": 100, "y": 200},
            "top_right": {"x": 300, "y": 200},
            "bottom_right": {"x": 300, "y": 250},
            "bottom_left": {"x": 100, "y": 250},
            "width": 200,
            "height": 50,
        }
        page = {
            "width": 1700,
            "height": 2200,
            "layout": [{"type": "text", "bounds": b}],
            "sections": [
                {
                    "page": 1,
                    "width": 1700,
                    "height": 2200,
                    "lines": [{"text": "a", "bounds": b, "words": [{"text": "a", "bounds": b}]}],
                }
            ],
        }
        out = rescale_page(page, 1224, 1584)
        assert (out["width"], out["height"]) == (1224, 1584)
        assert (out["sections"][0]["width"], out["sections"][0]["height"]) == (1224, 1584)
        for moved in (
            out["layout"][0]["bounds"],
            out["sections"][0]["lines"][0]["bounds"],
            out["sections"][0]["lines"][0]["words"][0]["bounds"],
        ):
            assert moved["top_left"] == {"x": 72, "y": 144}
            assert moved["bottom_right"] == {"x": 216, "y": 180}
            assert (moved["width"], moved["height"]) == (144, 36)

    def test_same_frame_is_untouched(self):
        page = {"width": 720, "height": 960, "sections": []}
        assert rescale_page(page, 720, 960) is page


class TestSpeakerAttribution:
    def test_max_overlap_beats_midpoint_on_interleaved_turns(self):
        """Overlap picks spk_1 (3.0 vs 1.5); midpoint 5.75 and NeMo's forward-only scan pick spk_0."""
        turns = [
            SpeakerTurn("spk_0", 0.0, 4.0),
            SpeakerTurn("spk_1", 4.0, 5.0),
            SpeakerTurn("spk_0", 5.0, 6.0),
            SpeakerTurn("spk_1", 6.0, 9.0),
        ]
        [word] = attribute_speakers([Word("mmhmm", 3.5, 8.0)], turns)
        assert word.speaker == "spk_1"

        midpoint = (3.5 + 8.0) / 2
        containing = [t.speaker for t in turns if t.start <= midpoint <= t.end]
        assert containing == ["spk_0"], "fixture must actually discriminate the two rules"

    def test_simple_word_lands_in_its_turn(self):
        turns = [SpeakerTurn("spk_0", 0.0, 5.0), SpeakerTurn("spk_1", 5.0, 10.0)]
        words = attribute_speakers([Word("hello", 1.0, 2.0), Word("world", 6.0, 7.0)], turns)
        assert [w.speaker for w in words] == ["spk_0", "spk_1"]

    def test_no_overlap_leaves_speaker_none_by_default(self):
        turns = [SpeakerTurn("spk_0", 10.0, 20.0)]
        [word] = attribute_speakers([Word("cough", 0.0, 1.0)], turns)
        assert word.speaker is None

    def test_fill_nearest_assigns_the_closest_turn(self):
        turns = [SpeakerTurn("spk_0", 10.0, 20.0), SpeakerTurn("spk_1", 100.0, 110.0)]
        [word] = attribute_speakers([Word("cough", 0.0, 1.0)], turns, fill_nearest=True)
        assert word.speaker == "spk_0"

    def test_long_turn_is_not_penalised(self):
        """IoU would divide by turn duration and pick the short turn. Overlap must not."""
        turns = [SpeakerTurn("spk_long", 0.0, 600.0), SpeakerTurn("spk_short", 5.0, 5.2)]
        [word] = attribute_speakers([Word("yes", 4.0, 5.1)], turns)
        assert word.speaker == "spk_long"


class TestGroupBySpeaker:
    def test_contiguous_words_collapse_into_one_chunk(self):
        words = [
            Word("the", 0.0, 0.5, "spk_0"),
            Word("total", 0.5, 1.0, "spk_0"),
            Word("forty", 1.2, 1.8, "spk_1"),
        ]
        chunks = group_by_speaker(words)
        assert len(chunks) == 2
        assert chunks[0] == {"speaker": "spk_0", "text": "the total", "timestamp": [0.0, 1.0]}
        assert chunks[1]["speaker"] == "spk_1"

    def test_speaker_alternation_splits_chunks(self):
        words = [
            Word("a", 0.0, 0.1, "spk_0"),
            Word("b", 0.1, 0.2, "spk_1"),
            Word("c", 0.2, 0.3, "spk_0"),
        ]
        assert len(group_by_speaker(words)) == 3


class TestReaderLayout:
    @staticmethod
    def _cell(category, text, bbox=(100, 100, 300, 140)):
        import json

        return json.dumps({"bbox": list(bbox), "category": category, "text": text})

    def _read(self, cells, sent=(1000, 800), page=(2000, 1600)):
        from PIL import Image

        return read_layout("[" + ", ".join(cells) + "]", sent, Image.new("RGB", page))

    def test_boxes_are_scaled_from_the_frame_the_reader_was_sent(self):
        [region], markdown = self._read([self._cell("Text", "TOTAL 166.20", (100, 50, 500, 120))])
        assert markdown == "TOTAL 166.20"
        assert region.bounds.as_dict()["top_left"] == {"x": 200, "y": 100}
        assert region.bounds.as_dict()["bottom_right"] == {"x": 1000, "y": 240}

    def test_a_small_page_is_not_read_as_a_thousand_grid(self):
        """Every value under 1000 is still pixels: a 720x960 receipt sent as itself."""
        [region], _ = self._read(
            [self._cell("Text", "centre", (310, 430, 410, 530))], sent=(720, 960), page=(720, 960)
        )
        assert region.bounds.as_dict()["top_left"] == {"x": 310, "y": 430}

    def test_page_furniture_is_a_region_but_not_document_text(self):
        regions, markdown = self._read(
            [
                self._cell("Page-header", "Journal of Things, Vol 3"),
                self._cell("Text", "Body text."),
                self._cell("Page-footer", "17"),
            ]
        )
        assert markdown == "Body text."
        assert [r.category for r in regions] == ["Page-header", "Text", "Page-footer"]

    def test_a_table_keeps_its_html_and_reads_as_plain_text(self):
        html = "<table><tr><td>Fees</td><td>&amp; costs</td></tr></table>"
        [region], markdown = self._read([self._cell("Table", html)])
        assert region.markup == html and region.text == "Fees & costs"
        assert markdown == html

    def test_pictures_carry_no_text(self):
        import json

        picture = json.dumps({"bbox": [0, 0, 500, 500], "category": "Picture"})
        assert self._read([picture]) == ([], "")

    def test_a_formula_is_fenced_once(self):
        _, markdown = self._read(
            [self._cell("Formula", "x^2 + y^2 = r^2"), self._cell("Formula", "$$a = b$$")]
        )
        assert markdown == "$$\nx^2 + y^2 = r^2\n$$\n\n$$\na = b\n$$"

    def test_output_cut_off_mid_cell_keeps_every_finished_cell(self):
        from PIL import Image

        raw = '[{"bbox": [1, 2, 30, 40], "category": "Title", "text": "Kept"}, {"bbox": [5, 6, 7'
        regions, markdown = read_layout(raw, (100, 100), Image.new("RGB", (100, 100)))
        assert markdown == "Kept" and len(regions) == 1

    def test_region_text_is_the_words_without_their_markdown(self):
        """Region text becomes each detected line's text: markup must not reach it."""
        regions, markdown = self._read(
            [
                self._cell("Text", "**TOTAL** 166.20 for $x^2$ items"),
                self._cell("List-item", "- see [the docs](http://a.b)"),
                self._cell("Section-header", "## Payment terms"),
            ]
        )
        assert [r.text for r in regions] == ["TOTAL 166.20 for x^2 items", "see the docs", "Payment terms"]
        assert markdown.startswith("**TOTAL** 166.20 for $x^2$ items")

    def test_a_dollar_sign_is_currency_unless_it_fences_math(self):
        regions, _ = self._read(
            [
                self._cell("Text", "**Total:** $166.20, was $180 and $5.00"),
                self._cell("Text", "where $\\beta$ and $x$ vary, $$a_1$$ and $$a = b$$"),
            ]
        )
        assert [r.text for r in regions] == [
            "Total: $166.20, was $180 and $5.00",
            "where \\beta and x vary, a_1 and a = b",
        ]

    def test_a_malformed_box_keeps_its_text(self):
        import json

        regions, markdown = self._read(
            [json.dumps({"bbox": [1, 2], "category": "Text", "text": "still here"})]
        )
        assert regions == [] and markdown == "still here"


class TestDocumentKinds:
    """Detected by content: ref-N temp files carry a suffix guessed from the MIME type."""

    def _kind(self, blob):
        return document_kind(blob)

    def test_pdf_detected_by_magic_bytes(self):
        assert self._kind(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n") == "pdf"

    def test_docx_detected_by_zip_plus_word_entry(self):
        assert self._kind(b"PK\x03\x04" + b"\x00" * 30 + b"word/document.xml") == "docx"

    def test_plain_zip_is_not_docx(self):
        assert self._kind(b"PK\x03\x04" + b"\x00" * 30 + b"xl/workbook.xml") == "image"

    def test_png_is_an_image(self):
        assert self._kind(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32) == "image"


class TestSectionPageFrame:
    """Bounds stay page-local as interfaze's (ocr/vocr.ts:594-605); stacking pages moved every later box."""

    def test_backfill_stamps_frame_without_moving_bounds(self):
        section = {
            "text": "x",
            "lines": [
                {
                    "text": "x",
                    "bounds": {
                        "top_left": {"x": 10, "y": 20},
                        "top_right": {"x": 90, "y": 20},
                        "bottom_right": {"x": 90, "y": 50},
                        "bottom_left": {"x": 10, "y": 50},
                        "width": 80,
                        "height": 30,
                    },
                }
            ],
        }
        [out] = backfill_sections([section], page=3, width=1240, height=1754)

        assert (out["page"], out["width"], out["height"]) == (3, 1240, 1754)
        assert out["lines"][0]["bounds"]["top_left"] == {"x": 10, "y": 20}

    def test_backfill_does_not_mutate_the_input(self):
        section = {"text": "x", "lines": []}
        backfill_sections([section], page=2, width=100, height=200)
        assert "page" not in section


class TestFileContentPart:
    """OpenAI's `file` part: interfaze's own harness sends audio this way, and it was silently ignored."""

    def _extract(self, part):
        return extract_from_messages(
            [{"role": "user", "content": [{"type": "text", "text": "transcribe"}, part]}]
        )

    def test_file_url_becomes_a_ref(self):
        refs, msgs = self._extract(
            {"type": "file", "file": {"filename": "call.wav", "file_url": "https://x.test/call.wav"}}
        )
        assert refs.resolve("ref-0") == "https://x.test/call.wav"
        assert "ref-0" in msgs[0]["content"][1]["text"]

    def test_data_uri_is_materialised(self):
        refs, _ = self._extract(
            {"type": "file", "file": {"filename": "a.txt", "file_data": "data:text/plain;base64,aGVsbG8="}}
        )
        assert refs.resolve("ref-0").endswith((".txt", ".bin")) or refs.resolve("ref-0")

    def test_bare_base64_uses_the_filename_for_mime(self):
        refs, _ = self._extract({"type": "file", "file": {"filename": "clip.wav", "file_data": "aGVsbG8="}})
        assert refs.refs["ref-0"].mime == "audio/x-wav" or "wav" in refs.refs["ref-0"].mime

    def test_empty_payload_is_left_alone_not_faked(self):
        refs, msgs = self._extract({"type": "file", "file": {"filename": "x.pdf"}})
        assert refs.refs == {}
        assert msgs[0]["content"][1]["type"] == "file"


class TestOneSectionPerPage:
    """interfaze consumers read sections[0] as the page; a second section per page misplaces every box."""

    def test_regions_collapse_into_a_single_section(self):
        a = OCRSection(text="Heading", lines=[OCRLine("Heading", box(10, 10, 200, 40), 0.9)])
        b = OCRSection(text="Body", lines=[OCRLine("Body", box(10, 60, 300, 90), 0.8)])
        page = collapse_to_page([a, b])
        assert page.text == "Heading\nBody"
        assert len(page.lines) == 2

    def test_lines_are_ordered_top_then_left(self):
        lower = OCRSection(text="b", lines=[OCRLine("b", box(10, 500, 100, 530), 0.9)])
        upper_right = OCRSection(text="a2", lines=[OCRLine("a2", box(400, 10, 500, 40), 0.9)])
        upper_left = OCRSection(text="a1", lines=[OCRLine("a1", box(10, 10, 100, 40), 0.9)])
        page = collapse_to_page([lower, upper_right, upper_left])
        assert [line.text for line in page.lines] == ["a1", "a2", "b"]

    def test_empty_region_text_is_not_joined_in(self):
        page = collapse_to_page(
            [
                OCRSection(text="real", lines=[]),
                OCRSection(text="   ", lines=[]),
            ]
        )
        assert page.text == "real"


class TestScanEnhancement:
    def _img(self, lo, hi):
        from PIL import Image

        img = Image.new("RGB", (40, 40), (hi, hi, hi))
        for x in range(0, 40, 2):
            for y in range(40):
                img.putpixel((x, y), (lo, lo, lo))
        return img

    def test_faded_scan_is_stretched(self):
        out = enhance_scan(self._img(90, 170))
        levels = out.convert("L").getextrema()
        assert levels[1] - levels[0] > 150, f"contrast not stretched: {levels}"

    def test_clean_page_is_untouched(self):
        original = self._img(0, 255)
        assert enhance_scan(original) is original


class TestDetectorRescale:
    def test_quad_scales_into_the_callers_frame(self):
        # A box found on a half-size copy must land in the same place on the full page.
        quad_small = [[100, 50], [200, 50], [200, 80], [100, 80]]
        sx = sy = 2.0
        scaled = [[x * sx, y * sy] for x, y in quad_small]
        bounds = to_bounds(scaled, src="quad", img_w=1600, img_h=1200)
        assert bounds.top_left.x == 200
        assert bounds.top_left.y == 100
        assert bounds.bottom_right.x == 400
        assert bounds.bottom_right.y == 160


class TestReadDocumentShape:
    """Both reader paths return (regions, markdown); the in-process one, unused on Modal, returned a list."""

    def test_every_return_goes_through_read_layout(self):
        source = MODELING.read_text()
        tree = ast.parse(source)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_read_document")
        returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value]
        assert len(returns) == 2, "expected the server return and the in-process return"
        for node in returns:
            assert (
                isinstance(node.value, ast.Call) and getattr(node.value.func, "attr", "") == "read_layout"
            ), f"_read_document returns something other than read_layout at line {node.lineno}"


@pytest.mark.skipif(
    not pathlib.Path("scripts/sync_hf.sh").exists(),
    reason="Hugging Face packaging lives with the model repo, not here",
)
class TestHuggingFacePackaging:
    """The generated HF repo must be self-contained: a module imported but never copied broke detection."""

    def test_every_sibling_module_reaches_the_user(self):
        """Siblings ship only if named by `from .x import` (trust_remote_code's pattern, used here)."""
        import re

        src = pathlib.Path("src/interfaze_lite")
        script = pathlib.Path("scripts/sync_hf.sh").read_text()
        todo, shipped = ["modeling_interfaze_lite"], set()
        while todo:
            name = todo.pop()
            if name in shipped:
                continue
            shipped.add(name)
            assert f"src/interfaze_lite/{name}.py" in script, f"scripts/sync_hf.sh does not copy {name}.py"
            text = (src / f"{name}.py").read_text()
            found = set(re.findall(r"^\s*import\s+\.(\S+)\s*$", text, re.M))
            found |= set(re.findall(r"^\s*from\s+\.(\S+)\s+import", text, re.M))
            for sibling in re.findall(r"^\s*from \. import ([\w, ]+)", text, re.M):
                for dep in re.split(r"[, ]+", sibling):
                    assert not dep or dep in found, (
                        f"{name}.py imports {dep} only as `from . import {dep}`, which "
                        "trust_remote_code does not ship"
                    )
            todo += [dep for dep in found if (src / f"{dep}.py").exists()]
        assert {"contracts", "grounding", "agent", "prompts"} <= shipped

    def test_the_published_config_is_bundled_and_names_no_component(self):
        """Components load from the bundle; naming upstream ids would say what the model is made of."""
        import json

        template = json.loads(pathlib.Path("hf_config.json.tmpl").read_text())
        assert template["bundle_weights"] is True
        assert not template.get("components")


class TestSourceIntegrity:
    """Modules must compile: a stale __pycache__ once hid conflict markers in brain.py from the suite."""

    def _shipped(self):
        return [p for p in [*pathlib.Path("src").rglob("*.py"), pathlib.Path("modal_app.py")] if p.exists()]

    def test_no_conflict_markers(self):
        for path in self._shipped():
            text = path.read_text()
            for marker in ("<" * 7, ">" * 7):
                assert marker not in text, f"{path} carries a conflict marker"

    def test_every_module_compiles(self):
        import ast

        for path in self._shipped():
            try:
                ast.parse(path.read_text(), filename=str(path))
            except SyntaxError as exc:
                raise AssertionError(f"{path} does not parse: {exc}") from exc


class TestErrorNormalisation:
    """Upstream error text is restated in capability terms; callers pass the ids, so these are invented."""

    VENDORS = ("acme/speech-xl-v3", "globex/segment-base", "initech/reader-2")

    def test_repo_ids_are_replaced(self):
        for vendor in self.VENDORS:
            out = normalise_error(f"{vendor} failed to load", self.VENDORS)
            assert "<component>" in out
            assert vendor not in out

    def test_bare_words_are_replaced(self):
        out = normalise_error("Initech reader blew up in native code", self.VENDORS)
        assert "initech" not in out.lower()

    def test_ordinary_errors_are_untouched(self):
        text = "request timed out after 600s"
        assert normalise_error(text, self.VENDORS) == text

    def test_no_identifiers_without_configuration(self):
        text = "acme/speech-xl-v3 failed"
        assert normalise_error(text, ()) == text

    def test_health_payloads_name_capabilities(self):
        for name in ("perception.py", "diarize.py"):
            source = pathlib.Path(f"src/interfaze_lite/services/{name}").read_text()
            health = source.split("def health(")[1].split("\n\n\n")[0]
            assert "DEFAULT_PIPELINE" not in health
            assert "capabilit" in health.lower() or "CAPABILITY" in health


class TestLayoutTextJoin:
    def _line(self, text, x1, y1, x2, y2):
        return OCRLine(
            text=text,
            bounds=Bounds(
                top_left=Point(x1, y1),
                top_right=Point(x2, y1),
                bottom_right=Point(x2, y2),
                bottom_left=Point(x1, y2),
                width=x2 - x1,
                height=y2 - y1,
            ),
            average_confidence=0.9,
            words=[],
        )

    def _block(self, kind, x1, y1, x2, y2):
        return {
            "type": kind,
            "score": 0.9,
            "bounds": {
                "top_left": {"x": x1, "y": y1},
                "top_right": {"x": x2, "y": y1},
                "bottom_right": {"x": x2, "y": y2},
                "bottom_left": {"x": x1, "y": y2},
                "width": x2 - x1,
                "height": y2 - y1,
            },
        }

    def test_blocks_come_in_the_readers_order_not_down_the_page(self):
        """Sorted by y, a paper's columns interleaved and the model dropped the left column's rest."""
        from interfaze_lite.contracts import attach_text_to_layout

        regions = [
            VLMRegion(t, box(*b))
            for t, b in (
                ("left top", (0, 100, 400, 300)),
                ("left foot", (0, 500, 400, 700)),
                ("right top", (500, 100, 900, 300)),
                ("right foot", (500, 500, 900, 700)),
            )
        ]
        blocks = [
            self._block("image", 0, 0, 900, 90),
            self._block("text", 0, 100, 400, 300),
            self._block("text", 500, 100, 900, 300),
            self._block("text", 0, 500, 400, 700),
            self._block("text", 500, 500, 900, 700),
        ]
        lines = [
            self._line(r.text, r.bounds.x1, r.bounds.y1 + 10, r.bounds.x2, r.bounds.y1 + 30) for r in regions
        ]
        out = attach_text_to_layout(blocks, lines, regions=regions)
        assert [b["text"] for b in out] == ["", "left top", "left foot", "right top", "right foot"]

    def test_each_block_gets_the_text_inside_it(self):
        lines = [
            self._line("Annual Report", 10, 10, 200, 40),
            self._line("Revenue rose", 10, 100, 300, 130),
            self._line("in every region.", 10, 140, 300, 170),
        ]
        blocks = [self._block("title", 0, 0, 400, 50), self._block("paragraph", 0, 90, 400, 180)]
        out = attach_text_to_layout(blocks, lines)
        assert out[0]["type"] == "title" and out[0]["text"] == "Annual Report"
        assert out[1]["type"] == "paragraph"
        assert out[1]["text"] == "Revenue rose in every region."
        assert out[1]["line_count"] == 2

    def test_text_outside_every_block_is_kept(self):
        lines = [self._line("boxed", 10, 10, 100, 40), self._line("orphan footnote", 10, 900, 300, 930)]
        out = attach_text_to_layout([self._block("title", 0, 0, 200, 50)], lines)
        assert out[-1]["type"] == "unclassified"
        assert out[-1]["text"] == "orphan footnote"

    def test_a_line_is_claimed_once(self):
        lines = [self._line("shared", 10, 10, 100, 40)]
        blocks = [self._block("title", 0, 0, 200, 50), self._block("paragraph", 0, 0, 200, 50)]
        out = attach_text_to_layout(blocks, lines)
        assert [b["line_count"] for b in out] == [1, 0]

    def _region(self, text, x1, y1, x2, y2):
        return VLMRegion(text=text, bounds=self._line("", x1, y1, x2, y2).bounds)

    def test_the_readers_text_wins_over_the_line_detectors(self):
        # The detector reads "evaluatereasoning" and "StructE- val"; the reader does not.
        lines = [
            self._line("models evaluatereasoning, StructE-", 10, 100, 300, 130),
            self._line("val and more", 10, 140, 300, 170),
        ]
        regions = [self._region("models evaluate reasoning, StructEval and more", 5, 95, 305, 175)]
        out = attach_text_to_layout([self._block("text", 0, 90, 400, 180)], lines, regions=regions)
        assert out[0]["text"] == "models evaluate reasoning, StructEval and more"
        assert out[0]["line_count"] == 2
        assert len(out) == 1

    def test_a_line_the_reader_missed_keeps_the_detectors_reading(self):
        lines = [self._line("Figure 1", 10, 10, 100, 40)]
        regions = [self._region("A paragraph.", 10, 100, 300, 170)]
        out = attach_text_to_layout([self._block("figure_title", 0, 0, 200, 50)], lines, regions=regions)
        assert [b["text"] for b in out] == ["Figure 1"]

    def test_the_readers_boxes_place_nothing(self):
        # A landscape form's reader boxes came back transposed (letterhead at x=509, y=44-1056).
        lines = [
            self._line("WESTVIEW ORTHOPAEDIC", 640, 40, 1270, 90),
            self._line("Patient Name: Sarah K. Miller", 400, 280, 870, 325),
        ]
        regions = [
            self._region("WESTVIEW ORTHOPAEDIC CLINIC", 509, 44, 734, 1056),
            self._region("Patient Name: Sarah K. Miller DOB: 1995-11-03", 379, 616, 734, 1100),
        ]
        out = attach_text_to_layout(
            [self._block("doc_title", 600, 30, 1300, 100), self._block("text", 390, 270, 900, 330)],
            lines,
            regions=regions,
        )
        assert [b["text"] for b in out] == ["WESTVIEW ORTHOPAEDIC", "Patient Name: Sarah K. Miller"]

    def test_side_by_side_columns_keep_their_own_lines(self):
        lines = [
            self._line("left column about apples", 10, 100, 190, 130),
            self._line("right column about rivers", 210, 100, 390, 130),
        ]
        blocks = [self._block("text", 0, 90, 200, 140), self._block("text", 200, 90, 400, 140)]
        regions = [
            self._region("right column about rivers.", 20, 95, 150, 135),
            self._region("left column about apples.", 10, 95, 180, 135),
        ]
        out = attach_text_to_layout(blocks, lines, regions=regions)
        by_x = {b["bounds"]["top_left"]["x"]: b["text"] for b in out}
        assert by_x == {0: "left column about apples.", 200: "right column about rivers."}

    def test_a_list_the_reader_kept_whole_is_split_across_its_item_blocks(self):
        # One List-Group region over three item blocks: each block gets its own item.
        lines = [
            self._line("1. firstitem", 10, 100, 300, 130),
            self._line("2. second item", 10, 150, 300, 180),
            self._line("3. third item", 10, 200, 300, 230),
        ]
        blocks = [
            self._block("text", 0, 95, 400, 135),
            self._block("text", 0, 145, 400, 185),
            self._block("text", 0, 195, 400, 235),
        ]
        markup = "<ol><li>1. first item</li><li>2. second item</li><li>3. third item</li></ol>"
        region = VLMRegion(
            text="1. first item 2. second item 3. third item",
            markup=markup,
            category="List-Group",
            bounds=self._line("", 10, 95, 300, 235).bounds,
        )
        out = attach_text_to_layout(blocks, lines, regions=[region])
        assert [b["text"] for b in out] == ["1. first item", "2. second item", "3. third item"]

    def test_a_list_that_does_not_line_up_is_not_repeated_as_unclassified(self):
        lines = [self._line("1. first", 10, 100, 300, 130), self._line("2. second", 10, 150, 300, 180)]
        blocks = [self._block("text", 0, 95, 400, 135), self._block("text", 0, 145, 400, 185)]
        region = VLMRegion(
            text="1. first 2. second 3. extra",
            markup="<ol><li>a</li><li>b</li><li>c</li></ol>",
            bounds=self._line("", 10, 95, 300, 185).bounds,
        )
        out = attach_text_to_layout(blocks, lines, regions=[region])
        assert [b["text"] for b in out] == ["1. first", "2. second"]
        assert len(out) == 2

    def test_tables_come_back_as_the_reader_produced_them(self):
        table = "<table><tr><td>Metric</td><td>Range</td></tr></table>"
        region = VLMRegion(
            text="Metric Range", markup=table, category="Table", bounds=self._line("", 700, 5, 900, 20).bounds
        )
        lines = [self._line("Metric Range", 10, 100, 300, 130)]
        out = attach_text_to_layout([self._block("table", 0, 90, 400, 180)], lines, regions=[region])
        assert out[0]["text"] == table

    def test_unboxed_reader_text_is_kept_once(self):
        lines = [self._line("orphan footnote", 10, 900, 300, 930)]
        regions = [self._region("orphan footnote", 5, 895, 305, 935)]
        out = attach_text_to_layout([self._block("title", 0, 0, 200, 50)], lines, regions=regions)
        assert out[-1]["type"] == "unclassified"
        assert out[-1]["text"] == "orphan footnote"
        assert len(out) == 2


class TestEveryReadPathClassifies:
    """Both read paths pass layout to `_assemble`; the PDF path once skipped it and no fixture noticed."""

    @staticmethod
    def _assemble_calls(func_name: str) -> list[ast.Call]:
        tree = ast.parse(MODELING.read_text())
        func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func_name)
        return [
            n
            for n in ast.walk(func)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_assemble"
        ]

    @pytest.mark.parametrize("path", ["_ocr_pdf", "_ocr_image"])
    def test_layout_is_passed(self, path):
        calls = self._assemble_calls(path)
        assert calls, f"{path} makes no _assemble call"
        for call in calls:
            passed = {kw.arg for kw in call.keywords}
            assert "layout" in passed, f"{path} drops layout"
            assert "layout_status" in passed, f"{path} drops layout_status"


class TestIdentifiersAreIntact:
    """No identifier literal may hold prose: a text-based edit once mangled the diarizer's repo id."""

    @pytest.mark.parametrize(
        "relative",
        [
            "src/interfaze_lite/config.py",
            "src/interfaze_lite/services/diarize.py",
            "src/interfaze_lite/configuration_interfaze_lite.py",
            "src/interfaze_lite/modeling_interfaze_lite.py",
            "modal_app.py",
        ],
    )
    def test_no_prose_in_identifier_literals(self, relative):
        if not pathlib.Path(relative).exists():
            pytest.skip(f"{relative} is not part of this checkout")
        tree = ast.parse(pathlib.Path(relative).read_text())
        bad = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            # One slash is a repo id `owner/name`; URLs and user-agents have several and may read as prose.
            and node.value.count("/") == 1
            and " " in node.value
            and not node.value.startswith(("http", "/", "#", "."))
            and not node.value.endswith("/")
            and "\n" not in node.value
            and len(node.value) < 80
        ]
        assert not bad, f"{relative}: identifier literals contain prose: {bad}"


class TestEmptyTaskTag:
    """An empty tag, as templates emit for an unset slot, routes nothing instead of a 400."""

    def test_empty_tag_is_no_task(self):
        assert extract_task([{"role": "system", "content": "<task></task>"}]) is None

    def test_whitespace_only_tag_is_no_task(self):
        assert extract_task([{"role": "system", "content": "<task>   </task>"}]) is None

    def test_a_named_task_still_routes(self):
        assert extract_task([{"role": "system", "content": "<task>ocr</task>"}]) == "ocr"

    def test_an_unknown_name_is_still_rejected(self):
        with pytest.raises(RequestError):
            extract_task([{"role": "system", "content": "<task>web_search</task>"}])


class TestEvalsReadTheKeysWeEmit:
    """Evals read the keys the tools emit: a renamed key once gave a clean table of zero detections."""

    def test_detection_evals_use_the_emitted_key(self):
        emitted = pathlib.Path("src/interfaze_lite/tools/detection.py").read_text()
        assert '"detected_objects": objects' in emitted, (
            "object_detection no longer emits detected_objects; update the evals"
        )

        stale = []
        for script in pathlib.Path("evals").rglob("*.py"):
            text = script.read_text()
            for line_no, line in enumerate(text.splitlines(), 1):
                # A legacy read paired with the current name is a deliberate fallback; a lone one is stale.
                if (
                    'get("objects")' in line
                    and "detected_objects" not in text[max(0, text.find(line) - 200) : text.find(line) + 200]
                ):
                    stale.append(f"{script.name}:{line_no}")
        assert not stale, f"evals read a key the tool no longer emits: {stale}"


class TestRoleContractMatchesInterfaze:
    """Roles match interfaze's, including `developer`, which recent OpenAI SDKs send instead of `system`."""

    def test_developer_is_accepted(self):
        validate_messages([{"role": "developer", "content": "be terse"}, {"role": "user", "content": "hi"}])

    def test_developer_becomes_system(self):
        out = normalise_roles(
            [{"role": "developer", "content": "be terse"}, {"role": "user", "content": "hi"}]
        )
        assert [m["role"] for m in out] == ["system", "user"]

    def test_a_task_tag_in_a_developer_turn_still_routes(self):
        """The reason normalising has to happen before anything reads a role."""
        messages = [
            {"role": "developer", "content": "<task>ocr</task>"},
            {"role": "user", "content": "read it"},
        ]
        assert extract_task(messages) is None, "precondition: raw developer is not seen"
        assert extract_task(normalise_roles(messages)) == "ocr"

    def test_other_roles_are_untouched(self):
        original = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "tool", "tool_call_id": "c1", "content": "{}"},
        ]
        assert normalise_roles(original) == original

    def test_an_unknown_role_is_still_rejected(self):
        with pytest.raises(RequestError):
            validate_messages([{"role": "wizard", "content": "hi"}])


class TestDetectionRunsInWorkerProcesses:
    """Detectors run in worker processes: threads shared Paddle's oneDNN cache and segfaulted under load."""

    @staticmethod
    def _body(func_name: str) -> str:
        tree = ast.parse(MODELING.read_text())
        func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func_name)
        return ast.unparse(func)

    @pytest.mark.parametrize(
        "func,call",
        [
            ("_detect_layout", "layout_pages"),
            ("_detect_text_lines", "line_rows"),
        ],
    )
    def test_calls_go_to_the_worker_processes(self, func, call):
        assert f"self._detection_runtime().run({call}" in self._body(func)

    def test_no_engine_is_built_in_this_process(self):
        source = MODELING.read_text()
        assert "PaddleOCR(" not in source and "LayoutDetection(" not in source


class TestModelInitIsComplete:
    """`__init__` assigns all the class reads: an edit once nested methods in it, orphaning `_dtype`."""

    @staticmethod
    def _init_assignments() -> set[str]:
        tree = ast.parse(MODELING.read_text())
        cls = next(
            n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "InterfazeLiteModel"
        )
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        names = {
            t.attr
            for node in ast.walk(init)
            if isinstance(node, ast.Assign)
            for t in node.targets
            if isinstance(t, ast.Attribute)
        }
        # `self._cache: dict[...] = {}` is an AnnAssign, not an Assign.
        names |= {
            node.target.attr
            for node in ast.walk(init)
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Attribute)
        }
        return names

    @pytest.mark.parametrize(
        "attribute",
        [
            "_cache",
            "_cache_lock",
            "_dtype",
            "_torch_device",
            "_runtime",
            "_runtime_lock",
        ],
    )
    def test_constructor_sets(self, attribute):
        assert attribute in self._init_assignments()

    def test_no_function_is_nested_inside_init(self):
        """A method defined inside `__init__` is what caused the truncation."""
        tree = ast.parse(MODELING.read_text())
        cls = next(
            n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "InterfazeLiteModel"
        )
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        nested = [n.name for n in init.body if isinstance(n, ast.FunctionDef)]
        assert not nested, f"methods defined inside __init__: {nested}"


class TestRecogniserOutput:
    def test_language_names_become_iso_codes(self):
        pytest.importorskip("transformers")
        from interfaze_lite.modeling_interfaze_lite import _language_code

        assert _language_code("english") == "en"
        assert _language_code("en") == "en"
        assert _language_code(None) is None

    def test_words_group_into_sentence_segments(self):
        pytest.importorskip("torch")
        from interfaze_lite.modeling_interfaze_lite import _segments

        words = [
            Word("Hi,", 0.0, 0.3),
            Word("there.", 0.3, 0.6),
            Word("Next", 2.0, 2.3),
            Word("one", 2.3, 2.6),
        ]
        spans = _segments(words)
        assert [(s.text, s.start, s.end) for s in spans] == [("Hi, there.", 0.0, 0.6), ("Next one", 2.0, 2.6)]


class TestSpeakerAlwaysNamed:
    def test_a_word_between_turns_takes_the_nearest_turn(self):
        turns = [SpeakerTurn("SPEAKER_00", 0.0, 10.0), SpeakerTurn("SPEAKER_01", 12.5, 13.0)]
        # Its midpoint is 0.5 s past the long turn and 2 s before the short one, whose centre is nearer.
        out = attribute_speakers([Word("gap", 10.2, 10.8)], turns, fill_nearest=True)
        assert out[0].speaker == "SPEAKER_00"

    def test_no_turns_at_all_is_still_a_named_speaker(self):
        assert attribute_speakers([Word("hi", 0, 1)], [], fill_nearest=True)[0].speaker == "SPEAKER_00"


class TestWordSplit:
    def _bounds(self, tl, tr, br, bl):
        return Bounds(
            top_left=Point(*tl),
            top_right=Point(*tr),
            bottom_right=Point(*br),
            bottom_left=Point(*bl),
            width=tr[0] - tl[0],
            height=bl[1] - tl[1],
        )

    def test_words_join_back_into_the_line(self):
        words = split_words("See back of receipt", self._bounds((0, 0), (190, 0), (190, 20), (0, 20)), 0.9)
        assert " ".join(w.text for w in words) == "See back of receipt"
        assert [w.bounds.top_left.x for w in words] == [0, 40, 90, 120]
        assert words[-1].bounds.top_right.x == 190
        assert all(set(w.as_dict()) == {"text", "bounds", "confidence"} for w in words)

    def test_a_tilted_line_keeps_tilted_words(self):
        words = split_words("ab cd", self._bounds((0, 0), (50, 10), (50, 30), (0, 20)), 0.9)
        assert words[1].bounds.top_right.y == 10 and words[0].bounds.top_left.y == 0

    def test_one_word_keeps_the_line_box(self):
        b = self._bounds((0, 0), (40, 0), (40, 10), (0, 10))
        assert split_words("Total", b, 0.8)[0].bounds == b


class TestDuplicateBoxes:
    def _el(self, label, x1, y1, x2, y2):
        return {
            "label": label,
            "bounds": Bounds(
                top_left=Point(x1, y1),
                top_right=Point(x2, y1),
                bottom_right=Point(x2, y2),
                bottom_left=Point(x1, y2),
                width=x2 - x1,
                height=y2 - y1,
            ).as_dict(),
        }

    def test_a_looped_box_collapses_to_the_first(self):
        loop = [self._el("sword", 508, 409, 933, 458)] + [
            self._el("sword", 508, 410 + i % 3, 933, 457) for i in range(43)
        ]
        assert len(suppress_duplicates(loop)) == 1

    def test_distinct_objects_and_labels_survive(self):
        els = [
            self._el("dog", 0, 0, 100, 100),
            self._el("dog", 300, 0, 400, 100),
            self._el("collar", 0, 0, 100, 100),
        ]
        assert len(suppress_duplicates(els)) == 3

    def test_a_ladder_of_abutting_identical_boxes_is_cut_to_its_first(self):
        """The same box stepped down the page by its own height: a loop, not a list."""
        before = [self._el("icon", 10, 10, 50, 30)]
        ladder = [self._el("icon", 513, 150 + 13 * i, 553, 163 + 13 * i) for i in range(20)]
        kept = cut_ladders(before + ladder)
        assert kept == before + ladder[:1]

    def test_a_ladder_whose_boxes_jitter_in_size_is_cut_too(self):
        """The screenshot demo's dock: widths 55, 55, 52, ... each abutting the last."""
        widths = [55, 55, 52, 55, 54, 53] * 3
        xs = [sum(widths[:i]) for i in range(len(widths) + 1)]
        ladder = [self._el("buttons", 20 + xs[i], 900, 20 + xs[i + 1], 960) for i in range(len(widths))]
        assert cut_ladders(ladder) == ladder[:1]

    def test_a_ladder_that_climbs_as_it_steps_is_cut(self):
        """A street photo's 40 "person" boxes: 11-13 units wide, each abutting the last and 2-5 higher."""
        person = self._el("person", 491, 487, 516, 586)
        widths, rises = [13, 12, 11, 13, 12, 13] * 4, [5, 2, 3, 2, 5, 2] * 4
        x, y, ladder = 531, 454, []
        for w, rise in zip(widths, rises, strict=True):
            ladder.append(self._el("person", x, y, x + w, y + 68))
            x, y = x + w, y - rise
        assert cut_ladders([person] + ladder) == [person, ladder[0]]

    def test_a_ladder_running_leftwards_is_cut(self):
        """Each box ending where the last began: 20 boxes 15 to 18 units wide."""
        widths = [17, 18, 17, 15, 18] * 4
        x, ladder = 983, []
        for w in widths:
            ladder.append(self._el("person", x - w, 654, x, 765))
            x -= w
        assert cut_ladders(ladder) == ladder[:1]

    def test_abutting_boxes_of_different_sizes_survive(self):
        """A tab bar: side by side, but each its own width."""
        widths = [60, 90, 45, 120, 70, 100, 50]
        xs = [sum(widths[:i]) for i in range(len(widths) + 1)]
        tabs = [self._el("tab", 10 + xs[i], 50, 10 + xs[i + 1], 80) for i in range(len(widths))]
        assert cut_ladders(tabs) == tabs

    def test_a_real_list_with_gaps_and_a_short_stack_survive(self):
        thumbnails = [self._el("icon", 63, 143 + 28 * i, 93, 169 + 28 * i) for i in range(7)]
        short_stack = [self._el("row", 300, 100 + 20 * i, 600, 120 + 20 * i) for i in range(5)]
        assert cut_ladders(thumbnails + short_stack) == thumbnails + short_stack


class TestSpeechWindows:
    RATE = 16_000

    def _speech(self, seconds):
        import numpy as np

        t = np.arange(int(seconds * self.RATE)) / self.RATE
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    def test_windows_are_cut_in_the_quiet_and_never_exceed_30_s(self):
        import numpy as np

        # Speech in 24 s stretches with a second of silence between them.
        gap = np.zeros(self.RATE, dtype=np.float32)
        audio = np.concatenate([self._speech(24), gap, self._speech(24), gap, self._speech(24)])
        windows = speech_windows(audio, self.RATE)
        assert all(e - s <= 30 * self.RATE for s, e in windows)
        assert all(a[1] <= b[0] for a, b in zip(windows, windows[1:], strict=False))
        # Every sample of speech is in a window: no cut fell inside a stretch of it.
        covered = np.zeros(len(audio), dtype=bool)
        for s, e in windows:
            covered[s:e] = True
        assert covered[np.abs(audio) > 0].all()

    def test_silence_around_speech_is_trimmed(self):
        import numpy as np

        audio = np.concatenate(
            [
                np.zeros(3 * self.RATE, dtype=np.float32),
                self._speech(10),
                np.zeros(3 * self.RATE, dtype=np.float32),
            ]
        )
        [(s, e)] = speech_windows(audio, self.RATE)
        assert 2.7 * self.RATE <= s <= 3 * self.RATE and 13 * self.RATE <= e <= 13.3 * self.RATE

    def test_a_silent_window_is_left_out(self):
        import numpy as np

        audio = np.concatenate(
            [self._speech(25), np.zeros(40 * self.RATE, dtype=np.float32), self._speech(25)]
        )
        windows = speech_windows(audio, self.RATE)
        assert len(windows) >= 2
        for s, e in windows:
            assert np.abs(audio[s:e]).max() > 0

    def test_short_audio_is_one_window(self):
        audio = self._speech(12)
        [(s, e)] = speech_windows(audio, self.RATE)
        assert s == 0 and len(audio) - e < 0.05 * self.RATE

    def test_a_batch_that_does_not_fit_is_freed_before_its_halves_run(self):
        """Halved inside the except, the failed batch stayed reachable and every half ran out of memory."""
        import weakref

        import numpy as np

        torch = pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from interfaze_lite.modeling_interfaze_lite import _transcribe_windows

        class Tensors:
            pass

        failed, alive_at_call, sizes = [], [], []

        def asr(inputs, **kwargs):
            alive_at_call.append(any(ref() is not None for ref in failed))
            sizes.append(len(inputs))
            if len(inputs) > 2:
                held = Tensors()  # what the failed batch had allocated, live in its frame
                failed.append(weakref.ref(held))
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return [{"chunks": [{"text": "hi", "timestamp": (0.0, 1.0)}]} for _ in inputs]

        gap = np.zeros(self.RATE, dtype=np.float32)
        audio = np.concatenate([np.concatenate([self._speech(24), gap]) for _ in range(8)])
        units, _ = _transcribe_windows(asr, audio, "en", words=True, batch_size=8)
        assert failed, "a batch ran out of memory"
        assert not any(alive_at_call)
        assert len(units) == len(speech_windows(audio, self.RATE))
        # Once two fit, the rest of the recording goes two at a time.
        assert sizes[:2] == [8, 4] and all(n <= 2 for n in sizes[2:])


class TestEstimatedWords:
    """Speakers joined to a recording too long to decode word timing for."""

    def test_a_segment_is_shared_among_its_words_by_length(self):
        words = estimate_words([Word("hi there everyone", 10.0, 13.0)])
        assert [w.text for w in words] == ["hi", "there", "everyone"]
        assert words[0].start == 10.0 and words[-1].end == pytest.approx(13.0)
        assert all(a.end == pytest.approx(b.start) for a, b in zip(words, words[1:], strict=False))
        assert words[2].end - words[2].start > words[0].end - words[0].start

    def test_a_change_of_speaker_inside_a_segment_splits_it(self):
        from interfaze_lite.contracts import (
            SpeakerTurn,
            Word,
            attribute_speakers,
            estimate_words,
            group_by_speaker,
        )

        segment = Word("yes I agree completely and what about Friday then", 0.0, 6.0)
        turns = [SpeakerTurn("A", 0.0, 3.5), SpeakerTurn("B", 3.5, 6.0)]
        chunks = group_by_speaker(attribute_speakers(estimate_words([segment]), turns))
        assert [c["speaker"] for c in chunks] == ["A", "B"]
        assert chunks[1]["text"].endswith("Friday then")

    def test_text_without_spaces_stays_one_unit(self):
        assert [(w.text, w.start, w.end) for w in estimate_words([Word("你好世界", 1.0, 2.0)])] == [
            ("你好世界", 1.0, 2.0)
        ]


class TestTranscriptChunks:
    def test_segments_are_cut_to_three_seconds_as_interfaze_cuts_them(self):
        """interfaze's own chunks for the sample clip, boundaries and words exactly."""
        segments = [
            {"text": "The little tales they tell are false", "timestamp": [0.0, 4.78]},
            {"text": "The door was barred, locked and bolted as well", "timestamp": [4.78, 9.48]},
            {"text": "Ripe pears are fit for a queen's table", "timestamp": [9.48, 13.06]},
        ]
        assert split_long_chunks(segments) == [
            {"text": "The little tales", "timestamp": [0.0, 2.39]},
            {"text": "they tell are false", "timestamp": [2.39, 4.78]},
            {"text": "The door was barred,", "timestamp": [4.78, 7.13]},
            {"text": "locked and bolted as well", "timestamp": [7.13, 9.48]},
            {"text": "Ripe pears are fit", "timestamp": [9.48, 11.27]},
            {"text": "for a queen's table", "timestamp": [11.27, 13.06]},
        ]

    def test_short_segments_words_and_untimed_chunks_are_left_alone(self):
        chunks = [
            {"text": "Hello there", "timestamp": [0.0, 2.5]},
            {"text": "word", "timestamp": [2.5, 6.0]},
            {"text": "no end time yet", "timestamp": [6.0, None]},
        ]
        assert split_long_chunks(chunks) == chunks
