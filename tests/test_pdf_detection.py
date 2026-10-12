"""object_detection on a PDF: each page rendered, searched, and its boxes tagged with it."""

import asyncio
import dataclasses
import os

import httpx
import pymupdf
import pytest
from PIL import Image

from interfaze_lite.config import Settings
from interfaze_lite.filerefs import FileRefs
from interfaze_lite.tools import pages
from interfaze_lite.tools.base import ToolContext
from interfaze_lite.tools.detection import _run_detection


def _pdf(path, count=3):
    doc = pymupdf.open()
    for i in range(count):
        doc.new_page(width=612, height=792).insert_text((72, 72), f"page {i + 1}")
    doc.save(path)
    doc.close()
    return str(path)


def _ctx(url, seen):
    async def ground(image, prompts, domain="ui"):
        with Image.open(image) as img:
            seen.append((image, img.size))
        # One box per page, on the 0-1000 grid the brain answers in.
        return {
            "gui_elements": [
                {
                    "label": prompts[0],
                    "bounds": {"top_left": {"x": 100, "y": 100}, "bottom_right": {"x": 500, "y": 500}},
                }
            ],
            "width": img.size[0],
            "height": img.size[1],
        }

    refs = FileRefs()
    refs.add(url, filename=os.path.basename(url))
    settings = dataclasses.replace(Settings(), detection_outlines=False)
    return ToolContext(refs=refs, http=httpx.AsyncClient(), settings=settings, ground=ground)


def test_every_page_is_searched_and_each_box_names_its_page(tmp_path):
    seen = []
    url = _pdf(tmp_path / "doc.pdf")
    result = asyncio.run(_run_detection({"file_ref_id": "ref-0", "prompts": ["logo"]}, _ctx(url, seen)))

    objects = result.model_facing["detected_objects"]
    assert [o["page"] for o in objects] == [1, 2, 3]
    # A letter page at 144 DPI, and the box in that page's pixels.
    assert all(size == (1224, 1584) for _, size in seen)
    assert objects[0]["bounds"]["top_left"] == {"x": 122, "y": 158}
    assert result.model_facing["pages"] == [{"page": n, "width": 1224, "height": 1584} for n in (1, 2, 3)]
    assert not any(os.path.exists(path) for path, _ in seen)


def test_page_range_limits_the_search(tmp_path):
    seen = []
    url = _pdf(tmp_path / "doc.pdf", count=5)
    result = asyncio.run(
        _run_detection({"file_ref_id": "ref-0", "prompts": ["logo"], "page_range": [2, 3]}, _ctx(url, seen))
    )
    assert [o["page"] for o in result.model_facing["detected_objects"]] == [2, 3]


def test_an_image_is_searched_as_before(tmp_path):
    seen = []
    url = str(tmp_path / "photo.png")
    Image.new("RGB", (800, 600), "white").save(url)
    result = asyncio.run(_run_detection({"file_ref_id": "ref-0", "prompts": ["dog"]}, _ctx(url, seen)))
    assert seen == [(url, (800, 600))]
    assert "page" not in result.model_facing["detected_objects"][0]
    assert "pages" not in result.model_facing


def test_a_url_that_does_not_say_pdf_is_not_fetched_here():
    refs = FileRefs()
    refs.add("https://x/photo.png")
    ctx = ToolContext(refs=refs, http=None, settings=Settings())
    # http is None: any fetch would raise, so returning None proves none happened.
    assert asyncio.run(pages.pdf_pages("https://x/photo.png", ctx)) is None


def test_a_range_past_the_end_says_so(tmp_path):
    with open(_pdf(tmp_path / "doc.pdf", count=2), "rb") as fh:
        data = fh.read()
    with pytest.raises(ValueError, match="2 pages"):
        pages._render(data, [5, 6])
