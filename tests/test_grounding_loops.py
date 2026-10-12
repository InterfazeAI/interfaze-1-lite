"""A grounding reply that loops: stopped when it starts, and the screenshot grounded in quarters."""

import asyncio
import base64
import dataclasses
import io
import json
import random

import httpx
from PIL import Image, ImageDraw

from interfaze_lite import grounding
from interfaze_lite.brain import BrainClient
from interfaze_lite.config import Settings


def _box(x1, y1, x2, y2, label="buttons"):
    return json.dumps({"bbox_2d": [x1, y1, x2, y2], "label": label})


# The screenshot demo's reply: boxes stepped across the search bar, then the same again.
LADDER = [_box(440 + 38 * i, 52, 462 + 38 * i, 87) for i in range(15)]
LOOPING = "[" + ", ".join(LADDER * 20) + "]"
HEALTHY = (
    "["
    + ", ".join(_box(50 + 30 * i, 100 + 20 * (i % 5), 70 + 30 * i, 115 + 20 * (i % 5)) for i in range(30))
    + "]"
)


def _stream(text, size=7):
    watch = grounding.LoopWatch()
    for i in range(0, len(text), size):
        if watch.feed(text[i : i + size]):
            return watch, True
    return watch, False


def test_a_looping_reply_is_stopped_at_the_tenth_repeat():
    watch, looped = _stream(LOOPING)
    assert looped and watch.wasted == grounding.LOOP_REPEATS
    assert len(grounding._OBJECT.findall(watch.raw)) == 15 + grounding.LOOP_REPEATS
    assert len(watch.raw) < len(LOOPING) / 10


def test_a_healthy_reply_runs_to_the_end():
    watch, looped = _stream(HEALTHY)
    assert not looped and watch.raw == HEALTHY


def test_the_quarters_cover_the_screenshot_and_overlap():
    quarters = grounding.tiles(3600, 2250)
    assert len(quarters) == 4
    assert min(t[0] for t in quarters) == 0 and max(t[2] for t in quarters) == 3600
    assert min(t[1] for t in quarters) == 0 and max(t[3] for t in quarters) == 2250
    left, right = quarters[0], quarters[1]
    assert left[2] > right[0], "the halves overlap, so an element on the seam is whole in one"


def test_a_box_on_a_quarter_lands_on_the_whole_screenshot():
    # The middle of the bottom-right quarter's grid is that quarter's middle on the page.
    tile = grounding.tiles(2000, 1000)[3]
    [moved] = grounding.from_tile(
        [{"label": "x", "bounds": {"top_left": {"x": 500, "y": 500}, "bottom_right": {"x": 500, "y": 500}}}],
        tile,
        2000,
        1000,
    )
    assert moved["bounds"]["top_left"] == {
        "x": round((tile[0] + tile[2]) / 2 / 2000 * 1000),
        "y": round((tile[1] + tile[3]) / 2 / 1000 * 1000),
    }


def test_a_box_cut_by_a_seam_is_left_to_the_tile_that_sees_it_whole():
    tiles = grounding.tiles(2000, 1000)

    def element(x1, y1, x2, y2):
        return {"label": "b", "bounds": {"top_left": {"x": x1, "y": y1}, "bottom_right": {"x": x2, "y": y2}}}

    # Top-right quarter: its left edge is a seam, its right and top edges are the page's.
    sliver, inside, at_page_edge = (
        element(0, 300, 40, 340),
        element(300, 300, 400, 340),
        element(950, 0, 1000, 40),
    )
    kept = grounding.from_tile([sliver, inside, at_page_edge], tiles[1], 2000, 1000)
    assert len(kept) == 2, "only the box against the seam goes"


def _png(width, height, elements=()):
    image = Image.new("RGB", (width, height), "white")
    for box in elements:
        ImageDraw.Draw(image).rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill="navy")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _element(x1, y1, x2, y2, label="b"):
    return {"label": label, "bounds": {"top_left": {"x": x1, "y": y1}, "bottom_right": {"x": x2, "y": y2}}}


def _edges(e):
    b = e["bounds"]
    return (b["top_left"]["x"], b["top_left"]["y"], b["bottom_right"]["x"], b["bottom_right"]["y"])


def _page(*elements, fill="gold"):
    image = Image.new("RGB", (1000, 1000), "white")
    for box in elements:
        ImageDraw.Draw(image).rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=fill)
    return image


def test_a_box_too_wide_is_trimmed_to_its_button():
    """Add to cart, boxed from the white page beside it."""
    [trimmed] = grounding.trim_to_content([_element(300, 400, 900, 460)], _page((650, 400, 900, 460)))
    assert _edges(trimmed) == (650, 400, 900, 460)


def test_a_box_already_on_its_element_is_untouched():
    box = _element(650, 400, 900, 460)
    assert grounding.trim_to_content([box], _page((650, 400, 900, 460))) == [box]


def test_a_box_with_nothing_in_it_is_dropped():
    assert grounding.trim_to_content([_element(100, 100, 300, 200)], _page((650, 400, 900, 460))) == []


def test_a_box_running_off_the_card_into_the_page_stops_at_the_input():
    """A Google Form input boxed past its card: the half-white, half-grey ring beat the background test."""
    image = Image.new("RGB", (1000, 1000), (237, 237, 237))
    draw = ImageDraw.Draw(image)
    draw.rectangle([300, 300, 699, 499], fill="white")  # the card
    draw.rectangle([320, 420, 400, 440], fill=(90, 90, 90))  # "Your answer"
    draw.line([317, 455, 640, 455], fill=(225, 225, 225))  # the input's underline
    [trimmed] = grounding.trim_to_content([_element(320, 410, 900, 456)], image)
    # The card's own white is the input's to keep: past the underline it is still the field.
    assert _edges(trimmed) == (320, 420, 700, 456)


def test_a_wide_box_is_not_trimmed_into_its_text():
    """Text's top rows, under 2% of a 1,228 px row, read as margin and the trim cut through the text."""
    image = Image.new("RGB", (1000, 1000), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([100, 420, 102, 424], fill=(90, 90, 90))  # an ascender: 3 px of the row
    draw.rectangle([100, 425, 150, 440], fill=(90, 90, 90))  # the rest of the text
    draw.line([100, 455, 640, 455], fill=(150, 150, 150))  # the input's underline
    [trimmed] = grounding.trim_to_content([_element(100, 400, 900, 456)], image)
    # The text's top is kept, and the underline -- one row of 56 -- still bounds the right.
    assert _edges(trimmed) == (100, 420, 641, 456)


def test_a_box_on_a_busy_background_is_left_alone():
    image = Image.frombytes("RGB", (1000, 1000), random.Random(0).randbytes(3_000_000))
    box = _element(100, 100, 300, 200)
    assert grounding.trim_to_content([box], image) == [box]


def _sse(text):
    return (
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": {"content": text}}]})
        + "\n\ndata: "
        + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        + "\n\ndata: [DONE]\n\n"
    )


def test_a_screenshot_that_loops_is_grounded_in_quarters():
    seen = []

    def handler(request):
        payload = json.loads(request.content)
        image = payload["messages"][0]["content"][0]["image_url"]["url"]
        with Image.open(io.BytesIO(base64.b64decode(image.split(",", 1)[1]))) as img:
            seen.append(img.size)
        # The whole screenshot loops; each quarter answers with one button in its middle.
        text = LOOPING if len(seen) == 1 else "[" + _box(400, 400, 600, 600) + "]"
        return httpx.Response(200, text=_sse(text), headers={"content-type": "text/event-stream"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            # Under the screenshot's 2 MP, so the whole is shrunk but each quarter fits at full size.
            settings = dataclasses.replace(Settings(), ground_max_pixels=700_000)
            # A button in the middle of each quarter, where each quarter's reply boxes it.
            buttons = [
                (440, 220, 660, 330),
                (1340, 220, 1560, 330),
                (440, 670, 660, 780),
                (1340, 670, 1560, 780),
            ]
            return await BrainClient(http, settings).ground(_png(2000, 1000, buttons), ["buttons"])

    result = asyncio.run(run())
    assert len(seen) == 5 and seen[0][0] * seen[0][1] <= 700_000
    assert all(size == (1100, 550) for size in seen[1:]), "quarters come from the original"
    # None of the ladder survives; one button per quarter, each at its quarter's middle.
    centres = sorted(
        (
            round((e["bounds"]["top_left"]["x"] + e["bounds"]["bottom_right"]["x"]) / 2),
            round((e["bounds"]["top_left"]["y"] + e["bounds"]["bottom_right"]["y"]) / 2),
        )
        for e in result["gui_elements"]
    )
    assert [(round(x / 5) * 5, round(y / 5) * 5) for x, y in centres] == [
        (275, 275),
        (275, 725),
        (725, 275),
        (725, 725),
    ]
    assert (result["width"], result["height"]) == (2000, 1000)


def test_a_photograph_that_loops_keeps_only_what_it_found_before_the_loop():
    calls = []
    found = [_box(100, 100, 300, 400, "dog"), _box(500, 200, 700, 500, "dog")]

    def handler(request):
        calls.append(request)
        reply = "[" + ", ".join(found + LADDER * 20) + "]"
        return httpx.Response(200, text=_sse(reply), headers={"content-type": "text/event-stream"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await BrainClient(http, Settings()).ground(_png(800, 600), ["dog"], domain="object")

    result = asyncio.run(run())
    assert len(calls) == 1, "only screenshots are grounded again in quarters"
    assert [e["label"] for e in result["gui_elements"]] == ["dog", "dog"]


def test_a_reply_that_did_not_loop_keeps_its_repeats_to_the_usual_filter():
    reply = "[" + ", ".join([_box(100, 100, 300, 400, "dog")] * 2 + [_box(500, 200, 700, 500, "dog")]) + "]"
    assert len(grounding.parse(reply, "dog")) == 2
    assert len(grounding.parse(reply, "dog", looped=True)) == 1


def test_a_screenshot_that_cannot_be_read_again_keeps_its_boxes():
    """The trim reads the image a second time; a failure there must not cost the boxes."""

    def handler(request):
        if request.method == "GET":
            return httpx.Response(
                200 if not getattr(handler, "served", False) else 404,
                content=base64.b64decode(_png(800, 600).split(",", 1)[1]),
            )
        return httpx.Response(
            200,
            text=_sse("[" + _box(100, 100, 300, 400) + "]"),
            headers={"content-type": "text/event-stream"},
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = BrainClient(http, Settings())
            original = client._image_bytes

            async def once(url):
                data = await original(url)
                handler.served = True
                return data

            client._image_bytes = once
            return await client.ground("https://x.test/shot.png", ["buttons"])

    result = asyncio.run(run())
    assert len(result["gui_elements"]) == 1


def test_a_small_photograph_is_seen_at_the_budget_and_boxed_on_the_original():
    """At 0.33 MP the crowd demo's cameras were all boxed about 5% too high."""
    seen = []

    def handler(request):
        image = json.loads(request.content)["messages"][0]["content"][0]["image_url"]["url"]
        with Image.open(io.BytesIO(base64.b64decode(image.split(",", 1)[1]))) as img:
            seen.append(img.size)
        return httpx.Response(
            200,
            text=_sse("[" + _box(100, 100, 300, 400, "dog") + "]"),
            headers={"content-type": "text/event-stream"},
        )

    async def run(domain):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await BrainClient(http, Settings()).ground(_png(400, 300), ["dog"], domain=domain)

    result = asyncio.run(run("object"))
    budget = Settings().object_ground_max_pixels
    assert 0.95 * budget <= seen[0][0] * seen[0][1] <= budget
    assert (result["width"], result["height"]) == (400, 300)
    # A screenshot is never enlarged: that is where loops start.
    asyncio.run(run("ui"))
    assert seen[1] == (400, 300)
