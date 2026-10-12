"""Locating things with the brain: the question it is asked, and how its answer is read.

Shared by the service (vLLM) and the transformers model (in-process), so a box found by one
is the box the other finds.
"""

from __future__ import annotations

import json
import re
from typing import Any

try:  # the interfaze_lite package, and the HF repo as trust_remote_code imports it
    from .contracts import _bounds_from_dict, iou, to_bounds
except ImportError:  # the files run flat, from a checkout
    from contracts import _bounds_from_dict, iou, to_bounds  # type: ignore

# A photo's box list is a few hundred tokens; 8192 only gave a degenerate reply room to run (233 boxes).
MAX_TOKENS = 1536
# "All buttons" lists every one at ~33 tokens each: 1536 stopped after 46, in the menu bar. interfaze
# allows 8192 on this model; a looping reply is cut down, not returned.
MAX_TOKENS_UI = 8192

# interfaze's GUI instruction word for word (object-detection-truss, dev_qwen38_gui_main.py), same model.
# Asking for each element's visible text instead boxed every piece of text and missed the buy box.
UI_INSTRUCTION = (
    "Locate every {phrase} in this UI screenshot.\n"
    "Respond with ONLY a JSON array and nothing else. Each element must be "
    '{{"bbox_2d": [x1, y1, x2, y2], "label": "<short name of the element>"}}.\n'
    "Coordinates are normalized to a 0-1000 grid, where (0,0) is the top-left corner "
    "and (1000,1000) is the bottom-right corner, regardless of the image's pixel size.\n"
    "Return every distinct match, not just the best one. "
    "If the screenshot contains no such element, respond with exactly []."
)


def max_tokens(domain: str) -> int:
    return MAX_TOKENS_UI if domain == "ui" else MAX_TOKENS


_OBJECT = re.compile(r'\{[^{}]*?"bbox_2d"\s*:\s*\[[^\]]*\][^{}]*?\}', re.S)


def prompt(prompts: list[str], domain: str = "ui") -> str:
    """The grounding question for `prompts` in a screenshot ("ui") or a photograph.

    The wording matters: asked for elements "in this UI screenshot", a photograph often returns [].
    """
    phrase = ", ".join(prompts)
    if domain == "ui":
        return UI_INSTRUCTION.format(phrase=phrase)
    # Ranking took RefCOCOg 77.5% -> 81.2%, but ScreenSpot-v2 fell to 51.2% with empty
    # replies 1.2% -> 18.8%, so screenshots skip it.
    return (
        f"Locate every {phrase} in this image.\n"
        "Label each match with the kind of object it is.\n"
        "Respond with ONLY a JSON array, each element "
        '{"bbox_2d": [x1,y1,x2,y2], "label": "<label>"}.\n'
        "Coordinates are normalized to a 0-1000 grid, (0,0) top-left, "
        "(1000,1000) bottom-right.\n"
        "Order the array so the element that best answers the request "
        "comes first; callers that want one answer take that one.\n"
        "Return every distinct match. If none, respond with exactly []."
    )


def _element(text: str, fallback_label: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
        bounds = to_bounds(obj["bbox_2d"], src="norm1000", img_w=1000, img_h=1000)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None
    if bounds.width <= 0 or bounds.height <= 0:
        return None
    return {
        "bounds": bounds.as_dict(),
        "label": obj.get("label") or fallback_label,
        "coordinate_space": "normalized_1000",
    }


def parse(raw: str, fallback_label: str, *, looped: bool = False) -> list[dict[str, Any]]:
    """The boxes in a grounding reply, on the 0-1000 grid, repeats removed.

    The grid is resolution-independent: `to_pixels` places the boxes on the image.
    A reply that `looped` also loses every box it repeated: those are the loop.
    """
    elements = [e for m in _OBJECT.finditer(raw or "") if (e := _element(m.group(0), fallback_label))]
    if looped:
        elements = _unrepeated(elements)
    return suppress_duplicates(cut_ladders(elements))


def _unrepeated(elements: list[dict[str, Any]], threshold: float = 0.85) -> list[dict[str, Any]]:
    """The boxes a looping reply gave once; what it went round and round is the loop, not a find."""
    repeated: set[int] = set()
    for i in range(len(elements)):
        j = next((j for j in range(i) if _same_box(elements[i], elements[j], threshold)), None)
        if j is not None:
            repeated.update((i, j))
    return [e for i, e in enumerate(elements) if i not in repeated]


# Wasted boxes (repeats, or off the grid) before a reply counts as looping; healthy lists repeat now and then.
LOOP_REPEATS = 10


class LoopWatch:
    """Reads a grounding reply as it streams and says when it has started looping.

    Stopping at LOOP_REPEATS gives back a minute of nothing (results.md, "GUI grounding looped").
    """

    def __init__(self, repeats: int = LOOP_REPEATS):
        self.repeats = repeats
        self.raw = ""
        self.wasted = 0
        self._pos = 0
        self._kept: list[dict[str, Any]] = []

    def feed(self, text: str) -> bool:
        """Add the next piece of the reply; True once it is looping."""
        self.raw += text
        for match in _OBJECT.finditer(self.raw, self._pos):
            self._pos = match.end()
            element = _element(match.group(0), "")
            if element is None or any(_same_box(element, k) for k in self._kept):
                self.wasted += 1
            else:
                self._kept.append(element)
        return self.wasted >= self.repeats


def _bounds_dict(x1: int, y1: int, x2: int, y2: int) -> dict[str, Any]:
    return {
        "top_left": {"x": x1, "y": y1},
        "top_right": {"x": x2, "y": y1},
        "bottom_right": {"x": x2, "y": y2},
        "bottom_left": {"x": x1, "y": y2},
        "width": x2 - x1,
        "height": y2 - y1,
    }


def tiles(width: int, height: int, overlap: float = 0.1) -> list[tuple[int, int, int, int]]:
    """A screenshot as four overlapping quarters, (x1, y1, x2, y2) in its pixels.

    What a looping screenshot is grounded in instead: fewer elements, seen at higher resolution.
    The overlap keeps an element on a seam whole in one of them.
    """
    half_w, half_h = width / 2, height / 2
    pad_w, pad_h = width * overlap / 2, height * overlap / 2
    xs = [(0, round(half_w + pad_w)), (round(half_w - pad_w), width)]
    ys = [(0, round(half_h + pad_h)), (round(half_h - pad_h), height)]
    return [(x1, y1, x2, y2) for y1, y2 in ys for x1, x2 in xs]


def from_tile(
    elements: list[dict[str, Any]], tile: tuple[int, int, int, int], width: int, height: int, margin: int = 5
) -> list[dict[str, Any]]:
    """Boxes on a tile's 0-1000 grid, moved onto the whole screenshot's grid.

    A box against a seam is dropped: the crop cut it in half, and the overlap shows it whole next door.
    """
    x1, y1, x2, y2 = tile
    tw, th = x2 - x1, y2 - y1

    def on_seam(b):
        return (
            (x1 > 0 and b["top_left"]["x"] <= margin)
            or (x2 < width and b["bottom_right"]["x"] >= 1000 - margin)
            or (y1 > 0 and b["top_left"]["y"] <= margin)
            or (y2 < height and b["bottom_right"]["y"] >= 1000 - margin)
        )

    def point(p):
        return {
            "x": round((x1 + p["x"] / 1000 * tw) / width * 1000),
            "y": round((y1 + p["y"] / 1000 * th) / height * 1000),
        }

    out = []
    for element in elements:
        if on_seam(element["bounds"]):
            continue
        tl, br = point(element["bounds"]["top_left"]), point(element["bounds"]["bottom_right"])
        out.append({**element, "bounds": _bounds_dict(tl["x"], tl["y"], br["x"], br["y"])})
    return out


def cut_ladders(elements: list[dict[str, Any]], run: int = 6) -> list[dict[str, Any]]:
    """Drop a stack of identical boxes each starting exactly where the last one ends; the first is kept.

    A looping reply's other shape: one box stepped across the page by its own size, any direction,
    with drift and a quarter (or three units) of jitter (results.md, "Ladders"). Real lists have gaps.
    """

    def edges(e):
        b = e["bounds"]
        return (b["top_left"]["x"], b["top_left"]["y"], b["bottom_right"]["x"], b["bottom_right"]["y"])

    def alike(u, v):
        return abs(u - v) <= max(3, 0.25 * max(u, v))

    def stacked(a, b):
        ax1, ay1, ax2, ay2 = edges(a)
        bx1, by1, bx2, by2 = edges(b)
        w, h = ax2 - ax1, ay2 - ay1
        if not (alike(w, bx2 - bx1) and alike(h, by2 - by1)):
            return False
        sideways = abs(bx1 - ax2) <= 1 or abs(bx2 - ax1) <= 1
        upright = abs(by1 - ay2) <= 1 or abs(by2 - ay1) <= 1
        return (sideways and abs(by1 - ay1) <= max(1, 0.15 * h)) or (
            upright and abs(bx1 - ax1) <= max(1, 0.15 * w)
        )

    drop: set[int] = set()
    start = 0
    for i in range(1, len(elements) + 1):
        if i < len(elements) and stacked(elements[i - 1], elements[i]):
            continue
        if i - start >= run:
            drop.update(range(start + 1, i))
        start = i
    return [e for i, e in enumerate(elements) if i not in drop]


def suppress_duplicates(elements: list[dict[str, Any]], threshold: float = 0.85) -> list[dict[str, Any]]:
    """Drop boxes that repeat an earlier box of the same label, as detectors do; replies are best-first."""
    kept: list[dict[str, Any]] = []
    for element in elements:
        if not any(_same_box(element, k, threshold) for k in kept):
            kept.append(element)
    return kept


def _same_box(a: dict[str, Any], b: dict[str, Any], threshold: float = 0.85) -> bool:
    """One element given twice: the same label on nearly the same box."""
    return (
        a["label"] == b["label"]
        and iou(_bounds_from_dict(a["bounds"]), _bounds_from_dict(b["bounds"])) >= threshold
    )


def trim_to_content(
    elements: list[dict[str, Any]], image, tolerance: int = 12, share: float = 1.0
) -> list[dict[str, Any]]:
    """Screenshot boxes with their plain-background margins cut off, on the 0-1000 grid.

    Each side is trimmed against the colour just outside it: an edge row or column is margin while
    `share` of it is that colour. A box only shrinks; one with nothing in it is dropped (results.md,
    "GUI boxes too wide"). `share` is all of it: 0.9 trimmed through thin borders, 0.98 lost underlines.
    """
    import numpy as np

    pixels = np.asarray(image.convert("RGB")).astype(np.int16)
    h, w = pixels.shape[:2]
    out = []
    for element in elements:
        b = element["bounds"]
        x1, y1 = max(0, round(b["top_left"]["x"] / 1000 * w)), max(0, round(b["top_left"]["y"] / 1000 * h))
        x2, y2 = (
            min(w, round(b["bottom_right"]["x"] / 1000 * w)),
            min(h, round(b["bottom_right"]["y"] / 1000 * h)),
        )
        if x2 - x1 <= 2 or y2 - y1 <= 2:
            out.append(element)
            continue
        t = _trim_box(pixels, x1, y1, x2, y2, tolerance, share)
        if t is None:
            continue
        if t == (x1, y1, x2, y2):
            out.append(element)  # nothing to trim: returned as it came, not re-rounded
            continue
        tl = {"x": round(t[0] / w * 1000), "y": round(t[1] / h * 1000)}
        br = {"x": round(t[2] / w * 1000), "y": round(t[3] / h * 1000)}
        out.append({**element, "bounds": _bounds_dict(tl["x"], tl["y"], br["x"], br["y"])})
    return out


def _trim_box(pixels, x1: int, y1: int, x2: int, y2: int, tolerance: int, share: float, ring: int = 4):
    """The box [x1, x2) x [y1, y2) with its margins trimmed, or None if all of it is margin."""
    import numpy as np

    h, w = pixels.shape[:2]

    def outside(band):
        # The colour of a strip beside the box, if one colour holds most of it.
        around = band.reshape(-1, 3)
        if not len(around):
            return None
        colour = np.median(around, axis=0)
        return colour if (np.abs(around - colour).max(axis=1) <= tolerance).mean() >= 0.6 else None

    def plain(strip, colour) -> bool:
        return colour is not None and (np.abs(strip - colour).max(axis=1) <= tolerance).mean() >= share

    left = outside(pixels[y1:y2, max(0, x1 - ring) : x1])
    while x1 < x2 and plain(pixels[y1:y2, x1], left):
        x1 += 1
    right = outside(pixels[y1:y2, x2 : min(w, x2 + ring)])
    while x1 < x2 and plain(pixels[y1:y2, x2 - 1], right):
        x2 -= 1
    top = outside(pixels[max(0, y1 - ring) : y1, x1:x2])
    while y1 < y2 and plain(pixels[y1, x1:x2], top):
        y1 += 1
    bottom = outside(pixels[y2 : min(h, y2 + ring), x1:x2])
    while y1 < y2 and plain(pixels[y2 - 1, x1:x2], bottom):
        y2 -= 1
    return (x1, y1, x2, y2) if x1 < x2 and y1 < y2 else None


def to_pixels(elements: list[dict], width: int | None, height: int | None) -> list[dict]:
    """Move boxes from the 0-1000 grid onto a `width` x `height` image.

    Every caller gets pixels: read as pixels, grid boxes once scored gui_detection 3.8% on ScreenSpot-v2.
    """
    if not width or not height:
        return elements
    out = []
    for element in elements:
        top_left = element["bounds"]["top_left"]
        bottom_right = element["bounds"]["bottom_right"]
        x1, y1 = round(top_left["x"] / 1000 * width), round(top_left["y"] / 1000 * height)
        x2, y2 = round(bottom_right["x"] / 1000 * width), round(bottom_right["y"] / 1000 * height)
        out.append(
            {k: v for k, v in element.items() if k != "coordinate_space"}
            | {"bounds": _bounds_dict(x1, y1, x2, y2)}
        )
    return out
