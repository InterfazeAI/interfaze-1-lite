"""Locating things: objects in photographs, elements in screenshots."""

from __future__ import annotations

import asyncio
import re

from ..grounding import _bounds_dict
from ..grounding import to_pixels as _to_pixels
from . import pages as _pages
from .base import Tool, ToolContext, ToolResult, _post, log
from .ocr import _read as _read_document


async def _run_detection(args: dict, ctx: ToolContext) -> ToolResult:
    """Boxes, labels and masks.

    Boxes come from the brain, already resident under vLLM: no extra VRAM, and none of the
    transformers-version skew that makes the alternate detector unloadable beside a 5.x brain.
    Masks are a second call to the segmenter, which is plain PyTorch.
    """
    url = ctx.refs.resolve(args["file_ref_id"])
    prompts = [p for p in (args.get("prompts") or []) if p and p.strip()]
    if not prompts:
        return ToolResult(
            model_facing={
                "error": "object_detection needs at least one prompt naming what to find.",
                "message": "Retry the tool call with concrete prompts, e.g. ['person', 'car'].",
            }
        )

    if ctx.ground is None:
        raise RuntimeError("object_detection requires a grounding callable on the context")
    want_masks = bool(args.get("return_masks", False))

    pages = await _pages.pdf_pages(url, ctx, args.get("page_range"))
    if pages is None:
        objects = await _detect_via_brain(url, prompts, ctx, want_masks=want_masks)
        extra: dict = {}
    else:
        # A PDF is searched page by page, all at once; each box is in its page's pixels.
        try:
            found = await asyncio.gather(
                *(_detect_via_brain(page.path, prompts, ctx, want_masks=want_masks) for page in pages)
            )
        finally:
            _pages.cleanup(pages)
        objects = [
            {**o, "page": page.number}
            for page, page_objects in zip(pages, found, strict=True)
            for o in page_objects
        ]
        extra = {"pages": [{"page": p.number, "width": p.width, "height": p.height} for p in pages]}

    # Given an empty result alone, the model asked again with near-identical prompts, to
    # the step cap.
    nothing = (
        {}
        if objects
        else {
            "message": (
                "Nothing matching these prompts was found. Do not call object_detection again with "
                "the same or similar prompts; tell the user none were found."
            )
        }
    )
    return ToolResult(
        # Masks are megabytes of base64, so they reach the caller through precontext only.
        model_facing={
            "detected_objects": [
                # The outline is small enough to read and answers questions a box cannot
                # -- which way something faces, whether two things touch.
                {k: o[k] for k in ("page", "bounds", "label", "polygon") if k in o}
                for o in objects
            ],
            **extra,
            **nothing,
        },
        # The contract's key: under `objects`, a consumer reading `detected_objects` found nothing.
        full={"detected_objects": objects, **extra},
    )


async def _detect_via_brain(
    url: str, prompts: list[str], ctx: ToolContext, *, want_masks: bool = False
) -> list[dict]:
    """Ground with the brain, then put the boxes in pixel space.

    The segmenter runs for outlines (`Settings.detection_outlines`) and for masks, which only
    callers asking for them get.
    """
    grounded = await ctx.ground(url, prompts, domain="object")
    elements = grounded.get("gui_elements") or []
    if not elements:
        return []
    pixel_boxes = _to_pixels(elements, grounded.get("width"), grounded.get("height"))

    # The 0-1000 grid: /segment rescales it against the image as perception fits it (`_fit_for_ocr`).
    boxes_norm = [
        [
            e["bounds"]["top_left"]["x"],
            e["bounds"]["top_left"]["y"],
            e["bounds"]["bottom_right"]["x"],
            e["bounds"]["bottom_right"]["y"],
        ]
        for e in elements
    ]

    masks: list[str] = []
    outlines: list = []
    if want_masks or ctx.settings.detection_outlines:
        try:
            seg = await _post(
                ctx,
                ctx.settings.perception_url,
                "/segment",
                {
                    "url": url,
                    "boxes": boxes_norm,
                    "normalized": True,
                    "return_masks": want_masks,
                    "return_outlines": ctx.settings.detection_outlines,
                },
            )
            masks = seg.get("masks") or []
            outlines = seg.get("outlines") or []
        except Exception as exc:
            # Outlines and masks are extras; losing them must not cost the detections.
            log.warning(
                "segmentation failed (%s); returning boxes only", type(exc).__name__ if ctx.zdr else exc
            )

    objects: list[dict] = []
    for i, element in enumerate(pixel_boxes):
        entry: dict = {"label": element.get("label"), "bounds": element["bounds"]}
        # The outline goes out whenever there is one, the mask only when it was asked for.
        if i < len(outlines) and outlines[i]:
            entry["polygon"] = outlines[i]
        if i < len(masks):
            entry["mask"] = masks[i]
        objects.append(entry)
    return objects


OBJECT_DETECTION = Tool(
    name="object_detection",
    description=(
        "Detect specific objects in images using text prompts in non GUI images. "
        "Returns detected objects with bounding boxes and labels. Works on images "
        "and on the pages of a PDF, where each object also carries its page number."
    ),
    parameters={
        "type": "object",
        "properties": {
            "file_ref_id": {
                "type": "string",
                "description": "The file reference id or URL of the image or PDF.",
            },
            "page_range": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
                "description": (
                    "PDF only: the 1-based [first, last] pages to search. Omit to "
                    f"search the first {_pages.DEFAULT_PAGES} pages."
                ),
            },
            "return_masks": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Set true only when the request asks for the raster "
                    "segmentation mask IMAGE of each object (e.g. 'cut out', "
                    "'transparent PNG', 'mask image'). Regardless of this "
                    "flag, each object carries a `polygon` -- the outline of "
                    "its largest region as coordinates -- whenever a contour "
                    "is available; prefer `bounds` when you need the object's "
                    "full extent, since `polygon` covers only the largest "
                    "region. Leave this false unless the caller specifically "
                    "needs the mask bitmap. Defaults to false."
                ),
            },
            "prompts": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Objects to detect, e.g. ['person', 'dog', 'car']. This is pure "
                    "grounding -- break complex requests into simple noun phrases. "
                    "When the request names a kind of thing ('vehicles', 'animals'), "
                    "pass that word itself, e.g. ['vehicle']: a list of its members "
                    "misses whatever the list leaves out."
                ),
            },
        },
        "required": ["file_ref_id", "prompts"],
        "additionalProperties": False,
    },
    execute=_run_detection,
)


async def _run_gui(args: dict, ctx: ToolContext) -> ToolResult:
    if ctx.ground is None:
        raise RuntimeError("gui_detection requires a grounding callable on the context")
    url = ctx.refs.resolve(args["file_ref_id"])
    prompts = [p for p in (args.get("prompts") or []) if p and p.strip()]

    everything = bool(args.get("all_elements")) or not prompts
    prompts = [_ALL_ELEMENTS] if everything else [_target(p) for p in prompts]

    # As interfaze grounds them: one call per phrase, each box labelled with the phrase
    # that found it (dev_qwen38_gui_main.py, label_as_phrase). Given a question-shaped
    # phrase, the model's own label echoes the instruction back. Every element on screen
    # is interfaze's detector path instead, whose elements are all typed "icon".
    # The screenshot's words, read while it is grounded: named by eye, "Add to cart" landed
    # on the Google logo. One cached read, so an ocr call afterwards costs nothing.
    reading = asyncio.ensure_future(_read_document(url, None, ctx))
    grounded = await asyncio.gather(*(ctx.ground(url, [phrase]) for phrase in prompts))
    try:
        words = _lines(await reading)
    except Exception as exc:
        log.info(
            "gui_detection could not read the screenshot's text: %s", type(exc).__name__ if ctx.zdr else exc
        )
        words = []
    elements: list[dict] = []
    seen: list[dict] = []
    for phrase, data in zip(prompts, grounded, strict=True):
        for e in _to_pixels(data.get("gui_elements") or [], data.get("width"), data.get("height")):
            # Interfaze's element is {type, bounds} (helpers/detection/detection.ts).
            bounds = _onto_its_line(e.get("bounds") or {}, words)
            elements.append({"type": "icon" if everything else phrase, "bounds": bounds})
            # The brain also sees what the box is. Handed only "icon", it paired "Health AI"
            # with whichever box came next, or grounded the page again on its own 0-1000 grid.
            label = _naming_label(e.get("label") or "", phrase)
            text = _text_inside(e.get("bounds") or {}, words)
            seen.append(
                {**elements[-1], **({"label": label} if label else {}), **({"text": text} if text else {})}
            )
    # The page's own lines, for any text the request asks to box: filled from the elements
    # instead, "Price history" landed on "plans" and "Or $112.25" on "other sellers".
    top_to_bottom = sorted(words, key=lambda word: (word[1][1], word[1][0]))
    page_text = [{"text": t, "bbox": list(b)} for t, b in top_to_bottom][:_MAX_PAGE_LINES]
    return ToolResult(
        model_facing={"gui_elements": seen, **({"page_text": page_text} if page_text else {})},
        full={"gui_elements": elements},
    )


_ALL_ELEMENTS = "interactive element (buttons, links, inputs, icons)"


_MAX_PAGE_LINES = 400


def _lines(reading: dict) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Each line of text read on the page: its words and its box."""
    out = []
    for section in reading.get("sections") or []:
        for line in section.get("lines") or []:
            tl, br = (
                (line.get("bounds") or {}).get("top_left"),
                (line.get("bounds") or {}).get("bottom_right"),
            )
            if tl and br and (line.get("text") or "").strip():
                out.append((line["text"].strip(), (tl["x"], tl["y"], br["x"], br["y"])))
    return out


def _box(bounds: dict) -> tuple[int, int, int, int] | None:
    tl, br = bounds.get("top_left"), bounds.get("bottom_right")
    return (tl["x"], tl["y"], br["x"], br["y"]) if tl and br else None


def _text_inside(bounds: dict, lines: list[tuple[str, tuple[int, int, int, int]]]) -> str:
    """The words of the lines whose centre is inside `bounds`, top to bottom."""
    b = _box(bounds)
    if not b:
        return ""
    inside = [
        (y1, x1, text)
        for text, (x1, y1, x2, y2) in lines
        if b[0] <= (x1 + x2) / 2 <= b[2] and b[1] <= (y1 + y2) / 2 <= b[3]
    ]
    return " ".join(text for _, _, text in sorted(inside))[:120]


def _onto_its_line(bounds: dict, lines: list[tuple[str, tuple[int, int, int, int]]]) -> dict:
    """`bounds`, or the text line it is a slightly misplaced copy of: a box within half
    again of one line's width and height that shares most of its area with it. A button
    larger than its label is left as it is."""
    b = _box(bounds)
    if not b:
        return bounds
    for _, line in lines:
        lw, lh, bw, bh = line[2] - line[0], line[3] - line[1], b[2] - b[0], b[3] - b[1]
        if lw <= 0 or lh <= 0 or not (0.67 <= bw / lw <= 1.5 and 0.67 <= bh / lh <= 1.5):
            continue
        ix = max(0, min(b[2], line[2]) - max(b[0], line[0]))
        iy = max(0, min(b[3], line[3]) - max(b[1], line[1]))
        if ix * iy >= 0.5 * min(lw * lh, bw * bh):
            return _bounds_dict(*line)
    return bounds


# Kinds of element, not names of one. Told a box was a "button", the brain named it that.
_GENERIC_LABELS = {
    "button",
    "icon",
    "link",
    "input",
    "text input",
    "text field",
    "field",
    "element",
    "checkbox",
    "tab",
}


def _naming_label(label: str, phrase: str) -> str | None:
    """The grounding model's label for a box, if it names the element rather than repeating
    the query or saying what kind of element it is."""
    word = label.strip().lower()
    if not word or word in _GENERIC_LABELS or word.rstrip("s") in _GENERIC_LABELS:
        return None
    if word == phrase.strip().lower() or word + "s" == phrase.strip().lower():
        return None
    return label.strip()


# A request's own verb and quantifier, ahead of what it asks for: "Detect all buttons".
_REQUEST = re.compile(
    r"^\s*(?:please\s+)?(?:detect|find|locate|identify|show(?:\s+me)?|list|highlight|mark|"
    r"get|spot|box|label|point\s+out)\s+(?:all\s+(?:of\s+)?(?:the\s+)?|every\s+|each\s+|the\s+|any\s+)?",
    re.I,
)


def _target(prompt: str) -> str:
    """What to locate, without the request around it: "Detect all buttons" -> "buttons".

    Grounded with the request left in, a page's buttons came back as 12 boxes; as "buttons",
    49. The noun stays as written: singularising it changes the referent ("check the letters")
    and cost interfaze half its ScreenSpot hits on the prompts it touched.
    """
    target = _REQUEST.sub("", prompt, count=1).strip()
    return target or prompt


GUI_DETECTION = Tool(
    name="gui_detection",
    # interfaze's wording, word for word (tools/index.ts). With a shorter all_elements
    # description the model read "Detect all buttons" as every element on screen and
    # never grounded "buttons".
    description=(
        "Detect all GUI elements on a screen or UI screenshot. Returns all interactive "
        "elements (buttons, inputs, links, etc.) with bounding boxes and labels. Use this "
        "for GUI detection, computer use, UI automation, and screen interaction tasks. "
        "Works with image URLs only."
    ),
    parameters={
        "type": "object",
        "properties": {
            "file_ref_id": {
                "type": "string",
                "description": (
                    "The file reference id or URL of the screenshot/UI image to detect "
                    "GUI elements in. This can only support image urls"
                ),
            },
            "prompts": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Pass the user prompt directly here without any edits. The "
                    "grounding model will handle the prompt."
                ),
            },
            "all_elements": {
                "type": "boolean",
                "default": False,
                "description": (
                    "True only when the user wants every element on the screen and "
                    "named nothing specific ('map this UI', 'what can I click here'). "
                    "Returns all elements with generic labels and skips the grounding "
                    "model entirely. Anything the user does name — even a bare noun "
                    "like 'trash' — means false, with that wording in prompts."
                ),
            },
        },
        "required": ["file_ref_id"],
        "additionalProperties": False,
    },
    execute=_run_gui,
)
