"""Reading documents: text, per-line geometry, and page layout."""

from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .base import Tool, ToolContext, ToolResult, _post

WEB_PAGE_TYPES = ("text/html", "application/xhtml+xml")
# Already text, so nothing to render: a CSV sent to the document reader failed, and the
# model retried it to the step limit.
TEXT_TYPES = (
    "text/",
    "application/json",
    "application/xml",
    "application/yaml",
    "application/x-yaml",
    "application/x-ndjson",
)
MAX_TEXT_BYTES = 10 * 1024 * 1024


async def _content_type(url: str, ctx: ToolContext) -> str:
    """The type the server reports for a URL, or "" when it will not say."""
    try:
        resp = await ctx.http.get(url, timeout=30, follow_redirects=True, headers={"Range": "bytes=0-0"})
    except Exception:
        return ""
    if not resp.is_success:
        return ""
    return (resp.headers.get("content-type") or "").split(";")[0].strip().lower()


def _declared_type(url: str, ctx: ToolContext) -> str:
    ref = next((r for r in ctx.refs.refs.values() if r.url == url), None)
    return (
        ref.mime
        if ref and ref.mime
        else mimetypes.guess_type(ref.filename if ref and ref.filename else url)[0] or ""
    )


async def _read_text(url: str, ctx: ToolContext) -> str:
    if not url.startswith(("http://", "https://")):
        path = Path(url)
        return (await asyncio.to_thread(path.read_bytes))[:MAX_TEXT_BYTES].decode("utf-8", "replace")
    body = bytearray()
    async with ctx.http.stream("GET", url, timeout=30, follow_redirects=True) as resp:
        resp.raise_for_status()
        async for chunk in resp.aiter_bytes():
            body += chunk
            if len(body) > MAX_TEXT_BYTES:
                break
    return bytes(body[:MAX_TEXT_BYTES]).decode("utf-8", "replace")


def _capped_text(text: str, limit: int) -> dict[str, Any]:
    """`extracted_text` cut to `limit` characters, with a note saying so when it was cut."""
    if len(text) <= limit:
        return {"extracted_text": text}
    return {
        "extracted_text": text[:limit],
        "truncated": True,
        "note": f"showing first {limit} of {len(text)} characters; the caller received the complete text",
    }


def _text_result(text: str, ctx: ToolContext) -> ToolResult:
    return ToolResult(
        model_facing=_capped_text(text, ctx.settings.max_context_chars),
        # Interfaze's shape, with no page geometry: a text file has none.
        full={"extracted_text": text, "sections": [], "width": None, "height": None},
    )


# Recent reads by what was read, since relay rounds and landing demos re-read the same document
# at 7-28 s a read. interfaze keeps them 14 days per project; this is one process's memory.
# An in-flight read is kept too, so an identical request waits on it.
_READ_TTL_S = 3600
_READ_LIMIT = 32
_reads: OrderedDict[str, tuple[float, dict | asyncio.Future]] = OrderedDict()


async def _read(url: str, page_range: list[int] | None, ctx: ToolContext) -> dict:
    payload = {"url": url, "return_markdown": True, "page_range": page_range}
    if ctx.zdr:
        # Nothing of a zero-data-retention caller's outlives their request.
        return await _post(ctx, ctx.settings.perception_url, "/ocr", payload)
    key = hashlib.sha256(f"{url}|{page_range}".encode()).hexdigest()
    held = _reads.get(key)
    if held is not None and time.monotonic() - held[0] <= _READ_TTL_S:
        _reads.move_to_end(key)
        if isinstance(held[1], dict):
            return held[1]
    else:
        task = asyncio.ensure_future(_post(ctx, ctx.settings.perception_url, "/ocr", payload))
        # Its failure is read here even when every caller waiting on it has gone.
        task.add_done_callback(lambda t: t.cancelled() or t.exception())
        held = _reads[key] = (time.monotonic(), task)
        while len(_reads) > _READ_LIMIT:
            _reads.popitem(last=False)
    try:
        # Shielded: one caller giving up does not cancel the read the others wait on.
        data = await asyncio.shield(held[1])
    except Exception:
        _forget(key, held)
        raise
    if not data.get("has_text") and not (data.get("text") or "").strip():
        # A read that found nothing may have been a bad fetch; the next one tries again.
        _forget(key, held)
    elif _reads.get(key) is held:
        _reads[key] = (held[0], data)
    return data


def _forget(key: str, held: tuple) -> None:
    if _reads.get(key) is held:
        del _reads[key]


async def _run_ocr(args: dict, ctx: ToolContext) -> ToolResult:
    url = ctx.refs.resolve(args["file_ref_id"])
    remote = url.startswith(("http://", "https://"))
    ctype = (await _content_type(url, ctx) if remote else "") or _declared_type(url, ctx)
    if remote and ctype.startswith("application/octet-stream"):
        # Storage often labels everything this way; the name the caller gave says more.
        ctype = _declared_type(url, ctx) or ctype
    if ctype.startswith(WEB_PAGE_TYPES):
        return ToolResult(
            model_facing={
                "error": (
                    f"{url} is a web page, not a document. ocr reads images, PDFs and Word "
                    f"documents the user supplied. To read a web page, use a scraping "
                    f"function if this request offers one; if none is offered, say that "
                    f"scraping is not available."
                ),
                "message": "Do not retry ocr on this URL.",
            }
        )
    if ctype.startswith(TEXT_TYPES):
        return _text_result(await _read_text(url, ctx), ctx)
    # One read of the document serves every option: its text, sections and layout are
    # the same whatever was asked, and the markdown costs nothing more. Asked again for
    # line boxes after the markdown, the receipt was read a second time, ~20 s each.
    key = f"/ocr|{url}|{args.get('page_range')}"
    data = ctx.cache.get(key)
    if data is None:
        data = await _read(url, args.get("page_range"), ctx)
        ctx.cache[key] = data

    sections = data.get("sections") or []
    # Straight from the document VLM: re-joined section text could carry detector text and
    # disagree with extracted_text.
    extracted = data.get("text") or data.get("context") or ""

    # Interfaze's shape key for key (total_pages only for a paged document; no layout blocks),
    # built explicitly so a new backend field stays out until named.
    full = {
        "extracted_text": extracted,
        "sections": sections,
        "width": data.get("width"),
        "height": data.get("height"),
        **({"total_pages": data.get("total_pages")} if data.get("pages_processed") else {}),
    }
    empty = not data.get("has_text") and not extracted.strip()
    if empty:
        full["should_not_return_to_user"] = True

    # The model reads a paged document with each page marked. Run together, a form's
    # two pages read as one, and the model asked for page 2 again -- a second ocr result
    # that clients showing the last one drew over page 1 with nothing.
    pages = data.get("page_texts") or []
    first_page = (data.get("pages_processed") or [1])[0]
    seen = (
        "\n\n".join(f"--- page {first_page + i} ---\n{text}" for i, text in enumerate(pages))
        if len(pages) > 1
        else extracted
    )
    return ToolResult(
        model_facing={
            **_capped_text(seen, ctx.settings.max_context_chars),
            "width": full["width"],
            "height": full["height"],
            "total_pages": data.get("total_pages"),
            **({"pages_read": data["pages_processed"]} if data.get("pages_processed") else {}),
            # Said, so the answer is "there is no text in this image", as interfaze's is.
            **({"note": "No text was found in this file."} if empty else {}),
            **(
                _layout_view(sections, data.get("layout") or [], ctx.settings.max_layout_blocks)
                if (args.get("return_bounds") or ctx.wants_geometry)
                else {}
            ),
        },
        full=full,
    )


def _layout_view(sections: list[dict], layout: list[dict], cap: int) -> dict[str, Any]:
    """A compact, model-readable index of what is on each page and where.

    The full `sections` would swamp the context, but with no geometry a layout request could
    only answer an empty list, since the system prompt forbids inventing coordinates. `layout`
    is the detector's typed blocks, for layout requests; `lines` each text line, for one field.
    Opt-in: attached to every call, the crowding cut a word-extraction test from 446 words to 392.
    """

    def flat(bounds: dict) -> list | None:
        top_left, bottom_right = bounds.get("top_left"), bounds.get("bottom_right")
        if not top_left or not bottom_right:
            return None
        return [top_left.get("x"), top_left.get("y"), bottom_right.get("x"), bottom_right.get("y")]

    blocks: list[dict[str, Any]] = []
    for block in layout:
        text, bbox = (block.get("text") or "").strip(), flat(block.get("bounds") or {})
        if bbox and (text or block.get("type") in _PICTURES):
            blocks.append(
                {
                    "page": block.get("page") or 1,
                    "type": _PLAIN.get(block.get("type"), block.get("type")),
                    "text": _clip(text),
                    "bbox": bbox,
                }
            )
    lines: list[dict[str, Any]] = []
    for section in sections:
        for line in section.get("lines") or []:
            text, bbox = (line.get("text") or "").strip(), flat(line.get("bounds") or {})
            if bbox and text:
                lines.append({"page": section.get("page") or 1, "text": text, "bbox": bbox})

    if not blocks and not lines:
        return {}
    out: dict[str, Any] = {
        "layout_note": (
            "Positions on each page. bbox is [top_left_x, top_left_y, "
            "bottom_right_x, bottom_right_y] in pixels, relative to the page "
            "named in `page`. `layout` holds the page's elements with their "
            "type, `lines` each line of text. Use these for any question about "
            "position, layout or bounding boxes; extracted_text has the full text."
        ),
    }
    if blocks:
        if len(blocks) > cap:
            out["layout_truncated"] = f"showing {cap} of {len(blocks)} blocks"
        out["layout"] = blocks[:cap]
    if len(lines) <= cap or not blocks:
        if len(lines) > cap:
            out["lines_truncated"] = f"showing {cap} of {len(lines)} lines"
        out["lines"] = lines[:cap]
    else:
        out["lines_omitted"] = (
            f"{len(lines)} lines is too many to list; call ocr again with a page_range for line-level boxes"
        )
    return out


# The layout detector's labels, in the words a request uses. Handed "paragraph_title",
# the model filed every section heading of a paper under "paragraph".
_PLAIN = {
    "doc_title": "title",
    "paragraph_title": "heading",
    "text": "paragraph",
    "figure_title": "caption",
    "table_title": "caption",
    "chart_title": "caption",
    "reference_content": "reference",
    "aside_text": "margin_text",
}

# Blocks worth a box even with no text in them.
_PICTURES = ("image", "chart", "figure", "seal", "header_image", "footer_image")


def _clip(text: str, limit: int = 240) -> str:
    # Enough to tell blocks apart and quote a heading; extracted_text has the rest.
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


OCR = Tool(
    name="ocr",
    description=(
        "Extract texts from images, PDF documents and Word documents (.docx) using OCR. "
        "Also reads text files (CSV, TXT, JSON, Markdown), returning their contents."
    ),
    parameters={
        "type": "object",
        "properties": {
            "file_ref_id": {
                "type": "string",
                "description": (
                    "The file reference id or url to OCR. It can handle urls for "
                    "images, PDF documents and Word documents (.docx)"
                ),
            },
            "return_bounds": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Set to true when the request involves bounding boxes, "
                    "coordinates, positions or document layout. Bounds always "
                    "reach the caller; this controls whether the answering "
                    "model sees them, so it must be true to answer any "
                    "question about where something is on the page."
                ),
            },
            "return_markdown": {
                "type": "boolean",
                "default": False,
                "description": "Whether to return the extracted text formatted as Markdown.",
            },
            "page_range": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
                "description": "PDF only. [startPage, endPage], 1-based and inclusive.",
            },
        },
        "required": ["file_ref_id"],
        "additionalProperties": False,
    },
    execute=_run_ocr,
)
