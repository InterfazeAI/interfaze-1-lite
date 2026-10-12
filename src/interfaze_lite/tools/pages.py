"""A PDF's pages as images, for the tools that only see images.

The grounding step behind detection cannot decode a PDF, so each page is rendered and
searched as an image.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
from dataclasses import dataclass

from ..contracts import UA
from .base import ToolContext

# The page frame OCR reports coordinates in, so a box found here and a line read there
# are in the same pixels.
PDF_DPI = 144
# Pages searched when the caller names none: each is a grounding call.
DEFAULT_PAGES = 10
MAX_PDF_BYTES = 50 * 1024 * 1024


@dataclass
class Page:
    number: int
    path: str
    width: int
    height: int


def _declared_pdf(url: str, ctx: ToolContext) -> bool:
    ref = next((r for r in ctx.refs.refs.values() if r.url == url), None)
    names = [ref.mime if ref else "", ref.filename if ref else "", url.split("?")[0]]
    return any(n and (n == "application/pdf" or n.lower().endswith(".pdf")) for n in names)


async def _pdf_bytes(url: str, ctx: ToolContext) -> bytes | None:
    """The document's bytes when it is a PDF; None when it is anything else."""
    if not url.startswith(("http://", "https://")):
        try:
            with open(url, "rb") as fh:
                head = fh.read(5)
                if head != b"%PDF-":
                    return None
                fh.seek(0)
                return fh.read(MAX_PDF_BYTES + 1)
        except OSError:
            return None
    if not _declared_pdf(url, ctx):
        # Most detection inputs are images; only a URL that says it is a PDF is fetched
        # here, so an image is still downloaded once, by the grounding step.
        return None
    # Fetched as the OCR tool fetches the same document.
    resp = await ctx.http.get(url, timeout=60, follow_redirects=True, headers={"User-Agent": UA})
    if not resp.is_success or not resp.content.startswith(b"%PDF-"):
        return None
    return resp.content


def _render(data: bytes, page_range: list[int] | None) -> list[Page]:
    import pymupdf

    if len(data) > MAX_PDF_BYTES:
        raise ValueError(f"PDF is too large to search ({len(data)} bytes; max {MAX_PDF_BYTES}).")
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        total = doc.page_count
        first, last = 1, min(total, DEFAULT_PAGES)
        if page_range:
            first = max(1, int(page_range[0]))
            last = min(total, int(page_range[-1]))
        if first > last:
            raise ValueError(f"page_range {page_range} is outside the document's {total} pages.")
        pages = []
        for number in range(first, last + 1):
            pix = doc[number - 1].get_pixmap(dpi=PDF_DPI)
            fd, path = tempfile.mkstemp(suffix=".png", prefix=f"page{number}-")
            os.close(fd)
            pix.save(path)
            pages.append(Page(number=number, path=path, width=pix.width, height=pix.height))
        return pages
    finally:
        doc.close()


async def pdf_pages(url: str, ctx: ToolContext, page_range: list[int] | None = None) -> list[Page] | None:
    """The PDF at `url` rendered page by page, or None when it is not a PDF.

    The caller removes the files with `cleanup` once it is done with them.
    """
    data = await _pdf_bytes(url, ctx)
    if data is None:
        return None
    return await asyncio.to_thread(_render, data, page_range)


def cleanup(pages: list[Page]) -> None:
    for page in pages:
        with contextlib.suppress(OSError):
            os.unlink(page.path)
