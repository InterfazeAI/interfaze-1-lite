"""Normalise every backend's native output into the shapes interfaze clients expect.

Mainly: geometry (xyxy, line quads and the brain's 0-1000 grid, all to four-corner `Bounds`),
composition (detector lines into sections/lines/words, the reader's text onto the detector's
geometry) and speaker attribution (`attribute_speakers`). The rest is input fetching, audio
windowing, scan cleanup and error wording the pipeline shares.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from html import unescape
from typing import Any, Literal

from rapidfuzz import fuzz

# Lines join a section when their heights match and their top and bottom edges agree,
# both within this many pixels.
ROW_TOLERANCE_PX = 5.0

BoxFormat = Literal["xyxy", "quad", "norm1000", "norm01", "cxcywh"]


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True)
class Bounds:
    """The one box shape every client consumes: four explicit corners plus derived extents."""

    top_left: Point
    top_right: Point
    bottom_right: Point
    bottom_left: Point
    width: int
    height: int

    def as_dict(self) -> dict[str, object]:
        return {
            "top_left": self.top_left.as_dict(),
            "top_right": self.top_right.as_dict(),
            "bottom_left": self.bottom_left.as_dict(),
            "bottom_right": self.bottom_right.as_dict(),
            "width": self.width,
            "height": self.height,
        }

    @property
    def x1(self) -> int:
        return self.top_left.x

    @property
    def y1(self) -> int:
        return self.top_left.y

    @property
    def x2(self) -> int:
        return self.bottom_right.x

    @property
    def y2(self) -> int:
        return self.bottom_right.y


def _axis_aligned(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> Bounds:
    # Sorted so misordered corners still give a positive box; rounded once, so width/height
    # equal the rounded corner difference.
    lo_x, hi_x = sorted((x1, x2))
    lo_y, hi_y = sorted((y1, y2))
    rx1 = max(0, round(lo_x))
    ry1 = max(0, round(lo_y))
    rx2 = min(img_w, round(hi_x))
    ry2 = min(img_h, round(hi_y))
    return Bounds(
        top_left=Point(rx1, ry1),
        top_right=Point(rx2, ry1),
        bottom_right=Point(rx2, ry2),
        bottom_left=Point(rx1, ry2),
        width=max(0, rx2 - rx1),
        height=max(0, ry2 - ry1),
    )


def to_bounds(
    raw: Sequence[float] | Sequence[Sequence[float]], *, src: BoxFormat, img_w: int, img_h: int
) -> Bounds:
    """Convert any backend's box into interfaze `Bounds`.

    `img_w`/`img_h` are required even for absolute formats: every path clamps to the image,
    since a detector run on a padded or downscaled copy emits coordinates past the edge.
    Raises ValueError on malformed or non-finite input; a NaN would serialise as `null`.
    """
    if src == "quad":
        pts = [(float(p[0]), float(p[1])) for p in raw]  # type: ignore[index]
        if len(pts) != 4:
            raise ValueError(f"quad needs 4 points, got {len(pts)}")
        if not all(math.isfinite(v) for pt in pts for v in pt):
            raise ValueError(f"non-finite coordinate in quad: {raw!r}")
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        # the line detector quads are near-rectangular but rotated by a degree or two. The
        # enclosing axis-aligned box is what the interfaze contract has always carried --
        # Azure's polygons were reduced the same way by `getBounds`.
        return _axis_aligned(min(xs), min(ys), max(xs), max(ys), img_w, img_h)

    vals = [float(v) for v in raw]  # type: ignore[arg-type]
    if len(vals) != 4:
        raise ValueError(f"{src} needs 4 values, got {len(vals)}")
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"non-finite coordinate in {src}: {raw!r}")

    if src == "xyxy":
        x1, y1, x2, y2 = vals
    elif src == "norm1000":
        # The brain's grounding: a 0-1000 grid, origin top-left. interfaze does the same
        # rescale in detection/modal.ts.
        x1, y1, x2, y2 = (
            vals[0] / 1000 * img_w,
            vals[1] / 1000 * img_h,
            vals[2] / 1000 * img_w,
            vals[3] / 1000 * img_h,
        )
    elif src == "norm01":
        x1, y1, x2, y2 = (vals[0] * img_w, vals[1] * img_h, vals[2] * img_w, vals[3] * img_h)
    elif src == "cxcywh":
        cx, cy, w, h = vals
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    else:
        raise ValueError(f"unknown box format {src!r}")

    return _axis_aligned(x1, y1, x2, y2, img_w, img_h)


@dataclass
class OCRWord:
    text: str
    bounds: Bounds
    confidence: float

    def as_dict(self) -> dict[str, object]:
        # Two places, as interfaze reports confidence: 0.99, not 0.9973307847976685.
        return {"text": self.text, "bounds": self.bounds.as_dict(), "confidence": round(self.confidence, 2)}


def split_words(text: str, bounds: Bounds, confidence: float) -> list[OCRWord]:
    """A line's words, each with its share of the line's box, as interfaze returns them.

    The detector's own word boxes segment its recognised text, which drops spaces between
    English words, so the quad is divided by character count instead: the words always join
    back into the line's text, and a tilted line keeps tilted word boxes.
    """
    tokens = text.split()
    if len(tokens) <= 1:
        return [OCRWord(text=text.strip() or text, bounds=bounds, confidence=confidence)]

    total = sum(len(t) for t in tokens) + len(tokens) - 1
    tl, tr, br, bl = bounds.top_left, bounds.top_right, bounds.bottom_right, bounds.bottom_left

    def along(a: Point, b: Point, f: float) -> Point:
        return Point(round(a.x + (b.x - a.x) * f), round(a.y + (b.y - a.y) * f))

    words: list[OCRWord] = []
    offset = 0
    for token in tokens:
        f0, f1 = offset / total, (offset + len(token)) / total
        top0, top1 = along(tl, tr, f0), along(tl, tr, f1)
        bot0, bot1 = along(bl, br, f0), along(bl, br, f1)
        words.append(
            OCRWord(
                text=token,
                confidence=confidence,
                bounds=Bounds(
                    top_left=top0,
                    top_right=top1,
                    bottom_right=bot1,
                    bottom_left=bot0,
                    width=round(((top1.x - top0.x) ** 2 + (top1.y - top0.y) ** 2) ** 0.5),
                    height=round(((bot0.x - top0.x) ** 2 + (bot0.y - top0.y) ** 2) ** 0.5),
                ),
            )
        )
        offset += len(token) + 1
    return words


@dataclass
class OCRLine:
    text: str
    bounds: Bounds
    average_confidence: float
    words: list[OCRWord] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "bounds": self.bounds.as_dict(),
            "average_confidence": round(self.average_confidence, 2),
            "words": [w.as_dict() for w in self.words],
        }


@dataclass
class OCRSection:
    text: str
    lines: list[OCRLine] = field(default_factory=list)
    # The 1-based page and THAT page's pixel frame: bounds are page-local, and the top-level
    # width/height spans pages.
    page: int | None = None
    width: int | None = None
    height: int | None = None

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {"text": self.text, "lines": [ln.as_dict() for ln in self.lines]}
        if self.page is not None:
            out["page"] = self.page
        if self.width is not None:
            out["width"] = self.width
        if self.height is not None:
            out["height"] = self.height
        return out


def merge_lines_into_sections(
    lines: Sequence[OCRLine], tolerance: float = ROW_TOLERANCE_PX
) -> list[OCRSection]:
    """Group lines that share a visual row into one section.

    A line joins the seed's section when heights match within `tolerance` and all four top
    (and bottom) corners sit within it of their mean, so same-row columns merge. Lines are
    consumed greedily in input order; each unclaimed line seeds a new section.
    """
    remaining = list(lines)
    sections: list[OCRSection] = []

    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        claimed: list[int] = []

        for idx, cand in enumerate(remaining):
            if abs(cand.bounds.height - seed.bounds.height) > tolerance:
                continue
            top_ys = [
                seed.bounds.top_left.y,
                seed.bounds.top_right.y,
                cand.bounds.top_left.y,
                cand.bounds.top_right.y,
            ]
            bottom_ys = [
                seed.bounds.bottom_left.y,
                seed.bounds.bottom_right.y,
                cand.bounds.bottom_left.y,
                cand.bounds.bottom_right.y,
            ]
            avg_top = sum(top_ys) / 4
            avg_bottom = sum(bottom_ys) / 4
            if all(abs(y - avg_top) <= tolerance for y in top_ys) and all(
                abs(y - avg_bottom) <= tolerance for y in bottom_ys
            ):
                group.append(cand)
                claimed.append(idx)

        for idx in reversed(claimed):
            remaining.pop(idx)

        group.sort(key=lambda ln: ln.bounds.x1)
        sections.append(OCRSection(text=" ".join(ln.text for ln in group), lines=group))

    return sections


def collapse_to_page(sections: Sequence[OCRSection]) -> OCRSection:
    """Flatten many regions into the single section that represents one page.

    Consumers read `sections[0]` as the page's text and `height / len(sections)` as its height,
    so a page split across sections draws its boxes in the wrong place, silently. Lines keep
    their geometry, ordered top-to-bottom, then left-to-right.
    """
    lines: list[OCRLine] = []
    for section in sections:
        lines.extend(section.lines)
    lines.sort(key=lambda ln: (ln.bounds.y1, ln.bounds.x1))
    return OCRSection(
        text="\n".join(s.text for s in sections if s.text.strip()),
        lines=lines,
    )


@dataclass
class VLMRegion:
    """A document VLM's output: text and structure, box quality varies, no confidence."""

    text: str
    bounds: Bounds
    category: str = ""
    # A table's HTML as the reader wrote it; empty for every other block.
    markup: str = ""


def iou(a: Bounds, b: Bounds) -> float:
    inter_w = min(a.x2, b.x2) - max(a.x1, b.x1)
    inter_h = min(a.y2, b.y2) - max(a.y1, b.y1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def contained_fraction(inner: Bounds, outer: Bounds) -> float:
    """What fraction of `inner` lies inside `outer`: unlike IoU, ~1.0 for every line of a paragraph."""
    inner_area = inner.width * inner.height
    if inner_area <= 0:
        return 0.0
    inter_w = min(inner.x2, outer.x2) - max(inner.x1, outer.x1)
    inter_h = min(inner.y2, outer.y2) - max(inner.y1, outer.y1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    return (inter_w * inter_h) / inner_area


def _fold(c: str) -> str:
    low = c.lower()
    return low if len(low) == 1 else c


class PageText:
    """The document reader's text for one page, and where each detector line is in it.

    Lines are found by what they say, not where the reader boxed them: its boxes came back
    squeezed or transposed, and matching by position dropped 62 of a memo's 98 lines.
    """

    def __init__(self, regions: Sequence[VLMRegion]):
        self.text = " ".join(" ".join(r.text for r in regions).split())
        self.pos = [i for i, c in enumerate(self.text) if c.isalnum()]
        # One character in, one out, so offsets in `flat` index `pos`: 'İ'.lower() is two.
        self.flat = "".join(_fold(self.text[i]) for i in self.pos)
        self.tokens = [(m.start(), m.end()) for m in re.finditer(r"\S+", self.text)]
        self.token_at = [0] * len(self.text)
        for k, (a, b) in enumerate(self.tokens):
            self.token_at[a:b] = [k] * (b - a)

    def read(
        self, lines: Sequence[OCRLine], accept: float = 88
    ) -> list[tuple[OCRLine, tuple[int, int] | None]]:
        """Each line with the reader's words for it, and where in the text they are.

        Each stretch of text goes to one line, closest match first, so near misses cannot all
        take it. A line not found keeps the detector's reading (unless unsure: `drop_unread`).
        """
        found = [self._find(ln.text, (), accept) for ln in lines]
        order = sorted(
            range(len(lines)), key=lambda i: (-(found[i][2] if found[i] else -1), -len(lines[i].text))
        )
        used: list[tuple[int, int]] = []
        out: list[tuple[OCRLine, tuple[int, int] | None]] = [(ln, None) for ln in lines]
        for i in order:
            hit = found[i]
            if hit:
                lo, hi = hit[1]
                if any(lo < b and a < hi for a, b in used):
                    hit = self._find(lines[i].text, used, accept)
            if not hit:
                continue
            text, span, _ = hit
            used.append(span)
            line = lines[i]
            out[i] = (
                OCRLine(
                    text=text,
                    bounds=line.bounds,
                    average_confidence=line.average_confidence,
                    words=split_words(text, line.bounds, line.average_confidence),
                ),
                span,
            )
        return out

    def _find(self, line: str, used, accept: float):
        wanted = "".join(c for c in line if c.isalnum())
        if not wanted or not self.flat:
            return None
        need = 100 if len(wanted) <= 4 else accept
        searched = self.flat
        for a, b in used:
            searched = _blanked(searched, a, b)
        for _ in range(4):
            hit = fuzz.partial_ratio_alignment("".join(map(_fold, wanted)), searched, score_cutoff=need)
            if hit is None or hit.dest_end <= hit.dest_start:
                return None
            text = self._words(line.strip(), self.pos[hit.dest_start], self.pos[hit.dest_end - 1] + 1)
            if text is not None:
                got = "".join(c for c in text if c.isalnum())
                score = fuzz.ratio(wanted, got)
                # The same letters in another case are another occurrence: the detector reads case off glyphs.
                if score >= need and not (got.lower() == wanted.lower() and got != wanted):
                    return text, (hit.dest_start, hit.dest_end), score
            searched = _blanked(searched, hit.dest_start, hit.dest_end)
        return None

    def _words(self, line: str, lo: int, hi: int) -> str | None:
        """The reader's text over [lo, hi), widened to whole words, or None if it splits one.

        Otherwise "150.02" matches inside barcode 005150025499; only a hyphen break may split a word.
        """
        t, tokens = self.text, self.tokens
        first, last = self.token_at[lo], self.token_at[hi - 1]
        start, end = tokens[first][0], tokens[last][1]
        if lo > start and _alnum(t[start:lo]):
            if not t[lo : tokens[first][1]].replace("-", "").isalpha():
                return None
            start = lo
        if hi < end and _alnum(t[hi:end]):
            if not line.endswith("-"):
                return None
            end = hi
        text = t[start:end]
        if line.endswith("-") and not text.endswith("-"):
            text += "-"
        # End punctuation only where the detector saw some (a bullet, not a form blank's underscores).
        if line[:1].isalnum():
            text = text.lstrip("".join({c for c in text if not c.isalnum()}))
        elif (
            first > 0
            and not _alnum(t[slice(*tokens[first - 1])])
            and set(t[slice(*tokens[first - 1])]) & set(line[:3])
        ):
            text = t[slice(*tokens[first - 1])] + " " + text
        if line[-1:].isalnum():
            text = text.rstrip("".join({c for c in text if not c.isalnum()}))
        elif (
            last + 1 < len(tokens)
            and not _alnum(t[slice(*tokens[last + 1])])
            and set(t[slice(*tokens[last + 1])]) & set(line[-3:])
        ):
            text = text + " " + t[slice(*tokens[last + 1])]
        return text.strip() or None

    def between(self, spans: Sequence[tuple[int, int]]) -> str | None:
        """The reader's text from the first span to the last, unless it is much longer than the spans."""
        if not spans:
            return None
        lo, hi = min(a for a, _ in spans), max(b for _, b in spans)
        if hi - lo > 1.3 * sum(b - a for a, b in spans) + 8:
            return None
        first, last = self.token_at[self.pos[lo]], self.token_at[self.pos[hi - 1]]
        return self.text[self.tokens[first][0] : self.tokens[last][1]]


def _blanked(text: str, start: int, end: int) -> str:
    """`text` with [start, end) overwritten by NULs, so a stretch already taken cannot match again."""
    return text[:start] + "\0" * (end - start) + text[end:]


def _alnum(text: str) -> str:
    return "".join(c for c in text if c.isalnum())


def drop_unread(
    regions: Sequence[VLMRegion],
    detector_lines: Sequence[OCRLine],
    min_confidence: float,
    alone_min_confidence: float,
) -> list[OCRLine]:
    """The detector lines that are text: those the reader also read, and those the detector is sure of.

    The recogniser reads something into every box (a textless photo gave "LE" at 0.35); where
    the reader read nothing at all, `alone_min_confidence` applies (a logo's bars read "II" at 0.73).
    """
    if not regions:
        return [line for line in detector_lines if line.average_confidence >= alone_min_confidence]
    read = PageText(regions).read(detector_lines)
    return [
        line
        for line, (_, span) in zip(detector_lines, read, strict=True)
        if span is not None or line.average_confidence >= min_confidence
    ]


# Quote marks and primes: what the document reader drops between letters it read right
# ("HGT 5-02\"" for 5'-02"). Not dots or commas, which specks give the detector.
_MARKS = set("'\"\u2018\u2019\u201c\u201d\u2032\u2033")
_MARK_MIN_CONFIDENCE = 0.9


def _with_marks(reader: str, detector: str) -> str:
    """The reader's words, with any quote mark the detector saw where the reader has punctuation of its own.

    Not between letters the reader joined: there the detector's marks were specks ("con'tains").
    """
    r_at = [i for i, c in enumerate(reader) if c.isalnum()]
    d_at = [i for i, c in enumerate(detector) if c.isalnum()]
    if "".join(reader[i] for i in r_at) != "".join(detector[i] for i in d_at):
        return reader
    out, last = [], 0
    for (r1, r2), (d1, d2) in zip(
        zip(r_at, r_at[1:], strict=False), zip(d_at, d_at[1:], strict=False), strict=True
    ):
        gap, seen = reader[r1 + 1 : r2], "".join(detector[d1 + 1 : d2].split())
        out.append(reader[last : r1 + 1])
        last = r2
        extra = list(seen)
        for c in gap:
            if c in extra:
                extra.remove(c)
        if (
            gap
            and "".join(gap.split()) == gap
            and _is_subsequence(gap, seen)
            and extra
            and set(extra) <= _MARKS
        ):
            gap = seen
        out.append(gap)
    return "".join(out) + reader[last:]


def _is_subsequence(short: str, long: str) -> bool:
    it = iter(long)
    return all(c in it for c in short)


def restore_marks(
    regions: Sequence[VLMRegion], markdown: str, detector_lines: Sequence[OCRLine]
) -> tuple[list[VLMRegion], str]:
    """Regions and markdown with the quote marks the reader dropped and a confident detector line saw.

    A fix goes in only where the reader's words for the line occur once.
    """
    sure = [line for line in detector_lines if line.average_confidence >= _MARK_MIN_CONFIDENCE]
    regions = list(regions)
    if not regions or not sure:
        return regions, markdown
    for line, (read, span) in zip(sure, PageText(regions).read(sure), strict=True):
        marked = _with_marks(read.text, line.text) if span is not None else read.text
        if marked == read.text:
            continue
        if sum(r.text.count(read.text) for r in regions) == 1:
            regions = [replace(r, text=r.text.replace(read.text, marked)) for r in regions]
        if markdown.count(read.text) == 1:
            markdown = markdown.replace(read.text, marked)
    return regions, markdown


def stitch(regions: Sequence[VLMRegion], detector_lines: Sequence[OCRLine]) -> list[OCRSection]:
    """One page: the document reader's text, on the detector's lines.

    The reader owns the text, order and structure; the detector owns every box and confidence.
    """
    lines = [line for line, _ in PageText(regions).read(detector_lines)]
    lines.sort(key=lambda ln: (ln.bounds.y1, ln.bounds.x1))
    return [OCRSection(text="\n".join(r.text for r in regions if r.text.strip()), lines=lines)]


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: str | None = None

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {"text": self.text, "timestamp": [self.start, self.end]}
        if self.speaker is not None:
            out["speaker"] = self.speaker
        return out


@dataclass
class SpeakerTurn:
    speaker: str
    start: float
    end: float


def attribute_speakers(
    words: Sequence[Word], turns: Sequence[SpeakerTurn], fill_nearest: bool = False
) -> list[Word]:
    """Assign each word the speaker whose turns overlap its full [start, end] span most, in TOTAL.

    As a common alignment toolkit's `assign_word_speakers` does. Deliberately NOT midpoint
    assignment (a boundary word goes by how much falls each side), NOT IoU (a long turn must not
    be penalised for its length), and not a forward-only pointer over turns, which mis-assigns
    when they interleave. `fill_nearest` gives a word overlapping no turn the turn nearest its
    midpoint, or SPEAKER_00 with no turns; interfaze never returns a null speaker, so callers set it.
    """
    out: list[Word] = []

    for word in words:
        totals: dict[str, float] = {}
        for turn in turns:
            overlap = min(turn.end, word.end) - max(turn.start, word.start)
            if overlap > 0:
                totals[turn.speaker] = totals.get(turn.speaker, 0.0) + overlap

        speaker: str | None = None
        if totals:
            speaker = max(totals.items(), key=lambda kv: kv[1])[0]
        elif fill_nearest and turns:
            # Distance to the turn's span, not its centre, so a long turn beats a short one centred nearer.
            mid = (word.start + word.end) / 2
            speaker = min(turns, key=lambda t: max(0.0, t.start - mid, mid - t.end)).speaker
        elif fill_nearest:
            speaker = "SPEAKER_00"

        out.append(Word(text=word.text, start=word.start, end=word.end, speaker=speaker))

    return out


def split_long_chunks(chunks: Sequence[dict], max_s: float = 3.0) -> list[dict]:
    """Transcript chunks cut to at most `max_s` seconds, as interfaze chunks them.

    A longer segment becomes equal-length pieces, its words divided between them by count.
    """
    out: list[dict] = []
    for chunk in chunks:
        start, end = (list(chunk.get("timestamp") or []) + [None, None])[:2]
        words = (chunk.get("text") or "").split()
        if start is None or end is None or end - start <= max_s or len(words) < 2:
            out.append(chunk)
            continue
        pieces = min(len(words), math.ceil((end - start) / max_s - 1e-9))
        step = (end - start) / pieces
        for i in range(pieces):
            part = words[i * len(words) // pieces : (i + 1) * len(words) // pieces]
            out.append(
                {
                    **chunk,
                    "text": " ".join(part),
                    "timestamp": [round(start + i * step, 2), round(start + (i + 1) * step, 2)],
                }
            )
    return out


def group_by_speaker(words: Sequence[Word]) -> list[dict[str, object]]:
    """Collapse speaker-attributed words into contiguous per-speaker chunks: the `by_speaker=true` shape."""
    chunks: list[dict[str, object]] = []
    for word in words:
        if chunks and chunks[-1]["speaker"] == word.speaker:
            chunks[-1]["text"] = f"{chunks[-1]['text']} {word.text}"
            chunks[-1]["timestamp"] = [chunks[-1]["timestamp"][0], word.end]  # type: ignore[index]
        else:
            chunks.append(
                {
                    "speaker": word.speaker,
                    "text": word.text,
                    "timestamp": [word.start, word.end],
                }
            )
    return chunks


def estimate_words(segments: Sequence[Word]) -> list[Word]:
    """Word timing estimated from segment timing: each segment's span shared among its words by length.

    For recordings too long to decode word timing for; joined by segment instead, one spanning
    a speaker change would go wholly to one speaker.
    """
    words: list[Word] = []
    for segment in segments:
        tokens = segment.text.split()
        if not tokens:
            continue
        weights = [len(token) + 1 for token in tokens]
        total, span, start = sum(weights), max(0.0, segment.end - segment.start), segment.start
        for token, weight in zip(tokens, weights, strict=True):
            end = start + span * weight / total
            words.append(Word(text=token, start=start, end=end))
            start = end
    return words


def speech_windows(
    samples, rate: int = 16_000, *, max_s: float = 30.0, min_s: float = 20.0, frame_s: float = 0.03
) -> list[tuple[int, int]]:
    """(start, end) sample ranges covering the audio, each at most `max_s` seconds.

    Each cut falls on the quietest frame between `min_s` and `max_s` after the last, so no word
    is split ("websitesince"); silence, where the recogniser invents "Thank you.", is trimmed.
    """
    import numpy as np

    n = len(samples)
    frame = max(1, int(frame_s * rate))
    count = n // frame
    if count == 0:
        return [(0, n)] if n else []
    rms = np.sqrt(
        np.mean(np.asarray(samples[: count * frame], dtype=np.float64).reshape(count, frame) ** 2, axis=1)
    )
    # Averaged over ~0.3 s, the minimum lies inside the pause, not against a word's end.
    span = max(1, int(0.3 / frame_s))
    smooth = np.convolve(rms, np.ones(span) / span, mode="same")
    longest, shortest = int(max_s * rate), int(min_s * rate)
    windows, start = [], 0
    while n - start > longest:
        lo, hi = (start + shortest) // frame, min(count, (start + longest) // frame)
        cut = (lo + int(np.argmin(smooth[lo:hi]))) * frame
        windows.append((start, cut))
        start = cut
    windows.append((start, n))

    # Trimmed to speech +-0.2 s, dropped under 0.3 s of it; timestamps are window-relative, so none move.
    floor = max(1e-3, 0.05 * float(np.percentile(rms, 95)))
    pad, minimum = int(0.2 * rate), max(1, int(0.3 / frame_s))
    kept = []
    for s, e in windows:
        first_frame = s // frame
        loud = np.nonzero(rms[first_frame : max(first_frame + 1, e // frame)] >= floor)[0]
        if loud.size < minimum:
            continue
        kept.append(
            (
                max(s, (first_frame + int(loud[0])) * frame - pad),
                min(e, (first_frame + int(loud[-1]) + 1) * frame + pad),
            )
        )
    return kept


UA = "interfaze-1-lite/0.1 (+https://github.com/InterfazeAI/interfaze-1-lite)"

# Left out of the document text, as the reader's authors do, but kept as regions for the lines inside.
_PAGE_FURNITURE = {"Page-header", "Page-footer"}


def layout_cells(raw: str) -> list[dict]:
    """The document reader's JSON list of {bbox, category, text} cells.

    Decoded one cell at a time, so a page cut off mid-array keeps every completed cell.
    """
    start = raw.find("[")
    if start == -1:
        return []
    decoder, cells, i = json.JSONDecoder(), [], start + 1
    while True:
        i = raw.find("{", i)
        if i == -1:
            break
        try:
            cell, i = decoder.raw_decode(raw, i)
        except json.JSONDecodeError:
            break
        if isinstance(cell, dict):
            cells.append(cell)
    return cells


def _formula(latex: str) -> str:
    """A display formula as markdown: $$-fenced unless the reader already fenced it."""
    latex = latex.strip().strip("`")
    if (latex.startswith("$$") and latex.endswith("$$")) or (
        latex.startswith("\\[") and latex.endswith("\\]")
    ):
        latex = latex[2:-2].strip()
    elif "$" in latex or "\\[" in latex:
        return latex
    return f"$$\n{latex}\n$$"


def read_layout(raw: str, sent_size: tuple[int, int], img) -> tuple[list[VLMRegion], str]:
    """Regions and markdown from the document reader's layout JSON.

    Its boxes are pixels on the image it was SENT (snapped to its patch grid), so they are scaled
    to `img`; guessing the frame from the values misplaces every box on a page under 1000 px.
    Tables keep their HTML as `markup`; pictures carry no text and are dropped.
    """
    sx, sy = img.width / sent_size[0], img.height / sent_size[1]
    regions: list[VLMRegion] = []
    blocks: list[str] = []
    for cell in layout_cells(raw):
        category, text = str(cell.get("category", "")), str(cell.get("text") or "").strip()
        if category == "Picture" or not text:
            continue
        markup = text if category == "Table" else ""
        if category not in _PAGE_FURNITURE:
            blocks.append(_formula(text) if category == "Formula" else text)
        bbox = cell.get("bbox")
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
            bounds = to_bounds(
                [x1 * sx, y1 * sy, x2 * sx, y2 * sy], src="xyxy", img_w=img.width, img_h=img.height
            )
        except (TypeError, ValueError):
            continue  # a malformed box must not take the page's text down with it
        regions.append(VLMRegion(text=_plain(_unmark(text)), bounds=bounds, category=category, markup=markup))
    return regions, "\n\n".join(blocks)


_LINE_MARKER = re.compile(r"^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s*)", re.M)
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_MATH = re.compile(r"\$\$?([^$]+?)\$\$?")


def _unfence(match: re.Match) -> str:
    """Math without its $ fences; anything else between two single $ is prices."""
    inner = match.group(1)
    is_math = (
        match.group(0).startswith("$$")
        or re.search(r"[\\^_{}]", inner)
        or re.fullmatch(r"[A-Za-z]\w*", inner)
    )
    return inner if is_math else match.group(0)


def _unmark(text: str) -> str:
    """A cell's markdown as the words on the page, since it becomes each matched line's text.

    A $ stays unless it fences math: "$166.20" is what the receipt says.
    """
    text = _MATH.sub(_unfence, _LINK.sub(r"\1", _LINE_MARKER.sub("", text)))
    return re.sub(r"\*\*|__|[*`]", "", text)


_INLINE_TAG = re.compile(r"</?(?:b|i|u|s|em|strong|span|sup|sub|a|mark|small|code|del|ins)\b[^>]*>", re.I)


def _plain(markup: str) -> str:
    """Markup as text: inline tags join ("<i>x</i>." is "x."), the rest separate, entities decode."""
    text = re.sub(r"<[^>]+>", " ", _INLINE_TAG.sub("", markup))
    return " ".join(unescape(text).split())


PDF_MAGIC = b"%PDF-"
ZIP_MAGIC = b"PK\x03\x04"


class InputFetchError(Exception):
    """The caller's file could not be fetched; kept apart because the remedy is theirs, not an outage."""


def fetch_url(url: str, *, timeout: float, headers: dict | None = None):
    import httpx

    try:
        resp = httpx.get(
            url, timeout=timeout, follow_redirects=True, headers={"User-Agent": UA, **(headers or {})}
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise InputFetchError(f"could not fetch {url}: HTTP {exc.response.status_code}") from exc
    except httpx.RequestError as exc:
        raise InputFetchError(f"could not fetch {url}: {type(exc).__name__}") from exc
    return resp


def fetch_bytes(source) -> bytes:
    """Read a URL, path or raw bytes into memory."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, str) and source.startswith(("http://", "https://")):
        return fetch_url(source, timeout=120).content
    with open(source, "rb") as fh:
        return fh.read()


def document_kind(source) -> str:
    """Classify by CONTENT, falling back to extension: temp-file suffixes come from guessed MIME types."""
    if hasattr(source, "convert"):  # already a PIL image
        return "image"
    try:
        head = fetch_bytes(source)[:4096]
    except Exception:
        head = b""

    if head.startswith(PDF_MAGIC):
        return "pdf"
    if head.startswith(ZIP_MAGIC) and b"word/" in head:
        return "docx"

    name = source.lower() if isinstance(source, str) else ""
    if name.endswith(".pdf"):
        return "pdf"
    if name.endswith(".docx"):
        return "docx"
    if name.endswith(TEXT_SUFFIXES) or _looks_like_text(head):
        return "text"
    return "image"


TEXT_SUFFIXES = (".csv", ".tsv", ".txt", ".md", ".json", ".log", ".yaml", ".yml")


def _looks_like_text(head: bytes) -> bool:
    """Whether these bytes decode as UTF-8; image and archive formats put non-UTF-8 bytes up front."""
    if not head or b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        # A multi-byte character may straddle the end of the slice.
        try:
            head[:-4].decode("utf-8")
        except UnicodeDecodeError:
            return False
    return True


def backfill_sections(sections: list[dict], *, page: int, width: int, height: int) -> list[dict]:
    """Stamp each section with its page and that page's pixel frame; no coordinate is
    translated (see `OCRSection.page`)."""
    return [{**s, "page": page, "width": width, "height": height} for s in sections]


def rescale_page(page: dict, width: int, height: int) -> dict:
    """One OCR'd page with every box moved into a `width` x `height` frame.

    The frame the caller measures against: the image's own pixels, or a PDF page at interfaze's
    render scale, which differ from the read resolution whenever a page is resized to be read.
    """
    sx, sy = width / (page.get("width") or width), height / (page.get("height") or height)
    if (sx, sy) == (1.0, 1.0):
        return page

    def box(b: dict) -> dict:
        corners = {
            k: {"x": round(b[k]["x"] * sx), "y": round(b[k]["y"] * sy)}
            for k in ("top_left", "top_right", "bottom_right", "bottom_left")
            if k in b
        }
        return {**b, **corners, "width": round(b["width"] * sx), "height": round(b["height"] * sy)}

    def line(ln: dict) -> dict:
        return {
            **ln,
            "bounds": box(ln["bounds"]),
            "words": [{**w, "bounds": box(w["bounds"])} for w in ln.get("words") or []],
        }

    sections = [
        {
            **s,
            "lines": [line(ln) for ln in s.get("lines") or []],
            **({"width": width, "height": height} if "width" in s else {}),
        }
        for s in page.get("sections") or []
    ]
    layout = [{**b, "bounds": box(b["bounds"])} for b in page.get("layout") or []]
    return {**page, "sections": sections, "layout": layout, "width": width, "height": height}


def speaker_tracks(output: Any):
    """Yield (segment, track, label) across the diarization pipeline versions.

    3.x returns an Annotation; 4.x / community-1 a DiarizeOutput, whose exclusive view is preferred:
    one speaker per instant, so a word cannot accrue overlap against two at once.
    """
    for attr in ("exclusive_speaker_diarization", "speaker_diarization"):
        annotation = getattr(output, attr, None)
        if annotation is not None and hasattr(annotation, "itertracks"):
            return annotation.itertracks(yield_label=True)
    if hasattr(output, "itertracks"):
        return output.itertracks(yield_label=True)
    raise RuntimeError(
        f"cannot read speaker turns from {type(output).__name__}; "
        f"attributes: {sorted(a for a in dir(output) if not a.startswith('_'))[:20]}"
    )


def pdf_render_dpi(width_pt: float, height_pt: float, *, dpi: int, max_pixels: int, small_pixels: int) -> int:
    """The DPI to rasterise a `width_pt` x `height_pt` PDF page at for reading.

    Fitted to `max_pixels` (a 16+ MP page cost 223 s); a page `dpi` leaves under `small_pixels`
    is read at the budget instead, since tiny print rendered small had the reader invent sentences.
    """
    inches = (width_pt / 72.0) * (height_pt / 72.0)
    if inches <= 0:
        return dpi
    budget = (max_pixels / inches) ** 0.5
    target = budget if inches * dpi * dpi < small_pixels else dpi
    return int(max(72, min(target, budget)))


def enhance_scan(img):
    """Normalise contrast on a page that reads like a degraded scan.

    olmOCR-Bench's old_scans were nearly right but for a few characters: a legibility problem.
    Only a low-contrast histogram is stretched, so a clean digital PDF keeps its antialiasing.
    """
    from PIL import ImageOps

    if img.mode not in ("L", "RGB"):
        return img

    grey = img.convert("L")
    histogram = grey.histogram()
    pixels = sum(histogram)
    if not pixels:
        return img

    # Ink and paper as the 5th and 95th percentile, so specks or a bright margin do not decide it.
    cumulative, low, high = 0, 0, 255
    for level, count in enumerate(histogram):
        cumulative += count
        if low == 0 and cumulative >= pixels * 0.05:
            low = level
        if cumulative >= pixels * 0.95:
            high = level
            break

    if high - low > 200:
        return img
    return ImageOps.autocontrast(img.convert("RGB"), cutoff=1)


def normalise_error(text: str, vendors: tuple[str, ...] = ()) -> str:
    """Rewrite an error so it names the capability that failed, not the component.

    A sidecar traceback reaches the model and the caller verbatim; `vendors`, from the caller's
    configuration, are the names replaced.
    """
    if not text or not vendors:
        return text

    tokens: set[str] = set()
    for vendor in vendors:
        for part in re.split(r"[/_\-\s]+", vendor.lower()):
            if len(part) > 3 and not part.isdigit():
                tokens.add(part)
        tokens.add(vendor.lower())

    cleaned = re.sub(
        r"\b[\w.-]+/[\w.-]+\b",
        lambda m: "<component>" if any(t in m.group(0).lower() for t in tokens) else m.group(0),
        text,
    )
    for token in sorted(tokens, key=len, reverse=True):
        cleaned = re.sub(re.escape(token), "<component>", cleaned, flags=re.I)
    return re.sub(r"(<component>[\s./-]*)+", "<component>", cleaned)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def attach_text_to_layout(
    blocks: list[dict], lines: list[OCRLine], min_containment: float = 0.5, regions: Sequence[VLMRegion] = ()
) -> list[dict]:
    """Give every layout block the text that falls inside it.

    The line detector says which lines lie in a block (it shares the layout detector's frame);
    the reader says what they say (`PageText`), and a table block takes the reader's matching
    table as HTML. Lines outside every block go to an "unclassified" block, never dropped.
    """
    page = PageText(regions)
    read = page.read(lines) if regions else [(line, None) for line in lines]
    boxes = [_bounds_from_dict(b["bounds"]) for b in blocks]
    tables = [r for r in regions if "<table" in r.markup.lower()]

    def reading_order(indices):
        return sorted(indices, key=lambda i: (lines[i].bounds.y1, lines[i].bounds.x1))

    # Each line to the smallest block holding it: blocks nest, and in list order the outer box took all.
    owner: dict[int, int] = {}
    for i, line in enumerate(lines):
        holding = [
            k for k, box in enumerate(boxes) if contained_fraction(line.bounds, box) >= min_containment
        ]
        if holding:
            owner[i] = min(holding, key=lambda k: (boxes[k].width * boxes[k].height, k))
    claimed = set(owner)
    out: list[dict] = []
    starts: list[int | None] = []
    for k, block in enumerate(blocks):
        mine = reading_order(i for i, o in owner.items() if o == k)
        spans = [read[i][1] for i in mine if read[i][1]]
        starts.append(min((a for a, _ in spans), default=None))
        text = (page.between(spans) if len(spans) >= 0.7 * len(mine) else None) or " ".join(
            read[i][0].text for i in mine
        )
        if tables and mine and "table" in str(block.get("type", "")):
            said = _words(text)
            best = max(tables, key=lambda r: len(said & _words(r.text)))
            if said and len(said & _words(best.text)) >= 0.5 * len(said):
                text = best.markup
        out.append({**block, "text": text.strip(), "line_count": len(mine)})
    out = _in_reading_order(out, boxes, starts)

    spare = reading_order(i for i in range(len(lines)) if i not in claimed)
    if spare:
        out.append(
            {
                "type": "unclassified",
                "score": 0.0,
                "bounds": _union_of([lines[i].bounds for i in spare]),
                "text": "\n".join(read[i][0].text for i in spare),
                "line_count": len(spare),
            }
        )
    return out


def _in_reading_order(blocks: list[dict], boxes: list[Bounds], starts: list[int | None]) -> list[dict]:
    """`blocks` in the reader's order rather than down the page.

    Sorted by y, two columns interleave. A block's place is where its words begin in the reader's
    text (`starts`); one it did not read, like a figure, follows the last placed block above it.
    """
    if all(start is None for start in starts):
        return blocks
    keys: list[float] = []
    for k, box in enumerate(boxes):
        if starts[k] is not None:
            keys.append(float(starts[k]))
            continue
        above = [
            starts[j]
            for j, other in enumerate(boxes)
            if starts[j] is not None and other.y2 <= box.y1 and min(box.x2, other.x2) > max(box.x1, other.x1)
        ]
        keys.append(max(above) + 0.5 if above else -0.5)
    order = sorted(range(len(blocks)), key=lambda k: (keys[k], boxes[k].y1, boxes[k].x1))
    return [blocks[k] for k in order]


def _union_of(boxes: list[Bounds]) -> dict:
    xs = [b.top_left.x for b in boxes] + [b.bottom_right.x for b in boxes]
    ys = [b.top_left.y for b in boxes] + [b.bottom_right.y for b in boxes]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return Bounds(
        top_left=Point(x1, y1),
        top_right=Point(x2, y1),
        bottom_right=Point(x2, y2),
        bottom_left=Point(x1, y2),
        width=x2 - x1,
        height=y2 - y1,
    ).as_dict()


def _bounds_from_dict(raw: dict) -> Bounds:
    tl, br = raw["top_left"], raw["bottom_right"]
    return Bounds(
        top_left=Point(tl["x"], tl["y"]),
        top_right=Point(br["x"], tl["y"]),
        bottom_right=Point(br["x"], br["y"]),
        bottom_left=Point(tl["x"], br["y"]),
        width=br["x"] - tl["x"],
        height=br["y"] - tl["y"],
    )
