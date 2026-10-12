"""Forecasting: interfaze's dataset parsing and pre-processing, and its TimesFM service's.

Ported so their behaviour does not move: interfaze `helpers/forecastDataset.ts` and
`helpers/forecast/index.ts` through `series`, JigsawStack's `prediction-timesfm` service after it.
Pure, apart from pandas/numpy imported where they are used.
"""

from __future__ import annotations

import ipaddress
import json
import math
import re
from datetime import datetime, timedelta
from urllib.parse import urlparse

MIN_FORECAST_POINTS = 5
MAX_FORECAST_POINTS = 1_000
MAX_DATASET_BYTES = 25 * 1024 * 1024
# TimesFM, as JigsawStack's service compiles it.
MAX_HORIZON = 256
MAX_CONTEXT = 1024

_DATE_HEADER = re.compile(r"date|time|day|month|year|timestamp|period|\bds\b", re.I)
_GROUPED = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")


class DatasetError(ValueError):
    """The data could not be turned into a series; the message says what to fix."""


def parse_csv(text: str) -> list[list[str]]:
    """RFC-4180-ish rows: quoted fields, "" escapes, CRLF/LF, a leading BOM. Blank lines dropped."""
    if text.startswith("﻿"):
        text = text[1:]
    rows: list[list[str]] = []
    field, row, quoted = "", [], False
    i = 0
    while i < len(text):
        ch = text[i]
        if quoted:
            if ch == '"':
                if i + 1 < len(text) and text[i + 1] == '"':
                    field += '"'
                    i += 1
                else:
                    quoted = False
            else:
                field += ch
        elif ch == '"':
            quoted = True
        elif ch == ",":
            row.append(field)
            field = ""
        elif ch == "\n":
            row.append(field)
            rows.append(row)
            field, row = "", []
        elif ch != "\r":
            field += ch
        i += 1
    row.append(field)
    rows.append(row)
    return [r for r in rows if not (len(r) == 1 and r[0].strip() == "")]


def coerce_number(raw: str) -> float:
    """A cell as a number: a leading currency sign dropped, thousands commas only when
    the value is unambiguously grouped. NaN when it is not a number."""
    trimmed = raw.strip()
    trimmed = re.sub(r"^[$€£]", "", trimmed)
    if trimmed:
        try:
            return float(trimmed)
        except ValueError:
            pass
    if _GROUPED.match(trimmed):
        return float(trimmed.replace(",", ""))
    return float("nan")


def looks_like_date(text: str) -> bool:
    """What JavaScript's Date.parse accepts, near enough: pandas parses it."""
    if not text or not text.strip():
        return False
    try:
        float(text)
        return False  # a bare number is not a date here, as in interfaze's header test
    except ValueError:
        pass
    import pandas as pd

    try:
        return not pd.isna(pd.to_datetime(text.strip(), errors="coerce"))
    except (ValueError, TypeError, OverflowError):
        return False


def rows_to_dataset(
    rows: list[list[str]], date_column: str | None = None, value_column: str | None = None
) -> list[dict]:
    """CSV rows as `{date, value}` points, choosing the columns the way interfaze does."""
    if not rows:
        return []
    first = [c.strip() for c in rows[0]]
    has_header = any(c and not _is_number(c) and not looks_like_date(c) for c in first)
    header = first if has_header else [f"column_{i}" for i in range(len(first))]
    lower = [h.lower() for h in header]
    count = len(header)

    def hint(name: str | None, kind: str) -> int:
        if not name:
            return -1
        try:
            return lower.index(name.strip().lower())
        except ValueError:
            raise DatasetError(
                f'{kind} column "{name}" not found. Available columns: {", ".join(header)}.'
            ) from None

    date_idx = hint(date_column, "Date")
    value_idx = hint(value_column, "Value")
    if date_idx == -1 and has_header:
        date_idx = next((i for i, h in enumerate(lower) if _DATE_HEADER.search(h)), -1)

    if count == 2:
        if date_idx == -1 and value_idx == -1:
            date_idx, value_idx = 0, 1
        elif value_idx == -1:
            value_idx = 1 if date_idx == 0 else 0
        elif date_idx == -1:
            date_idx = 1 if value_idx == 0 else 0

    # More than two columns, no value hint: exactly one numeric non-date column is taken;
    # several is genuinely ambiguous (date,open,close) and asks for value_column.
    if value_idx == -1 and date_idx != -1:
        sample = (rows[1:] if has_header else rows)[:5]
        numeric = [
            i
            for i in range(count)
            if i != date_idx
            and sample
            and all(i < len(r) and math.isfinite(coerce_number(r[i])) for r in sample)
        ]
        if len(numeric) == 1:
            value_idx = numeric[0]

    if date_idx == -1 or value_idx == -1 or date_idx == value_idx:
        raise DatasetError(
            f"Could not determine date and value columns from: {', '.join(header)}. "
            "Specify date_column and value_column explicitly."
        )

    dataset = []
    for r in rows[1:] if has_header else rows:
        if date_idx >= len(r) or value_idx >= len(r):
            continue
        date = (r[date_idx] or "").strip()
        value = coerce_number(r[value_idx] or "")
        if not date or not looks_like_date(date) or not math.isfinite(value):
            continue
        dataset.append({"date": date, "value": value})
    return dataset


def json_to_dataset(parsed, date_column: str | None = None, value_column: str | None = None) -> list[dict]:
    """`[{date, value}]`, `[[date, value]]`, or either wrapped in {data|dataset|series|predictions}."""
    arr = parsed
    if isinstance(parsed, dict):
        # JavaScript's `??`: the first key that is present, even when it holds [].
        arr = next(
            (parsed[k] for k in ("data", "dataset", "series", "predictions") if parsed.get(k) is not None),
            parsed,
        )
    if not isinstance(arr, list):
        return []
    date_keys = [k for k in (date_column, "date", "ds", "timestamp", "time", "day", "period") if k]
    value_keys = [k for k in (value_column, "value", "y", "val", "amount", "count") if k]
    dataset = []
    for row in arr:
        date, value = None, None
        if isinstance(row, list) and len(row) >= 2:
            date, value = str(row[0]).strip(), coerce_number(str(row[1]))
        elif isinstance(row, dict):
            by_lower = {k.lower(): k for k in row}
            dk = next((by_lower[k.lower()] for k in date_keys if k.lower() in by_lower), None)
            vk = next((by_lower[k.lower()] for k in value_keys if k.lower() in by_lower), None)
            if dk is not None:
                date = str(row[dk]).strip()
            if vk is not None:
                value = coerce_number(str(row[vk]))
        if not date or value is None or not looks_like_date(date) or not math.isfinite(value):
            continue
        dataset.append({"date": date, "value": value})
    return dataset


def parse_dataset(
    text: str,
    *,
    url: str = "",
    content_type: str = "",
    date_column: str | None = None,
    value_column: str | None = None,
) -> list[dict]:
    """A dataset file's text as points, JSON or CSV by content type, extension or first byte."""
    looks_json = (
        "json" in content_type.lower()
        or re.search(r"\.json(\?|#|$)", url, re.I)
        or re.match(r"^\s*[\[{]", text)
    )
    if looks_json:
        return json_to_dataset(json.loads(text), date_column, value_column)
    return rows_to_dataset(parse_csv(text), date_column, value_column)


def dataset_from_text(text: str) -> list[dict]:
    """The series written in a message: its longest run of rows with one comma count, as CSV; empty if none.

    Spares the model copying a pasted table into a tool call, which ran past the token limit (CUT_OFF_CALL).
    """
    best: list[str] = []
    run: list[str] = []
    shape = None
    for line in (text or "").splitlines():
        commas = line.count(",")
        if line.strip() and commas:
            if commas != shape:
                run, shape = [], commas
            run.append(line)
            if len(run) > len(best):
                best = list(run)
        else:
            run, shape = [], None
    # Drop a lead-in with the rows' comma count ("Forecast this, please:") until the rest reads as a table.
    for start in range(min(3, len(best))):
        if len(best) - start < MIN_FORECAST_POINTS:
            break
        try:
            points = rows_to_dataset(parse_csv("\n".join(best[start:])))
        except (DatasetError, ValueError):
            continue
        if len(points) >= MIN_FORECAST_POINTS:
            return points
    return []


def assert_fetchable_url(url: str) -> None:
    """Refuse non-http(s) URLs and addresses back into our own network, as interfaze does."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise DatasetError("Only http(s) dataset URLs are supported.")
    host = (parsed.hostname or "").lower()
    blocked = host in ("localhost", "0.0.0.0", "::1", "::") or host.endswith(".localhost")
    try:
        ip = ipaddress.ip_address(host)
        blocked = blocked or ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_unspecified
    except ValueError:
        pass
    if blocked:
        raise DatasetError(f"Refusing to fetch dataset from a non-public address ({parsed.hostname}).")


def series(dataset: list[dict]) -> dict[str, float]:
    """interfaze's pre-processing: a time on every date, sorted, same-date values summed."""
    points = [{**p, "date": p["date"] if ":" in p["date"] else f"{p['date']} 00:00:00"} for p in dataset]
    points.sort(key=lambda p: _timestamp(p["date"]))
    y: dict[str, float] = {}
    for p in points:
        v = float(p["value"])
        y[p["date"]] = y[p["date"]] + v if p["date"] in y else v
    if len(y) < MIN_FORECAST_POINTS:
        raise DatasetError("At least 5 unique dates are required")
    return y


# Ported from JigsawStack's prediction-timesfm service.


def parse_dates_sorted(y: dict[str, float]) -> tuple[list[datetime], list[float]]:
    import pandas as pd

    keys = list(y)
    idx = pd.to_datetime(keys, errors="coerce")
    bad = [keys[i] for i, ts in enumerate(idx) if pd.isna(ts)]
    if bad:
        raise DatasetError(f"Invalid date keys: {bad[:8]}{'...' if len(bad) > 8 else ''}")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    order = idx.argsort()
    return [idx[i].to_pydatetime() for i in order], [float(y[keys[i]]) for i in order]


def infer_step(dts: list[datetime]) -> timedelta:
    import numpy as np
    import pandas as pd

    if len(dts) < 3:
        return timedelta(days=1)
    try:
        guessed = pd.infer_freq(pd.DatetimeIndex(dts))
        if guessed:
            g = guessed.upper()
            if g.endswith("D"):
                return timedelta(days=int(g[:-1] or "1"))
            if g.endswith("H"):
                return timedelta(hours=int(g[:-1] or "1"))
            if g.endswith(("T", "MIN")):
                n = g[:-1] if g.endswith("T") else g.replace("MIN", "")
                return timedelta(minutes=int(n or "1"))
            if g.endswith("S"):
                return timedelta(seconds=int(g[:-1] or "1"))
    except Exception:
        pass
    diffs = np.diff(np.array(dts, dtype="datetime64[s]")).astype("timedelta64[s]").astype(int)
    diffs = diffs[diffs > 0]
    return timedelta(seconds=int(np.median(diffs))) if diffs.size else timedelta(days=1)


def future_timestamps(last: datetime, horizon: int, step: timedelta) -> list[str]:
    start = last + step
    return [(start + i * step).strftime("%Y-%m-%d %H:%M:%S") for i in range(horizon)]


def all_integers(values) -> bool:
    return all(
        not isinstance(v, bool) and isinstance(v, (int, float)) and abs(v - round(v)) < 1e-9 for v in values
    )


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _timestamp(text: str) -> float:
    import pandas as pd

    ts = pd.to_datetime(text, errors="coerce")
    return float("inf") if pd.isna(ts) else ts.value
