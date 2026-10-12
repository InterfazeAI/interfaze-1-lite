"""Forecasting a time series with TimesFM.

interfaze's `forecast` tool: the same parameters, the same dataset handling
(`forecasting.py`), the same errors, and the same `{predictions: [{date, value}]}`
result, oldest first. The model runs in the perception service.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from .. import forecasting
from .base import Tool, ToolContext, ToolResult, _post


async def _read_file(ctx: ToolContext, file_ref_id: str) -> tuple[str, str, str]:
    """(text, url-or-path, content type) of a referenced dataset file."""
    location = ctx.refs.resolve(file_ref_id)
    if location.startswith(("http://", "https://")):
        forecasting.assert_fetchable_url(location)
        try:
            async with ctx.http.stream("GET", location, timeout=10) as resp:
                if resp.status_code >= 400:
                    raise forecasting.DatasetError(
                        f"Failed to fetch dataset from {location} (HTTP {resp.status_code})."
                    )
                length = resp.headers.get("content-length")
                if length and length.isdigit() and int(length) > forecasting.MAX_DATASET_BYTES:
                    raise forecasting.DatasetError(
                        f"Dataset file is too large ({length} bytes; max {forecasting.MAX_DATASET_BYTES})."
                    )
                body = bytearray()
                async for chunk in resp.aiter_bytes():
                    body += chunk
                    if len(body) > forecasting.MAX_DATASET_BYTES:
                        raise forecasting.DatasetError(
                            f"Dataset file is too large (exceeds {forecasting.MAX_DATASET_BYTES} bytes)."
                        )
                return bytes(body).decode("utf-8", "replace"), location, resp.headers.get("content-type", "")
        except httpx.HTTPError as exc:
            raise forecasting.DatasetError(
                f"Failed to fetch dataset from {location} ({type(exc).__name__})."
            ) from exc

    path = Path(location)
    if path.stat().st_size > forecasting.MAX_DATASET_BYTES:
        raise forecasting.DatasetError(
            f"Dataset file is too large (exceeds {forecasting.MAX_DATASET_BYTES} bytes)."
        )
    ref = next((r for r in ctx.refs.refs.values() if r.url == location), None)
    text = await asyncio.to_thread(path.read_text, "utf-8", "replace")
    return text, (ref.filename if ref and ref.filename else location), (ref.mime if ref else "")


async def _run_forecast(args: dict, ctx: ToolContext) -> ToolResult:
    retry = (
        "Retry with a valid file_ref_id/URL or an inline dataset (and date_column/value_column "
        "if the file has more than 2 columns)."
    )
    try:
        steps = int(args.get("steps"))
    except (TypeError, ValueError):
        return ToolResult(model_facing={"error": "steps must be a number", "message": retry})

    try:
        dataset = args.get("dataset")
        file_ref_id = args.get("file_ref_id")
        if file_ref_id:
            text, where, content_type = await _read_file(ctx, file_ref_id)
            dataset = forecasting.parse_dataset(
                text,
                url=where,
                content_type=content_type,
                date_column=args.get("date_column"),
                value_column=args.get("value_column"),
            )
            if len(dataset) < forecasting.MIN_FORECAST_POINTS:
                raise forecasting.DatasetError(
                    f"Parsed only {len(dataset)} valid data point(s) from {where}; need at least "
                    f"{forecasting.MIN_FORECAST_POINTS}. Check the file format and column names."
                )

        inline_is_usable = isinstance(dataset, list) and len(dataset) >= forecasting.MIN_FORECAST_POINTS
        if not file_ref_id and not inline_is_usable:
            # The series as written in the message, read here instead of copied into
            # the call -- the tool description says to leave it out.
            dataset = forecasting.dataset_from_text(ctx.prompt) or dataset

        if not dataset or len(dataset) < forecasting.MIN_FORECAST_POINTS:
            return ToolResult(
                model_facing={
                    "error": f"Need at least {forecasting.MIN_FORECAST_POINTS} historical data points to forecast.",
                    "message": "Provide the historical series via file_ref_id (CSV/JSON file or URL) or an inline dataset.",
                }
            )
        # Refused rather than truncated: a partial series gives a misleading forecast.
        if len(dataset) > forecasting.MAX_FORECAST_POINTS:
            return ToolResult(
                model_facing={
                    "error": (
                        f"We support a maximum of {forecasting.MAX_FORECAST_POINTS} rows per forecast "
                        f"request; received {len(dataset)}."
                    ),
                    "forecast_row_limit_exceeded": True,
                }
            )

        y = forecasting.series(dataset)
        out = await _post(ctx, ctx.settings.perception_url, "/forecast", {"fh": steps, "y": y})
        predictions = [{"date": d, "value": v} for d, v in zip(out["timestamp"], out["value"], strict=True)]
        return ToolResult(model_facing={"predictions": predictions})
    except Exception as exc:
        return ToolResult(model_facing={"error": str(exc) or type(exc).__name__, "message": retry})


FORECAST = Tool(
    name="forecast",
    description=(
        "Forecast/predict future values of a numeric time series. Never reproduce the "
        "historical data in tool params: when it is in a file or at a URL, pass `file_ref_id` "
        "(a ref-N from the file references block, or a CSV/JSON URL); when it is written out "
        "in the user's message, omit both `file_ref_id` and `dataset` and it is read from the "
        "message. Use inline `dataset` only for a series that appears in neither. "
        "`steps` is how many future points to predict — read it from the user's request "
        '(e.g. "next 7 days" => 7). Use only when the user asks for a forecast or prediction '
        "of future values."
    ),
    parameters={
        "type": "object",
        "properties": {
            "steps": {
                "type": "number",
                "description": (
                    "The number of future steps to forecast ahead (from the user's "
                    "request, e.g. 'next 7 days' => 7)"
                ),
            },
            "file_ref_id": {
                "type": "string",
                "description": (
                    "Preferred. A file reference id (e.g. ref-0) or a CSV/JSON URL of the "
                    "historical series. Reads the data server-side so it never needs to "
                    "be reproduced here. Use whenever the data is in a file or at a URL."
                ),
            },
            "date_column": {
                "type": "string",
                "description": (
                    "Name of the date column in the file. Optional; inferred for 2-column "
                    "files. Provide when the file has more than 2 columns."
                ),
            },
            "value_column": {
                "type": "string",
                "description": (
                    "Name of the numeric value column in the file. Optional; inferred for "
                    "2-column files. Provide when the file has more than 2 columns."
                ),
            },
            "dataset": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number", "description": "The numerical value of the data"},
                        "date": {
                            "type": "string",
                            "description": "The date of the data in YYYY-MM-DD format",
                        },
                    },
                    "required": ["value", "date"],
                    "additionalProperties": False,
                },
                "description": (
                    "Only for a series that is in neither a file nor the user's "
                    "message. Data written in the message is read from it; leave "
                    "this out."
                ),
            },
        },
        "required": ["steps"],
        "additionalProperties": False,
    },
    execute=_run_forecast,
)
