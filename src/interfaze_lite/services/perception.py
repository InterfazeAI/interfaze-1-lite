"""Perception sidecar: OCR, segmentation, ASR, guardrails and forecasting on one GPU.

Every model here shares a torch build and so a process; diarization runs apart (services/diarize.py).
The work is `InterfazeLiteModel`'s, so this and the `transformers` repo cannot drift: HTTP in, shapes out.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..contracts import InputFetchError

log = logging.getLogger("interfaze.perception")

_model: Any = None
_model_lock = threading.Lock()


@asynccontextmanager
async def _lifespan(_app):
    preload()
    yield


app = FastAPI(title="interfaze-lite perception", docs_url=None, redoc_url=None, lifespan=_lifespan)


def _free_vram() -> None:
    """Release cached CUDA blocks after a failure, so the next request does not OOM on memory nothing uses."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# Set by a failed request. FastAPI's threadpool holds its exception, and through the traceback every
# tensor, in a cycle: an OOM'd transcription still held 16 GB until one collection freed it.
# Collecting inside the failed request frees nothing, as the exception is still live.
_failed = threading.Event()


def _collect_failure() -> None:
    """Free what a failed request left allocated, once, before the next request runs."""
    if _failed.is_set():
        _failed.clear()
        gc.collect()
        _free_vram()


def _describe(exc: BaseException) -> str:
    """Always include the exception type. Several exceptions -- StopIteration most
    painfully -- stringify to the empty string, turning a 500 into "ocr failed: "."""
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def model() -> Any:
    """Instantiate the composite model once; its components still load on first use."""
    global _model
    if _model is None:
        with _model_lock:
            # Re-checked in the lock: three concurrent first requests each built a model while
            # transformers flipped the process-wide dtype, and the one kept was part bf16, part float32.
            if _model is None:
                from ..configuration_interfaze_lite import InterfazeLiteConfig
                from ..modeling_interfaze_lite import InterfazeLiteModel

                log.info("initialising composite model (components load on first use)")
                _model = InterfazeLiteModel(InterfazeLiteConfig(lazy_load=True))
    return _model


def preload() -> None:
    """Load the components named in PERCEPTION_PRELOAD before taking any request.

    A first-use load ran beside live requests, changing process-wide torch state, and charged the
    first request after each deploy. Unset, everything stays lazy.
    """
    names = [n.strip() for n in os.environ.get("PERCEPTION_PRELOAD", "").split(",") if n.strip()]
    for name in names:
        log.info("preloading %s", name)
        model()._component(name)


class OCRRequest(BaseModel):
    url: str
    return_markdown: bool = False
    page_range: list[int] | None = None


class SegmentRequest(BaseModel):
    url: str
    boxes: list[list[float]] = Field(default_factory=list)
    # Boxes on a 0-1000 grid, rescaled here: asking for the dimensions first is a second download.
    normalized: bool = False
    # Masks run the segmenter's image encoder, this endpoint's whole cost; boxes alone are arithmetic.
    return_masks: bool = True
    # The same encoder pass as masks, but a few dozen coordinates to return, not a megabyte of base64.
    return_outlines: bool = False


class TranscribeRequest(BaseModel):
    url: str
    language: str = "auto"
    word_timestamps: bool = False
    # Words for joining speakers to: decoded, or estimated on a long recording.
    words_for_speakers: bool = False
    # Speakers are joined by the orchestrator against the diarization sidecar: the two stay independent.


class GuardRequest(BaseModel):
    text: str = ""


class ForecastRequest(BaseModel):
    # interfaze's prediction payload: the horizon, and the series as date -> value.
    fh: int
    y: dict[str, float]


CAPABILITY = {
    "asr": "stt",
    "asr_fallback": "stt",
    "line_detector": "text_detection",
    "segmenter": "segmentation",
    "ocr_vlm": "document_reading",
    "diarizer": "diarization",
    "brain": "reasoning",
    "guard": "guardrails",
    "forecaster": "forecasting",
}


@app.get("/health")
def health() -> dict[str, Any]:
    # Capability names, not components: which checkpoint backs a capability is no promise to callers.
    loaded = sorted({CAPABILITY.get(name, name) for name in (_model._cache if _model is not None else ())})
    token_vars = [
        k
        for k in (
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "HUGGINGFACE_TOKEN",
            "hf_token",
            "HUGGINGFACE_HUB_TOKEN",
        )
        if os.environ.get(k)
    ]
    return {
        "ok": True,
        "initialised": _model is not None,
        "components_loaded": loaded,
        # Names only, never values.
        "hf_token_vars": token_vars,
    }


@app.post("/ocr")
def ocr(req: OCRRequest) -> dict[str, Any]:
    _collect_failure()
    try:
        return model().ocr(req.url, return_markdown=req.return_markdown, page_range=req.page_range)
    except InputFetchError as exc:
        # The caller's file, not this service: 422, so the tool layer can say so.
        raise HTTPException(status_code=422, detail={"input_fetch": str(exc)}) from exc
    except Exception as exc:  # surfaced to the tool layer, which reports it to the model
        _free_vram()
        _failed.set()
        log.exception("ocr failed")
        raise HTTPException(status_code=500, detail=f"ocr failed: {_describe(exc)}") from exc


@app.post("/segment")
def segment(req: SegmentRequest) -> dict[str, Any]:
    """Masks and outlines from box prompts.

    The boxes come from the vLLM-served brain; this turns them into masks and outlines.
    SAM 2 is plain PyTorch, with no transformers dependency to skew against the brain's.
    """
    _collect_failure()
    try:
        composite = model()
        img = composite._fit_for_ocr(req.url)
        if not req.boxes:
            # Dimensions are still useful to a caller holding normalised boxes.
            return {"masks": [], "outlines": [], "boxes": [], "width": img.width, "height": img.height}

        if req.normalized:
            boxes = [
                [x1 / 1000 * img.width, y1 / 1000 * img.height, x2 / 1000 * img.width, y2 / 1000 * img.height]
                for x1, y1, x2, y2 in req.boxes
            ]
        else:
            boxes = [[float(v) for v in b] for b in req.boxes]

        masks: list[str] = []
        outlines: list[Any] = []
        if req.return_masks or req.return_outlines:
            masks, outlines = composite._segment_boxes_with_outlines(img, boxes)
            if not req.return_masks:
                masks = []

        return {
            "masks": masks,
            "outlines": outlines,
            # Echoed back in pixel space so the caller does not have to redo the maths.
            "boxes": [[round(v) for v in b] for b in boxes],
            "width": img.width,
            "height": img.height,
        }
    except InputFetchError as exc:
        raise HTTPException(status_code=422, detail={"input_fetch": str(exc)}) from exc
    except Exception as exc:
        _free_vram()
        _failed.set()
        log.exception("segment failed")
        raise HTTPException(status_code=500, detail=f"segment failed: {_describe(exc)}") from exc


@app.post("/transcribe")
def transcribe(req: TranscribeRequest) -> dict[str, Any]:
    # The word-timestamp pass is this service's largest transient allocation, on a shared card:
    # returning cached blocks first stops it OOMing on memory nothing uses.
    _collect_failure()
    _free_vram()
    try:
        # Words only when asked: they cost four times what segments do. The model takes the
        # recogniser per batch, not per recording (`_RECOGNISER`), so a long file holds up no one.
        result = model().transcribe(
            req.url,
            by_speaker=False,
            language=req.language,
            word_timestamps=req.word_timestamps,
            words_for_speakers=req.words_for_speakers,
        )
    except NotImplementedError as exc:
        # Logged: returned only, a model-loading failure showed in the logs as a bare "501".
        log.exception("transcribe unsupported")
        raise HTTPException(status_code=501, detail=_describe(exc)) from exc
    except InputFetchError as exc:
        raise HTTPException(status_code=422, detail={"input_fetch": str(exc)}) from exc
    except Exception as exc:
        _free_vram()
        _failed.set()
        log.exception("transcribe failed")
        raise HTTPException(status_code=500, detail=f"transcribe failed: {_describe(exc)}") from exc
    # Given back now: diarization runs next on the same card, and after a 95-minute transcription
    # it ran out of memory on 158 MB.
    _free_vram()
    return result


# One at a time, like transcription: each is a short GPU pass on a card other models
# share, and serialising them keeps their peak allocation to one request's.
_guard_lock = threading.Lock()
_forecast_lock = threading.Lock()


@app.post("/guard")
def guard(req: GuardRequest) -> dict[str, Any]:
    _collect_failure()
    try:
        with _guard_lock:
            return model().moderate(req.text)
    except Exception as exc:
        _free_vram()
        _failed.set()
        log.exception("guard failed")
        raise HTTPException(status_code=500, detail=f"guard failed: {_describe(exc)}") from exc


@app.post("/forecast")
def forecast(req: ForecastRequest) -> dict[str, Any]:
    from ..forecasting import DatasetError

    _collect_failure()
    try:
        with _forecast_lock:
            return model().forecast(req.y, req.fh)
    except (DatasetError, ValueError) as exc:
        # The series, not this service: interfaze returns these as the caller's error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _free_vram()
        _failed.set()
        log.exception("forecast failed")
        raise HTTPException(status_code=500, detail=f"forecast failed: {_describe(exc)}") from exc
