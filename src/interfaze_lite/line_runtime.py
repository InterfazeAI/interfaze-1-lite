"""The line detector and layout detector, each engine in a process of its own.

Threads share Paddle's process-wide oneDNN context, whose cached-object count is read
unlocked (`OneDNNContext::GetCachedObjectsNumber`); under concurrent OCR that segfaulted
the whole service. A process per engine has nothing to race, and a crash costs one retried call.
Kept free of torch and this package's other modules, so workers start with nothing but Paddle.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import functools
import logging
import multiprocessing
import os
import tempfile
import threading
from concurrent.futures.process import BrokenProcessPool
from typing import Any

try:
    import fcntl
except ImportError:  # Windows has no flock; there the first builds are not serialised
    fcntl = None

log = logging.getLogger("interfaze.line_runtime")

# Engines built in this worker process, from the settings the parent handed it.
_ENGINES: dict[str, Any] = {}
_SETTINGS: dict[str, Any] = {}


def _init(line_kwargs: dict, layout_kwargs: dict | None) -> None:
    _SETTINGS["line"], _SETTINGS["layout"] = line_kwargs, layout_kwargs


_BUILD_LOCK_PATH = os.path.join(tempfile.gettempdir(), "interfaze-line-runtime.lock")


@contextlib.contextmanager
def _build_lock():
    """One engine build at a time across workers: first builds unpacking at once read half-written models."""
    if fcntl is None:
        yield
        return
    with open(_BUILD_LOCK_PATH, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _engine(kind: str):
    if kind not in _ENGINES:
        with _build_lock():
            if kind == "line":
                from paddleocr import PaddleOCR

                try:  # 3.x
                    _ENGINES[kind] = PaddleOCR(**_SETTINGS["line"])
                except (TypeError, ValueError):  # 2.x
                    _ENGINES[kind] = PaddleOCR(use_angle_cls=True, lang="en")
            else:
                from paddleocr import LayoutDetection

                _ENGINES[kind] = LayoutDetection(**_SETTINGS["layout"])
    return _ENGINES[kind]


def _dropping_on_failure(kind: str):
    """Drop an engine that raised; an aborted predictor stays broken (PaddleOCR #16238)."""

    def wrap(fn):
        @functools.wraps(fn)
        def run(*args):
            try:
                return fn(*args)
            except Exception:
                _ENGINES.pop(kind, None)
                raise

        return run

    return wrap


@_dropping_on_failure("line")
def line_rows(arr) -> list[tuple[list, str, float]]:
    """Detected lines in an RGB array: (quad, text, score) for each."""
    engine = _engine("line")
    rows: list[tuple[list, str, float]] = []
    if hasattr(engine, "predict"):
        for page in engine.predict(input=arr) or []:
            data = page.get("res", page) if isinstance(page, dict) else page
            polys = data.get("rec_polys") or data.get("dt_polys") or []
            texts = data.get("rec_texts") or []
            scores = data.get("rec_scores") or []
            for poly, text, score in zip(polys, texts, scores, strict=True):
                rows.append(([[float(x), float(y)] for x, y in poly], str(text), float(score)))
    else:
        for quad, (text, score) in (engine.ocr(arr, cls=True) or [[]])[0] or []:
            rows.append(([[float(x), float(y)] for x, y in quad], str(text), float(score)))
    return rows


@_dropping_on_failure("layout")
def layout_pages(arr) -> list:
    """Layout detection on an RGB array, as plain data that can cross back to the parent."""
    pages = _engine("layout").predict(arr, batch_size=1, layout_nms=True)
    return [_plain(getattr(page, "json", page)) for page in (pages or [])]


def _plain(value):
    """Result objects and numpy values as builtins, so they pickle across processes."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if callable(getattr(value, "tolist", None)):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class Runtime:
    """A pool of engine processes. A worker that dies is replaced and its call retried once."""

    def __init__(self, workers: int, line_kwargs: dict, layout_kwargs: dict | None):
        self._workers = max(1, workers)
        self._init_args = (line_kwargs, layout_kwargs)
        self._pool: concurrent.futures.ProcessPoolExecutor | None = None
        self._lock = threading.Lock()

    def _current(self) -> concurrent.futures.ProcessPoolExecutor:
        with self._lock:
            if self._pool is None:
                # spawn, not fork: a forked child inherits the parent's CUDA context broken.
                self._pool = concurrent.futures.ProcessPoolExecutor(
                    max_workers=self._workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_init,
                    initargs=self._init_args,
                )
            return self._pool

    def _replace(self, broken: concurrent.futures.ProcessPoolExecutor) -> None:
        with self._lock:
            if self._pool is broken:  # another caller may have replaced it already
                self._pool = None
                broken.shutdown(wait=False, cancel_futures=True)

    def run(self, fn, *args):
        for attempt in (1, 2):
            pool = self._current()
            try:
                return pool.submit(fn, *args).result()
            except BrokenProcessPool:
                log.warning("a detection worker died (attempt %d); replacing the pool", attempt)
                self._replace(pool)
                if attempt == 2:
                    raise
        raise RuntimeError("unreachable")
