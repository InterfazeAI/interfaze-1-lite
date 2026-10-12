"""The detector worker pool: a worker that dies is replaced, and results cross processes."""

import array
import functools
import multiprocessing
import os
import pickle
import time

import pytest

from interfaze_lite import line_runtime
from interfaze_lite.line_runtime import Runtime, _plain


def test_a_dead_worker_is_replaced_and_the_pool_keeps_serving():
    # os._exit stands in for a detector segfault, which used to take the whole service down.
    runtime = Runtime(1, {}, None)
    with pytest.raises(line_runtime.BrokenProcessPool):
        runtime.run(functools.partial(os._exit, 1))  # dies on the first try and the retry
    assert runtime.run(pow, 2, 10) == 1024


def test_results_cross_back_as_plain_data():
    class Result:
        tolist = None

    value = {
        "res": {"boxes": [{"coordinate": array.array("d", [1.0, 2.0]), "label": "text"}]},
        "obj": Result(),
    }
    plain = _plain(value)
    assert plain["res"]["boxes"][0]["coordinate"] == [1.0, 2.0]
    assert isinstance(plain["obj"], str)


def test_an_engine_that_raised_is_not_reused(monkeypatch):
    """A predictor that aborted stays broken; the worker would keep it for its life."""

    class Faulted:
        def predict(self, **_):
            raise RuntimeError("std::exception")

    monkeypatch.setitem(line_runtime._ENGINES, "line", Faulted())
    with pytest.raises(RuntimeError):
        line_runtime.line_rows(None)
    assert "line" not in line_runtime._ENGINES


def test_the_worker_functions_pickle_by_name():
    # Worker processes are spawned: every function sent to them must pickle by reference.
    for fn in (line_runtime.line_rows, line_runtime.layout_pages):
        assert pickle.loads(pickle.dumps(fn)) is fn


def _hold_build_lock(marker, overlaps):
    with line_runtime._build_lock():
        if os.path.exists(marker):
            overlaps.value += 1
        open(marker, "w").close()
        time.sleep(0.2)
        os.remove(marker)


@pytest.mark.skipif(line_runtime.fcntl is None, reason="no flock on this platform")
def test_engine_builds_run_one_at_a_time_across_processes(tmp_path):
    """Concurrent engine builds read each other's half-unpacked model folders (`inference.yml` not found)."""
    ctx = multiprocessing.get_context("fork")
    overlaps = ctx.Value("i", 0)
    marker = str(tmp_path / "building")
    workers = [ctx.Process(target=_hold_build_lock, args=(marker, overlaps)) for _ in range(3)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(10)
    assert overlaps.value == 0
