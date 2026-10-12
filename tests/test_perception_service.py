"""The perception service's model: built once, and loaded before it takes traffic."""

import sys
import threading
import time
import types

import pytest
from fastapi import HTTPException

from interfaze_lite.services import perception


@pytest.fixture
def fake_model(monkeypatch):
    """The composite model and its config, without torch: counts builds and component loads."""
    built, loaded = [], []

    class FakeModel:
        def __init__(self, config):
            time.sleep(0.05)  # long enough for concurrent first callers to overlap
            built.append(self)

        def _component(self, name):
            loaded.append(name)

    modeling = types.ModuleType("interfaze_lite.modeling_interfaze_lite")
    modeling.InterfazeLiteModel = FakeModel
    configuration = types.ModuleType("interfaze_lite.configuration_interfaze_lite")
    configuration.InterfazeLiteConfig = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "interfaze_lite.modeling_interfaze_lite", modeling)
    monkeypatch.setitem(sys.modules, "interfaze_lite.configuration_interfaze_lite", configuration)
    monkeypatch.setattr(perception, "_model", None)
    return built, loaded


def test_concurrent_first_requests_build_one_model(fake_model):
    """Three at once each built their own, and the one kept decoded garbage."""
    built, _ = fake_model
    seen = []
    threads = [threading.Thread(target=lambda: seen.append(perception.model())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(built) == 1
    assert all(m is built[0] for m in seen)


def test_the_named_components_load_before_traffic(fake_model, monkeypatch):
    _, loaded = fake_model
    monkeypatch.setenv("PERCEPTION_PRELOAD", "asr_fallback, segmenter,guard,forecaster")
    perception.preload()
    assert loaded == ["asr_fallback", "segmenter", "guard", "forecaster"]


def test_nothing_loads_early_unless_asked(fake_model, monkeypatch):
    built, loaded = fake_model
    monkeypatch.delenv("PERCEPTION_PRELOAD", raising=False)
    perception.preload()
    assert built == [] and loaded == []


def test_a_failed_request_is_collected_before_the_next_runs(monkeypatch):
    """An OOM transcription held 16 GB after answering; collecting inside the failure freed nothing."""

    class Model:
        def transcribe(self, *args, **kwargs):
            raise RuntimeError("CUDA out of memory")

        def moderate(self, text):
            return {"safe": True}

    collected = []
    monkeypatch.setattr(perception, "model", lambda: Model())
    monkeypatch.setattr(perception.gc, "collect", lambda: collected.append(1))
    perception._failed.clear()

    perception.guard(perception.GuardRequest(text="hi"))
    assert collected == []
    with pytest.raises(HTTPException):
        perception.transcribe(perception.TranscribeRequest(url="https://example.com/a.mp3"))
    assert collected == []
    perception.guard(perception.GuardRequest(text="hi"))
    assert collected == [1]
    perception.guard(perception.GuardRequest(text="hi"))
    assert collected == [1]
