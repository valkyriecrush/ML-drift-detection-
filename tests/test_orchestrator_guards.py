"""
Regression tests for the production guard on in-process-memory backends
(checkpointer + stateful agent registry).
"""

import importlib

import pytest


@pytest.fixture
def orchestrator_module():
    import orchestrator

    importlib.reload(orchestrator)
    yield orchestrator
    importlib.reload(orchestrator)  # reset module-level singletons for other tests


def test_default_checkpointer_raises_in_production_without_override(orchestrator_module, monkeypatch):
    monkeypatch.setenv("AGENTIC_ENV", "production")
    monkeypatch.delenv("AGENTIC_ALLOW_INMEMORY_STATE", raising=False)

    with pytest.raises(RuntimeError, match="AGENTIC_ENV=production"):
        orchestrator_module.get_default_checkpointer()


def test_default_checkpointer_allowed_with_explicit_override(orchestrator_module, monkeypatch):
    monkeypatch.setenv("AGENTIC_ENV", "production")
    monkeypatch.setenv("AGENTIC_ALLOW_INMEMORY_STATE", "1")

    checkpointer = orchestrator_module.get_default_checkpointer()
    assert checkpointer is not None


def test_default_checkpointer_works_outside_production(orchestrator_module, monkeypatch):
    monkeypatch.delenv("AGENTIC_ENV", raising=False)
    checkpointer = orchestrator_module.get_default_checkpointer()
    assert checkpointer is not None


def test_batch_registry_round_trip_is_thread_safe(orchestrator_module):
    import threading

    errors = []

    def worker(i):
        try:
            orchestrator_module.stage_batch(f"batch-{i}", features_df=None, y_true=i)
            data = orchestrator_module.get_batch(f"batch-{i}")
            assert data["y_true"] == i
            orchestrator_module.clear_batch(f"batch-{i}")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
