"""
Regression tests for the fix to silent CRITICAL/PAGE alert loss in
AlertDispatcher.
"""

import json

import pytest

from alert_agent import Alert, AlertChannel, AlertDispatcher, AlertPriority


def _make_dispatcher(tmp_path, routing, max_retries=3):
    return AlertDispatcher(
        routing=routing,
        max_retries=max_retries,
        backoff_base_seconds=0.0,
        dead_letter_path=str(tmp_path / "dead_letter.jsonl"),
        sleep_fn=lambda _seconds: None,
    )


def test_page_alert_retries_before_giving_up(tmp_path):
    calls = {"n": 0}

    def flaky_sender(alert):
        calls["n"] += 1
        raise RuntimeError("smtp down")

    dispatcher = _make_dispatcher(tmp_path, {AlertPriority.PAGE: [AlertChannel.PAGER]}, max_retries=3)
    dispatcher.register(AlertChannel.PAGER, flaky_sender)

    alert = Alert(source_agent="performance_drift", priority=AlertPriority.PAGE, title="t", message="m")
    dispatcher.dispatch([alert])

    assert calls["n"] == 3


def test_page_alert_all_channels_failing_writes_dead_letter(tmp_path):
    def always_fails(alert):
        raise RuntimeError("down")

    dead_letter_path = tmp_path / "dead_letter.jsonl"
    dispatcher = AlertDispatcher(
        routing={AlertPriority.PAGE: [AlertChannel.PAGER, AlertChannel.EMAIL]},
        max_retries=2,
        backoff_base_seconds=0.0,
        dead_letter_path=str(dead_letter_path),
        sleep_fn=lambda _seconds: None,
    )
    dispatcher.register(AlertChannel.PAGER, always_fails)
    dispatcher.register(AlertChannel.EMAIL, always_fails)

    alert = Alert(source_agent="performance_drift", priority=AlertPriority.PAGE, title="collapse", message="m")
    dispatcher.dispatch([alert])

    assert dead_letter_path.exists()
    lines = dead_letter_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["alert"]["title"] == "collapse"
    assert set(entry["channel_failures"]) == {"pager", "email"}


def test_page_alert_succeeds_on_second_try_no_dead_letter(tmp_path):
    calls = {"n": 0}

    def succeeds_on_second_try(alert):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient")

    dead_letter_path = tmp_path / "dead_letter.jsonl"
    dispatcher = _make_dispatcher(tmp_path, {AlertPriority.CRITICAL: [AlertChannel.SLACK]}, max_retries=3)
    dispatcher.register(AlertChannel.SLACK, succeeds_on_second_try)

    alert = Alert(source_agent="data_drift", priority=AlertPriority.CRITICAL, title="t", message="m")
    dispatcher.dispatch([alert])

    assert calls["n"] == 2
    assert not dead_letter_path.exists()


def test_info_priority_is_not_retried():
    """Only CRITICAL/PAGE justify the cost of a retry -- an INFO alert that
    fails once shouldn't retry 3 times for nothing."""
    calls = {"n": 0}

    def always_fails(alert):
        calls["n"] += 1
        raise RuntimeError("down")

    dispatcher = AlertDispatcher(
        routing={AlertPriority.INFO: [AlertChannel.LOG]},
        max_retries=3,
        backoff_base_seconds=0.0,
        dead_letter_path="/tmp/unused_dead_letter.jsonl",
        sleep_fn=lambda _seconds: None,
    )
    dispatcher.register(AlertChannel.LOG, always_fails)

    alert = Alert(source_agent="data_drift", priority=AlertPriority.INFO, title="t", message="m")
    dispatcher.dispatch([alert])

    assert calls["n"] == 1
