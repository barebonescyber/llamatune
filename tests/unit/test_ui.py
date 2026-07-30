from __future__ import annotations

import io
import json
from typing import Any, cast

import pytest

from llamatune import ui
from llamatune.types import ProgressEvent


def _event(event_kind: str, **payload: object) -> ProgressEvent:
    return ProgressEvent(kind=event_kind, ts="2026-01-01T00:00:00+00:00", payload=dict(payload))


def test_none_reporter() -> None:
    assert ui.make_reporter("none", err=io.StringIO()) is None


def test_json_reporter_emits_ndjson() -> None:
    err = io.StringIO()
    reporter = ui.make_reporter("json", err=err)
    assert reporter is not None
    reporter.emit(_event("warning", message="careful"))
    assert json.loads(err.getvalue()) == {
        "kind": "warning",
        "ts": "2026-01-01T00:00:00+00:00",
        "message": "careful",
    }


def test_plain_reporter_throttles_heartbeats() -> None:
    err = io.StringIO()
    reporter = ui.make_reporter("plain", err=err)
    assert reporter is not None
    reporter.emit(_event("exec_heartbeat", label="trial", elapsed_s=2.0, timeout_s=60.0))
    reporter.emit(_event("exec_heartbeat", label="trial", elapsed_s=10.0, timeout_s=60.0))
    reporter.emit(_event("exec_heartbeat", label="trial", elapsed_s=32.0, timeout_s=60.0))
    assert err.getvalue().count("[running]") == 2


def test_plain_reporter_core_events() -> None:
    err = io.StringIO()
    reporter = ui.make_reporter("plain", err=err)
    assert reporter is not None
    events = [
        _event("session_start", model_name="model", session_dir="session"),
        _event("stage", stage="search"),
        _event("exec_start", kind="trial", label="ngl=1"),
        _event("exec_end", label="ngl=1", status="ok", wall_s=1.2, pp=2.0, tg=3.0),
        _event("incumbent", score=1.2),
        _event("depth_profile", depth=32768),
        _event("envelope", ctx=65536, status="failed"),
        _event("warning", message="noise"),
        _event("session_end", exit_code=0, winner_trial_id="abc"),
    ]
    for event in events:
        reporter.emit(event)
    text = err.getvalue()
    assert "[session]" in text
    assert "[result]" in text
    assert "winner=abc" in text
    assert "[depth profile] d=32768" in text
    assert "[envelope] ctx=65536 — failed" in text


def test_auto_non_tty_uses_plain() -> None:
    err = io.StringIO()
    reporter = ui.make_reporter("auto", err=err)
    assert isinstance(reporter, ui.PlainReporter)
    reporter.emit(_event("warning", message="plain cmd output"))
    assert "\x1b" not in err.getvalue()


class _TtyBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_auto_tty_uses_rich(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = ui.PlainReporter(io.StringIO())
    monkeypatch.setattr(ui, "RichReporter", lambda err: sentinel)
    assert ui.make_reporter("auto", err=_TtyBuffer()) is sentinel


def test_rich_failure_warns_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(err: io.StringIO) -> ui.RichReporter:
        raise RuntimeError("rich failed")

    err = io.StringIO()
    monkeypatch.setattr(ui, "RichReporter", fail)
    reporter = ui.make_reporter("rich", err=err)
    assert isinstance(reporter, ui.PlainReporter)
    assert "Rich progress unavailable" in err.getvalue()


def test_plain_reporter_stop_hint_status_without_metrics_and_unknown_event() -> None:
    err = io.StringIO()
    reporter = ui.PlainReporter(err)
    reporter.emit(
        _event(
            "session_start",
            model_name="model",
            session_dir="session",
            stop_hint="Ctrl-C once to confirm best; twice to abort",
        )
    )
    reporter.emit(_event("exec_end", label="ngl=24", status="cuda_error", wall_s=2.5))
    reporter.emit(_event("not_a_real_event"))
    text = err.getvalue()
    assert "Ctrl-C once to confirm best; twice to abort" in text
    assert "cuda_error" in text
    assert "pp=" not in text


class _FakeLive:
    def __init__(self) -> None:
        self.updates: list[tuple[str, bool]] = []
        self.stopped = False

    def update(self, text: str, *, refresh: bool) -> None:
        self.updates.append((text, refresh))

    def stop(self) -> None:
        self.stopped = True


def test_rich_reporter_all_event_updates() -> None:
    reporter = object.__new__(ui.RichReporter)
    reporter._lines = ["llamatune starting…"]
    live = _FakeLive()
    reporter._live = cast(Any, live)
    events = [
        _event("session_start", model_name="model", session_dir="session", stop_hint="stop hint"),
        _event("stage", stage="search"),
        _event("exec_start", label="trial", elapsed_s=0),
        _event("exec_heartbeat", label="trial", elapsed_s=12),
        _event("exec_end", label="trial", status="ok"),
        _event("incumbent", score=1.25),
        *[_event("warning", message=f"warning-{index}") for index in range(14)],
        _event("session_end", exit_code=0),
    ]
    for event in events:
        reporter.emit(event)
    rendered = live.updates[-1][0]
    assert "done: exit 0" in rendered
    assert "warning-13" in rendered
    assert "warning-0" not in rendered
    assert live.stopped is True
