"""Unit tests for the shared evidence IO foundation (issues #7, #8, #29)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from llamatune.evidence import (
    EvidenceWriter,
    InterruptState,
    PathEscapeError,
    confined_path,
    create_unique_dir,
    install_interrupt_handlers,
    jsonable,
    read_journal_lines,
    resolve_deadline,
    utc_iso,
)
from llamatune.marathon import MarathonPathError
from llamatune.nightshift import NightshiftPathError
from llamatune.session import SessionPathError


def test_jsonable_sorts_set_members_deterministically() -> None:
    payload = {"capabilities": frozenset({"fa", "ctk", "d"}), "names": {"b", "a"}}
    first = json.dumps(jsonable(payload))
    second = json.dumps(jsonable(payload))
    assert first == second
    assert jsonable(payload)["capabilities"] == ["ctk", "d", "fa"]


def test_jsonable_llama_report_capabilities_is_stable_across_hash_seeds() -> None:
    snippet = (
        "import json\n"
        "from pathlib import Path\n"
        "from llamatune.evidence import jsonable\n"
        "from llamatune.types import LlamaCppReport\n"
        "report = LlamaCppReport(\n"
        "    bench_path=Path('llama-bench'), cli_path=None, server_path=None,\n"
        "    capabilities=frozenset(['fa', 'ctk', 'd', 'ncmoe', 'tb']),\n"
        "    help_sha256='abc', build_commit=None, build_number=None,\n"
        ")\n"
        "print(json.dumps(jsonable(report), sort_keys=True))\n"
    )
    outputs = set()
    for seed in ("0", "4242", "12345"):
        result = subprocess.run(  # noqa: S603 - fixed in-process snippet, no input
            [sys.executable, "-c", snippet],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1
    report_data = json.loads(next(iter(outputs)))
    assert report_data["capabilities"] == ["ctk", "d", "fa", "ncmoe", "tb"]


def test_jsonable_handles_dataclasses_mappings_paths_and_sequences() -> None:
    @dataclass(frozen=True)
    class Inner:
        path: Path
        values: tuple[int, ...]

    @dataclass(frozen=True)
    class Outer:
        inner: Inner
        tags: frozenset[str]

    value = Outer(inner=Inner(path=Path("models/m.gguf"), values=(3, 1)), tags=frozenset({"x"}))
    assert jsonable(value) == {
        "inner": {"path": "models/m.gguf", "values": [3, 1]},
        "tags": ["x"],
    }
    assert jsonable({1: "a"}) == {"1": "a"}
    assert jsonable([Path("/z")]) == ["/z"]
    assert jsonable(7) == 7


def test_utc_iso_is_utc_and_parseable() -> None:
    parsed = datetime.fromisoformat(utc_iso())
    assert parsed.tzinfo is UTC
    assert abs(datetime.now(UTC) - parsed) < timedelta(seconds=5)


@pytest.mark.parametrize(
    ("until", "max_hours"),
    [
        (None, None),
        (None, 2.0),
        ("23:30", None),
        ("09:00", 2.5),
    ],
)
def test_resolve_deadline_matches_previous_orchestrator_semantics(
    until: str | None, max_hours: float | None
) -> None:
    start = datetime(2026, 8, 25, 10, 0, tzinfo=UTC)
    deadline = resolve_deadline(start, until, max_hours, local_tz=UTC)
    candidates: list[datetime] = []
    if until is not None:
        hour, minute = (int(part) for part in until.split(":"))
        candidate = start.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= start:
            candidate += timedelta(days=1)
        candidates.append(candidate)
    if max_hours is not None:
        candidates.append(start + timedelta(hours=max_hours))
    expected = min(candidates) if candidates else None
    assert deadline == expected


def test_resolve_deadline_rolls_past_wall_time_to_tomorrow() -> None:
    start = datetime(2026, 8, 25, 23, 45, tzinfo=UTC)
    deadline = resolve_deadline(start, "06:00", None, local_tz=UTC)
    assert deadline == datetime(2026, 8, 26, 6, 0, tzinfo=UTC)


def test_confined_path_allows_inside_and_returns_joined_path(tmp_path: Path) -> None:
    target = confined_path(tmp_path, "sub", "file.json")
    assert target == tmp_path / "sub" / "file.json"


def test_confined_path_rejects_escaping_candidates(tmp_path: Path) -> None:
    with pytest.raises(PathEscapeError):
        confined_path(tmp_path, "../escape.json")
    with pytest.raises(PathEscapeError):
        confined_path(tmp_path, "/etc/passwd")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "error_type",
    [SessionPathError, NightshiftPathError, MarathonPathError],
)
def test_per_module_error_types_keep_cli_exception_mapping(error_type: type[Exception]) -> None:
    assert issubclass(error_type, PathEscapeError)
    with pytest.raises(error_type):
        confined_path(Path("/nonexistent-root-wp7"), "../x", error=error_type)


def test_read_journal_lines_skips_corrupt_lines_and_reports_them(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    journal.write_text(
        '{"type": "a", "n": 1}\n'
        "\n"
        "{broken middle\n"
        '{"type": "b", "n": 2}\n'
        '"not an object"\n'
        '{"type": "c", "n": 3\n',
        encoding="utf-8",
    )
    entries, warnings = read_journal_lines(journal)
    assert [entry["type"] for entry in entries] == ["a", "b"]
    assert len(warnings) == 3
    assert any("line 3" in warning for warning in warnings)
    assert any("line 5" in warning for warning in warnings)
    assert any("line 6" in warning for warning in warnings)


def test_read_journal_lines_missing_file_yields_nothing(tmp_path: Path) -> None:
    entries, warnings = read_journal_lines(tmp_path / "absent.jsonl")
    assert entries == []
    assert warnings == []


def test_create_unique_dir_allocates_distinct_expected_names(tmp_path: Path) -> None:
    first = create_unique_dir(tmp_path, "model", error=PathEscapeError)
    second = create_unique_dir(tmp_path, "model", error=PathEscapeError)
    third = create_unique_dir(tmp_path, "", error=PathEscapeError)
    assert len({first, second, third}) == 3
    assert first.name.startswith("model-")
    assert not third.name.startswith("-")


def test_create_unique_dir_exhausts_retries_and_raises_given_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed = datetime(2026, 8, 25, 12, 0, 0)
    fake_clock = SimpleNamespace(now=lambda _tz: fixed)
    monkeypatch.setattr("llamatune.evidence.datetime", fake_clock)
    monkeypatch.setattr("secrets.token_hex", lambda _n: "cafe")
    monkeypatch.setattr("llamatune.evidence.CREATE_RETRIES", 3)
    occupied = tmp_path / "m-20260825-120000-cafe"
    occupied.mkdir()
    with pytest.raises(SessionPathError) as excinfo:
        create_unique_dir(tmp_path, "m", error=SessionPathError)
    assert isinstance(excinfo.value.__cause__, FileExistsError)
    assert list(tmp_path.iterdir()) == [occupied]


class _Writer(EvidenceWriter):
    def __init__(self, root: Path) -> None:
        self.dir = root


def test_evidence_writer_writes_deterministic_json_and_journal(tmp_path: Path) -> None:
    writer = _Writer(tmp_path)
    writer.write_json("nested/thing.json", {"b": 1, "a": {"k": [1, 2]}})
    raw = (tmp_path / "nested" / "thing.json").read_text()
    assert raw.endswith("\n")
    assert raw.index('"a"') < raw.index('"b"')

    writer.append({"type": "event", "detail": Path("/x")})
    record = json.loads((tmp_path / "journal.jsonl").read_text().strip())
    assert record["type"] == "event"
    assert record["detail"] == "/x"
    assert "ts" in record

    entries, warnings = read_journal_lines(tmp_path / "journal.jsonl")
    assert [entry["type"] for entry in entries] == ["event"]
    assert warnings == []

    with pytest.raises(PathEscapeError):
        writer.write_json("../escape.json", {})


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery")
def test_install_interrupt_handlers_installs_and_restores_handlers() -> None:
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        with install_interrupt_handlers() as state:
            installed = signal.getsignal(signal.SIGINT)
            assert callable(installed) and installed not in (previous_sigint, signal.SIG_DFL)
            assert signal.getsignal(signal.SIGTERM) is installed
        assert signal.getsignal(signal.SIGINT) is previous_sigint
        assert signal.getsignal(signal.SIGTERM) is previous_sigterm
        assert state.stop_requested is False
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery")
def test_two_stage_protocol_first_sets_stop_second_raises_promptly() -> None:
    with install_interrupt_handlers() as state:
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        signum = int(signal.SIGINT)

        handler(signum, None)  # first press: graceful stop request
        assert state.stop_requested is True
        assert state.second_signal is False
        assert state.drain_events() == [(signum, False)]

        with pytest.raises(KeyboardInterrupt):  # second press: prompt exit
            handler(signum, None)
        assert state.second_signal is True
        assert state.drain_events()[-1] == (signum, True)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery")
def test_real_sigint_delivery_flips_stop_flag() -> None:
    with install_interrupt_handlers() as state:
        os.kill(os.getpid(), signal.SIGINT)
        assert state.stop_requested is True
        assert state.count == 1


def test_interrupt_state_drain_clears_events() -> None:
    state = InterruptState(events=[(2, False)])
    assert state.drain_events() == [(2, False)]
    assert state.drain_events() == []
