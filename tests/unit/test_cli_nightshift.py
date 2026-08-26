"""Night Shift CLI validation and option assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from llamatune.cli import app
from llamatune.types import NightshiftOptions, NightshiftOutcome

runner = CliRunner()


def _fake_outcome(tmp_path: Path, **kwargs: Any) -> NightshiftOutcome:
    return NightshiftOutcome(run_dir=tmp_path / "run", summary={}, exit_code=0)


def test_help_labels_night_shift_experimental() -> None:
    result = runner.invoke(app, ["nightshift", "--help"])
    assert result.exit_code == 0
    assert "Experimental:" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--until", "25:00"], "--until must be HH:MM"),
        (["--drift-threshold", "-0.1"], "--drift-threshold must be >= 0"),
        (["--calibration-runs", "1"], "--calibration-runs must be >= 2"),
        (["--duplicates", "many"], "--duplicates must be 'one' or 'both'"),
    ],
)
def test_validation_errors_exit_2(tmp_path: Path, args: list[str], message: str) -> None:
    result = runner.invoke(app, ["nightshift", str(tmp_path), *args])
    assert result.exit_code == 2
    assert f"error: {message}" in result.output


def test_options_are_assembled_and_json_summary_emitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[NightshiftOptions] = []

    def fake_run(
        options: NightshiftOptions,
        *,
        reporter: Any = None,
        follow_symlinks: bool = False,
        now_fn: Any = None,
    ) -> NightshiftOutcome:
        captured.append(options)
        assert reporter is None  # json default keeps stdout clean
        assert follow_symlinks is False
        return NightshiftOutcome(
            run_dir=tmp_path / "run", summary={"schema_version": 1}, exit_code=0
        )

    monkeypatch.setattr("llamatune.nightshift.run_nightshift", fake_run)
    result = runner.invoke(
        app,
        [
            "nightshift",
            str(tmp_path),
            "--include",
            "Qwen*",
            "--exclude",
            "*F16*",
            "--duplicates",
            "both",
            "--profile",
            "standard",
            "--dry-run",
            "--budget-trials",
            "80",
            "--ctx-size",
            "8192,16384,32768",
            "--json",
        ],
    )
    assert result.exit_code == 0
    assert '"schema_version": 1' in result.output
    options = captured[0]
    assert options.include == ("Qwen*",)
    assert options.exclude == ("*F16*",)
    assert options.profile == "standard"
    assert options.duplicates == "both"
    assert options.dry_run is True
    assert options.budget_trials == 80
    assert options.ctx_size == 8192
    assert options.ctx_ladder == (16384, 32768)


def test_context_ladder_must_be_distinct_and_ascending(tmp_path: Path) -> None:
    result = runner.invoke(app, ["nightshift", str(tmp_path), "--ctx-size", "8192,4096"])
    assert result.exit_code == 2
    assert "distinct and ascending" in result.output


def test_human_output_names_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    monkeypatch.setattr(
        "llamatune.nightshift.run_nightshift",
        lambda _options, **_kwargs: NightshiftOutcome(run_dir=run_dir, summary={}, exit_code=1),
    )
    result = runner.invoke(app, ["nightshift", str(tmp_path)])
    assert result.exit_code == 1
    assert str(run_dir / "nightshift-report.md") in result.output


def test_human_dry_run_prints_full_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {
        "items": [
            {
                "kind": "tune",
                "model_path": str(tmp_path / "model.gguf"),
                "estimated_minutes": 42.0,
                "reason": "no completed session",
            }
        ]
    }
    monkeypatch.setattr(
        "llamatune.nightshift.run_nightshift",
        lambda _options, **_kwargs: NightshiftOutcome(
            run_dir=tmp_path / "run", summary=summary, exit_code=0
        ),
    )
    result = runner.invoke(app, ["nightshift", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0
    assert "Experimental Night Shift plan:" in result.output
    assert "tune:" in result.output
    assert str(tmp_path / "model.gguf") in result.output
    assert "42.0 min" in result.output
    assert "no completed session" in result.output


@pytest.mark.parametrize(
    ("args", "expect_plain", "expect_json", "expect_follow"),
    [
        ([], True, False, False),
        (["--progress", "plain"], True, False, False),
        (["--follow-symlinks"], True, False, True),
        (["--progress", "json"], False, True, False),
        (["--json"], False, False, False),
        (["--quiet"], False, False, False),
    ],
)
def test_progress_options_thread_reporter_and_follow_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    expect_plain: bool,
    expect_json: bool,
    expect_follow: bool,
) -> None:
    from llamatune.ui import JsonReporter, PlainReporter

    captured: dict[str, Any] = {}

    def fake_run(
        options: NightshiftOptions,
        *,
        reporter: Any = None,
        follow_symlinks: bool = False,
        now_fn: Any = None,
    ) -> NightshiftOutcome:
        captured["reporter"] = reporter
        captured["follow_symlinks"] = follow_symlinks
        return _fake_outcome(tmp_path)

    monkeypatch.setattr("llamatune.nightshift.run_nightshift", fake_run)
    result = runner.invoke(app, ["nightshift", str(tmp_path), *args])
    assert result.exit_code == 0
    assert captured["follow_symlinks"] is expect_follow
    reporter = captured["reporter"]
    if expect_plain:
        assert isinstance(reporter, PlainReporter)
    elif expect_json:
        assert isinstance(reporter, JsonReporter)
    else:
        assert reporter is None


def test_quiet_suppresses_reporter_and_json_keeps_stdout_pure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(
        options: NightshiftOptions,
        *,
        reporter: Any = None,
        follow_symlinks: bool = False,
        now_fn: Any = None,
    ) -> NightshiftOutcome:
        captured["reporter"] = reporter
        return NightshiftOutcome(
            run_dir=tmp_path / "run", summary={"schema_version": 1}, exit_code=0
        )

    monkeypatch.setattr("llamatune.nightshift.run_nightshift", fake_run)
    quiet_result = runner.invoke(app, ["nightshift", str(tmp_path), "--quiet"])
    assert quiet_result.exit_code == 0
    assert captured["reporter"] is None


def test_tui_and_quiet_are_mutually_exclusive(tmp_path: Path) -> None:
    result = runner.invoke(app, ["nightshift", str(tmp_path), "--tui", "--quiet"])
    assert result.exit_code == 2
    assert "mutually exclusive" in result.stderr
