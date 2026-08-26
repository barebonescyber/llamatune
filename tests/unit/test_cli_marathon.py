"""Marathon CLI validation and option assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from llamatune.cli import app
from llamatune.types import LlamaCppReport, MarathonOptions, MarathonOutcome

runner = CliRunner()


def test_help_labels_marathon_experimental() -> None:
    result = runner.invoke(app, ["marathon", "--help"])
    assert result.exit_code == 0
    assert "Experimental:" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--until", "25:00"], "--until must be HH:MM"),
        (["--rounds-max", "0"], "--rounds-max must be >= 1"),
        (["--converge-rounds", "0"], "--converge-rounds must be >= 1"),
        (["--ab-blocks", "2"], "--ab-blocks must be >= 3"),
        (["--depth-grid", "8192,0"], "--depth-grid values must be distinct and ascending"),
    ],
)
def test_validation_errors_exit_2(tmp_path: Path, args: list[str], message: str) -> None:
    result = runner.invoke(app, ["marathon", str(tmp_path / "model.gguf"), *args])
    assert result.exit_code == 2
    assert f"error: {message}" in result.output


def _llama(tmp_path: Path, capabilities: frozenset[str] = frozenset({"d"})) -> LlamaCppReport:
    return LlamaCppReport(
        bench_path=tmp_path / "llama-bench",
        cli_path=None,
        server_path=None,
        capabilities=capabilities,
        help_sha256="h",
        build_commit=None,
        build_number=None,
    )


def test_depth_capability_error_precedes_run_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def fake_run(_options: MarathonOptions, **_kwargs: Any) -> MarathonOutcome:
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(
        "llamatune.llama.discover_llama", lambda _path: _llama(tmp_path, frozenset())
    )
    monkeypatch.setattr("llamatune.marathon.run_marathon", fake_run)
    result = runner.invoke(app, ["marathon", str(tmp_path / "model.gguf")])
    assert result.exit_code == 2
    assert str(tmp_path / "llama-bench") in result.output
    assert called is False


def test_options_assembled_and_json_emitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[MarathonOptions] = []

    def fake_run(
        options: MarathonOptions, *, reporter: Any = None, now_fn: Any = None
    ) -> MarathonOutcome:
        captured.append(options)
        assert reporter is None  # json default resolves to no reporter
        return MarathonOutcome(run_dir=tmp_path / "run", summary={"schema_version": 1}, exit_code=0)

    monkeypatch.setattr("llamatune.llama.discover_llama", lambda _path: _llama(tmp_path))
    monkeypatch.setattr("llamatune.marathon.run_marathon", fake_run)
    result = runner.invoke(
        app,
        [
            "marathon",
            str(tmp_path / "model.gguf"),
            "--rounds-max",
            "4",
            "--ab-blocks",
            "7",
            "--depth-grid",
            "0,4096",
            "--ctx-size",
            "8192,16384",
            "--budget-trials",
            "300",
            "--no-matrix-refine",
            "--json",
        ],
    )
    assert result.exit_code == 0
    assert '"schema_version": 1' in result.output
    options = captured[0]
    assert options.rounds_max == 4
    assert options.ab_blocks == 7
    assert options.depth_grid == (0, 4096)
    assert options.ctx_size == 8192
    assert options.ctx_ladder == (16384,)
    assert options.budget_trials == 300
    assert options.matrix_refine is False


def test_human_dry_run_includes_full_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llamatune.llama.discover_llama", lambda _path: _llama(tmp_path))
    monkeypatch.setattr(
        "llamatune.marathon.run_marathon",
        lambda _options, **_kwargs: MarathonOutcome(
            run_dir=tmp_path / "run",
            summary={
                "plan": {
                    "recon_runs": 10,
                    "round_1_budget": 240,
                    "tiers": "A, B, C, D",
                    "matrix_cells": 6,
                }
            },
            exit_code=0,
        ),
    )
    result = runner.invoke(app, ["marathon", str(tmp_path / "model.gguf"), "--dry-run"])
    assert result.exit_code == 0
    assert "Experimental Marathon plan:" in result.output
    assert "reconnaissance: 10 runs" in result.output
    assert "round 1 budget: 240 trials" in result.output
    assert "matrix cells: 6" in result.output
    assert "A/B blocks: 5" in result.output


@pytest.mark.parametrize(
    ("args", "expect_plain", "expect_json"),
    [
        ([], True, False),
        (["--progress", "plain"], True, False),
        (["--progress", "json"], False, True),
        (["--json"], False, False),
        (["--quiet"], False, False),
    ],
)
def test_progress_options_thread_reporter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    expect_plain: bool,
    expect_json: bool,
) -> None:
    from llamatune.ui import JsonReporter, PlainReporter

    captured: dict[str, Any] = {}

    def fake_run(
        options: MarathonOptions, *, reporter: Any = None, now_fn: Any = None
    ) -> MarathonOutcome:
        captured["reporter"] = reporter
        return MarathonOutcome(run_dir=tmp_path / "run", summary={}, exit_code=0)

    monkeypatch.setattr("llamatune.llama.discover_llama", lambda _path: _llama(tmp_path))
    monkeypatch.setattr("llamatune.marathon.run_marathon", fake_run)
    result = runner.invoke(app, ["marathon", str(tmp_path / "model.gguf"), *args])
    assert result.exit_code == 0
    reporter = captured["reporter"]
    if expect_plain:
        assert isinstance(reporter, PlainReporter)
    elif expect_json:
        assert isinstance(reporter, JsonReporter)
    else:
        assert reporter is None


def test_tui_and_quiet_are_mutually_exclusive(tmp_path: Path) -> None:
    result = runner.invoke(app, ["marathon", str(tmp_path / "m.gguf"), "--tui", "--quiet"])
    assert result.exit_code == 2
    assert "mutually exclusive" in result.stderr
