"""CLI contract tests for deterministic quality evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.core import TyperGroup, TyperOption
from typer.main import get_command
from typer.testing import CliRunner

from llamatune.cli import app
from llamatune.types import QualityOptions, QualityOutcome

runner = CliRunner()


def _outcome(tmp_path: Path, *, exit_code: int = 0) -> QualityOutcome:
    return QualityOutcome(
        run_dir=tmp_path / "quality" / "run",
        summary={"schema_version": 1, "overall": 0.75, "warnings": []},
        exit_code=exit_code,
    )


def test_quality_help_and_list_suites() -> None:
    help_result = runner.invoke(app, ["quality", "--help"])
    assert help_result.exit_code == 0
    assert "Experimental:" in help_result.output
    root_command = get_command(app)
    assert isinstance(root_command, TyperGroup)
    quality_command = root_command.commands["quality"]
    registered_options = {
        option
        for parameter in quality_command.params
        if isinstance(parameter, TyperOption)
        for option in parameter.opts
    }
    for option in (
        "--config-session",
        "--compare-lossless",
        "--suite",
        "--exec",
        "--resume",
        "--dry-run",
    ):
        assert option in registered_options

    listed = runner.invoke(app, ["quality", "--list-suites"])
    assert listed.exit_code == 0
    for suite in ("coding@", "tooluse@", "agentic@", "ifollow@", "perplexity@"):
        assert suite in listed.output


def test_quality_assembles_all_options_and_emits_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("quality corpus", encoding="utf-8")

    def capture(options: QualityOptions, **_kwargs: Any) -> QualityOutcome:
        seen["options"] = options
        return _outcome(tmp_path)

    monkeypatch.setattr("llamatune.quality.run_quality", capture)
    result = runner.invoke(
        app,
        [
            "quality",
            str(tmp_path / "model.gguf"),
            "--llama-bin",
            str(tmp_path / "bin"),
            "--sessions-dir",
            str(tmp_path / "sessions"),
            "--config-session",
            str(tmp_path / "source-session"),
            "--strict-config",
            "--compare-lossless",
            "--suite",
            "perplexity",
            "--tasks",
            "needle-*",
            "--ctx-size",
            "4096",
            "--quality-corpus",
            str(corpus),
            "--reps",
            "3",
            "--max-tokens",
            "512",
            "--request-timeout",
            "12.5",
            "--server-start-timeout",
            "25",
            "--seed",
            "7",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["overall"] == 0.75
    options = seen["options"]
    assert options.config_mode == "session"
    assert options.config_session == tmp_path / "source-session"
    assert options.strict_config is True
    assert options.compare_lossless is True
    assert options.suites == ("perplexity",)
    assert options.task_filters == ("needle-*",)
    assert options.ctx_size == 4096
    assert options.quality_corpus == corpus
    assert options.reps == 3
    assert options.max_tokens == 512
    assert options.request_timeout_s == 12.5
    assert options.server_start_timeout_s == 25.0
    assert options.seed == 7
    assert options.dry_run is True


def test_quality_defaults_and_human_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[QualityOptions] = []

    def capture(options: QualityOptions, **_kwargs: Any) -> QualityOutcome:
        seen.append(options)
        return _outcome(tmp_path, exit_code=1)

    monkeypatch.setattr("llamatune.quality.run_quality", capture)
    result = runner.invoke(app, ["quality", str(tmp_path / "model.gguf")])

    assert result.exit_code == 1
    assert "quality run:" in result.stdout
    assert "overall: 0.75" in result.stdout
    assert seen[0].config_mode == "best"
    assert seen[0].suites == ("coding", "tooluse", "agentic", "ifollow")


def test_quality_human_dry_run_and_returned_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def dry(options: QualityOptions, **_kwargs: Any) -> QualityOutcome:
        return QualityOutcome(
            run_dir=Path(),
            summary={
                "dry_run": True,
                "config_source": "defaults",
                "config_provenance": None,
                "suites": [{"suite_id": "coding@1+abc", "tasks": 20}],
                "server_argv": ["llama-server", "--host", "127.0.0.1"],
                "exec_enabled": options.exec_enabled,
            },
            exit_code=0,
        )

    monkeypatch.setattr("llamatune.quality.run_quality", dry)
    result = runner.invoke(app, ["quality", str(tmp_path / "model.gguf"), "--dry-run"])
    assert result.exit_code == 0
    assert "quality dry run" in result.stdout
    assert "coding@1+abc (20 tasks)" in result.stdout
    assert "127.0.0.1" in result.stdout

    monkeypatch.setattr(
        "llamatune.quality.run_quality",
        lambda options, **_kwargs: QualityOutcome(
            run_dir=Path(), summary={"error": "model unreadable"}, exit_code=3
        ),
    )
    failed = runner.invoke(app, ["quality", str(tmp_path / "model.gguf")])
    assert failed.exit_code == 3
    assert "error: model unreadable" in failed.stderr


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ([], "MODEL is required"),
        (["model.gguf", "--config", "nearby"], "--config must be"),
        (
            ["model.gguf", "--config", "best", "--config-session", "session"],
            "mutually exclusive",
        ),
        (["model.gguf", "--reps", "0"], "--reps"),
        (["model.gguf", "--reps", "6"], "--reps"),
        (["model.gguf", "--ctx-size", "0"], "--ctx-size values must be > 0"),
        (["model.gguf", "--ctx-size", "8192,4096"], "distinct and ascending"),
        (["model.gguf", "--ctx-size", "abc"], "comma-separated list of integers"),
        (["model.gguf", "--max-tokens", "0"], "must be positive"),
        (["model.gguf", "--request-timeout", "0"], "timeouts"),
        (["model.gguf", "--server-start-timeout", "0"], "timeouts"),
        (["model.gguf", "--request-timeout", "nan"], "finite"),
        (["model.gguf", "--server-start-timeout", "inf"], "finite"),
        (["model.gguf", "--suite", "missing-suite"], "unknown suite"),
        (["model.gguf", "--suite", "perplexity"], "required iff"),
        (["model.gguf", "--quality-corpus", "corpus.txt"], "required iff"),
    ],
)
def test_quality_validation_exits_2(args: list[str], message: str) -> None:
    result = runner.invoke(app, ["quality", *args])
    assert result.exit_code == 2
    assert message in result.stderr


def test_quality_exec_platform_validation_precedes_run_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("llamatune.cli._quality_exec_supported", lambda: False)
    result = runner.invoke(
        app,
        ["quality", str(tmp_path / "model.gguf"), "--sessions-dir", str(tmp_path), "--exec"],
    )
    assert result.exit_code == 2
    assert "POSIX resource limits" in result.stderr
    assert not (tmp_path / "quality").exists()


def test_quality_resume_wiring_and_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def capture(
        run_dir: Path,
        *,
        llama_bin: Path | None = None,
        now_fn: Any = None,
        reporter: Any = None,
    ) -> QualityOutcome:
        seen.update(run_dir=run_dir, llama_bin=llama_bin, now_fn=now_fn, reporter=reporter)
        return _outcome(tmp_path)

    monkeypatch.setattr("llamatune.quality.resume_quality", capture)
    result = runner.invoke(
        app,
        ["quality", "--resume", str(tmp_path / "run"), "--llama-bin", str(tmp_path / "bin")],
    )
    assert result.exit_code == 0
    assert seen["run_dir"] == tmp_path / "run"
    assert seen["llama_bin"] == tmp_path / "bin"

    conflict = runner.invoke(
        app,
        ["quality", "model.gguf", "--resume", str(tmp_path / "run")],
    )
    assert conflict.exit_code == 2
    assert "mutually exclusive" in conflict.stderr

    explicit_default = runner.invoke(
        app,
        ["quality", "--resume", str(tmp_path / "run"), "--ctx-size", "8192"],
    )
    assert explicit_default.exit_code == 2


@pytest.mark.parametrize(("exception", "exit_code"), [(ValueError("bad"), 2), (OSError("bad"), 3)])
def test_quality_maps_orchestrator_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception: Exception,
    exit_code: int,
) -> None:
    def fail(options: QualityOptions, **_kwargs: Any) -> QualityOutcome:
        raise exception

    monkeypatch.setattr("llamatune.quality.run_quality", fail)
    result = runner.invoke(app, ["quality", str(tmp_path / "model.gguf")])
    assert result.exit_code == exit_code
    assert "error: bad" in result.stderr


def test_quality_ctx_size_accepts_csv_and_bare_int(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[QualityOptions] = []

    def capture(options: QualityOptions, **_kwargs: Any) -> QualityOutcome:
        captured.append(options)
        return _outcome(tmp_path)

    monkeypatch.setattr("llamatune.quality.run_quality", capture)
    csv_result = runner.invoke(
        app, ["quality", str(tmp_path / "model.gguf"), "--ctx-size", "4096,8192"]
    )
    assert csv_result.exit_code == 0
    assert captured[0].ctx_size == 4096

    bare_result = runner.invoke(app, ["quality", str(tmp_path / "m.gguf"), "--ctx-size", "4096"])
    assert bare_result.exit_code == 0
    assert captured[1].ctx_size == 4096


def test_quality_default_ctx_size_is_8192(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[QualityOptions] = []

    def capture(options: QualityOptions, **_kwargs: Any) -> QualityOutcome:
        captured.append(options)
        return _outcome(tmp_path)

    monkeypatch.setattr("llamatune.quality.run_quality", capture)
    result = runner.invoke(app, ["quality", str(tmp_path / "model.gguf")])
    assert result.exit_code == 0
    assert captured[0].ctx_size == 8192


@pytest.mark.parametrize(
    ("args", "expect_reporter"),
    [
        ([], True),
        (["--progress", "plain"], True),
        (["--quiet"], False),
        (["--json"], False),
    ],
)
def test_quality_threads_reporter_from_progress_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    expect_reporter: bool,
) -> None:
    from llamatune.ui import PlainReporter

    captured: dict[str, Any] = {}

    def capture(options: QualityOptions, *, reporter: Any = None) -> QualityOutcome:
        captured["reporter"] = reporter
        return _outcome(tmp_path)

    monkeypatch.setattr("llamatune.quality.run_quality", capture)
    result = runner.invoke(app, ["quality", str(tmp_path / "model.gguf"), *args])
    assert result.exit_code == 0
    if expect_reporter:
        # auto resolves to the plain renderer on a non-TTY stderr.
        assert isinstance(captured["reporter"], PlainReporter)
    else:
        assert captured["reporter"] is None
