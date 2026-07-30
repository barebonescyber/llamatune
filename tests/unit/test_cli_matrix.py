"""CLI contract tests for the Results Matrix sub-application."""

from __future__ import annotations

import json
import runpy
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from llamatune import resultsmatrix
from llamatune.cli import (
    _json_default,
    _matrix_config_summary,
    _matrix_rank_value,
    _matrix_show_payload,
    _refresh_results_matrix,
    app,
)

write_matrix_evidence = cast(
    Callable[..., dict[str, Path]],
    runpy.run_path(str(Path(__file__).parents[1] / "fixtures" / "matrix_evidence.py"))[
        "write_matrix_evidence"
    ],
)

runner = CliRunner()


def test_matrix_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["matrix", "--help"])

    assert result.exit_code == 0
    for command in ("build", "query", "show", "export"):
        assert command in result.output


def test_matrix_build_json_and_empty_matrix(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["matrix", "build", "--sessions-dir", str(tmp_path), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["rows"] == 0
    assert payload["models"] == 0
    assert (tmp_path / "matrix" / "results-matrix.json").is_file()
    assert (tmp_path / "matrix" / "results-matrix.md").is_file()

    built = runner.invoke(
        app,
        ["matrix", "build", "--sessions-dir", str(tmp_path)],
    )
    assert built.exit_code == 0
    assert "matrix: 0 rows, 0 models, 1 roots, 0 warnings" in built.stdout

    shown = runner.invoke(app, ["matrix", "show", "--sessions-dir", str(tmp_path)])
    assert shown.exit_code == 0
    assert "0 rows" in shown.stdout


@pytest.mark.parametrize("subcommand", ["build", "query", "show", "export"])
def test_matrix_missing_root_exits_3(tmp_path: Path, subcommand: str) -> None:
    args = ["matrix", subcommand, "--sessions-dir", str(tmp_path / "missing")]
    if subcommand == "query":
        args.extend(["--sort", "perf.tg"])
    elif subcommand == "export":
        args.extend(["--format", "json"])
    result = runner.invoke(app, args)

    assert result.exit_code == 3
    assert "error: sessions root is missing or unreadable" in result.stderr


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ([], "one of --use-case or --sort is required"),
        (["--use-case", "max-tg", "--sort", "perf.tg"], "mutually exclusive"),
        (["--use-case", "not-a-use-case"], "unknown use case"),
        (["--sort", "perf.tg", "--kind", "not-a-kind"], "unknown kind"),
        (["--sort", "perf.tg", "--compat", "nearby"], "--compat must be"),
        (["--sort", "perf.tg", "--compat", "current"], "requires --llama-bin"),
        (["--sort", "perf.tg", "--limit", "-1"], "--limit must be"),
    ],
)
def test_matrix_query_validation(tmp_path: Path, args: list[str], message: str) -> None:
    result = runner.invoke(
        app,
        ["matrix", "query", "--sessions-dir", str(tmp_path), *args],
    )

    assert result.exit_code == 2
    assert message in result.stderr


def test_matrix_query_assembles_spec_and_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_matrix_evidence(tmp_path)
    seen: dict[str, Any] = {}

    def capture(matrix: Any, spec: Any, identity: tuple[str, str] | None) -> dict[str, Any]:
        assert identity is not None
        seen.update(matrix=matrix, spec=spec, identity=identity)
        return {
            "sort": spec.sort,
            "filters": {},
            "identity": {
                "hardware_hash": identity[0],
                "build_discriminator": identity[1],
            },
            "groups": [],
            "excluded": {},
            "warnings": [],
        }

    monkeypatch.setattr("llamatune.cli._matrix_identity", lambda path: ("hardware", "build"))
    monkeypatch.setattr("llamatune.matrixquery.apply", capture)
    result = runner.invoke(
        app,
        [
            "matrix",
            "query",
            "--sessions-dir",
            str(tmp_path),
            "--sort",
            "perf.tg",
            "--ascending",
            "--model",
            "fixture",
            "--quant",
            "Q4_K_M",
            "--kind",
            "recommendation",
            "--suite",
            "coding",
            "--ctx",
            "4096",
            "--depth",
            "2048",
            "--min-pp",
            "1",
            "--min-tg",
            "2",
            "--include-unconfirmed",
            "--include-superseded",
            "--compat",
            "current",
            "--llama-bin",
            str(tmp_path),
            "--limit",
            "0",
            "--json",
        ],
    )

    assert result.exit_code == 0
    spec = seen["spec"]
    assert spec.sort == "perf.tg"
    assert spec.ascending is True
    assert spec.model == "fixture"
    assert spec.quant == "Q4_K_M"
    assert spec.kinds == ("recommendation",)
    assert spec.suite == "coding"
    assert spec.ctx_min == 4096
    assert spec.depth == 2048
    assert spec.min_pp == 1.0
    assert spec.min_tg == 2.0
    assert spec.confirmed_only is False
    assert spec.current_only is False
    assert spec.compat == "current"
    assert spec.limit == 0
    assert seen["identity"] == ("hardware", "build")


def test_matrix_query_probe_failure_exits_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llamatune.llama import LlamaDiscoveryError

    def fail(path: Path) -> tuple[str, str]:
        raise LlamaDiscoveryError("no usable llama-bench")

    monkeypatch.setattr("llamatune.cli._matrix_identity", fail)
    result = runner.invoke(
        app,
        [
            "matrix",
            "query",
            "--sessions-dir",
            str(tmp_path),
            "--sort",
            "perf.tg",
            "--llama-bin",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 3
    assert "error: could not probe current identity" in result.stderr


@pytest.mark.parametrize(
    ("use_case", "expected"),
    [
        ("max-pp", 400.0),
        ("max-tg", 25.0),
        ("balanced", 100.0),
        ("max-context", 8192.0),
        ("coding", 0.8),
        ("tool-use", 0.7),
        ("agentic", 0.6),
        ("instruction", 0.9),
        ("quality-overall", 0.75),
    ],
)
def test_matrix_human_rank_metric_for_named_use_cases(use_case: str, expected: float) -> None:
    row = {
        "metrics": {
            "perf.pp": 400.0,
            "perf.tg": 25.0,
            "ctx.validated": 8192.0,
            "quality.coding.score": 0.8,
            "quality.tooluse.score": 0.7,
            "quality.agentic.score": 0.6,
            "quality.ifollow.score": 0.9,
            "quality.overall": 0.75,
        }
    }

    assert _matrix_rank_value({"use_case": use_case}, row) == pytest.approx(expected)


def test_matrix_human_formatting_helpers_cover_fallbacks() -> None:
    assert _json_default(frozenset({"b", "a"})) == ["a", "b"]
    with pytest.raises(TypeError, match="not JSON serializable"):
        _json_default(object())
    assert _matrix_config_summary(None) == "defaults"
    assert _matrix_config_summary({"gpu_layers": 12, "flash_attn": True}).startswith(
        "ngl=12/ncmoe=-/fa=1"
    )
    assert _matrix_rank_value({"sort": "custom"}, {"metrics": {"custom": 7}}) == 7
    assert (
        _matrix_rank_value(
            {"use_case": "balanced"}, {"metrics": {"perf.pp": "unknown", "perf.tg": 2}}
        )
        is None
    )


def test_matrix_human_query_renders_rows_groups_and_warnings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from llamatune.cli import _echo_matrix_query

    _echo_matrix_query({"groups": [], "excluded": {}, "warnings": []})
    assert "0 rows" in capsys.readouterr().out

    recommendation = {
        "rank": 1,
        "kind": "recommendation",
        "model_name": "Model",
        "model_path": "/models/model.gguf",
        "quant": "Q4_K_M",
        "config": {"gpu_layers": 12, "flash_attn": True},
        "ctx": 8192,
        "depth": 0,
        "metrics": {"perf.pp": 400.0, "perf.tg": 25.0},
        "confirmed": True,
        "replicated": True,
        "compat": "current",
        "evidence_dir": "/evidence/recommendation",
    }
    unquantized = {
        **recommendation,
        "rank": 2,
        "kind": "tune",
        "model_name": None,
        "quant": None,
        "confirmed": False,
        "replicated": False,
        "evidence_dir": "/evidence/tune",
    }
    payload = {
        "sort": "perf.tg",
        "groups": [
            {"hardware_hash": "first", "rows": [recommendation, unquantized]},
            {"hardware_hash": "empty", "rows": []},
            {"hardware_hash": "tune", "rows": [unquantized]},
        ],
        "excluded": {"stale": 1},
        "warnings": ["mixed hardware"],
    }

    _echo_matrix_query(payload)
    captured = capsys.readouterr()
    assert "Model (Q4_K_M)" in captured.out
    expected_reproduction = Path("/evidence/recommendation") / "recommended.sh"
    assert f"reproduce: {expected_reproduction}" in captured.out
    assert "excluded: stale=1" in captured.out
    assert "warning: mixed hardware" in captured.err


def test_matrix_show_and_exports(tmp_path: Path) -> None:
    write_matrix_evidence(tmp_path)

    shown = runner.invoke(
        app,
        ["matrix", "show", "--sessions-dir", str(tmp_path), "--json"],
    )
    assert shown.exit_code == 0
    summary = json.loads(shown.stdout)
    assert summary["rows"] > 0
    assert summary["models"][0]["model_fingerprint"] == "fingerprint-a"
    assert "recommendation" in summary["models"][0]["kinds"]

    for export_format in ("json", "csv", "md"):
        result = runner.invoke(
            app,
            [
                "matrix",
                "export",
                "--sessions-dir",
                str(tmp_path),
                "--format",
                export_format,
            ],
        )
        assert result.exit_code == 0
        assert result.stdout
    assert (
        json.loads(
            runner.invoke(
                app,
                ["matrix", "export", "--sessions-dir", str(tmp_path), "--format", "json"],
            ).stdout
        )["row_count"]
        > 0
    )

    destination = tmp_path / "rows.csv"
    written = runner.invoke(
        app,
        [
            "matrix",
            "export",
            "--sessions-dir",
            str(tmp_path),
            "--format",
            "csv",
            "--output",
            str(destination),
        ],
    )
    assert written.exit_code == 0
    assert destination.is_file()


def test_matrix_show_uses_current_rows_and_geometric_balanced_score(tmp_path: Path) -> None:
    write_matrix_evidence(tmp_path)
    matrix = resultsmatrix.harvest((tmp_path,))
    recommendation = next(row for row in matrix.rows if row.kind == "recommendation")
    stale = replace(
        recommendation,
        current=False,
        metrics={"perf.pp": 10000.0, "perf.tg": 10000.0},
    )

    payload = _matrix_show_payload(replace(matrix, rows=(*matrix.rows, stale)))
    model = payload["models"][0]
    current = [row for row in matrix.rows if row.current]

    assert model["best_pp"] == max(
        row.metrics["perf.pp"] for row in current if "perf.pp" in row.metrics
    )
    assert model["best_tg"] == max(
        row.metrics["perf.tg"] for row in current if "perf.tg" in row.metrics
    )
    assert model["best_balanced"] == pytest.approx(
        max(
            (row.metrics["perf.pp"] * row.metrics["perf.tg"]) ** 0.5
            for row in current
            if "perf.pp" in row.metrics and "perf.tg" in row.metrics
        )
    )
    assert model["stale_identity"] is True


def test_matrix_build_and_export_reject_unusable_output(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("file", encoding="utf-8")

    built = runner.invoke(
        app,
        [
            "matrix",
            "build",
            "--sessions-dir",
            str(tmp_path),
            "--output",
            str(blocker / "matrix"),
        ],
    )
    assert built.exit_code == 3

    exported = runner.invoke(
        app,
        [
            "matrix",
            "export",
            "--sessions-dir",
            str(tmp_path),
            "--format",
            "json",
            "--output",
            str(blocker / "rows.json"),
        ],
    )
    assert exported.exit_code == 3


def test_refresh_guard_preserves_exit_and_reports_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[Path] = []

    def fail(root: Path) -> None:
        calls.append(root)
        raise OSError("read-only")

    monkeypatch.setattr("llamatune.resultsmatrix.refresh", fail)
    _refresh_results_matrix(tmp_path, 2)
    assert calls == []

    _refresh_results_matrix(tmp_path, 1)
    assert calls == [tmp_path]
    assert "warning: results matrix refresh failed: read-only" in capsys.readouterr().err
