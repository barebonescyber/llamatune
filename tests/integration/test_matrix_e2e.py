"""End-to-end Results Matrix workflows over real and synthetic evidence."""

from __future__ import annotations

import json
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from llamatune.cli import app
from llamatune.types import GPUInfo, HardwareReport

write_matrix_evidence = cast(
    Callable[..., dict[str, Path]],
    runpy.run_path(str(Path(__file__).parents[1] / "fixtures" / "matrix_evidence.py"))[
        "write_matrix_evidence"
    ],
)

runner = CliRunner()


def _gpu_hardware(llama_bin: Path | None = None) -> HardwareReport:
    return HardwareReport(
        os_name="Linux",
        arch="x86_64",
        cpu_model="Fake CPU",
        physical_cores=8,
        logical_cores=16,
        perf_cores=None,
        ram_mb=32768,
        gpus=(GPUInfo(vendor="nvidia", name="Fake GPU", vram_mb=24000, method="test"),),
        warnings=(),
    )


def _invoke_tune(model: Path, llama_bin: Path, root: Path, *, baseline_only: bool = False) -> Any:
    args = [
        "tune",
        str(model),
        "--llama-bin",
        str(llama_bin),
        "--sessions-dir",
        str(root),
        "--budget-trials",
        "12",
        "--initial-gpu-layers",
        "33",
        "--max-gpu-layers",
        "33",
        "--reps-search",
        "1",
        "--reps-confirm",
        "3",
        "--thermal-wait-cap-s",
        "0",
        "--quiet",
    ]
    if baseline_only:
        args.append("--baseline-only")
    return runner.invoke(app, args)


def test_real_tune_build_and_query_rank_recommendation(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("llamatune.hardware.assess_hardware", _gpu_hardware)
    root = tmp_path / "sessions"
    tuned = _invoke_tune(tiny_gguf, fake_bin_dir, root)
    assert tuned.exit_code == 0
    assert (root / "matrix" / "results-matrix.json").is_file()

    built = runner.invoke(
        app,
        ["matrix", "build", "--sessions-dir", str(root), "--json"],
    )
    assert built.exit_code == 0
    summary = json.loads(built.stdout)
    assert summary["kinds"]["baseline"] == 1
    assert summary["kinds"]["recommendation"] == 1
    document = json.loads((root / "matrix" / "results-matrix.json").read_text())
    assert document["row_count"] >= 2

    queried = runner.invoke(
        app,
        [
            "matrix",
            "query",
            "--sessions-dir",
            str(root),
            "--use-case",
            "max-tg",
            "--json",
        ],
    )
    assert queried.exit_code == 0
    payload = json.loads(queried.stdout)
    assert payload["groups"][0]["rows"][0]["kind"] == "recommendation"


def test_superseded_recommendations_are_retained_and_selectable(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("llamatune.hardware.assess_hardware", _gpu_hardware)
    root = tmp_path / "sessions"
    monkeypatch.setenv("LLAMATUNE_FAKE_SPEED_SCALE", "1.0")
    first = _invoke_tune(tiny_gguf, fake_bin_dir, root)
    monkeypatch.setenv("LLAMATUNE_FAKE_SPEED_SCALE", "1.2")
    second = _invoke_tune(tiny_gguf, fake_bin_dir, root)
    assert first.exit_code == second.exit_code == 0

    current = runner.invoke(
        app,
        [
            "matrix",
            "query",
            "--sessions-dir",
            str(root),
            "--sort",
            "perf.tg",
            "--kind",
            "recommendation",
            "--json",
        ],
    )
    history = runner.invoke(
        app,
        [
            "matrix",
            "query",
            "--sessions-dir",
            str(root),
            "--sort",
            "perf.tg",
            "--kind",
            "recommendation",
            "--include-superseded",
            "--json",
        ],
    )

    assert current.exit_code == history.exit_code == 0
    current_rows = json.loads(current.stdout)["groups"][0]["rows"]
    history_rows = json.loads(history.stdout)["groups"][0]["rows"]
    assert len(current_rows) == 1
    assert len(history_rows) == 2
    assert sum(row["current"] for row in history_rows) == 1
    assert current_rows[0]["metrics"]["perf.tg"] == max(
        row["metrics"]["perf.tg"] for row in history_rows
    )


def test_multi_root_build_preserves_provenance_and_uses_first_root(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_matrix_evidence(first, fingerprint="first-model")
    write_matrix_evidence(second, fingerprint="second-model", hardware_name="Second GPU")

    result = runner.invoke(
        app,
        [
            "matrix",
            "build",
            "--sessions-dir",
            str(first),
            "--sessions-dir",
            str(second),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert (first / "matrix" / "results-matrix.json").is_file()
    assert not (second / "matrix").exists()
    document = json.loads((first / "matrix" / "results-matrix.json").read_text())
    assert document["roots"] == [str(first.resolve()), str(second.resolve())]
    assert {row["source_root"] for row in document["rows"]} == {
        str(first.resolve()),
        str(second.resolve()),
    }


def test_refresh_failure_does_not_change_tune_result_or_evidence(
    tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    (root / "matrix").write_text("blocks artifact directory", encoding="utf-8")

    result = _invoke_tune(tiny_gguf, fake_bin_dir, root, baseline_only=True)

    assert result.exit_code == 0
    assert "warning: results matrix refresh failed" in result.stderr
    sessions = [path for path in root.iterdir() if (path / "session.json").is_file()]
    assert len(sessions) == 1
    assert (sessions[0] / "analysis.json").is_file()


def test_corrupt_session_warns_without_hiding_valid_rows(tmp_path: Path) -> None:
    write_matrix_evidence(tmp_path)
    corrupt = tmp_path / "corrupt-session"
    corrupt.mkdir()
    (corrupt / "session.json").write_text("{not-json", encoding="utf-8")

    result = runner.invoke(
        app,
        ["matrix", "build", "--sessions-dir", str(tmp_path), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["rows"] > 0
    assert any(str(corrupt) in warning for warning in payload["warnings"])
