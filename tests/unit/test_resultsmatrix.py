"""Unit tests for deterministic Results Matrix harvesting."""

from __future__ import annotations

import json
import runpy
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Never, cast

import pytest

from llamatune import resultsmatrix
from llamatune.types import ResultRow

write_matrix_evidence = cast(
    Callable[..., dict[str, Path]],
    runpy.run_path(str(Path(__file__).parents[1] / "fixtures" / "matrix_evidence.py"))[
        "write_matrix_evidence"
    ],
)


def test_harvest_produces_every_kind_with_frozen_field_mapping(tmp_path: Path) -> None:
    paths = write_matrix_evidence(tmp_path / "sessions")

    matrix = resultsmatrix.harvest((paths["root"],))

    assert {row.kind for row in matrix.rows} == {
        "baseline",
        "recommendation",
        "lossless_recommendation",
        "context_envelope",
        "depth_profile",
        "operating_point",
        "ab_verification",
        "calibration",
        "quality_suite",
    }
    assert len(matrix.rows) == 12
    assert matrix.generated == "2026-01-05T00:00:00+00:00"
    assert matrix.warnings == ()

    recommendation = next(row for row in matrix.rows if row.kind == "recommendation")
    assert recommendation.metrics["perf.pp"] == 720.0
    assert recommendation.metrics["perf.tg"] == 48.0
    assert recommendation.confirmed is True
    assert recommendation.reps == 5
    assert recommendation.ctx == 8192
    assert recommendation.depth == 4096
    assert recommendation.quant == "Q4_K_M"
    lossless = next(row for row in matrix.rows if row.kind == "lossless_recommendation")
    assert lossless.metrics["perf.pp"] == 710.0
    assert lossless.metrics["perf.tg"] == 47.0

    ab = next(row for row in matrix.rows if row.kind == "ab_verification")
    assert ab.replicated is True
    assert ab.status == "replicated"
    calibration = next(row for row in matrix.rows if row.kind == "calibration")
    assert calibration.metrics["perf.drift_pp_pct"] == pytest.approx(1.0)
    assert calibration.config is not None

    quality = next(row for row in matrix.rows if row.suite_id == "coding@1+abcdef123456")
    assert quality.metrics == {
        "quality.coding.score": 0.9,
        "quality.coding.pass_rate": 0.8,
        "quality.overall": 0.9,
    }


def test_safe_fallback_and_legacy_quality_gate_are_conservative(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    write_matrix_evidence(root, safe_fallback=True)

    matrix = resultsmatrix.harvest((root,))

    baseline = next(row for row in matrix.rows if row.kind == "baseline")
    assert baseline.status == "safe_fallback"
    assert baseline.confirmed is False
    assert baseline.config is not None and baseline.config.gpu_layers == 0
    legacy = next(row for row in matrix.rows if row.suite_id == "perplexity-gate@legacy")
    assert legacy.metrics["quality.perplexity.ppl"] == 9.5
    assert legacy.metrics["quality.perplexity.delta_pct"] == 1.06


def test_result_row_round_trips_through_canonical_json(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    write_matrix_evidence(root)
    row = resultsmatrix.harvest((root,)).rows[0]

    encoded = json.loads(json.dumps(row.to_dict(), sort_keys=True))

    assert ResultRow.from_dict(encoded) == row


def test_identity_is_stable_and_newest_evidence_is_current(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    first = write_matrix_evidence(root, created="2026-01-01T00:00:00+00:00")
    original = first["session"]
    newer = root / "session-z"
    newer.mkdir()
    for name in (
        "session.json",
        "model.json",
        "hardware.json",
        "llamacpp.json",
        "analysis.json",
        "journal.jsonl",
    ):
        (newer / name).write_bytes((original / name).read_bytes())
    session = json.loads((newer / "session.json").read_text())
    session["created"] = "2026-01-02T00:00:00+00:00"
    (newer / "session.json").write_text(json.dumps(session))

    matrix = resultsmatrix.harvest((root,))
    recommendations = [row for row in matrix.rows if row.kind == "recommendation"]

    assert len(recommendations) == 2
    assert len({row.row_id for row in recommendations}) == 1
    assert sum(row.current for row in recommendations) == 1
    assert next(row for row in recommendations if row.current).evidence_dir == newer


def test_hardware_and_build_changes_change_identity(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    write_matrix_evidence(root_a, hardware_name="GPU A", build="build-a")
    write_matrix_evidence(root_b, hardware_name="GPU B", build="build-b")

    matrix = resultsmatrix.harvest((root_a, root_b))
    recommendations = [row for row in matrix.rows if row.kind == "recommendation"]

    assert len({row.hardware_hash for row in recommendations}) == 2
    assert len({row.build_discriminator for row in recommendations}) == 2
    assert len({row.row_id for row in recommendations}) == 2


def test_corrupt_unit_warns_without_hiding_valid_rows(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    write_matrix_evidence(root)
    corrupt = root / "quality" / "broken"
    corrupt.mkdir()
    (corrupt / "quality.json").write_text("{broken")

    matrix = resultsmatrix.harvest((root,))

    assert len(matrix.rows) == 12
    assert len(matrix.warnings) == 1
    assert str(corrupt) in matrix.warnings[0]


def test_build_is_byte_deterministic_confined_and_loadable(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    output = tmp_path / "output"
    write_matrix_evidence(root)

    first = resultsmatrix.build((root,), output)
    json_bytes = (output / "results-matrix.json").read_bytes()
    markdown_bytes = (output / "results-matrix.md").read_bytes()
    second = resultsmatrix.build((root,), output)

    assert (output / "results-matrix.json").read_bytes() == json_bytes
    assert (output / "results-matrix.md").read_bytes() == markdown_bytes
    assert first == second
    assert first["rows"] == 12
    assert first["current_rows"] == 12
    loaded = resultsmatrix.load_artifact(output / "results-matrix.json")
    assert loaded == resultsmatrix.harvest((root,))
    assert not list(output.glob(".*results-matrix*"))


def test_refresh_preserves_existing_multi_root_configuration(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_matrix_evidence(first, fingerprint="first-model")
    write_matrix_evidence(second, fingerprint="second-model", hardware_name="Second GPU")
    output = first / "matrix"
    resultsmatrix.build((first, second), output)

    resultsmatrix.refresh(first)

    refreshed = resultsmatrix.load_artifact(output / "results-matrix.json")
    assert refreshed.roots == (first.resolve(), second.resolve())
    assert {row.source_root for row in refreshed.rows} == {first.resolve(), second.resolve()}


def test_refresh_ignores_foreign_root_configuration(tmp_path: Path) -> None:
    owner = tmp_path / "owner"
    foreign = tmp_path / "foreign"
    write_matrix_evidence(owner, fingerprint="owner-model")
    write_matrix_evidence(foreign, fingerprint="foreign-model")
    output = owner / "matrix"
    resultsmatrix.build((foreign,), output)

    resultsmatrix.refresh(owner)

    refreshed = resultsmatrix.load_artifact(output / "results-matrix.json")
    assert refreshed.roots == (owner.resolve(),)
    assert {row.source_root for row in refreshed.rows} == {owner.resolve()}


@pytest.mark.parametrize(
    "document",
    (
        "{broken",
        '{"schema_version": 2, "roots": [], "rows": [], "warnings": []}',
        '{"schema_version": 1, "roots": [1], "rows": [], "warnings": []}',
    ),
)
def test_refresh_repairs_invalid_existing_artifact_from_owner_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], document: str
) -> None:
    owner = tmp_path / "owner"
    write_matrix_evidence(owner, fingerprint="owner-model")
    output = owner / "matrix"
    resultsmatrix.build((owner,), output)
    (output / "results-matrix.json").write_text(document)

    resultsmatrix.refresh(owner)

    refreshed = resultsmatrix.load_artifact(output / "results-matrix.json")
    assert refreshed.roots == (owner.resolve(),)
    assert {row.source_root for row in refreshed.rows} == {owner.resolve()}
    assert "existing results matrix configuration ignored" in capsys.readouterr().err


def test_refresh_roots_recovers_when_existing_artifact_is_unreadable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = tmp_path / "owner"
    artifact = owner / "matrix" / "results-matrix.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}")

    def unreadable(path: Path) -> Never:
        raise PermissionError(f"unreadable: {path}")

    monkeypatch.setattr(resultsmatrix, "load_artifact", unreadable)

    assert resultsmatrix._refresh_roots(owner) == (owner.resolve(),)
    assert "unreadable" in capsys.readouterr().err


def test_load_rejects_unknown_schema_and_refresh_reports_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "matrix.json"
    artifact.write_text('{"schema_version": 2, "roots": [], "rows": []}')
    with pytest.raises(ValueError, match="schema version"):
        resultsmatrix.load_artifact(artifact)

    def fail(*args: object, **kwargs: object) -> dict[str, object]:
        raise OSError("read-only")

    monkeypatch.setattr(resultsmatrix, "build", fail)
    resultsmatrix.refresh(tmp_path)
    assert "warning: results matrix refresh failed: read-only" in capsys.readouterr().err


def test_current_tie_break_uses_evidence_directory_name(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    paths = write_matrix_evidence(root)
    original = next(
        row for row in resultsmatrix.harvest((root,)).rows if row.kind == "recommendation"
    )
    duplicate = replace(original, evidence_dir=paths["session"].with_name("session-z"))

    marked = resultsmatrix._mark_current((original, duplicate))

    assert [row.current for row in marked] == [False, True]
