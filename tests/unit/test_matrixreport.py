from __future__ import annotations

import csv
import io
from dataclasses import replace
from pathlib import Path
from typing import Any

from llamatune.matrixreport import render_csv, render_markdown
from llamatune.types import ResultRow, ResultsMatrix, TrialConfig

_CONFIG = TrialConfig(
    gpu_layers=33,
    moe_cpu_layers=0,
    flash_attn=True,
    ubatch=512,
    batch=2048,
    threads=8,
    mmap=True,
    no_kv_offload=False,
    cache_type_k="f16",
    cache_type_v="f16",
)


def _row(row_id: str, **changes: Any) -> ResultRow:
    values: dict[str, Any] = {
        "row_id": row_id,
        "kind": "recommendation",
        "current": True,
        "model_fingerprint": "abcdef0123456789model",
        "model_name": "Alpha | Model",
        "model_path": "/models/alpha.gguf",
        "quant": "Q4_K_M",
        "hardware_hash": "hardware-a",
        "hardware_signature": ("Linux", "x86_64", "GPU A"),
        "build_discriminator": "build-a",
        "build_commit": "commit-a",
        "config": _CONFIG,
        "ctx": 8192,
        "depth": 4096,
        "pp_workload": 512,
        "tg_workload": 128,
        "suite_id": None,
        "metrics": {"perf.pp": 100.0, "perf.tg": 10.0},
        "status": "ok",
        "confirmed": True,
        "replicated": True,
        "reps": 5,
        "noise_floor_cv": 0.01,
        "source_root": Path("/evidence"),
        "evidence_dir": Path(f"/evidence/{row_id}"),
        "ts": "2026-01-01T00:00:00+00:00",
    }
    values.update(changes)
    return ResultRow(**values)


def _matrix(*rows: ResultRow) -> ResultsMatrix:
    return ResultsMatrix(
        schema_version=1,
        generated="2026-01-05T00:00:00+00:00",
        roots=(Path("/evidence-a"), Path("/evidence-b")),
        rows=rows,
        warnings=("one | warning",),
    )


def test_markdown_has_all_required_sections_and_is_repeatable() -> None:
    recommendation = _row("recommendation")
    operating = _row(
        "operating",
        kind="operating_point",
        metrics={"perf.pp": 90.0, "perf.tg": 9.0},
    )
    envelope = _row(
        "envelope",
        kind="context_envelope",
        metrics={"ctx.validated": 16384.0},
        confirmed=False,
    )
    quality = _row(
        "quality",
        kind="quality_suite",
        suite_id="coding@1+abcdef",
        metrics={"quality.coding.score": 0.9, "quality.overall": 0.8},
    )
    superseded = replace(recommendation, row_id="old", current=False)
    other_model = _row(
        "beta",
        model_fingerprint="beta-fingerprint",
        model_name="Beta Model",
        metrics={"perf.pp": 80.0, "perf.tg": 8.0},
    )
    matrix = _matrix(other_model, quality, envelope, superseded, operating, recommendation)

    rendered = render_markdown(matrix)
    assert rendered == render_markdown(matrix)
    assert rendered.startswith("# Results Matrix\n")
    assert "Generated: `2026-01-05T00:00:00+00:00`" in rendered
    assert "Counts: 6 rows, 5 current, 2 models" in rendered
    assert "- one \\| warning" in rendered
    assert rendered.index("## Alpha \\| Model") < rendered.index("## Beta Model")
    assert "fingerprint `abcdef0123456789`, size `unavailable`" in rendered
    assert "### Best results" in rendered
    assert "### Operating points" in rendered
    assert "### Quality" in rendered
    assert "History: 1 superseded row(s)." in rendered
    assert "| Use case | Hardware | Config | Result | Compat | Evidence |" in rendered
    assert "| max-pp | hardware-a |" in rendered
    assert "| coding | hardware-a |" in rendered


def test_markdown_handles_empty_document_without_optional_sections() -> None:
    rendered = render_markdown(_matrix())
    assert "Counts: 0 rows, 0 current, 0 models" in rendered
    assert "## Warnings" in rendered
    assert "### Operating points" not in rendered
    assert "### Quality" not in rendered


def test_markdown_best_ignores_failed_rows_and_matches_default_config_ties() -> None:
    recommendation = _row("defaults-rec", config=None)
    matched = _row(
        "defaults-quality",
        kind="quality_suite",
        config=None,
        suite_id="coding@1+matched",
        metrics={"quality.coding.score": 0.9},
    )
    unmatched = _row(
        "unmatched-quality",
        kind="quality_suite",
        config=replace(_CONFIG, gpu_layers=20),
        suite_id="coding@1+unmatched",
        metrics={"quality.coding.score": 0.9},
        ts="2026-01-03T00:00:00+00:00",
    )
    failed = _row(
        "failed-quality",
        kind="quality_suite",
        suite_id="coding@1+failed",
        metrics={"quality.coding.score": 1.0},
        status="failed",
    )
    rendered = render_markdown(_matrix(unmatched, failed, matched, recommendation))
    evidence = Path("/evidence/defaults-quality")
    assert f"| coding | hardware-a | defaults | 0.9 |  | {evidence} |" in rendered
    assert "| coding | hardware-a | defaults | 1 |" not in rendered


def test_csv_uses_fixed_scalars_then_sorted_metric_union() -> None:
    first = _row("first", metrics={"z.metric": 2.0, "a.metric": 1.0})
    second = _row("second", metrics={"middle.metric": 3.0})
    rendered = render_csv((first, second))
    assert rendered == render_csv((first, second))
    reader = csv.DictReader(io.StringIO(rendered))
    assert reader.fieldnames is not None
    assert reader.fieldnames[-3:] == ["a.metric", "middle.metric", "z.metric"]
    assert "hardware_signature" not in reader.fieldnames
    assert "config" not in reader.fieldnames
    assert "metrics" not in reader.fieldnames
    records = list(reader)
    assert records[0]["a.metric"] == "1.0"
    assert records[0]["middle.metric"] == ""
    assert records[1]["z.metric"] == ""
    assert records[1]["source_root"] == str(Path("/evidence"))
    assert rendered.endswith("\n")


def test_csv_empty_rows_still_has_deterministic_header() -> None:
    rendered = render_csv(())
    assert rendered.splitlines() == [
        ",".join(
            (
                "row_id",
                "kind",
                "current",
                "model_fingerprint",
                "model_name",
                "model_path",
                "quant",
                "hardware_hash",
                "build_discriminator",
                "build_commit",
                "ctx",
                "depth",
                "pp_workload",
                "tg_workload",
                "suite_id",
                "status",
                "confirmed",
                "replicated",
                "reps",
                "noise_floor_cv",
                "source_root",
                "evidence_dir",
                "ts",
            )
        )
    ]
