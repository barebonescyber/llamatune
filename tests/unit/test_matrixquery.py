from __future__ import annotations

import runpy
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from llamatune.matrixquery import USE_CASES, apply
from llamatune.resultsmatrix import harvest
from llamatune.types import MatrixQuerySpec, ResultRow, ResultsMatrix, TrialConfig

write_matrix_evidence = cast(
    Callable[..., dict[str, Path]],
    runpy.run_path(str(Path(__file__).parents[1] / "fixtures" / "matrix_evidence.py"))[
        "write_matrix_evidence"
    ],
)

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
        "model_fingerprint": "fixture-fingerprint",
        "model_name": f"Model {row_id}",
        "model_path": f"/models/{row_id}.gguf",
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
        generated="2026-01-02T00:00:00+00:00",
        roots=(Path("/evidence"),),
        rows=rows,
        warnings=("fixture warning",),
    )


def _spec(use_case: str | None = "max-pp", **changes: Any) -> MatrixQuerySpec:
    values: dict[str, Any] = {
        "use_case": use_case,
        "sort": None,
        "ascending": False,
        "model": None,
        "quant": None,
        "kinds": (),
        "suite": None,
        "ctx_min": None,
        "depth": None,
        "min_pp": None,
        "min_tg": None,
        "confirmed_only": True,
        "current_only": True,
        "compat": "all",
        "limit": 20,
    }
    values.update(changes)
    return MatrixQuerySpec(**values)


def _ids(payload: dict[str, Any]) -> list[str]:
    return [row["row_id"] for group in payload["groups"] for row in group["rows"]]


def test_use_case_registry_is_exact() -> None:
    assert tuple(USE_CASES) == (
        "max-pp",
        "max-tg",
        "balanced",
        "max-context",
        "coding",
        "tool-use",
        "agentic",
        "instruction",
        "quality-overall",
    )


@pytest.mark.parametrize(
    ("use_case", "metric", "suite"),
    [
        ("coding", "quality.coding.score", "coding"),
        ("tool-use", "quality.tooluse.score", "tooluse"),
        ("agentic", "quality.agentic.score", "agentic"),
        ("instruction", "quality.ifollow.score", "ifollow"),
        ("quality-overall", "quality.overall", "coding"),
    ],
)
def test_each_quality_use_case_filters_and_ranks(use_case: str, metric: str, suite: str) -> None:
    low = _row(
        "low",
        kind="quality_suite",
        suite_id=f"{suite}@1+abc",
        metrics={metric: 0.5},
    )
    high = _row(
        "high",
        kind="quality_suite",
        suite_id=f"{suite}@1+def",
        metrics={metric: 0.9},
    )
    wrong_pool = _row("wrong", metrics={metric: 99.0})
    missing = _row(
        "missing",
        kind="quality_suite",
        suite_id=f"{suite}@1+missing",
        metrics={"quality.unrelated": 1.0},
    )
    payload = apply(_matrix(low, wrong_pool, missing, high), _spec(use_case), None)
    assert _ids(payload) == ["high", "low"]
    assert payload["excluded"] == {"missing_metric": 1}


def test_performance_use_cases_use_raw_values_and_exact_tiebreaks() -> None:
    older = _row(
        "older",
        metrics={"perf.pp": 100.0, "perf.tg": 20.0, "perf.improvement_pp_pct": 999.0},
        ts="2026-01-01T00:00:00+00:00",
    )
    newer = _row(
        "newer",
        metrics={"perf.pp": 100.0, "perf.tg": 10.0, "perf.improvement_pp_pct": 1.0},
        ts="2026-01-02T00:00:00+00:00",
    )
    assert _ids(apply(_matrix(newer, older), _spec("max-pp"), None)) == ["older", "newer"]
    assert _ids(apply(_matrix(newer, older), _spec("max-pp", ascending=True), None)) == [
        "older",
        "newer",
    ]

    tg_a = replace(older, row_id="tg-a", metrics={"perf.pp": 90.0, "perf.tg": 30.0})
    tg_b = replace(newer, row_id="tg-b", metrics={"perf.pp": 110.0, "perf.tg": 30.0})
    assert _ids(apply(_matrix(tg_a, tg_b), _spec("max-tg"), None)) == ["tg-b", "tg-a"]

    balanced_old = replace(older, row_id="bal-old", metrics={"perf.pp": 100.0, "perf.tg": 4.0})
    balanced_new = replace(newer, row_id="bal-new", metrics={"perf.pp": 25.0, "perf.tg": 16.0})
    assert _ids(apply(_matrix(balanced_old, balanced_new), _spec("balanced"), None)) == [
        "bal-new",
        "bal-old",
    ]
    assert _ids(
        apply(_matrix(balanced_old, balanced_new), _spec("balanced", ascending=True), None)
    ) == ["bal-new", "bal-old"]


def test_max_context_pool_metric_tiebreak_and_excluded_count() -> None:
    fast = _row(
        "fast",
        kind="context_envelope",
        metrics={"ctx.validated": 32768.0, "perf.tg": 12.0},
    )
    slow = _row(
        "slow",
        kind="depth_profile",
        metrics={"ctx.validated": 32768.0, "perf.tg": 8.0},
        ts="2026-01-02T00:00:00+00:00",
    )
    missing = _row("missing", kind="context_envelope", metrics={"perf.tg": 99.0})
    payload = apply(
        _matrix(missing, slow, fast),
        _spec("max-context", confirmed_only=False),
        None,
    )
    assert _ids(payload) == ["fast", "slow"]
    assert payload["excluded"] == {"missing_metric": 1}


def test_quality_tie_prefers_same_config_recommendation_tg_then_newer() -> None:
    other_config = replace(_CONFIG, gpu_layers=20)
    recommendation = _row("recommendation", metrics={"perf.pp": 80.0, "perf.tg": 7.0})
    matched = _row(
        "matched",
        kind="quality_suite",
        suite_id="coding@1+one",
        metrics={"quality.coding.score": 0.9},
        ts="2026-01-01T00:00:00+00:00",
    )
    unmatched = _row(
        "unmatched",
        kind="quality_suite",
        suite_id="coding@1+two",
        config=other_config,
        metrics={"quality.coding.score": 0.9},
        ts="2026-01-03T00:00:00+00:00",
    )
    assert _ids(apply(_matrix(unmatched, matched, recommendation), _spec("coding"), None)) == [
        "matched",
        "unmatched",
    ]
    older = replace(unmatched, row_id="older", ts="2026-01-01T00:00:00+00:00")
    newer = replace(unmatched, row_id="newer", ts="2026-01-02T00:00:00+00:00")
    assert _ids(apply(_matrix(older, newer), _spec("coding"), None)) == ["newer", "older"]

    defaults_recommendation = replace(recommendation, row_id="defaults-rec", config=None)
    defaults_quality = replace(matched, row_id="defaults-quality", config=None)
    assert _ids(
        apply(
            _matrix(unmatched, defaults_quality, defaults_recommendation),
            _spec("coding"),
            None,
        )
    ) == ["defaults-quality", "unmatched"]


def test_rankings_are_grouped_by_hardware_and_current_group_is_first() -> None:
    foreign = _row(
        "foreign",
        hardware_hash="hardware-b",
        hardware_signature=("Linux", "arm64", "GPU B"),
        metrics={"perf.pp": 1000.0, "perf.tg": 20.0},
    )
    local = _row("local", metrics={"perf.pp": 10.0, "perf.tg": 2.0})
    payload = apply(_matrix(foreign, local), _spec(), ("hardware-a", "build-a"))
    assert [group["hardware_hash"] for group in payload["groups"]] == [
        "hardware-a",
        "hardware-b",
    ]
    assert [row["rank"] for group in payload["groups"] for row in group["rows"]] == [1, 1]


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"current": False}, []),
        ({"model": "target", "model_name": "Target Model"}, ["model"]),
        ({"quant": "Q8_0"}, []),
        ({"kind": "baseline"}, ["model"]),
        ({"suite_id": "coding@1+x"}, []),
        ({"ctx": 1024}, []),
        ({"depth": 2048}, []),
        ({"metrics": {"perf.pp": 9.0, "perf.tg": 10.0}}, []),
        ({"metrics": {"perf.pp": 100.0, "perf.tg": 1.0}}, []),
        ({"confirmed": False}, []),
    ],
)
def test_individual_filters(change: dict[str, Any], expected: list[str]) -> None:
    row_change = dict(change)
    spec_change: dict[str, Any] = {}
    if "model" in row_change:
        spec_change["model"] = row_change.pop("model")
    elif "quant" in row_change:
        spec_change["quant"] = "Q4_K_M"
    elif "kind" in row_change:
        spec_change["kinds"] = ("baseline",)
    elif "suite_id" in row_change:
        spec_change["suite"] = "tooluse"
    elif "ctx" in row_change:
        spec_change["ctx_min"] = 2048
    elif "depth" in row_change:
        spec_change["depth"] = 4096
    elif "metrics" in row_change:
        if row_change["metrics"]["perf.pp"] == 9.0:
            spec_change["min_pp"] = 10.0
        else:
            spec_change["min_tg"] = 2.0
    row = _row("model", **row_change)
    assert _ids(apply(_matrix(row), _spec(**spec_change), None)) == expected


def test_unconfirmed_superseded_and_combined_filters() -> None:
    row = _row("all", current=False, confirmed=False)
    widened = _spec(
        model="Model all",
        quant="q4_k_m",
        kinds=("recommendation",),
        ctx_min=8192,
        depth=4096,
        min_pp=100.0,
        min_tg=10.0,
        current_only=False,
        confirmed_only=False,
    )
    assert _ids(apply(_matrix(row), widened, None)) == ["all"]


def test_all_compatibility_annotations_and_filter() -> None:
    rows = (
        _row("current"),
        _row("build", build_discriminator="build-b"),
        _row("hardware", hardware_hash="hardware-b"),
        _row("both", hardware_hash="hardware-b", build_discriminator="build-b"),
    )
    payload = apply(_matrix(*rows), _spec(), ("hardware-a", "build-a"))
    annotations = {
        row["row_id"]: row["compat"] for group in payload["groups"] for row in group["rows"]
    }
    assert annotations == {
        "current": "current",
        "build": "build_changed",
        "hardware": "hardware_changed",
        "both": "both",
    }
    unknown = apply(_matrix(rows[0]), _spec(), None)
    assert unknown["groups"][0]["rows"][0]["compat"] == "unknown"
    filtered = apply(_matrix(*rows), _spec(compat="current"), ("hardware-a", "build-a"))
    assert _ids(filtered) == ["current"]


def test_custom_sort_ascending_limit_and_payload_shape() -> None:
    first = _row("first", metrics={"custom": 1.0})
    second = _row("second", metrics={"custom": 2.0})
    missing = _row("missing", metrics={})
    payload = apply(
        _matrix(second, missing, first),
        _spec(None, sort="custom", ascending=True, limit=1),
        None,
    )
    assert payload["sort"] == "custom"
    assert "use_case" not in payload
    assert _ids(payload) == ["first"]
    assert payload["excluded"] == {"missing_metric": 1}
    assert payload["warnings"] == ["fixture warning"]

    unlimited = apply(_matrix(second, first), _spec(None, sort="custom", limit=0), None)
    assert _ids(unlimited) == ["second", "first"]


def test_fixture_quality_row_is_queryable_before_quality_wave(tmp_path: Path) -> None:
    write_matrix_evidence(tmp_path)
    matrix = harvest((tmp_path,))
    payload = apply(matrix, _spec("coding"), None)
    assert len(_ids(payload)) == 1
    row = payload["groups"][0]["rows"][0]
    assert row["suite_id"].startswith("coding@")
    assert row["metrics"]["quality.coding.score"] == 0.9


def test_invalid_query_specs_are_rejected() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        apply(_matrix(), _spec("max-pp", sort="perf.pp"), None)
    with pytest.raises(ValueError, match="one of"):
        apply(_matrix(), _spec(None), None)
    with pytest.raises(ValueError, match="unknown"):
        apply(_matrix(), _spec("made-up"), None)
    with pytest.raises(ValueError, match="non-negative"):
        apply(_matrix(), _spec(limit=-1), None)
