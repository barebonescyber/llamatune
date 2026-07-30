from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from llamatune import matrix
from llamatune.matrix import _fallbacks, _refinements, run_matrix
from llamatune.types import LlamaCppReport, MarathonOptions, ModelReport, TrialConfig


def config() -> TrialConfig:
    return TrialConfig(
        gpu_layers=4,
        moe_cpu_layers=0,
        flash_attn=True,
        ubatch=512,
        batch=1024,
        threads=4,
        mmap=True,
        no_kv_offload=False,
        cache_type_k="f16",
        cache_type_v="f16",
    )


def options(tmp_path: Path) -> MarathonOptions:
    return MarathonOptions(
        model_path=tmp_path / "m.gguf",
        llama_bin=None,
        sessions_dir=tmp_path,
        until=None,
        max_hours=None,
        rounds_max=1,
        converge_rounds=1,
        ab_blocks=3,
        depth_grid=(0, 8192),
        ctx_size=8192,
        ctx_ladder=(16384,),
        matrix_refine=True,
        drift_threshold=0.05,
        dry_run=False,
        target="balanced",
        allow_lossy=False,
        vram_reserve_mb=None,
        cooldown_s=0,
        full_hash=False,
        pp=512,
        tg=128,
        quality_corpus=None,
        ot_search=False,
        budget_trials=None,
        reps_search=None,
        reps_confirm=None,
        baseline_runs=None,
    )


def model(tmp_path: Path) -> ModelReport:
    return ModelReport(
        path=tmp_path / "m.gguf",
        size_bytes=1,
        architecture="x",
        n_layer=8,
        ngl_all=9,
        expert_count=8,
        moe=True,
        name=None,
        fingerprint="f" * 64,
        full_sha256=None,
    )


class Run:
    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.entries: list[dict[str, Any]] = []
        self.champion_evidence: dict[str, Any] | None = None

    def matrix_dir(self, ctx: int, depth: int, refine: int | None = None) -> Path:
        path = self.dir / f"{ctx}-{depth}-{refine}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def append(self, entry: dict[str, Any]) -> None:
        self.entries.append(entry)

    def write_json(self, name: str, payload: dict[str, Any]) -> None:
        raise AssertionError("deferred matrix must not execute")


def test_refinement_clamps_and_fallback_is_deterministic(tmp_path: Path) -> None:
    candidate = config()
    assert [value.ubatch for value in _refinements(candidate)] == [256, 1024]
    values = _fallbacks(candidate, model(tmp_path))
    assert values == _fallbacks(candidate, model(tmp_path))
    assert values[-1].gpu_layers == 0


def test_exhausted_time_defers_every_cell(tmp_path: Path) -> None:
    llama = LlamaCppReport(
        bench_path=tmp_path / "bench",
        cli_path=None,
        server_path=None,
        capabilities=frozenset({"d"}),
        help_sha256="h",
        build_commit=None,
        build_number=None,
        backends=None,
    )
    rows = run_matrix(
        Run(tmp_path),
        config(),
        model(tmp_path),
        llama,
        options(tmp_path),
        remaining_minutes_fn=lambda: 0,
    )
    assert len(rows) == 4
    assert {row.status for row in rows} == {"deferred"}


def test_matrix_measures_and_selects_better_bounded_refinement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llama = LlamaCppReport(
        bench_path=tmp_path / "bench",
        cli_path=None,
        server_path=None,
        capabilities=frozenset({"d"}),
        help_sha256="h",
        build_commit=None,
        build_number=None,
        backends=None,
    )
    values = iter(((1.0, 1.0), (10.0, 10.0), (20.0, 20.0), (15.0, 15.0)))
    monkeypatch.setattr(matrix, "_execute", lambda *_args: next(values))
    opts = dataclasses.replace(options(tmp_path), depth_grid=(0,), ctx_ladder=())
    run = Run(tmp_path)
    rows = run_matrix(
        run,
        config(),
        model(tmp_path),
        llama,
        opts,
        remaining_minutes_fn=lambda: 60.0,
    )
    assert len(rows) == 1
    assert rows[0].status == "ok"
    assert rows[0].refined is True
    assert rows[0].pp == 20.0
    assert rows[0].config is not None and rows[0].config.ubatch == 256
    assert run.entries[-1]["type"] == "matrix_cell"


def test_required_depth_zero_reuses_confirmed_champion_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llama = LlamaCppReport(
        bench_path=tmp_path / "bench",
        cli_path=None,
        server_path=None,
        capabilities=frozenset({"d"}),
        help_sha256="h",
        build_commit=None,
        build_number=None,
    )
    champion = config()
    run = Run(tmp_path)
    run.champion_evidence = {
        "ctx": 8192,
        "config": champion.to_dict(),
        "pp": 123.0,
        "tg": 45.0,
        "evidence": str(tmp_path / "analysis.json"),
    }
    monkeypatch.setattr(matrix, "_execute", lambda *_args: pytest.fail("reuse executed"))
    opts = dataclasses.replace(options(tmp_path), depth_grid=(0,), ctx_ladder=())
    rows = run_matrix(
        run,
        champion,
        model(tmp_path),
        llama,
        opts,
        remaining_minutes_fn=lambda: 60.0,
    )
    assert rows[0].pp == 123.0
    assert run.entries[-1]["reused"] is True


def test_matrix_uses_fallback_after_champion_probe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llama = LlamaCppReport(
        bench_path=tmp_path / "bench",
        cli_path=None,
        server_path=None,
        capabilities=frozenset({"d"}),
        help_sha256="h",
        build_commit=None,
        build_number=None,
        backends=None,
    )
    values = iter((None, (1.0, 1.0), (12.0, 4.0)))
    monkeypatch.setattr(matrix, "_execute", lambda *_args: next(values))
    opts = dataclasses.replace(
        options(tmp_path), depth_grid=(0,), ctx_ladder=(), matrix_refine=False
    )
    run = Run(tmp_path)
    rows = run_matrix(
        run,
        config(),
        model(tmp_path),
        llama,
        opts,
        remaining_minutes_fn=lambda: 60.0,
    )
    assert rows[0].status == "ok"
    assert rows[0].config is not None
    assert rows[0].config != config()
    assert rows[0].pp == 12.0


def test_matrix_prunes_larger_context_after_all_placements_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llama = LlamaCppReport(
        bench_path=tmp_path / "bench",
        cli_path=None,
        server_path=None,
        capabilities=frozenset({"d"}),
        help_sha256="h",
        build_commit=None,
        build_number=None,
        backends=None,
    )
    monkeypatch.setattr(matrix, "_execute", lambda *_args: None)
    opts = dataclasses.replace(options(tmp_path), depth_grid=(0,), ctx_ladder=(16384,))
    rows = run_matrix(
        Run(tmp_path),
        config(),
        model(tmp_path),
        llama,
        opts,
        remaining_minutes_fn=lambda: 60.0,
    )
    assert [row.status for row in rows] == ["failed", "pruned"]
