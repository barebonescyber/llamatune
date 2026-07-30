from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llamatune import abtest
from llamatune.abtest import _block_verdict, _pooled_verdict
from llamatune.types import LlamaCppReport, MarathonOptions, ModelReport, TrialConfig


def test_block_threshold_is_strict() -> None:
    verdict, _ = _block_verdict((101.0, 100.0), (100.0, 100.0), threshold=0.01, weights=(1.0, 0.0))
    assert verdict == "tie"
    verdict, _ = _block_verdict((101.01, 100.0), (100.0, 100.0), threshold=0.01, weights=(1.0, 0.0))
    assert verdict == "a"
    verdict, margin = _block_verdict(
        (90.0, 100.0), (100.0, 100.0), threshold=0.01, weights=(1.0, 0.0)
    )
    assert verdict == "b" and margin < 0


def test_majority_requires_dominant_metric_cross_check() -> None:
    assert (
        _pooled_verdict(
            a_wins=3,
            b_wins=1,
            blocks=5,
            a_pp=110,
            a_tg=100,
            b_pp=100,
            b_tg=102,
            threshold=0.01,
            target="generation",
        )
        == "tie"
    )
    assert (
        _pooled_verdict(
            a_wins=3,
            b_wins=1,
            blocks=5,
            a_pp=110,
            a_tg=103,
            b_pp=100,
            b_tg=102,
            threshold=0.01,
            target="generation",
        )
        == "a"
    )
    assert (
        _pooled_verdict(
            a_wins=1,
            b_wins=3,
            blocks=5,
            a_pp=100,
            a_tg=100,
            b_pp=103,
            b_tg=103,
            threshold=0.01,
            target="prompt",
        )
        == "b"
    )


class Run:
    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.entries: list[dict[str, Any]] = []
        self.decision_threshold = 0.01

    def append(self, entry: dict[str, Any]) -> None:
        self.entries.append(entry)

    def ab_dir(self, label: str, block: int, slot: str) -> Path:
        return self.dir / label / str(block) / slot

    def write_json(self, name: str, payload: dict[str, Any]) -> None:
        raise AssertionError("mocked measurement must not write")


def test_run_ab_executes_abba_blocks_and_pools_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = Run(tmp_path)
    config = TrialConfig(
        gpu_layers=1,
        moe_cpu_layers=0,
        flash_attn=False,
        ubatch=128,
        batch=512,
        threads=1,
        mmap=True,
        no_kv_offload=False,
        cache_type_k="f16",
        cache_type_v="f16",
    )
    model = ModelReport(
        path=tmp_path / "m.gguf",
        size_bytes=1,
        architecture="fake",
        n_layer=1,
        ngl_all=1,
        expert_count=0,
        moe=False,
        name=None,
        fingerprint="f" * 64,
        full_sha256=None,
    )
    llama = LlamaCppReport(
        bench_path=tmp_path / "bench",
        cli_path=None,
        server_path=None,
        capabilities=frozenset(),
        help_sha256="h",
        build_commit=None,
        build_number=None,
    )
    options = MarathonOptions(
        model_path=model.path,
        llama_bin=None,
        sessions_dir=tmp_path,
        until=None,
        max_hours=None,
        rounds_max=1,
        converge_rounds=1,
        ab_blocks=3,
        depth_grid=(0,),
        ctx_size=8192,
        ctx_ladder=(),
        matrix_refine=False,
        drift_threshold=0.05,
        dry_run=False,
        target="balanced",
        allow_lossy=False,
        vram_reserve_mb=None,
        cooldown_s=0,
        full_hash=False,
        pp=16,
        tg=8,
        quality_corpus=None,
        ot_search=False,
        budget_trials=None,
        reps_search=None,
        reps_confirm=None,
        baseline_runs=None,
    )
    assert abtest._reps(options) == 12
    assert abtest._cooldown(options) == 0
    tuned_argv = abtest._argv(config, model, llama, options)
    default_argv = abtest._argv(None, model, llama, options)
    assert "-ngl" in tuned_argv
    assert "-ngl" not in default_argv
    calls: list[tuple[int, str]] = []

    def measure(
        _run: object,
        measured_config: TrialConfig | None,
        *,
        block: int,
        slot: str,
        **_kwargs: object,
    ) -> tuple[float, float]:
        calls.append((block, slot))
        return (110.0, 110.0) if measured_config is config else (100.0, 100.0)

    monkeypatch.setattr(abtest, "_measure", measure)
    result = abtest.run_ab(
        run, config, None, blocks=3, model=model, llama=llama, options=options, label="final"
    )
    assert calls == [(block, slot) for block in range(1, 4) for slot in ("a1", "b1", "b2", "a2")]
    assert result.verdict == "a"
    assert result.a_wins == 3
    assert len(run.entries) == 3
    with pytest.raises(ValueError, match="positive"):
        abtest.run_ab(
            run, config, None, blocks=0, model=model, llama=llama, options=options, label="bad"
        )
