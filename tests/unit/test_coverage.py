"""Pure Marathon coverage tier and ledger tests."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from llamatune.coverage import TIER_D_CAP, build_ledger, enumerate_space
from llamatune.types import (
    GPUInfo,
    HardwareReport,
    LlamaCppReport,
    MarathonOptions,
    ModelReport,
    TrialConfig,
)


def _inputs(tmp_path: Path) -> tuple[ModelReport, LlamaCppReport, HardwareReport, MarathonOptions]:
    model = ModelReport(
        path=tmp_path / "model.gguf",
        size_bytes=1024,
        architecture="qwen",
        n_layer=8,
        ngl_all=9,
        expert_count=8,
        moe=True,
        name="test",
        fingerprint="f" * 64,
        full_sha256=None,
    )
    llama = LlamaCppReport(
        bench_path=tmp_path / "llama-bench",
        cli_path=None,
        server_path=None,
        capabilities=frozenset({"ncmoe", "fa", "mmp", "nkvo", "ctk", "ctv", "tb", "d"}),
        help_sha256="h",
        build_commit=None,
        build_number=None,
        backends="CUDA",
    )
    hardware = HardwareReport(
        os_name="Linux",
        arch="x86_64",
        cpu_model="fake",
        physical_cores=8,
        logical_cores=16,
        perf_cores=None,
        ram_mb=65536,
        gpus=(GPUInfo(vendor="NVIDIA", name="fake", vram_mb=24000, method="test"),),
        warnings=(),
    )
    options = MarathonOptions(
        model_path=model.path,
        llama_bin=None,
        sessions_dir=tmp_path,
        until=None,
        max_hours=None,
        rounds_max=6,
        converge_rounds=2,
        ab_blocks=5,
        depth_grid=(0, 8192),
        ctx_size=8192,
        ctx_ladder=(16384,),
        matrix_refine=True,
        drift_threshold=0.05,
        dry_run=False,
        target="balanced",
        allow_lossy=True,
        vram_reserve_mb=1536,
        cooldown_s=None,
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
    return model, llama, hardware, options


def _champion() -> TrialConfig:
    return TrialConfig(
        gpu_layers=9,
        moe_cpu_layers=4,
        flash_attn=True,
        ubatch=512,
        batch=2048,
        threads=8,
        mmap=True,
        no_kv_offload=False,
        cache_type_k="f16",
        cache_type_v="f16",
    )


def test_enumeration_is_deterministic_valid_and_capability_gated(tmp_path: Path) -> None:
    model, llama, hardware, options = _inputs(tmp_path)
    champion = _champion()
    space = enumerate_space(model, llama, hardware, options, champion=champion, known_placements=())
    again = enumerate_space(model, llama, hardware, options, champion=champion, known_placements=())
    assert space == again
    assert space["A"] and space["B"] and space["C"]
    assert space["D"] == ()
    assert all(config.ubatch <= config.batch for configs in space.values() for config in configs)
    no_optional = dataclasses.replace(llama, capabilities=frozenset())
    gated = enumerate_space(
        model, no_optional, hardware, options, champion=champion, known_placements=()
    )
    assert all(config.flash_attn == champion.flash_attn for config in gated["A"])


def test_tier_d_uses_responsive_dimensions_and_is_capped(tmp_path: Path) -> None:
    model, llama, hardware, options = _inputs(tmp_path)
    champion = _champion()
    space = enumerate_space(
        model,
        llama,
        hardware,
        options,
        champion=champion,
        known_placements={
            "placements": [(9, 4)],
            "responsive": ("ubatch", "batch", "threads", "moe_cpu_layers"),
        },
    )
    assert 0 < len(space["D"]) <= TIER_D_CAP
    assert len({config.trial_id for config in space["D"]}) == len(space["D"])


def test_ledger_classification_and_responsiveness_boundary(tmp_path: Path) -> None:
    champion = _champion()
    changed = dataclasses.replace(champion, threads=4)
    space = {"A": (champion, changed)}
    boundary = build_ledger(
        space,
        {champion.trial_id},
        {changed.trial_id},
        trials=(
            {"config": champion, "status": "ok", "score": 100.0, "tau": 0.01},
            {"config": changed, "status": "ok", "score": 101.0},
        ),
    )
    assert boundary.tiers["A"].executed == 1
    assert boundary.tiers["A"].pruned == 1
    assert boundary.tiers["A"].remaining == 0
    assert boundary.responsive == ()
    responsive = build_ledger(
        space,
        (),
        (),
        trials=(
            {"config": champion, "status": "ok", "score": 100.0, "tau": 0.01},
            {"config": changed, "status": "ok", "score": 101.01},
        ),
    )
    assert responsive.responsive == ("threads",)
