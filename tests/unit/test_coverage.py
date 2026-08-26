"""Pure Marathon coverage tier and ledger tests."""

from __future__ import annotations

import dataclasses
import itertools
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from llamatune.config import DIMENSION_ORDER, dimension_value
from llamatune.coverage import (
    TIER_D_CAP,
    _responsive_dimensions,
    _trial_parts,
    build_ledger,
    enumerate_space,
)
from llamatune.types import (
    GPUInfo,
    HardwareReport,
    LlamaCppReport,
    MarathonOptions,
    MetricStats,
    ModelReport,
    TrialConfig,
    TrialResult,
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


def _reference_responsive(trials: tuple[Any, ...], *, tau: float) -> tuple[str, ...]:
    """Historical pairwise algorithm, kept verbatim as the test oracle."""
    ok = [parts for trial in trials if (parts := _trial_parts(trial))[1] == "ok"]
    responsive: list[str] = []
    for dim in DIMENSION_ORDER:
        for left, right in itertools.combinations(ok, 2):
            a, _, a_score = left
            b, _, b_score = right
            if a_score is None or b_score is None or min(a_score, b_score) <= 0:
                continue
            differences = [
                name
                for name in DIMENSION_ORDER
                if dimension_value(a, name) != dimension_value(b, name)
            ]
            if differences == [dim] and max(a_score, b_score) / min(a_score, b_score) > 1 + tau:
                responsive.append(dim)
                break
    return tuple(responsive)


def _trial_result(config: TrialConfig) -> TrialResult:
    return TrialResult(
        trial_id=config.trial_id,
        config=config,
        status="ok",
        pp=MetricStats(mean=100.0, stdev=1.0, cv=0.01, n=2),
        tg=MetricStats(mean=50.0, stdev=1.0, cv=0.01, n=2),
        wall_s=1.0,
        exit_code=0,
        oom_pattern=None,
        artifact_dir=None,
        flags=(),
    )


def test_responsive_dimensions_matches_pairwise_oracle_on_synthetic_set() -> None:
    rng = random.Random(20260825)  # noqa: S311 -- deterministic synthetic evidence
    base = _champion()
    mutation_dims = ("ubatch", "batch", "threads", "mmap", "kv_offload", "flash_attn")
    grids: dict[str, tuple[Any, ...]] = {
        "ubatch": (128, 256, 512, 1024),
        "batch": (512, 2048, 8192),
        "threads": (1, 2, 4, 8),
        "mmap": (True, False),
        "kv_offload": (False, True),
        "flash_attn": (False, True),
    }
    trials: list[dict[str, Any] | TrialResult] = []
    while len(trials) < 64:
        config = base
        touched = [dim for dim in mutation_dims if rng.random() < 0.5 or len(trials) % 7 == 0] or [
            rng.choice(mutation_dims)
        ]
        for dim in touched:
            value = rng.choice(grids[dim])
            field = {
                "ubatch": "ubatch",
                "batch": "batch",
                "threads": "threads",
                "mmap": "mmap",
                "kv_offload": "no_kv_offload",
                "flash_attn": "flash_attn",
            }[dim]
            config = dataclasses.replace(config, **{field: value})
        roll = rng.random()
        score: float | None
        if roll < 0.12:
            status, score = "error", None
        elif roll < 0.18:
            status, score = "ok", None
        elif roll < 0.24:
            status, score = "ok", -abs(rng.uniform(1.0, 50.0))
        else:
            status, score = "ok", rng.uniform(10.0, 400.0)
        entry: dict[str, Any] = {"config": config, "status": status}
        if score is not None:
            entry["score"] = score
        if rng.random() < 0.3:
            entry["tau"] = rng.choice([0.01, 0.05, 0.2])
        trials.append(entry)
    trials.append({"config": base, "status": "ok", "score": 250.0})
    trials.append(_trial_result(dataclasses.replace(base, ubatch=128)))
    materialized = tuple(trials)

    taus = [float(t["tau"]) for t in trials if isinstance(t, dict) and "tau" in t]
    tau = max(0.01, *taus)
    assert len(materialized) >= 50
    assert _responsive_dimensions(materialized, tau=tau) == _reference_responsive(
        materialized, tau=tau
    )
    assert build_ledger({}, (), (), trials=materialized).responsive == _reference_responsive(
        materialized, tau=tau
    )


def test_responsive_dimensions_identical_pair_never_marks_dimension() -> None:
    first = _champion()
    second = dataclasses.replace(first, threads=_champion().threads)
    assert first == second
    mapping_first: Mapping[str, Any] = {"config": first, "status": "ok", "score": 100.0}
    mapping_second: Mapping[str, Any] = {"config": second, "status": "ok", "score": 300.0}
    assert _responsive_dimensions((mapping_first, mapping_second), tau=0.01) == ()
