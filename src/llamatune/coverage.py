"""Pure Marathon search-space enumeration and coverage accounting."""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from llamatune.config import (
    BATCH_CANDIDATES,
    CACHE_TYPE_CANDIDATES,
    DIMENSION_ORDER,
    FLASH_ATTN_CANDIDATES,
    KV_OFFLOAD_CANDIDATES,
    MMAP_CANDIDATES,
    UBATCH_CANDIDATES,
    dimension_value,
    gpu_available,
    gpu_layer_candidates,
    is_valid_config,
    moe_cpu_layer_candidates,
    mutate,
    thread_candidates,
)
from llamatune.types import (
    CoverageLedger,
    HardwareReport,
    LlamaCppReport,
    MarathonOptions,
    ModelReport,
    TierCoverage,
    TrialConfig,
    TrialResult,
)

TIER_D_CAP = 256


def _unique(configs: Iterable[TrialConfig]) -> tuple[TrialConfig, ...]:
    by_id: dict[str, TrialConfig] = {}
    for config in configs:
        by_id.setdefault(config.trial_id, config)
    return tuple(by_id[key] for key in sorted(by_id))


def _valid(
    config: TrialConfig,
    *,
    model: ModelReport,
    llama: LlamaCppReport,
    hardware: HardwareReport,
) -> bool:
    return is_valid_config(config, model=model, llama=llama, hardware=hardware)


def _placement_inputs(
    known_placements: Sequence[Any] | Mapping[str, Any],
) -> tuple[Sequence[Any], tuple[str, ...]]:
    if isinstance(known_placements, Mapping):
        placements = known_placements.get("placements", ())
        responsive = tuple(str(value) for value in known_placements.get("responsive", ()))
        return placements, responsive
    return known_placements, ()


def _placement_pair(value: Any) -> tuple[int, int]:
    if isinstance(value, TrialConfig):
        return value.gpu_layers, value.moe_cpu_layers
    if isinstance(value, Mapping):
        return int(value["gpu_layers"]), int(value["moe_cpu_layers"])
    pair = tuple(value)
    return int(pair[0]), int(pair[1])


def _dimension_candidates(
    dim: str,
    *,
    champion: TrialConfig,
    model: ModelReport,
    llama: LlamaCppReport,
    hardware: HardwareReport,
    options: MarathonOptions,
) -> tuple[Any, ...]:
    gpu = gpu_available(hardware, llama)
    if dim == "gpu_layers":
        return gpu_layer_candidates(model.ngl_all) if gpu else (0,)
    if dim == "moe_cpu_layers":
        if not (gpu and model.moe and "ncmoe" in llama.capabilities):
            return (0,)
        return moe_cpu_layer_candidates(model.n_layer)
    if dim == "flash_attn":
        return FLASH_ATTN_CANDIDATES if "fa" in llama.capabilities else (champion.flash_attn,)
    if dim == "ubatch":
        return UBATCH_CANDIDATES
    if dim == "batch":
        return BATCH_CANDIDATES
    if dim in {"threads", "threads_batch"}:
        if dim == "threads_batch" and "tb" not in llama.capabilities:
            return ()
        return thread_candidates(
            hardware.physical_cores, hardware.logical_cores, hardware.perf_cores
        )
    if dim == "mmap":
        return MMAP_CANDIDATES if "mmp" in llama.capabilities else ()
    if dim == "kv_offload":
        return KV_OFFLOAD_CANDIDATES if gpu and "nkvo" in llama.capabilities else ()
    if dim == "cache_type_k":
        return CACHE_TYPE_CANDIDATES if options.allow_lossy and "ctk" in llama.capabilities else ()
    if dim == "cache_type_v":
        if options.allow_lossy and "ctv" in llama.capabilities and champion.flash_attn:
            return CACHE_TYPE_CANDIDATES
        return ()
    return ()


def enumerate_space(
    model: ModelReport,
    llama: LlamaCppReport,
    hardware: HardwareReport,
    options: MarathonOptions,
    *,
    champion: TrialConfig,
    known_placements: Sequence[Any] | Mapping[str, Any],
) -> dict[str, tuple[TrialConfig, ...]]:
    """Build the deterministic, capability-gated Marathon tier universe.

    ``known_placements`` is ordered best-first.  A mapping form may additionally
    carry ``responsive`` dimension names for Tier D while retaining the frozen
    public call signature.
    """
    supplied, responsive = _placement_inputs(known_placements)
    placements = [_placement_pair(value) for value in supplied[:2]]
    if not placements:
        gpu_values = gpu_layer_candidates(model.ngl_all) if gpu_available(hardware, llama) else (0,)
        moe_values = (
            moe_cpu_layer_candidates(model.n_layer)
            if model.moe and "ncmoe" in llama.capabilities and gpu_values[-1] > 0
            else (0,)
        )
        placements.append((gpu_values[-1], moe_values[-1]))
        if len(gpu_values) > 1:
            placements.append((gpu_values[-2], moe_values[-1]))

    tier_a: list[TrialConfig] = []
    flash_values = FLASH_ATTN_CANDIDATES if "fa" in llama.capabilities else (champion.flash_attn,)
    for (gpu_layers, moe_layers), ubatch, batch, flash in itertools.product(
        placements, UBATCH_CANDIDATES, BATCH_CANDIDATES, flash_values
    ):
        candidate = dataclasses.replace(
            champion,
            gpu_layers=gpu_layers,
            moe_cpu_layers=moe_layers if gpu_layers > 0 else 0,
            ubatch=ubatch,
            batch=batch,
            flash_attn=flash,
        )
        if _valid(candidate, model=model, llama=llama, hardware=hardware):
            tier_a.append(candidate)

    tier_b: list[TrialConfig] = []
    threads = thread_candidates(
        hardware.physical_cores, hardware.logical_cores, hardware.perf_cores
    )
    moe_values = (
        moe_cpu_layer_candidates(model.n_layer)
        if champion.gpu_layers > 0 and model.moe and "ncmoe" in llama.capabilities
        else (0,)
    )
    for thread, moe_layers in itertools.product(threads, moe_values):
        candidate = dataclasses.replace(champion, threads=thread, moe_cpu_layers=moe_layers)
        if _valid(candidate, model=model, llama=llama, hardware=hardware):
            tier_b.append(candidate)

    tier_c: list[TrialConfig] = []
    for dim in DIMENSION_ORDER:
        if dim in {"gpu_layers", "moe_cpu_layers", "flash_attn", "ubatch", "batch"}:
            continue
        for value in _dimension_candidates(
            dim,
            champion=champion,
            model=model,
            llama=llama,
            hardware=hardware,
            options=options,
        ):
            candidate = mutate(champion, dim, value)
            if _valid(candidate, model=model, llama=llama, hardware=hardware):
                tier_c.append(candidate)

    responsive = tuple(dim for dim in DIMENSION_ORDER if dim in set(responsive))
    tier_d_ranked: list[tuple[int, TrialConfig]] = []
    if responsive:
        value_grid = [
            _dimension_candidates(
                dim,
                champion=champion,
                model=model,
                llama=llama,
                hardware=hardware,
                options=options,
            )
            for dim in responsive
        ]
        if all(value_grid):
            for values in itertools.product(*value_grid):
                candidate = champion
                distance = 0
                for dim, value in zip(responsive, values, strict=True):
                    distance += value != dimension_value(champion, dim)
                    candidate = mutate(candidate, dim, value)
                if _valid(candidate, model=model, llama=llama, hardware=hardware):
                    tier_d_ranked.append((distance, candidate))
            ordered = sorted(tier_d_ranked, key=lambda item: (item[0], item[1].trial_id))
            tier_d_configs = _unique(item[1] for item in ordered[:TIER_D_CAP])
        else:
            tier_d_configs = ()
    else:
        tier_d_configs = ()

    return {
        "A": _unique(tier_a),
        "B": _unique(tier_b),
        "C": _unique(tier_c),
        "D": tier_d_configs,
    }


def _trial_parts(trial: TrialResult | Mapping[str, Any]) -> tuple[TrialConfig, str, float | None]:
    if isinstance(trial, TrialResult):
        score = None
        if trial.pp is not None and trial.tg is not None:
            score = (trial.pp.mean * trial.tg.mean) ** 0.5
        return trial.config, trial.status, score
    raw_config = trial.get("config")
    if isinstance(raw_config, TrialConfig):
        config = raw_config
    elif isinstance(raw_config, dict):
        config = TrialConfig.from_dict(raw_config)
    else:
        raise TypeError("trial config must be TrialConfig or a config dictionary")
    raw_score = trial.get("score")
    score = float(raw_score) if raw_score is not None else None
    return config, str(trial.get("status", "ok")), score


def _responsive_dimensions(
    trials: Iterable[TrialResult | Mapping[str, Any]], *, tau: float
) -> tuple[str, ...]:
    """Return dimensions with an ok pair differing only there beyond 1 + tau.

    Equivalent to the historical pairwise scan but linear in trials: configs
    parse once, each trial reduces to one value tuple across
    ``DIMENSION_ORDER``, and trials are grouped per dimension by their
    all-dims-except-one value signature. Within a group only the best and
    worst score per differing value can form the extremal pair, so each
    group collapses to a min/max fold over its distinct values.
    """
    usable = [
        (_config_values(config), score)
        for config, status, score in (_trial_parts(trial) for trial in trials)
        if status == "ok" and score is not None and score > 0
    ]
    threshold = 1.0 + tau
    responsive: list[str] = []
    for index in range(len(DIMENSION_ORDER)):
        groups: dict[tuple[Any, ...], dict[Any, tuple[float, float]]] = {}
        for values, score in usable:
            key = values[:index] + values[index + 1 :]
            per_value = groups.setdefault(key, {})
            low, high = per_value.get(values[index], (score, score))
            per_value[values[index]] = (min(low, score), max(high, score))
        if any(
            max(high1, high2) / min(low1, low2) > threshold
            for per_value in groups.values()
            if len(per_value) > 1
            for (_, (low1, high1)), (_, (low2, high2)) in itertools.combinations(
                per_value.items(), 2
            )
        ):
            responsive.append(DIMENSION_ORDER[index])
    return tuple(responsive)


def _config_values(config: TrialConfig) -> tuple[Any, ...]:
    return tuple(dimension_value(config, dim) for dim in DIMENSION_ORDER)


def build_ledger(
    space: Mapping[str, Sequence[TrialConfig]],
    executed_ids: Iterable[str],
    pruned_ids: Iterable[str],
    *,
    trials: Iterable[TrialResult | Mapping[str, Any]],
) -> CoverageLedger:
    """Classify every enumerated config and derive responsive dimensions."""
    executed = set(executed_ids)
    pruned = set(pruned_ids) - executed
    materialized_trials = tuple(trials)
    tau = 0.01
    for trial in materialized_trials:
        if isinstance(trial, Mapping) and trial.get("tau") is not None:
            tau = max(tau, float(trial["tau"]))
    tiers: dict[str, TierCoverage] = {}
    for tier, configs in space.items():
        ids = tuple(dict.fromkeys(config.trial_id for config in configs))
        executed_count = sum(trial_id in executed for trial_id in ids)
        pruned_count = sum(trial_id in pruned for trial_id in ids)
        remaining_ids = tuple(
            trial_id for trial_id in ids if trial_id not in executed and trial_id not in pruned
        )
        tiers[tier] = TierCoverage(
            enumerated=len(ids),
            executed=executed_count,
            pruned=pruned_count,
            remaining=len(remaining_ids),
            remaining_ids=remaining_ids,
        )
    return CoverageLedger(
        tiers=tiers,
        responsive=_responsive_dimensions(materialized_trials, tau=tau),
    )
