"""Search-space construction, constraints, and feasibility (DESIGN §9).

Pure functions only: no I/O, no process execution, no randomness. This
module imports nothing above the standard library plus :mod:`llamatune.types`
(see DESIGN §13 layering rules).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from llamatune.types import (
    HardwareReport,
    LlamaCppReport,
    ModelReport,
    RamEstimate,
    TrialConfig,
    TuneOptions,
    VramCalibration,
    VramEstimate,
)

#: Coordinate-ascent visitation order (DESIGN §10 step 2), by expected impact.
DIMENSION_ORDER: tuple[str, ...] = (
    "gpu_layers",
    "moe_cpu_layers",
    "flash_attn",
    "ubatch",
    "batch",
    "threads",
    "threads_batch",
    "mmap",
    "kv_offload",
    "cache_type_k",
    "cache_type_v",
)

#: Dimensions whose non-default candidate values may affect output quality.
LOSSY_DIMENSIONS: frozenset[str] = frozenset({"cache_type_k", "cache_type_v"})

#: Maps a search dimension name to the TrialConfig field it mutates. Differs
#: from the identity mapping only for "kv_offload", which controls the
#: `no_kv_offload` field.
_DIMENSION_FIELD: dict[str, str] = {
    "gpu_layers": "gpu_layers",
    "moe_cpu_layers": "moe_cpu_layers",
    "flash_attn": "flash_attn",
    "ubatch": "ubatch",
    "batch": "batch",
    "threads": "threads",
    "threads_batch": "threads_batch",
    "mmap": "mmap",
    "kv_offload": "no_kv_offload",
    "cache_type_k": "cache_type_k",
    "cache_type_v": "cache_type_v",
}

_FRACTIONS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

UBATCH_CANDIDATES: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096)
BATCH_CANDIDATES: tuple[int, ...] = (512, 1024, 2048, 4096)
CACHE_TYPE_CANDIDATES: tuple[str, ...] = ("f16", "q8_0", "q4_0")
FLASH_ATTN_CANDIDATES: tuple[bool, ...] = (False, True)
MMAP_CANDIDATES: tuple[bool, ...] = (True, False)
KV_OFFLOAD_CANDIDATES: tuple[bool, ...] = (False, True)


def multi_gpu_placement_candidates(
    hardware: HardwareReport, llama: LlamaCppReport, options: TuneOptions
) -> tuple[tuple[tuple[float, ...], str], ...]:
    """Return the bounded, deterministic multi-GPU placement grid."""
    count = len(hardware.gpus)
    if not options.multi_gpu or count < 2 or not {"ts", "sm"}.issubset(llama.capabilities):
        return ()
    capacities = tuple(float(gpu.vram_mb or 0) for gpu in hardware.gpus)
    total = sum(capacities)
    proportional = capacities if total > 0 else tuple(1.0 for _ in range(count))
    splits = [proportional, tuple(1.0 for _ in range(count))]
    splits.extend(
        tuple(1.0 if index == selected else 0.0 for index in range(count))
        for selected in range(count)
    )
    unique: list[tuple[float, ...]] = []
    for split in splits:
        if split not in unique:
            unique.append(split)
    return tuple((split, mode) for split in unique for mode in ("layer", "row"))


def multi_gpu_capacity_check(
    config: TrialConfig,
    hardware: HardwareReport,
    estimated_total_mb: float,
    reserve_mb: int,
) -> tuple[bool | None, tuple[float, ...]]:
    """Advisory per-device placement feasibility; runtime remains authoritative."""
    if config.tensor_split is None or len(config.tensor_split) != len(hardware.gpus):
        return None, ()
    weight_total = sum(config.tensor_split)
    if weight_total <= 0:
        return False, ()
    estimates = tuple(estimated_total_mb * weight / weight_total for weight in config.tensor_split)
    capacities = tuple(gpu.vram_mb for gpu in hardware.gpus)
    if any(capacity is None for capacity in capacities):
        return None, estimates
    known_capacities = tuple(capacity for capacity in capacities if capacity is not None)
    fits = all(
        estimate <= max(0, capacity - reserve_mb)
        for estimate, capacity in zip(estimates, known_capacities, strict=True)
    )
    return fits, estimates


def gpu_layer_candidates(ngl_all: int) -> tuple[int, ...]:
    """{0, 1/4, 1/2, 3/4, 1} x ngl_all, rounded, clamped, deduplicated."""
    if ngl_all <= 0:
        return (0,)
    values = {max(0, min(ngl_all, round(f * ngl_all))) for f in _FRACTIONS}
    return tuple(sorted(values))


def moe_cpu_layer_candidates(n_layer: int) -> tuple[int, ...]:
    """{0, 1/4, 1/2, 3/4, 1} x n_layer, rounded, clamped, deduplicated."""
    if n_layer <= 0:
        return (0,)
    values = {max(0, min(n_layer, round(f * n_layer))) for f in _FRACTIONS}
    return tuple(sorted(values))


def thread_candidates(
    physical_cores: int, logical_cores: int, perf_cores: int | None
) -> tuple[int, ...]:
    """Return DESIGN §9.1's measured physical-to-logical thread search.

    Logical-core candidates remain intentional for CPU and hybrid workloads:
    the search measures the installed llama.cpp build instead of assuming that
    simultaneous multithreading is always slower.
    """
    half = max(1, round(physical_cores / 2))
    raw = {
        physical_cores,
        half,
        logical_cores,
        round((half + physical_cores) / 2),
        round((physical_cores + logical_cores) / 2),
    }
    if perf_cores is not None and perf_cores > 0:
        raw.add(perf_cores)
    clamped = {max(1, min(logical_cores, v)) for v in raw}
    return tuple(sorted(clamped))


def gpu_present(hardware: HardwareReport) -> bool:
    return len(hardware.gpus) > 0


def total_vram_mb(hardware: HardwareReport, *, multi_gpu: bool = False) -> int | None:
    """Known GPU capacity, pooled only when the multi-GPU mode is enabled.

    The default feasibility estimate remains anchored on one device; the
    opt-in placement sweep uses the sum across devices with known capacity.
    """
    vram_values = [g.vram_mb for g in hardware.gpus if g.vram_mb is not None]
    if not vram_values:
        return None
    return sum(vram_values) if multi_gpu else max(vram_values)


def free_vram_mb(hardware: HardwareReport, *, multi_gpu: bool = False) -> int | None:
    values = [gpu.vram_free_mb for gpu in hardware.gpus if gpu.vram_free_mb is not None]
    if not values:
        return None
    return sum(values) if multi_gpu else max(values)


def hardware_signature(hardware: HardwareReport) -> tuple[Any, ...]:
    """Pure identity tuple shared by resume validation and the registry."""
    return (
        hardware.os_name,
        hardware.arch,
        hardware.physical_cores,
        hardware.logical_cores,
        hardware.ram_mb,
        tuple((gpu.vendor, gpu.name, gpu.vram_mb) for gpu in hardware.gpus),
    )


def est_vram_mb(
    *, gpu_layers: int, moe_cpu_layers: int, model_size_bytes: int, ngl_all: int, n_layer: int
) -> float:
    """Estimated VRAM footprint (MiB) for a config (DESIGN §9.3)."""
    if ngl_all <= 0 or n_layer <= 0:
        return 0.0
    model_size_mb = model_size_bytes / (1024 * 1024)
    offload_fraction = gpu_layers / ngl_all
    moe_discount = 1 - 0.6 * (moe_cpu_layers / n_layer)
    return model_size_mb * offload_fraction * moe_discount


def estimate_vram(
    *,
    config: TrialConfig,
    model: ModelReport,
    ctx: int | None,
    vram_reserve_mb: int,
    vram_total_mb: int | None,
    vram_free_mb: int | None = None,
    calibration: VramCalibration | None = None,
) -> VramEstimate:
    """Advisory GPU-memory estimate; runtime probes remain authoritative.

    The deliberately coarse terms are documented in DESIGN §9.3.  They are useful
    for candidate ordering, not as a claim about exact llama.cpp allocations.
    """
    mib = 1024 * 1024
    gpu_layers = model.ngl_all if config.gpu_layers == -1 else config.gpu_layers
    gpu_fraction = min(1.0, max(0.0, gpu_layers / max(1, model.ngl_all)))
    expert_gpu_fraction = gpu_fraction * max(
        0.0, 1.0 - config.moe_cpu_layers / max(1, model.n_layer)
    )
    if model.expert_bytes is not None and model.dense_bytes is not None:
        weights_mb = (
            model.dense_bytes / mib * gpu_fraction + model.expert_bytes / mib * expert_gpu_fraction
        )
    else:
        weights_mb = est_vram_mb(
            gpu_layers=config.gpu_layers,
            moe_cpu_layers=config.moe_cpu_layers,
            model_size_bytes=model.size_bytes,
            ngl_all=model.ngl_all,
            n_layer=model.n_layer,
        )

    workload_tokens = ctx if ctx is not None else 640
    cache_scale = {
        "f32": 2.0,
        "f16": 1.0,
        "bf16": 1.0,
        "q8_0": 0.5,
        "q5_1": 0.375,
        "q5_0": 0.375,
        "q4_1": 0.25,
        "q4_0": 0.25,
    }
    kv_type_scale = (
        cache_scale.get(config.cache_type_k.lower(), 1.0)
        + cache_scale.get(config.cache_type_v.lower(), 1.0)
    ) / 2
    kv_basis = "metadata" if model.kv_bytes_per_token_f16 is not None else "heuristic"
    if config.no_kv_offload or gpu_layers <= 0:
        kv_mb = 0.0
    elif model.kv_bytes_per_token_f16 is not None:
        kv_mb = workload_tokens * model.kv_bytes_per_token_f16 * kv_type_scale * gpu_fraction / mib
    else:
        kv_mb = model.n_layer * workload_tokens / 32 * kv_type_scale * gpu_fraction
    compute_mb = 0.0 if gpu_layers <= 0 else 128.0 + config.batch / 16 + config.ubatch / 8
    if calibration is not None:
        weights_mb *= calibration.weights_scale
        kv_mb *= calibration.kv_scale
        compute_mb *= calibration.compute_scale
    total_mb = weights_mb + kv_mb + compute_mb
    total_budget = (
        None if vram_total_mb is None else max(0.0, float(vram_total_mb - vram_reserve_mb))
    )
    free_budget = None if vram_free_mb is None else max(0.0, float(vram_free_mb - 256))
    budget_mb: float | None
    if total_budget is not None and free_budget is not None:
        budget_mb = min(total_budget, free_budget)
        budget_basis = "observed-free" if free_budget <= total_budget else "total-reserve"
    else:
        budget_mb = total_budget if total_budget is not None else free_budget
        budget_basis = (
            "observed-free" if total_budget is None and free_budget is not None else "total-reserve"
        )
    return VramEstimate(
        weights_mb=weights_mb,
        kv_mb=kv_mb,
        compute_mb=compute_mb,
        total_mb=total_mb,
        reserve_mb=float(vram_reserve_mb),
        budget_mb=budget_mb,
        kv_basis=kv_basis,
        calibrated=calibration is not None,
        budget_basis=budget_basis,
    )


def estimate_ram(config: TrialConfig, model: ModelReport, ctx: int | None) -> RamEstimate:
    """Advisory host-RAM estimate for CPU weights and host-side KV cache."""
    mib = 1024 * 1024
    gpu_fraction = min(1.0, config.gpu_layers / max(1, model.ngl_all))
    expert_gpu_fraction = gpu_fraction * max(
        0.0, 1.0 - config.moe_cpu_layers / max(1, model.n_layer)
    )
    if model.expert_bytes is not None and model.dense_bytes is not None:
        weights_mb = model.dense_bytes / mib * (1.0 - gpu_fraction) + model.expert_bytes / mib * (
            1.0 - expert_gpu_fraction
        )
    else:
        gpu_weights_mb = est_vram_mb(
            gpu_layers=config.gpu_layers,
            moe_cpu_layers=config.moe_cpu_layers,
            model_size_bytes=model.size_bytes,
            ngl_all=model.ngl_all,
            n_layer=model.n_layer,
        )
        weights_mb = max(0.0, model.size_bytes / mib - gpu_weights_mb)
    kv_mb = 0.0
    if config.no_kv_offload:
        workload_tokens = ctx if ctx is not None else 640
        if model.kv_bytes_per_token_f16 is not None:
            scales = {"f16": 1.0, "bf16": 1.0, "q8_0": 0.5, "q4_0": 0.25}
            scale = (
                scales.get(config.cache_type_k.lower(), 1.0)
                + scales.get(config.cache_type_v.lower(), 1.0)
            ) / 2
            kv_mb = workload_tokens * model.kv_bytes_per_token_f16 * scale / mib
        else:
            kv_mb = model.n_layer * workload_tokens / 32
    return RamEstimate(weights_mb=weights_mb, kv_mb=kv_mb, total_mb=weights_mb + kv_mb)


def estimate_reason(
    *,
    config: TrialConfig,
    model: ModelReport,
    ctx: int | None,
    vram_reserve_mb: int,
    vram_total_mb: int | None,
) -> str | None:
    """Explain an advisory over-budget estimate, or return ``None``."""
    estimate = estimate_vram(
        config=config,
        model=model,
        ctx=ctx,
        vram_reserve_mb=vram_reserve_mb,
        vram_total_mb=vram_total_mb,
    )
    if (
        estimate.budget_mb is None
        or vram_total_mb is None
        or estimate.total_mb <= estimate.budget_mb
    ):
        return None
    return (
        f"estimated {estimate.total_mb / 1024:.1f} GiB exceeds budget "
        f"{estimate.budget_mb / 1024:.1f} GiB "
        f"({vram_total_mb / 1024:.1f} total - {vram_reserve_mb / 1024:.1f} reserve)"
    )


def memory_pressure(est_vram: float, vram_mb: int | None) -> float:
    """Ratio of estimated VRAM need to detected VRAM capacity; 0.0 if unknown."""
    if vram_mb is None or vram_mb <= 0:
        return 0.0
    return est_vram / vram_mb


def is_fully_offloaded(gpu_layers: int, ngl_all: int) -> bool:
    return gpu_layers >= ngl_all


#: A fully-offloaded run whose tg is within this factor of a pure-CPU
#: reference is treated as host-memory spill suspected (issue #36).
SPILL_TG_TOLERANCE = 1.3

#: A fully-offloaded run whose observed device-memory delta is below this
#: fraction of the estimated total need is treated as host-memory spill
#: suspected (issue #36).
SPILL_VRAM_DELTA_FRACTION = 0.5


def is_cpu_class_speed(tg_mean: float, cpu_reference_tg: float) -> bool:
    """Whether a measured tg is indistinguishable from the CPU reference."""
    return cpu_reference_tg > 0 and tg_mean <= cpu_reference_tg * SPILL_TG_TOLERANCE


def is_spill_vram_delta(observed_used_delta_mb: float, estimated_total_mb: float) -> bool:
    """Whether an observed device-memory delta is far below the estimate."""
    return estimated_total_mb > 0 and observed_used_delta_mb < (
        estimated_total_mb * SPILL_VRAM_DELTA_FRACTION
    )


def has_gpu_backend(llama: LlamaCppReport) -> bool | None:
    """Whether the baseline reported a non-CPU llama.cpp backend."""
    if llama.backends is None:
        return None
    cpu_backends = {"CPU", "BLAS", "RPC"}
    return any(token.strip().upper() not in cpu_backends for token in llama.backends.split(","))


def gpu_available(hardware: HardwareReport, llama: LlamaCppReport) -> bool:
    """Whether physical hardware and the llama.cpp build permit GPU use."""
    return gpu_present(hardware) and has_gpu_backend(llama) is not False


def gpu_layer_cap(model: ModelReport, options: TuneOptions) -> int:
    """Return the effective hard GPU-layer boundary cap."""
    requested = options.max_gpu_layers
    return min(model.ngl_all, requested if requested is not None else model.ngl_all)


def applicable_dimensions(
    *,
    hardware: HardwareReport,
    model: ModelReport,
    llama: LlamaCppReport,
    options: TuneOptions,
    incumbent: TrialConfig,
) -> frozenset[str]:
    """Which of the 10 search dimensions apply, given the current incumbent.

    Several "applies when" conditions in DESIGN §9.1 reference the current
    value of another dimension (e.g. moe_cpu_layers applies only when
    gpu_layers > 0), so applicability is evaluated relative to `incumbent`
    rather than being a static, hardware-only property.
    """
    applicable: set[str] = set()
    has_gpu = gpu_available(hardware, llama)
    caps = llama.capabilities

    if has_gpu and gpu_layer_cap(model, options) > 0:
        applicable.add("gpu_layers")

    if model.moe and "ncmoe" in caps and incumbent.gpu_layers > 0:
        applicable.add("moe_cpu_layers")

    if "fa" in caps:
        applicable.add("flash_attn")

    applicable.add("ubatch")
    applicable.add("batch")

    fully_offloaded = incumbent.gpu_layers >= model.ngl_all
    if (not fully_offloaded) or (not has_gpu) or incumbent.moe_cpu_layers > 0:
        applicable.add("threads")
        if "tb" in caps:
            applicable.add("threads_batch")

    if "mmp" in caps:
        applicable.add("mmap")

    if has_gpu and "nkvo" in caps:
        pressure = memory_pressure(
            est_vram_mb(
                gpu_layers=incumbent.gpu_layers,
                moe_cpu_layers=incumbent.moe_cpu_layers,
                model_size_bytes=model.size_bytes,
                ngl_all=model.ngl_all,
                n_layer=model.n_layer,
            ),
            total_vram_mb(hardware, multi_gpu=options.multi_gpu),
        )
        if pressure > 0.8 or incumbent.no_kv_offload:
            applicable.add("kv_offload")

    if options.allow_lossy and "ctk" in caps:
        applicable.add("cache_type_k")

    if options.allow_lossy and "ctv" in caps and incumbent.flash_attn:
        applicable.add("cache_type_v")

    return frozenset(applicable)


def candidates_for(
    dim: str,
    *,
    hardware: HardwareReport,
    model: ModelReport,
    llama: LlamaCppReport,
    options: TuneOptions,
    incumbent: TrialConfig,
) -> tuple[Any, ...]:
    """Raw candidate values for `dim`, or () when inapplicable.

    Candidates are only clamped/deduplicated within their own dimension;
    cross-dimension constraints (e.g. ubatch <= batch) are enforced by
    :func:`is_valid_config` once the caller has built a full mutated
    TrialConfig.
    """
    if dim not in DIMENSION_ORDER:
        msg = f"unknown search dimension: {dim!r}"
        raise ValueError(msg)

    applicable = applicable_dimensions(
        hardware=hardware, model=model, llama=llama, options=options, incumbent=incumbent
    )
    if dim not in applicable:
        return ()

    if dim == "gpu_layers":
        return gpu_layer_candidates(gpu_layer_cap(model, options))
    if dim == "moe_cpu_layers":
        return moe_cpu_layer_candidates(model.n_layer)
    if dim == "flash_attn":
        return FLASH_ATTN_CANDIDATES
    if dim == "ubatch":
        return UBATCH_CANDIDATES
    if dim == "batch":
        return BATCH_CANDIDATES
    if dim == "threads":
        return thread_candidates(
            hardware.physical_cores, hardware.logical_cores, hardware.perf_cores
        )
    if dim == "threads_batch":
        return thread_candidates(
            hardware.physical_cores, hardware.logical_cores, hardware.perf_cores
        )
    if dim == "mmap":
        return MMAP_CANDIDATES
    if dim == "kv_offload":
        return KV_OFFLOAD_CANDIDATES
    if dim == "cache_type_k":
        return CACHE_TYPE_CANDIDATES
    if dim == "cache_type_v":
        return CACHE_TYPE_CANDIDATES

    msg = f"unknown search dimension: {dim!r}"
    raise ValueError(msg)


def build_search_plan(
    *,
    hardware: HardwareReport,
    model: ModelReport,
    llama: LlamaCppReport,
    options: TuneOptions,
    incumbent: TrialConfig,
    calibration: VramCalibration | None = None,
) -> dict[str, Any]:
    """Build a pure, non-executing description of the prospective search."""
    reserve_mb = options.vram_reserve_mb if options.vram_reserve_mb is not None else 1536
    total_mb = total_vram_mb(hardware, multi_gpu=options.multi_gpu)
    free_mb = free_vram_mb(hardware, multi_gpu=options.multi_gpu)
    dimensions = {
        dim: list(
            candidates_for(
                dim,
                hardware=hardware,
                model=model,
                llama=llama,
                options=options,
                incumbent=incumbent,
            )
        )
        for dim in DIMENSION_ORDER
        if dim
        in applicable_dimensions(
            hardware=hardware,
            model=model,
            llama=llama,
            options=options,
            incumbent=incumbent,
        )
    }
    cap = gpu_layer_cap(model, options)
    ladder = list(moe_cpu_layer_candidates(model.n_layer)) if model.moe and cap > 0 else []
    if options.initial_cpu_moe is not None and model.moe and cap > 0:
        initial_moe = min(model.n_layer, options.initial_cpu_moe)
        ladder = [initial_moe, *(value for value in ladder if value != initial_moe)]
    configs: list[tuple[str, TrialConfig]] = [("defaults", incumbent)]
    if gpu_available(hardware, llama):
        full = dataclasses.replace(
            incumbent,
            gpu_layers=cap,
        )
        configs.append(("full_offload", full))
        configs.extend(
            (f"full_offload_ncmoe_{ncmoe}", dataclasses.replace(full, moe_cpu_layers=ncmoe))
            for ncmoe in ladder
            if ncmoe != full.moe_cpu_layers
        )
    estimates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for label, trial in configs:
        if trial.trial_id in seen:
            continue
        seen.add(trial.trial_id)
        estimate = estimate_vram(
            config=trial,
            model=model,
            ctx=options.ctx_size,
            vram_reserve_mb=reserve_mb,
            vram_total_mb=total_mb,
            vram_free_mb=free_mb,
            calibration=calibration,
        )
        estimates.append(
            {"label": label, "config": trial.to_dict(), "estimate": dataclasses.asdict(estimate)}
        )
    budget = estimates[0]["estimate"] if estimates else {}
    budgets: dict[str, Any] = {
        "vram_total_mb": total_mb,
        "vram_free_mb": free_mb,
        "reserve_mb": reserve_mb,
        "budget_mb": budget.get("budget_mb"),
        "budget_basis": budget.get("budget_basis"),
    }
    result: dict[str, Any] = {
        "workload": {
            "pp": options.pp,
            "tg": options.tg,
            **({"depth": options.depth} if options.depth is not None else {}),
        },
        "ctx_ladder": [
            *([options.ctx_size] if options.ctx_size is not None else []),
            *options.ctx_ladder,
        ],
        "dimensions": dimensions,
        "ncmoe_ladder": ladder,
        "budgets": budgets,
        "estimates": estimates,
    }
    if options.multi_gpu:
        budgets["capacity_basis"] = "pooled-known-devices"
        budgets["known_capacity_devices"] = sum(gpu.vram_mb is not None for gpu in hardware.gpus)
        budgets["known_free_devices"] = sum(gpu.vram_free_mb is not None for gpu in hardware.gpus)
        placements = multi_gpu_placement_candidates(hardware, llama, options)
        if placements:
            result["multi_gpu_placements"] = [
                {"tensor_split": list(split), "split_mode": mode} for split, mode in placements
            ]
    return result


def mutate(config: TrialConfig, dim: str, value: Any) -> TrialConfig:
    """Return a copy of `config` with search dimension `dim` set to `value`."""
    field = _DIMENSION_FIELD[dim]
    return dataclasses.replace(config, **{field: value})


def dimension_value(config: TrialConfig, dim: str) -> Any:
    """Return the config value controlled by search dimension ``dim``."""
    field = _DIMENSION_FIELD[dim]
    return getattr(config, field)


def is_valid_config(
    config: TrialConfig,
    *,
    hardware: HardwareReport,
    model: ModelReport,
    llama: LlamaCppReport | None = None,
) -> bool:
    """Enforce the cross-dimension constraints in DESIGN §9.1."""
    if not (0 <= config.gpu_layers <= model.ngl_all):
        return False
    if not (0 <= config.moe_cpu_layers <= model.n_layer):
        return False
    if not model.moe and config.moe_cpu_layers != 0:
        return False
    if config.ubatch > config.batch:
        return False
    if config.cache_type_v != "f16" and not config.flash_attn:
        return False
    if config.split_mode not in (None, "layer", "row"):
        return False
    if config.tensor_split is not None and (
        len(config.tensor_split) != len(hardware.gpus)
        or any(value < 0 for value in config.tensor_split)
        or not any(value > 0 for value in config.tensor_split)
    ):
        return False
    if not (1 <= config.threads <= hardware.logical_cores):
        return False
    gpu_usable = gpu_present(hardware) if llama is None else gpu_available(hardware, llama)
    return not (not gpu_usable and config.gpu_layers != 0)
