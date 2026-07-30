"""Unit tests for llamatune.config (search-space, constraints, feasibility)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from llamatune import config
from llamatune.types import (
    GPUInfo,
    HardwareReport,
    LlamaCppReport,
    ModelReport,
    TrialConfig,
    TuneOptions,
    VramCalibration,
)

_ALL_CAPS = frozenset({"fa", "mmp", "nkvo", "ctk", "ctv", "ncmoe", "ot", "r", "o"})


def _hardware(*, gpus: tuple[GPUInfo, ...] = (), logical_cores: int = 16) -> HardwareReport:
    return HardwareReport(
        os_name="Linux",
        arch="x86_64",
        cpu_model="Fake CPU",
        physical_cores=8,
        logical_cores=logical_cores,
        perf_cores=None,
        ram_mb=32768,
        gpus=gpus,
        warnings=(),
    )


def _model(*, moe: bool = False, n_layer: int = 32, size_bytes: int = 4_000_000_000) -> ModelReport:
    return ModelReport(
        path=Path("/models/model.gguf"),
        size_bytes=size_bytes,
        architecture="llama",
        n_layer=n_layer,
        ngl_all=n_layer + 1,
        expert_count=8 if moe else 0,
        moe=moe,
        name="tiny",
        fingerprint="abc",
        full_sha256=None,
    )


def _llama(
    capabilities: frozenset[str] = _ALL_CAPS, *, backends: str | None = None
) -> LlamaCppReport:
    return LlamaCppReport(
        bench_path=Path("/usr/bin/llama-bench"),
        cli_path=None,
        server_path=None,
        capabilities=capabilities,
        help_sha256="deadbeef",
        build_commit=None,
        build_number=None,
        backends=backends,
    )


def _options(allow_lossy: bool = False) -> TuneOptions:
    return TuneOptions(
        target="balanced",
        budget_trials=60,
        budget_minutes=None,
        reps_search=3,
        reps_confirm=5,
        baseline_runs=3,
        pp=512,
        tg=128,
        allow_lossy=allow_lossy,
        cooldown_s=0.0,
        thermal_wait_cap_s=0.0,
        baseline_only=False,
        llama_bin=None,
        sessions_dir=Path("./llamatune-sessions"),
        full_hash=False,
    )


def _default_config(**overrides: object) -> TrialConfig:
    base: dict[str, object] = {
        "gpu_layers": 33,
        "moe_cpu_layers": 0,
        "flash_attn": False,
        "ubatch": 512,
        "batch": 2048,
        "threads": 8,
        "mmap": True,
        "no_kv_offload": False,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
    }
    base.update(overrides)
    return TrialConfig(**base)  # type: ignore[arg-type]


# -- candidate generators -----------------------------------------------------


def test_gpu_layer_candidates_fractions_of_ngl_all() -> None:
    # 33 * {0, .25, .5, .75, 1} = 0, 8.25, 16.5, 24.75, 33; round() uses
    # banker's rounding so 16.5 -> 16 (nearest even).
    assert config.gpu_layer_candidates(33) == (0, 8, 16, 25, 33)


def test_gpu_layer_candidates_zero_ngl_all() -> None:
    assert config.gpu_layer_candidates(0) == (0,)


def test_moe_cpu_layer_candidates_fractions_of_n_layer() -> None:
    assert config.moe_cpu_layer_candidates(32) == (0, 8, 16, 24, 32)


def test_thread_candidates_dedup_and_clamped() -> None:
    candidates = config.thread_candidates(physical_cores=8, logical_cores=16, perf_cores=None)
    assert candidates == (4, 6, 8, 12, 16)


def test_thread_candidates_include_perf_cores() -> None:
    candidates = config.thread_candidates(physical_cores=8, logical_cores=16, perf_cores=6)
    assert 6 in candidates


def test_multi_gpu_placement_grid_is_bounded_and_gated() -> None:
    gpus = (
        GPUInfo(vendor="nvidia", name="a", vram_mb=16000, method="test"),
        GPUInfo(vendor="nvidia", name="b", vram_mb=24000, method="test"),
    )
    hardware = _hardware(gpus=gpus)
    llama = _llama(_ALL_CAPS | {"ts", "sm"})
    enabled = dataclasses.replace(_options(), multi_gpu=True)
    assert config.multi_gpu_placement_candidates(hardware, llama, enabled) == (
        ((16000.0, 24000.0), "layer"),
        ((16000.0, 24000.0), "row"),
        ((1.0, 1.0), "layer"),
        ((1.0, 1.0), "row"),
        ((1.0, 0.0), "layer"),
        ((1.0, 0.0), "row"),
        ((0.0, 1.0), "layer"),
        ((0.0, 1.0), "row"),
    )
    assert config.multi_gpu_placement_candidates(hardware, llama, _options()) == ()
    assert (
        config.multi_gpu_placement_candidates(hardware, _llama(_ALL_CAPS | {"ts"}), enabled) == ()
    )
    assert config.multi_gpu_placement_candidates(_hardware(gpus=gpus[:1]), llama, enabled) == ()


def test_multi_gpu_capacity_check_is_per_device_and_advisory() -> None:
    gpus = (
        GPUInfo(vendor="nvidia", name="a", vram_mb=10000, method="test"),
        GPUInfo(vendor="nvidia", name="b", vram_mb=20000, method="test"),
    )
    trial = _default_config(tensor_split=(1.0, 2.0), split_mode="layer")
    feasible, estimates = config.multi_gpu_capacity_check(
        trial, _hardware(gpus=gpus), 24000.0, 1000
    )
    assert feasible is True
    assert estimates == (8000.0, 16000.0)
    feasible, _ = config.multi_gpu_capacity_check(
        dataclasses.replace(trial, tensor_split=(2.0, 1.0)),
        _hardware(gpus=gpus),
        24000.0,
        1000,
    )
    assert feasible is False
    unknown = dataclasses.replace(gpus[1], vram_mb=None)
    assert (
        config.multi_gpu_capacity_check(trial, _hardware(gpus=(gpus[0], unknown)), 24000.0, 1000)[0]
        is None
    )
    assert config.multi_gpu_capacity_check(
        dataclasses.replace(trial, tensor_split=None), _hardware(gpus=gpus), 24000.0, 1000
    ) == (None, ())
    assert config.multi_gpu_capacity_check(
        dataclasses.replace(trial, tensor_split=(0.0, 0.0)),
        _hardware(gpus=gpus),
        24000.0,
        1000,
    ) == (False, ())


def test_multi_gpu_zero_capacity_grid_deduplicates_even_and_proportional() -> None:
    gpus = (
        GPUInfo(vendor="nvidia", name="a", vram_mb=None, method="test"),
        GPUInfo(vendor="nvidia", name="b", vram_mb=None, method="test"),
    )
    placements = config.multi_gpu_placement_candidates(
        _hardware(gpus=gpus),
        _llama(_ALL_CAPS | {"ts", "sm"}),
        dataclasses.replace(_options(), multi_gpu=True),
    )
    assert len(placements) == 6
    assert placements.count(((1.0, 1.0), "layer")) == 1


def test_ubatch_and_batch_candidates_are_fixed() -> None:
    assert config.UBATCH_CANDIDATES == (128, 256, 512, 1024, 2048, 4096)
    assert config.BATCH_CANDIDATES == (512, 1024, 2048, 4096)


# -- feasibility ---------------------------------------------------------------


def test_est_vram_mb_matches_fake_bench_formula() -> None:
    # Mirrors the fake llama-bench OOM model in DESIGN §15.1.
    est = config.est_vram_mb(
        gpu_layers=33,
        moe_cpu_layers=20,
        model_size_bytes=4000 * 1024 * 1024,
        ngl_all=33,
        n_layer=32,
    )
    assert est == pytest.approx(2500.0, rel=1e-6)


def test_estimate_vram_models_moe_context_cache_and_reserve() -> None:
    model = dataclasses.replace(
        _model(moe=True, n_layer=32),
        expert_bytes=3 * 1024 * 1024 * 1024,
        dense_bytes=1 * 1024 * 1024 * 1024,
    )
    base = _default_config(gpu_layers=33)
    estimate = config.estimate_vram(
        config=base, model=model, ctx=512, vram_reserve_mb=1536, vram_total_mb=16384
    )
    more_cpu_moe = config.estimate_vram(
        config=dataclasses.replace(base, moe_cpu_layers=16),
        model=model,
        ctx=512,
        vram_reserve_mb=1536,
        vram_total_mb=16384,
    )
    long_context = config.estimate_vram(
        config=base, model=model, ctx=8192, vram_reserve_mb=1536, vram_total_mb=16384
    )
    q8 = config.estimate_vram(
        config=dataclasses.replace(base, cache_type_k="q8_0", cache_type_v="q8_0"),
        model=model,
        ctx=8192,
        vram_reserve_mb=1536,
        vram_total_mb=16384,
    )
    host_kv = config.estimate_vram(
        config=dataclasses.replace(base, no_kv_offload=True),
        model=model,
        ctx=8192,
        vram_reserve_mb=1536,
        vram_total_mb=16384,
    )
    assert more_cpu_moe.weights_mb < estimate.weights_mb
    assert long_context.kv_mb > estimate.kv_mb
    assert q8.kv_mb < long_context.kv_mb
    assert host_kv.kv_mb == 0
    assert estimate.budget_mb == 16384 - 1536
    assert estimate.kv_basis == "heuristic"


def test_estimate_vram_treats_negative_one_as_full_offload() -> None:
    model = dataclasses.replace(
        _model(),
        expert_bytes=0,
        dense_bytes=4 * 1024 * 1024 * 1024,
    )
    sentinel = config.estimate_vram(
        config=_default_config(gpu_layers=-1),
        model=model,
        ctx=8192,
        vram_reserve_mb=1536,
        vram_total_mb=16384,
    )
    explicit = config.estimate_vram(
        config=_default_config(gpu_layers=model.ngl_all),
        model=model,
        ctx=8192,
        vram_reserve_mb=1536,
        vram_total_mb=16384,
    )

    assert sentinel == explicit
    assert sentinel.weights_mb == 4096


def test_estimate_vram_uses_metadata_kv_geometry() -> None:
    model = dataclasses.replace(
        _model(moe=True, n_layer=40),
        ngl_all=40,
        kv_bytes_per_token_f16=40 * 8 * (128 + 128) * 2,
        n_kv_layers=40,
    )
    trial = _default_config(gpu_layers=40)
    estimate = config.estimate_vram(
        config=trial,
        model=model,
        ctx=8192,
        vram_reserve_mb=1536,
        vram_total_mb=16384,
    )
    assert estimate.kv_mb == pytest.approx(1280.0, abs=1.0)
    assert estimate.kv_basis == "metadata"


def test_estimate_vram_uses_observed_free_budget() -> None:
    estimate = config.estimate_vram(
        config=_default_config(),
        model=_model(),
        ctx=8192,
        vram_reserve_mb=1536,
        vram_total_mb=16303,
        vram_free_mb=13561,
    )
    assert estimate.budget_mb == 13305
    assert estimate.budget_basis == "observed-free"


def test_estimate_vram_budget_basis_fallbacks() -> None:
    free_only = config.estimate_vram(
        config=_default_config(),
        model=_model(),
        ctx=None,
        vram_reserve_mb=1536,
        vram_total_mb=None,
        vram_free_mb=4096,
    )
    assert free_only.budget_mb == 3840
    assert free_only.budget_basis == "observed-free"
    total_limited = config.estimate_vram(
        config=_default_config(),
        model=_model(),
        ctx=None,
        vram_reserve_mb=1536,
        vram_total_mb=8192,
        vram_free_mb=8000,
    )
    assert total_limited.budget_mb == 6656
    assert total_limited.budget_basis == "total-reserve"


def test_estimate_ram_tracks_cpu_weights_and_host_kv() -> None:
    huge = _model(size_bytes=200 * 1024**3)
    all_cpu = config.estimate_ram(_default_config(gpu_layers=0), huge, 8192)
    assert all_cpu.total_mb == pytest.approx(200 * 1024)
    validated_like = dataclasses.replace(
        _model(moe=True, n_layer=40, size_bytes=22 * 1024**3),
        ngl_all=41,
        dense_bytes=4 * 1024**3,
        expert_bytes=18 * 1024**3,
    )
    mixed = config.estimate_ram(
        _default_config(gpu_layers=23, moe_cpu_layers=18), validated_like, 8192
    )
    assert mixed.total_mb < 0.9 * 126 * 1024


def test_estimate_ram_scales_quantized_host_kv_and_heuristic_expert_spill() -> None:
    metadata_model = dataclasses.replace(_model(), kv_bytes_per_token_f16=4096)
    f16 = config.estimate_ram(_default_config(no_kv_offload=True), metadata_model, 8192)
    q4 = config.estimate_ram(
        _default_config(no_kv_offload=True, cache_type_k="q4_0", cache_type_v="q4_0"),
        metadata_model,
        8192,
    )
    assert q4.kv_mb == pytest.approx(f16.kv_mb * 0.25)
    heuristic_moe = _model(moe=True, size_bytes=20 * 1024**3)
    spilled = config.estimate_ram(
        _default_config(gpu_layers=heuristic_moe.ngl_all, moe_cpu_layers=16),
        heuristic_moe,
        None,
    )
    assert spilled.weights_mb > 0


def test_estimate_vram_calibration_scales_each_term() -> None:
    calibration = VramCalibration(
        schema_version=1,
        fitted_at="2026-07-17T00:00:00+00:00",
        samples=3,
        weights_scale=2.0,
        kv_scale=3.0,
        compute_scale=4.0,
    )
    base = config.estimate_vram(
        config=_default_config(),
        model=_model(),
        ctx=8192,
        vram_reserve_mb=1536,
        vram_total_mb=16384,
    )
    fitted = config.estimate_vram(
        config=_default_config(),
        model=_model(),
        ctx=8192,
        vram_reserve_mb=1536,
        vram_total_mb=16384,
        calibration=calibration,
    )
    assert fitted.weights_mb == pytest.approx(base.weights_mb * 2)
    assert fitted.kv_mb == pytest.approx(base.kv_mb * 3)
    assert fitted.compute_mb == pytest.approx(base.compute_mb * 4)
    assert fitted.calibrated is True


def test_estimate_vram_unknown_capacity_never_explains_infeasibility() -> None:
    model = _model()
    trial = _default_config()
    estimate = config.estimate_vram(
        config=trial, model=model, ctx=None, vram_reserve_mb=1536, vram_total_mb=None
    )
    assert estimate.budget_mb is None
    assert (
        config.estimate_reason(
            config=trial,
            model=model,
            ctx=None,
            vram_reserve_mb=1536,
            vram_total_mb=None,
        )
        is None
    )


def test_estimate_reason_explains_over_budget() -> None:
    reason = config.estimate_reason(
        config=_default_config(),
        model=_model(size_bytes=20 * 1024 * 1024 * 1024),
        ctx=8192,
        vram_reserve_mb=1536,
        vram_total_mb=16384,
    )
    assert reason is not None
    assert "exceeds budget" in reason


def test_memory_pressure_ratio() -> None:
    assert config.memory_pressure(2500.0, 2500) == pytest.approx(1.0)
    assert config.memory_pressure(2500.0, None) == 0.0
    assert config.memory_pressure(2500.0, 0) == 0.0


def test_total_vram_mb_uses_largest_gpu() -> None:
    gpus = (
        GPUInfo(vendor="nvidia", name="a", vram_mb=8000, method="nvidia-smi"),
        GPUInfo(vendor="nvidia", name="b", vram_mb=24000, method="nvidia-smi"),
    )
    assert config.total_vram_mb(_hardware(gpus=gpus)) == 24000


def test_multi_gpu_capacity_sums_known_total_and_free_values() -> None:
    gpus = (
        GPUInfo(vendor="nvidia", name="a", vram_mb=16000, vram_free_mb=12000, method="test"),
        GPUInfo(vendor="nvidia", name="b", vram_mb=24000, vram_free_mb=18000, method="test"),
        GPUInfo(vendor="nvidia", name="unknown", vram_mb=None, method="test"),
    )
    hardware = _hardware(gpus=gpus)
    assert config.total_vram_mb(hardware) == 24000
    assert config.free_vram_mb(hardware) == 18000
    assert config.total_vram_mb(hardware, multi_gpu=True) == 40000
    assert config.free_vram_mb(hardware, multi_gpu=True) == 30000
    single = _hardware(gpus=(gpus[0],))
    assert config.total_vram_mb(single, multi_gpu=True) == config.total_vram_mb(single)
    assert config.free_vram_mb(single, multi_gpu=True) == config.free_vram_mb(single)


def test_total_vram_mb_none_when_no_gpu_vram_known() -> None:
    gpus = (GPUInfo(vendor="amd", name="a", vram_mb=None, method="absent"),)
    assert config.total_vram_mb(_hardware(gpus=gpus)) is None


def test_hardware_signature_is_stable_and_plan_is_pure() -> None:
    gpu = GPUInfo(
        vendor="nvidia",
        name="x",
        vram_mb=16303,
        method="nvidia-smi",
        vram_free_mb=13561,
    )
    hardware = _hardware(gpus=(gpu,))
    assert config.hardware_signature(hardware)[-1] == (("nvidia", "x", 16303),)
    plan = config.build_search_plan(
        hardware=hardware,
        model=_model(moe=True),
        llama=_llama(capabilities=_ALL_CAPS | {"tb"}, backends="CUDA"),
        options=_options(),
        incumbent=_default_config(gpu_layers=20),
    )
    assert plan["budgets"]["budget_mb"] == 13305
    assert plan["budgets"]["budget_basis"] == "observed-free"
    assert plan["dimensions"]["threads_batch"] == [4, 6, 8, 12, 16]
    assert plan["ncmoe_ladder"] == [0, 8, 16, 24, 32]
    assert plan["estimates"]

    pooled = config.build_search_plan(
        hardware=_hardware(gpus=(gpu, dataclasses.replace(gpu, name="y"))),
        model=_model(moe=True),
        llama=_llama(capabilities=_ALL_CAPS | {"tb"}, backends="CUDA"),
        options=dataclasses.replace(_options(), multi_gpu=True),
        incumbent=_default_config(gpu_layers=20),
    )
    assert pooled["budgets"]["vram_total_mb"] == 32606
    assert pooled["budgets"]["vram_free_mb"] == 27122
    assert pooled["budgets"]["capacity_basis"] == "pooled-known-devices"
    assert pooled["budgets"]["known_capacity_devices"] == 2


# -- applicability ---------------------------------------------------------------


def test_applicable_dimensions_no_gpu_excludes_gpu_dims() -> None:
    applicable = config.applicable_dimensions(
        hardware=_hardware(gpus=()),
        model=_model(),
        llama=_llama(),
        options=_options(),
        incumbent=_default_config(gpu_layers=0),
    )
    assert "gpu_layers" not in applicable
    assert "moe_cpu_layers" not in applicable
    assert "kv_offload" not in applicable
    assert "threads" in applicable  # CPU-only always needs threads


@pytest.mark.parametrize(
    ("backends", "gpu_expected"),
    [("CPU", False), ("CPU,BLAS", False), ("CUDA", True), (None, True)],
)
def test_applicable_dimensions_respect_reported_backends(
    backends: str | None, gpu_expected: bool
) -> None:
    hardware = _hardware(gpus=(GPUInfo(vendor="nvidia", name="x", vram_mb=100, method="test"),))
    applicable = config.applicable_dimensions(
        hardware=hardware,
        model=_model(size_bytes=4_000_000_000),
        llama=_llama(backends=backends),
        options=_options(),
        incumbent=_default_config(),
    )
    assert ("gpu_layers" in applicable) is gpu_expected
    assert ("kv_offload" in applicable) is gpu_expected
    assert ("threads" in applicable) is not gpu_expected


def test_applicable_dimensions_moe_requires_gpu_layers_positive() -> None:
    gpus = (GPUInfo(vendor="nvidia", name="x", vram_mb=24000, method="nvidia-smi"),)
    hardware = _hardware(gpus=gpus)
    model = _model(moe=True)

    not_offloaded = config.applicable_dimensions(
        hardware=hardware,
        model=model,
        llama=_llama(),
        options=_options(),
        incumbent=_default_config(gpu_layers=0),
    )
    assert "moe_cpu_layers" not in not_offloaded

    offloaded = config.applicable_dimensions(
        hardware=hardware,
        model=model,
        llama=_llama(),
        options=_options(),
        incumbent=_default_config(gpu_layers=33),
    )
    assert "moe_cpu_layers" in offloaded


def test_applicable_dimensions_lossy_requires_allow_lossy_and_flash_attn() -> None:
    gpus = (GPUInfo(vendor="nvidia", name="x", vram_mb=24000, method="nvidia-smi"),)
    hardware = _hardware(gpus=gpus)
    model = _model()

    without_flag = config.applicable_dimensions(
        hardware=hardware,
        model=model,
        llama=_llama(),
        options=_options(allow_lossy=False),
        incumbent=_default_config(flash_attn=True),
    )
    assert "cache_type_k" not in without_flag
    assert "cache_type_v" not in without_flag

    with_flag_no_fa = config.applicable_dimensions(
        hardware=hardware,
        model=model,
        llama=_llama(),
        options=_options(allow_lossy=True),
        incumbent=_default_config(flash_attn=False),
    )
    assert "cache_type_k" in with_flag_no_fa
    assert "cache_type_v" not in with_flag_no_fa  # requires flash_attn=1

    with_flag_and_fa = config.applicable_dimensions(
        hardware=hardware,
        model=model,
        llama=_llama(),
        options=_options(allow_lossy=True),
        incumbent=_default_config(flash_attn=True),
    )
    assert "cache_type_v" in with_flag_and_fa


def test_applicable_dimensions_kv_offload_requires_pressure_or_disabled_incumbent() -> None:
    gpus = (GPUInfo(vendor="nvidia", name="x", vram_mb=2500, method="nvidia-smi"),)
    hardware = _hardware(gpus=gpus)
    # model_size 4000 MiB fully offloaded -> pressure 4000/2500 = 1.6 > 0.8
    model = _model(size_bytes=4000 * 1024 * 1024)

    high_pressure = config.applicable_dimensions(
        hardware=hardware,
        model=model,
        llama=_llama(),
        options=_options(),
        incumbent=_default_config(gpu_layers=33),
    )
    assert "kv_offload" in high_pressure

    low_pressure = config.applicable_dimensions(
        hardware=hardware,
        model=model,
        llama=_llama(),
        options=_options(),
        incumbent=_default_config(gpu_layers=1),
    )
    assert "kv_offload" not in low_pressure

    disabled_at_low_pressure = config.applicable_dimensions(
        hardware=hardware,
        model=model,
        llama=_llama(),
        options=_options(),
        incumbent=_default_config(gpu_layers=1, no_kv_offload=True),
    )
    assert "kv_offload" in disabled_at_low_pressure


def test_candidates_for_returns_empty_when_inapplicable() -> None:
    result = config.candidates_for(
        "gpu_layers",
        hardware=_hardware(gpus=()),
        model=_model(),
        llama=_llama(),
        options=_options(),
        incumbent=_default_config(gpu_layers=0),
    )
    assert result == ()


def test_candidates_for_gpu_layers() -> None:
    gpus = (GPUInfo(vendor="nvidia", name="x", vram_mb=24000, method="nvidia-smi"),)
    result = config.candidates_for(
        "gpu_layers",
        hardware=_hardware(gpus=gpus),
        model=_model(),
        llama=_llama(),
        options=_options(),
        incumbent=_default_config(),
    )
    assert result == config.gpu_layer_candidates(33)


def test_candidates_for_unknown_dimension_raises() -> None:
    with pytest.raises(ValueError, match="unknown search dimension"):
        config.candidates_for(
            "bogus",
            hardware=_hardware(),
            model=_model(),
            llama=_llama(),
            options=_options(),
            incumbent=_default_config(),
        )


# -- mutate / validity ---------------------------------------------------------------


def test_mutate_gpu_layers() -> None:
    base = _default_config()
    mutated = config.mutate(base, "gpu_layers", 16)
    assert mutated.gpu_layers == 16
    assert base.gpu_layers == 33  # base unaffected (frozen dataclass)


def test_mutate_kv_offload_maps_to_no_kv_offload_field() -> None:
    base = _default_config(no_kv_offload=False)
    mutated = config.mutate(base, "kv_offload", True)
    assert mutated.no_kv_offload is True


def test_dimension_value_maps_kv_offload_to_no_kv_offload_field() -> None:
    config_value = _default_config(no_kv_offload=True)
    assert config.dimension_value(config_value, "kv_offload") is True
    assert config.dimension_value(config_value, "gpu_layers") == config_value.gpu_layers


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gpu_layers", -1),
        ("gpu_layers", 34),
        ("moe_cpu_layers", 33),
        ("threads", 0),
        ("threads", 17),
    ],
)
def test_is_valid_config_rejects_out_of_range_values(field: str, value: object) -> None:
    hardware = _hardware(gpus=(GPUInfo(vendor="nvidia", name="x", vram_mb=100, method="m"),))
    model = _model()
    cfg = config.mutate(_default_config(), field, value)
    assert config.is_valid_config(cfg, hardware=hardware, model=model) is False


def test_is_valid_config_rejects_ubatch_greater_than_batch() -> None:
    hardware = _hardware(gpus=(GPUInfo(vendor="nvidia", name="x", vram_mb=100, method="m"),))
    cfg = _default_config(ubatch=4096, batch=512)
    assert config.is_valid_config(cfg, hardware=hardware, model=_model()) is False


def test_is_valid_config_rejects_moe_offload_for_dense_model() -> None:
    hardware = _hardware(gpus=(GPUInfo(vendor="nvidia", name="x", vram_mb=100, method="m"),))
    cfg = _default_config(moe_cpu_layers=1)
    assert config.is_valid_config(cfg, hardware=hardware, model=_model(moe=False)) is False


def test_is_valid_config_rejects_lossy_ctv_without_flash_attn() -> None:
    hardware = _hardware(gpus=(GPUInfo(vendor="nvidia", name="x", vram_mb=100, method="m"),))
    cfg = _default_config(flash_attn=False, cache_type_v="q8_0")
    assert config.is_valid_config(cfg, hardware=hardware, model=_model()) is False


def test_is_valid_config_accepts_lossy_ctv_with_flash_attn() -> None:
    hardware = _hardware(gpus=(GPUInfo(vendor="nvidia", name="x", vram_mb=100, method="m"),))
    cfg = _default_config(flash_attn=True, cache_type_v="q8_0")
    assert config.is_valid_config(cfg, hardware=hardware, model=_model()) is True


def test_is_valid_config_no_gpu_forces_zero_gpu_layers() -> None:
    hardware = _hardware(gpus=())
    valid = _default_config(gpu_layers=0)
    invalid = _default_config(gpu_layers=10)
    model = _model()
    assert config.is_valid_config(valid, hardware=hardware, model=model) is True
    assert config.is_valid_config(invalid, hardware=hardware, model=model) is False


def test_is_valid_config_cpu_backend_forces_zero_gpu_layers() -> None:
    hardware = _hardware(gpus=(GPUInfo(vendor="nvidia", name="x", vram_mb=24000, method="test"),))
    assert config.is_valid_config(
        _default_config(gpu_layers=0),
        hardware=hardware,
        model=_model(),
        llama=_llama(backends="CPU"),
    )
    assert not config.is_valid_config(
        _default_config(gpu_layers=10),
        hardware=hardware,
        model=_model(),
        llama=_llama(backends="CPU"),
    )


def test_dimension_order_matches_design_spec() -> None:
    assert config.DIMENSION_ORDER == (
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
