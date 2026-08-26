"""Unit tests for llamatune.types."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from llamatune.types import (
    BaselineResult,
    GPUInfo,
    HardwareReport,
    MetricStats,
    ModelReport,
    TrialConfig,
    TrialResult,
    TuneOptions,
    TuneOutcome,
)

_ALL_CAPS = frozenset({"fa", "mmp", "nkvo", "ctk", "ctv", "ncmoe", "ot", "r", "o"})


def _config(**overrides: object) -> TrialConfig:
    base: dict[str, object] = {
        "gpu_layers": 33,
        "moe_cpu_layers": 0,
        "flash_attn": True,
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


def test_trial_id_is_deterministic() -> None:
    a = _config()
    b = _config()
    assert a.trial_id == b.trial_id
    assert len(a.trial_id) == 16
    int(a.trial_id, 16)  # must be valid hex


def test_trial_id_matches_direct_canonical_json_computation() -> None:
    config = _config(threads_batch=4, tensor_split=(16.0, 24.0), split_mode="row")
    canonical = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    assert config.trial_id == expected


def test_trial_id_memoized_once_at_construction() -> None:
    config = _config()
    # PERF-008: the identifier is computed in __post_init__ into the private
    # slot; the property is a pure accessor over it.
    assert config._trial_id == config.trial_id
    assert config.trial_id == config.trial_id


def test_dataclasses_replace_recomputes_trial_id() -> None:
    base = _config(gpu_layers=33)
    mutated = dataclasses.replace(base, gpu_layers=1)
    assert mutated.trial_id != base.trial_id
    canonical = json.dumps(mutated.to_dict(), sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    assert mutated.trial_id == expected


def test_memoized_field_excluded_from_equality_hash_and_repr() -> None:
    a = _config()
    b = _config()
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
    assert "_trial_id" not in repr(a)
    # The public serialization contract is untouched by the cache.
    assert a.to_dict() == b.to_dict()
    assert "trial_id" not in a.to_dict()


def test_trial_id_changes_with_any_field() -> None:
    base = _config()
    mutated = _config(threads=4)
    assert base.trial_id != mutated.trial_id


def test_trial_config_to_dict_from_dict_roundtrip() -> None:
    config = _config(cache_type_v="q8_0", flash_attn=True)
    data = config.to_dict()
    restored = TrialConfig.from_dict(data)
    assert restored == config
    assert restored.trial_id == config.trial_id


def test_stage3_none_fields_preserve_trial_identity() -> None:
    config = _config()
    legacy_data = {
        "gpu_layers": 33,
        "moe_cpu_layers": 0,
        "flash_attn": True,
        "ubatch": 512,
        "batch": 2048,
        "threads": 8,
        "mmap": True,
        "no_kv_offload": False,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
    }
    assert config.to_dict() == legacy_data
    assert TrialConfig.from_dict(legacy_data).trial_id == config.trial_id


def test_multi_gpu_fields_roundtrip_and_emit_capability_gated_flags() -> None:
    legacy = _config()
    config = _config(tensor_split=(16.0, 24.0), split_mode="row")
    assert TrialConfig.from_dict(config.to_dict()) == config
    args = config.bench_args(frozenset({"ts", "sm"}))
    assert args[args.index("-ts") + 1] == "16,24"
    assert args[args.index("-sm") + 1] == "row"
    assert "tensor_split" not in legacy.to_dict()
    assert "split_mode" not in legacy.to_dict()
    assert TrialConfig.from_dict(legacy.to_dict()).trial_id == legacy.trial_id


def test_stage3_optional_flags_are_capability_gated() -> None:
    config = _config(threads_batch=12, ot_spec="^blk\\.(1|3)\\.ffn_.*_exps=CPU")
    args = config.bench_args(frozenset({"tb", "ot", "ncmoe"}))
    assert args[args.index("-tb") + 1] == "12"
    assert args[args.index("-ot") + 1] == "^blk\\.(1|3)\\.ffn_.*_exps=CPU"
    assert "-ncmoe" not in args
    assert TrialConfig.from_dict(config.to_dict()) == config


def test_trial_config_is_frozen() -> None:
    config = _config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.gpu_layers = 0  # type: ignore[misc]


def test_bench_args_emits_ungated_flags_even_without_capabilities() -> None:
    config = _config()
    args = config.bench_args(frozenset())
    assert args == ("-ngl", "33", "-ub", "512", "-b", "2048", "-t", "8")


def test_bench_args_emits_gated_flags_when_capability_present() -> None:
    config = _config(flash_attn=False, mmap=False, no_kv_offload=True, cache_type_v="q8_0")
    args = config.bench_args(_ALL_CAPS)
    assert "-fa" in args
    assert args[args.index("-fa") + 1] == "0"
    assert "-mmp" in args
    assert args[args.index("-mmp") + 1] == "0"
    assert "-nkvo" in args
    assert args[args.index("-nkvo") + 1] == "1"
    assert "-ctk" in args
    assert args[args.index("-ctk") + 1] == "f16"
    assert "-ctv" in args
    assert args[args.index("-ctv") + 1] == "q8_0"
    assert "-ncmoe" in args


def test_bench_args_omits_flag_when_capability_absent() -> None:
    config = _config()
    caps = frozenset({"mmp"})  # only mmap supported
    args = config.bench_args(caps)
    assert "-fa" not in args
    assert "-nkvo" not in args
    assert "-ctk" not in args
    assert "-ctv" not in args
    assert "-ncmoe" not in args
    assert "-mmp" in args


def test_bench_args_always_emits_applicable_dims_even_at_default_values() -> None:
    # flash_attn=False is llama-bench's own default, but since "fa" capability
    # is present the flag must still be emitted explicitly.
    config = _config(flash_attn=False)
    args = config.bench_args(frozenset({"fa"}))
    assert "-fa" in args
    assert args[args.index("-fa") + 1] == "0"


def test_gpu_info_and_hardware_report_construction() -> None:
    gpu = GPUInfo(vendor="nvidia", name="RTX 4090", vram_mb=24564, method="nvidia-smi")
    hw = HardwareReport(
        os_name="Linux",
        arch="x86_64",
        cpu_model="Fake CPU",
        physical_cores=8,
        logical_cores=16,
        perf_cores=None,
        ram_mb=32768,
        gpus=(gpu,),
        warnings=(),
    )
    assert hw.gpus[0].vram_mb == 24564


def test_model_report_construction() -> None:
    model = ModelReport(
        path=Path("/models/model.gguf"),
        size_bytes=1234,
        architecture="llama",
        n_layer=32,
        ngl_all=33,
        expert_count=0,
        moe=False,
        name="tiny",
        fingerprint="abc123",
        full_sha256=None,
    )
    assert model.ngl_all == model.n_layer + 1


def test_metric_stats_and_trial_result_and_baseline_result() -> None:
    stats = MetricStats(mean=100.0, stdev=1.0, cv=0.01, n=3)
    config = _config()
    result = TrialResult(
        trial_id=config.trial_id,
        config=config,
        status="ok",
        pp=stats,
        tg=stats,
        wall_s=1.23,
        exit_code=0,
        oom_pattern=None,
        artifact_dir=None,
        flags=(),
    )
    assert result.status == "ok"

    baseline = BaselineResult(
        runs=3, pp=stats, tg=stats, noise_floor_cv=0.01, fallback=None, resolved_defaults={}
    )
    assert baseline.runs == 3


def test_tune_options_and_outcome_construction() -> None:
    options = TuneOptions(
        target="balanced",
        budget_trials=60,
        budget_minutes=None,
        reps_search=3,
        reps_confirm=5,
        baseline_runs=3,
        pp=512,
        tg=128,
        allow_lossy=False,
        cooldown_s=0.0,
        thermal_wait_cap_s=0.0,
        baseline_only=False,
        llama_bin=None,
        sessions_dir=Path("./llamatune-sessions"),
        full_hash=False,
    )
    outcome = TuneOutcome(session_dir=Path("/var/lib/llamatune/session"), analysis={}, exit_code=0)
    assert options.target == "balanced"
    assert outcome.exit_code == 0
