"""Unit tests for llamatune.report (report.md rendering from analysis.json)."""

from __future__ import annotations

import json
from pathlib import PureWindowsPath
from typing import Any

import pytest

from llamatune import report

_HARDWARE: dict[str, Any] = {
    "os_name": "Linux",
    "arch": "x86_64",
    "cpu_model": "Fake CPU",
    "physical_cores": 8,
    "logical_cores": 16,
    "perf_cores": None,
    "ram_mb": 32768,
    "gpus": [{"vendor": "nvidia", "name": "Fake GPU", "vram_mb": 24000, "method": "nvidia-smi"}],
    "warnings": [],
}

_MODEL: dict[str, Any] = {
    "path": "/models/tiny.gguf",
    "size_bytes": 4_000_000_000,
    "architecture": "llama",
    "n_layer": 32,
    "ngl_all": 33,
    "expert_count": 0,
    "moe": False,
    "name": "tiny-model",
    "fingerprint": "abc123",
    "full_sha256": None,
}

_LLAMACPP: dict[str, Any] = {
    "bench_path": "/usr/bin/llama-bench",
    "cli_path": None,
    "server_path": None,
    "capabilities": ["fa", "mmp", "nkvo", "ctk", "ctv", "ncmoe", "r", "o"],
    "help_sha256": "deadbeef",
    "build_commit": "fake1234",
    "build_number": 9999,
}

_SESSION_META: dict[str, Any] = {
    "schema_version": 1,
    "tool_version": "0.1.0",
    "argv": ["llamatune", "tune", "tiny.gguf"],
    "created": "2026-07-14T00:00:00+00:00",
    "options": {
        "target": "balanced",
        "budget_trials": 60,
        "budget_minutes": None,
        "reps_search": 3,
        "reps_confirm": 5,
        "baseline_runs": 3,
        "pp": 512,
        "tg": 128,
        "allow_lossy": False,
        "cooldown_s": 0.0,
        "baseline_only": False,
    },
}


def _config_dict(**overrides: object) -> dict[str, Any]:
    base: dict[str, Any] = {
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
    return base


def _baseline() -> dict[str, Any]:
    return {
        "runs": 3,
        "pp": {"mean": 500.0, "stdev": 5.0, "cv": 0.01, "n": 3},
        "tg": {"mean": 35.0, "stdev": 0.5, "cv": 0.014, "n": 3},
        "noise_floor_cv": 0.01,
        "fallback": None,
        "resolved_defaults": {},
    }


def _analysis_with_winner(*, lossy: bool = False) -> dict[str, Any]:
    config = _config_dict(cache_type_v="q8_0") if lossy else _config_dict()
    winner = {
        "trial_id": "winner0000000001",
        "config": config,
        "pp": 650.0,
        "tg": 42.0,
        "score": 1.2,
        "improvement_pct": {"pp": 30.0, "tg": 20.0, "score": 20.0},
        "confirmed": True,
        "confirmation": {
            "runs": 5,
            "pp": {"mean": 650.0, "stdev": 3.0, "cv": 0.005, "n": 5},
            "tg": {"mean": 42.0, "stdev": 0.3, "cv": 0.007, "n": 5},
        },
    }
    top = [
        {
            "trial_id": "winner0000000001",
            "status": "ok",
            "config": config,
            "pp_mean": 650.0,
            "tg_mean": 42.0,
            "score": 1.2,
            "flags": [],
        }
    ]
    return {
        "schema_version": 1,
        "target": "balanced",
        "baseline": _baseline(),
        "counts": {
            "executed": 10,
            "ok": 8,
            "unstable": 0,
            "oom": 1,
            "timeout": 0,
            "crash": 0,
            "parse_error": 0,
            "pruned": 1,
        },
        "pareto": top,
        "top": top,
        "winner": winner,
        "lossless_winner": None if not lossy else {**winner, "trial_id": "lossless000000001"},
        "warnings": ["1-minute load average high"],
    }


def _analysis_no_winner() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "target": "balanced",
        "baseline": _baseline(),
        "counts": {
            "executed": 5,
            "ok": 5,
            "unstable": 0,
            "oom": 0,
            "timeout": 0,
            "crash": 0,
            "parse_error": 0,
            "pruned": 0,
        },
        "pareto": [],
        "top": [],
        "winner": None,
        "lossless_winner": None,
        "warnings": [],
    }


def test_render_includes_all_sections() -> None:
    text = report.render(_analysis_with_winner(), _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP)
    for heading in (
        "## Summary",
        "## Hardware",
        "## Model",
        "## llama.cpp build",
        "## Baseline",
        "## Method",
        "## Top trials",
        "## Pareto set",
        "## Failures and prunes",
        "## Warnings",
        "## Reproduce",
    ):
        assert heading in text


def test_render_winner_summary_mentions_trial_and_improvement() -> None:
    text = report.render(_analysis_with_winner(), _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP)
    assert "winner0000000001" in text
    assert "+30.00%" in text
    assert "Quality-affecting" not in text


def test_render_lossy_winner_is_labeled_quality_affecting() -> None:
    analysis = _analysis_with_winner(lossy=True)
    text = report.render(analysis, _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP)
    assert "Quality-affecting" in text
    assert "lossless000000001" in text


def test_render_no_winner_reports_defaults_optimal() -> None:
    text = report.render(_analysis_no_winner(), _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP)
    assert "No confirmed improvement" in text
    assert "No confirmed winner" in text  # reproduce section
    assert "_None recorded._" in text  # empty top/pareto tables


def test_render_reproduce_section_has_llama_bench_command() -> None:
    text = report.render(_analysis_with_winner(), _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP)
    assert "llama-bench" in text
    assert "-m /models/tiny.gguf" in text
    assert "-ngl 33" in text
    assert "-fa 1" in text


def test_render_hardware_section_contains_gpu_info() -> None:
    text = report.render(_analysis_with_winner(), _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP)
    assert "Fake GPU" in text
    assert "nvidia" in text
    assert "24000" in text


def test_render_hardware_driver_version_when_present() -> None:
    hardware = {**_HARDWARE, "gpus": [{**_HARDWARE["gpus"][0], "driver_version": "595.80"}]}
    text = report.render(_analysis_with_winner(), _SESSION_META, hardware, _MODEL, _LLAMACPP)
    assert "driver: 595.80" in text


def test_render_counts_table_has_all_outcomes() -> None:
    text = report.render(_analysis_with_winner(), _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP)
    for outcome in (
        "executed",
        "ok",
        "unstable",
        "oom",
        "cuda_error",
        "timeout",
        "crash",
        "parse_error",
        "pruned",
    ):
        assert f"| {outcome} |" in text
    assert "Counted benchmark measurements consumed:" not in text


def test_render_counts_table_distinguishes_budget_consumption() -> None:
    analysis = _analysis_with_winner()
    analysis["counts"]["budget_consumed"] = 17

    text = report.render(analysis, _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP)

    assert "Counted benchmark measurements consumed: 17." in text
    assert "| executed | 10 |" in text


def test_render_warnings_section_lists_warnings() -> None:
    text = report.render(_analysis_with_winner(), _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP)
    assert "1-minute load average high" in text


def test_render_no_warnings_shows_placeholder() -> None:
    text = report.render(_analysis_no_winner(), _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP)
    warnings_section = text.split("## Warnings")[1].split("## Reproduce")[0]
    assert "_None._" in warnings_section


def test_render_new_feasibility_evidence() -> None:
    analysis = _analysis_with_winner()
    analysis.update(
        {
            "default_probe": {
                "status": "gpu_resource",
                "classification": "gpu_resource",
                "pattern": "failed to load model",
                "evidence": "baseline/run-1",
            },
            "baseline_kind": "safe_fallback",
            "feasibility": {
                "workload": {"pp": 512, "tg": 128},
                "ctx_validated": 8192,
                "boundaries": [
                    {"moe_cpu_layers": 0, "max_ok_ngl": 23, "min_fail_ngl": 24, "probes": 7}
                ],
                "max_fitting": {"gpu_layers": 23, "moe_cpu_layers": 0},
                "best_measured": {
                    "gpu_layers": 23,
                    "moe_cpu_layers": 0,
                    "pp": 1114.0,
                    "tg": 42.95,
                    "score": 1.4,
                },
                "recommended": {
                    "gpu_layers": 22,
                    "moe_cpu_layers": 0,
                    "reason": "passed full-context validation",
                },
                "vram_reserve_mb": 1536,
                "reserve_provenance": "auto-default",
                "estimate": {
                    "weights_mb": 12000.0,
                    "kv_mb": 512.0,
                    "compute_mb": 256.0,
                    "total_mb": 12768.0,
                    "budget_mb": 14302.0,
                },
            },
            "context_validation": {"ctx": 8192, "status": "ok", "evidence": "probe/ctx"},
        }
    )
    analysis["baseline"]["kind"] = "safe_fallback"
    text = report.render(analysis, _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP)
    assert "failed to load model" in text
    assert "safe_fallback" in text
    assert "| 0 | 23 | 24 | 7 |" in text
    assert "Maximum fitting placement" in text
    assert "Safety-adjusted recommendation" in text
    assert "llama-bench -m /models/tiny.gguf -p 512 -n 128 -r 5" in text
    assert "llama-cli -m /models/tiny.gguf -ngl 22 -b 2048 -ub 512 -t 8" in text
    assert "-fa on" in text
    assert "-c 8192" in text


def test_render_legacy_analysis_marks_new_sections_not_evaluated() -> None:
    text = report.render(_analysis_no_winner(), _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP)
    assert text.count("Not evaluated") >= 3
    assert "no full-context validation was performed" in text


def test_render_warm_start_boundary_as_search_cap() -> None:
    analysis = _analysis_with_winner()
    analysis["feasibility"] = {
        "boundaries": [
            {
                "moe_cpu_layers": 12,
                "max_ok_ngl": 11,
                "min_fail_ngl": None,
                "probes": 3,
                "cap_ngl": 11,
                "cap_source": "warm_start",
            }
        ]
    }
    text = report.render(analysis, _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP)
    assert "| 12 | ≤ 11 (search cap) | - | 3 |" in text
    assert "not a measured maximum" in text


def test_render_legacy_boundary_remains_bare_maximum() -> None:
    analysis = _analysis_with_winner()
    analysis["feasibility"] = {
        "boundaries": [{"moe_cpu_layers": 0, "max_ok_ngl": 23, "min_fail_ngl": None, "probes": 4}]
    }
    text = report.render(analysis, _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP)
    assert "| 0 | 23 | - | 4 |" in text
    assert "search cap" not in text


def test_render_failed_zero_boundary_does_not_claim_a_fitting_placement() -> None:
    analysis = _analysis_with_winner()
    analysis["feasibility"] = {
        "boundaries": [{"moe_cpu_layers": 0, "max_ok_ngl": 0, "min_fail_ngl": 0, "probes": 2}]
    }

    text = report.render(analysis, _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP)

    assert "| 0 | none | 0 | 2 |" in text
    assert "| 0 | 0 | 0 | 2 |" not in text


def test_render_boundary_and_max_fitting_annotate_host_spill_suspected() -> None:
    analysis = _analysis_with_winner()
    analysis["feasibility"] = {
        "boundaries": [
            {
                "moe_cpu_layers": 0,
                "max_ok_ngl": 23,
                "min_fail_ngl": 24,
                "probes": 7,
                "spill_suspected": True,
            }
        ],
        "max_fitting": {
            "gpu_layers": 23,
            "moe_cpu_layers": 0,
            "spill_suspected": True,
        },
    }

    text = report.render(analysis, _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP)

    assert "| 0 | 23 (host-spill suspected) | 24 | 7 |" in text
    assert "host-spill suspected" in text.split("## Feasibility and placement")[1].split("## ")[0]
    assert "_Host-memory spill was suspected" in text


def _recommended_export(*, moe: bool = True) -> dict[str, Any]:
    config = _config_dict(moe_cpu_layers=18 if moe else 0)
    if moe:
        config["ot_spec"] = "^blk\\.(1|3)\\.ffn_.*_exps=CPU"
    return {
        "target": "balanced",
        "confirmed": True,
        "config": config,
        "model": {"path": "/models/model.gguf", "fingerprint": "fp"},
        "expected": {},
    }


def test_export_llama_server_and_cli_include_runtime_flags() -> None:
    meta = {"options": {"ctx_size": 8192}}
    recommended = _recommended_export()
    recommended["config"]["threads_batch"] = 12
    server = report.render_export(recommended, meta, "llama-server")
    cli = report.render_export(_recommended_export(), meta, "llama-cli")
    assert server.startswith("llama-server -m /models/model.gguf")
    assert cli.startswith("llama-cli -m /models/model.gguf")
    assert "-ot" in server and "--n-cpu-moe" not in server
    assert "-tb 12" in server
    assert "-c 8192" in server


def test_export_systemd_llama_swap_and_json() -> None:
    recommended = _recommended_export(moe=False)
    systemd = report.render_export(recommended, {"options": {}}, "systemd")
    swap = report.render_export(recommended, {"options": {}}, "llama-swap")
    exported_json = report.render_export(recommended, {"options": {}}, "json")
    assert "ExecStart=llama-server" in systemd
    assert "Restart=on-failure" in systemd
    assert "models:" in swap and "cmd: llama-server" in swap
    assert json.loads(exported_json) == recommended


def test_export_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="unknown export format"):
        report.render_export(_recommended_export(), {"options": {}}, "unknown")


def test_stage3_report_blocks_are_get_tolerant() -> None:
    analysis = _analysis_with_winner()
    analysis.update(
        {
            "coverage": {
                "threads": {
                    "candidates": [4, 8, 12],
                    "executed": [4, 8],
                    "pruned": [],
                    "skipped": [{"value": 12, "reason": "budget"}],
                }
            },
            "telemetry": {
                "samples": 8,
                "peak_temperature_c": 72.0,
                "peak_power_w": 320.0,
                "throttled": True,
                "thermal_pause_count": 3,
                "thermal_wait_s": 25.0,
            },
            "quality_gate": {"ppl_lossy": 7.5, "ppl_f16": 7.4, "delta_pct": 1.35},
            "feasibility": {"estimate": {"calibrated": True}},
        }
    )
    text = report.render(analysis, _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP)
    assert "## Search coverage" in text
    assert "12: budget" in text
    assert "throttled=True" in text
    assert "thermal pauses=3, thermal wait=25.0 s" in text
    assert "lossy PPL=7.5000" in text
    assert "calibrated from observed sessions" in text


def test_method_reports_multi_gpu_only_when_enabled() -> None:
    enabled = {
        **_SESSION_META,
        "options": {**_SESSION_META["options"], "multi_gpu": True},
    }
    enabled_text = report.render(_analysis_with_winner(), enabled, _HARDWARE, _MODEL, _LLAMACPP)
    default_text = report.render(
        _analysis_with_winner(), _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP
    )
    assert "Multi-GPU pooled capacity: enabled" in enabled_text
    assert "Multi-GPU pooled capacity" not in default_text


def test_method_reports_night_shift_allocation_next_to_budget() -> None:
    session_meta = {
        **_SESSION_META,
        "nightshift": {"budget_trials": 120, "budget_minutes": 263.5532032333333},
    }
    text = report.render(_analysis_with_winner(), session_meta, _HARDWARE, _MODEL, _LLAMACPP)
    default_text = report.render(
        _analysis_with_winner(), _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP
    )
    assert "Budget: 60 trials, unlimited minutes" in text
    assert "Night Shift allocated up to 120 trials and 263.55 minutes of the shift window" in text
    assert "Night Shift allocated" not in default_text


def test_method_reports_night_shift_allocation_without_minutes_verbatim() -> None:
    session_meta = {
        **_SESSION_META,
        "nightshift": {"budget_trials": 120, "budget_minutes": None},
    }
    text = report.render(_analysis_with_winner(), session_meta, _HARDWARE, _MODEL, _LLAMACPP)
    assert "Night Shift allocated up to 120 trials and - minutes of the shift window" in text


def test_report_real_quality_shape_computes_delta_and_formats_stage3_config() -> None:
    analysis = _analysis_with_winner()
    analysis["quality_gate"] = {"status": "ok", "ppl_lossy": 7.5, "ppl_f16": 7.4}
    analysis["winner"]["config"].update(
        {"threads_batch": 12, "ot_spec": "^blk\\.1\\.ffn_.*_exps=CPU"}
    )
    text = report.render(analysis, _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP)
    assert "delta=1.35%" in text
    assert "tb=12" in text
    assert "ot=^blk\\.1\\.ffn_.*_exps=CPU" in text


def test_render_search_plan_and_registry_results() -> None:
    plan = {
        "dimensions": {"threads": [4, 8]},
        "ncmoe_ladder": [0, 10],
        "budgets": {"budget_mb": 14000, "budget_basis": "observed-free"},
        "estimates": [{"label": "full", "estimate": {"total_mb": 12000, "calibrated": True}}],
    }
    text = report.render_search_plan(plan, _HARDWARE, _MODEL)
    assert "threads: [4, 8]" in text
    assert "observed-free" in text and "calibrated" in text
    assert "match: /sessions/a" in report.render_registry_lookup(
        {"status": "hit", "record": {"session_dir": "/sessions/a", "config": {}}}
    )
    assert "stale: build" in report.render_registry_lookup(
        {"status": "stale", "stale_reasons": ["build"]}
    )


def test_windows_model_path_is_rendered_without_posix_conversion() -> None:
    model = {**_MODEL, "path": str(PureWindowsPath("C:/Models/model.gguf"))}
    text = report.render(_analysis_with_winner(), _SESSION_META, _HARDWARE, model, _LLAMACPP)
    assert r"C:\Models\model.gguf" in text


def test_depth_profile_context_envelope_depth_method_and_binary_hash() -> None:
    analysis = _analysis_with_winner()
    analysis["depth_profile"] = {
        "trial_id": "winner0000000001",
        "rows": [
            {"d": 0, "pp": 650.0, "tg": 40.0},
            {"d": 32768, "pp": 500.0, "tg": 30.0},
        ],
    }
    analysis["context_envelope"] = [
        {
            "ctx": 32768,
            "status": "ok",
            "config": _config_dict(gpu_layers=33),
            "fallback_config": None,
            "evidence": "probes/context-32768",
        },
        {
            "ctx": 65536,
            "status": "ok",
            "config": _config_dict(gpu_layers=33),
            "fallback_config": _config_dict(gpu_layers=28, moe_cpu_layers=4),
            "evidence": "probes/context-65536",
        },
    ]
    session = {**_SESSION_META, "options": {**_SESSION_META["options"], "depth": 32768}}
    llama = {**_LLAMACPP, "bench_sha256": "a" * 64}
    text = report.render(analysis, session, _HARDWARE, _MODEL, llama)
    assert "## Depth profile" in text
    assert "| 32768 | 500.00 | 30.00 | -25.00% |" in text
    assert "## Context envelope" in text
    assert "ngl=28 ncmoe=4" in text
    assert "Workload: pp=512, tg=128, depth=32768" in text
    assert f"llama-bench SHA-256: `{'a' * 64}`" in text


def test_new_report_sections_are_absence_tolerant_for_old_schema() -> None:
    text = report.render(_analysis_with_winner(), _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP)
    assert "## Depth profile" not in text
    assert "## Context envelope" not in text
    assert "llama-bench SHA-256" not in text
    assert "Workload: pp=512, tg=128\n" in text


def test_old_schema_render_is_identical_with_additive_none_defaults() -> None:
    old = report.render(_analysis_with_winner(), _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP)
    additive_session = {
        **_SESSION_META,
        "options": {**_SESSION_META["options"], "depth": None, "depth_profile": None},
    }
    additive_analysis = {**_analysis_with_winner()}
    assert report.render(additive_analysis, additive_session, _HARDWARE, _MODEL, _LLAMACPP) == old


def test_partial_depth_and_sparse_envelope_rows_render_conservatively() -> None:
    analysis = _analysis_with_winner()
    analysis["depth_profile"] = {
        "trial_id": "winner0000000001",
        "rows": ["corrupt", {"d": 8192, "pp": None, "tg": 31.0}],
    }
    analysis["context_envelope"] = [
        "corrupt",
        {
            "ctx": 98304,
            "status": "skipped",
            "config": None,
            "fallback_config": None,
            "evidence": None,
        },
    ]
    text = report.render(analysis, _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP)
    assert "| 8192 | - | 31.00 | - |" in text
    assert "| 98304 | skipped | `-` | `-` |" in text


def test_empty_depth_profile_and_all_envelope_statuses_render() -> None:
    analysis = _analysis_with_winner()
    analysis["depth_profile"] = {"trial_id": "winner0000000001", "rows": []}
    analysis["context_envelope"] = [
        {
            "ctx": 32768,
            "status": "failed",
            "config": _config_dict(gpu_layers=33),
            "fallback_config": _config_dict(gpu_layers=24, moe_cpu_layers=8),
            "evidence": "probes/fallback",
        },
        {
            "ctx": 65536,
            "status": "pruned",
            "config": None,
            "fallback_config": None,
            "evidence": None,
        },
    ]
    text = report.render(analysis, _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP)
    assert "## Depth profile\n\n| depth | pp | tg | tg vs depth 0 |" in text
    assert "| 32768 | failed | `ngl=24 ncmoe=8` | `probes/fallback` |" in text
    assert "| 65536 | pruned | `-` | `-` |" in text


def test_context_envelope_labels_lossy_cache_fallback() -> None:
    analysis = _analysis_with_winner()
    fallback = _config_dict(gpu_layers=24, moe_cpu_layers=8)
    fallback["cache_type_k"] = "q8_0"
    analysis["context_envelope"] = [
        {
            "ctx": 65536,
            "status": "failed",
            "config": _config_dict(gpu_layers=33),
            "fallback_config": fallback,
            "evidence": "probes/fallback",
        }
    ]

    text = report.render(analysis, _SESSION_META, _HARDWARE, _MODEL, _LLAMACPP)

    assert "ngl=24 ncmoe=8 ctk=q8_0 ctv=f16" in text


def test_present_null_binary_hash_is_explicitly_unknown() -> None:
    text = report.render(
        _analysis_with_winner(),
        _SESSION_META,
        _HARDWARE,
        _MODEL,
        {**_LLAMACPP, "bench_sha256": None},
    )
    assert "llama-bench SHA-256: `(unknown)`" in text


def test_context_section_reports_work_scaled_timeout_and_retries() -> None:
    """Issue #38: the context-validation section names the timeout policy."""
    analysis = {
        "context_validation": {"ctx": 65536, "status": "ok", "evidence": "probes/x"},
        "probe_timeout": {
            "scale": 2.5,
            "floor_s": None,
            "trial_timeout_s": 122.7,
            "max_s": 3600.0,
            "retries": 1,
        },
    }
    text = report._context_section(analysis)
    assert "work-scaled at 2.5x" in text
    assert "122.7s" in text
    assert "one work-scaled retry ran" in text

    no_timeout = report._context_section({"context_validation": {"ctx": 1, "status": "ok"}})
    assert "Probe timeout" not in no_timeout
