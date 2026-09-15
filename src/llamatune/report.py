"""Render report.md from analysis.json content (DESIGN §11.2).

`render` is a pure function over already-loaded JSON dicts (the same shapes
written to session.json / hardware.json / model.json / llamacpp.json /
analysis.json) -- it performs no file or process I/O itself.
"""

from __future__ import annotations

import dataclasses
import json
import shlex
from typing import Any

from llamatune.config import DIMENSION_ORDER
from llamatune.runtimeflags import runtime_flags
from llamatune.types import TrialConfig

_LOSSY_CACHE_TYPES = {"cache_type_k", "cache_type_v"}


def _fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:+.{digits}f}%"


def _format_config(config: dict[str, Any]) -> str:
    text = (
        f"ngl={config.get('gpu_layers')} ncmoe={config.get('moe_cpu_layers')} "
        f"fa={int(bool(config.get('flash_attn')))} ub={config.get('ubatch')} "
        f"b={config.get('batch')} t={config.get('threads')} "
        f"mmap={int(bool(config.get('mmap')))} "
        f"nkvo={int(bool(config.get('no_kv_offload')))} "
        f"ctk={config.get('cache_type_k')} ctv={config.get('cache_type_v')}"
    )
    if config.get("threads_batch") is not None:
        text += f" tb={config['threads_batch']}"
    if config.get("ot_spec") is not None:
        text += f" ot={config['ot_spec']}"
    return text


def _runtime_flags(config: TrialConfig) -> list[str]:
    return runtime_flags(config)


def render_export(
    recommended: dict[str, Any], session_meta: dict[str, Any], export_format: str
) -> str:
    """Render one ready-to-use recommendation export format."""
    if export_format == "json":
        return json.dumps(recommended, indent=2, sort_keys=True) + "\n"
    config = TrialConfig.from_dict(recommended["config"])
    model_path = str((recommended.get("model") or {}).get("path", "MODEL.gguf"))
    flags = _runtime_flags(config)
    ctx = (session_meta.get("options") or {}).get("ctx_size")
    if ctx is not None:
        flags += ["-c", str(ctx)]
    server_argv = ["llama-server", "-m", model_path, *flags]
    cli_argv = ["llama-cli", "-m", model_path, *flags]
    if export_format == "llama-server":
        return shlex.join(server_argv) + "\n"
    if export_format == "llama-cli":
        return shlex.join(cli_argv) + "\n"
    if export_format == "systemd":
        return "\n".join(
            [
                "[Unit]",
                "Description=llama.cpp server (llamatune recommendation)",
                "After=network.target",
                "",
                "[Service]",
                f"ExecStart={shlex.join(server_argv)}",
                "Restart=on-failure",
                "",
                "[Install]",
                "WantedBy=default.target",
                "",
            ]
        )
    if export_format == "llama-swap":
        args = json.dumps(server_argv[1:])
        return "\n".join(
            [
                "models:",
                "  llamatune-recommended:",
                "    cmd: llama-server",
                f"    args: {args}",
                "",
            ]
        )
    raise ValueError(f"unknown export format: {export_format}")


def render_search_plan(
    plan: dict[str, Any], hardware: dict[str, Any], model: dict[str, Any]
) -> str:
    """Render the pure dry-run search plan returned by config.build_search_plan."""
    lines = [
        "llamatune dry-run search plan",
        f"model: {model.get('name') or model.get('path', '-')}",
        f"hardware: {hardware.get('cpu_model', '-')} / {len(hardware.get('gpus') or [])} GPU(s)",
        "dimensions:",
    ]
    dimensions = plan.get("dimensions") or {}
    lines.extend(f"  {name}: {values}" for name, values in dimensions.items())
    lines.append(f"ncmoe ladder: {plan.get('ncmoe_ladder') or []}")
    budgets = plan.get("budgets") or {}
    lines.append(
        f"VRAM budget: {_fmt(budgets.get('budget_mb'))} MiB "
        f"({budgets.get('budget_basis') or 'unavailable'})"
    )
    lines.append("estimates:")
    for item in plan.get("estimates") or []:
        estimate = item.get("estimate") or {}
        marker = " calibrated" if estimate.get("calibrated") else ""
        lines.append(f"  {item.get('label', '-')}: {_fmt(estimate.get('total_mb'))} MiB{marker}")
    return "\n".join(lines) + "\n"


def _is_lossy(config: dict[str, Any]) -> bool:
    return config.get("cache_type_k") != "f16" or config.get("cache_type_v") != "f16"


def _trials_table(trials: list[dict[str, Any]]) -> str:
    if not trials:
        return "_None recorded._\n"
    lines = [
        "| trial_id | status | pp (t/s) | tg (t/s) | score | config |",
        "|---|---|---|---|---|---|",
    ]
    for trial in trials:
        lines.append(
            "| {trial_id} | {status} | {pp} | {tg} | {score} | `{config}` |".format(
                trial_id=trial.get("trial_id", "-"),
                status=trial.get("status", "-"),
                pp=_fmt(trial.get("pp_mean")),
                tg=_fmt(trial.get("tg_mean")),
                score=_fmt(trial.get("score"), 4),
                config=_format_config(trial.get("config", {})),
            )
        )
    return "\n".join(lines) + "\n"


def _hardware_section(hardware: dict[str, Any]) -> str:
    lines = [
        f"- OS: {hardware.get('os_name', '-')} ({hardware.get('arch', '-')})",
        f"- CPU: {hardware.get('cpu_model', '-')}",
        f"- Physical cores: {hardware.get('physical_cores', '-')}, "
        f"logical cores: {hardware.get('logical_cores', '-')}"
        + (
            f", perf cores: {hardware['perf_cores']}"
            if hardware.get("perf_cores") is not None
            else ""
        ),
        f"- RAM: {hardware.get('ram_mb', '-')} MiB",
    ]
    gpus = hardware.get("gpus") or []
    if gpus:
        for gpu in gpus:
            lines.append(
                f"- GPU: {gpu.get('name', '-')} ({gpu.get('vendor', '-')}), "
                f"VRAM: {gpu.get('vram_mb', '-')} MiB, method: {gpu.get('method', '-')}"
                + (
                    f", driver: {gpu['driver_version']}"
                    if gpu.get("driver_version") is not None
                    else ""
                )
            )
            if gpu.get("vram_free_mb") is not None:
                lines.append(
                    f"  - Free VRAM observed: {gpu['vram_free_mb']} MiB, "
                    f"method: {gpu.get('vram_free_method') or '-'}, "
                    f"at: {gpu.get('vram_free_at') or '-'}"
                )
    else:
        lines.append("- GPU: none detected")
    if hardware.get("warnings"):
        lines.append("- Warnings:")
        lines.extend(f"  - {w}" for w in hardware["warnings"])
    return "\n".join(lines) + "\n"


def _model_section(model: dict[str, Any]) -> str:
    lines = [
        f"- Path: `{model.get('path', '-')}`",
        f"- Name: {model.get('name') or '(none)'}",
        f"- Architecture: {model.get('architecture', '-')}",
        f"- Layers: {model.get('n_layer', '-')} (ngl_all={model.get('ngl_all', '-')})",
        f"- MoE: {model.get('moe', False)} (expert_count={model.get('expert_count', 0)})",
        f"- Size: {model.get('size_bytes', 0):,} bytes",
        f"- Fingerprint: `{model.get('fingerprint', '-')}`",
    ]
    if model.get("full_sha256"):
        lines.append(f"- Full SHA-256: `{model['full_sha256']}`")
    return "\n".join(lines) + "\n"


def _llamacpp_section(llamacpp: dict[str, Any]) -> str:
    build_number = llamacpp.get("build_number")
    lines = [
        f"- llama-bench: `{llamacpp.get('bench_path', '-')}`",
        f"- llama-cli: `{llamacpp.get('cli_path') or '(not found)'}`",
        f"- llama-server: `{llamacpp.get('server_path') or '(not found)'}`",
        f"- Capabilities: {', '.join(sorted(llamacpp.get('capabilities') or [])) or '(none)'}",
        f"- Build commit: {llamacpp.get('build_commit') or '(unknown)'}",
        f"- Build number: {build_number if build_number is not None else '(unknown)'}",
    ]
    if "bench_sha256" in llamacpp:
        lines.append(f"- llama-bench SHA-256: `{llamacpp.get('bench_sha256') or '(unknown)'}`")
    return "\n".join(lines) + "\n"


def _baseline_section(baseline: dict[str, Any]) -> str:
    pp = baseline.get("pp", {})
    tg = baseline.get("tg", {})
    lines = [
        f"- Runs: {baseline.get('runs', '-')}",
        f"- pp: mean={_fmt(pp.get('mean'))} t/s, stdev={_fmt(pp.get('stdev'))}, "
        f"cv={_fmt(pp.get('cv'), 4)}, n={pp.get('n', '-')}",
        f"- tg: mean={_fmt(tg.get('mean'))} t/s, stdev={_fmt(tg.get('stdev'))}, "
        f"cv={_fmt(tg.get('cv'), 4)}, n={tg.get('n', '-')}",
        f"- Noise floor (cv): {_fmt(baseline.get('noise_floor_cv'), 4)}",
        f"- Fallback: {baseline.get('fallback') or '(none)'}",
        f"- Provenance: {baseline.get('kind', 'defaults')}",
    ]
    resolved = baseline.get("resolved_defaults") or {}
    if resolved:
        lines.append(f"- Resolved defaults: `{resolved}`")
    return "\n".join(lines) + "\n"


def _method_section(session_meta: dict[str, Any], baseline: dict[str, Any]) -> str:
    options = session_meta.get("options", {})
    lines = [
        f"- Dimensions searched (order): {' -> '.join(DIMENSION_ORDER)}",
        f"- Target: {options.get('target', '-')}",
        f"- Budget: {options.get('budget_trials', '-')} trials, "
        f"{options.get('budget_minutes') or 'unlimited'} minutes",
        f"- Repetitions: search={options.get('reps_search', '-')}, "
        f"confirm={options.get('reps_confirm', '-')}",
        f"- Baseline runs: {options.get('baseline_runs', '-')}",
        f"- Workload: pp={options.get('pp', '-')}, tg={options.get('tg', '-')}"
        + (f", depth={options['depth']}" if options.get("depth") is not None else ""),
        f"- Lossy dimensions allowed: {options.get('allow_lossy', False)}",
        f"- Cooldown: {options.get('cooldown_s', 0)} s",
        *(["- Multi-GPU pooled capacity: enabled"] if options.get("multi_gpu") else []),
        f"- Noise floor (cv): {_fmt(baseline.get('noise_floor_cv'), 4)}",
    ]
    return "\n".join(lines) + "\n"


def _depth_profile_section(analysis: dict[str, Any]) -> str | None:
    profile = analysis.get("depth_profile")
    if not isinstance(profile, dict):
        return None
    rows = profile.get("rows") or []
    depth_zero = next(
        (row.get("tg") for row in rows if isinstance(row, dict) and row.get("d") == 0), None
    )
    lines = ["| depth | pp | tg | tg vs depth 0 |", "|---:|---:|---:|---:|"]
    for row in rows:
        if not isinstance(row, dict):
            continue
        tg = row.get("tg")
        delta = "-"
        if isinstance(tg, (int, float)) and isinstance(depth_zero, (int, float)) and depth_zero:
            delta = f"{(tg / depth_zero - 1) * 100:+.2f}%"
        lines.append(f"| {row.get('d', '-')} | {_fmt(row.get('pp'))} | {_fmt(tg)} | {delta} |")
    return "\n".join(lines) + "\n"


def _context_envelope_section(analysis: dict[str, Any]) -> str | None:
    envelope = analysis.get("context_envelope")
    if not isinstance(envelope, list):
        return None
    lines = ["| ctx | status | placement | evidence |", "|---:|---|---|---|"]
    for row in envelope:
        if not isinstance(row, dict):
            continue
        config = row.get("fallback_config") or row.get("config")
        placement = "-"
        if isinstance(config, dict):
            placement = (
                f"ngl={config.get('gpu_layers', '-')} ncmoe={config.get('moe_cpu_layers', '-')}"
            )
            cache_k = config.get("cache_type_k")
            cache_v = config.get("cache_type_v")
            if cache_k not in (None, "f16") or cache_v not in (None, "f16"):
                placement += f" ctk={cache_k or '-'} ctv={cache_v or '-'}"
        lines.append(
            f"| {row.get('ctx', '-')} | {row.get('status', '-')} | "
            f"`{placement}` | `{row.get('evidence') or '-'}` |"
        )
    return "\n".join(lines) + "\n"


def _counts_section(counts: dict[str, Any]) -> str:
    order = [
        "executed",
        "ok",
        "unstable",
        "oom",
        "cuda_error",
        "gpu_resource",
        "timeout",
        "crash",
        "parse_error",
        "pruned",
    ]
    lines = ["| outcome | count |", "|---|---|"]
    lines.extend(f"| {key} | {counts.get(key, 0)} |" for key in order)
    if "budget_consumed" in counts:
        lines = [
            f"Counted benchmark measurements consumed: {counts['budget_consumed']}.",
            "",
            *lines,
        ]
    return "\n".join(lines) + "\n"


def _warnings_section(warnings: list[str]) -> str:
    if not warnings:
        return "_None._\n"
    return "\n".join(f"- {w}" for w in warnings) + "\n"


def _summary_section(analysis: dict[str, Any]) -> str:
    winner = analysis.get("winner")
    lossless_winner = analysis.get("lossless_winner")
    target = analysis.get("target", "balanced")

    if winner is None:
        lines = [
            f"No confirmed improvement over baseline was found for target `{target}`; "
            "llama.cpp defaults are optimal within the measured noise floor.",
        ]
        if lossless_winner is not None:
            lines.append(
                "A lossless candidate outperformed the baseline on paper but did not "
                "confirm beyond the noise floor; see the Top trials table below."
            )
        return "\n".join(lines) + "\n"

    improvement = winner.get("improvement_pct", {})
    lines = [
        f"Winner `{winner.get('trial_id', '-')}` for target `{target}`: "
        f"pp {_fmt(winner.get('pp'))} t/s ({_fmt_pct(improvement.get('pp'))}), "
        f"tg {_fmt(winner.get('tg'))} t/s ({_fmt_pct(improvement.get('tg'))}), "
        f"score {_fmt_pct(improvement.get('score'))} vs baseline.",
        f"Config: `{_format_config(winner.get('config', {}))}`",
    ]
    if _is_lossy(winner.get("config", {})):
        lines.append(
            "**Quality-affecting** -- this winner uses a lossy KV-cache quantization "
            "setting; validate output quality before adoption. "
            "The best all-lossless configuration is reported separately below."
        )
    if lossless_winner is not None and lossless_winner.get("trial_id") != winner.get("trial_id"):
        lines.append(
            f"Best all-lossless configuration: `{lossless_winner.get('trial_id', '-')}` "
            f"(`{_format_config(lossless_winner.get('config', {}))}`)"
        )
    return "\n".join(lines) + "\n"


def _reproduce_section(
    analysis: dict[str, Any],
    session_meta: dict[str, Any],
    model: dict[str, Any],
    llamacpp: dict[str, Any],
) -> str:
    winner = analysis.get("winner") or analysis.get("lossless_winner")
    if winner is None:
        return (
            "No confirmed winner; llama.cpp defaults (as measured in the baseline) "
            "are already optimal within noise.\n"
        )

    options = session_meta.get("options", {})
    capabilities = frozenset(llamacpp.get("capabilities") or [])
    try:
        config = TrialConfig.from_dict(winner["config"])
    except (KeyError, TypeError, ValueError):
        return "_Could not reconstruct the winning configuration._\n"

    argv = [
        str(llamacpp.get("bench_path", "llama-bench")),
        "-m",
        str(model.get("path", "MODEL.gguf")),
        "-p",
        str(options.get("pp", 512)),
        "-n",
        str(options.get("tg", 128)),
        "-r",
        str(options.get("reps_confirm", 5)),
        "-o",
        "json",
        *config.bench_args(capabilities),
    ]
    return "```sh\n" + shlex.join(argv) + "\n```\n"


def _default_probe_section(analysis: dict[str, Any]) -> str:
    probe = analysis.get("default_probe")
    if not isinstance(probe, dict):
        return "_Not evaluated._\n"
    return (
        "\n".join(
            [
                f"- Status: {probe.get('status', '-')}",
                f"- Classification: {probe.get('classification', '-')}",
                f"- Pattern: {probe.get('pattern') or '(none)'}",
                f"- Evidence: `{probe.get('evidence', '-')}`",
            ]
        )
        + "\n"
    )


def _pair(label: str, value: Any) -> str:
    if not isinstance(value, dict):
        return f"- {label}: not evaluated"
    placement = f"ngl={value.get('gpu_layers', '-')} ncmoe={value.get('moe_cpu_layers', '-')}"
    metrics = ""
    if value.get("score") is not None:
        metrics = (
            f", pp={_fmt(value.get('pp'))}, tg={_fmt(value.get('tg'))}, "
            f"score={_fmt(value.get('score'), 4)}"
        )
    reason = f" — {value['reason']}" if value.get("reason") else ""
    return f"- {label}: `{placement}`{metrics}{reason}"


def _feasibility_section(analysis: dict[str, Any]) -> str:
    feasibility = analysis.get("feasibility")
    if not isinstance(feasibility, dict):
        return "_Not evaluated._\n"
    workload = feasibility.get("workload") or {}
    workload_line = f"- Workload: pp={workload.get('pp', '-')} tg={workload.get('tg', '-')}"
    if workload.get("depth") is not None:
        workload_line += f" depth={workload['depth']}"
    lines = [
        workload_line,
        f"- Context validated: {feasibility.get('ctx_validated') or 'not evaluated'}",
        f"- VRAM reserve: {feasibility.get('vram_reserve_mb', '-')} MiB "
        f"({feasibility.get('reserve_provenance', '-')})",
    ]
    boundaries = feasibility.get("boundaries") or []
    if boundaries:
        lines += [
            "",
            "| ncmoe | max fitting ngl | min failing ngl | probes |",
            "|---:|---:|---:|---:|",
        ]
        for boundary in boundaries:
            min_fail = boundary.get("min_fail_ngl")
            max_ok = boundary.get("max_ok_ngl", "-")
            no_fit = max_ok == 0 and min_fail == 0
            warm_start_capped = min_fail is None and boundary.get("cap_source") == "warm_start"
            max_display = (
                "none"
                if no_fit
                else f"≤ {max_ok} (search cap)"
                if warm_start_capped
                else str(max_ok)
            )
            lines.append(
                f"| {boundary.get('moe_cpu_layers', '-')} | {max_display} | "
                f"{min_fail if min_fail is not None else '-'} | "
                f"{boundary.get('probes', '-')} |"
            )
        if any(
            boundary.get("min_fail_ngl") is None and boundary.get("cap_source") == "warm_start"
            for boundary in boundaries
        ):
            lines += [
                "",
                "_A search cap is inherited from another boundary; it is not a measured maximum._",
            ]
    else:
        lines.append("- Boundaries: not evaluated")
    lines += [
        _pair("Maximum fitting placement", feasibility.get("max_fitting")),
        _pair("Best measured placement", feasibility.get("best_measured")),
        _pair("Safety-adjusted recommendation", feasibility.get("recommended")),
    ]
    estimate = feasibility.get("estimate")
    if isinstance(estimate, dict):
        lines.append(
            "- Estimated memory pressure: "
            f"weights={_fmt(estimate.get('weights_mb'))} MiB, "
            f"KV={_fmt(estimate.get('kv_mb'))} MiB, "
            f"compute={_fmt(estimate.get('compute_mb'))} MiB, "
            f"total={_fmt(estimate.get('total_mb'))} MiB, "
            f"budget={_fmt(estimate.get('budget_mb'))} MiB"
        )
        if estimate.get("kv_basis") == "heuristic":
            lines.append("- Warning: KV estimate used fallback heuristic")
        if estimate.get("calibrated"):
            lines.append("- Estimate calibration: calibrated from observed sessions")
    else:
        lines.append("- Estimated memory pressure: not evaluated")
    return "\n".join(lines) + "\n"


def _context_section(analysis: dict[str, Any]) -> str:
    validation = analysis.get("context_validation")
    if not isinstance(validation, dict):
        return "**Warning:** no full-context validation was performed.\n"
    return (
        "\n".join(
            [
                f"- Context: {validation.get('ctx', '-')}",
                f"- Status: {validation.get('status', '-')}",
                f"- Evidence: `{validation.get('evidence', '-')}`",
            ]
        )
        + "\n"
    )


def _runtime_validation_section(analysis: dict[str, Any]) -> str:
    lines: list[str] = []
    cli_validation = analysis.get("cli_validation")
    if isinstance(cli_validation, dict):
        lines.append(
            f"- llama-cli: {cli_validation.get('status', '-')} "
            f"(`{cli_validation.get('evidence', '-')}`)"
        )
    else:
        lines.append("- llama-cli: not evaluated")
    observed = analysis.get("estimate_vs_observed")
    if isinstance(observed, dict):
        lines.append(
            "- Estimate vs observed: "
            f"estimated={_fmt(observed.get('estimated_total_mb'))} MiB, "
            f"observed delta={_fmt(observed.get('observed_used_delta_mb'))} MiB, "
            f"observed peak={_fmt(observed.get('observed_peak_used_mb'))} MiB "
            f"({observed.get('samples', 0)} samples)"
        )
    else:
        lines.append("- Estimate vs observed: not evaluated")
    quality = analysis.get("quality_gate")
    if isinstance(quality, dict):
        delta_pct = quality.get("delta_pct")
        if (
            delta_pct is None
            and isinstance(quality.get("ppl_lossy"), (int, float))
            and isinstance(quality.get("ppl_f16"), (int, float))
            and quality["ppl_f16"] != 0
        ):
            delta_pct = (quality["ppl_lossy"] / quality["ppl_f16"] - 1) * 100
        lines.append(
            "- KV quality gate: "
            f"lossy PPL={_fmt(quality.get('ppl_lossy'), 4)}, "
            f"f16 PPL={_fmt(quality.get('ppl_f16'), 4)}, "
            f"delta={_fmt(delta_pct, 2)}%"
        )
    telemetry = analysis.get("telemetry")
    if isinstance(telemetry, dict):
        telemetry_line = (
            "- Telemetry: "
            f"samples={telemetry.get('samples', 0)}, "
            f"peak temperature={_fmt(telemetry.get('peak_temperature_c'))} C, "
            f"peak power={_fmt(telemetry.get('peak_power_w'))} W, "
            f"throttled={bool(telemetry.get('throttled', False))}"
        )
        if telemetry.get("thermal_pause_count"):
            telemetry_line += (
                f", thermal pauses={telemetry['thermal_pause_count']}, "
                f"thermal wait={_fmt(telemetry.get('thermal_wait_s'), 1)} s"
            )
        lines.append(telemetry_line)
    return "\n".join(lines) + "\n"


def _coverage_section(analysis: dict[str, Any]) -> str:
    coverage = analysis.get("coverage")
    if not isinstance(coverage, dict):
        return "_Not evaluated._\n"
    lines = [
        "| dimension | candidates | executed | pruned | skipped |",
        "|---|---|---|---|---|",
    ]
    for dimension, details in coverage.items():
        if not isinstance(details, dict):
            continue
        candidates = details.get("candidates") or []
        executed = details.get("executed") or []
        pruned = details.get("pruned") or []
        skipped = details.get("skipped") or []
        skipped_text = ", ".join(
            f"{item.get('value', '-')}: {item.get('reason', '-')}"
            if isinstance(item, dict)
            else str(item)
            for item in skipped
        )
        lines.append(
            f"| {dimension} | {candidates} | {executed} | {pruned} | {skipped_text or '-'} |"
        )
    return "\n".join(lines) + "\n"


def render_registry_lookup(result: dict[str, Any]) -> str:
    """Render a registry lookup hit, miss, or stale result."""
    status = str(result.get("status", "miss"))
    record = result.get("record")
    if status == "hit" and isinstance(record, dict):
        return f"match: {record.get('session_dir', '-')}\nconfig: {record.get('config', {})}\n"
    reasons = result.get("stale_reasons") or []
    suffix = f": {', '.join(map(str, reasons))}" if reasons else ""
    return f"{status}{suffix}\n"


def render_revalidation(result: dict[str, Any]) -> str:
    """Render a revalidation outcome returned by the engine."""
    if result.get("previous_score") is not None and result.get("current_score") is not None:
        comparison = (
            f"previous={_fmt(result.get('previous_score'), 4)} "
            f"current={_fmt(result.get('current_score'), 4)} "
            f"noise_floor={_fmt(result.get('noise_floor_cv'), 4)}"
        )
    else:
        comparison = str(result.get("comparison", {}))
    return f"revalidation: {result.get('status', 'unknown')}\ncomparison: {comparison}\n"


def _recommended_runtime_section(
    analysis: dict[str, Any],
    session_meta: dict[str, Any],
    model: dict[str, Any],
    llamacpp: dict[str, Any],
) -> str:
    feasibility = analysis.get("feasibility")
    recommended = feasibility.get("recommended") if isinstance(feasibility, dict) else None
    if not isinstance(recommended, dict):
        return "_Not evaluated._\n"
    argv = [
        str(llamacpp.get("cli_path") or "llama-cli"),
        "-m",
        str(model.get("path", "MODEL.gguf")),
    ]
    measured = analysis.get("winner") or analysis.get("lossless_winner")
    config_data = measured.get("config") if isinstance(measured, dict) else None
    try:
        config = TrialConfig.from_dict(config_data) if isinstance(config_data, dict) else None
    except (KeyError, TypeError, ValueError):
        config = None
    if config is not None:
        config = dataclasses.replace(
            config,
            gpu_layers=int(recommended.get("gpu_layers", config.gpu_layers)),
            moe_cpu_layers=int(recommended.get("moe_cpu_layers", config.moe_cpu_layers)),
        )
        argv += _runtime_flags(config)
    else:
        argv += ["-ngl", str(recommended.get("gpu_layers", 0))]
        if recommended.get("moe_cpu_layers") is not None:
            argv += ["--n-cpu-moe", str(recommended["moe_cpu_layers"])]
    validation = analysis.get("context_validation")
    ctx = validation.get("ctx") if isinstance(validation, dict) else None
    if ctx is None:
        ctx = (session_meta.get("options") or {}).get("ctx_size")
    if ctx is not None:
        argv += ["-c", str(ctx)]
    return "```sh\n" + shlex.join(argv) + "\n```\n"


def render(
    analysis: dict[str, Any],
    session_meta: dict[str, Any],
    hardware: dict[str, Any],
    model: dict[str, Any],
    llamacpp: dict[str, Any],
) -> str:
    """Render report.md content from analysis.json + session metadata."""
    baseline = analysis.get("baseline", {})
    depth_profile = _depth_profile_section(analysis)
    context_envelope = _context_envelope_section(analysis)
    sections = [
        "# llamatune report\n",
        "## Summary\n",
        _summary_section(analysis),
        "## Hardware\n",
        _hardware_section(hardware),
        "## Model\n",
        _model_section(model),
        "## llama.cpp build\n",
        _llamacpp_section(llamacpp),
        "## Baseline\n",
        _baseline_section(baseline),
        "## Default probe\n",
        _default_probe_section(analysis),
        "## Feasibility and placement\n",
        _feasibility_section(analysis),
        "## Context validation\n",
        _context_section(analysis),
        *(["## Context envelope\n", context_envelope] if context_envelope is not None else []),
        "## Runtime validation\n",
        _runtime_validation_section(analysis),
        "## Method\n",
        _method_section(session_meta, baseline),
        *(["## Depth profile\n", depth_profile] if depth_profile is not None else []),
        "## Search coverage\n",
        _coverage_section(analysis),
        "## Top trials\n",
        _trials_table(analysis.get("top") or []),
        "## Pareto set\n",
        _trials_table(analysis.get("pareto") or []),
        "## Failures and prunes\n",
        _counts_section(analysis.get("counts", {})),
        "## Warnings\n",
        _warnings_section(analysis.get("warnings") or []),
        "## Reproduce\n",
        _reproduce_section(analysis, session_meta, model, llamacpp),
        "## Recommended runtime\n",
        _recommended_runtime_section(analysis, session_meta, model, llamacpp),
    ]
    return "\n".join(sections)
