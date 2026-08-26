"""Fit conservative VRAM-estimator calibration from completed sessions."""

from __future__ import annotations

import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from llamatune import bench, executor, hardware, stats
from llamatune.types import (
    CalibrationResult,
    LlamaCppReport,
    ModelReport,
    NightshiftOptions,
    RegistryRecord,
    TrialConfig,
)


class NightshiftRun(Protocol):
    """The artifact-writing subset of the frozen NightshiftRun interface."""

    dir: Path

    def calibration_dir(self, fingerprint16: str, n: int) -> Path: ...

    def write_json(self, name: str, payload: dict[str, Any]) -> None: ...


_CALIBRATION_TIMEOUT_S = 1800.0


def _required_capabilities(config: TrialConfig) -> tuple[str, ...]:
    """Return gated flags whose non-default semantics must be preserved."""
    required: list[str] = []
    if config.ot_spec is not None:
        required.append("ot")
    elif config.moe_cpu_layers > 0:
        required.append("ncmoe")
    if config.flash_attn:
        required.append("fa")
    if not config.mmap:
        required.append("mmp")
    if config.no_kv_offload:
        required.append("nkvo")
    if config.cache_type_k != "f16":
        required.append("ctk")
    if config.cache_type_v != "f16":
        required.append("ctv")
    if config.threads_batch is not None:
        required.append("tb")
    return tuple(required)


def _cooldown_s(options: NightshiftOptions) -> float:
    if options.cooldown_s is not None:
        return options.cooldown_s
    return 5.0 if options.profile == "deep" else 0.0


def _build_argv(
    record: RegistryRecord, model: ModelReport, llama: LlamaCppReport
) -> tuple[str, ...]:
    if record.reference_config is None:
        return bench.build_baseline_argv(
            bench_path=llama.bench_path,
            model_path=model.path,
            pp=record.pp_workload,
            tg=record.tg_workload,
            reps=record.reps_confirm,
            capabilities=llama.capabilities,
            depth=record.depth_workload,
        )
    return bench.build_bench_argv(
        bench_path=llama.bench_path,
        model_path=model.path,
        pp=record.pp_workload,
        tg=record.tg_workload,
        reps=record.reps_confirm,
        config=record.reference_config,
        capabilities=llama.capabilities,
        depth=record.depth_workload,
    )


def _command_payload(
    argv: tuple[str, ...],
    result: executor.ExecResult,
    load: tuple[float, float, float] | None,
) -> dict[str, Any]:
    return {
        "argv": list(argv),
        "timeout_s": _CALIBRATION_TIMEOUT_S,
        "core_dump_control": executor.core_dump_control_status(True),
        "load_average": list(load) if load is not None else None,
        "exit_code": result.exit_code,
        "wall_s": result.wall_s,
        "timed_out": result.timed_out,
        "started": result.started,
        "ended": result.ended,
        "env_names": list(result.env_names),
        "stdout": {
            "sha256": result.stdout.sha256,
            "size_bytes": result.stdout.size_bytes,
            "truncated": result.stdout.truncated,
        },
        "stderr": {
            "sha256": result.stderr.sha256,
            "size_bytes": result.stderr.size_bytes,
            "truncated": result.stderr.truncated,
        },
    }


def _failure_reason(result: executor.ExecResult) -> str | None:
    if result.timed_out:
        return "timeout"
    if result.stdout.truncated:
        return "parse_error"
    if result.exit_code == 0:
        return None
    stderr = result.stderr.path.read_text(encoding="utf-8", errors="replace")
    return bench.classify_failure(stderr) or "crash"


def _result(
    *,
    record: RegistryRecord,
    model: ModelReport,
    llama: LlamaCppReport,
    threshold: float,
    verdict: str,
    runs: int,
    reason: str | None = None,
    pp: Any = None,
    tg: Any = None,
    drift_pp: float | None = None,
    drift_tg: float | None = None,
    artifact_dir: Path | None = None,
) -> CalibrationResult:
    return CalibrationResult(
        fingerprint=model.fingerprint,
        reference_session=record.session_dir,
        verdict=verdict,
        pp=pp,
        tg=tg,
        drift_pp=drift_pp,
        drift_tg=drift_tg,
        threshold=threshold,
        runs=runs,
        reason=reason,
        transfer_from=record.fingerprint if record.fingerprint != model.fingerprint else None,
        build_changed=(
            record.bench_sha256 != llama.bench_sha256
            if record.bench_sha256 is not None and llama.bench_sha256 is not None
            else record.help_sha256 != llama.help_sha256
        ),
        artifact_dir=artifact_dir,
    )


def run_calibration(
    run: NightshiftRun,
    record: RegistryRecord,
    model: ModelReport,
    llama: LlamaCppReport,
    options: NightshiftOptions,
) -> CalibrationResult:
    """Re-measure recorded evidence and return a noise-aware drift verdict."""
    threshold = max(options.drift_threshold, 2.0 * record.noise_floor_cv)
    if record.depth_workload is not None and "d" not in llama.capabilities:
        return _result(
            record=record,
            model=model,
            llama=llama,
            threshold=threshold,
            verdict="error",
            runs=0,
            reason="capability_lost:d",
        )
    if record.reference_config is not None:
        for capability in _required_capabilities(record.reference_config):
            if capability not in llama.capabilities:
                return _result(
                    record=record,
                    model=model,
                    llama=llama,
                    threshold=threshold,
                    verdict="error",
                    runs=0,
                    reason=f"capability_lost:{capability}",
                )
    if record.reference_pp <= 0 or record.reference_tg <= 0:
        field = "pp" if record.reference_pp <= 0 else "tg"
        return _result(
            record=record,
            model=model,
            llama=llama,
            threshold=threshold,
            verdict="error",
            runs=0,
            reason=f"invalid_reference:{field}",
        )

    argv = _build_argv(record, model, llama)
    pp_values: list[float] = []
    tg_values: list[float] = []
    artifact_dir: Path | None = None
    completed = 0
    for number in range(1, options.calibration_runs + 1):
        run_dir = run.calibration_dir(model.fingerprint[:16], number)
        artifact_dir = run_dir.parent
        load = hardware.detect_load()
        try:
            exec_result = executor.run(
                argv,
                timeout_s=_CALIBRATION_TIMEOUT_S,
                stdout_path=run_dir / "stdout.json",
                stderr_path=run_dir / "stderr.log",
            )
        except OSError:
            return _result(
                record=record,
                model=model,
                llama=llama,
                threshold=threshold,
                verdict="error",
                runs=completed,
                reason="crash",
                artifact_dir=artifact_dir,
            )
        relative_command = (run_dir / "command.json").relative_to(run.dir)
        run.write_json(str(relative_command), _command_payload(argv, exec_result, load))
        completed += 1

        reason = _failure_reason(exec_result)
        if reason is not None:
            return _result(
                record=record,
                model=model,
                llama=llama,
                threshold=threshold,
                verdict="error",
                runs=completed,
                reason=reason,
                artifact_dir=artifact_dir,
            )
        try:
            sample = bench.parse_bench_output(exec_result.stdout.path.read_bytes())
        except (OSError, bench.BenchParseError):
            return _result(
                record=record,
                model=model,
                llama=llama,
                threshold=threshold,
                verdict="error",
                runs=completed,
                reason="parse_error",
                artifact_dir=artifact_dir,
            )
        pp_values.append(sample.pp_avg)
        tg_values.append(sample.tg_avg)
        if number < options.calibration_runs and _cooldown_s(options) > 0:
            time.sleep(_cooldown_s(options))

    pp = stats.metric_stats(pp_values)
    tg = stats.metric_stats(tg_values)
    drift_pp = abs(pp.mean - record.reference_pp) / record.reference_pp
    drift_tg = abs(tg.mean - record.reference_tg) / record.reference_tg
    verdict = "consistent" if drift_pp <= threshold and drift_tg <= threshold else "drift"
    return _result(
        record=record,
        model=model,
        llama=llama,
        threshold=threshold,
        verdict=verdict,
        runs=completed,
        pp=pp,
        tg=tg,
        drift_pp=drift_pp,
        drift_tg=drift_tg,
        artifact_dir=artifact_dir,
    )


def calibrate(sessions_dir: Path) -> dict[str, Any]:
    """Write ``calibration.json`` using robust total observed/estimated ratios.

    Runtime telemetry cannot identify weights, KV, and compute independently.
    We therefore apply the same median total correction to all three terms rather
    than inventing a component attribution. Fewer than three usable sessions keep
    the neutral 1.0 scale required by C13.
    """
    ratios: list[float] = []
    if sessions_dir.is_dir():
        for path in sessions_dir.iterdir():
            analysis_path = path / "analysis.json"
            try:
                analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
                observed = analysis.get("estimate_vs_observed") or {}
                estimated = float(observed.get("estimated_total_mb") or 0.0)
                actual = float(observed.get("observed_used_delta_mb") or 0.0)
                if estimated > 0 and actual > 0:
                    ratios.append(actual / estimated)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
    scale = statistics.median(ratios) if len(ratios) >= 3 else 1.0
    result = {
        "schema_version": 1,
        "fitted_at": datetime.now(UTC).isoformat(),
        "samples": len(ratios),
        "weights_scale": scale,
        "kv_scale": scale,
        "compute_scale": scale,
    }
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "calibration.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
