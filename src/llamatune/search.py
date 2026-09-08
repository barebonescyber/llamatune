"""Search orchestration: baseline, coordinate ascent, confirmation (DESIGN §§6, 10).

Replaces Packet A's interface stub. Implements the two DESIGN §13.3 entry
points. The engine drives every llama-bench invocation through
:mod:`llamatune.executor`, parses results with :mod:`llamatune.bench`, scores
with :mod:`llamatune.stats`, journals every outcome through
:class:`llamatune.session.Session`, and emits evidence via
:mod:`llamatune.recommend`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import signal
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from llamatune import bench, executor, recommend, stats
from llamatune.config import (
    DIMENSION_ORDER,
    applicable_dimensions,
    candidates_for,
    dimension_value,
    estimate_ram,
    estimate_reason,
    estimate_vram,
    free_vram_mb,
    gpu_available,
    gpu_layer_cap,
    gpu_present,
    hardware_signature,
    has_gpu_backend,
    is_valid_config,
    moe_cpu_layer_candidates,
    multi_gpu_capacity_check,
    multi_gpu_placement_candidates,
    mutate,
    total_vram_mb,
)
from llamatune.session import Session, SessionCorruptionError, SessionPathError
from llamatune.types import (
    BaselineResult,
    FeasibilityBoundary,
    GpuSample,
    HardwareReport,
    LlamaCppReport,
    ModelReport,
    ProgressEvent,
    Reporter,
    TrialConfig,
    TuneOptions,
    TuneOutcome,
    VramCalibration,
)

#: Maximum full coordinate-ascent passes (DESIGN §10 step 5).
MAX_PASSES = 3

_BASELINE_TIMEOUT_S = 1800.0
_TRIAL_TIMEOUT_MIN_S = 120.0
_TRIAL_TIMEOUT_MAX_S = 3600.0
_DEEP_TRIAL_TIMEOUT_MAX_S = 7200.0
_THERMAL_WAIT_CYCLE_S = 10.0


def _sample_gpu_state() -> GpuSample | None:
    from llamatune.hardware import sample_gpu_state

    return sample_gpu_state()


def _detect_gpu_throttle(samples: list[GpuSample]) -> bool:
    from llamatune.hardware import detect_throttle

    return detect_throttle(samples)


def _boundary_has_fit(boundary: FeasibilityBoundary) -> bool:
    """Whether a boundary contains at least one observed fitting placement."""
    return not (boundary.max_ok_ngl == 0 and boundary.min_fail_ngl == 0)


class _BudgetExhaustedError(Exception):
    """Internal signal: a budget is exhausted; jump to confirmation."""


def _budget_reason(engine: _Engine) -> str:
    """Name the budget that actually tripped (DESIGN §3; issue #39)."""
    if engine.executed_count >= engine._trial_limit():
        return "trial budget was exhausted"
    minutes = engine.options.budget_minutes
    if minutes is not None:
        used = min((_monotonic() - engine.start) / 60.0, minutes)
        reason = (
            f"the time budget was exhausted "
            f"({_fmt_minutes(used)} of {_fmt_minutes(minutes)} minutes used)"
        )
        return reason
    return "trial budget was exhausted"


def _fmt_minutes(value: float) -> str:
    if value >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


class _NoFeasibleConfigError(Exception):
    """Internal signal: no measured configuration satisfies a binding cap."""


class _BaselineError(Exception):
    """Internal signal: the baseline cannot be measured (DESIGN exit code 3)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True, slots=True)
class _Trial:
    status: str
    pp: float | None
    tg: float | None
    score: float | None


@dataclass(frozen=True, slots=True)
class _Measured:
    status: str
    pp: Any
    tg: Any
    entry: dict[str, Any] | None
    oom_pattern: str | None


@dataclass(frozen=True, slots=True)
class _Confirm:
    confirmed: bool
    pp: Any
    tg: Any
    score: float


@dataclass(frozen=True, slots=True)
class _RunObservation:
    before: GpuSample | None
    after: GpuSample | None
    samples: tuple[GpuSample, ...]


@dataclass(frozen=True, slots=True)
class _ThermalRetest:
    result: executor.ExecResult
    observation: _RunObservation
    measured: _Measured
    contaminated: bool
    retried: bool
    retry_contaminated: bool | None


@dataclass(frozen=True, slots=True)
class _StabilityRerun:
    measured: _Measured
    contaminated: bool
    retried: bool
    retry_contaminated: bool | None
    thermal_rejected: bool


def run_tuning(
    session: Session,
    hardware: HardwareReport,
    model: ModelReport,
    llama: LlamaCppReport,
    options: TuneOptions,
    *,
    reporter: Reporter | None = None,
    calibration: VramCalibration | None = None,
) -> TuneOutcome:
    """Run the full tune pipeline: baseline, search, confirmation, emission."""
    return _Engine(
        session, hardware, model, llama, options, reporter=reporter, calibration=calibration
    ).run()


def resume_tuning(session_dir: Path, *, reporter: Reporter | None = None) -> TuneOutcome:
    """Resume a tuning session, re-validating identity and skipping journaled trials."""
    from llamatune.hardware import assess_hardware
    from llamatune.llama import LlamaDiscoveryError, discover_llama
    from llamatune.model import ModelInspectionError, inspect_model

    try:
        session = Session.load(session_dir)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        SessionCorruptionError,
        SessionPathError,
    ) as exc:
        # Unreadable or malformed session directory: a usage/configuration
        # error (DESIGN §3, exit code 2); there is no journal to append to.
        return TuneOutcome(
            session_dir=Path(session_dir),
            analysis={},
            exit_code=2,
            failure_stage="session",
            failure_reason=str(exc),
        )
    options = session.options

    try:
        llama = discover_llama(options.llama_bin)
        model = inspect_model(session.model.path, full_hash=options.full_hash)
    except (LlamaDiscoveryError, ModelInspectionError) as exc:
        session.append({"type": "session_end", "exit_code": 3, "reason": str(exc)})
        return TuneOutcome(
            session_dir=session.dir,
            analysis={},
            exit_code=3,
            failure_stage="resume_validation",
            failure_reason=str(exc),
        )

    if (
        model.fingerprint != session.model.fingerprint
        or llama.help_sha256 != session.llama.help_sha256
    ):
        reason = "model or llama.cpp identity mismatch"
        session.append({"type": "session_end", "exit_code": 3, "reason": "identity mismatch"})
        return TuneOutcome(
            session_dir=session.dir,
            analysis={},
            exit_code=3,
            failure_stage="resume_validation",
            failure_reason=reason,
        )

    hardware = assess_hardware()
    drift_warnings: list[str] = []
    if (
        session.llama.bench_sha256 is not None
        and llama.bench_sha256 is not None
        and llama.bench_sha256 != session.llama.bench_sha256
    ):
        drift_warnings.append(
            "llama-bench binary changed since this session was created (rebuild?); "
            "results may mix builds"
        )
    if _hardware_signature(hardware) != _hardware_signature(session.hardware):
        drift_warnings.append("hardware drift detected on resume; using session-recorded hardware")
    calibration = _load_calibration(options.sessions_dir)
    if (options.sessions_dir / "calibration.json").is_file() and calibration is None:
        drift_warnings.append("ignoring corrupt calibration file")

    session.append(
        {
            "type": "stage",
            "stage": "resumed",
            "journaled_trials": len(session.journaled_trial_ids),
            "warnings": drift_warnings + list(session.resume_warnings),
        }
    )

    # The search runs against the session-recorded hardware/model (persisted
    # evidence), keeping a resumed search deterministic; the fresh scans above
    # exist only to validate identity and detect drift (DESIGN §12).
    engine = _Engine(
        session,
        session.hardware,
        model,
        llama,
        options,
        extra_warnings=drift_warnings,
        reporter=reporter,
        calibration=calibration,
    )
    return engine.run()


def _load_calibration(sessions_dir: Path) -> VramCalibration | None:
    try:
        data = json.loads((sessions_dir / "calibration.json").read_text(encoding="utf-8"))
        return VramCalibration(**data)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def revalidate_session(session_dir: Path) -> TuneOutcome:
    """Re-run only the confirmation batch for a recorded recommendation."""
    from llamatune.hardware import assess_hardware
    from llamatune.llama import LlamaDiscoveryError, discover_llama
    from llamatune.model import ModelInspectionError, inspect_model

    try:
        session = Session.load(session_dir)
        analysis = session.read_json("analysis.json")
        winner = analysis.get("winner")
        if not isinstance(winner, dict):
            raise ValueError("session has no confirmed winner")
        llama = discover_llama(session.options.llama_bin)
        model = inspect_model(session.model.path, full_hash=session.options.full_hash)
        config = TrialConfig.from_dict(winner["config"])
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        SessionCorruptionError,
        SessionPathError,
        LlamaDiscoveryError,
        ModelInspectionError,
    ) as exc:
        return TuneOutcome(
            session_dir=Path(session_dir),
            analysis={},
            exit_code=3,
            failure_stage="revalidation",
            failure_reason=str(exc),
        )
    if (
        model.fingerprint != session.model.fingerprint
        or llama.help_sha256 != session.llama.help_sha256
    ):
        return TuneOutcome(
            session_dir=session.dir,
            analysis={},
            exit_code=3,
            failure_stage="revalidation",
            failure_reason="model or llama.cpp identity mismatch",
        )
    engine = _Engine(session, assess_hardware(), model, llama, session.options)
    engine._tuning_budget_enforced = False
    engine._finalizing = True
    baseline_stage = engine._find_baseline_stage()
    if baseline_stage is None:
        return TuneOutcome(
            session_dir=session.dir,
            analysis={},
            exit_code=3,
            failure_stage="revalidation",
            failure_reason="session has no completed baseline evidence",
        )
    try:
        engine._load_baseline_stage(baseline_stage)
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        return TuneOutcome(
            session_dir=session.dir,
            analysis={},
            exit_code=3,
            failure_stage="revalidation",
            failure_reason=f"invalid baseline evidence: {exc}",
        )
    confirmation = engine._confirm(config)
    previous_score = float(winner.get("score", 0.0))
    within_noise = confirmation.score >= previous_score * (1 - engine.baseline.noise_floor_cv)
    result = {
        "status": "reproduced" if within_noise else "regressed",
        "comparison": {
            "previous_score": previous_score,
            "current_score": confirmation.score,
            "noise_floor_cv": engine.baseline.noise_floor_cv,
            "within_noise": within_noise,
        },
    }
    engine._append({"type": "stage", "stage": "revalidation", **result})
    return TuneOutcome(session_dir=session.dir, analysis=result, exit_code=0 if within_noise else 1)


def _hardware_signature(hardware: HardwareReport) -> tuple[Any, ...]:
    """Compatibility alias for the pure config-layer identity helper."""
    return hardware_signature(hardware)


class _Engine:
    """Stateful driver for one (possibly resumed) tuning run."""

    def __init__(
        self,
        session: Session,
        hardware: HardwareReport,
        model: ModelReport,
        llama: LlamaCppReport,
        options: TuneOptions,
        *,
        extra_warnings: list[str] | None = None,
        reporter: Reporter | None = None,
        calibration: VramCalibration | None = None,
    ) -> None:
        self.session = session
        self.hardware = hardware
        self.model = model
        self.llama = llama
        self.options = options
        self.reporter = reporter
        self.calibration = calibration
        self._reporter_failures = 0
        self._reporter_warning_added = False
        self.caps = llama.capabilities
        self.weights = stats.weights_for_target(options.target)

        self.known: dict[str, dict[str, Any]] = {}
        self.probes: dict[str, dict[str, Any]] = {}
        self.oom_points: list[tuple[dict[str, Any], str]] = []
        for entry in session.entries:
            if entry.get("type") == "trial" and "trial_id" in entry:
                self.known[entry["trial_id"]] = entry
                if entry.get("status") in ("oom", "gpu_resource"):
                    self.oom_points.append((dict(entry["config"]), entry["trial_id"]))
            elif entry.get("type") == "probe" and "probe_id" in entry:
                self.probes[entry["probe_id"]] = entry

        self.executed_count = _count_executed(session.entries)
        self.start = _monotonic()
        self.extra_warnings = list(extra_warnings or [])
        self.load_warnings: set[str] = set()
        self._load_total = 0
        self._load_exceeded = 0
        self._load_peak = 0.0
        self._load_peak_threshold = 0.0
        self._load_warning: str | None = None
        self.trial_timeout = _TRIAL_TIMEOUT_MIN_S
        self.vram_reserve_mb = (
            options.vram_reserve_mb if options.vram_reserve_mb is not None else 1536
        )
        self.reserve_provenance = "cli" if options.vram_reserve_mb is not None else "auto-default"
        self.boundaries: list[FeasibilityBoundary] = []
        self.context_validation: dict[str, Any] | None = None
        self.depth_profile: dict[str, Any] | None = None
        self.context_envelope: list[dict[str, Any]] | None = None
        self.cli_validation: dict[str, Any] | None = None
        self.estimate_vs_observed: dict[str, Any] | None = None
        self._observations: dict[str, _RunObservation] = {}
        self.coverage: dict[str, dict[str, Any]] = {}
        self.quality_gate: dict[str, Any] | None = None
        self._thermal_samples: list[GpuSample] = []
        thermal_pauses = [
            entry
            for entry in session.entries
            if entry.get("type") == "stage" and entry.get("stage") == "thermal_pause"
        ]
        self.thermal_pause_count = len(thermal_pauses)
        self.thermal_wait_s = sum(float(entry.get("waited_s", 0.0)) for entry in thermal_pauses)
        self._stop_requested = False
        self._signal_count = 0
        self._tuning_budget_enforced = True
        self._finalizing = False
        self.validated_configs: set[str] = set()
        self.default_probe: dict[str, Any] | None = self._find_stage("default_probe")

        # Populated by _establish_baseline / _seed_incumbent.
        self.baseline: BaselineResult
        self.default_config: TrialConfig
        self.measured_default: TrialConfig
        self.incumbent_config: TrialConfig
        self.incumbent_pp = 0.0
        self.incumbent_tg = 0.0
        self.incumbent_score = 1.0

    # -- top level ------------------------------------------------------

    def run(self) -> TuneOutcome:
        self._emit(
            "session_start",
            session_dir=str(self.session.dir),
            model_name=self.model.name or self.model.path.name,
            target=self.options.target,
            budget_trials=self.options.budget_trials,
            budget_minutes=self.options.budget_minutes,
            ctx_size=self.options.ctx_size,
            stop_hint="Ctrl-C once to confirm best-so-far; twice to abort",
        )
        try:
            self._ensure_plan()
        except OSError as exc:
            return self._io_failure(exc)
        previous_handler: Any = None
        installed = False
        if threading.current_thread() is threading.main_thread() and hasattr(signal, "SIGINT"):
            previous_handler = signal.getsignal(signal.SIGINT)

            def handle_sigint(_signum: int, _frame: Any) -> None:
                self._signal_count += 1
                if self._signal_count == 1:
                    self._stop_requested = True
                    warning = "search interrupted by user; confirming best-so-far"
                    self.extra_warnings.append(warning)
                    self._emit("warning", message=warning)
                    return
                raise KeyboardInterrupt

            signal.signal(signal.SIGINT, handle_sigint)
            installed = True
        try:
            return self._run()
        except KeyboardInterrupt:
            try:
                self._session_end(4, reason="interrupted")
            except OSError as exc:
                return self._io_failure(exc)
            return TuneOutcome(
                session_dir=self.session.dir,
                analysis={},
                exit_code=4,
                failure_stage="interruption",
                failure_reason="interrupted by user",
            )
        except OSError as exc:
            return self._io_failure(exc)
        finally:
            if installed:
                signal.signal(signal.SIGINT, previous_handler)

    def _io_failure(self, exc: OSError) -> TuneOutcome:
        """Return a resumable mid-run failure even when evidence storage is full."""
        reason = f"session I/O failure: {exc}"
        # The same unwritable/full filesystem may prevent the terminal
        # journal record. Preserve the controlled outcome; completed
        # journal entries remain resumable after storage is repaired.
        with contextlib.suppress(OSError):
            self._session_end(4, reason=reason)
        return TuneOutcome(
            session_dir=self.session.dir,
            analysis={},
            exit_code=4,
            failure_stage="evidence",
            failure_reason=reason,
        )

    def _run(self) -> TuneOutcome:
        try:
            self._establish_baseline()
        except _BaselineError as exc:
            self._session_end(3, reason=str(exc))
            return TuneOutcome(
                session_dir=self.session.dir,
                analysis={},
                exit_code=3,
                failure_stage="baseline",
                failure_reason=exc.message,
            )

        self._check_backend_mismatch()

        if self.options.baseline_only:
            return self._finalize_or_no_feasible(baseline_only=True)

        try:
            self._seed_incumbent()
            self._search()
        except _BudgetExhaustedError:
            self._append(
                {
                    "type": "stage",
                    "stage": "search_stopped" if self._stop_requested else "budget_exhausted",
                }
            )
        self._finalizing = True
        if not self._stop_requested:
            self._validate_recommendation()
        return self._finalize_or_no_feasible(baseline_only=False)

    def _finalize_or_no_feasible(self, *, baseline_only: bool) -> TuneOutcome:
        try:
            return self._finalize(baseline_only=baseline_only)
        except _NoFeasibleConfigError as exc:
            reason = str(exc)
            self.session.invalidate_derived_outputs()
            self._append({"type": "stage", "stage": "no_feasible_config", "reason": reason})
            self._session_end(3, reason=reason)
            return TuneOutcome(
                session_dir=self.session.dir,
                analysis={},
                exit_code=3,
                failure_stage="recommendation",
                failure_reason=reason,
            )

    def _emit(self, event_kind: str, **payload: Any) -> None:
        if self.reporter is None:
            return
        try:
            self.reporter.emit(
                ProgressEvent(kind=event_kind, ts=datetime.now(UTC).isoformat(), payload=payload)
            )
        except Exception:
            self._reporter_failures += 1
            if not self._reporter_warning_added:
                self.extra_warnings.append("progress reporter failed; tuning continued")
                self._reporter_warning_added = True
            if self._reporter_failures >= 3:
                self.reporter = None

    def _append(self, record: dict[str, Any]) -> None:
        self.session.append(record)
        if record.get("type") == "stage":
            payload = {k: v for k, v in record.items() if k not in ("type", "ts")}
            self._emit("stage", **payload)

    def _session_end(
        self, exit_code: int, *, reason: str | None = None, winner_trial_id: str | None = None
    ) -> None:
        record: dict[str, Any] = {"type": "session_end", "exit_code": exit_code}
        if reason is not None:
            record["reason"] = reason
        self.session.append(record)
        self._emit(
            "session_end",
            exit_code=exit_code,
            winner_trial_id=winner_trial_id,
            session_dir=str(self.session.dir),
        )

    def _run_child(
        self,
        *,
        kind: str,
        label: str,
        run_id: str,
        argv: tuple[str, ...],
        timeout_s: float,
        stdout_path: Path,
        stderr_path: Path,
    ) -> tuple[executor.ExecResult, _RunObservation]:
        from llamatune.hardware import sample_gpu_states

        samples: list[GpuSample] = []
        track_devices = self.options.multi_gpu and len(self.hardware.gpus) > 1

        def observed_samples() -> tuple[GpuSample, ...]:
            values = sample_gpu_states() if self.options.observe_vram else ()
            if track_devices:
                return values
            return tuple(dataclasses.replace(sample, device_index=None) for sample in values[:1])

        before_samples = observed_samples()
        before = before_samples[0] if before_samples else None
        samples.extend(before_samples)
        last_sample_s = 0.0
        self._emit(
            "exec_start",
            kind=kind,
            label=label,
            id=run_id,
            timeout_s=timeout_s,
            executed=self.executed_count,
            budget_trials=self.options.budget_trials,
            elapsed_s=_monotonic() - self.start,
        )

        def heartbeat(elapsed_s: float) -> None:
            nonlocal last_sample_s
            self._emit("exec_heartbeat", label=label, elapsed_s=elapsed_s, timeout_s=timeout_s)
            if (
                self.options.observe_vram
                and elapsed_s - last_sample_s >= 10.0
                and len(samples) < 29
            ):
                samples.extend(observed_samples())
                last_sample_s = elapsed_s

        result = executor.run(
            argv,
            timeout_s=timeout_s,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            disable_core_dumps=not self.options.allow_core_dumps,
            on_heartbeat=heartbeat,
        )
        after_samples = observed_samples()
        after = after_samples[0] if after_samples else None
        samples.extend(after_samples)
        observation = _RunObservation(before, after, tuple(samples[:30]))
        self._observations[run_id] = observation
        if samples:
            self._thermal_samples.extend(samples)
            self._thermal_samples = self._thermal_samples[-30:]
        return result, observation

    def _emit_exec_end(
        self,
        label: str,
        run_id: str,
        result: executor.ExecResult,
        measured: _Measured,
        score: float | None = None,
    ) -> None:
        self._emit(
            "exec_end",
            label=label,
            id=run_id,
            status=measured.status,
            wall_s=result.wall_s,
            pp=measured.pp.mean if measured.pp is not None else None,
            tg=measured.tg.mean if measured.tg is not None else None,
            score=score,
        )

    def _thermal_retest(
        self,
        *,
        kind: str,
        trial_id: str,
        run_id: str,
        label: str,
        argv: tuple[str, ...],
        timeout_s: float,
        reps: int,
        threads: int,
        retry_dir: Path,
        retry_rel_dir: str,
        result: executor.ExecResult,
        observation: _RunObservation,
        measured: _Measured,
    ) -> _ThermalRetest:
        contaminated = bool(
            measured.status == "ok"
            and measured.pp is not None
            and measured.tg is not None
            and self.options.observe_vram
            and observation.samples
            and _detect_gpu_throttle(list(observation.samples))
        )
        if not contaminated:
            return _ThermalRetest(result, observation, measured, False, False, None)

        # Close the original execution under its own identity before emitting a
        # replacement execution. Callers must not re-emit this run id afterward.
        self._emit_exec_end(label, run_id, result, measured)

        # This is the normal bounded cooldown, moved between the contaminated
        # attempt and its single replacement measurement.
        self._cooldown()
        can_retry = self._can_execute()
        retry_reason = None
        if not can_retry:
            retry_reason = "stop" if self._stop_requested else "budget"
        retry_run_id = f"{run_id}-thermal-retry"
        self._append(
            {
                "type": "stage",
                "stage": "thermal_contamination",
                "run_kind": kind,
                "trial_id": trial_id,
                "run_id": run_id,
                "sample_count": len(observation.samples),
                "retry_scheduled": can_retry,
                "retry_reason": retry_reason,
                "retry_run_id": retry_run_id if can_retry else None,
            }
        )
        if not can_retry:
            rejected = dataclasses.replace(measured, status="unstable")
            return _ThermalRetest(result, observation, rejected, True, False, None)

        retry_dir.mkdir(parents=True, exist_ok=True)
        retry_label = f"thermal retry {label}"
        retry_result, retry_observation = self._run_child(
            kind=kind,
            label=retry_label,
            run_id=retry_run_id,
            argv=argv,
            timeout_s=timeout_s,
            stdout_path=retry_dir / "stdout.json",
            stderr_path=retry_dir / "stderr.log",
        )
        self.executed_count += 1
        retry_measured = self._classify(retry_result, reps)
        retry_contaminated = bool(
            retry_measured.status == "ok"
            and retry_measured.pp is not None
            and retry_measured.tg is not None
            and retry_observation.samples
            and _detect_gpu_throttle(list(retry_observation.samples))
        )
        if retry_contaminated:
            retry_measured = dataclasses.replace(retry_measured, status="unstable")
        self._write_command(
            retry_rel_dir,
            argv,
            retry_result,
            reps,
            self._record_load(threads),
            retry_observation,
        )
        self._append(
            {
                "type": "thermal_retry",
                "run_kind": kind,
                "trial_id": trial_id,
                "run_id": run_id,
                "retry_run_id": retry_run_id,
                "status": retry_measured.status,
                "pp_mean": (retry_measured.pp.mean if retry_measured.pp is not None else None),
                "tg_mean": (retry_measured.tg.mean if retry_measured.tg is not None else None),
                "sample_count": len(retry_observation.samples),
                "thermally_contaminated": retry_contaminated,
            }
        )
        self._emit_exec_end(retry_label, retry_run_id, retry_result, retry_measured)
        self._cooldown()
        return _ThermalRetest(
            retry_result,
            retry_observation,
            retry_measured,
            True,
            True,
            retry_contaminated,
        )

    def _config_label(self, cfg: TrialConfig) -> str:
        return (
            f"ngl={cfg.gpu_layers} ncmoe={cfg.moe_cpu_layers} "
            f"fa={int(cfg.flash_attn)} ub={cfg.ubatch} b={cfg.batch}"
        )

    # -- baseline -------------------------------------------------------

    def _establish_baseline(self) -> None:
        stage = self._find_baseline_stage()
        if stage is not None:
            self._load_baseline_stage(stage)
            return

        self._quiet_gate("baseline")

        walls: list[float] = []
        probe = self._run_baseline_once(1, ngl=None, walls=walls)
        fallback: str | None = None
        ngl: int | None = None
        runs: list[_Measured] = []
        self.default_probe = {
            "type": "stage",
            "stage": "default_probe",
            "status": probe.status,
            "classification": probe.status,
            "pattern": probe.oom_pattern,
            "evidence": "baseline/run-1",
        }
        self._append(self.default_probe)
        if probe.status != "ok":
            fallback = "cpu"
            ngl = 0
        else:
            runs.append(probe)

        last_index = self.options.baseline_runs + (1 if fallback == "cpu" else 0)
        for index in range(2, last_index + 1):
            measured = self._run_baseline_once(
                index,
                ngl=ngl,
                walls=walls,
                display_index=index - 1 if fallback == "cpu" else index,
                display_prefix="CPU fallback" if fallback == "cpu" else "baseline",
            )
            if measured.status != "ok":
                msg = (
                    f"default probe failed ({probe.oom_pattern or probe.status}) and "
                    f"-ngl 0 fallback failed ({measured.oom_pattern or measured.status}); "
                    f"see baseline/run-1 and baseline/run-{index}"
                )
                raise _BaselineError(msg)
            runs.append(measured)

        pp_agg = stats.metric_stats([m.pp.mean for m in runs])
        tg_agg = stats.metric_stats([m.tg.mean for m in runs])
        cv_nf = stats.noise_floor_cv(pp_agg, tg_agg)
        median_wall = _median(walls)
        timeout_max = (
            _DEEP_TRIAL_TIMEOUT_MAX_S
            if self.options.depth is not None and self.options.depth > 0
            else _TRIAL_TIMEOUT_MAX_S
        )
        self.trial_timeout = min(max(6.0 * median_wall, _TRIAL_TIMEOUT_MIN_S), timeout_max)

        last_entry = runs[-1].entry
        if last_entry is None:
            raise _BaselineError(
                f"baseline/run-{last_index} reported success without a benchmark entry"
            )
        backends = last_entry.get("backends") or last_entry.get("backend")
        self.llama = dataclasses.replace(self.llama, backends=backends)
        force = None if gpu_available(self.hardware, self.llama) else 0
        use_multi_gpu_placement = self.options.multi_gpu and len(self.hardware.gpus) > 1
        self.measured_default = self._normalize_resolved_config(
            bench.resolved_config(last_entry, multi_gpu=use_multi_gpu_placement)
        )
        self.default_config = self._normalize_resolved_config(
            bench.resolved_config(
                last_entry,
                force_gpu_layers=force,
                multi_gpu=use_multi_gpu_placement,
            )
        )

        self.baseline = BaselineResult(
            runs=len(runs),
            pp=pp_agg,
            tg=tg_agg,
            noise_floor_cv=cv_nf,
            fallback=fallback,
            resolved_defaults=self.default_config.to_dict(),
            kind="safe_fallback" if fallback else "defaults",
        )
        raw_commit = last_entry.get("build_commit")
        raw_number = _opt_int(last_entry.get("build_number"))
        build_commit, build_number = bench.normalize_build_info(raw_commit, raw_number)
        if (raw_commit is not None or raw_number is not None) and (
            build_commit != raw_commit or build_number != raw_number
        ):
            self.extra_warnings.append(
                "llama.cpp build reports no embedded version info (commit 'unknown'); "
                "rebuild from a git checkout for traceable reports"
            )
        self.session.record_build_info(build_commit, build_number, backends)
        self.llama = self.session.llama
        self._append(
            {
                "type": "stage",
                "stage": "baseline_complete",
                "pp": stats.metric_to_dict(pp_agg),
                "tg": stats.metric_to_dict(tg_agg),
                "noise_floor_cv": cv_nf,
                "fallback": fallback,
                "fallback_ngl": 0 if fallback else None,
                "kind": "safe_fallback" if fallback else "defaults",
                "resolved_defaults": self.default_config.to_dict(),
                "measured_defaults": self.measured_default.to_dict(),
                "trial_timeout_s": self.trial_timeout,
                "backends": backends,
                **({"depth": self.options.depth} if self.options.depth is not None else {}),
            }
        )

    def _find_baseline_stage(self) -> dict[str, Any] | None:
        for entry in self.session.entries:
            if entry.get("type") == "stage" and entry.get("stage") == "baseline_complete":
                return entry
        return None

    def _find_stage(self, name: str) -> dict[str, Any] | None:
        return next(
            (
                e
                for e in reversed(self.session.entries)
                if e.get("type") == "stage" and e.get("stage") == name
            ),
            None,
        )

    def _ensure_plan(self) -> None:
        if self._find_stage("plan") is None:
            total = total_vram_mb(self.hardware, multi_gpu=self.options.multi_gpu)
            free = free_vram_mb(self.hardware, multi_gpu=self.options.multi_gpu)
            total_budget = None if total is None else max(0, total - self.vram_reserve_mb)
            free_budget = None if free is None else max(0, free - 256)
            budgets = [value for value in (total_budget, free_budget) if value is not None]
            budget_mb = min(budgets) if budgets else None
            budget_basis = (
                "observed-free"
                if free_budget is not None and budget_mb == free_budget
                else "total-reserve"
            )
            self._append(
                {
                    "type": "stage",
                    "stage": "plan",
                    "vram_reserve_mb": self.vram_reserve_mb,
                    "reserve_provenance": self.reserve_provenance,
                    "budget_mb": budget_mb,
                    "budget_basis": budget_basis,
                    **(
                        {"capacity_basis": "pooled-known-devices"} if self.options.multi_gpu else {}
                    ),
                    "initial_gpu_layers": self.options.initial_gpu_layers,
                    "initial_gpu_layers_provenance": "cli"
                    if self.options.initial_gpu_layers is not None
                    else "auto",
                    "initial_cpu_moe": self.options.initial_cpu_moe,
                    "initial_cpu_moe_provenance": "cli"
                    if self.options.initial_cpu_moe is not None
                    else "auto",
                }
            )

    def _load_baseline_stage(self, stage: dict[str, Any]) -> None:
        self.llama = dataclasses.replace(
            self.session.llama,
            backends=stage.get("backends"),
        )
        pp_agg = stats.metric_from_dict(stage["pp"])
        tg_agg = stats.metric_from_dict(stage["tg"])
        self.default_config = self._normalize_resolved_config(
            TrialConfig.from_dict(stage["resolved_defaults"])
        )
        self.measured_default = self._normalize_resolved_config(
            TrialConfig.from_dict(stage["measured_defaults"])
        )
        self.trial_timeout = float(stage.get("trial_timeout_s", _TRIAL_TIMEOUT_MIN_S))
        self.baseline = BaselineResult(
            runs=pp_agg.n,
            pp=pp_agg,
            tg=tg_agg,
            noise_floor_cv=float(stage["noise_floor_cv"]),
            fallback=stage.get("fallback"),
            resolved_defaults=stage["resolved_defaults"],
            kind=stage.get("kind", "defaults"),
        )

    def _run_baseline_once(
        self,
        index: int,
        *,
        ngl: int | None,
        walls: list[float],
        display_index: int | None = None,
        display_prefix: str = "baseline",
    ) -> _Measured:
        if self.executed_count >= self.options.budget_trials:
            raise _BaselineError("trial budget exhausted before baseline completed")
        argv = bench.build_baseline_argv(
            bench_path=self.llama.bench_path,
            model_path=self.model.path,
            pp=self.options.pp,
            tg=self.options.tg,
            reps=self.options.reps_confirm,
            ngl=ngl,
            capabilities=self.caps,
            depth=self.options.depth,
        )
        run_dir = self.session.baseline_dir(index)
        load = self._record_load(self.hardware.physical_cores)
        progress_index = index if display_index is None else display_index
        label = f"{display_prefix} {progress_index}/{self.options.baseline_runs}"
        run_id = f"baseline-{index}"
        result, observation = self._run_child(
            kind="baseline",
            label=label,
            run_id=run_id,
            argv=argv,
            timeout_s=_BASELINE_TIMEOUT_S,
            stdout_path=run_dir / "stdout.json",
            stderr_path=run_dir / "stderr.log",
        )
        self.executed_count += 1
        walls.append(result.wall_s)
        measured = self._classify(result, self.options.reps_confirm)
        self._write_command(
            f"baseline/run-{index}", argv, result, self.options.reps_confirm, load, observation
        )
        self._append(
            {
                "type": "baseline_run",
                "run": index,
                "ngl": ngl,
                "status": measured.status,
                "pp_mean": measured.pp.mean if measured.pp is not None else None,
                "tg_mean": measured.tg.mean if measured.tg is not None else None,
                "oom_pattern": measured.oom_pattern,
            }
        )
        self._emit_exec_end(label, run_id, result, measured)
        self._emit(
            "baseline",
            run=progress_index,
            runs=self.options.baseline_runs,
            status=measured.status,
            pp=measured.pp.mean if measured.pp is not None else None,
            tg=measured.tg.mean if measured.tg is not None else None,
        )
        self._cooldown()
        return measured

    # -- incumbent seeding ---------------------------------------------

    def _seed_incumbent(self) -> None:
        """Choose and measure the search starting point (DESIGN §10 step 1).

        Always anchors the incumbent on the baseline first, so a budget
        exhausted (or failing) seed trial still leaves a measured reference.
        """
        self._set_incumbent(
            self.default_config,
            self.baseline.pp.mean,
            self.baseline.tg.mean,
            self._score(self.baseline.pp.mean, self.baseline.tg.mean),
        )

        # Boundary discovery, rather than an unproven full-offload seed,
        # advances from this measured anchor.

    def _set_incumbent(self, config: TrialConfig, pp: float, tg: float, score: float) -> None:
        self.incumbent_config = config
        self.incumbent_pp = pp
        self.incumbent_tg = tg
        self.incumbent_score = score
        self._emit("incumbent", config=config.to_dict(), pp=pp, tg=tg, score=score)

    # -- coordinate ascent ---------------------------------------------

    def _search(self) -> None:
        if gpu_available(self.hardware, self.llama):
            self._joint_discovery()
            self._multi_gpu_placement_sweep()
        dimensions = tuple(
            dim for dim in DIMENSION_ORDER if dim not in ("gpu_layers", "moe_cpu_layers")
        )
        for pass_no in range(1, MAX_PASSES + 1):
            self._append({"type": "stage", "stage": "pass_start", "pass": pass_no})
            score_before = self.incumbent_score
            for dim in dimensions:
                self._sweep(dim)
                if dim in {"threads", "threads_batch"}:
                    self._refine_thread(dim)
            relative_gain = (self.incumbent_score / score_before) - 1 if score_before > 0 else 0.0
            if relative_gain < self.baseline.noise_floor_cv:
                break
        self._pairwise_refine()
        if gpu_available(self.hardware, self.llama):
            self._refine_joint()
            self._ot_refine()

    def _multi_gpu_placement_sweep(self) -> None:
        placements = multi_gpu_placement_candidates(self.hardware, self.llama, self.options)
        if not placements:
            return
        self._append(
            {"type": "stage", "stage": "multi_gpu_placement_start", "candidates": len(placements)}
        )
        base = self.incumbent_config
        for tensor_split, split_mode in placements:
            if not self._can_execute():
                break
            candidate = dataclasses.replace(base, tensor_split=tensor_split, split_mode=split_mode)
            estimate = estimate_vram(
                config=candidate,
                model=self.model,
                ctx=self.options.ctx_size,
                vram_reserve_mb=self.vram_reserve_mb,
                vram_total_mb=total_vram_mb(self.hardware, multi_gpu=True),
                vram_free_mb=free_vram_mb(self.hardware, multi_gpu=True),
                calibration=self.calibration,
            )
            feasible, per_device_mb = multi_gpu_capacity_check(
                candidate, self.hardware, estimate.total_mb, self.vram_reserve_mb
            )
            self._append(
                {
                    "type": "stage",
                    "stage": "multi_gpu_placement_estimate",
                    "tensor_split": list(tensor_split),
                    "split_mode": split_mode,
                    "per_device_mb": list(per_device_mb),
                    "estimated_feasible": feasible,
                    "advisory": True,
                }
            )
            trial = self._evaluate(candidate, "multi_gpu_placement")
            if (
                trial.status in ("ok", "unstable")
                and trial.score is not None
                and trial.pp is not None
                and trial.tg is not None
                and trial.score > self.incumbent_score
            ):
                self._set_incumbent(candidate, trial.pp, trial.tg, trial.score)

    def _joint_discovery(self) -> None:
        ladder = [0]
        if gpu_layer_cap(self.model, self.options) > 0 and self.model.moe and "ncmoe" in self.caps:
            ladder = list(reversed(moe_cpu_layer_candidates(self.model.n_layer)))
            if self.options.initial_cpu_moe is not None:
                initial = min(self.model.n_layer, self.options.initial_cpu_moe)
                ladder = [initial, *(v for v in ladder if v != initial)]
        warm_cap: int | None = None
        for ncmoe in ladder:
            boundary = self._discover_boundary(self.incumbent_config, ncmoe, warm_cap)
            self.boundaries.append(boundary)
            self._append({"type": "stage", "stage": "boundary", **dataclasses.asdict(boundary)})
            self._emit("boundary", **dataclasses.asdict(boundary))
            warm_cap = boundary.max_ok_ngl if _boundary_has_fit(boundary) else None
        self._bisect_minimum_spill()

    def _bisect_minimum_spill(self) -> None:
        if not self.model.moe or not self.boundaries:
            return
        cap = gpu_layer_cap(self.model, self.options)
        successful = [b for b in self.boundaries if _boundary_has_fit(b) and b.max_ok_ngl == cap]
        failing = [b for b in self.boundaries if _boundary_has_fit(b) and b.max_ok_ngl < cap]
        if not successful or not failing:
            return
        high = min(b.moe_cpu_layers for b in successful)
        lows = [b.moe_cpu_layers for b in failing if b.moe_cpu_layers < high]
        if not lows:
            return
        low = max(lows)
        probes = 0
        while high - low > 1 and self._can_execute():
            mid = (low + high) // 2
            cfg = dataclasses.replace(self.incumbent_config, gpu_layers=cap, moe_cpu_layers=mid)
            if self._probe(cfg, "boundary", None) == "ok":
                high = mid
            else:
                low = mid
            probes += 1
        boundary = FeasibilityBoundary(
            moe_cpu_layers=high,
            max_ok_ngl=cap,
            min_fail_ngl=None,
            probes=probes,
            cap_ngl=cap,
            cap_source="model" if self.options.max_gpu_layers is None else "cli",
        )
        if all(b.moe_cpu_layers != high for b in self.boundaries):
            self.boundaries.append(boundary)
            self._append({"type": "stage", "stage": "boundary", **dataclasses.asdict(boundary)})
        frontier = dataclasses.replace(self.incumbent_config, gpu_layers=cap, moe_cpu_layers=high)
        trial = self._evaluate(frontier, "joint_frontier")
        if (
            trial.status in ("ok", "unstable")
            and trial.score is not None
            and trial.pp is not None
            and trial.tg is not None
            and trial.score > self.incumbent_score
            and (
                self.options.ctx_size is None
                or self._probe(frontier, "context", self.options.ctx_size) == "ok"
            )
        ):
            self._set_incumbent(frontier, trial.pp, trial.tg, trial.score)

    def _discover_boundary(
        self, base: TrialConfig, ncmoe: int, warm_cap: int | None = None
    ) -> FeasibilityBoundary:
        hard_cap = gpu_layer_cap(self.model, self.options)
        cap = hard_cap
        cap_source = (
            "cli"
            if self.options.max_gpu_layers is not None
            and self.options.max_gpu_layers < self.model.ngl_all
            else "model"
        )
        if warm_cap is not None and warm_cap < cap:
            cap = min(cap, warm_cap)
            cap_source = "warm_start"
        base = dataclasses.replace(base, moe_cpu_layers=ncmoe)
        low: int | None = None
        failed, used = None, 0
        limit = 2 * max(1, max(1, self.model.ngl_all).bit_length()) + 4
        baseline_verified = (
            hasattr(self, "baseline")
            and self.baseline.kind == "defaults"
            and base.trial_id == self.default_config.trial_id
            and base.gpu_layers <= cap
        )
        if baseline_verified:
            low = base.gpu_layers
        else:
            guess = self.options.initial_gpu_layers
            if guess is None or guess > cap:
                guess = min(cap, 1)
            status = self._probe(dataclasses.replace(base, gpu_layers=guess), "boundary", None)
            used += 1
            if status == "ok":
                low = guess
            else:
                failed = guess
                if guess > 0 and used < limit:
                    status = self._probe(dataclasses.replace(base, gpu_layers=0), "boundary", None)
                    used += 1
                    if status == "ok":
                        low = 0
                    else:
                        failed = 0
        while low is not None and low < cap and failed is None and used < limit:
            guess = min(cap, max(low + 1, low * 2))
            cfg = dataclasses.replace(base, gpu_layers=guess)
            status = self._probe(cfg, "boundary", None)
            used += 1
            if status == "ok":
                low = guess
            else:
                failed = guess
        while low is not None and failed is not None and failed - low > 1 and used < limit:
            guess = (low + failed) // 2
            if self._probe(dataclasses.replace(base, gpu_layers=guess), "boundary", None) == "ok":
                low = guess
            else:
                failed = guess
            used += 1
        compliant: list[tuple[TrialConfig, float, float, float]] = []
        neighbors = (
            dict.fromkeys(max(0, low - delta) for delta in (0, 1, 2)) if low is not None else ()
        )
        for ngl in neighbors:
            cfg = dataclasses.replace(base, gpu_layers=ngl)
            trial = self._evaluate(cfg, "boundary_neighbor")
            if (
                trial.status in ("ok", "unstable")
                and trial.score is not None
                and trial.pp is not None
                and trial.tg is not None
            ):
                compliant.append((cfg, trial.pp, trial.tg, trial.score))
            if (
                trial.status in ("ok", "unstable")
                and trial.score is not None
                and trial.pp is not None
                and trial.tg is not None
                and trial.score > self.incumbent_score
            ):
                self._set_incumbent(cfg, trial.pp, trial.tg, trial.score)
        if self.incumbent_config.gpu_layers > hard_cap and compliant:
            cfg, pp, tg, score = max(compliant, key=lambda item: item[3])
            self._set_incumbent(cfg, pp, tg, score)
        return FeasibilityBoundary(
            moe_cpu_layers=ncmoe,
            max_ok_ngl=low if low is not None else 0,
            min_fail_ngl=failed,
            probes=used,
            cap_ngl=cap,
            cap_source=cap_source,
        )

    def _refine_joint(self) -> None:
        center = self.incumbent_config
        cap = gpu_layer_cap(self.model, self.options)
        offsets = [(0, -2), (0, -1), (0, 1), (0, 2)]
        if cap > 0 and self.model.moe and "ncmoe" in self.caps:
            offsets = [(-2, 0), (-1, 0), (1, 0), (2, 0), *offsets]
        for dc, dg in offsets:
            cfg = dataclasses.replace(
                center,
                gpu_layers=min(cap, max(0, center.gpu_layers + dg)),
                moe_cpu_layers=min(self.model.n_layer, max(0, center.moe_cpu_layers + dc)),
            )
            if not is_valid_config(cfg, hardware=self.hardware, model=self.model, llama=self.llama):
                continue
            if self._pruned_by(cfg) is not None:
                continue
            trial = self._evaluate(cfg, "joint_refine")
            if (
                trial.status in ("ok", "unstable")
                and trial.score is not None
                and trial.pp is not None
                and trial.tg is not None
                and trial.score > self.incumbent_score
            ):
                if (
                    self.options.ctx_size is not None
                    and self._probe(cfg, "context", self.options.ctx_size) != "ok"
                ):
                    continue
                self._set_incumbent(cfg, trial.pp, trial.tg, trial.score)
        direction = _sign(self.incumbent_config.moe_cpu_layers - center.moe_cpu_layers)
        if direction:
            self._hill_climb_moe(direction)

    def _hill_climb_moe(self, direction: int) -> None:
        steps = (1, 2, 4, 8)
        step_index = 0
        misses = 0
        started = self.executed_count
        while misses < 2 and self.executed_count - started < 12 and self._can_execute():
            current = self.incumbent_config
            value = min(
                self.model.n_layer,
                max(0, current.moe_cpu_layers + direction * steps[step_index]),
            )
            if value == current.moe_cpu_layers:
                break
            cfg = dataclasses.replace(current, moe_cpu_layers=value)
            before = self.executed_count
            trial = self._evaluate(cfg, "joint_refine")
            improved = bool(
                trial.status in ("ok", "unstable")
                and trial.score is not None
                and trial.pp is not None
                and trial.tg is not None
                and trial.score > self.incumbent_score * (1 + self.baseline.noise_floor_cv)
            )
            if (
                trial.score is not None
                and trial.pp is not None
                and trial.tg is not None
                and improved
                and (
                    self.options.ctx_size is None
                    or self._probe(cfg, "context", self.options.ctx_size) == "ok"
                )
            ):
                self._set_incumbent(cfg, trial.pp, trial.tg, trial.score)
                misses = 0
                step_index = min(step_index + 1, len(steps) - 1)
            else:
                misses += 1
                step_index = 0
            if self.executed_count == before:
                misses += 1

    def _probe(
        self,
        cfg: TrialConfig,
        purpose: str,
        ctx: int | None,
    ) -> str:
        probe_id = _probe_id(purpose, cfg, ctx)
        if probe_id in self.probes and self._hard_cap_ancestor(cfg) is None:
            return str(self.probes[probe_id]["status"])
        cap_ancestor = self._hard_cap_ancestor(cfg)
        if cap_ancestor is not None:
            record = {
                "type": "probe",
                "probe_id": probe_id,
                "purpose": purpose,
                "config": cfg.to_dict(),
                "ctx": ctx,
                "status": "pruned",
                "pruned_from": cap_ancestor,
            }
            self._append(record)
            self.probes[probe_id] = record
            return "pruned"
        ancestor = self._pruned_by(cfg)
        if ancestor is not None and purpose == "boundary":
            record = {
                "type": "probe",
                "probe_id": probe_id,
                "purpose": purpose,
                "config": cfg.to_dict(),
                "ctx": ctx,
                "status": "pruned",
                "pruned_from": ancestor,
            }
            self._append(record)
            self.probes[probe_id] = record
            return "pruned"
        if not self._can_execute():
            raise _BudgetExhaustedError
        argv = (
            bench.build_context_probe_argv(
                bench_path=self.llama.bench_path,
                model_path=self.model.path,
                ctx=ctx,
                config=cfg,
                capabilities=self.caps,
            )
            if ctx is not None
            else bench.build_bench_argv(
                bench_path=self.llama.bench_path,
                model_path=self.model.path,
                pp=self.options.pp,
                tg=self.options.tg,
                reps=1,
                config=cfg,
                capabilities=self.caps,
                depth=self.options.depth,
            )
        )
        probe_dir = self.session.probe_dir(probe_id)
        label = f"{purpose} {self._config_label(cfg)}"
        result, observation = self._run_child(
            kind="probe",
            label=label,
            run_id=probe_id,
            argv=argv,
            timeout_s=self.trial_timeout,
            stdout_path=probe_dir / "stdout.json",
            stderr_path=probe_dir / "stderr.log",
        )
        self.executed_count += 1
        measured = self._classify(result, 1)
        record = {
            "type": "probe",
            "probe_id": probe_id,
            "purpose": purpose,
            "config": cfg.to_dict(),
            "ctx": ctx,
            "status": measured.status,
            "pattern": measured.oom_pattern,
            "estimate_reason": estimate_reason(
                config=cfg,
                model=self.model,
                ctx=ctx,
                vram_reserve_mb=self.vram_reserve_mb,
                vram_total_mb=total_vram_mb(self.hardware, multi_gpu=self.options.multi_gpu),
            ),
        }
        self._write_command(
            f"probes/{probe_id}", argv, result, 1, self._record_load(cfg.threads), observation
        )
        self._append(record)
        self._emit_exec_end(label, probe_id, result, measured)
        self.probes[probe_id] = record
        if measured.status in ("oom", "gpu_resource"):
            self.oom_points.append((cfg.to_dict(), probe_id))
        self._cooldown()
        return measured.status

    def _sweep(self, dim: str) -> None:
        candidates = candidates_for(
            dim,
            hardware=self.hardware,
            model=self.model,
            llama=self.llama,
            options=self.options,
            incumbent=self.incumbent_config,
        )
        if not candidates:
            return

        best_cfg: TrialConfig | None = None
        best_score = self.incumbent_score
        best_pp = self.incumbent_pp
        best_tg = self.incumbent_tg
        coverage: dict[str, Any] = {
            "candidates": list(candidates),
            "executed": [],
            "cached_hit": [],
            "pruned": [],
            "skipped": [],
        }

        ordered_configs = [
            mutate(self.incumbent_config, dim, value)
            for value in _order_candidates(dim, candidates)
        ]
        batch_results: dict[str, _Trial] = {}
        if self.options.batched_trials and dim in {
            "threads",
            "threads_batch",
            "mmap",
            "ubatch",
            "batch",
        }:
            batchable = [
                cfg
                for cfg in ordered_configs
                if cfg.trial_id not in self.known
                and cfg.to_dict() != self.incumbent_config.to_dict()
                and is_valid_config(cfg, hardware=self.hardware, model=self.model, llama=self.llama)
                and self._pruned_by(cfg) is None
                and (
                    dim not in {"ubatch", "batch"}
                    or dimension_value(cfg, dim) <= dimension_value(self.incumbent_config, dim)
                )
            ]
            if len(batchable) > 1:
                batch_results = self._execute_batch(batchable, dim)

        for cfg in ordered_configs:
            value = dimension_value(cfg, dim)
            if cfg.to_dict() == self.incumbent_config.to_dict():
                coverage["skipped"].append({"value": value, "reason": "incumbent"})
                continue
            if not is_valid_config(cfg, hardware=self.hardware, model=self.model, llama=self.llama):
                coverage["skipped"].append({"value": value, "reason": "invalid"})
                continue
            was_known = cfg.trial_id in self.known and cfg.trial_id not in batch_results
            if cfg.trial_id not in self.known:
                ancestor = self._pruned_by(cfg)
                if ancestor is not None:
                    self._journal_pruned(cfg, dim, ancestor)
                    coverage["pruned"].append({"value": value, "ancestor": ancestor})
                    continue
            try:
                trial = batch_results.get(cfg.trial_id) or self._evaluate(cfg, dim)
            except _BudgetExhaustedError:
                coverage["skipped"].append({"value": value, "reason": "budget"})
                self.coverage[dim] = coverage
                raise
            coverage["cached_hit" if was_known else "executed"].append(value)
            self.coverage[dim] = coverage
            pair_allows = True
            if dim == "mmap" and trial.status in ("ok", "unstable"):
                pair_allows = self._pair_check_mmap(self.incumbent_config, cfg, trial)
            if trial.status in ("ok", "unstable") and dim in {
                "ubatch",
                "batch",
                "kv_offload",
                "cache_type_k",
                "cache_type_v",
            }:
                validation = self._probe(
                    cfg,
                    "context" if self.options.ctx_size is not None else "boundary",
                    self.options.ctx_size,
                )
                if validation != "ok":
                    self._append(
                        {
                            "type": "stage",
                            "stage": "revalidation_rejected",
                            "dim": dim,
                            "config": cfg.to_dict(),
                            "reason": validation,
                        }
                    )
                    continue
            if (
                trial.status in ("ok", "unstable")
                and trial.score is not None
                and trial.pp is not None
                and trial.tg is not None
                and pair_allows
                and trial.score > best_score
            ):
                best_cfg, best_score, best_pp, best_tg = cfg, trial.score, trial.pp, trial.tg

        if best_cfg is not None and best_score > self.incumbent_score * (
            1 + self.baseline.noise_floor_cv
        ):
            self._set_incumbent(best_cfg, best_pp, best_tg, best_score)
        self.coverage[dim] = coverage

    def _pairwise_refine(self) -> None:
        ubatches = candidates_for(
            "ubatch",
            hardware=self.hardware,
            model=self.model,
            llama=self.llama,
            options=self.options,
            incumbent=self.incumbent_config,
        )
        batches = candidates_for(
            "batch",
            hardware=self.hardware,
            model=self.model,
            llama=self.llama,
            options=self.options,
            incumbent=self.incumbent_config,
        )
        ub_center = _neighbor_values(ubatches, self.incumbent_config.ubatch)
        b_center = _neighbor_values(batches, self.incumbent_config.batch)
        tried = 0
        for ubatch in ub_center:
            for batch in b_center:
                if tried >= 8 or not self._can_execute():
                    return
                cfg = dataclasses.replace(self.incumbent_config, ubatch=ubatch, batch=batch)
                if cfg.trial_id in self.known or not is_valid_config(
                    cfg, hardware=self.hardware, model=self.model, llama=self.llama
                ):
                    continue
                trial = self._evaluate(cfg, "pairwise")
                tried += 1
                if (
                    trial.status in ("ok", "unstable")
                    and trial.score is not None
                    and trial.pp is not None
                    and trial.tg is not None
                    and trial.score > self.incumbent_score * (1 + self.baseline.noise_floor_cv)
                ):
                    self._set_incumbent(cfg, trial.pp, trial.tg, trial.score)
        self._threads_moe_cross()

    def _refine_thread(self, dim: str) -> None:
        candidates = candidates_for(
            dim,
            hardware=self.hardware,
            model=self.model,
            llama=self.llama,
            options=self.options,
            incumbent=self.incumbent_config,
        )
        current = getattr(self.incumbent_config, dim)
        if current is None:
            return
        ordered = sorted({int(value) for value in candidates} | {int(current)})
        index = ordered.index(int(current))
        refinements: list[int] = []
        if index > 0:
            refinements.append((ordered[index - 1] + int(current)) // 2)
        if index + 1 < len(ordered):
            refinements.append((ordered[index + 1] + int(current)) // 2)
        for value in dict.fromkeys(refinements):
            if value == current or not self._can_execute():
                continue
            cfg = mutate(self.incumbent_config, dim, value)
            if cfg.trial_id in self.known:
                continue
            trial = self._evaluate(cfg, f"{dim}_refine")
            if (
                trial.status in ("ok", "unstable")
                and trial.score is not None
                and trial.pp is not None
                and trial.tg is not None
                and trial.score > self.incumbent_score * (1 + self.baseline.noise_floor_cv)
            ):
                self._set_incumbent(cfg, trial.pp, trial.tg, trial.score)

    def _threads_moe_cross(self) -> None:
        if self.incumbent_config.moe_cpu_layers <= 0:
            return
        threads = candidates_for(
            "threads",
            hardware=self.hardware,
            model=self.model,
            llama=self.llama,
            options=self.options,
            incumbent=self.incumbent_config,
        )
        neighbors = _neighbor_values(threads, self.incumbent_config.threads)
        thread_values = [value for value in neighbors if value != self.incumbent_config.threads]
        tried = 0
        for thread in thread_values:
            for delta in (-1, 1):
                if tried >= 4 or not self._can_execute():
                    return
                ncmoe = min(
                    self.model.n_layer, max(0, self.incumbent_config.moe_cpu_layers + delta)
                )
                cfg = dataclasses.replace(
                    self.incumbent_config, threads=thread, moe_cpu_layers=ncmoe
                )
                if cfg.trial_id in self.known:
                    continue
                trial = self._evaluate(cfg, "pairwise")
                tried += 1
                if (
                    trial.status in ("ok", "unstable")
                    and trial.score is not None
                    and trial.pp is not None
                    and trial.tg is not None
                    and trial.score > self.incumbent_score * (1 + self.baseline.noise_floor_cv)
                ):
                    self._set_incumbent(cfg, trial.pp, trial.tg, trial.score)

    def _ot_refine(self) -> None:
        current = self.incumbent_config
        if not (
            self.options.ot_search
            and self.model.moe
            and "ot" in self.caps
            and current.moe_cpu_layers > 0
        ):
            return
        for spec in _ot_specs(self.model.n_layer, current.moe_cpu_layers):
            if not self._can_execute():
                break
            cfg = dataclasses.replace(current, moe_cpu_layers=0, ot_spec=spec)
            trial = self._evaluate(cfg, "ot_search")
            if (
                trial.status in ("ok", "unstable")
                and trial.score is not None
                and trial.pp is not None
                and trial.tg is not None
                and trial.score > self.incumbent_score * (1 + self.baseline.noise_floor_cv)
                and (
                    self.options.ctx_size is None
                    or self._probe(cfg, "context", self.options.ctx_size) == "ok"
                )
            ):
                self._set_incumbent(cfg, trial.pp, trial.tg, trial.score)
                current = cfg

    def _execute_batch(self, configs: list[TrialConfig], dim: str) -> dict[str, _Trial]:
        if any(self._hard_cap_ancestor(config) is not None for config in configs):
            return {}
        if len(configs) > self._trial_limit() - self.executed_count:
            return {}
        batch_id = hashlib.sha256(
            "".join(config.trial_id for config in configs).encode()
        ).hexdigest()[:16]
        argv = bench.build_bench_batch_argv(
            bench_path=self.llama.bench_path,
            model_path=self.model.path,
            pp=self.options.pp,
            tg=self.options.tg,
            reps=self.options.reps_search,
            configs=configs,
            capabilities=self.caps,
            depth=self.options.depth,
        )
        batch_dir = self.session.batch_dir(batch_id)
        label = f"{dim} batch ({len(configs)} combos)"
        result, observation = self._run_child(
            kind="trial",
            label=label,
            run_id=batch_id,
            argv=argv,
            timeout_s=self.trial_timeout * len(configs),
            stdout_path=batch_dir / "stdout.json",
            stderr_path=batch_dir / "stderr.log",
        )
        self._write_command(
            f"batches/{batch_id}",
            argv,
            result,
            self.options.reps_search,
            self._record_load(max(config.threads for config in configs)),
            observation,
        )
        if result.exit_code != 0 or result.timed_out or result.stdout.truncated:
            status = (
                "timeout"
                if result.timed_out
                else "parse_error"
                if result.stdout.truncated
                else "crash"
            )
            self._emit(
                "exec_end",
                label=label,
                id=batch_id,
                status=status,
                wall_s=result.wall_s,
                pp=None,
                tg=None,
                score=None,
            )
            return {}
        try:
            samples = bench.parse_bench_output_multi(result.stdout.path.read_bytes())
        except bench.BenchParseError:
            self._emit(
                "exec_end",
                label=label,
                id=batch_id,
                status="parse_error",
                wall_s=result.wall_s,
                pp=None,
                tg=None,
                score=None,
            )
            return {}
        attributed: list[tuple[TrialConfig, bench.BenchSample]] = []
        used_samples: set[int] = set()
        for cfg in configs:
            matches = [
                (index, item)
                for index, item in enumerate(samples)
                if index not in used_samples and _sample_matches(item.config_fields, cfg)
            ]
            if len(matches) != 1:
                return {}
            index, sample = matches[0]
            used_samples.add(index)
            attributed.append((cfg, sample))
        if len(used_samples) != len(samples):
            return {}
        matched: dict[str, _Trial] = {}
        for cfg, sample in attributed:
            pp = stats.sample_stats(sample.pp_avg, sample.pp_stddev, self.options.reps_search)
            tg = stats.sample_stats(sample.tg_avg, sample.tg_stddev, self.options.reps_search)
            score = self._score(pp.mean, tg.mean)
            status = "unstable" if stats.is_unstable(pp, tg) else "ok"
            record = {
                "type": "trial",
                "trial_id": cfg.trial_id,
                "config": cfg.to_dict(),
                "status": status,
                "pp_mean": pp.mean,
                "tg_mean": tg.mean,
                "score": score,
                "flags": list(cfg.bench_args(self.caps)),
                "dim": dim,
                "oom_pattern": None,
                "pruned_from": None,
                "batch_id": batch_id,
                "evidence": f"batches/{batch_id}",
            }
            self._append(record)
            self.known[cfg.trial_id] = record
            matched[cfg.trial_id] = _Trial(status, pp.mean, tg.mean, score)
            self._emit(
                "exec_end",
                label=self._config_label(cfg),
                id=cfg.trial_id,
                status=status,
                wall_s=result.wall_s,
                pp=pp.mean,
                tg=tg.mean,
                score=score,
            )
        self.executed_count += len(configs)
        return matched

    def _pair_check_mmap(
        self, incumbent: TrialConfig, challenger: TrialConfig, challenger_trial: _Trial
    ) -> bool:
        if not self._can_execute():
            return False
        pair_no = sum(1 for e in self.session.entries if e.get("type") == "pair_check") + 1
        argv = bench.build_bench_argv(
            bench_path=self.llama.bench_path,
            model_path=self.model.path,
            pp=self.options.pp,
            tg=self.options.tg,
            reps=self.options.reps_search,
            config=incumbent,
            capabilities=self.caps,
            depth=self.options.depth,
        )
        pair_dir = self.session.trial_dir(incumbent.trial_id) / f"pair-{pair_no}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"{incumbent.trial_id}-pair-{pair_no}"
        label = f"pair_check {self._config_label(incumbent)}"
        result, observation = self._run_child(
            kind="pair_check",
            label=label,
            run_id=run_id,
            argv=argv,
            timeout_s=self.trial_timeout,
            stdout_path=pair_dir / "stdout.json",
            stderr_path=pair_dir / "stderr.log",
        )
        self.executed_count += 1
        measured = self._classify(result, self.options.reps_search)
        pair_score = None
        if measured.status == "ok" and measured.pp is not None and measured.tg is not None:
            pair_score = self._score(measured.pp.mean, measured.tg.mean)
        accepted = bool(
            pair_score is not None
            and challenger_trial.score is not None
            and challenger_trial.score > pair_score * (1 + self.baseline.noise_floor_cv)
        )
        rel = f"trials/{incumbent.trial_id}/pair-{pair_no}"
        self._write_command(
            rel,
            argv,
            result,
            self.options.reps_search,
            self._record_load(incumbent.threads),
            observation,
        )
        self._append(
            {
                "type": "pair_check",
                "dim": "mmap",
                "incumbent_id": incumbent.trial_id,
                "challenger_id": challenger.trial_id,
                "pp": measured.pp.mean if measured.pp is not None else None,
                "tg": measured.tg.mean if measured.tg is not None else None,
                "accepted": accepted,
                "evidence": rel,
            }
        )
        self._emit_exec_end(label, run_id, result, measured, pair_score)
        self._cooldown()
        return accepted

    def _pruned_by(self, cfg: TrialConfig) -> str | None:
        """Monotone OOM pruning ancestor for `cfg`, or None (DESIGN §10 step 4).

        An OOM at (gpu_layers=g, moe_cpu_layers=c) prunes candidates with
        gpu_layers >= g and moe_cpu_layers <= c only when every other
        (memory-relevant) field is equal -- the most conservative reading.
        """
        cfg_dict = cfg.to_dict()
        for oom_cfg, trial_id in self.oom_points:
            if (
                cfg.gpu_layers >= int(oom_cfg["gpu_layers"])
                and cfg.moe_cpu_layers <= int(oom_cfg["moe_cpu_layers"])
                and _other_fields(cfg_dict) == _other_fields(oom_cfg)
            ):
                return trial_id
        return None

    def _hard_cap_ancestor(self, cfg: TrialConfig) -> str | None:
        cap = gpu_layer_cap(self.model, self.options)
        return f"max_gpu_layers:{cap}" if cfg.gpu_layers > cap else None

    def _journal_pruned(self, cfg: TrialConfig, dim: str, ancestor: str) -> None:
        if cfg.trial_id in self.known:
            return
        record = {
            "type": "trial",
            "trial_id": cfg.trial_id,
            "config": cfg.to_dict(),
            "status": "pruned",
            "pp_mean": None,
            "tg_mean": None,
            "score": None,
            "flags": list(cfg.bench_args(self.caps)),
            "dim": dim,
            "oom_pattern": None,
            "pruned_from": ancestor,
        }
        self._append(record)
        self.known[cfg.trial_id] = record

    # -- trial evaluation ----------------------------------------------

    def _evaluate(self, cfg: TrialConfig, dim: str) -> _Trial:
        cap_ancestor = self._hard_cap_ancestor(cfg)
        if cap_ancestor is not None:
            self._journal_pruned(cfg, dim, cap_ancestor)
            return _Trial(status="pruned", pp=None, tg=None, score=None)
        known = self.known.get(cfg.trial_id)
        if known is not None:
            return _Trial(
                status=known["status"],
                pp=known.get("pp_mean"),
                tg=known.get("tg_mean"),
                score=known.get("score"),
            )
        return self._execute_trial(cfg, dim)

    def _execute_trial(self, cfg: TrialConfig, dim: str) -> _Trial:
        if not self._can_execute():
            raise _BudgetExhaustedError
        argv = bench.build_bench_argv(
            bench_path=self.llama.bench_path,
            model_path=self.model.path,
            pp=self.options.pp,
            tg=self.options.tg,
            reps=self.options.reps_search,
            config=cfg,
            capabilities=self.caps,
            depth=self.options.depth,
        )
        trial_dir = self.session.trial_dir(cfg.trial_id)
        load = self._record_load(cfg.threads)
        label = f"{dim} {self._config_label(cfg)}"
        result, observation = self._run_child(
            kind="trial",
            label=label,
            run_id=cfg.trial_id,
            argv=argv,
            timeout_s=self.trial_timeout,
            stdout_path=trial_dir / "stdout.json",
            stderr_path=trial_dir / "stderr.log",
        )
        self.executed_count += 1
        measured = self._classify(result, self.options.reps_search)
        primary_result = result

        self._write_command(
            f"trials/{cfg.trial_id}", argv, result, self.options.reps_search, load, observation
        )
        thermal = self._thermal_retest(
            kind="trial",
            trial_id=cfg.trial_id,
            run_id=cfg.trial_id,
            label=label,
            argv=argv,
            timeout_s=self.trial_timeout,
            reps=self.options.reps_search,
            threads=cfg.threads,
            retry_dir=trial_dir / "thermal-retry",
            retry_rel_dir=f"trials/{cfg.trial_id}/thermal-retry",
            result=result,
            observation=observation,
            measured=measured,
        )
        measured = thermal.measured
        primary_exec_emitted = thermal.contaminated

        stability: _StabilityRerun | None = None
        if measured.status == "ok" and stats.is_unstable(measured.pp, measured.tg):
            if not primary_exec_emitted:
                self._emit_exec_end(label, cfg.trial_id, primary_result, measured)
                primary_exec_emitted = True
            stability = self._rerun_for_stability(cfg, measured)
            measured = stability.measured

        pp_mean = measured.pp.mean if measured.pp is not None else None
        tg_mean = measured.tg.mean if measured.tg is not None else None
        primary_thermal_rejected = thermal.contaminated and (
            not thermal.retried or thermal.retry_contaminated is True
        )
        thermal_rejected = primary_thermal_rejected or bool(
            stability is not None and stability.thermal_rejected
        )
        thermally_contaminated = thermal.contaminated or bool(
            stability is not None and stability.contaminated
        )
        thermal_retried = thermal.retried or bool(stability is not None and stability.retried)
        retry_states = [
            state
            for state in (
                thermal.retry_contaminated,
                stability.retry_contaminated if stability is not None else None,
            )
            if state is not None
        ]
        thermal_retry_contaminated = True if any(retry_states) else False if retry_states else None
        score = (
            self._score(pp_mean, tg_mean)
            if measured.status in ("ok", "unstable")
            and not thermal_rejected
            and pp_mean is not None
            and tg_mean is not None
            else None
        )
        if measured.status in ("oom", "gpu_resource"):
            self.oom_points.append((cfg.to_dict(), cfg.trial_id))

        record = {
            "type": "trial",
            "trial_id": cfg.trial_id,
            "config": cfg.to_dict(),
            "status": measured.status,
            "pp_mean": pp_mean,
            "tg_mean": tg_mean,
            "score": score,
            "flags": list(cfg.bench_args(self.caps)),
            "dim": dim,
            "oom_pattern": measured.oom_pattern,
            "pruned_from": None,
            "thermally_contaminated": thermally_contaminated,
            "thermal_retried": thermal_retried,
            "thermal_retry_contaminated": thermal_retry_contaminated,
            "thermal_rejected": thermal_rejected,
        }
        self._append(record)
        self.known[cfg.trial_id] = record
        if not primary_exec_emitted:
            self._emit_exec_end(label, cfg.trial_id, primary_result, measured, score)
        if not thermal.contaminated:
            self._cooldown()
        return _Trial(measured.status, pp_mean, tg_mean, score)

    def _rerun_for_stability(self, cfg: TrialConfig, first: _Measured) -> _StabilityRerun:
        if not self._can_execute():
            return _StabilityRerun(
                dataclasses.replace(first, status="unstable"), False, False, None, False
            )
        argv = bench.build_bench_argv(
            bench_path=self.llama.bench_path,
            model_path=self.model.path,
            pp=self.options.pp,
            tg=self.options.tg,
            reps=self.options.reps_search,
            config=cfg,
            capabilities=self.caps,
            depth=self.options.depth,
        )
        rerun_dir = self.session.trial_dir(cfg.trial_id) / "rerun"
        rerun_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"{cfg.trial_id}-rerun"
        label = f"rerun {self._config_label(cfg)}"
        result, observation = self._run_child(
            kind="rerun",
            label=label,
            run_id=run_id,
            argv=argv,
            timeout_s=self.trial_timeout,
            stdout_path=rerun_dir / "stdout.json",
            stderr_path=rerun_dir / "stderr.log",
        )
        self.executed_count += 1
        second = self._classify(result, self.options.reps_search)
        self._write_command(
            f"trials/{cfg.trial_id}/rerun",
            argv,
            result,
            self.options.reps_search,
            self._record_load(cfg.threads),
            observation,
        )
        self._append(
            {
                "type": "stability_rerun",
                "trial_id": cfg.trial_id,
                "run_id": run_id,
                "status": second.status,
                "pp_mean": second.pp.mean if second.pp is not None else None,
                "tg_mean": second.tg.mean if second.tg is not None else None,
                "evidence": f"trials/{cfg.trial_id}/rerun",
            }
        )
        thermal = self._thermal_retest(
            kind="rerun",
            trial_id=cfg.trial_id,
            run_id=run_id,
            label=label,
            argv=argv,
            timeout_s=self.trial_timeout,
            reps=self.options.reps_search,
            threads=cfg.threads,
            retry_dir=rerun_dir / "thermal-retry",
            retry_rel_dir=f"trials/{cfg.trial_id}/rerun/thermal-retry",
            result=result,
            observation=observation,
            measured=second,
        )
        second = thermal.measured
        if not thermal.contaminated:
            self._emit_exec_end(label, run_id, result, second)
            self._cooldown()
        thermal_rejected = thermal.contaminated and (
            not thermal.retried or thermal.retry_contaminated is True
        )
        if thermal_rejected:
            return _StabilityRerun(
                second,
                thermal.contaminated,
                thermal.retried,
                thermal.retry_contaminated,
                True,
            )
        if second.status == "ok" and not stats.is_unstable(second.pp, second.tg):
            selected = second
        else:
            selected = dataclasses.replace(first, status="unstable")
        return _StabilityRerun(
            selected,
            thermal.contaminated,
            thermal.retried,
            thermal.retry_contaminated,
            False,
        )

    # -- classification -------------------------------------------------

    def _classify(self, result: executor.ExecResult, reps: int) -> _Measured:
        if result.timed_out:
            return _Measured("timeout", None, None, None, None)
        if result.stdout.truncated:
            return _Measured("parse_error", None, None, None, None)
        if result.exit_code == 0:
            raw = result.stdout.path.read_bytes()
            try:
                sample = bench.parse_bench_output(raw)
            except bench.BenchParseError:
                return _Measured("parse_error", None, None, None, None)
            pp = stats.sample_stats(sample.pp_avg, sample.pp_stddev, reps)
            tg = stats.sample_stats(sample.tg_avg, sample.tg_stddev, reps)
            return _Measured("ok", pp, tg, sample.pp_entry, None)
        stderr_text = result.stderr.path.read_text(encoding="utf-8", errors="replace")
        classification = bench.classify_failure(stderr_text)
        if classification is not None:
            return _Measured(
                classification, None, None, None, bench.failure_pattern(stderr_text, classification)
            )
        return _Measured("crash", None, None, None, None)

    def _validate_recommendation(self) -> None:
        if self.options.ctx_size is None:
            return
        candidates = self._context_candidates(self.incumbent_config)
        for cfg in candidates:
            try:
                status = self._probe(cfg, "context", self.options.ctx_size)
            except _BudgetExhaustedError:
                self.context_validation = {
                    "ctx": self.options.ctx_size,
                    "status": "skipped",
                    "evidence": None,
                    "reason": _budget_reason(self),
                }
                warning = f"required context validation skipped because {_budget_reason(self)}"
                if warning not in self.extra_warnings:
                    self.extra_warnings.append(warning)
                    self._emit("warning", message=warning)
                return
            if status == "ok":
                self.validated_configs.add(cfg.trial_id)
                self.context_validation = {
                    "ctx": self.options.ctx_size,
                    "status": "ok",
                    "evidence": f"probes/{_probe_id('context', cfg, self.options.ctx_size)}",
                }
                if cfg.trial_id != self.incumbent_config.trial_id:
                    trial = self._evaluate(cfg, "context_safe")
                    if (
                        trial.status in ("ok", "unstable")
                        and trial.pp is not None
                        and trial.tg is not None
                    ):
                        self._set_incumbent(cfg, trial.pp, trial.tg, trial.score or 0.0)
                return
        self.context_validation = {
            "ctx": self.options.ctx_size,
            "status": "failed",
            "evidence": None,
        }

    def _context_candidates(self, base: TrialConfig) -> list[TrialConfig]:
        cap = gpu_layer_cap(self.model, self.options)
        candidates = [base] if base.gpu_layers <= cap else []
        for boundary in sorted(
            (boundary for boundary in self.boundaries if _boundary_has_fit(boundary)),
            key=lambda b: b.max_ok_ngl,
            reverse=True,
        ):
            candidates.extend(
                dataclasses.replace(base, gpu_layers=ngl, moe_cpu_layers=boundary.moe_cpu_layers)
                for ngl in range(boundary.max_ok_ngl, -1, -1)
            )
        result: list[TrialConfig] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.trial_id not in seen:
                result.append(candidate)
                seen.add(candidate.trial_id)
        return result

    def _context_envelope_candidates(self, base: TrialConfig) -> list[TrialConfig]:
        """Return placement fallbacks with opt-in, least-lossy KV tiers interleaved."""
        result: list[TrialConfig] = []
        seen: set[str] = set()
        for placement in self._context_candidates(base):
            variants = [placement]
            if self.options.allow_lossy:
                ranks = {"f16": 0, "q8_0": 1, "q4_0": 2}
                k_rank = ranks.get(placement.cache_type_k)
                v_rank = ranks.get(placement.cache_type_v)
                can_k = "ctk" in self.caps and k_rank is not None
                can_v = "ctv" in self.caps and placement.flash_attn and v_rank is not None
                for tier in ("q8_0", "q4_0"):
                    tier_rank = ranks[tier]
                    k_variant = placement
                    if can_k and k_rank is not None and tier_rank >= k_rank:
                        k_variant = dataclasses.replace(placement, cache_type_k=tier)
                        variants.append(k_variant)
                    if can_v and v_rank is not None and tier_rank >= v_rank:
                        if can_k and k_rank is not None and tier_rank >= k_rank:
                            variants.append(dataclasses.replace(k_variant, cache_type_v=tier))
                        else:
                            variants.append(dataclasses.replace(placement, cache_type_v=tier))
            for candidate in variants:
                if candidate.trial_id not in seen:
                    result.append(candidate)
                    seen.add(candidate.trial_id)
        return result

    # -- confirmation and finalization ---------------------------------

    def _run_depth_profile(self, config: TrialConfig) -> None:
        depths = self.options.depth_profile
        if not depths:
            return
        rows: list[dict[str, Any]] = []
        existing = {
            int(entry["depth"]): entry
            for entry in self.session.entries
            if entry.get("type") == "depth_profile_run"
            and entry.get("trial_id") == config.trial_id
            and isinstance(entry.get("depth"), int)
        }
        for depth in depths:
            entry = existing.get(depth)
            if entry is None:
                if not self._can_execute():
                    if self._find_stage("depth_profile_skipped") is None:
                        self._append(
                            {"type": "stage", "stage": "depth_profile_skipped", "reason": "budget"}
                        )
                    break
                self._emit("depth_profile", depth=depth, trial_id=config.trial_id)
                argv = bench.build_bench_argv(
                    bench_path=self.llama.bench_path,
                    model_path=self.model.path,
                    pp=self.options.pp,
                    tg=self.options.tg,
                    reps=self.options.reps_search,
                    config=config,
                    capabilities=self.caps,
                    depth=depth,
                )
                run_dir = self.session.trial_dir(config.trial_id) / f"depth-{depth}"
                run_dir.mkdir(parents=True, exist_ok=True)
                result, observation = self._run_child(
                    kind="depth_profile",
                    label=f"depth profile {depth} {self._config_label(config)}",
                    run_id=f"{config.trial_id}-depth-{depth}",
                    argv=argv,
                    timeout_s=self.trial_timeout,
                    stdout_path=run_dir / "stdout.json",
                    stderr_path=run_dir / "stderr.log",
                )
                self.executed_count += 1
                measured = self._classify(result, self.options.reps_search)
                self._write_command(
                    f"trials/{config.trial_id}/depth-{depth}",
                    argv,
                    result,
                    self.options.reps_search,
                    self._record_load(config.threads),
                    observation,
                )
                entry = {
                    "type": "depth_profile_run",
                    "trial_id": config.trial_id,
                    "depth": depth,
                    "status": measured.status,
                    "pp_mean": measured.pp.mean if measured.pp is not None else None,
                    "tg_mean": measured.tg.mean if measured.tg is not None else None,
                }
                self._append(entry)
            if entry.get("status") == "ok" and entry.get("pp_mean") is not None:
                rows.append(
                    {"d": depth, "pp": float(entry["pp_mean"]), "tg": float(entry["tg_mean"])}
                )
        self.depth_profile = {"trial_id": config.trial_id, "rows": rows}

    def _record_context_pruned(self, config: TrialConfig, ctx: int, source_ctx: int) -> None:
        probe_id = _probe_id("context", config, ctx)
        if probe_id in self.probes:
            return
        record = {
            "type": "probe",
            "probe_id": probe_id,
            "purpose": "context",
            "config": config.to_dict(),
            "ctx": ctx,
            "status": "pruned",
            "pruned_from": f"context:{source_ctx}",
        }
        self._append(record)
        self.probes[probe_id] = record

    def _run_context_envelope(
        self, config: TrialConfig, lossless_winner: dict[str, Any] | None
    ) -> None:
        if not self.options.ctx_ladder or self.options.ctx_size is None:
            return
        required = {
            "ctx": self.options.ctx_size,
            "status": (self.context_validation or {}).get("status", "skipped"),
            "config": config.to_dict(),
            "fallback_config": None,
            "evidence": (self.context_validation or {}).get("evidence"),
        }
        rows = [required]
        original_failed_at: int | None = None
        fallback: TrialConfig | None = None
        total_start = self.executed_count
        truncated = False
        budget_exhausted = False
        self._emit("envelope", ctx=self.options.ctx_size, status=required["status"])
        for ctx in self.options.ctx_ladder:
            if budget_exhausted:
                rows.append(
                    {
                        "ctx": ctx,
                        "status": "skipped",
                        "config": None,
                        "fallback_config": None,
                        "evidence": None,
                    }
                )
                continue
            if self.executed_count - total_start >= 24:
                truncated = True
                rows.append(
                    {
                        "ctx": ctx,
                        "status": "skipped",
                        "config": None,
                        "fallback_config": None,
                        "evidence": None,
                    }
                )
                continue
            rung_start = self.executed_count
            primary_status = "pruned" if original_failed_at is not None else "skipped"
            evidence: str | None = None
            if original_failed_at is not None:
                self._record_context_pruned(config, ctx, original_failed_at)
            else:
                try:
                    probe_status = self._probe(config, "context", ctx)
                except _BudgetExhaustedError:
                    truncated = True
                    budget_exhausted = True
                    rows.append(
                        {
                            "ctx": ctx,
                            "status": "skipped",
                            "config": None,
                            "fallback_config": None,
                            "evidence": None,
                        }
                    )
                    continue
                evidence = f"probes/{_probe_id('context', config, ctx)}"
                primary_status = "ok" if probe_status == "ok" else "failed"
                if primary_status == "failed":
                    original_failed_at = ctx

            passing = config if primary_status == "ok" else None
            if passing is None:
                candidates = self._context_envelope_candidates(fallback or config)
                for candidate in candidates:
                    if candidate.trial_id == config.trial_id:
                        continue
                    if (
                        self.executed_count - rung_start >= 8
                        or self.executed_count - total_start >= 24
                    ):
                        truncated = True
                        break
                    try:
                        status = self._probe(candidate, "context", ctx)
                    except _BudgetExhaustedError:
                        truncated = True
                        budget_exhausted = True
                        break
                    if status == "ok":
                        passing = candidate
                        fallback = candidate
                        evidence = f"probes/{_probe_id('context', candidate, ctx)}"
                        break
            row = {
                "ctx": ctx,
                "status": primary_status,
                "config": config.to_dict(),
                "fallback_config": (
                    passing.to_dict()
                    if passing is not None and passing.trial_id != config.trial_id
                    else None
                ),
                "evidence": evidence,
            }
            rows.append(row)
            self._emit("envelope", ctx=ctx, status=primary_status)

        if truncated and self._find_stage("envelope_truncated") is None:
            self._append(
                {
                    "type": "stage",
                    "stage": "envelope_truncated",
                    "reason": "budget",
                    "ctx": rows[-1]["ctx"],
                }
            )
        self._run_lossless_context_checks(rows, config, lossless_winner)
        self.context_envelope = rows

    def _max_validated_context(self) -> int | None:
        """Largest context validated for the exact recommended configuration."""
        validated: list[int] = []
        if self.context_validation and self.context_validation.get("status") == "ok":
            validated.append(int(self.context_validation["ctx"]))
        if self.context_envelope:
            validated.extend(
                int(row["ctx"])
                for row in self.context_envelope
                if row.get("status") == "ok" and row.get("fallback_config") is None
            )
        return max(validated) if validated else None

    def _run_lossless_context_checks(
        self,
        rows: list[dict[str, Any]],
        config: TrialConfig,
        lossless_winner: dict[str, Any] | None,
    ) -> None:
        if lossless_winner is None or not self.options.allow_lossy:
            return
        lossless = TrialConfig.from_dict(lossless_winner["config"])
        if lossless.trial_id == config.trial_id:
            return
        passing_ctx = [
            int(row["ctx"])
            for row in rows
            if row["status"] == "ok" or row.get("fallback_config") is not None
        ]
        targets = list(
            dict.fromkeys([int(rows[0]["ctx"]), max(passing_ctx, default=int(rows[0]["ctx"]))])
        )
        by_ctx = {int(row["ctx"]): row for row in rows}
        for ctx in targets[:2]:
            try:
                status = self._probe(lossless, "context", ctx)
            except _BudgetExhaustedError:
                status = "skipped"
            by_ctx[ctx]["lossless_check"] = {
                "status": status,
                "config": lossless.to_dict(),
                "evidence": (
                    f"probes/{_probe_id('context', lossless, ctx)}" if status != "skipped" else None
                ),
            }

    def _finalize(self, *, baseline_only: bool) -> TuneOutcome:
        if baseline_only and self._hard_cap_ancestor(self.default_config) is not None:
            cap = gpu_layer_cap(self.model, self.options)
            raise _NoFeasibleConfigError(
                f"baseline defaults exceed max_gpu_layers={cap}; "
                "no cap-compliant configuration was measured"
            )
        records = self._analysis_records()
        winner: dict[str, Any] | None = None
        recommend_config = self.default_config
        expected = self._defaults_expected()
        confirmed = False
        context_rejected = False

        if not baseline_only:
            winner, recommend_config, expected, confirmed = self._determine_winner()
            context_confirmed = self.options.ctx_size is None or (
                self.context_validation is not None
                and self.context_validation.get("status") == "ok"
                and recommend_config.trial_id in self.validated_configs
            )
            if confirmed and not context_confirmed:
                confirmed = False
                context_rejected = True
                if winner is not None:
                    winner["confirmed"] = False
            if confirmed:
                self._run_depth_profile(recommend_config)
            self._validate_with_cli(recommend_config)
            self._run_quality_gate(recommend_config)

        lossless_winner = recommend.best_lossless(records) if self.options.allow_lossy else None
        if not baseline_only and confirmed:
            self._run_context_envelope(recommend_config, lossless_winner)
        telemetry = self._telemetry_summary(recommend_config)
        warnings = self._collect_warnings(winner)
        if context_rejected:
            warnings.append(
                "measured improvement rejected because the recommended configuration "
                "did not pass required context validation"
            )
        ok_records = [
            r
            for r in records
            if r.get("status") in ("ok", "unstable") and r.get("score") is not None
        ]
        best_record = max(ok_records, key=lambda r: r["score"]) if ok_records else None
        fitting_boundaries = [
            boundary for boundary in self.boundaries if _boundary_has_fit(boundary)
        ]
        max_boundary = (
            max(fitting_boundaries, key=lambda b: b.max_ok_ngl) if fitting_boundaries else None
        )
        estimate = estimate_vram(
            config=recommend_config,
            model=self.model,
            ctx=self.options.ctx_size,
            vram_reserve_mb=self.vram_reserve_mb,
            vram_total_mb=total_vram_mb(self.hardware, multi_gpu=self.options.multi_gpu),
            vram_free_mb=free_vram_mb(self.hardware, multi_gpu=self.options.multi_gpu),
            calibration=self.calibration,
        )
        ram_estimate = estimate_ram(recommend_config, self.model, self.options.ctx_size)
        if self.hardware.ram_mb and ram_estimate.total_mb > 0.9 * self.hardware.ram_mb:
            warnings.append(
                f"estimated host RAM use {ram_estimate.total_mb:.0f} MiB "
                "exceeds 90% of available RAM"
            )
        self.estimate_vs_observed = self._estimate_observation(recommend_config, estimate.total_mb)
        self._complete_coverage()
        if winner is not None and best_record is not None and float(best_record["score"]) > 0:
            drop = (float(best_record["score"]) - float(winner["score"])) / float(
                best_record["score"]
            )
            if drop > 2 * self.baseline.noise_floor_cv:
                warning = (
                    f"confirmed score {float(winner['score']):.3f} is below best search sample "
                    f"{float(best_record['score']):.3f} by more than twice the noise floor"
                )
                warnings.append(warning)
                self._emit("warning", message=warning)
        feasibility = {
            "workload": {
                "pp": self.options.pp,
                "tg": self.options.tg,
                **({"depth": self.options.depth} if self.options.depth is not None else {}),
            },
            "ctx_validated": self.options.ctx_size
            if self.context_validation and self.context_validation.get("status") == "ok"
            else None,
            "boundaries": [dataclasses.asdict(b) for b in self.boundaries],
            "max_fitting": (
                {
                    "gpu_layers": max_boundary.max_ok_ngl,
                    "moe_cpu_layers": max_boundary.moe_cpu_layers,
                }
                if max_boundary
                else None
            ),
            "best_measured": (
                {
                    "gpu_layers": best_record["config"]["gpu_layers"],
                    "moe_cpu_layers": best_record["config"]["moe_cpu_layers"],
                    "pp": best_record["pp_mean"],
                    "tg": best_record["tg_mean"],
                    "score": best_record["score"],
                }
                if best_record
                else None
            ),
            "recommended": {
                "gpu_layers": recommend_config.gpu_layers,
                "moe_cpu_layers": recommend_config.moe_cpu_layers,
                "reason": (
                    "passed full-context validation"
                    if self.context_validation
                    and self.context_validation.get("status") == "ok"
                    and recommend_config.trial_id in self.validated_configs
                    else "full-context validation failed"
                    if self.options.ctx_size is not None
                    else "best measured; no full-context validation"
                ),
            },
            "vram_reserve_mb": self.vram_reserve_mb,
            "reserve_provenance": self.reserve_provenance,
            "estimate": dataclasses.asdict(estimate),
            "ram_estimate": dataclasses.asdict(ram_estimate),
        }
        analysis = recommend.build_analysis(
            target=self.options.target,
            baseline=self.baseline,
            records=records,
            winner=winner,
            lossless_winner=lossless_winner,
            warnings=warnings,
            default_probe=self.default_probe,
            feasibility=feasibility,
            context_validation=self.context_validation,
            cli_validation=self.cli_validation,
            estimate_vs_observed=self.estimate_vs_observed,
            coverage=self.coverage,
            quality_gate=self.quality_gate,
            telemetry=telemetry,
            depth_profile=self.depth_profile,
            context_envelope=self.context_envelope,
            budget_consumed=self.executed_count,
        )
        recommend.write_outputs(
            self.session,
            analysis,
            model=self.model,
            llama=self.llama,
            options=self.options,
            recommend_config=recommend_config,
            expected=expected,
            confirmed=confirmed,
        )
        if confirmed:
            from llamatune.registry import append_record

            append_record(
                self.options.sessions_dir / "registry.jsonl",
                model_report=self.model,
                llama_report=self.llama,
                hardware_signature=hardware_signature(self.hardware),
                session_dir=self.session.dir,
                target=self.options.target,
                ctx_size=self._max_validated_context(),
                pp=self.options.pp,
                tg=self.options.tg,
                depth=self.options.depth,
                config=recommend_config,
                expected=expected,
            )
        self._append({"type": "analysis_written"})

        exit_code = 0 if confirmed else 1
        if baseline_only:
            exit_code = 0
        self._session_end(
            exit_code,
            reason="stopped_by_user" if self._stop_requested else None,
            winner_trial_id=winner.get("trial_id") if winner is not None else None,
        )
        return TuneOutcome(session_dir=self.session.dir, analysis=analysis, exit_code=exit_code)

    def _analysis_records(self) -> list[dict[str, Any]]:
        """Exclude successful resumed measurements that violate the active hard cap."""
        return [
            record
            for record in self.known.values()
            if record.get("status") not in ("ok", "unstable")
            or self._hard_cap_ancestor(TrialConfig.from_dict(record["config"])) is None
        ]

    def _validate_with_cli(self, config: TrialConfig) -> None:
        if self._hard_cap_ancestor(config) is not None:
            return
        if not (
            self.options.validate_with_cli
            and self.llama.cli_path is not None
            and self.options.ctx_size is not None
        ):
            return
        argv = bench.build_cli_context_argv(
            cli_path=self.llama.cli_path,
            model_path=self.model.path,
            config=config,
            ctx=self.options.ctx_size,
            moe=self.model.moe,
        )
        run_id = f"cli-{config.trial_id}"
        probe_dir = self.session.probe_dir(run_id)
        label = f"llama-cli {self._config_label(config)}"
        result, observation = self._run_child(
            kind="cli_validation",
            label=label,
            run_id=run_id,
            argv=argv,
            timeout_s=self.trial_timeout,
            stdout_path=probe_dir / "stdout.log",
            stderr_path=probe_dir / "stderr.log",
        )
        self.executed_count += 1
        status = "ok" if result.exit_code == 0 and not result.timed_out else "failed"
        evidence = f"probes/{run_id}"
        self._write_command(
            evidence, argv, result, 1, self._record_load(config.threads), observation
        )
        self.cli_validation = {"status": status, "evidence": evidence}
        self._append({"type": "stage", "stage": "cli_validation", **self.cli_validation})
        measured = _Measured(status, None, None, None, None)
        self._emit_exec_end(label, run_id, result, measured)
        if status != "ok":
            warning = (
                "recommendation passed llama-bench context probe but failed llama-cli; "
                "see cli_validation evidence"
            )
            self.extra_warnings.append(warning)
            self._emit("warning", message=warning)

    def _run_quality_gate(self, config: TrialConfig) -> None:
        if self._hard_cap_ancestor(config) is not None:
            return
        if not (
            self.options.quality_corpus is not None
            and self.llama.perplexity_path is not None
            and (config.cache_type_k != "f16" or config.cache_type_v != "f16")
        ):
            return
        results: dict[str, float | None] = {}
        for name, candidate in (
            ("lossy", config),
            ("f16", dataclasses.replace(config, cache_type_k="f16", cache_type_v="f16")),
        ):
            argv = bench.build_perplexity_argv(
                perplexity_path=self.llama.perplexity_path,
                model_path=self.model.path,
                config=candidate,
                corpus=self.options.quality_corpus,
                ctx=self.options.ctx_size or self.options.pp + self.options.tg,
                capabilities=self.caps,
            )
            run_id = f"ppl-{name}-{candidate.trial_id}"
            probe_dir = self.session.probe_dir(run_id)
            result, observation = self._run_child(
                kind="probe",
                label=f"perplexity {name}",
                run_id=run_id,
                argv=argv,
                timeout_s=self.trial_timeout * 2,
                stdout_path=probe_dir / "stdout.log",
                stderr_path=probe_dir / "stderr.log",
            )
            self.executed_count += 1
            self._write_command(
                f"probes/{run_id}",
                argv,
                result,
                1,
                self._record_load(candidate.threads),
                observation,
            )
            self._append(
                {
                    "type": "quality_gate_run",
                    "name": name,
                    "trial_id": candidate.trial_id,
                    "run_id": run_id,
                    "status": (
                        "ok" if result.exit_code == 0 and not result.timed_out else "failed"
                    ),
                    "evidence": f"probes/{run_id}",
                }
            )
            raw = result.stdout.path.read_bytes() + b"\n" + result.stderr.path.read_bytes()
            results[name] = bench.parse_perplexity_output(raw)
        self.quality_gate = {
            "status": "ok" if all(value is not None for value in results.values()) else "failed",
            "ppl_lossy": results["lossy"],
            "ppl_f16": results["f16"],
            "delta_pct": (
                (results["lossy"] / results["f16"] - 1) * 100
                if results["lossy"] is not None and results["f16"] not in (None, 0)
                else None
            ),
        }
        self._append({"type": "stage", "stage": "quality_gate", **self.quality_gate})
        if (
            results["lossy"] is not None
            and results["f16"] is not None
            and results["lossy"] > 1.01 * results["f16"]
        ):
            warning = "lossy KV perplexity exceeds the f16 reference by more than 1%"
            self.extra_warnings.append(warning)
            self._emit("warning", message=warning)

    def _estimate_observation(
        self, config: TrialConfig, estimated_total_mb: float
    ) -> dict[str, Any] | None:
        observations = [
            observation
            for run_id, observation in self._observations.items()
            if run_id.startswith(config.trial_id)
        ]
        samples = [sample for observation in observations for sample in observation.samples]
        if not samples:
            return None
        baseline_used = min(sample.vram_used_mb for sample in samples)
        peak_used = max(sample.vram_used_mb for sample in samples)
        result: dict[str, Any] = {
            "estimated_total_mb": estimated_total_mb,
            "observed_used_delta_mb": peak_used - baseline_used,
            "observed_peak_used_mb": peak_used,
            "samples": len(samples),
        }
        per_device: dict[str, dict[str, int]] = {}
        for device_index in sorted(
            {sample.device_index for sample in samples if sample.device_index is not None}
        ):
            device_samples = [s for s in samples if s.device_index == device_index]
            device_baseline = min(sample.vram_used_mb for sample in device_samples)
            device_peak = max(sample.vram_used_mb for sample in device_samples)
            per_device[str(device_index)] = {
                "observed_used_delta_mb": device_peak - device_baseline,
                "observed_peak_used_mb": device_peak,
                "samples": len(device_samples),
            }
        if per_device:
            result["per_device"] = per_device
        return result

    def _telemetry_summary(self, config: TrialConfig) -> dict[str, Any] | None:
        from llamatune.hardware import detect_throttle

        samples = [
            sample
            for run_id, observation in self._observations.items()
            if run_id.startswith(config.trial_id)
            for sample in observation.samples
        ]
        if not samples and self.thermal_pause_count == 0:
            return None
        throttled = detect_throttle(samples) if samples else False
        if throttled:
            warning = "GPU clock telemetry indicates possible thermal or power throttling"
            self.extra_warnings.append(warning)
            self._emit("warning", message=warning)
        temperatures = [s.temperature_c for s in samples if s.temperature_c is not None]
        powers = [s.power_w for s in samples if s.power_w is not None]
        result = {
            "samples": len(samples),
            "peak_temperature_c": max(temperatures) if temperatures else None,
            "peak_power_w": max(powers) if powers else None,
            "throttled": throttled,
        }
        if self.thermal_pause_count:
            result["thermal_pause_count"] = self.thermal_pause_count
            result["thermal_wait_s"] = self.thermal_wait_s
        return result

    def _complete_coverage(self) -> None:
        if not hasattr(self, "incumbent_config"):
            return
        applicable = applicable_dimensions(
            hardware=self.hardware,
            model=self.model,
            llama=self.llama,
            options=self.options,
            incumbent=self.incumbent_config,
        )
        for dim in DIMENSION_ORDER:
            if dim not in applicable:
                continue
            self.coverage.setdefault(
                dim,
                {
                    "candidates": list(
                        candidates_for(
                            dim,
                            hardware=self.hardware,
                            model=self.model,
                            llama=self.llama,
                            options=self.options,
                            incumbent=self.incumbent_config,
                        )
                    ),
                    "executed": [],
                    "cached_hit": [],
                    "pruned": [],
                    "skipped": [],
                },
            )
        for details in self.coverage.values():
            accounted: list[Any] = []
            for bucket in ("executed", "cached_hit", "pruned", "skipped"):
                for item in details.get(bucket, []):
                    accounted.append(item.get("value") if isinstance(item, dict) else item)
            for value in details.get("candidates", []):
                if value not in accounted:
                    details["skipped"].append(
                        {
                            "value": value,
                            "reason": "budget" if not self._can_execute() else "not_reached",
                        }
                    )

    def _determine_winner(
        self,
    ) -> tuple[dict[str, Any] | None, TrialConfig, dict[str, Any], bool]:
        no_improvement = (
            self.baseline.fallback is None
            and self.incumbent_config.to_dict() == self.measured_default.to_dict()
        )
        if no_improvement:
            config, expected = self._fallback_recommendation()
            return None, config, expected, False

        if self._hard_cap_ancestor(self.incumbent_config) is not None:
            config, expected = self._fallback_recommendation()
            return None, config, expected, False

        confirm = self._confirm(self.incumbent_config)
        if confirm.confirmed:
            winner = self._winner_dict(self.incumbent_config, confirm)
            return winner, self.incumbent_config, self._winner_expected(confirm), True

        runner_up = self._runner_up(self.incumbent_config)
        if runner_up is not None:
            confirm2 = self._confirm(runner_up)
            if confirm2.confirmed:
                winner = self._winner_dict(runner_up, confirm2)
                return winner, runner_up, self._winner_expected(confirm2), True

        config, expected = self._fallback_recommendation()
        return None, config, expected, False

    def _fallback_recommendation(self) -> tuple[TrialConfig, dict[str, Any]]:
        """Return defaults, or the best measured config satisfying a binding cap."""
        cap = gpu_layer_cap(self.model, self.options)
        if self.default_config.gpu_layers <= cap:
            return self.default_config, self._defaults_expected()
        candidates = [
            record
            for record in self.known.values()
            if record.get("status") in ("ok", "unstable")
            and record.get("score") is not None
            and int(record["config"]["gpu_layers"]) <= cap
            and (self.options.ctx_size is None or record["trial_id"] in self.validated_configs)
        ]
        if candidates:
            best = max(candidates, key=lambda record: (record["score"], record["trial_id"]))
            expected = {
                "pp": best["pp_mean"],
                "tg": best["tg_mean"],
                "improvement_pct": {
                    "pp": _improvement(float(best["pp_mean"]), self.baseline.pp.mean),
                    "tg": _improvement(float(best["tg_mean"]), self.baseline.tg.mean),
                    "score": (float(best["score"]) - 1) * 100,
                },
            }
            return TrialConfig.from_dict(best["config"]), expected
        raise _NoFeasibleConfigError(
            f"no successful measured configuration satisfies max_gpu_layers={cap}"
        )

    def _runner_up(self, winner_config: TrialConfig) -> TrialConfig | None:
        winner_dict = winner_config.to_dict()
        candidates = [
            record
            for record in self.known.values()
            if record["status"] in ("ok", "unstable")
            and record.get("score") is not None
            and record["config"] != winner_dict
            and int(record["config"]["gpu_layers"]) <= gpu_layer_cap(self.model, self.options)
            and (self.options.ctx_size is None or record["trial_id"] in self.validated_configs)
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda record: (record["score"], record["trial_id"]))
        return TrialConfig.from_dict(best["config"])

    def _confirm(self, config: TrialConfig) -> _Confirm:
        if self._hard_cap_ancestor(config) is not None:
            return _Confirm(False, None, None, 0.0)
        self._quiet_gate("confirmation")
        if (
            self._tuning_budget_enforced
            and self.executed_count + self.options.baseline_runs > self.options.budget_trials
        ):
            warning = "confirmation skipped because trial budget was exhausted"
            if warning not in self.extra_warnings:
                self.extra_warnings.append(warning)
                self._emit("warning", message=warning)
            return _Confirm(False, None, None, 0.0)
        pp_means: list[float] = []
        tg_means: list[float] = []
        for index in range(1, self.options.baseline_runs + 1):
            if not self._can_execute():
                warning = f"confirmation skipped because {_budget_reason(self)}"
                if warning not in self.extra_warnings:
                    self.extra_warnings.append(warning)
                    self._emit("warning", message=warning)
                return _Confirm(False, None, None, 0.0)
            argv = bench.build_bench_argv(
                bench_path=self.llama.bench_path,
                model_path=self.model.path,
                pp=self.options.pp,
                tg=self.options.tg,
                reps=self.options.reps_confirm,
                config=config,
                capabilities=self.caps,
                depth=self.options.depth,
            )
            confirm_dir = self.session.trial_dir(config.trial_id) / f"confirm-{index}"
            confirm_dir.mkdir(parents=True, exist_ok=True)
            run_id = f"{config.trial_id}-confirm-{index}"
            label = (
                f"confirmation {index}/{self.options.baseline_runs} {self._config_label(config)}"
            )
            result, observation = self._run_child(
                kind="confirm",
                label=label,
                run_id=run_id,
                argv=argv,
                timeout_s=self.trial_timeout,
                stdout_path=confirm_dir / "stdout.json",
                stderr_path=confirm_dir / "stderr.log",
            )
            self.executed_count += 1
            measured = self._classify(result, self.options.reps_confirm)
            self._write_command(
                f"trials/{config.trial_id}/confirm-{index}",
                argv,
                result,
                self.options.reps_confirm,
                self._record_load(config.threads),
                observation,
            )
            thermal = self._thermal_retest(
                kind="confirmation",
                trial_id=config.trial_id,
                run_id=run_id,
                label=label,
                argv=argv,
                timeout_s=self.trial_timeout,
                reps=self.options.reps_confirm,
                threads=config.threads,
                retry_dir=confirm_dir / "thermal-retry",
                retry_rel_dir=(f"trials/{config.trial_id}/confirm-{index}/thermal-retry"),
                result=result,
                observation=observation,
                measured=measured,
            )
            result = thermal.result
            measured = thermal.measured
            self._append(
                {
                    "type": "confirmation_run",
                    "trial_id": config.trial_id,
                    "run": index,
                    "status": measured.status,
                    "pp_mean": measured.pp.mean if measured.pp is not None else None,
                    "tg_mean": measured.tg.mean if measured.tg is not None else None,
                    "thermally_contaminated": thermal.contaminated,
                    "thermal_retried": thermal.retried,
                    "thermal_retry_contaminated": thermal.retry_contaminated,
                }
            )
            self._emit(
                "confirmation",
                run=index,
                runs=self.options.baseline_runs,
                status=measured.status,
                pp=measured.pp.mean if measured.pp is not None else None,
                tg=measured.tg.mean if measured.tg is not None else None,
            )
            if not thermal.contaminated:
                self._emit_exec_end(label, run_id, result, measured)
            if not thermal.contaminated:
                self._cooldown()
            if measured.status != "ok" or measured.pp is None or measured.tg is None:
                return _Confirm(False, None, None, 0.0)
            pp_means.append(measured.pp.mean)
            tg_means.append(measured.tg.mean)

        agg_pp = stats.metric_stats(pp_means)
        agg_tg = stats.metric_stats(tg_means)
        conf_score = self._score(agg_pp.mean, agg_tg.mean)
        threshold = 1 + max(2 * self.baseline.noise_floor_cv, 0.03)
        return _Confirm(conf_score > threshold, agg_pp, agg_tg, conf_score)

    def _winner_dict(self, config: TrialConfig, confirm: _Confirm) -> dict[str, Any]:
        return {
            "trial_id": config.trial_id,
            "config": config.to_dict(),
            "pp": confirm.pp.mean,
            "tg": confirm.tg.mean,
            "score": confirm.score,
            "improvement_pct": {
                "pp": _improvement(confirm.pp.mean, self.baseline.pp.mean),
                "tg": _improvement(confirm.tg.mean, self.baseline.tg.mean),
                "score": (confirm.score - 1) * 100,
            },
            "confirmed": True,
            "confirmation": {
                "runs": confirm.pp.n,
                "pp": stats.metric_to_dict(confirm.pp),
                "tg": stats.metric_to_dict(confirm.tg),
            },
        }

    def _winner_expected(self, confirm: _Confirm) -> dict[str, Any]:
        return {
            "pp": confirm.pp.mean,
            "tg": confirm.tg.mean,
            "improvement_pct": {
                "pp": _improvement(confirm.pp.mean, self.baseline.pp.mean),
                "tg": _improvement(confirm.tg.mean, self.baseline.tg.mean),
                "score": (confirm.score - 1) * 100,
            },
        }

    def _defaults_expected(self) -> dict[str, Any]:
        return {
            "pp": self.baseline.pp.mean,
            "tg": self.baseline.tg.mean,
            "improvement_pct": {"pp": 0.0, "tg": 0.0, "score": 0.0},
        }

    def _collect_warnings(self, winner: dict[str, Any] | None) -> list[str]:
        warnings = list(self.hardware.warnings)
        warnings.extend(self.session.resume_warnings)
        warnings.extend(self.extra_warnings)
        warnings.extend(sorted(self.load_warnings))
        if self.baseline.fallback == "cpu":
            default_status = (
                self.default_probe.get("classification") if self.default_probe is not None else None
            )
            failure = {
                "oom": "an out-of-memory failure",
                "gpu_resource": "a GPU-resource failure",
                "cuda_error": "a CUDA failure",
            }.get(str(default_status), "a default-configuration failure")
            warnings.append(
                f"baseline fell back to CPU (-ngl 0) after {failure} at default settings"
            )
        if any(
            record.get("status") == "unstable" and not record.get("thermal_rejected", False)
            for record in self.known.values()
        ):
            warnings.append("one or more trials were flagged unstable (internal cv > 0.10)")
        if any(record.get("thermal_rejected") is True for record in self.known.values()):
            warnings.append(
                "one or more thermally contaminated trial measurements were excluded from scoring"
            )
        if self.options.ctx_size is None:
            warnings.append("no full-context validation was performed")
        if winner is not None and (
            winner["config"].get("cache_type_k") != "f16"
            or winner["config"].get("cache_type_v") != "f16"
        ):
            warnings.append(
                "winner uses lossy KV-cache quantization; validate output quality before adoption"
            )
        if self.context_envelope and any(
            isinstance(fallback := row.get("fallback_config"), dict)
            and (fallback.get("cache_type_k") != "f16" or fallback.get("cache_type_v") != "f16")
            for row in self.context_envelope
        ):
            warnings.append(
                "context envelope includes a lossy KV-cache fallback; validate output quality "
                "before using that alternate"
            )
        return warnings

    def _check_backend_mismatch(self) -> None:
        if not gpu_present(self.hardware) or has_gpu_backend(self.llama) is not False:
            return
        gpu_names = [gpu.name for gpu in self.hardware.gpus]
        backends = self.llama.backends
        warning = (
            f"a GPU was detected ({', '.join(gpu_names)}) but the llama.cpp build reports "
            f"CPU-only backends ('{backends}'); GPU dimensions were skipped — verify "
            "--llama-bin points at the intended build (e.g. a CUDA/Metal build directory)"
        )
        if self._find_stage("backend_mismatch") is None:
            self._append(
                {
                    "type": "stage",
                    "stage": "backend_mismatch",
                    "gpus": gpu_names,
                    "backends": backends,
                }
            )
        if warning not in self.extra_warnings:
            self.extra_warnings.append(warning)
            self._emit("warning", message=warning)

    # -- helpers --------------------------------------------------------

    def _normalize_resolved_config(self, config: TrialConfig) -> TrialConfig:
        if config.gpu_layers == -1:
            return dataclasses.replace(config, gpu_layers=self.model.ngl_all)
        return config

    def _score(self, pp: float, tg: float) -> float:
        return stats.score(
            pp=pp,
            tg=tg,
            pp0=self.baseline.pp.mean,
            tg0=self.baseline.tg.mean,
            weights=self.weights,
        )

    def _can_execute(self) -> bool:
        if self._stop_requested:
            return False
        if not self._tuning_budget_enforced:
            return True
        if self.executed_count >= self._trial_limit():
            return False
        if self.options.budget_minutes is not None:
            elapsed = _monotonic() - self.start
            if elapsed >= self.options.budget_minutes * 60.0:
                return False
        return True

    def _trial_limit(self) -> int:
        reserve = 0
        if not self._finalizing and not self.options.baseline_only:
            reserve = self.options.baseline_runs
            if self.options.ctx_size is not None:
                reserve += 1
        return max(0, self.options.budget_trials - reserve)

    def _cooldown(self) -> None:
        if self.options.cooldown_s > 0:
            _sleep(self.options.cooldown_s)
        cap = self.options.thermal_wait_cap_s
        if cap <= 0:
            return
        sample = _sample_gpu_state()
        if sample is None:
            return
        self._thermal_samples.append(sample)
        self._thermal_samples = self._thermal_samples[-30:]
        waited = 0.0
        while waited < cap:
            temperature_hot = (
                sample.temperature_c is not None
                and sample.temperature_c > self.options.thermal_threshold_c
            )
            throttled = _detect_gpu_throttle(self._thermal_samples)
            if not temperature_hot and not throttled:
                break
            delay = min(_THERMAL_WAIT_CYCLE_S, cap - waited)
            budget_exhausted = False
            if self.options.budget_minutes is not None:
                remaining = self.options.budget_minutes * 60.0 - (_monotonic() - self.start)
                if remaining <= 0:
                    break
                delay = min(delay, remaining)
                budget_exhausted = delay >= remaining
            if delay <= 0:
                break
            before_temperature = sample.temperature_c
            reason = "temperature" if temperature_hot else "throttle"
            _sleep(delay)
            waited += delay
            self.thermal_wait_s += delay
            self.thermal_pause_count += 1
            next_sample = _sample_gpu_state()
            if next_sample is not None:
                sample = next_sample
                self._thermal_samples.append(sample)
                self._thermal_samples = self._thermal_samples[-30:]
            self._append(
                {
                    "type": "stage",
                    "stage": "thermal_pause",
                    "reason": reason,
                    "waited_s": delay,
                    "total_waited_s": waited,
                    "temperature_before_c": before_temperature,
                    "temperature_after_c": (
                        next_sample.temperature_c if next_sample is not None else None
                    ),
                    "threshold_c": self.options.thermal_threshold_c,
                    "cap_reached": waited >= cap,
                    "budget_exhausted": budget_exhausted,
                }
            )
            if next_sample is None or budget_exhausted:
                break

    def _record_load(self, expected_threads: int | None = None) -> float | None:
        load = _load_avg()
        if load is None:
            return None
        # llama-bench itself contributes roughly one runnable task per worker;
        # warn only about load materially beyond that expected self-load.
        threads = self.hardware.physical_cores if expected_threads is None else expected_threads
        threshold = threads + max(2.0, self.hardware.physical_cores / 2)
        self._load_total += 1
        if load > threshold:
            self._load_exceeded += 1
            if load > self._load_peak:
                self._load_peak = load
                self._load_peak_threshold = threshold
        if self._load_warning is not None:
            self.load_warnings.discard(self._load_warning)
            self._load_warning = None
        if self._load_exceeded:
            self._load_warning = (
                "system load exceeded the contention threshold before "
                f"{self._load_exceeded} of {self._load_total} invocations "
                f"(peak {self._load_peak:.2f}, threshold {self._load_peak_threshold:.1f}); "
                "affected measurements may be pessimistic"
            )
            self.load_warnings.add(self._load_warning)
        return load

    def _quiet_gate(self, phase: str) -> None:
        if self.options.quiet_wait_s <= 0:
            return
        threshold = self.options.quiet_load
        if threshold is None:
            threshold = self.hardware.physical_cores / 2
        before = _load_avg()
        if before is None:
            self._append(
                {
                    "type": "stage",
                    "stage": "quiet_gate",
                    "phase": phase,
                    "status": "unsupported",
                    "waited_s": 0.0,
                    "load_before": None,
                    "load_after": None,
                    "threshold": threshold,
                    "gave_up": False,
                }
            )
            return
        waited = 0.0
        current = before
        while current > threshold and waited < self.options.quiet_wait_s:
            delay = min(5.0, self.options.quiet_wait_s - waited)
            _sleep(delay)
            waited += delay
            sampled = _load_avg()
            if sampled is None:
                break
            current = sampled
        gave_up = current > threshold
        self._append(
            {
                "type": "stage",
                "stage": "quiet_gate",
                "phase": phase,
                "status": "gave_up" if gave_up else "ok",
                "waited_s": waited,
                "load_before": before,
                "load_after": current,
                "threshold": threshold,
                "gave_up": gave_up,
            }
        )
        if gave_up:
            warning = f"quiet gate gave up after {waited:.1f}s at load {current:.2f}"
            self.load_warnings.add(warning)
            self._emit("warning", message=warning)

    def _write_command(
        self,
        rel_dir: str,
        argv: tuple[str, ...],
        result: executor.ExecResult,
        reps: int,
        load: float | None,
        observation: _RunObservation | None = None,
    ) -> None:
        core_dump_control = executor.core_dump_control_status(not self.options.allow_core_dumps)
        payload = {
            "argv": list(argv),
            "reps": reps,
            "env_names": list(result.env_names),
            "started": result.started,
            "ended": result.ended,
            "wall_s": result.wall_s,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "load_avg_1m": load,
            "core_dumps_disabled": core_dump_control == "disabled",
            "core_dump_control": core_dump_control,
            "stdout": _capture_dict(result.stdout, self.session.dir),
            "stderr": _capture_dict(result.stderr, self.session.dir),
        }
        if observation is not None and self.options.observe_vram:
            include_device = self.options.multi_gpu and len(self.hardware.gpus) > 1
            payload["gpu_before"] = (
                _gpu_sample_dict(observation.before, include_device=include_device)
                if observation.before is not None
                else None
            )
            payload["gpu_after"] = (
                _gpu_sample_dict(observation.after, include_device=include_device)
                if observation.after is not None
                else None
            )
            payload["gpu_samples"] = [
                _gpu_sample_dict(sample, include_device=include_device)
                for sample in observation.samples
            ]
        self.session.write_text(
            f"{rel_dir}/command.json", json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )


def _capture_dict(capture: executor.CaptureInfo, session_dir: Path) -> dict[str, Any]:
    try:
        rel = str(capture.path.relative_to(session_dir))
    except ValueError:
        rel = str(capture.path)
    return {
        "path": rel,
        "sha256": capture.sha256,
        "size_bytes": capture.size_bytes,
        "truncated": capture.truncated,
    }


def _other_fields(config: dict[str, Any]) -> dict[str, Any]:
    """The config minus the two dimensions the OOM-pruning order is over."""
    return {k: v for k, v in config.items() if k not in ("gpu_layers", "moe_cpu_layers")}


def _gpu_sample_dict(sample: GpuSample, *, include_device: bool) -> dict[str, Any]:
    data = dataclasses.asdict(sample)
    if not include_device:
        data.pop("device_index", None)
    return data


def _probe_id(purpose: str, config: TrialConfig, ctx: int | None) -> str:
    canonical = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{purpose}{canonical}{ctx}".encode()).hexdigest()[:16]


def _order_candidates(dim: str, candidates: tuple[Any, ...]) -> list[Any]:
    if dim == "gpu_layers":
        return sorted(candidates)
    if dim == "moe_cpu_layers":
        return sorted(candidates, reverse=True)
    return list(candidates)


def _count_executed(entries: tuple[dict[str, Any], ...]) -> int:
    count = 0
    for entry in entries:
        kind = entry.get("type")
        if (
            (
                kind
                in (
                    "baseline_run",
                    "confirmation_run",
                    "probe",
                    "depth_profile_run",
                    "stability_rerun",
                    "pair_check",
                    "quality_gate_run",
                )
                and entry.get("status") != "pruned"
            )
            or (kind == "trial" and entry.get("status") != "pruned")
            or kind == "thermal_retry"
            or (kind == "stage" and entry.get("stage") == "cli_validation")
        ):
            count += 1
    return count


def _improvement(value: float, baseline: float) -> float:
    if baseline <= 0:
        return 0.0
    return (value / baseline - 1) * 100


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return statistics.median(values)


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _sample_matches(fields: dict[str, Any], config: TrialConfig) -> bool:
    expected = {
        "n_gpu_layers": config.gpu_layers,
        "n_cpu_moe": config.moe_cpu_layers,
        "flash_attn": int(config.flash_attn),
        "n_ubatch": config.ubatch,
        "n_batch": config.batch,
        "n_threads": config.threads,
        "n_threads_batch": config.threads_batch,
        "use_mmap": config.mmap,
        "no_kv_offload": config.no_kv_offload,
        "type_k": config.cache_type_k,
        "type_v": config.cache_type_v,
        "tensor_buft_overrides": config.ot_spec or "none",
    }
    return all(expected.get(key) == value for key, value in fields.items())


def _neighbor_values(candidates: tuple[int, ...], value: int) -> tuple[int, ...]:
    ordered = sorted(set(candidates) | {value})
    index = ordered.index(value)
    return tuple(ordered[max(0, index - 1) : index + 2])


def _ot_specs(n_layer: int, count: int) -> tuple[str, ...]:
    count = min(n_layer, max(1, count))
    head = list(range(count))
    tail = list(range(n_layer - count, n_layer))
    stride = max(1, n_layer // count)
    spread = list(range(0, n_layer, stride))[:count]
    component_count = min(n_layer, 2 * count)
    component_stride = max(1, n_layer // component_count)
    component = list(range(0, n_layer, component_stride))[:component_count]

    def spec(indices: list[int], tensor: str = "ffn_.*_exps") -> str:
        joined = "|".join(str(index) for index in indices)
        return rf"^blk\.({joined})\.{tensor}=CPU$"

    return tuple(
        dict.fromkeys((spec(head), spec(tail), spec(spread), spec(component, "ffn_(up|down)_exps")))
    )


def _monotonic() -> float:
    return time.monotonic()


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _load_avg() -> float | None:
    getloadavg = cast(
        Callable[[], tuple[float, float, float]] | None,
        getattr(os, "getloadavg", None),
    )
    if getloadavg is None:
        return None  # pragma: no cover - platform dependent
    try:
        return getloadavg()[0]
    except OSError:  # pragma: no cover - platform dependent
        return None
