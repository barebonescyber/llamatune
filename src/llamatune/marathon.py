"""Serial orchestration and confined evidence writer for Marathon."""

from __future__ import annotations

import dataclasses
import json
import math
import statistics
import sys
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self, cast

from llamatune._version import __version__
from llamatune.calibrate import run_calibration
from llamatune.evidence import (
    EvidenceWriter,
    InterruptState,
    PathEscapeError,
    confined_path,
    create_unique_dir,
    install_interrupt_handlers,
    read_journal_lines,
)
from llamatune.evidence import (
    jsonable as _jsonable,
)
from llamatune.evidence import (
    resolve_deadline as resolve_deadline,
)
from llamatune.evidence import (
    utc_iso as _utc_iso,
)
from llamatune.sanitize import strip_control_chars
from llamatune.types import (
    CoverageLedger,
    HardwareReport,
    LlamaCppReport,
    MarathonOptions,
    MarathonOutcome,
    ModelReport,
    NightshiftOptions,
    RegistryRecord,
    Reporter,
    TrialConfig,
    TuneOptions,
)

MIN_ROUND_MINUTES = 30.0
MATRIX_MIN_MINUTES = 15.0
SHUTDOWN_MARGIN_MIN = 5.0
BRACKET_FRESH_MINUTES = 15.0
RECON_RUNS = 10
BASE_BUDGET = 240
MAX_BUDGET = 1000
REPS_SEARCH = 8
REPS_CONFIRM = 12
BASELINE_RUNS = 5
COOLDOWN_S = 10.0
_RECON_TIMEOUT_S = 1800.0


class MarathonPathError(PathEscapeError):
    """A requested artifact path escaped its Marathon run directory."""


def _confine(base: Path, *parts: str) -> Path:
    return confined_path(base, *parts, error=MarathonPathError)


class MarathonRun(EvidenceWriter):
    """The sole writer inside one Marathon evidence directory."""

    path_error: type[PathEscapeError] = MarathonPathError

    def __init__(self, run_dir: Path) -> None:
        self._dir = Path(run_dir)
        self.dir = self._dir
        self._bracket = 1
        self.decision_threshold = 0.01
        self.champion_evidence: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        sessions_dir: Path,
        *,
        options: MarathonOptions,
        hardware: HardwareReport,
        llama: LlamaCppReport,
        model: ModelReport,
        argv: list[str],
    ) -> Self:
        root = sessions_dir / "marathon"
        stem = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in model.path.stem)
        run_dir = create_unique_dir(root, stem, error=MarathonPathError)
        run = cls(run_dir)
        run.write_json(
            "run.json",
            {
                "schema_version": 1,
                "tool_version": __version__,
                "argv": argv,
                "options": _jsonable(options),
                "created": _utc_iso(),
            },
        )
        run.write_json("hardware.json", cast(dict[str, Any], _jsonable(hardware)))
        run.write_json("llamacpp.json", cast(dict[str, Any], _jsonable(llama)))
        run.write_json("model.json", cast(dict[str, Any], _jsonable(model)))
        run.append(
            {
                "type": "marathon_start",
                "tool_version": __version__,
                "model_fingerprint": model.fingerprint,
            }
        )
        return run

    @classmethod
    def load(cls, run_dir: Path) -> Self:
        path = Path(run_dir)
        if not (path / "run.json").is_file():
            raise FileNotFoundError(f"not a Marathon run: {path}")
        return cls(path)

    def _mkdir(self, *parts: str) -> Path:
        path = _confine(self._dir, *parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def recon_dir(self, n: int) -> Path:
        return self._mkdir("recon", f"run-{n}")

    def bracket_dir(self, n: int, k: int) -> Path:
        return self._mkdir("brackets", str(n), f"run-{k}")

    def ab_dir(self, label: str, block: int, slot: str) -> Path:
        if slot not in {"a1", "b1", "b2", "a2"}:
            raise MarathonPathError(f"invalid A/B slot: {slot}")
        return self._mkdir("ab", label, f"block-{block}", slot)

    def matrix_dir(self, ctx: int, depth: int, refine: int | None = None) -> Path:
        parts = ["matrix", f"ctx-{ctx}-d-{depth}"]
        if refine is not None:
            parts.append(f"refine-{refine}")
        return self._mkdir(*parts)

    def calibration_dir(self, fingerprint16: str, n: int) -> Path:
        if not fingerprint16 or any(ch not in "0123456789abcdefABCDEF" for ch in fingerprint16):
            raise MarathonPathError("calibration fingerprint must be hexadecimal")
        return self.bracket_dir(self._bracket, n)

    def select_bracket(self, number: int) -> None:
        self._bracket = number


def round_budget(base: int, index: int) -> int:
    """Return the deterministic 1-based round escalation."""
    if index < 1:
        raise ValueError("round index must be positive")
    return min(round(base * 1.5 ** (index - 1)), MAX_BUDGET)


def coverage_percent(ledger: CoverageLedger) -> float:
    total = sum(t.enumerated for t in ledger.tiers.values())
    covered = sum(t.executed + t.pruned for t in ledger.tiers.values())
    return 100.0 if total == 0 else 100.0 * covered / total


def coverage_complete(ledger: CoverageLedger) -> bool:
    return all(
        ledger.tiers.get(tier) is None or ledger.tiers[tier].remaining == 0
        for tier in ("A", "B", "C")
    )


def has_converged(no_change_rounds: int, required: int, ledger: CoverageLedger) -> bool:
    return no_change_rounds >= required and coverage_complete(ledger)


def remaining_minutes(deadline: datetime | None, now: datetime) -> float | None:
    return None if deadline is None else max(0.0, (deadline - now).total_seconds() / 60.0)


def round_fits(remaining: float | None, *, ab_reserved: float = 0.0) -> bool:
    return remaining is None or remaining >= MIN_ROUND_MINUTES + SHUTDOWN_MARGIN_MIN + ab_reserved


def ab_reserved_minutes(ab_blocks: int, estimate_invocation_minutes: float = 5.0) -> float:
    """Reserve two invocations per block (the two sides, ABBA-amortized)."""
    return float(math.ceil(ab_blocks * 2 * estimate_invocation_minutes))


def identity(options: MarathonOptions, fingerprint: str) -> dict[str, Any]:
    return {
        "model_fingerprint": fingerprint,
        "target": options.target,
        "allow_lossy": options.allow_lossy,
        "pp": options.pp,
        "tg": options.tg,
        "depth_grid": list(options.depth_grid),
        "ctx_ladder": list(options.ctx_ladder),
    }


def identity_matches(
    run_meta: Mapping[str, Any], options: MarathonOptions, fingerprint: str
) -> bool:
    stored = run_meta.get("identity")
    return isinstance(stored, Mapping) and dict(stored) == identity(options, fingerprint)


def _entries(run_dir: Path) -> list[dict[str, Any]]:
    """Load journal entries; corrupt lines are skipped and warned, never fatal.

    The shared reader guarantees later valid entries still load after a
    corrupt line (issue #8): resume scans must not silently truncate.
    """
    entries, corruption = read_journal_lines(run_dir / "journal.jsonl")
    for warning in corruption:
        warnings.warn(warning, RuntimeWarning, stacklevel=2)
    return entries


def find_reentry(
    sessions_dir: Path, options: MarathonOptions, fingerprint: str
) -> tuple[Path | None, tuple[Path, ...]]:
    """Find the newest matching unfinished run and retain mismatches as warnings."""
    root = sessions_dir / "marathon"
    matches: list[Path] = []
    mismatches: list[Path] = []
    for candidate in sorted(root.iterdir(), reverse=True) if root.is_dir() else ():
        try:
            meta = json.loads((candidate / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        entries = _entries(candidate)
        if any(entry.get("type") == "marathon_end" for entry in entries):
            continue
        (matches if identity_matches(meta, options, fingerprint) else mismatches).append(candidate)
    return (matches[0] if matches else None, tuple(mismatches))


def _profile(options: MarathonOptions, index: int, remaining: float | None) -> TuneOptions:
    cooldown = COOLDOWN_S if options.cooldown_s is None else options.cooldown_s
    return TuneOptions(
        target=options.target,
        budget_trials=round_budget(options.budget_trials or BASE_BUDGET, index),
        budget_minutes=(None if remaining is None else max(1.0, remaining - SHUTDOWN_MARGIN_MIN)),
        reps_search=options.reps_search or REPS_SEARCH,
        reps_confirm=options.reps_confirm or REPS_CONFIRM,
        baseline_runs=options.baseline_runs or BASELINE_RUNS,
        pp=options.pp,
        tg=options.tg,
        allow_lossy=options.allow_lossy,
        cooldown_s=cooldown,
        baseline_only=False,
        llama_bin=options.llama_bin,
        sessions_dir=options.sessions_dir,
        full_hash=options.full_hash,
        ctx_size=options.ctx_size,
        ctx_ladder=options.ctx_ladder,
        vram_reserve_mb=options.vram_reserve_mb,
        quality_corpus=options.quality_corpus,
        ot_search=options.ot_search,
    )


def _plan(options: MarathonOptions) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "recon_runs": RECON_RUNS,
        "rounds_max": options.rounds_max,
        "round_1_budget": options.budget_trials or BASE_BUDGET,
        "round_budgets": [
            round_budget(options.budget_trials or BASE_BUDGET, n)
            for n in range(1, options.rounds_max + 1)
        ],
        "steering": "budget-shaped coverage tiers A-D",
        "matrix_cells": [
            {"ctx": ctx, "depth": depth}
            for ctx in (options.ctx_ladder or ((options.ctx_size or 0),))
            for depth in options.depth_grid
        ],
        "ab": {
            "blocks": options.ab_blocks,
            "challenge_blocks": max(3, options.ab_blocks - 2),
            "order": ["A", "B", "B", "A"],
        },
    }


def _recon(
    run: MarathonRun,
    model: ModelReport,
    llama: LlamaCppReport,
    options: MarathonOptions,
) -> dict[str, Any] | None:
    from llamatune import bench, executor, stats

    argv = bench.build_baseline_argv(
        bench_path=llama.bench_path,
        model_path=model.path,
        pp=options.pp,
        tg=options.tg,
        reps=options.reps_confirm or REPS_CONFIRM,
        capabilities=llama.capabilities,
    )
    pp: list[float] = []
    tg: list[float] = []
    default_config: TrialConfig | None = None
    for number in range(1, RECON_RUNS + 1):
        target = run.recon_dir(number)
        try:
            result = executor.run(
                argv,
                timeout_s=_RECON_TIMEOUT_S,
                stdout_path=target / "stdout.json",
                stderr_path=target / "stderr.log",
            )
            sample = (
                bench.parse_bench_output((target / "stdout.json").read_bytes())
                if result.exit_code == 0 and not result.timed_out
                else None
            )
        except (OSError, ValueError):
            sample = None
        run.write_json(
            str((target / "command.json").relative_to(run.dir)),
            {
                "argv": list(argv),
                "timeout_s": _RECON_TIMEOUT_S,
                "exit_code": None if sample is None else 0,
            },
        )
        run.append({"type": "recon_run", "run": number, "ok": sample is not None})
        if sample is None:
            return None
        if default_config is None:
            default_config = bench.resolved_config(sample.pp_entry)
        pp.append(sample.pp_avg)
        tg.append(sample.tg_avg)
        if (options.cooldown_s or COOLDOWN_S) > 0 and number != RECON_RUNS:
            time.sleep(options.cooldown_s or COOLDOWN_S)
    pp_stats, tg_stats = stats.metric_stats(pp), stats.metric_stats(tg)
    cv = max(0.01, pp_stats.cv, tg_stats.cv)
    threshold = max(options.drift_threshold, 2 * cv)
    drift_pp = abs(statistics.fmean(pp[5:]) / statistics.fmean(pp[:5]) - 1)
    drift_tg = abs(statistics.fmean(tg[5:]) / statistics.fmean(tg[:5]) - 1)
    return {
        "pp": pp_stats.mean,
        "tg": tg_stats.mean,
        "cv_ref": cv,
        "runs": RECON_RUNS,
        "warmup_drift": drift_pp > threshold or drift_tg > threshold,
        "trend": {"pp": drift_pp, "tg": drift_tg},
        "default_config": default_config.to_dict() if default_config is not None else None,
    }


def _synth_record(
    reference: Mapping[str, Any],
    model: ModelReport,
    llama: LlamaCppReport,
    options: MarathonOptions,
    run: MarathonRun,
) -> RegistryRecord:
    return RegistryRecord(
        fingerprint=model.fingerprint,
        session_dir=run.dir,
        created=_utc_iso(),
        outcome="marathon_reference",
        reference_config=None,
        reference_pp=float(reference["pp"]),
        reference_tg=float(reference["tg"]),
        noise_floor_cv=float(reference["cv_ref"]),
        pp_workload=options.pp,
        tg_workload=options.tg,
        reps_confirm=options.reps_confirm or REPS_CONFIRM,
        target=options.target,
        build_commit=llama.build_commit,
        help_sha256=llama.help_sha256,
        median_trial_wall_s=None,
        session_wall_s=None,
        bench_sha256=llama.bench_sha256,
        ctx_size=options.ctx_size,
    )


def _bracket_options(options: MarathonOptions) -> NightshiftOptions:
    return NightshiftOptions(
        models_dir=options.model_path.parent,
        llama_bin=options.llama_bin,
        sessions_dir=options.sessions_dir,
        until=None,
        max_hours=None,
        profile="deep",
        drift_threshold=options.drift_threshold,
        calibration_runs=3,
        duplicates="one",
        include=(),
        exclude=(),
        dry_run=False,
        target=options.target,
        allow_lossy=options.allow_lossy,
        ctx_size=options.ctx_size,
        vram_reserve_mb=options.vram_reserve_mb,
        cooldown_s=options.cooldown_s,
        full_hash=options.full_hash,
        budget_trials=None,
        reps_search=None,
        reps_confirm=None,
        baseline_runs=None,
        ctx_ladder=options.ctx_ladder,
    )


def _trial_evidence(
    session_dirs: Sequence[Path],
) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    executed: set[str] = set()
    pruned: set[str] = set()
    trials: list[dict[str, Any]] = []
    for session_dir in session_dirs:
        try:
            entries = _entries(session_dir)
        except OSError:
            continue
        for entry in entries:
            if entry.get("type") == "trial" and isinstance(entry.get("trial_id"), str):
                executed.add(entry["trial_id"])
                trials.append(entry)
            if entry.get("type") in {"prune", "pruned"}:
                pruned.update(str(item) for item in entry.get("trial_ids", ()))
    return executed, pruned, trials


def _finalize(
    run: MarathonRun,
    summary: dict[str, Any],
    exit_code: int,
    *,
    interrupt_state: InterruptState | None = None,
) -> MarathonOutcome:
    if interrupt_state is not None:
        for signum, immediate in interrupt_state.drain_events():
            run.append({"type": "interrupted", "signal": signum, "immediate": immediate})
    summary["exit_code"] = exit_code
    run.write_json("marathon.json", summary)
    try:
        from llamatune.marathonreport import render

        report = render(summary)
    except (ImportError, AttributeError):
        report = "# Marathon\n\n" + json.dumps(summary, indent=2, sort_keys=True)
    run.write_text("marathon-report.md", report)
    run.append(
        {
            "type": "marathon_end",
            "exit_code": exit_code,
            "stop_reason": summary.get("stop_reason"),
        }
    )
    return MarathonOutcome(run_dir=run.dir, summary=summary, exit_code=exit_code)


def _announce(
    reporter: Reporter | None,
    message: str,
    payload: dict[str, Any],
) -> None:
    """Emit one concise progress line for an orchestrator phase or round."""
    if reporter is None:
        return
    message = strip_control_chars(message)
    from llamatune.types import ProgressEvent

    reporter.emit(
        ProgressEvent(
            kind="orchestrator_item",
            ts=_utc_iso(),
            payload={"message": message, **payload},
        )
    )
    from llamatune import ui

    if isinstance(reporter, ui.PlainReporter):
        reporter.err.write(message + "\n")
        reporter.err.flush()


@dataclasses.dataclass
class _MarathonState:
    """Run-scope context threaded explicitly through Marathon phases."""

    run: MarathonRun
    model: ModelReport
    llama: LlamaCppReport
    options: MarathonOptions
    hardware: HardwareReport
    summary: dict[str, Any]
    clock: Callable[[], datetime]
    deadline: datetime | None
    interrupt_state: InterruptState
    reporter: Reporter | None
    tracker: BracketTracker
    default_config: TrialConfig
    ledger: CoverageLedger
    completed_rounds: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    champion: TrialConfig | None = None
    champion_session: Path | None = None
    session_dirs: list[Path] = dataclasses.field(default_factory=list)
    executed: set[str] = dataclasses.field(default_factory=set)
    pruned: set[str] = dataclasses.field(default_factory=set)
    journaled_trials: list[dict[str, Any]] = dataclasses.field(default_factory=list)


class BracketTracker:
    """Owns bracket numbering, error counting, and reference re-baselining."""

    def __init__(
        self,
        run: MarathonRun,
        model: ModelReport,
        llama: LlamaCppReport,
        options: MarathonOptions,
        summary: dict[str, Any],
        clock: Callable[[], datetime],
    ) -> None:
        self._run = run
        self._model = model
        self._llama = llama
        self._options = options
        self._summary = summary
        self._clock = clock
        self.number = 0
        self.errors = 0
        self.last_at: datetime | None = None
        self.reference: dict[str, Any] = {}

    def arm(self, reference: dict[str, Any]) -> None:
        """Adopt a reconnaissance reference and derive the decision threshold."""
        self.reference = reference
        self._run.decision_threshold = max(0.01, 2.0 * float(reference["cv_ref"]))

    def fresh(self, now: datetime) -> bool:
        """Whether the last bracket is recent enough to skip recalibration."""
        return (
            self.last_at is not None
            and (now - self.last_at).total_seconds() < BRACKET_FRESH_MINUTES * 60
        )

    def bracket(self, phase: str) -> bool:
        """Run one environment bracket; False trips the circuit breaker."""
        current = self._clock()
        if self.fresh(current):
            return True
        self.number += 1
        self._run.select_bracket(self.number)
        result = run_calibration(
            self._run,
            _synth_record(self.reference, self._model, self._llama, self._options, self._run),
            self._model,
            self._llama,
            _bracket_options(self._options),
        )
        payload = cast(dict[str, Any], _jsonable(result))
        self._run.append({"type": "bracket", "phase": phase, "number": self.number, **payload})
        self.last_at = current
        if result.verdict == "error":
            self.errors += 1
            self._summary["failed"].append(
                {"kind": "bracket", "phase": phase, "reason": result.reason}
            )
            return self.errors < 3
        self.errors = 0
        if result.verdict == "drift" and result.pp is not None and result.tg is not None:
            old = dict(self.reference)
            self.arm(
                {
                    **self.reference,
                    "pp": result.pp.mean,
                    "tg": result.tg.mean,
                    "cv_ref": max(0.01, result.pp.cv, result.tg.cv),
                }
            )
            self._summary["reference"]["current"] = self.reference
            self._summary["reference"]["rebaselines"].append(
                {"phase": phase, "old": old, "new": dict(self.reference)}
            )
            self._run.append(
                {
                    "type": "environment_drift",
                    "phase": phase,
                    "old_reference": old,
                    "new_reference": self.reference,
                }
            )
            if len(self._summary["reference"]["rebaselines"]) >= 2:
                self._summary["warnings"].append(
                    "environment unstable: consecutive phase brackets drifted"
                )
        return True


def _journal_interrupts(run: MarathonRun, state: InterruptState) -> None:
    """Journal queued interrupt events at the next safe main-flow point."""
    for signum, immediate in state.drain_events():
        run.append({"type": "interrupted", "signal": signum, "immediate": immediate})


def _resume_incomplete_round(
    run: MarathonRun, incomplete: Mapping[str, Any], reporter: Reporter | None
) -> None:
    """Finish an interrupted round's session before continuing the plan."""
    from llamatune.search import resume_tuning

    outcome = resume_tuning(Path(str(incomplete["session_dir"])), reporter=reporter)
    run.append(
        {
            "type": "round_end" if outcome.exit_code in (0, 1) else "round_failed",
            "index": incomplete["index"],
            "session_dir": str(outcome.session_dir),
            "exit_code": outcome.exit_code,
            "resumed": True,
        }
    )


def _resolve_reference(
    run: MarathonRun,
    model: ModelReport,
    llama: LlamaCppReport,
    options: MarathonOptions,
    recon_entry: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Reuse prior reconnaissance evidence or measure it (one retry)."""
    reference_value = (
        dict(recon_entry["reference"]) if recon_entry else _recon(run, model, llama, options)
    )
    if reference_value is None:
        reference_value = _recon(run, model, llama, options)
    return reference_value


def _restore_champion(
    completed_rounds: Sequence[Mapping[str, Any]],
) -> tuple[TrialConfig | None, Path | None]:
    """Replay round-end journals to recover the current champion."""
    champion: TrialConfig | None = None
    champion_session: Path | None = None
    for completed in completed_rounds:
        config = completed.get("winner_config")
        if completed.get("champion_changed") and isinstance(config, dict):
            champion = TrialConfig.from_dict(config)
            champion_session = Path(str(completed["session_dir"]))
    return champion, champion_session


def _fallback_default_config(hardware: HardwareReport) -> TrialConfig:
    """Conservative all-CPU defaults used when recon recorded none."""
    return TrialConfig(
        gpu_layers=0,
        moe_cpu_layers=0,
        flash_attn=False,
        ubatch=512,
        batch=2048,
        threads=max(1, hardware.physical_cores),
        mmap=True,
        no_kv_offload=False,
        cache_type_k="f16",
        cache_type_v="f16",
    )


def _initial_default_config(reference: Mapping[str, Any], hardware: HardwareReport) -> TrialConfig:
    default_data = reference.get("default_config")
    if isinstance(default_data, dict):
        return TrialConfig.from_dict(default_data)
    return _fallback_default_config(hardware)


def _champion_evidence(state: _MarathonState) -> dict[str, Any] | None:
    """Load confirmed champion context evidence from its session analysis."""
    session = state.champion_session
    if session is None:
        return None
    try:
        champion_analysis = json.loads((session / "analysis.json").read_text(encoding="utf-8"))
        winner = champion_analysis.get("winner")
        validated = champion_analysis.get("feasibility", {}).get("ctx_validated")
        if (
            isinstance(winner, dict)
            and winner.get("confirmed") is True
            and isinstance(validated, int)
            and state.options.ctx_size is not None
            and validated >= state.options.ctx_size
        ):
            return {
                "ctx": state.options.ctx_size,
                "config": winner.get("config"),
                "pp": winner.get("pp"),
                "tg": winner.get("tg"),
                "evidence": str(session / "analysis.json"),
            }
    except (OSError, TypeError, ValueError):
        return None
    return None


def _run_rounds_phase(state: _MarathonState) -> int | None:
    """Execute tuning rounds; returns a finalize exit code, or None to continue."""
    from llamatune.coverage import build_ledger, enumerate_space
    from llamatune.hardware import assess_hardware

    run, model, llama, options = state.run, state.model, state.llama, state.options
    summary = state.summary
    run.append({"type": "phase", "phase": "rounds"})
    _announce(state.reporter, "[marathon] phase rounds", {"phase": "rounds"})
    if not state.tracker.bracket("rounds"):
        summary["stop_reason"] = "bracket_circuit_breaker"
        return 3
    no_change = 0
    failures = 0
    for index in range(len(state.completed_rounds) + 1, options.rounds_max + 1):
        _journal_interrupts(run, state.interrupt_state)
        if state.interrupt_state.stop_requested:
            break
        remain = remaining_minutes(state.deadline, state.clock())
        if not round_fits(
            remain,
            ab_reserved=(ab_reserved_minutes(options.ab_blocks) if state.deadline else 0.0),
        ):
            summary["stop_reason"] = "deadline"
            summary["deferred"].append(
                {"kind": "round", "index": index, "reason": "insufficient time"}
            )
            run.append({"type": "deferred", **summary["deferred"][-1]})
            break
        tune_options = _profile(options, index, remain)
        if state.champion is not None:
            tune_options = dataclasses.replace(
                tune_options,
                initial_gpu_layers=state.champion.gpu_layers,
                initial_cpu_moe=state.champion.moe_cpu_layers,
            )
        from llamatune.search import run_tuning
        from llamatune.session import Session

        session = Session.create(
            options.sessions_dir,
            model=model,
            hardware=assess_hardware(),
            llama=llama,
            options=tune_options,
            argv=list(sys.argv),
        )
        run.append(
            {
                "type": "round_start",
                "index": index,
                "session_dir": str(session.dir),
                "budget": tune_options.budget_trials,
                "steering": {
                    tier: coverage.remaining_ids for tier, coverage in state.ledger.tiers.items()
                },
            }
        )
        _announce(
            state.reporter,
            f"[marathon] round {index}/{options.rounds_max} budget={tune_options.budget_trials}",
            {"phase": "rounds", "round": index, "budget": tune_options.budget_trials},
        )
        before = state.clock()
        outcome = run_tuning(
            session, state.hardware, model, llama, tune_options, reporter=state.reporter
        )
        wall = max(0.0, (state.clock() - before).total_seconds())
        if outcome.exit_code not in (0, 1):
            failures += 1
            item = {
                "type": "round_failed",
                "index": index,
                "session_dir": str(outcome.session_dir),
                "exit_code": outcome.exit_code,
                "wall_s": wall,
            }
            run.append(item)
            summary["failed"].append(item)
            if failures >= 3:
                summary["stop_reason"] = "circuit_breaker"
                return 3
            continue
        failures = 0
        state.session_dirs.append(outcome.session_dir)
        # Fold in only the just-finished session's journal: rescanning every
        # prior session per round made ledger builds O(R^2).
        delta_executed, delta_pruned, delta_trials = _trial_evidence((outcome.session_dir,))
        state.executed |= delta_executed
        state.pruned |= delta_pruned
        state.journaled_trials.extend(delta_trials)
        winner = outcome.analysis.get("winner")
        contender = (
            TrialConfig.from_dict(winner["config"])
            if isinstance(winner, dict) and isinstance(winner.get("config"), dict)
            else None
        )
        changed = False
        challenge_payload: dict[str, Any] | None = None
        if contender is not None and (
            state.champion is None or contender.trial_id != state.champion.trial_id
        ):
            from llamatune.abtest import run_ab

            challenge = run_ab(
                run,
                state.champion,
                contender,
                blocks=max(3, options.ab_blocks - 2),
                model=model,
                llama=llama,
                options=options,
                label=f"challenge-{index}",
            )
            challenge_payload = cast(dict[str, Any], _jsonable(challenge))
            summary["challenges"].append(challenge_payload)
            run.append({"type": "challenge", "round": index, **challenge_payload})
            if challenge.verdict == "b":
                state.champion, state.champion_session, changed = (
                    contender,
                    outcome.session_dir,
                    True,
                )
        no_change = 0 if changed else no_change + 1
        known = (
            ()
            if state.champion is None
            else ((state.champion.gpu_layers, state.champion.moe_cpu_layers),)
        )
        state.ledger = build_ledger(
            enumerate_space(
                model,
                llama,
                state.hardware,
                options,
                champion=state.champion or state.default_config,
                known_placements=known,
            ),
            state.executed,
            state.pruned,
            trials=state.journaled_trials,
        )
        item = {
            "type": "round_end",
            "index": index,
            "session_dir": str(outcome.session_dir),
            "exit_code": outcome.exit_code,
            "winner_config": contender.to_dict() if contender else None,
            "champion_changed": changed,
            "wall_s": wall,
            "coverage_pct": coverage_percent(state.ledger),
            "challenge": challenge_payload,
        }
        run.append(item)
        summary["rounds"].append(item)
        run.append(
            {
                "type": "coverage_snapshot",
                "round": index,
                "ledger": _jsonable(state.ledger),
            }
        )
        if has_converged(no_change, options.converge_rounds, state.ledger):
            summary["stop_reason"] = "converged"
            break
    else:
        summary["stop_reason"] = "rounds_max"
    summary["ledger"] = _jsonable(state.ledger)
    summary["champion"] = {
        "config": state.champion.to_dict() if state.champion else None,
        "session_dir": str(state.champion_session) if state.champion_session else None,
        "defaults": state.champion is None,
    }
    return None


def _run_matrix_phase(state: _MarathonState) -> int | None:
    """Run the context-by-depth matrix; returns a finalize exit code or None."""
    from llamatune.matrix import run_matrix

    run, model, llama, options = state.run, state.model, state.llama, state.options
    summary = state.summary
    run.append({"type": "phase", "phase": "matrix"})
    _announce(state.reporter, "[marathon] phase matrix", {"phase": "matrix"})
    if not state.tracker.bracket("matrix"):
        summary["stop_reason"] = "bracket_circuit_breaker"
        return 3
    remain = remaining_minutes(state.deadline, state.clock())
    if remain is None or remain >= MATRIX_MIN_MINUTES:
        run.champion_evidence = _champion_evidence(state)
        cells = run_matrix(
            run,
            state.champion or state.default_config,
            model,
            llama,
            options,
            remaining_minutes_fn=lambda: (
                float("inf")
                if state.deadline is None
                else max(
                    0.0,
                    cast(float, remaining_minutes(state.deadline, state.clock()))
                    - ab_reserved_minutes(options.ab_blocks),
                )
            ),
        )
        summary["matrix"] = _jsonable(cells)
    else:
        summary["deferred"].append({"kind": "matrix", "reason": "insufficient time"})
        run.append({"type": "deferred", **summary["deferred"][-1]})
    return None


def _run_verification_phase(state: _MarathonState) -> int:
    """Verify the champion against defaults; returns the final exit code."""
    run, summary = state.run, state.summary
    exit_code = 1 if summary["failed"] else 0
    run.append({"type": "phase", "phase": "verification"})
    _announce(state.reporter, "[marathon] phase verification", {"phase": "verification"})
    if not state.tracker.bracket("verification"):
        summary["stop_reason"] = "bracket_circuit_breaker"
        return 3
    if state.champion is None:
        summary["warnings"].append("final verification skipped: defaults remain champion")
        summary["verification"] = {
            "status": "skipped",
            "reason": "defaults champion",
        }
        return 1
    from llamatune.abtest import run_ab

    verified = run_ab(
        run,
        None,
        state.champion,
        blocks=state.options.ab_blocks,
        model=state.model,
        llama=state.llama,
        options=state.options,
        label="final",
    )
    summary["ab"] = _jsonable(verified)
    replicated = verified.verdict == "b"
    summary["verification"] = {
        "status": "replicated" if replicated else "not replicated",
        "verdict": verified.verdict,
    }
    if not replicated:
        summary["warnings"].append("not replicated: champion did not beat defaults in final A/B")
        return 1
    return exit_code


def run_marathon(
    options: MarathonOptions,
    *,
    now_fn: Callable[[], Any] | None = None,
    reporter: Reporter | None = None,
) -> MarathonOutcome:
    """Run or naturally resume an exhaustive, strictly serial Marathon."""
    from llamatune.coverage import build_ledger, enumerate_space
    from llamatune.hardware import assess_hardware
    from llamatune.llama import discover_llama
    from llamatune.model import inspect_model

    clock = now_fn or (lambda: datetime.now(UTC))
    start = clock()
    if not isinstance(start, datetime):
        raise TypeError("now_fn must return datetime")
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    deadline = resolve_deadline(start, options.until, options.max_hours)
    try:
        hardware, llama, model = (
            assess_hardware(),
            discover_llama(options.llama_bin),
            inspect_model(options.model_path, full_hash=options.full_hash),
        )
    except Exception as exc:
        return MarathonOutcome(
            run_dir=options.sessions_dir / "marathon",
            summary={"schema_version": 1, "error": str(exc)},
            exit_code=3,
        )
    prior, mismatches = find_reentry(options.sessions_dir, options, model.fingerprint)
    run = (
        MarathonRun.load(prior)
        if prior
        else MarathonRun.create(
            options.sessions_dir,
            options=options,
            hardware=hardware,
            llama=llama,
            model=model,
            argv=list(sys.argv),
        )
    )
    meta = json.loads((run.dir / "run.json").read_text(encoding="utf-8"))
    meta.update(
        identity=identity(options, model.fingerprint),
        resolved_deadline=deadline.isoformat() if deadline else None,
    )
    run.write_json("run.json", meta)
    plan = _plan(options)
    run.write_json("plan.json", plan)
    run.append({"type": "plan", **plan})
    summary: dict[str, Any] = {
        "schema_version": 1,
        "options": _jsonable(options),
        "window": {
            "started": start.isoformat(),
            "deadline": deadline.isoformat() if deadline else None,
        },
        "plan": plan,
        "rounds": [],
        "challenges": [],
        "matrix": [],
        "ab": None,
        "warnings": [
            f"left unmatched interrupted marathon untouched: {path}" for path in mismatches
        ],
        "deferred": [],
        "failed": [],
    }
    if options.dry_run:
        summary["stop_reason"] = "dry_run"
        return _finalize(run, summary, 0)

    with install_interrupt_handlers() as interrupt_state:
        try:
            entries = _entries(run.dir)
            recon_entry = next(
                (entry for entry in reversed(entries) if entry.get("type") == "recon_complete"),
                None,
            )
            run.append({"type": "phase", "phase": "resume"})
            _announce(reporter, "[marathon] phase resume", {"phase": "resume"})
            incomplete = next(
                (
                    entry
                    for entry in reversed(entries)
                    if entry.get("type") == "round_start"
                    and not any(
                        end.get("type") in {"round_end", "round_failed"}
                        and end.get("index") == entry.get("index")
                        for end in entries
                    )
                ),
                None,
            )
            if incomplete and not interrupt_state.stop_requested:
                _resume_incomplete_round(run, incomplete, reporter)
                entries = _entries(run.dir)
            run.append({"type": "phase", "phase": "reconnaissance"})
            _announce(reporter, "[marathon] phase reconnaissance", {"phase": "reconnaissance"})
            reference = _resolve_reference(run, model, llama, options, recon_entry)
            if reference is None:
                summary["stop_reason"] = "reconnaissance_failed"
                summary["failed"].append(
                    {"kind": "reconnaissance", "reason": "both attempts failed"}
                )
                return _finalize(run, summary, 3, interrupt_state=interrupt_state)
            if recon_entry is None:
                run.append({"type": "recon_complete", "reference": reference})
            summary["reference"] = {
                "original": reference,
                "current": reference,
                "rebaselines": [],
            }
            tracker = BracketTracker(run, model, llama, options, summary, clock)
            tracker.arm(reference)
            if reference.get("warmup_drift"):
                summary["warnings"].append(
                    "warmup_drift: reconnaissance halves differed beyond threshold"
                )

            completed_rounds = [
                entry for entry in _entries(run.dir) if entry.get("type") == "round_end"
            ]
            champion, champion_session = _restore_champion(completed_rounds)
            summary["rounds"] = [dict(entry) for entry in completed_rounds]
            default_config = _initial_default_config(reference, hardware)
            session_dirs = [
                Path(str(entry["session_dir"]))
                for entry in completed_rounds
                if entry.get("session_dir")
            ]
            executed, pruned, journaled_trials = _trial_evidence(session_dirs)
            ledger = build_ledger(
                enumerate_space(
                    model,
                    llama,
                    hardware,
                    options,
                    champion=champion or default_config,
                    known_placements=(),
                ),
                executed,
                pruned,
                trials=journaled_trials,
            )
            state = _MarathonState(
                run=run,
                model=model,
                llama=llama,
                options=options,
                hardware=hardware,
                summary=summary,
                clock=clock,
                deadline=deadline,
                interrupt_state=interrupt_state,
                reporter=reporter,
                tracker=tracker,
                completed_rounds=completed_rounds,
                champion=champion,
                champion_session=champion_session,
                default_config=default_config,
                ledger=ledger,
                session_dirs=session_dirs,
                executed=executed,
                pruned=pruned,
                journaled_trials=journaled_trials,
            )

            code = _run_rounds_phase(state)
            if code is not None:
                return _finalize(run, summary, code, interrupt_state=interrupt_state)

            _journal_interrupts(run, interrupt_state)
            if interrupt_state.stop_requested:
                summary["stop_reason"] = "interrupted"
                return _finalize(run, summary, 4, interrupt_state=interrupt_state)
            code = _run_matrix_phase(state)
            if code is not None:
                return _finalize(run, summary, code, interrupt_state=interrupt_state)
            exit_code = _run_verification_phase(state)
            run.append({"type": "phase", "phase": "report"})
            _announce(reporter, "[marathon] phase report", {"phase": "report"})
            return _finalize(run, summary, exit_code, interrupt_state=interrupt_state)
        except KeyboardInterrupt:
            summary["stop_reason"] = "interrupted"
            _journal_interrupts(run, interrupt_state)
            run.append({"type": "interrupted", "immediate": interrupt_state.second_signal})
            return _finalize(run, summary, 4, interrupt_state=interrupt_state)
