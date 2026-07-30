"""Serial orchestration and confined evidence writer for Marathon."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import secrets
import signal
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any, Self, cast

from llamatune._version import __version__
from llamatune.types import (
    CoverageLedger,
    HardwareReport,
    LlamaCppReport,
    MarathonOptions,
    MarathonOutcome,
    ModelReport,
    NightshiftOptions,
    RegistryRecord,
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
_CREATE_RETRIES = 5
_RECON_TIMEOUT_S = 1800.0


class MarathonPathError(Exception):
    """A requested artifact path escaped its Marathon run directory."""


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def _confine(base: Path, *parts: str) -> Path:
    root = base.resolve()
    candidate = base.joinpath(*parts)
    try:
        candidate.resolve().relative_to(root)
    except ValueError:
        raise MarathonPathError(f"path {candidate} escapes marathon directory {base}") from None
    return candidate


class MarathonRun:
    """The sole writer inside one Marathon evidence directory."""

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
        root.mkdir(parents=True, exist_ok=True)
        stem = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in model.path.stem)
        run_dir: Path | None = None
        for _ in range(_CREATE_RETRIES):
            candidate = root / f"{stem}-{datetime.now(UTC):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
            try:
                candidate.mkdir(exist_ok=False)
            except FileExistsError:
                continue
            run_dir = candidate
            break
        if run_dir is None:
            raise MarathonPathError(f"could not allocate a unique run under {root}")
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

    def append(self, entry: dict[str, Any]) -> None:
        record = cast(dict[str, Any], _jsonable(dict(entry)))
        record.setdefault("ts", _utc_iso())
        path = _confine(self._dir, "journal.jsonl")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

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

    def write_json(self, name: str, payload: dict[str, Any]) -> None:
        path = _confine(self._dir, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def write_text(self, name: str, text: str) -> None:
        path = _confine(self._dir, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def resolve_deadline(
    start: datetime,
    until: str | None,
    max_hours: float | None,
    *,
    local_tz: tzinfo | None = None,
) -> datetime | None:
    """Resolve the earlier supplied deadline, with past wall times meaning tomorrow."""
    candidates: list[datetime] = []
    if until is not None:
        hour, minute = (int(part) for part in until.split(":"))
        zone = local_tz or datetime.now().astimezone().tzinfo or UTC
        local = start.astimezone(zone)
        candidate = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= local:
            candidate += timedelta(days=1)
        candidates.append(candidate.astimezone(UTC))
    if max_hours is not None:
        candidates.append(start + timedelta(hours=max_hours))
    return min(candidates) if candidates else None


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
    path = run_dir / "journal.jsonl"
    if not path.is_file():
        return []
    result: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            break
        if isinstance(value, dict):
            result.append(value)
    return result


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


def _finalize(run: MarathonRun, summary: dict[str, Any], exit_code: int) -> MarathonOutcome:
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


def run_marathon(
    options: MarathonOptions, *, now_fn: Callable[[], Any] | None = None
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

    stop = False
    second = False
    old_handlers: dict[signal.Signals, Any] = {}

    def handle(signum: int, _frame: Any) -> None:
        nonlocal stop, second
        if stop:
            second = True
            raise KeyboardInterrupt
        stop = True
        run.append({"type": "interrupted", "signal": signum, "immediate": False})

    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, handle)
    try:
        entries = _entries(run.dir)
        recon_entry = next(
            (entry for entry in reversed(entries) if entry.get("type") == "recon_complete"),
            None,
        )
        run.append({"type": "phase", "phase": "resume"})
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
        if incomplete and not stop:
            from llamatune.search import resume_tuning

            outcome = resume_tuning(Path(str(incomplete["session_dir"])))
            run.append(
                {
                    "type": "round_end" if outcome.exit_code in (0, 1) else "round_failed",
                    "index": incomplete["index"],
                    "session_dir": str(outcome.session_dir),
                    "exit_code": outcome.exit_code,
                    "resumed": True,
                }
            )
            entries = _entries(run.dir)
        run.append({"type": "phase", "phase": "reconnaissance"})
        reference_value = (
            dict(recon_entry["reference"]) if recon_entry else _recon(run, model, llama, options)
        )
        if reference_value is None:
            reference_value = _recon(run, model, llama, options)
        if reference_value is None:
            summary["stop_reason"] = "reconnaissance_failed"
            summary["failed"].append({"kind": "reconnaissance", "reason": "both attempts failed"})
            return _finalize(run, summary, 3)
        reference: dict[str, Any] = reference_value
        if recon_entry is None:
            run.append({"type": "recon_complete", "reference": reference})
        summary["reference"] = {
            "original": reference,
            "current": reference,
            "rebaselines": [],
        }
        run.decision_threshold = max(0.01, 2.0 * float(reference["cv_ref"]))
        if reference.get("warmup_drift"):
            summary["warnings"].append(
                "warmup_drift: reconnaissance halves differed beyond threshold"
            )

        bracket_number = 0
        bracket_errors = 0
        last_bracket: datetime | None = None

        def bracket(phase: str) -> bool:
            nonlocal bracket_number, bracket_errors, last_bracket, reference
            current = clock()
            if (
                last_bracket is not None
                and (current - last_bracket).total_seconds() < BRACKET_FRESH_MINUTES * 60
            ):
                return True
            from llamatune.calibrate import run_calibration

            bracket_number += 1
            run.select_bracket(bracket_number)
            result = run_calibration(
                run,
                _synth_record(reference, model, llama, options, run),
                model,
                llama,
                _bracket_options(options),
            )
            payload = cast(dict[str, Any], _jsonable(result))
            run.append({"type": "bracket", "phase": phase, "number": bracket_number, **payload})
            last_bracket = current
            if result.verdict == "error":
                bracket_errors += 1
                summary["failed"].append(
                    {"kind": "bracket", "phase": phase, "reason": result.reason}
                )
                return bracket_errors < 3
            bracket_errors = 0
            if result.verdict == "drift" and result.pp is not None and result.tg is not None:
                old = dict(reference)
                reference = {
                    **reference,
                    "pp": result.pp.mean,
                    "tg": result.tg.mean,
                    "cv_ref": max(0.01, result.pp.cv, result.tg.cv),
                }
                run.decision_threshold = max(0.01, 2.0 * float(reference["cv_ref"]))
                summary["reference"]["current"] = reference
                summary["reference"]["rebaselines"].append(
                    {"phase": phase, "old": old, "new": dict(reference)}
                )
                run.append(
                    {
                        "type": "environment_drift",
                        "phase": phase,
                        "old_reference": old,
                        "new_reference": reference,
                    }
                )
                if len(summary["reference"]["rebaselines"]) >= 2:
                    summary["warnings"].append(
                        "environment unstable: consecutive phase brackets drifted"
                    )
            return True

        champion: TrialConfig | None = None
        champion_session: Path | None = None
        completed_rounds = [
            entry for entry in _entries(run.dir) if entry.get("type") == "round_end"
        ]
        for completed in completed_rounds:
            config = completed.get("winner_config")
            if completed.get("champion_changed") and isinstance(config, dict):
                champion = TrialConfig.from_dict(config)
                champion_session = Path(str(completed["session_dir"]))
        summary["rounds"] = [dict(entry) for entry in completed_rounds]
        default_data = reference.get("default_config")
        default_config = (
            TrialConfig.from_dict(default_data)
            if isinstance(default_data, dict)
            else TrialConfig(
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
        )
        session_dirs = [
            Path(str(entry["session_dir"]))
            for entry in completed_rounds
            if entry.get("session_dir")
        ]
        no_change = 0
        failures = 0
        ledger = build_ledger(
            enumerate_space(
                model,
                llama,
                hardware,
                options,
                champion=champion or default_config,
                known_placements=(),
            ),
            *_trial_evidence(session_dirs)[:2],
            trials=_trial_evidence(session_dirs)[2],
        )
        run.append({"type": "phase", "phase": "rounds"})
        if not bracket("rounds"):
            summary["stop_reason"] = "bracket_circuit_breaker"
            return _finalize(run, summary, 3)
        for index in range(len(completed_rounds) + 1, options.rounds_max + 1):
            if stop:
                break
            remain = remaining_minutes(deadline, clock())
            if not round_fits(
                remain,
                ab_reserved=(ab_reserved_minutes(options.ab_blocks) if deadline else 0.0),
            ):
                summary["stop_reason"] = "deadline"
                summary["deferred"].append(
                    {"kind": "round", "index": index, "reason": "insufficient time"}
                )
                run.append({"type": "deferred", **summary["deferred"][-1]})
                break
            tune_options = _profile(options, index, remain)
            if champion is not None:
                tune_options = dataclasses.replace(
                    tune_options,
                    initial_gpu_layers=champion.gpu_layers,
                    initial_cpu_moe=champion.moe_cpu_layers,
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
                        tier: coverage.remaining_ids for tier, coverage in ledger.tiers.items()
                    },
                }
            )
            before = clock()
            outcome = run_tuning(session, hardware, model, llama, tune_options)
            wall = max(0.0, (clock() - before).total_seconds())
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
                    return _finalize(run, summary, 3)
                continue
            failures = 0
            session_dirs.append(outcome.session_dir)
            winner = outcome.analysis.get("winner")
            contender = (
                TrialConfig.from_dict(winner["config"])
                if isinstance(winner, dict) and isinstance(winner.get("config"), dict)
                else None
            )
            changed = False
            challenge_payload: dict[str, Any] | None = None
            if contender is not None and (
                champion is None or contender.trial_id != champion.trial_id
            ):
                from llamatune.abtest import run_ab

                challenge = run_ab(
                    run,
                    champion,
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
                    champion, champion_session, changed = (
                        contender,
                        outcome.session_dir,
                        True,
                    )
            no_change = 0 if changed else no_change + 1
            executed, pruned, trials = _trial_evidence(session_dirs)
            known = () if champion is None else ((champion.gpu_layers, champion.moe_cpu_layers),)
            ledger = build_ledger(
                enumerate_space(
                    model,
                    llama,
                    hardware,
                    options,
                    champion=champion or default_config,
                    known_placements=known,
                ),
                executed,
                pruned,
                trials=trials,
            )
            item = {
                "type": "round_end",
                "index": index,
                "session_dir": str(outcome.session_dir),
                "exit_code": outcome.exit_code,
                "winner_config": contender.to_dict() if contender else None,
                "champion_changed": changed,
                "wall_s": wall,
                "coverage_pct": coverage_percent(ledger),
                "challenge": challenge_payload,
            }
            run.append(item)
            summary["rounds"].append(item)
            run.append(
                {
                    "type": "coverage_snapshot",
                    "round": index,
                    "ledger": _jsonable(ledger),
                }
            )
            if has_converged(no_change, options.converge_rounds, ledger):
                summary["stop_reason"] = "converged"
                break
        else:
            summary["stop_reason"] = "rounds_max"
        summary["ledger"] = _jsonable(ledger)
        summary["champion"] = {
            "config": champion.to_dict() if champion else None,
            "session_dir": str(champion_session) if champion_session else None,
            "defaults": champion is None,
        }

        if stop:
            summary["stop_reason"] = "interrupted"
            return _finalize(run, summary, 4)
        run.append({"type": "phase", "phase": "matrix"})
        if not bracket("matrix"):
            summary["stop_reason"] = "bracket_circuit_breaker"
            return _finalize(run, summary, 3)
        remain = remaining_minutes(deadline, clock())
        if remain is None or remain >= MATRIX_MIN_MINUTES:
            from llamatune.matrix import run_matrix

            if champion_session is not None:
                try:
                    champion_analysis = json.loads(
                        (champion_session / "analysis.json").read_text(encoding="utf-8")
                    )
                    winner = champion_analysis.get("winner")
                    validated = champion_analysis.get("feasibility", {}).get("ctx_validated")
                    if (
                        isinstance(winner, dict)
                        and winner.get("confirmed") is True
                        and isinstance(validated, int)
                        and options.ctx_size is not None
                        and validated >= options.ctx_size
                    ):
                        run.champion_evidence = {
                            "ctx": options.ctx_size,
                            "config": winner.get("config"),
                            "pp": winner.get("pp"),
                            "tg": winner.get("tg"),
                            "evidence": str(champion_session / "analysis.json"),
                        }
                except (OSError, TypeError, ValueError):
                    run.champion_evidence = None

            cells = run_matrix(
                run,
                champion or default_config,
                model,
                llama,
                options,
                remaining_minutes_fn=lambda: (
                    float("inf")
                    if deadline is None
                    else max(
                        0.0,
                        cast(float, remaining_minutes(deadline, clock()))
                        - ab_reserved_minutes(options.ab_blocks),
                    )
                ),
            )
            summary["matrix"] = _jsonable(cells)
        else:
            summary["deferred"].append({"kind": "matrix", "reason": "insufficient time"})
            run.append({"type": "deferred", **summary["deferred"][-1]})

        exit_code = 1 if summary["failed"] else 0
        run.append({"type": "phase", "phase": "verification"})
        if not bracket("verification"):
            summary["stop_reason"] = "bracket_circuit_breaker"
            return _finalize(run, summary, 3)
        if champion is None:
            summary["warnings"].append("final verification skipped: defaults remain champion")
            summary["verification"] = {
                "status": "skipped",
                "reason": "defaults champion",
            }
            exit_code = 1
        else:
            from llamatune.abtest import run_ab

            verified = run_ab(
                run,
                None,
                champion,
                blocks=options.ab_blocks,
                model=model,
                llama=llama,
                options=options,
                label="final",
            )
            summary["ab"] = _jsonable(verified)
            replicated = verified.verdict == "b"
            summary["verification"] = {
                "status": "replicated" if replicated else "not replicated",
                "verdict": verified.verdict,
            }
            if not replicated:
                summary["warnings"].append(
                    "not replicated: champion did not beat defaults in final A/B"
                )
                exit_code = 1
        run.append({"type": "phase", "phase": "report"})
        return _finalize(run, summary, exit_code)
    except KeyboardInterrupt:
        summary["stop_reason"] = "interrupted"
        run.append({"type": "interrupted", "immediate": second})
        return _finalize(run, summary, 4)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
