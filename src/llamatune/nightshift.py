"""Serial, evidence-first orchestration for unattended tuning shifts."""

from __future__ import annotations

import json
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

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
from llamatune.session import JournalTailIncomplete, scan_journal_tail
from llamatune.types import (
    CalibrationResult,
    DiscoveredModel,
    HardwareReport,
    LlamaCppReport,
    NightshiftOptions,
    NightshiftOutcome,
    RegistryRecord,
    Reporter,
    TuneOptions,
    WorkItem,
)

MIN_TUNE_MINUTES = 20.0
SHUTDOWN_MARGIN_MIN = 5.0
_CALIBRATION_FALLBACK_S = 300.0
_CALIBRATION_SAFETY_FACTOR = 1.5

_PROFILES: dict[str, dict[str, float | int]] = {
    "standard": {
        "budget_trials": 60,
        "reps_search": 3,
        "reps_confirm": 5,
        "baseline_runs": 3,
        "cooldown_s": 0.0,
    },
    "deep": {
        "budget_trials": 120,
        "reps_search": 5,
        "reps_confirm": 8,
        "baseline_runs": 5,
        "cooldown_s": 5.0,
    },
}


class NightshiftPathError(PathEscapeError):
    """A requested run artifact path escaped the Night Shift directory."""


def _confine(base: Path, *parts: str) -> Path:
    return confined_path(base, *parts, error=NightshiftPathError)


class NightshiftRun(EvidenceWriter):
    """The only writer allowed inside one Night Shift evidence directory."""

    path_error: type[PathEscapeError] = NightshiftPathError

    def __init__(self, run_dir: Path) -> None:
        self._dir = run_dir
        self.dir = run_dir

    @classmethod
    def create(
        cls,
        sessions_dir: Path,
        *,
        options: NightshiftOptions,
        hardware: HardwareReport,
        llama: LlamaCppReport,
        argv: Sequence[str],
    ) -> NightshiftRun:
        root = sessions_dir / "nightshift"
        run_dir = create_unique_dir(root, "", error=NightshiftPathError)
        run = cls(run_dir)
        _confine(run_dir, "calibrations").mkdir(exist_ok=True)
        run.write_json(
            "run.json",
            {
                "schema_version": 1,
                "tool_version": __version__,
                "argv": list(argv),
                "options": _jsonable(options),
                "created": _utc_iso(),
            },
        )
        run.write_json("hardware.json", cast(dict[str, Any], _jsonable(hardware)))
        run.write_json("llamacpp.json", cast(dict[str, Any], _jsonable(llama)))
        run.append({"type": "nightshift_start", "tool_version": __version__})
        return run

    @classmethod
    def load(cls, run_dir: Path) -> NightshiftRun:
        path = Path(run_dir)
        if not (path / "run.json").is_file():
            raise FileNotFoundError(f"not a Night Shift run: {path}")
        return cls(path)

    def calibration_dir(self, fingerprint16: str, n: int) -> Path:
        invalid = not fingerprint16 or any(
            character not in "0123456789abcdefABCDEF" for character in fingerprint16
        )
        if invalid and fingerprint16 != "bootstrap":
            raise NightshiftPathError("calibration fingerprint must be hexadecimal")
        path = _confine(self._dir, "calibrations", fingerprint16, f"run-{n}")
        path.mkdir(parents=True, exist_ok=True)
        return path


def profile_values(options: NightshiftOptions, *, deepen: bool = False) -> dict[str, float | int]:
    """Return resolved profile values without mutating scheduler state."""
    values = dict(_PROFILES[options.profile])
    if deepen:
        values.update(
            budget_trials=240, reps_search=5, reps_confirm=8, baseline_runs=5, cooldown_s=5.0
        )
    for name in ("budget_trials", "reps_search", "reps_confirm", "baseline_runs", "cooldown_s"):
        override = getattr(options, name)
        if override is not None:
            values[name] = override
    return values


def deepening_changes_profile(options: NightshiftOptions) -> bool:
    """Whether a spare-time pass would do more work than the initial tune."""
    return profile_values(options, deepen=True) != profile_values(options, deepen=False)


def calibration_estimate_minutes(record: RegistryRecord, runs: int) -> float:
    seconds = record.median_trial_wall_s or _CALIBRATION_FALLBACK_S
    return runs * seconds * _CALIBRATION_SAFETY_FACTOR / 60.0


def _context_compatible_records(
    records: dict[str, RegistryRecord], ctx_size: int | None
) -> dict[str, RegistryRecord]:
    if ctx_size is None:
        return records
    return {
        fingerprint: record
        for fingerprint, record in records.items()
        if record.ctx_size is not None and record.ctx_size >= ctx_size
    }


def build_initial_plan(
    models: Sequence[DiscoveredModel],
    records: dict[str, RegistryRecord],
    incomplete: Sequence[tuple[Path, str]],
    *,
    calibration_runs: int,
) -> tuple[WorkItem, ...]:
    """Build phases 0--2 deterministically from an evidence snapshot."""
    by_fingerprint = {model.report.fingerprint: model for model in models}
    incomplete_fingerprints: set[str] = set()
    plan: list[WorkItem] = []
    for session_dir, fingerprint in incomplete:
        model = by_fingerprint.get(fingerprint)
        if model is None:
            continue
        incomplete_fingerprints.add(fingerprint)
        plan.append(
            WorkItem(
                kind="resume",
                model_path=model.path,
                fingerprint=fingerprint,
                session_dir=session_dir,
                reference_fingerprint=None,
                estimated_minutes=None,
                reason="incomplete in-scope session",
            )
        )

    new_models = sorted(
        (
            model
            for model in models
            if model.representative
            and model.report.fingerprint not in records
            and model.report.fingerprint not in incomplete_fingerprints
        ),
        key=lambda model: (model.report.size_bytes, str(model.path)),
    )
    plan.extend(
        WorkItem(
            kind="tune",
            model_path=model.path,
            fingerprint=model.report.fingerprint,
            session_dir=None,
            reference_fingerprint=None,
            estimated_minutes=None,
            reason="no completed session",
        )
        for model in new_models
    )

    own = sorted(
        (model for model in models if model.report.fingerprint in records),
        key=lambda model: (records[model.report.fingerprint].created, str(model.path)),
    )
    for model in own:
        record = records[model.report.fingerprint]
        plan.append(
            WorkItem(
                kind="calibrate",
                model_path=model.path,
                fingerprint=model.report.fingerprint,
                session_dir=None,
                reference_fingerprint=model.report.fingerprint,
                estimated_minutes=calibration_estimate_minutes(record, calibration_runs),
                reason="verify latest completed session",
            )
        )

    representatives = {
        model.group_key: model for model in models if model.group_key and model.representative
    }
    siblings = sorted(
        (
            model
            for model in models
            if not model.representative and model.report.fingerprint not in records
        ),
        key=lambda model: str(model.path),
    )
    for model in siblings:
        representative = (
            representatives.get(model.group_key) if model.group_key is not None else None
        )
        if representative is None:
            continue
        reference = records.get(representative.report.fingerprint)
        if reference is None:
            continue
        plan.append(
            WorkItem(
                kind="calibrate",
                model_path=model.path,
                fingerprint=model.report.fingerprint,
                session_dir=None,
                reference_fingerprint=reference.fingerprint,
                estimated_minutes=calibration_estimate_minutes(reference, calibration_runs),
                reason="transfer calibration from content-group representative",
            )
        )
    return tuple(plan)


def _session_fingerprint(path: Path) -> str | None:
    try:
        payload = json.loads((path / "model.json").read_text(encoding="utf-8"))
        return str(payload["fingerprint"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _interrupted_by_end(entry: Mapping[str, Any]) -> bool:
    return entry.get("type") == "session_end" and entry.get("reason") in {
        "stopped_by_user",
        "interrupted",
    }


def _interrupted_session(path: Path) -> bool:
    found = scan_journal_tail(path / "journal.jsonl", _interrupted_by_end)
    if isinstance(found, JournalTailIncomplete):
        entries, _corruption = read_journal_lines(path / "journal.jsonl")
        return any(_interrupted_by_end(entry) for entry in entries)
    return found is not None


def _tune_options(
    options: NightshiftOptions,
    hardware: HardwareReport,
    remaining: float | None,
    *,
    deepen: bool,
    depth: int | None = None,
) -> TuneOptions:
    profile = profile_values(options, deepen=deepen)
    return TuneOptions(
        target=options.target,
        budget_trials=int(profile["budget_trials"]),
        budget_minutes=(remaining - SHUTDOWN_MARGIN_MIN if remaining is not None else None),
        reps_search=int(profile["reps_search"]),
        reps_confirm=int(profile["reps_confirm"]),
        baseline_runs=int(profile["baseline_runs"]),
        pp=512,
        tg=128,
        allow_lossy=options.allow_lossy,
        cooldown_s=float(profile["cooldown_s"]),
        baseline_only=False,
        llama_bin=options.llama_bin,
        sessions_dir=options.sessions_dir,
        full_hash=options.full_hash,
        ctx_size=options.ctx_size,
        ctx_ladder=options.ctx_ladder,
        depth=depth if depth is not None else options.depth,
        vram_reserve_mb=options.vram_reserve_mb,
        quiet_load=hardware.physical_cores / 2,
    )


def _item_dict(item: WorkItem) -> dict[str, Any]:
    return cast(dict[str, Any], _jsonable(item))


def _announce_item(reporter: Reporter | None, index: int, total: int, item: WorkItem) -> None:
    """Emit one concise progress line for an orchestrator work item."""
    if reporter is None:
        return
    raw_label = (
        Path(str(item.model_path)).stem
        if item.model_path is not None
        else item.fingerprint or "item"
    )
    label = strip_control_chars(raw_label)
    message = f"[nightshift] item {index}/{total} {item.kind} {label}"
    from llamatune.types import ProgressEvent

    reporter.emit(
        ProgressEvent(
            kind="orchestrator_item",
            ts=_utc_iso(),
            payload={
                "message": message,
                "index": index,
                "total": total,
                "item_kind": item.kind,
                "model": label,
            },
        )
    )
    from llamatune import ui

    if isinstance(reporter, ui.PlainReporter):
        reporter.err.write(message + "\n")
        reporter.err.flush()


def _remaining_minutes(deadline: datetime | None, now: datetime) -> float | None:
    return None if deadline is None else max(0.0, (deadline - now).total_seconds() / 60.0)


def item_fits(item: WorkItem, remaining_minutes: float | None) -> bool:
    """Return whether an item may start in the remaining window."""
    if remaining_minutes is None:
        return True
    if item.kind == "calibrate":
        return item.estimated_minutes is None or item.estimated_minutes <= remaining_minutes
    return remaining_minutes >= MIN_TUNE_MINUTES


def deepen_order(
    models: Sequence[DiscoveredModel], records: dict[str, RegistryRecord]
) -> tuple[DiscoveredModel, ...]:
    """Return own-record models in least-recently-tuned order."""
    return tuple(
        sorted(
            (model for model in models if model.report.fingerprint in records),
            key=lambda model: (records[model.report.fingerprint].created, str(model.path)),
        )
    )


def tune_failure_breaker(history: Sequence[str]) -> bool:
    """Three consecutive failures for distinct models trip the breaker."""
    return len(history) >= 3 and len(set(history[-3:])) == 3


def _summary_counts(items: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = f"{item.get('kind', 'unknown')}:{item.get('outcome', 'unknown')}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _total_invocations(items: Sequence[dict[str, Any]]) -> int:
    total = 0
    for item in items:
        calibration = item.get("calibration")
        if isinstance(calibration, dict):
            total += int(calibration.get("runs", 0))
        session_dir = item.get("session_dir")
        if not isinstance(session_dir, str):
            continue
        try:
            entries, _corruption = read_journal_lines(Path(session_dir) / "journal.jsonl")
        except OSError:
            continue
        total += sum(
            1
            for entry in entries
            if entry.get("type") in {"baseline_run", "trial", "confirmation_run"}
        )
    return total


def _content_groups(models: Sequence[DiscoveredModel]) -> list[dict[str, Any]]:
    grouped: dict[str, list[DiscoveredModel]] = {}
    for model in models:
        if model.group_key is not None:
            grouped.setdefault(model.group_key, []).append(model)
    return [
        {
            "group_key": key,
            "representative": str(
                next((model.path for model in members if model.representative), members[0].path)
            ),
            "members": [str(model.path) for model in members],
        }
        for key, members in sorted(grouped.items())
    ]


def _finalize(
    run: NightshiftRun,
    options: NightshiftOptions,
    start: datetime,
    deadline: datetime | None,
    items: list[dict[str, Any]],
    warning_messages: list[str],
    exit_code: int,
    *,
    hardware: HardwareReport,
    llama: LlamaCppReport,
    models: Sequence[DiscoveredModel],
    stopped: bool = False,
    interrupt_state: InterruptState | None = None,
) -> NightshiftOutcome:
    if interrupt_state is not None:
        for signum, immediate in interrupt_state.drain_events():
            run.append({"type": "interrupted", "signal": signum, "immediate": immediate})
    summary: dict[str, Any] = {
        "schema_version": 1,
        "options": _jsonable(options),
        "window": {
            "started": start.isoformat(),
            "ended": _utc_iso(),
            "deadline": deadline.isoformat() if deadline else None,
            "outcome": "interrupted" if stopped else "completed",
        },
        "items": items,
        "counts": _summary_counts(items),
        "total_invocations": _total_invocations(items),
        "hardware": _jsonable(hardware),
        "llamacpp": _jsonable(llama),
        "content_groups": _content_groups(models),
        "warnings": warning_messages,
        "exit_code": exit_code,
        "constants": {
            "min_tune_minutes": MIN_TUNE_MINUTES,
            "shutdown_margin_minutes": SHUTDOWN_MARGIN_MIN,
        },
    }
    run.append({"type": "nightshift_end", "exit_code": exit_code, "stopped": stopped})
    run.write_json("nightshift.json", summary)
    from llamatune.nightreport import render

    run.write_text("nightshift-report.md", render(summary))
    return NightshiftOutcome(run_dir=run.dir, summary=summary, exit_code=exit_code)


@dataclass
class _ShiftState:
    """Mutable scheduling state threaded explicitly through Night Shift phases."""

    items: list[dict[str, Any]] = field(default_factory=list)
    warning_messages: list[str] = field(default_factory=list)
    failed: bool = False
    tune_failure_models: list[str] = field(default_factory=list)
    calibrated_fingerprints: set[str] = field(default_factory=set)
    tuned_fingerprints: set[str] = field(default_factory=set)
    total_items: int = 0
    started_items: int = 0
    phase_queues: dict[str, list[WorkItem]] = field(default_factory=dict)
    consumed: dict[str, int] = field(default_factory=dict)


def _journal_interrupts(run: NightshiftRun, state: InterruptState) -> None:
    """Journal queued interrupt events at the next safe main-flow point."""
    for signum, immediate in state.drain_events():
        run.append({"type": "interrupted", "signal": signum, "immediate": immediate})


def _deferred_item(item: WorkItem, reason: str) -> dict[str, Any]:
    return {**_item_dict(item), "outcome": "deferred", "reason": reason, "wall_s": 0.0}


def _startup_evidence(
    options: NightshiftOptions,
) -> tuple[HardwareReport, LlamaCppReport, list[str]]:
    """Assess hardware and locate llama.cpp, degrading to stubs on failure.

    Returns the reports plus accumulated startup error messages; a non-empty
    message list means the shift must end immediately with exit code 3.
    """
    from llamatune.hardware import assess_hardware
    from llamatune.llama import discover_llama

    startup_errors: list[str] = []
    try:
        hardware = assess_hardware()
    except Exception as exc:
        startup_errors.append(f"hardware assessment failed: {exc}")
        hardware = HardwareReport(
            os_name="unknown",
            arch="unknown",
            cpu_model="unknown",
            physical_cores=1,
            logical_cores=1,
            perf_cores=None,
            ram_mb=0,
            gpus=(),
            warnings=tuple(startup_errors),
        )
    try:
        llama = discover_llama(options.llama_bin)
    except Exception as exc:
        startup_errors.append(f"llama.cpp discovery failed: {exc}")
        llama = LlamaCppReport(
            bench_path=options.llama_bin or Path("llama-bench"),
            cli_path=None,
            server_path=None,
            capabilities=frozenset(),
            help_sha256="unavailable",
            build_commit=None,
            build_number=None,
        )
    if options.depth is not None and "d" not in llama.capabilities:
        startup_errors.append("llama-bench does not support -d required by --depth")
    return hardware, llama, startup_errors


def _discover_scope(
    options: NightshiftOptions, follow_symlinks: bool
) -> tuple[tuple[DiscoveredModel, ...], dict[str, RegistryRecord], tuple[Path, ...], list[str]]:
    """Discover models, build the registry, and list incomplete sessions."""
    from llamatune.discovery import discover_models
    from llamatune.registry import build_registry, incomplete_sessions

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        models = discover_models(
            options.models_dir,
            options.include,
            options.exclude,
            duplicates=options.duplicates,
            full_hash=options.full_hash,
            follow_symlinks=follow_symlinks,
        )
        records = _context_compatible_records(
            build_registry(options.sessions_dir), options.ctx_size
        )
        incomplete_paths = incomplete_sessions(options.sessions_dir)
    discovery_warnings = [str(item.message) for item in caught]
    return models, records, incomplete_paths, discovery_warnings


def _run_calibrate_item(
    run: NightshiftRun,
    state: _ShiftState,
    item: WorkItem,
    *,
    model: DiscoveredModel,
    reference: RegistryRecord,
    llama: LlamaCppReport,
    options: NightshiftOptions,
) -> tuple[dict[str, Any], WorkItem | None]:
    """Calibrate one model against its reference; maybe enqueue a retune.

    Returns the result record (without ``wall_s``) and an optional retune item
    the caller must queue and journal.
    """
    try:
        calibration = run_calibration(run, reference, model.report, llama, options)
    except Exception as exc:
        calibration = CalibrationResult(
            fingerprint=model.report.fingerprint,
            reference_session=reference.session_dir,
            verdict="error",
            pp=None,
            tg=None,
            drift_pp=None,
            drift_tg=None,
            threshold=max(options.drift_threshold, 2 * reference.noise_floor_cv),
            runs=options.calibration_runs,
            reason=str(exc),
            transfer_from=(
                reference.fingerprint if reference.fingerprint != model.report.fingerprint else None
            ),
            build_changed=reference.help_sha256 != llama.help_sha256,
            artifact_dir=None,
        )
    state.calibrated_fingerprints.add(model.report.fingerprint)
    result_record = {
        **_item_dict(item),
        "outcome": calibration.verdict,
        "calibration": _jsonable(calibration),
    }
    retune: WorkItem | None = None
    if calibration.verdict in {"drift", "error"}:
        retune = WorkItem(
            kind="retune",
            model_path=model.path,
            fingerprint=model.report.fingerprint,
            session_dir=None,
            reference_fingerprint=None,
            estimated_minutes=None,
            reason=f"calibration verdict: {calibration.verdict}",
            depth_workload=reference.depth_workload,
        )
    return result_record, retune


def _run_tune_class_item(
    run: NightshiftRun,
    state: _ShiftState,
    item: WorkItem,
    *,
    options: NightshiftOptions,
    models: Sequence[DiscoveredModel],
    by_fingerprint: dict[str, DiscoveredModel],
    records: dict[str, RegistryRecord],
    llama: LlamaCppReport,
    reporter: Reporter | None,
    remaining: float | None,
    interrupt_state: InterruptState,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[WorkItem]]:
    """Resume or tune one model item with per-item error recovery.

    Returns the result record (without ``wall_s``), deferred sibling records
    journaled after this item ends, and transfer calibrations to enqueue.
    """
    from llamatune.hardware import assess_hardware
    from llamatune.registry import absorb_session
    from llamatune.search import resume_tuning, run_tuning
    from llamatune.session import Session

    session_dir = item.session_dir
    try:
        if item.kind == "resume":
            if session_dir is None:
                raise ValueError("resume item has no session directory")
            outcome = resume_tuning(session_dir, reporter=reporter)
        else:
            model = by_fingerprint[item.fingerprint or ""]
            current_hardware = assess_hardware()
            tune_options = _tune_options(
                options,
                current_hardware,
                remaining,
                deepen=item.kind == "deepen",
                depth=item.depth_workload,
            )
            session = Session.create(
                options.sessions_dir,
                model=model.report,
                hardware=current_hardware,
                llama=llama,
                options=tune_options,
                argv=sys.argv,
            )
            session_dir = session.dir
            outcome = run_tuning(
                session,
                current_hardware,
                model.report,
                llama,
                tune_options,
                reporter=reporter,
            )
        result_record = {
            **_item_dict(item),
            "session_dir": str(session_dir),
            "outcome": ("succeeded" if outcome.exit_code in {0, 1} else "failed"),
            "tune_exit_code": outcome.exit_code,
        }
    except Exception as exc:
        result_record = {
            **_item_dict(item),
            "session_dir": str(session_dir) if session_dir else None,
            "outcome": "failed",
            "error": str(exc),
        }
    if session_dir is not None and _interrupted_session(session_dir):
        interrupt_state.stop_requested = True
        result_record["outcome"] = "interrupted"
    post_item_records: list[dict[str, Any]] = []
    transfers: list[WorkItem] = []
    if result_record["outcome"] == "failed":
        state.failed = True
        fingerprint = item.fingerprint or str(item.model_path)
        if not state.tune_failure_models or state.tune_failure_models[-1] != fingerprint:
            state.tune_failure_models.append(fingerprint)
        if item.kind == "tune" and item.fingerprint:
            representative = by_fingerprint[item.fingerprint]
            for sibling in models:
                if (
                    sibling.representative
                    or sibling.group_key != representative.group_key
                    or sibling.report.fingerprint in records
                ):
                    continue
                post_item_records.append(
                    {
                        "kind": "calibrate",
                        "model_path": str(sibling.path),
                        "fingerprint": sibling.report.fingerprint,
                        "reference_fingerprint": item.fingerprint,
                        "outcome": "deferred",
                        "reason": "representative tune failed",
                        "wall_s": 0.0,
                    }
                )
    else:
        state.tune_failure_models.clear()
        if item.fingerprint:
            state.tuned_fingerprints.add(item.fingerprint)
        if session_dir is not None:
            # Fold the just-finished session into the registry incrementally;
            # a full rescan is unnecessary under strictly serial scheduling.
            absorb_session(records, session_dir, ctx_size=options.ctx_size)
        if item.kind == "tune" and item.fingerprint in records:
            representative = by_fingerprint[item.fingerprint]
            for sibling in models:
                if (
                    sibling.representative
                    or sibling.group_key != representative.group_key
                    or sibling.report.fingerprint in records
                    or sibling.report.fingerprint in state.calibrated_fingerprints
                ):
                    continue
                reference = records[item.fingerprint]
                transfers.append(
                    WorkItem(
                        kind="calibrate",
                        model_path=sibling.path,
                        fingerprint=sibling.report.fingerprint,
                        session_dir=None,
                        reference_fingerprint=item.fingerprint,
                        estimated_minutes=calibration_estimate_minutes(
                            reference, options.calibration_runs
                        ),
                        reason="dynamic transfer after representative tune",
                    )
                )
    return result_record, post_item_records, transfers


def _run_deepen_candidate(
    item: WorkItem,
    *,
    options: NightshiftOptions,
    model: DiscoveredModel,
    llama: LlamaCppReport,
    remaining: float | None,
    clock: Callable[[], datetime],
) -> tuple[dict[str, Any], bool]:
    """Execute one spare-time deepen item; returns (record, interrupted)."""
    from llamatune.hardware import assess_hardware
    from llamatune.search import run_tuning
    from llamatune.session import Session

    item_started = clock()
    deepen_session_dir: Path | None = None
    interrupted: bool
    try:
        current_hardware = assess_hardware()
        tune_options = _tune_options(options, current_hardware, remaining, deepen=True)
        session = Session.create(
            options.sessions_dir,
            model=model.report,
            hardware=current_hardware,
            llama=llama,
            options=tune_options,
            argv=sys.argv,
        )
        deepen_session_dir = session.dir
        outcome = run_tuning(
            session,
            current_hardware,
            model.report,
            llama,
            tune_options,
        )
        interrupted = _interrupted_session(session.dir)
        result_record = {
            **_item_dict(item),
            "session_dir": str(session.dir),
            "outcome": (
                "interrupted"
                if interrupted
                else "succeeded"
                if outcome.exit_code in {0, 1}
                else "failed"
            ),
            "tune_exit_code": outcome.exit_code,
        }
    except Exception as exc:
        interrupted = False
        result_record = {
            **_item_dict(item),
            "session_dir": (str(deepen_session_dir) if deepen_session_dir else None),
            "outcome": "failed",
            "error": str(exc),
        }
    result_record["wall_s"] = max(0.0, (clock() - item_started).total_seconds())
    return result_record, interrupted


def _run_deepening(
    run: NightshiftRun,
    state: _ShiftState,
    *,
    options: NightshiftOptions,
    models: Sequence[DiscoveredModel],
    records: dict[str, RegistryRecord],
    llama: LlamaCppReport,
    deadline: datetime | None,
    clock: Callable[[], datetime],
    reporter: Reporter | None,
    interrupt_state: InterruptState,
) -> None:
    """Spend spare time deepening least-recently-tuned evidence, serially.

    ``records`` is maintained incrementally after each finished item (see
    absorb_session); it already equals a from-scratch rebuild under strictly
    serial scheduling.
    """
    for model in deepen_order(models, records):
        _journal_interrupts(run, interrupt_state)
        remaining = _remaining_minutes(deadline, clock())
        if remaining is None or remaining < MIN_TUNE_MINUTES or interrupt_state.stop_requested:
            break
        item = WorkItem(
            kind="deepen",
            model_path=model.path,
            fingerprint=model.report.fingerprint,
            session_dir=None,
            reference_fingerprint=None,
            estimated_minutes=None,
            reason="deadline has spare time; deepen least-recently-tuned evidence",
        )
        state.phase_queues["retune"].append(item)
        state.total_items += 1
        run.append({"type": "item_start", **_item_dict(item)})
        state.started_items += 1
        _announce_item(reporter, state.started_items, state.total_items, item)
        result_record, interrupted = _run_deepen_candidate(
            item, options=options, model=model, llama=llama, remaining=remaining, clock=clock
        )
        state.items.append(result_record)
        run.append({"type": "item_end", **result_record})
        if interrupted:
            interrupt_state.stop_requested = True
            break
        if result_record["outcome"] == "failed":
            state.failed = True
            if (
                not state.tune_failure_models
                or state.tune_failure_models[-1] != model.report.fingerprint
            ):
                state.tune_failure_models.append(model.report.fingerprint)
            if tune_failure_breaker(state.tune_failure_models):
                state.warning_messages.append(
                    "circuit breaker: three consecutive tune-class model failures"
                )
                interrupt_state.stop_requested = True
                break
        else:
            state.tune_failure_models.clear()


def _unconsumed_plan_items(
    phase_queues: Mapping[str, Sequence[Any]], consumed: Mapping[str, int]
) -> bool:
    """True when any phase queue still holds items the plan never reached."""
    return any(len(queue) > consumed.get(name, 0) for name, queue in phase_queues.items())


def _nightshift_exit_code(
    *,
    stop_requested: bool,
    second_signal: bool,
    signal_count: int,
    items: Sequence[Mapping[str, Any]],
    phase_queues: Mapping[str, Sequence[Any]],
    consumed: Mapping[str, int],
    failed: bool,
) -> int:
    """Resolve the process exit code from interruption, failure, and plan state."""
    interrupted_anywhere = any(item.get("outcome") == "interrupted" for item in items)
    if stop_requested and (
        second_signal
        or interrupted_anywhere
        or (signal_count > 0 and _unconsumed_plan_items(phase_queues, consumed))
    ):
        return 4
    return 1 if failed else 0


def _run_plan_phases(
    run: NightshiftRun,
    state: _ShiftState,
    *,
    options: NightshiftOptions,
    models: Sequence[DiscoveredModel],
    by_fingerprint: dict[str, DiscoveredModel],
    records: dict[str, RegistryRecord],
    llama: LlamaCppReport,
    deadline: datetime | None,
    clock: Callable[[], datetime],
    reporter: Reporter | None,
    interrupt_state: InterruptState,
) -> None:
    """Consume the resume/tune/calibrate/retune phase queues strictly serially."""
    for phase in ("resume", "tune", "calibrate", "retune"):
        queue = state.phase_queues[phase]
        index = 0
        while index < len(queue) and not interrupt_state.stop_requested:
            _journal_interrupts(run, interrupt_state)
            item = queue[index]
            index += 1
            state.consumed[phase] = index
            remaining = _remaining_minutes(deadline, clock())
            if item.kind != "calibrate" and not item_fits(item, remaining):
                record = _deferred_item(item, "insufficient time for tune-class item")
                state.items.append(record)
                run.append({"type": "deferred", **record})
                continue
            if item.kind == "calibrate" and not item_fits(item, remaining):
                record = _deferred_item(item, "calibration estimate exceeds remaining time")
                state.items.append(record)
                run.append({"type": "deferred", **record})
                continue
            run.append({"type": "item_start", **_item_dict(item)})
            state.started_items += 1
            _announce_item(reporter, state.started_items, state.total_items, item)
            item_started = clock()
            result_record: dict[str, Any]
            post_item_records: list[dict[str, Any]] = []
            if item.kind == "calibrate":
                model = by_fingerprint[item.fingerprint or ""]
                reference = records[item.reference_fingerprint or ""]
                result_record, retune = _run_calibrate_item(
                    run, state, item, model=model, reference=reference, llama=llama, options=options
                )
                run.append({"type": "calibration", **result_record})
                if retune is not None:
                    state.phase_queues["retune"].append(retune)
                    state.total_items += 1
                    run.append({"type": "retune_enqueued", **_item_dict(retune)})
                if result_record["outcome"] == "error":
                    state.failed = True
            else:
                result_record, post_item_records, transfers = _run_tune_class_item(
                    run,
                    state,
                    item,
                    options=options,
                    models=models,
                    by_fingerprint=by_fingerprint,
                    records=records,
                    llama=llama,
                    reporter=reporter,
                    remaining=remaining,
                    interrupt_state=interrupt_state,
                )
                state.phase_queues["calibrate"].extend(transfers)
                state.total_items += len(transfers)
            result_record["wall_s"] = max(0.0, (clock() - item_started).total_seconds())
            state.items.append(result_record)
            run.append({"type": "item_end", **result_record})
            for deferred in post_item_records:
                state.items.append(deferred)
                run.append({"type": "deferred", **deferred})
            if tune_failure_breaker(state.tune_failure_models):
                state.warning_messages.append(
                    "circuit breaker: three consecutive tune-class model failures"
                )
                interrupt_state.stop_requested = True
                break


def run_nightshift(
    options: NightshiftOptions,
    *,
    now_fn: Callable[[], datetime] | None = None,
    reporter: Reporter | None = None,
    follow_symlinks: bool = False,
) -> NightshiftOutcome:
    """Run one strictly serial Night Shift schedule."""
    clock = now_fn or (lambda: datetime.now(UTC))
    start = clock()
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    deadline = resolve_deadline(start, options.until, options.max_hours)
    hardware, llama, startup_errors = _startup_evidence(options)
    run = NightshiftRun.create(
        options.sessions_dir, options=options, hardware=hardware, llama=llama, argv=sys.argv
    )
    run_meta = json.loads((run.dir / "run.json").read_text(encoding="utf-8"))
    run_meta["resolved_deadline"] = deadline.isoformat() if deadline else None
    run.write_json("run.json", run_meta)
    warning_messages: list[str] = list(startup_errors)
    if startup_errors:
        return _finalize(
            run,
            options,
            start,
            deadline,
            [],
            warning_messages,
            3,
            hardware=hardware,
            llama=llama,
            models=(),
        )
    models, records, incomplete_paths, discovery_warnings = _discover_scope(
        options, follow_symlinks
    )
    warning_messages.extend(discovery_warnings)
    incomplete = tuple(
        (path, fingerprint)
        for path in incomplete_paths
        if (fingerprint := _session_fingerprint(path)) is not None
    )
    plan = build_initial_plan(
        models, records, incomplete, calibration_runs=options.calibration_runs
    )
    plan_payload = {"schema_version": 1, "items": [_item_dict(item) for item in plan]}
    run.write_json("plan.json", plan_payload)
    run.append({"type": "plan", "items": plan_payload["items"]})
    items: list[dict[str, Any]] = []
    if not models:
        warning_messages.append("no GGUF model survived discovery and filters")
        return _finalize(
            run,
            options,
            start,
            deadline,
            items,
            warning_messages,
            3,
            hardware=hardware,
            llama=llama,
            models=models,
        )
    if options.dry_run:
        items.extend({**_item_dict(item), "outcome": "planned", "wall_s": 0.0} for item in plan)
        return _finalize(
            run,
            options,
            start,
            deadline,
            items,
            warning_messages,
            0,
            hardware=hardware,
            llama=llama,
            models=models,
        )

    state = _ShiftState(total_items=len(plan), warning_messages=warning_messages)
    state.phase_queues = {
        "resume": [item for item in plan if item.kind == "resume"],
        "tune": [item for item in plan if item.kind == "tune"],
        "calibrate": [item for item in plan if item.kind == "calibrate"],
        "retune": [],
    }
    by_fingerprint = {model.report.fingerprint: model for model in models}

    with install_interrupt_handlers() as interrupt_state:
        try:
            _run_plan_phases(
                run,
                state,
                options=options,
                models=models,
                by_fingerprint=by_fingerprint,
                records=records,
                llama=llama,
                deadline=deadline,
                clock=clock,
                reporter=reporter,
                interrupt_state=interrupt_state,
            )
            spare_time = (
                deadline is not None
                and not interrupt_state.stop_requested
                and deepening_changes_profile(options)
            )
            if spare_time:
                _run_deepening(
                    run,
                    state,
                    options=options,
                    models=models,
                    records=records,
                    llama=llama,
                    deadline=deadline,
                    clock=clock,
                    reporter=reporter,
                    interrupt_state=interrupt_state,
                )
            elif deadline is not None and not interrupt_state.stop_requested:
                state.warning_messages.append(
                    "spare-time deepening skipped because its resolved profile "
                    "matches the initial tune"
                )
        except KeyboardInterrupt:
            interrupt_state.stop_requested = True
            interrupt_state.second_signal = True

    _journal_interrupts(run, interrupt_state)

    exit_code = _nightshift_exit_code(
        stop_requested=interrupt_state.stop_requested,
        second_signal=interrupt_state.second_signal,
        signal_count=interrupt_state.count,
        items=state.items,
        phase_queues=state.phase_queues,
        consumed=state.consumed,
        failed=state.failed,
    )
    return _finalize(
        run,
        options,
        start,
        deadline,
        state.items,
        state.warning_messages,
        exit_code,
        hardware=hardware,
        llama=llama,
        models=models,
        stopped=interrupt_state.stop_requested,
        interrupt_state=interrupt_state,
    )
