"""Serial, evidence-first orchestration for unattended tuning shifts."""

from __future__ import annotations

import dataclasses
import json
import os
import secrets
import signal
import sys
import warnings
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any, cast

from llamatune._version import __version__
from llamatune.types import (
    CalibrationResult,
    DiscoveredModel,
    HardwareReport,
    LlamaCppReport,
    NightshiftOptions,
    NightshiftOutcome,
    RegistryRecord,
    TuneOptions,
    WorkItem,
)

MIN_TUNE_MINUTES = 20.0
SHUTDOWN_MARGIN_MIN = 5.0
_CALIBRATION_FALLBACK_S = 300.0
_CALIBRATION_SAFETY_FACTOR = 1.5
_CREATE_RETRIES = 5

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


class NightshiftPathError(Exception):
    """A requested run artifact path escaped the Night Shift directory."""


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return cast(
            dict[str, Any],
            {
                field.name: _jsonable(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        )
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, frozenset, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def _confine(base: Path, *parts: str) -> Path:
    resolved = base.resolve()
    candidate = base.joinpath(*parts)
    try:
        candidate.resolve().relative_to(resolved)
    except ValueError:
        raise NightshiftPathError(f"path {candidate} escapes nightshift directory {base}") from None
    return candidate


class NightshiftRun:
    """The only writer allowed inside one Night Shift evidence directory."""

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
        root.mkdir(parents=True, exist_ok=True)
        run_dir: Path | None = None
        for _ in range(_CREATE_RETRIES):
            name = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
            candidate = root / name
            try:
                candidate.mkdir(exist_ok=False)
            except FileExistsError:
                continue
            run_dir = candidate
            break
        if run_dir is None:
            raise NightshiftPathError(f"could not allocate a unique run directory under {root}")
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
        run.write_json("hardware.json", _jsonable(hardware))
        run.write_json("llamacpp.json", _jsonable(llama))
        run.append({"type": "nightshift_start", "tool_version": __version__})
        return run

    @classmethod
    def load(cls, run_dir: Path) -> NightshiftRun:
        path = Path(run_dir)
        if not (path / "run.json").is_file():
            raise FileNotFoundError(f"not a Night Shift run: {path}")
        return cls(path)

    def append(self, entry: dict[str, Any]) -> None:
        record = _jsonable(dict(entry))
        record.setdefault("ts", _utc_iso())
        path = _confine(self._dir, "journal.jsonl")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def calibration_dir(self, fingerprint16: str, n: int) -> Path:
        invalid = not fingerprint16 or any(
            character not in "0123456789abcdefABCDEF" for character in fingerprint16
        )
        if invalid and fingerprint16 != "bootstrap":
            raise NightshiftPathError("calibration fingerprint must be hexadecimal")
        path = _confine(self._dir, "calibrations", fingerprint16, f"run-{n}")
        path.mkdir(parents=True, exist_ok=True)
        return path

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
    """Resolve the earliest supplied deadline; pure when ``local_tz`` is supplied."""
    candidates: list[datetime] = []
    if until is not None:
        hour, minute = (int(part) for part in until.split(":"))
        zone = local_tz or datetime.now().astimezone().tzinfo or UTC
        local_start = start.astimezone(zone)
        candidate = local_start.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= local_start:
            candidate += timedelta(days=1)
        candidates.append(candidate.astimezone(UTC))
    if max_hours is not None:
        candidates.append(start + timedelta(hours=max_hours))
    return min(candidates) if candidates else None


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


def _interrupted_session(path: Path) -> bool:
    try:
        lines = (path / "journal.jsonl").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    entries: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            return False
        if isinstance(entry, dict):
            entries.append(entry)
    return any(
        entry.get("type") == "session_end"
        and entry.get("reason") in {"stopped_by_user", "interrupted"}
        for entry in entries
        if isinstance(entry, dict)
    )


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
            lines = (Path(session_dir) / "journal.jsonl").read_text(encoding="utf-8").splitlines()
            entries = (json.loads(line) for line in lines if line.strip())
            total += sum(
                1
                for entry in entries
                if isinstance(entry, dict)
                and entry.get("type") in {"baseline_run", "trial", "confirmation_run"}
            )
        except (OSError, json.JSONDecodeError):
            continue
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
) -> NightshiftOutcome:
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


def run_nightshift(
    options: NightshiftOptions, *, now_fn: Callable[[], datetime] | None = None
) -> NightshiftOutcome:
    """Run one strictly serial Night Shift schedule."""
    from llamatune.discovery import discover_models
    from llamatune.hardware import assess_hardware
    from llamatune.llama import discover_llama
    from llamatune.registry import build_registry, incomplete_sessions
    from llamatune.search import resume_tuning, run_tuning
    from llamatune.session import Session

    clock = now_fn or (lambda: datetime.now(UTC))
    start = clock()
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    deadline = resolve_deadline(start, options.until, options.max_hours)
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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        models = discover_models(
            options.models_dir,
            options.include,
            options.exclude,
            duplicates=options.duplicates,
            full_hash=options.full_hash,
        )
        records = _context_compatible_records(
            build_registry(options.sessions_dir), options.ctx_size
        )
        incomplete_paths = incomplete_sessions(options.sessions_dir)
    warning_messages.extend(str(item.message) for item in caught)
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

    by_fingerprint = {model.report.fingerprint: model for model in models}
    stop = False
    second_signal = False
    signalled = False
    old_handlers: dict[signal.Signals, Any] = {}

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal stop, second_signal, signalled
        if stop:
            second_signal = True
            run.append({"type": "interrupted", "signal": signum, "immediate": True})
            raise KeyboardInterrupt
        signalled = True
        stop = True
        run.append({"type": "interrupted", "signal": signum, "immediate": False})

    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, handle_signal)

    failed = False
    tune_failure_models: list[str] = []
    calibrated_fingerprints: set[str] = set()
    tuned_fingerprints: set[str] = set()
    phase_queues: dict[str, list[WorkItem]] = {
        "resume": [item for item in plan if item.kind == "resume"],
        "tune": [item for item in plan if item.kind == "tune"],
        "calibrate": [item for item in plan if item.kind == "calibrate"],
        "retune": [],
    }
    consumed: dict[str, int] = {}

    try:
        for phase in ("resume", "tune", "calibrate", "retune"):
            queue = phase_queues[phase]
            index = 0
            while index < len(queue) and not stop:
                item = queue[index]
                index += 1
                consumed[phase] = index
                remaining = _remaining_minutes(deadline, clock())
                if item.kind != "calibrate" and not item_fits(item, remaining):
                    record = {
                        **_item_dict(item),
                        "outcome": "deferred",
                        "reason": "insufficient time for tune-class item",
                        "wall_s": 0.0,
                    }
                    items.append(record)
                    run.append({"type": "deferred", **record})
                    continue
                if item.kind == "calibrate" and not item_fits(item, remaining):
                    record = {
                        **_item_dict(item),
                        "outcome": "deferred",
                        "reason": "calibration estimate exceeds remaining time",
                        "wall_s": 0.0,
                    }
                    items.append(record)
                    run.append({"type": "deferred", **record})
                    continue
                run.append({"type": "item_start", **_item_dict(item)})
                item_started = clock()
                result_record: dict[str, Any]
                post_item_records: list[dict[str, Any]] = []
                if item.kind == "calibrate":
                    from llamatune.calibrate import run_calibration

                    model = by_fingerprint[item.fingerprint or ""]
                    reference = records[item.reference_fingerprint or ""]
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
                                reference.fingerprint
                                if reference.fingerprint != model.report.fingerprint
                                else None
                            ),
                            build_changed=reference.help_sha256 != llama.help_sha256,
                            artifact_dir=None,
                        )
                    calibrated_fingerprints.add(model.report.fingerprint)
                    result_record = {
                        **_item_dict(item),
                        "outcome": calibration.verdict,
                        "calibration": _jsonable(calibration),
                    }
                    run.append({"type": "calibration", **result_record})
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
                        phase_queues["retune"].append(retune)
                        run.append({"type": "retune_enqueued", **_item_dict(retune)})
                    if calibration.verdict == "error":
                        failed = True
                else:
                    session_dir = item.session_dir
                    try:
                        if item.kind == "resume":
                            if session_dir is None:
                                raise ValueError("resume item has no session directory")
                            outcome = resume_tuning(session_dir)
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
                                session, current_hardware, model.report, llama, tune_options
                            )
                        result_record = {
                            **_item_dict(item),
                            "session_dir": str(session_dir),
                            "outcome": "succeeded" if outcome.exit_code in {0, 1} else "failed",
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
                        stop = True
                        result_record["outcome"] = "interrupted"
                    if result_record["outcome"] == "failed":
                        failed = True
                        fingerprint = item.fingerprint or str(item.model_path)
                        if not tune_failure_models or tune_failure_models[-1] != fingerprint:
                            tune_failure_models.append(fingerprint)
                        if item.kind == "tune" and item.fingerprint:
                            representative = by_fingerprint[item.fingerprint]
                            for sibling in models:
                                if (
                                    sibling.representative
                                    or sibling.group_key != representative.group_key
                                    or sibling.report.fingerprint in records
                                ):
                                    continue
                                deferred = {
                                    "kind": "calibrate",
                                    "model_path": str(sibling.path),
                                    "fingerprint": sibling.report.fingerprint,
                                    "reference_fingerprint": item.fingerprint,
                                    "outcome": "deferred",
                                    "reason": "representative tune failed",
                                    "wall_s": 0.0,
                                }
                                post_item_records.append(deferred)
                    else:
                        tune_failure_models.clear()
                        if item.fingerprint:
                            tuned_fingerprints.add(item.fingerprint)
                        records = _context_compatible_records(
                            build_registry(options.sessions_dir), options.ctx_size
                        )
                        if item.kind == "tune" and item.fingerprint in records:
                            representative = by_fingerprint[item.fingerprint]
                            for sibling in models:
                                if (
                                    sibling.representative
                                    or sibling.group_key != representative.group_key
                                    or sibling.report.fingerprint in records
                                    or sibling.report.fingerprint in calibrated_fingerprints
                                ):
                                    continue
                                reference = records[item.fingerprint]
                                phase_queues["calibrate"].append(
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
                wall_s = max(0.0, (clock() - item_started).total_seconds())
                result_record["wall_s"] = wall_s
                items.append(result_record)
                run.append({"type": "item_end", **result_record})
                for deferred in post_item_records:
                    items.append(deferred)
                    run.append({"type": "deferred", **deferred})
                if tune_failure_breaker(tune_failure_models):
                    warning_messages.append(
                        "circuit breaker: three consecutive tune-class model failures"
                    )
                    stop = True
                    break

        if deadline is not None and not stop and deepening_changes_profile(options):
            records = _context_compatible_records(
                build_registry(options.sessions_dir), options.ctx_size
            )
            candidates = deepen_order(models, records)
            for model in candidates:
                remaining = _remaining_minutes(deadline, clock())
                if remaining is None or remaining < MIN_TUNE_MINUTES or stop:
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
                phase_queues["retune"].append(item)
                run.append({"type": "item_start", **_item_dict(item)})
                item_started = clock()
                deepen_session_dir: Path | None = None
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
                        session, current_hardware, model.report, llama, tune_options
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
                items.append(result_record)
                run.append({"type": "item_end", **result_record})
                if interrupted:
                    stop = True
                    break
                if result_record["outcome"] == "failed":
                    failed = True
                    if (
                        not tune_failure_models
                        or tune_failure_models[-1] != model.report.fingerprint
                    ):
                        tune_failure_models.append(model.report.fingerprint)
                    if tune_failure_breaker(tune_failure_models):
                        warning_messages.append(
                            "circuit breaker: three consecutive tune-class model failures"
                        )
                        stop = True
                        break
                else:
                    tune_failure_models.clear()
        elif deadline is not None and not stop:
            warning_messages.append(
                "spare-time deepening skipped because its resolved profile matches the initial tune"
            )
    except KeyboardInterrupt:
        stop = True
        second_signal = True
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)

    def _unconsumed_plan_items() -> bool:
        """True when any phase queue still holds items the plan never reached."""
        return any(len(phase_queues[name]) > consumed.get(name, 0) for name in phase_queues)

    exit_code = (
        4
        if stop
        and (
            second_signal
            or any(item.get("outcome") == "interrupted" for item in items)
            or (signalled and _unconsumed_plan_items())
        )
        else 1
        if failed
        else 0
    )
    return _finalize(
        run,
        options,
        start,
        deadline,
        items,
        warning_messages,
        exit_code,
        hardware=hardware,
        llama=llama,
        models=models,
        stopped=stop,
    )
