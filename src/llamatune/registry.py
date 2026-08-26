"""Append-only registry of confirmed tuning recommendations (contract C10)."""

from __future__ import annotations

import json
import os
import statistics
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llamatune.types import LlamaCppReport, ModelReport, RegistryRecord, TrialConfig


def load_all(path: Path) -> list[dict[str, Any]]:
    """Load registry records, tolerating an incomplete final line."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise ValueError(f"registry line {index + 1} is corrupt") from None
        if not isinstance(record, dict):
            raise ValueError(f"registry line {index + 1} is not an object")
        records.append(record)
    return records


def append_record(
    path: Path,
    *,
    model_report: ModelReport,
    llama_report: LlamaCppReport,
    hardware_signature: tuple[Any, ...],
    session_dir: Path,
    target: str,
    ctx_size: int | None,
    pp: int = 512,
    tg: int = 128,
    depth: int | None = None,
    config: TrialConfig,
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Append one confirmed recommendation and return its serialized record."""
    record: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "model_fingerprint": model_report.fingerprint,
        "model_name": model_report.name,
        "model_path": str(model_report.path),
        "help_sha256": llama_report.help_sha256,
        "build_commit": llama_report.build_commit,
        "bench_sha256": llama_report.bench_sha256,
        "hardware_signature": json.loads(json.dumps(hardware_signature)),
        "session_dir": str(session_dir),
        "target": target,
        "ctx_size": ctx_size,
        "workload": {"pp": pp, "tg": tg, "depth": depth},
        "config": config.to_dict(),
        "expected": expected,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        raw = path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            last_newline = raw.rfind(b"\n")
            path.write_bytes(raw[: last_newline + 1] if last_newline >= 0 else b"")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


def lookup(
    path: Path,
    model_report: ModelReport,
    llama_report: LlamaCppReport,
    hardware_signature: tuple[Any, ...],
    *,
    pp: int | None = None,
    tg: int | None = None,
    depth: int | None = None,
    ctx_size: int | None = None,
) -> dict[str, Any]:
    """Return the newest matching record or explicit identity staleness reasons."""
    fingerprint_matches = [
        record
        for record in load_all(path)
        if record.get("model_fingerprint") == model_report.fingerprint
    ]
    if not fingerprint_matches:
        return {"status": "miss", "record": None, "stale_reasons": []}

    expected_hardware = json.loads(json.dumps(hardware_signature))
    for record in reversed(fingerprint_matches):
        workload = record.get("workload")
        workload_matches = pp is None or workload == {"pp": pp, "tg": tg, "depth": depth}
        recorded_ctx_size = record.get("ctx_size")
        context_matches = ctx_size is None or (
            isinstance(recorded_ctx_size, int)
            and not isinstance(recorded_ctx_size, bool)
            and recorded_ctx_size >= ctx_size
        )
        build_matches = (
            record.get("bench_sha256") == llama_report.bench_sha256
            if record.get("bench_sha256") is not None and llama_report.bench_sha256 is not None
            else record.get("help_sha256") == llama_report.help_sha256
        )
        if (
            build_matches
            and record.get("hardware_signature") == expected_hardware
            and workload_matches
            and context_matches
        ):
            return {"status": "hit", "record": record, "stale_reasons": []}

    reasons: list[str] = []
    newest = fingerprint_matches[-1]
    if (
        newest.get("bench_sha256") != llama_report.bench_sha256
        if newest.get("bench_sha256") is not None and llama_report.bench_sha256 is not None
        else newest.get("help_sha256") != llama_report.help_sha256
    ):
        reasons.append("llama.cpp build changed")
    if newest.get("hardware_signature") != expected_hardware:
        reasons.append("hardware changed")
    workload = newest.get("workload")
    if pp is not None and workload != {"pp": pp, "tg": tg, "depth": depth}:
        reasons.append("workload changed")
    recorded_ctx_size = newest.get("ctx_size")
    if ctx_size is not None and (
        not isinstance(recorded_ctx_size, int)
        or isinstance(recorded_ctx_size, bool)
        or recorded_ctx_size < ctx_size
    ):
        reasons.append("context size increased")
    return {"status": "stale", "record": newest, "stale_reasons": reasons}


def _json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} is not an object")
    return data


def _journal(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    entries: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        if isinstance(value, dict):
            entries.append(value)
    return entries


def _session_dirs(sessions_dir: Path) -> tuple[Path, ...]:
    if not sessions_dir.is_dir():
        return ()
    return tuple(
        sorted(
            (
                child
                for child in sessions_dir.iterdir()
                if child.is_dir()
                and child.name != "nightshift"
                and (child / "session.json").is_file()
            ),
            key=lambda path: str(path),
        )
    )


def _completed(entries: list[dict[str, Any]], analysis_path: Path) -> bool:
    return analysis_path.is_file() and any(
        entry.get("type") in {"analysis_written", "session_end"} for entry in entries
    )


def _wall_seconds(entries: list[dict[str, Any]]) -> float | None:
    timestamps: list[datetime] = []
    for entry in entries:
        raw = entry.get("ts")
        if isinstance(raw, str):
            try:
                timestamps.append(datetime.fromisoformat(raw.replace("Z", "+00:00")))
            except ValueError:
                continue
    return (max(timestamps) - min(timestamps)).total_seconds() if len(timestamps) >= 2 else None


def _record(session_dir: Path, entries: list[dict[str, Any]]) -> RegistryRecord:
    session = _json_object(session_dir / "session.json")
    model = _json_object(session_dir / "model.json")
    llama = _json_object(session_dir / "llamacpp.json")
    analysis = _json_object(session_dir / "analysis.json")
    options = session["options"]
    baseline = analysis["baseline"]
    winner = analysis.get("winner")
    if isinstance(winner, dict):
        confirmation = winner["confirmation"]
        config = TrialConfig.from_dict(winner["config"])
        pp = float(confirmation["pp"]["mean"])
        tg = float(confirmation["tg"]["mean"])
        outcome = "winner"
    else:
        config = None
        pp = float(baseline["pp"]["mean"])
        tg = float(baseline["tg"]["mean"])
        outcome = "defaults_optimal"
    trial_walls = [
        float(entry["wall_s"])
        for entry in entries
        if entry.get("type") == "trial" and isinstance(entry.get("wall_s"), (int, float))
    ]
    validated_contexts: list[int] = []
    feasibility = analysis.get("feasibility")
    if isinstance(feasibility, dict):
        required_ctx = feasibility.get("ctx_validated")
        if isinstance(required_ctx, int) and not isinstance(required_ctx, bool):
            validated_contexts.append(required_ctx)
    envelope = analysis.get("context_envelope")
    if isinstance(envelope, list):
        validated_contexts.extend(
            int(row["ctx"])
            for row in envelope
            if isinstance(row, dict)
            and isinstance(row.get("ctx"), int)
            and not isinstance(row.get("ctx"), bool)
            and row.get("status") == "ok"
            and row.get("fallback_config") is None
        )
    return RegistryRecord(
        fingerprint=str(model["fingerprint"]),
        session_dir=session_dir,
        created=str(session["created"]),
        outcome=outcome,
        reference_config=config,
        reference_pp=pp,
        reference_tg=tg,
        noise_floor_cv=float(baseline["noise_floor_cv"]),
        pp_workload=int(options["pp"]),
        tg_workload=int(options["tg"]),
        depth_workload=(int(options["depth"]) if options.get("depth") is not None else None),
        reps_confirm=int(options["reps_confirm"]),
        target=str(options["target"]),
        build_commit=(
            str(llama["build_commit"]) if llama.get("build_commit") is not None else None
        ),
        help_sha256=str(llama["help_sha256"]),
        bench_sha256=(str(llama["bench_sha256"]) if llama.get("bench_sha256") else None),
        median_trial_wall_s=statistics.median(trial_walls) if trial_walls else None,
        session_wall_s=_wall_seconds(entries),
        ctx_size=max(validated_contexts) if validated_contexts else None,
    )


def build_registry(sessions_dir: Path, *, ctx_size: int | None = None) -> dict[str, RegistryRecord]:
    """Derive the latest completed session record for every fingerprint."""
    records: dict[str, RegistryRecord] = {}
    for session_dir in _session_dirs(sessions_dir):
        try:
            entries = _journal(session_dir / "journal.jsonl")
            if not _completed(entries, session_dir / "analysis.json"):
                continue
            record = _record(session_dir, entries)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            warnings.warn(
                f"skipping corrupt session {session_dir}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if ctx_size is not None and (record.ctx_size is None or record.ctx_size < ctx_size):
            continue
        previous = records.get(record.fingerprint)
        if previous is None or record.created > previous.created:
            records[record.fingerprint] = record
    return records


def absorb_session(
    records: dict[str, RegistryRecord],
    session_dir: Path,
    *,
    ctx_size: int | None = None,
) -> bool:
    """Fold one just-finished session into ``records`` without a full rescan.

    Applies the same completion, corruption, context-filter, and
    newest-``created`` rules as :func:`build_registry` so incremental folds
    converge to exactly the from-scratch registry under serial scheduling.
    Returns whether ``records`` gained or updated a fingerprint.
    """
    try:
        entries = _journal(session_dir / "journal.jsonl")
        if not _completed(entries, session_dir / "analysis.json"):
            return False
        record = _record(session_dir, entries)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        warnings.warn(
            f"skipping corrupt session {session_dir}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    if ctx_size is not None and (record.ctx_size is None or record.ctx_size < ctx_size):
        return False
    previous = records.get(record.fingerprint)
    if previous is not None and record.created <= previous.created:
        return False
    records[record.fingerprint] = record
    return True


def incomplete_sessions(sessions_dir: Path) -> tuple[Path, ...]:
    """Return incomplete ordinary tuning sessions, oldest first."""
    candidates: list[tuple[str, Path]] = []
    for session_dir in _session_dirs(sessions_dir):
        try:
            session = _json_object(session_dir / "session.json")
            entries = _journal(session_dir / "journal.jsonl")
            if not _completed(entries, session_dir / "analysis.json"):
                candidates.append((str(session["created"]), session_dir))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            warnings.warn(
                f"skipping corrupt session {session_dir}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return tuple(path for _, path in sorted(candidates, key=lambda item: (item[0], str(item[1]))))
