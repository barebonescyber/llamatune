"""Deterministic Results Matrix harvesting and artifact materialization."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import tempfile
import warnings
from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import typer

from llamatune.config import hardware_signature
from llamatune.evidence import PathEscapeError, confined_path, read_journal_lines
from llamatune.types import GPUInfo, HardwareReport, ResultRow, ResultsMatrix, TrialConfig

_SCHEMA_VERSION = 1
REFRESH_EXIT_CODES = frozenset({0, 1, 4})

# Files whose (size, mtime_ns) identity decides whether a cached source's
# rows may be reused instead of re-reading and re-parsing its evidence.
_SESSION_FILES = (
    "session.json",
    "model.json",
    "hardware.json",
    "llamacpp.json",
    "analysis.json",
    "journal.jsonl",
)
_MARATHON_FILES = ("marathon.json", "run.json", "model.json", "hardware.json", "llamacpp.json")
_NIGHTSHIFT_FILES = ("nightshift.json", "run.json", "hardware.json", "llamacpp.json")
_QUALITY_FILES = ("quality.json",)

_CACHE_VERSION = 1
_SourceCache = dict[str, dict[str, Any]]


class MatrixPathError(PathEscapeError):
    """A matrix artifact path escaped its output directory."""


_KINDS = frozenset(
    {
        "baseline",
        "recommendation",
        "lossless_recommendation",
        "context_envelope",
        "depth_profile",
        "operating_point",
        "ab_verification",
        "calibration",
        "quality_suite",
    }
)
_QUANT_RE = re.compile(r"(?i)(?:UD-)?Q\d+(?:_[A-Z0-9]+)+")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _journal(path: Path) -> list[dict[str, Any]]:
    entries, corruption = read_journal_lines(path)
    for warning in corruption:
        warnings.warn(warning, RuntimeWarning, stacklevel=2)
    return entries


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _integer(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _freeze(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _mean(value: Any) -> float | None:
    if isinstance(value, dict):
        return _number(value.get("mean"))
    return _number(value)


def _metrics(**values: float | None) -> dict[str, float]:
    return {name: value for name, value in values.items() if value is not None}


def _config(value: Any) -> TrialConfig | None:
    if not isinstance(value, dict):
        return None
    try:
        return TrialConfig.from_dict(value)
    except (KeyError, TypeError, ValueError):
        return None


def _quant(name: str | None, path: str) -> str | None:
    match = _QUANT_RE.search(f"{name or ''} {Path(path).name}")
    return match.group(0).upper() if match else None


def _hardware_report(value: dict[str, Any]) -> HardwareReport:
    raw_gpus = value.get("gpus", ())
    gpus = tuple(
        GPUInfo(
            vendor=str(gpu.get("vendor", "unknown")),
            name=str(gpu.get("name", "unknown")),
            vram_mb=_integer(gpu.get("vram_mb")),
            method=str(gpu.get("method", "unknown")),
        )
        for gpu in raw_gpus
        if isinstance(gpu, dict)
    )
    return HardwareReport(
        os_name=str(value["os_name"]),
        arch=str(value["arch"]),
        cpu_model=str(value.get("cpu_model", "unknown")),
        physical_cores=int(value["physical_cores"]),
        logical_cores=int(value["logical_cores"]),
        perf_cores=_integer(value.get("perf_cores")),
        ram_mb=int(value["ram_mb"]),
        gpus=gpus,
        warnings=(),
    )


def _signature(value: dict[str, Any]) -> tuple[Any, ...]:
    return hardware_signature(_hardware_report(value))


def _signature_hash(signature: tuple[Any, ...]) -> str:
    canonical = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _build_identity(value: dict[str, Any]) -> tuple[str, str | None]:
    discriminator = value.get("bench_sha256") or value.get("help_sha256")
    if not isinstance(discriminator, str) or not discriminator:
        raise ValueError("missing build discriminator")
    commit = value.get("build_commit")
    return discriminator, str(commit) if commit is not None else None


def _row_id(
    *,
    fingerprint: str,
    hardware_hash: str,
    build_discriminator: str,
    config: TrialConfig | None,
    ctx: int | None,
    depth: int | None,
    pp_workload: int | None,
    tg_workload: int | None,
    kind: str,
    suite_id: str | None,
) -> str:
    identity = (
        fingerprint,
        hardware_hash,
        build_discriminator,
        config.trial_id if config is not None else "defaults",
        ctx,
        depth,
        pp_workload,
        tg_workload,
        kind,
        suite_id,
    )
    canonical = json.dumps(identity, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _make_row(
    *,
    kind: str,
    root: Path,
    evidence_dir: Path,
    model: dict[str, Any],
    signature: tuple[Any, ...],
    build_discriminator: str,
    build_commit: str | None,
    config: TrialConfig | None,
    ctx: int | None,
    depth: int | None,
    pp_workload: int | None,
    tg_workload: int | None,
    suite_id: str | None,
    metrics: dict[str, float],
    status: str,
    confirmed: bool,
    replicated: bool | None,
    reps: int | None,
    noise_floor_cv: float | None,
    ts: str,
) -> ResultRow:
    if kind not in _KINDS:
        raise ValueError(f"unknown result kind: {kind}")
    fingerprint = str(model["fingerprint"])
    model_path = str(model["path"])
    model_name = str(model["name"]) if model.get("name") is not None else None
    hardware_hash = _signature_hash(signature)
    return ResultRow(
        row_id=_row_id(
            fingerprint=fingerprint,
            hardware_hash=hardware_hash,
            build_discriminator=build_discriminator,
            config=config,
            ctx=ctx,
            depth=depth,
            pp_workload=pp_workload,
            tg_workload=tg_workload,
            kind=kind,
            suite_id=suite_id,
        ),
        kind=kind,
        current=True,
        model_fingerprint=fingerprint,
        model_name=model_name,
        model_path=model_path,
        quant=_quant(model_name, model_path),
        hardware_hash=hardware_hash,
        hardware_signature=signature,
        build_discriminator=build_discriminator,
        build_commit=build_commit,
        config=config,
        ctx=ctx,
        depth=depth,
        pp_workload=pp_workload,
        tg_workload=tg_workload,
        suite_id=suite_id,
        metrics=metrics,
        status=status,
        confirmed=confirmed,
        replicated=replicated,
        reps=reps,
        noise_floor_cv=noise_floor_cv,
        source_root=root,
        evidence_dir=evidence_dir,
        ts=ts,
    )


def _winner_metrics(value: dict[str, Any]) -> dict[str, float]:
    confirmation = value.get("confirmation")
    source = confirmation if isinstance(confirmation, dict) else value
    improvement = value.get("improvement_pct")
    improvement = improvement if isinstance(improvement, dict) else {}
    pp = _mean(source.get("pp"))
    tg = _mean(source.get("tg"))
    return _metrics(
        **{
            "perf.pp": pp if pp is not None else _number(source.get("pp_mean")),
            "perf.tg": tg if tg is not None else _number(source.get("tg_mean")),
            "perf.improvement_pp_pct": _number(improvement.get("pp")),
            "perf.improvement_tg_pct": _number(improvement.get("tg")),
            "perf.improvement_score_pct": _number(improvement.get("score")),
        }
    )


def _validated_context(analysis: dict[str, Any]) -> int | None:
    feasibility = analysis.get("feasibility")
    if isinstance(feasibility, dict):
        value = _integer(feasibility.get("ctx_validated"))
        if value is not None:
            return value
    envelope = analysis.get("context_envelope")
    if not isinstance(envelope, list):
        return None
    values = [
        int(row["ctx"])
        for row in envelope
        if isinstance(row, dict)
        and _integer(row.get("ctx")) is not None
        and row.get("status") == "ok"
        and row.get("fallback_config") is None
    ]
    return max(values) if values else None


def _session_rows(root: Path, session_dir: Path) -> list[ResultRow]:
    session = _json(session_dir / "session.json")
    model = _json(session_dir / "model.json")
    hardware = _json(session_dir / "hardware.json")
    llama = _json(session_dir / "llamacpp.json")
    analysis = _json(session_dir / "analysis.json")
    entries = _journal(session_dir / "journal.jsonl")
    if not any(entry.get("type") in {"analysis_written", "session_end"} for entry in entries):
        return []
    options = session.get("options")
    if not isinstance(options, dict):
        raise ValueError("session options missing")
    signature = _signature(hardware)
    discriminator, commit = _build_identity(llama)
    created = str(session["created"])
    pp_workload = _integer(options.get("pp"))
    tg_workload = _integer(options.get("tg"))
    depth = _integer(options.get("depth"))
    baseline = analysis.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("analysis baseline missing")
    noise = _number(baseline.get("noise_floor_cv"))
    baseline_kind = str(analysis.get("baseline_kind") or baseline.get("kind") or "defaults")
    fallback = baseline_kind != "defaults" or baseline.get("fallback") is not None
    baseline_config = _config(baseline.get("resolved_defaults")) if fallback else None
    rows = [
        _make_row(
            kind="baseline",
            root=root,
            evidence_dir=session_dir,
            model=model,
            signature=signature,
            build_discriminator=discriminator,
            build_commit=commit,
            config=baseline_config,
            ctx=_integer(options.get("ctx_size")),
            depth=depth,
            pp_workload=pp_workload,
            tg_workload=tg_workload,
            suite_id=None,
            metrics=_metrics(
                **{"perf.pp": _mean(baseline.get("pp")), "perf.tg": _mean(baseline.get("tg"))}
            ),
            status="safe_fallback" if fallback else "ok",
            confirmed=False,
            replicated=None,
            reps=_integer(baseline.get("runs")),
            noise_floor_cv=noise,
            ts=created,
        )
    ]
    winner = analysis.get("winner")
    winner_id: str | None = None
    if isinstance(winner, dict):
        winner_config = _config(winner.get("config"))
        winner_id = winner_config.trial_id if winner_config is not None else None
        confirmation_ts = next(
            (
                str(entry["ts"])
                for entry in reversed(entries)
                if entry.get("type") == "confirmation_run" and isinstance(entry.get("ts"), str)
            ),
            created,
        )
        confirmation = winner.get("confirmation")
        rows.append(
            _make_row(
                kind="recommendation",
                root=root,
                evidence_dir=session_dir,
                model=model,
                signature=signature,
                build_discriminator=discriminator,
                build_commit=commit,
                config=winner_config,
                ctx=_validated_context(analysis),
                depth=depth,
                pp_workload=pp_workload,
                tg_workload=tg_workload,
                suite_id=None,
                metrics=_winner_metrics(winner),
                status="ok",
                confirmed=winner.get("confirmed") is True,
                replicated=None,
                reps=(
                    _integer(confirmation.get("runs")) if isinstance(confirmation, dict) else None
                ),
                noise_floor_cv=noise,
                ts=confirmation_ts,
            )
        )
    lossless = analysis.get("lossless_winner")
    if isinstance(lossless, dict):
        lossless_config = _config(lossless.get("config"))
        if lossless_config is not None and lossless_config.trial_id != winner_id:
            rows.append(
                _make_row(
                    kind="lossless_recommendation",
                    root=root,
                    evidence_dir=session_dir,
                    model=model,
                    signature=signature,
                    build_discriminator=discriminator,
                    build_commit=commit,
                    config=lossless_config,
                    ctx=_validated_context(analysis),
                    depth=depth,
                    pp_workload=pp_workload,
                    tg_workload=tg_workload,
                    suite_id=None,
                    metrics=_winner_metrics(lossless),
                    status=str(lossless.get("status", "ok")),
                    confirmed=lossless.get("confirmed") is True,
                    replicated=None,
                    reps=None,
                    noise_floor_cv=noise,
                    ts=created,
                )
            )
    envelope = analysis.get("context_envelope")
    if isinstance(envelope, list):
        for item in envelope:
            if not isinstance(item, dict) or _integer(item.get("ctx")) is None:
                continue
            config = _config(item.get("fallback_config")) or _config(item.get("config"))
            ctx = int(item["ctx"])
            status = str(item.get("status", "unknown"))
            rows.append(
                _make_row(
                    kind="context_envelope",
                    root=root,
                    evidence_dir=session_dir,
                    model=model,
                    signature=signature,
                    build_discriminator=discriminator,
                    build_commit=commit,
                    config=config,
                    ctx=ctx,
                    depth=depth,
                    pp_workload=pp_workload,
                    tg_workload=tg_workload,
                    suite_id=None,
                    metrics={"ctx.validated": float(ctx)} if status == "ok" else {},
                    status=status,
                    confirmed=False,
                    replicated=None,
                    reps=1,
                    noise_floor_cv=noise,
                    ts=created,
                )
            )
    profile = analysis.get("depth_profile")
    if isinstance(profile, dict) and isinstance(profile.get("rows"), list):
        profile_config = _config(winner.get("config")) if isinstance(winner, dict) else None
        for item in profile["rows"]:
            if not isinstance(item, dict) or _integer(item.get("d")) is None:
                continue
            rows.append(
                _make_row(
                    kind="depth_profile",
                    root=root,
                    evidence_dir=session_dir,
                    model=model,
                    signature=signature,
                    build_discriminator=discriminator,
                    build_commit=commit,
                    config=profile_config,
                    ctx=_integer(options.get("ctx_size")),
                    depth=int(item["d"]),
                    pp_workload=pp_workload,
                    tg_workload=tg_workload,
                    suite_id=None,
                    metrics=_metrics(
                        **{"perf.pp": _number(item.get("pp")), "perf.tg": _number(item.get("tg"))}
                    ),
                    status="ok",
                    confirmed=False,
                    replicated=None,
                    reps=_integer(options.get("reps_search")),
                    noise_floor_cv=noise,
                    ts=created,
                )
            )
    gate = analysis.get("quality_gate")
    if isinstance(gate, dict):
        gate_metrics = _metrics(
            **{
                "quality.perplexity.ppl": _number(gate.get("ppl_lossy")),
                "quality.perplexity.delta_pct": _number(gate.get("delta_pct")),
            }
        )
        if gate_metrics:
            rows.append(
                _make_row(
                    kind="quality_suite",
                    root=root,
                    evidence_dir=session_dir,
                    model=model,
                    signature=signature,
                    build_discriminator=discriminator,
                    build_commit=commit,
                    config=_config(winner.get("config")) if isinstance(winner, dict) else None,
                    ctx=_integer(options.get("ctx_size")),
                    depth=depth,
                    pp_workload=pp_workload,
                    tg_workload=tg_workload,
                    suite_id="perplexity-gate@legacy",
                    metrics=gate_metrics,
                    status=str(gate.get("status", "unknown")),
                    confirmed=True,
                    replicated=None,
                    reps=1,
                    noise_floor_cv=noise,
                    ts=created,
                )
            )
    return rows


def _run_identity(run_dir: Path) -> tuple[dict[str, Any], tuple[Any, ...], str, str | None, str]:
    run = _json(run_dir / "run.json")
    model = _json(run_dir / "model.json")
    signature = _signature(_json(run_dir / "hardware.json"))
    discriminator, commit = _build_identity(_json(run_dir / "llamacpp.json"))
    return model, signature, discriminator, commit, str(run["created"])


def _marathon_rows(root: Path, run_dir: Path) -> list[ResultRow]:
    summary = _json(run_dir / "marathon.json")
    if summary.get("schema_version") != 1:
        raise ValueError("unsupported marathon schema")
    model, signature, discriminator, commit, created = _run_identity(run_dir)
    raw_options = summary.get("options")
    options: dict[str, Any] = raw_options if isinstance(raw_options, dict) else {}
    pp_workload = _integer(options.get("pp"))
    tg_workload = _integer(options.get("tg"))
    rows: list[ResultRow] = []
    matrix = summary.get("matrix")
    if isinstance(matrix, list):
        for item in matrix:
            if not isinstance(item, dict):
                continue
            ctx, depth = _integer(item.get("ctx")), _integer(item.get("depth"))
            if ctx is None or depth is None:
                continue
            status = str(item.get("status", "unknown"))
            metrics = _metrics(
                **{"perf.pp": _number(item.get("pp")), "perf.tg": _number(item.get("tg"))}
            )
            if status == "ok":
                metrics["ctx.validated"] = float(ctx)
            rows.append(
                _make_row(
                    kind="operating_point",
                    root=root,
                    evidence_dir=run_dir,
                    model=model,
                    signature=signature,
                    build_discriminator=discriminator,
                    build_commit=commit,
                    config=_config(item.get("config")),
                    ctx=ctx,
                    depth=depth,
                    pp_workload=pp_workload,
                    tg_workload=tg_workload,
                    suite_id=None,
                    metrics=metrics,
                    status=status,
                    confirmed=False,
                    replicated=None,
                    reps=_integer(options.get("reps_search")),
                    noise_floor_cv=None,
                    ts=created,
                )
            )
    ab = summary.get("ab")
    if isinstance(ab, dict):
        rounds = summary.get("rounds")
        champion: TrialConfig | None = None
        if isinstance(rounds, list):
            for item in reversed(rounds):
                if isinstance(item, dict):
                    champion = _config(item.get("winner_config"))
                    if champion is not None:
                        break
        verdict = str(ab.get("verdict", "unknown"))
        rows.append(
            _make_row(
                kind="ab_verification",
                root=root,
                evidence_dir=run_dir,
                model=model,
                signature=signature,
                build_discriminator=discriminator,
                build_commit=commit,
                config=champion,
                ctx=_integer(options.get("ctx_size")),
                depth=None,
                pp_workload=pp_workload,
                tg_workload=tg_workload,
                suite_id=None,
                metrics=_metrics(
                    **{"perf.pp": _number(ab.get("b_pp")), "perf.tg": _number(ab.get("b_tg"))}
                ),
                status=(
                    "replicated"
                    if verdict == "b"
                    else "tie"
                    if verdict == "tie"
                    else "not_replicated"
                ),
                confirmed=True,
                replicated=verdict == "b",
                reps=_integer(ab.get("blocks")),
                noise_floor_cv=None,
                ts=created,
            )
        )
    return rows


def _reference_config(root: Path, calibration: dict[str, Any]) -> TrialConfig | None:
    raw = calibration.get("reference_config")
    config = _config(raw)
    if config is not None:
        return config
    reference = calibration.get("reference_session")
    if not isinstance(reference, str):
        return None
    reference_path = Path(reference).resolve()
    if not reference_path.is_relative_to(root.resolve()):
        return None
    try:
        analysis = _json(reference_path / "analysis.json")
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    winner = analysis.get("winner")
    return _config(winner.get("config")) if isinstance(winner, dict) else None


def _nightshift_rows(root: Path, run_dir: Path) -> list[ResultRow]:
    summary = _json(run_dir / "nightshift.json")
    if summary.get("schema_version") != 1:
        raise ValueError("unsupported nightshift schema")
    run = _json(run_dir / "run.json")
    hardware = summary.get("hardware")
    llama = summary.get("llamacpp")
    if not isinstance(hardware, dict) or not isinstance(llama, dict):
        hardware, llama = _json(run_dir / "hardware.json"), _json(run_dir / "llamacpp.json")
    signature = _signature(hardware)
    discriminator, commit = _build_identity(llama)
    created = str(run["created"])
    rows: list[ResultRow] = []
    items = summary.get("items")
    if not isinstance(items, list):
        return rows
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("calibration"), dict):
            continue
        calibration = item["calibration"]
        fingerprint = calibration.get("fingerprint") or item.get("fingerprint")
        model_path = item.get("model_path")
        if not isinstance(fingerprint, str) or not isinstance(model_path, str):
            continue
        drift_pp = _number(calibration.get("drift_pp"))
        drift_tg = _number(calibration.get("drift_tg"))
        model = {"fingerprint": fingerprint, "name": Path(model_path).stem, "path": model_path}
        rows.append(
            _make_row(
                kind="calibration",
                root=root,
                evidence_dir=run_dir,
                model=model,
                signature=signature,
                build_discriminator=discriminator,
                build_commit=commit,
                config=_reference_config(root, calibration),
                ctx=None,
                depth=_integer(item.get("depth_workload")),
                pp_workload=None,
                tg_workload=None,
                suite_id=None,
                metrics=_metrics(
                    **{
                        "perf.pp": _mean(calibration.get("pp")),
                        "perf.tg": _mean(calibration.get("tg")),
                        "perf.drift_pp_pct": drift_pp * 100 if drift_pp is not None else None,
                        "perf.drift_tg_pct": drift_tg * 100 if drift_tg is not None else None,
                    }
                ),
                status=str(calibration.get("verdict", "unknown")),
                confirmed=True,
                replicated=None,
                reps=_integer(calibration.get("runs")),
                noise_floor_cv=None,
                ts=created,
            )
        )
    return rows


def _quality_rows(root: Path, run_dir: Path) -> list[ResultRow]:
    summary = _json(run_dir / "quality.json")
    if summary.get("schema_version") != 1:
        raise ValueError("unsupported quality schema")
    model = summary.get("model")
    build = summary.get("build")
    signature_value = summary.get("hardware_signature")
    if (
        not isinstance(model, dict)
        or not isinstance(build, dict)
        or not isinstance(signature_value, list)
    ):
        raise ValueError("quality identity missing")
    signature = tuple(_freeze(item) for item in signature_value)
    discriminator, commit = _build_identity(build)
    created = str(summary["created"])
    config = _config(summary.get("config"))
    ctx = _integer(summary.get("ctx"))
    reps = _integer(summary.get("reps"))
    overall = _number(summary.get("overall"))
    rows: list[ResultRow] = []
    suites = summary.get("suites")
    if not isinstance(suites, list):
        raise ValueError("quality suites missing")
    for suite in suites:
        if not isinstance(suite, dict) or not isinstance(suite.get("metrics"), dict):
            continue
        name, suite_id = suite.get("name"), suite.get("suite_id")
        if not isinstance(name, str) or not isinstance(suite_id, str):
            continue
        metrics = {
            f"quality.{name}.{key}": float(value)
            for key, value in suite["metrics"].items()
            if _number(value) is not None
        }
        if overall is not None:
            metrics["quality.overall"] = overall
        rows.append(
            _make_row(
                kind="quality_suite",
                root=root,
                evidence_dir=run_dir,
                model=model,
                signature=signature,
                build_discriminator=discriminator,
                build_commit=commit,
                config=config,
                ctx=ctx,
                depth=None,
                pp_workload=None,
                tg_workload=None,
                suite_id=suite_id,
                metrics=metrics,
                status="ok",
                confirmed=True,
                replicated=None,
                reps=reps,
                noise_floor_cv=None,
                ts=created,
            )
        )
    return rows


def _units(root: Path, category: str, artifact: str) -> tuple[Path, ...]:
    parent = root / category
    if not parent.is_dir():
        return ()
    return tuple(
        path
        for path in sorted(parent.iterdir(), key=lambda item: item.name)
        if (path / artifact).is_file()
    )


def _ordinary_sessions(root: Path) -> tuple[Path, ...]:
    if not root.is_dir():
        return ()
    excluded = {"marathon", "matrix", "nightshift", "quality"}
    return tuple(
        path
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if path.is_dir() and path.name not in excluded and (path / "session.json").is_file()
    )


def _mark_current(rows: Iterable[ResultRow]) -> tuple[ResultRow, ...]:
    materialized = list(rows)
    winners: dict[str, int] = {}
    for index, row in enumerate(materialized):
        previous = winners.get(row.row_id)
        if previous is None or (row.ts, row.evidence_dir.name) > (
            materialized[previous].ts,
            materialized[previous].evidence_dir.name,
        ):
            winners[row.row_id] = index
    return tuple(
        replace(row, current=winners[row.row_id] == index) for index, row in enumerate(materialized)
    )


def _sort_key(row: ResultRow) -> tuple[Any, ...]:
    return (
        row.model_name or "",
        row.model_fingerprint,
        row.kind,
        row.ctx if row.ctx is not None else -1,
        row.depth if row.depth is not None else -1,
        row.ts,
        row.row_id,
    )


def _source_digest(source: Path, names: tuple[str, ...]) -> list[list[Any]]:
    """Return a change-detecting fingerprint of one source's evidence files."""
    digest: list[list[Any]] = []
    for name in names:
        try:
            info = (source / name).stat()
        except OSError:
            digest.append([name, None, None])
            continue
        digest.append([name, info.st_size, info.st_mtime_ns])
    return digest


def _revive_rows(raw_rows: object) -> list[ResultRow] | None:
    """Deserialize cached row dicts, or None when they are unusable."""
    if not isinstance(raw_rows, list):
        return None
    try:
        return [ResultRow.from_dict(row) for row in raw_rows if isinstance(row, dict)]
    except (KeyError, TypeError, ValueError):
        return None


def harvest(roots: tuple[Path, ...]) -> ResultsMatrix:
    """Read all named evidence under ``roots`` into a deterministic matrix."""
    resolved = tuple(root.resolve() for root in roots)
    rows, warnings_list, _cache = _harvest(resolved, None)
    current = sorted(_mark_current(rows), key=_sort_key)
    generated = max((row.ts for row in current), default="")
    return ResultsMatrix(
        schema_version=_SCHEMA_VERSION,
        generated=generated,
        roots=resolved,
        rows=tuple(current),
        warnings=tuple(warnings_list),
    )


def _harvest(
    resolved: tuple[Path, ...],
    prior_cache: _SourceCache | None,
) -> tuple[list[ResultRow], list[str], _SourceCache]:
    """Collect rows per source in deterministic order, reusing cached rows.

    ``prior_cache`` maps an evidence directory to its last digest and
    serialized rows; unchanged sources skip re-parsing entirely. Passing
    ``None`` performs a pure from-scratch read with no digest overhead.
    """
    rows: list[ResultRow] = []
    warnings: list[str] = []
    cache: _SourceCache = {}
    for root in resolved:
        if not root.is_dir():
            warnings.append(f"{root}: root is not a readable directory")
            continue
        sources = (
            ((_session_rows, path, _SESSION_FILES) for path in _ordinary_sessions(root)),
            (
                (_marathon_rows, path, _MARATHON_FILES)
                for path in _units(root, "marathon", "marathon.json")
            ),
            (
                (_nightshift_rows, path, _NIGHTSHIFT_FILES)
                for path in _units(root, "nightshift", "nightshift.json")
            ),
            (
                (_quality_rows, path, _QUALITY_FILES)
                for path in _units(root, "quality", "quality.json")
            ),
        )
        for group in sources:
            for reader, path, names in group:
                cached = prior_cache.get(str(path)) if prior_cache is not None else None
                digest = _source_digest(path, names)
                revived = None if cached is None else _revive_rows(cached.get("rows"))
                if cached is not None and revived is not None and cached.get("digest") == digest:
                    rows.extend(revived)
                    cache[str(path)] = {"digest": digest, "rows": cached["rows"]}
                    continue
                try:
                    produced = reader(root, path)
                except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                    warnings.append(f"{path}: {exc}")
                    continue
                rows.extend(produced)
                cache[str(path)] = {
                    "digest": digest,
                    "rows": [row.to_dict() for row in produced],
                }
    return rows, warnings, cache


def _load_source_cache(artifact: Path) -> _SourceCache | None:
    """Load the incremental-refresh cache from an existing artifact, if any.

    Missing, stale-versioned, or malformed caches yield ``None`` so callers
    fall back to a full rebuild; older artifacts stay fully readable.
    """
    try:
        data = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    cache = data.get("source_cache")
    if not isinstance(cache, dict) or cache.get("version") != _CACHE_VERSION:
        return None
    sources = cache.get("sources")
    if not isinstance(sources, dict):
        return None
    return {str(key): value for key, value in sources.items() if isinstance(value, dict)}


def _document(matrix: ResultsMatrix) -> dict[str, Any]:
    return {
        "schema_version": matrix.schema_version,
        "generated": matrix.generated,
        "roots": [str(root) for root in matrix.roots],
        "row_count": len(matrix.rows),
        "model_count": len({row.model_fingerprint for row in matrix.rows}),
        "rows": [row.to_dict() for row in matrix.rows],
        "warnings": list(matrix.warnings),
    }


def _atomic_text(output_dir: Path, name: str, content: str) -> None:
    target = confined_path(output_dir, name, error=MatrixPathError)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=output_dir)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def build(roots: tuple[Path, ...], output_dir: Path) -> dict[str, Any]:
    """Harvest roots and atomically materialize the JSON and Markdown artifacts.

    When a previous artifact with a current-version source cache exists in
    ``output_dir``, unchanged sources reuse their cached rows instead of
    being re-parsed; the resulting rows are identical to a full rebuild.
    """
    resolved = tuple(root.resolve() for root in roots)
    prior_cache = _load_source_cache(output_dir / "results-matrix.json")
    rows, warnings_list, cache = _harvest(resolved, prior_cache)
    current = sorted(_mark_current(rows), key=_sort_key)
    generated = max((row.ts for row in current), default="")
    matrix = ResultsMatrix(
        schema_version=_SCHEMA_VERSION,
        generated=generated,
        roots=resolved,
        rows=tuple(current),
        warnings=tuple(warnings_list),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise OSError(f"matrix output is not a directory: {output_dir}")
    if any((output_dir / marker).exists() for marker in ("session.json", "run.json")):
        raise ValueError(f"matrix output cannot be a session or run directory: {output_dir}")
    try:
        module = importlib.import_module("llamatune.matrixreport")
        renderer = cast(
            Callable[[ResultsMatrix], str],
            module.render_markdown,
        )
        markdown = renderer(matrix)
    except ImportError:
        markdown = f"# Results Matrix\n\nRows: {len(matrix.rows)}\n"
    document = _document(matrix)
    document["source_cache"] = {"version": _CACHE_VERSION, "sources": cache}
    _atomic_text(
        output_dir,
        "results-matrix.json",
        json.dumps(document, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(output_dir, "results-matrix.md", markdown)
    kinds: dict[str, int] = {}
    for row in matrix.rows:
        kinds[row.kind] = kinds.get(row.kind, 0) + 1
    return {
        "output": str(output_dir.resolve()),
        "rows": len(matrix.rows),
        "current_rows": sum(row.current for row in matrix.rows),
        "models": len({row.model_fingerprint for row in matrix.rows}),
        "roots": [str(root) for root in matrix.roots],
        "kinds": dict(sorted(kinds.items())),
        "warnings": list(matrix.warnings),
    }


def _artifact_roots(artifact: Path) -> tuple[Path, ...]:
    """Read just an artifact's schema and roots without materializing rows."""
    data = _json(artifact)
    version = data.get("schema_version")
    if version != _SCHEMA_VERSION:
        raise ValueError(f"unsupported results matrix schema version: {version}")
    roots = data.get("roots")
    if not isinstance(roots, list) or not all(isinstance(root, str) and root for root in roots):
        raise ValueError("invalid results matrix roots")
    return tuple(Path(str(root)) for root in roots)


def _refresh_roots(root: Path) -> tuple[Path, ...]:
    """Preserve an existing matrix's configured roots when refreshing its owner."""
    resolved = root.resolve()
    artifact = resolved / "matrix" / "results-matrix.json"
    if not artifact.is_file():
        return (resolved,)
    try:
        configured = _artifact_roots(artifact)
    except Exception as exc:
        typer.echo(
            f"warning: existing results matrix configuration ignored; "
            f"refreshing only {resolved}: {exc}",
            err=True,
        )
        return (resolved,)
    normalized = tuple(dict.fromkeys(path.resolve() for path in configured))
    return normalized if resolved in normalized else (resolved,)


def refresh(root: Path) -> None:
    """Best-effort artifact refresh used by terminal command epilogues."""
    resolved = root.resolve()
    try:
        build(_refresh_roots(resolved), resolved / "matrix")
    except Exception as exc:  # refresh must never alter the owning command's outcome
        typer.echo(f"warning: results matrix refresh failed: {exc}", err=True)


def load_artifact(path: Path) -> ResultsMatrix:
    """Read and validate a schema-v1 materialized matrix artifact."""
    data = _json(path)
    version = data.get("schema_version")
    if version != _SCHEMA_VERSION:
        raise ValueError(f"unsupported results matrix schema version: {version}")
    roots = data.get("roots")
    rows = data.get("rows")
    warnings = data.get("warnings", ())
    if not isinstance(roots, list) or not isinstance(rows, list) or not isinstance(warnings, list):
        raise ValueError("invalid results matrix document")
    if not all(isinstance(root, str) and root for root in roots):
        raise ValueError("invalid results matrix roots")
    return ResultsMatrix(
        schema_version=_SCHEMA_VERSION,
        generated=str(data.get("generated", "")),
        roots=tuple(Path(str(root)) for root in roots),
        rows=tuple(ResultRow.from_dict(row) for row in rows if isinstance(row, dict)),
        warnings=tuple(str(warning) for warning in warnings),
    )
