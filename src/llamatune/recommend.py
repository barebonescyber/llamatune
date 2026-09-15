"""Scoring roll-up, analysis.json assembly, and recommendation emission.

Builds the DESIGN §11.1 ``analysis.json`` object from journaled trial
records, then writes ``recommended.json``, ``recommended.sh`` and (via
:func:`llamatune.report.render`) ``report.md``. All session writes go through
:class:`llamatune.session.Session`; this module only *reads* the already
written evidence files it needs to render the report.
"""

from __future__ import annotations

import json
import shlex
from typing import TYPE_CHECKING, Any

from llamatune import report as report_module
from llamatune import stats
from llamatune.runtimeflags import runtime_flags
from llamatune.types import BaselineResult, LlamaCppReport, ModelReport, TrialConfig, TuneOptions

if TYPE_CHECKING:
    from llamatune.session import Session

_SCHEMA_VERSION = 2
_TOP_LIMIT = 10
_LOSSLESS = ("f16", "f16")


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "trial_id": record["trial_id"],
        "status": record["status"],
        "config": record["config"],
        "pp_mean": record.get("pp_mean"),
        "tg_mean": record.get("tg_mean"),
        "score": record.get("score"),
        "flags": list(record.get("flags", [])),
    }


def _is_lossless(config: dict[str, Any]) -> bool:
    return (config.get("cache_type_k"), config.get("cache_type_v")) == _LOSSLESS


def compute_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    """Outcome tallies for the analysis ``counts`` block (DESIGN §11.1)."""
    counts = {
        "executed": 0,
        "ok": 0,
        "unstable": 0,
        "oom": 0,
        "cuda_error": 0,
        "gpu_resource": 0,
        "timeout": 0,
        "crash": 0,
        "parse_error": 0,
        "pruned": 0,
    }
    for record in records:
        status = record["status"]
        if status in counts:
            counts[status] += 1
        if status != "pruned":
            counts["executed"] += 1
    return counts


def _ok_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _summary(record)
        for record in records
        if record["status"] in ("ok", "unstable") and record.get("score") is not None
    ]


def compute_top(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Up to ten measured trials, best geometric score first."""
    summaries = _ok_summaries(records)
    summaries.sort(key=lambda s: (s["score"], s["trial_id"]), reverse=True)
    return summaries[:_TOP_LIMIT]


def compute_pareto(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The Pareto front over (pp_mean, tg_mean) across measured trials."""
    summaries = _ok_summaries(records)
    points = [(float(s["pp_mean"]), float(s["tg_mean"])) for s in summaries]
    front = stats.pareto_front(points)
    chosen = [summaries[i] for i in front]
    chosen.sort(key=lambda s: (s["pp_mean"], s["tg_mean"]), reverse=True)
    return chosen


def best_lossless(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Highest-scoring all-lossless measured trial, or None (DESIGN §9.2)."""
    candidates = [s for s in _ok_summaries(records) if _is_lossless(s["config"])]
    if not candidates:
        return None
    candidates.sort(key=lambda s: (s["score"], s["trial_id"]), reverse=True)
    return candidates[0]


def build_analysis(
    *,
    target: str,
    baseline: BaselineResult,
    records: list[dict[str, Any]],
    winner: dict[str, Any] | None,
    lossless_winner: dict[str, Any] | None,
    warnings: list[str],
    default_probe: dict[str, Any] | None = None,
    feasibility: dict[str, Any] | None = None,
    context_validation: dict[str, Any] | None = None,
    cli_validation: dict[str, Any] | None = None,
    estimate_vs_observed: dict[str, Any] | None = None,
    coverage: dict[str, Any] | None = None,
    quality_gate: dict[str, Any] | None = None,
    telemetry: dict[str, Any] | None = None,
    depth_profile: dict[str, Any] | None = None,
    context_envelope: list[dict[str, Any]] | None = None,
    budget_consumed: int | None = None,
) -> dict[str, Any]:
    """Assemble the complete DESIGN §11.1 analysis object."""
    counts = compute_counts(records)
    counts["budget_consumed"] = counts["executed"] if budget_consumed is None else budget_consumed
    result = {
        "schema_version": _SCHEMA_VERSION,
        "target": target,
        "baseline": {
            "runs": baseline.runs,
            "pp": stats.metric_to_dict(baseline.pp),
            "tg": stats.metric_to_dict(baseline.tg),
            "noise_floor_cv": baseline.noise_floor_cv,
            "fallback": baseline.fallback,
            "resolved_defaults": baseline.resolved_defaults,
            "kind": baseline.kind,
        },
        "default_probe": default_probe,
        "baseline_kind": baseline.kind,
        "feasibility": feasibility,
        "context_validation": context_validation,
        "cli_validation": cli_validation,
        "estimate_vs_observed": estimate_vs_observed,
        "coverage": coverage,
        "quality_gate": quality_gate,
        "telemetry": telemetry,
        "counts": counts,
        "pareto": compute_pareto(records),
        "top": compute_top(records),
        "winner": winner,
        "lossless_winner": lossless_winner,
        "warnings": warnings,
    }
    if depth_profile is not None:
        result["depth_profile"] = depth_profile
    if context_envelope is not None:
        result["context_envelope"] = context_envelope
    return result


def build_recommended_json(
    *,
    config: TrialConfig,
    capabilities: frozenset[str],
    expected: dict[str, Any],
    confirmed: bool,
    target: str,
    model: ModelReport,
    llama: LlamaCppReport,
    pp: int | None = None,
    tg: int | None = None,
    depth: int | None = None,
) -> dict[str, Any]:
    """The ``recommended.json`` payload (DESIGN §11.2)."""
    result = {
        "target": target,
        "confirmed": confirmed,
        "config": config.to_dict(),
        "bench_flags": list(config.bench_args(capabilities)),
        "expected": expected,
        "model": {
            "path": str(model.path),
            "fingerprint": model.fingerprint,
            "full_sha256": model.full_sha256,
        },
        "llamacpp": {
            "build_commit": llama.build_commit,
            "build_number": llama.build_number,
            **({"bench_sha256": llama.bench_sha256} if llama.bench_sha256 is not None else {}),
        },
    }
    if pp is not None and tg is not None:
        result = {
            **result,
            "workload": {"pp": pp, "tg": tg, **({"depth": depth} if depth is not None else {})},
        }
    return result


def _cli_flags(config: TrialConfig, *, moe: bool) -> list[str]:
    return runtime_flags(config, moe=moe)


def build_recommended_sh(
    *,
    config: TrialConfig,
    model: ModelReport,
    expected: dict[str, Any],
    confirmed: bool,
    target: str,
    moe: bool,
    ctx_size: int | None = None,
    context_envelope: list[dict[str, Any]] | None = None,
) -> str:
    """A commented, non-executable reference snippet (DESIGN §11.2)."""
    runtime_flags = _cli_flags(config, moe=moe)
    if ctx_size is not None:
        runtime_flags += ["-c", str(ctx_size)]
    model_path = str(model.path)
    improvement = expected.get("improvement_pct", {})
    verdict = "confirmed improvement" if confirmed else "no confirmed improvement (defaults)"
    lines = [
        "# llamatune recommendation -- reference only, not executable by default.",
        f"# Target: {target} ({verdict}).",
        f"# Expected: pp {_num(expected.get('pp'))} t/s ({_pct(improvement.get('pp'))}), "
        f"tg {_num(expected.get('tg'))} t/s ({_pct(improvement.get('tg'))}).",
    ]
    if config.flash_attn:
        lines.append(
            "# Note: '-fa on' is the modern flash-attention spelling; older"
            " llama.cpp builds accept a bare '-fa'."
        )
    lines += [
        "#",
        "# llama-server:",
        "# " + shlex.join(["llama-server", "-m", model_path, *runtime_flags]),
        "#",
        "# llama-cli:",
        "# " + shlex.join(["llama-cli", "-m", model_path, *runtime_flags]),
        "",
    ]
    seen_alternates: set[tuple[int, str]] = set()
    for row in context_envelope or []:
        fallback = row.get("fallback_config")
        if not isinstance(fallback, dict):
            continue
        alternate = TrialConfig.from_dict(fallback)
        alternate_key = (int(row["ctx"]), alternate.trial_id)
        if alternate_key in seen_alternates:
            continue
        seen_alternates.add(alternate_key)
        alternate_flags = [*_cli_flags(alternate, moe=moe), "-c", str(row["ctx"])]
        lines.append(f"# Alternate for {row['ctx']} context:")
        if not _is_lossless(alternate.to_dict()):
            lines.append(
                "# QUALITY-AFFECTING: lossy KV-cache alternate; validate output quality "
                "before adoption."
            )
        lines += [
            "# " + shlex.join(["llama-server", "-m", model_path, *alternate_flags]),
            "# " + shlex.join(["llama-cli", "-m", model_path, *alternate_flags]),
            "",
        ]
    return "\n".join(lines)


def _num(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{float(value):.2f}"


def _pct(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{float(value):+.2f}%"


def write_outputs(
    session: Session,
    analysis: dict[str, Any],
    *,
    model: ModelReport,
    llama: LlamaCppReport,
    options: TuneOptions,
    recommend_config: TrialConfig,
    expected: dict[str, Any],
    confirmed: bool,
) -> None:
    """Write analysis.json, recommended.json, recommended.sh and report.md."""
    session.write_analysis(analysis)

    recommended = build_recommended_json(
        config=recommend_config,
        capabilities=llama.capabilities,
        expected=expected,
        confirmed=confirmed,
        target=options.target,
        model=model,
        llama=llama,
        pp=options.pp,
        tg=options.tg,
        depth=options.depth,
    )
    session.write_text("recommended.json", json.dumps(recommended, indent=2, sort_keys=True) + "\n")

    session.write_text(
        "recommended.sh",
        build_recommended_sh(
            config=recommend_config,
            model=model,
            expected=expected,
            confirmed=confirmed,
            target=options.target,
            moe=model.moe,
            ctx_size=options.ctx_size,
            context_envelope=analysis.get("context_envelope"),
        ),
    )

    session_meta = _read_json(session, "session.json")
    hardware = _read_json(session, "hardware.json")
    model_meta = _read_json(session, "model.json")
    llamacpp = _read_json(session, "llamacpp.json")
    report_text = report_module.render(analysis, session_meta, hardware, model_meta, llamacpp)
    session.write_text("report.md", report_text)


def _read_json(session: Session, name: str) -> dict[str, Any]:
    return session.read_json(name)
