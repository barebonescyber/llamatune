"""Pure, deterministic Results Matrix filtering and ranking."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from llamatune.types import MatrixQuerySpec, ResultRow, ResultsMatrix

_GOOD_STATUSES = frozenset({"ok", "replicated", "consistent"})
_RECOMMENDATION_KINDS = frozenset({"recommendation", "lossless_recommendation"})
_PERFORMANCE_KINDS = frozenset({"recommendation", "baseline", "operating_point"})


@dataclass(frozen=True, slots=True)
class UseCaseDef:
    """One named, frozen Results Matrix ranking definition."""

    name: str
    description: str
    kinds: frozenset[str]
    suite: str | None
    metric_fn: Callable[[ResultRow], float | None]
    tiebreak_fn: Callable[[ResultRow], tuple[float, ...]]


def _metric(name: str) -> Callable[[ResultRow], float | None]:
    return lambda row: row.metrics.get(name)


def _balanced(row: ResultRow) -> float | None:
    pp = row.metrics.get("perf.pp")
    tg = row.metrics.get("perf.tg")
    if pp is None or tg is None or pp < 0.0 or tg < 0.0:
        return None
    return math.sqrt(pp * tg)


def _metric_tiebreak(name: str) -> Callable[[ResultRow], tuple[float, ...]]:
    return lambda row: (row.metrics.get(name, -math.inf),)


def _no_tiebreak(_row: ResultRow) -> tuple[float, ...]:
    return ()


USE_CASES: dict[str, UseCaseDef] = {
    "max-pp": UseCaseDef(
        "max-pp",
        "maximum prompt-processing throughput",
        _PERFORMANCE_KINDS,
        None,
        _metric("perf.pp"),
        _metric_tiebreak("perf.tg"),
    ),
    "max-tg": UseCaseDef(
        "max-tg",
        "maximum token-generation throughput",
        _PERFORMANCE_KINDS,
        None,
        _metric("perf.tg"),
        _metric_tiebreak("perf.pp"),
    ),
    "balanced": UseCaseDef(
        "balanced",
        "geometric mean of prompt and generation throughput",
        _PERFORMANCE_KINDS,
        None,
        _balanced,
        _no_tiebreak,
    ),
    "max-context": UseCaseDef(
        "max-context",
        "largest validated context",
        frozenset(),
        None,
        _metric("ctx.validated"),
        _metric_tiebreak("perf.tg"),
    ),
    "coding": UseCaseDef(
        "coding",
        "coding quality score",
        frozenset({"quality_suite"}),
        "coding",
        _metric("quality.coding.score"),
        _no_tiebreak,
    ),
    "tool-use": UseCaseDef(
        "tool-use",
        "tool-use quality score",
        frozenset({"quality_suite"}),
        "tooluse",
        _metric("quality.tooluse.score"),
        _no_tiebreak,
    ),
    "agentic": UseCaseDef(
        "agentic",
        "agentic quality score",
        frozenset({"quality_suite"}),
        "agentic",
        _metric("quality.agentic.score"),
        _no_tiebreak,
    ),
    "instruction": UseCaseDef(
        "instruction",
        "instruction-following quality score",
        frozenset({"quality_suite"}),
        "ifollow",
        _metric("quality.ifollow.score"),
        _no_tiebreak,
    ),
    "quality-overall": UseCaseDef(
        "quality-overall",
        "overall quality score",
        frozenset({"quality_suite"}),
        None,
        _metric("quality.overall"),
        _no_tiebreak,
    ),
}


def _suite_name(row: ResultRow) -> str | None:
    if row.suite_id is None:
        return None
    return row.suite_id.split("@", 1)[0]


def _compatibility(row: ResultRow, identity: tuple[str, str] | None) -> str:
    if identity is None:
        return "unknown"
    same_hardware = row.hardware_hash == identity[0]
    same_build = row.build_discriminator == identity[1]
    if same_hardware and same_build:
        return "current"
    if same_hardware:
        return "build_changed"
    if same_build:
        return "hardware_changed"
    return "both"


def _matches_model(row: ResultRow, requested: str) -> bool:
    needle = requested.casefold()
    values = (row.model_name or "", row.model_path, row.model_fingerprint)
    return any(needle in value.casefold() for value in values)


def _base_filter(
    row: ResultRow,
    spec: MatrixQuerySpec,
    identity: tuple[str, str] | None,
) -> bool:
    if spec.current_only and not row.current:
        return False
    if row.status not in _GOOD_STATUSES and row.kind not in _RECOMMENDATION_KINDS:
        return False
    if spec.model is not None and not _matches_model(row, spec.model):
        return False
    if spec.quant is not None and (row.quant or "").casefold() != spec.quant.casefold():
        return False
    if spec.kinds and row.kind not in spec.kinds:
        return False
    if spec.suite is not None and _suite_name(row) != spec.suite:
        return False
    if spec.ctx_min is not None and (row.ctx is None or row.ctx < spec.ctx_min):
        return False
    if spec.depth is not None and row.depth != spec.depth:
        return False
    if spec.min_pp is not None:
        pp = row.metrics.get("perf.pp")
        if pp is None or pp < spec.min_pp:
            return False
    if spec.min_tg is not None:
        tg = row.metrics.get("perf.tg")
        if tg is None or tg < spec.min_tg:
            return False
    if spec.confirmed_only and not row.confirmed:
        return False
    compatibility = _compatibility(row, identity)
    return spec.compat in {"all", "any"} or compatibility == spec.compat


def _use_case_filter(row: ResultRow, definition: UseCaseDef) -> bool:
    if definition.kinds and row.kind not in definition.kinds:
        return False
    return definition.suite is None or _suite_name(row) == definition.suite


def _recommendation_tg(rows: list[ResultRow]) -> dict[tuple[Any, ...], float]:
    throughputs: dict[tuple[Any, ...], float] = {}
    for row in rows:
        if row.kind != "recommendation" or not row.current:
            continue
        tg = row.metrics.get("perf.tg")
        if tg is None:
            continue
        key = (
            row.model_fingerprint,
            row.hardware_hash,
            row.build_discriminator,
            row.config,
        )
        throughputs[key] = max(tg, throughputs.get(key, -math.inf))
    return throughputs


def _quality_tiebreak(
    row: ResultRow,
    recommendation_tg: dict[tuple[Any, ...], float],
) -> tuple[float, float]:
    key = (
        row.model_fingerprint,
        row.hardware_hash,
        row.build_discriminator,
        row.config,
    )
    tg = recommendation_tg.get(key)
    return (1.0, tg) if tg is not None else (0.0, -math.inf)


def _rank_rows(
    rows: list[ResultRow],
    metric_fn: Callable[[ResultRow], float | None],
    definition: UseCaseDef | None,
    ascending: bool,
    recommendation_tg: dict[tuple[Any, ...], float],
) -> list[ResultRow]:
    ranked = sorted(rows, key=lambda row: row.row_id)
    ranked.sort(key=lambda row: row.ts, reverse=True)
    if definition is not None and definition.kinds == frozenset({"quality_suite"}):
        ranked.sort(key=lambda row: _quality_tiebreak(row, recommendation_tg), reverse=True)
    elif definition is not None:
        ranked.sort(key=definition.tiebreak_fn, reverse=True)
    reverse = not ascending

    def rank_key(row: ResultRow) -> tuple[int, float]:
        value = metric_fn(row)
        if value is None:
            return (1, 0.0)
        return (0, -value) if reverse else (0, value)

    ranked.sort(key=rank_key)
    return ranked


def _filters(spec: MatrixQuerySpec) -> dict[str, Any]:
    return {
        "model": spec.model,
        "quant": spec.quant,
        "kinds": list(spec.kinds),
        "suite": spec.suite,
        "ctx_min": spec.ctx_min,
        "depth": spec.depth,
        "min_pp": spec.min_pp,
        "min_tg": spec.min_tg,
        "confirmed_only": spec.confirmed_only,
        "current_only": spec.current_only,
        "compat": spec.compat,
        "limit": spec.limit,
        "ascending": spec.ascending,
    }


def apply(
    matrix: ResultsMatrix,
    spec: MatrixQuerySpec,
    identity: tuple[str, str] | None,
) -> dict[str, Any]:
    """Filter, group, and deterministically rank a Results Matrix document."""
    if spec.use_case is not None and spec.sort is not None:
        raise ValueError("use_case and sort are mutually exclusive")
    if spec.use_case is None and spec.sort is None:
        raise ValueError("one of use_case or sort is required")
    if spec.limit < 0:
        raise ValueError("limit must be non-negative")

    definition: UseCaseDef | None = None
    if spec.use_case is not None:
        try:
            definition = USE_CASES[spec.use_case]
        except KeyError as exc:
            raise ValueError(f"unknown matrix use case: {spec.use_case}") from exc
        metric_fn = definition.metric_fn
    else:
        sort_name = spec.sort
        if sort_name is None:  # guarded above; keeps the narrowing explicit
            raise ValueError("one of use_case or sort is required")
        metric_fn = _metric(sort_name)

    all_eligible = [row for row in matrix.rows if _base_filter(row, spec, identity)]
    pool = (
        [row for row in all_eligible if _use_case_filter(row, definition)]
        if definition is not None
        else all_eligible
    )
    measurable = [row for row in pool if metric_fn(row) is not None]
    excluded = {"missing_metric": len(pool) - len(measurable)}
    recommendation_tg = _recommendation_tg(list(matrix.rows))

    by_hardware: dict[str, list[ResultRow]] = {}
    for row in measurable:
        by_hardware.setdefault(row.hardware_hash, []).append(row)

    current_hardware = identity[0] if identity is not None else None
    hardware_order = sorted(
        by_hardware,
        key=lambda value: (value != current_hardware, value),
    )
    groups: list[dict[str, Any]] = []
    for hardware_hash in hardware_order:
        rows = _rank_rows(
            by_hardware[hardware_hash],
            metric_fn,
            definition,
            spec.ascending if definition is None else False,
            recommendation_tg,
        )
        if spec.limit:
            rows = rows[: spec.limit]
        rendered_rows: list[dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            rendered = row.to_dict()
            rendered["rank"] = rank
            rendered["compat"] = _compatibility(row, identity)
            rendered_rows.append(rendered)
        groups.append(
            {
                "hardware_hash": hardware_hash,
                "hardware_signature": list(rows[0].hardware_signature),
                "rows": rendered_rows,
            }
        )

    payload: dict[str, Any] = {
        "filters": _filters(spec),
        "identity": (
            {"hardware_hash": identity[0], "build_discriminator": identity[1]}
            if identity is not None
            else None
        ),
        "groups": groups,
        "excluded": excluded,
        "warnings": list(matrix.warnings),
    }
    payload["use_case" if spec.use_case is not None else "sort"] = (
        spec.use_case if spec.use_case is not None else spec.sort
    )
    return payload
