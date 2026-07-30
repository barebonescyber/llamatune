"""Pure deterministic Markdown and CSV rendering for Results Matrix documents."""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Callable
from typing import Any

from llamatune.types import ResultRow, ResultsMatrix

_GOOD_STATUSES = frozenset({"ok", "replicated", "consistent"})
_RECOMMENDATION_KINDS = frozenset({"recommendation", "lossless_recommendation"})
_CSV_FIELDS = (
    "row_id",
    "kind",
    "current",
    "model_fingerprint",
    "model_name",
    "model_path",
    "quant",
    "hardware_hash",
    "build_discriminator",
    "build_commit",
    "ctx",
    "depth",
    "pp_workload",
    "tg_workload",
    "suite_id",
    "status",
    "confirmed",
    "replicated",
    "reps",
    "noise_floor_cv",
    "source_root",
    "evidence_dir",
    "ts",
)


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def _config(row: ResultRow) -> str:
    config = row.config
    if config is None:
        return "defaults"
    return "/".join(
        (
            f"ngl={config.gpu_layers}",
            f"ncmoe={config.moe_cpu_layers}",
            f"fa={int(config.flash_attn)}",
            f"ub={config.ubatch}",
            f"b={config.batch}",
            f"t={config.threads}",
        )
    )


def _balanced(row: ResultRow) -> float | None:
    pp = row.metrics.get("perf.pp")
    tg = row.metrics.get("perf.tg")
    if pp is None or tg is None or pp < 0 or tg < 0:
        return None
    return math.sqrt(pp * tg)


_BEST_CASES: tuple[
    tuple[str, frozenset[str], str | None, Callable[[ResultRow], float | None]], ...
] = (
    (
        "max-pp",
        frozenset({"recommendation", "baseline", "operating_point"}),
        None,
        lambda r: r.metrics.get("perf.pp"),
    ),
    (
        "max-tg",
        frozenset({"recommendation", "baseline", "operating_point"}),
        None,
        lambda r: r.metrics.get("perf.tg"),
    ),
    ("balanced", frozenset({"recommendation", "baseline", "operating_point"}), None, _balanced),
    ("max-context", frozenset(), None, lambda r: r.metrics.get("ctx.validated")),
    (
        "coding",
        frozenset({"quality_suite"}),
        "coding",
        lambda r: r.metrics.get("quality.coding.score"),
    ),
    (
        "tool-use",
        frozenset({"quality_suite"}),
        "tooluse",
        lambda r: r.metrics.get("quality.tooluse.score"),
    ),
    (
        "agentic",
        frozenset({"quality_suite"}),
        "agentic",
        lambda r: r.metrics.get("quality.agentic.score"),
    ),
    (
        "instruction",
        frozenset({"quality_suite"}),
        "ifollow",
        lambda r: r.metrics.get("quality.ifollow.score"),
    ),
    (
        "quality-overall",
        frozenset({"quality_suite"}),
        None,
        lambda r: r.metrics.get("quality.overall"),
    ),
)


def _suite(row: ResultRow) -> str | None:
    return row.suite_id.split("@", 1)[0] if row.suite_id is not None else None


def _best(
    rows: list[ResultRow],
    name: str,
    kinds: frozenset[str],
    suite: str | None,
    metric: Callable[[ResultRow], float | None],
) -> list[tuple[ResultRow, float]]:
    candidates = [
        row
        for row in rows
        if row.current
        and row.confirmed
        and (row.status in _GOOD_STATUSES or row.kind in _RECOMMENDATION_KINDS)
        and (not kinds or row.kind in kinds)
        and (suite is None or _suite(row) == suite)
        and metric(row) is not None
    ]
    grouped: dict[str, list[ResultRow]] = {}
    for row in candidates:
        grouped.setdefault(row.hardware_hash, []).append(row)
    winners: list[tuple[ResultRow, float]] = []
    recommendation_tg: dict[tuple[Any, ...], float] = {}
    for row in rows:
        if not row.current or row.kind != "recommendation" or "perf.tg" not in row.metrics:
            continue
        key = (row.hardware_hash, row.build_discriminator, row.config)
        recommendation_tg[key] = max(row.metrics["perf.tg"], recommendation_tg.get(key, -math.inf))
    for hardware_hash in sorted(grouped):
        ordered = sorted(grouped[hardware_hash], key=lambda row: row.row_id)
        ordered.sort(key=lambda row: row.ts, reverse=True)
        if name in {"coding", "tool-use", "agentic", "instruction", "quality-overall"}:
            ordered.sort(
                key=lambda row: (
                    (row.hardware_hash, row.build_discriminator, row.config) in recommendation_tg,
                    recommendation_tg.get(
                        (row.hardware_hash, row.build_discriminator, row.config), -math.inf
                    ),
                ),
                reverse=True,
            )
        elif name == "max-pp":
            ordered.sort(key=lambda row: row.metrics.get("perf.tg", -math.inf), reverse=True)
        elif name == "max-tg":
            ordered.sort(key=lambda row: row.metrics.get("perf.pp", -math.inf), reverse=True)
        elif name == "max-context":
            ordered.sort(key=lambda row: row.metrics.get("perf.tg", -math.inf), reverse=True)
        ordered.sort(key=lambda row: metric(row) or 0.0, reverse=True)
        winner = ordered[0]
        value = metric(winner)
        if value is None:  # candidates exclude this case
            continue
        winners.append((winner, value))
    return winners


def _model_label(row: ResultRow) -> str:
    return row.model_name or row.model_path


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_escape(value) for value in row) + " |" for row in rows)
    return lines


def render_markdown(matrix: ResultsMatrix) -> str:
    """Render an identity-neutral Results Matrix Markdown artifact."""
    current = sum(row.current for row in matrix.rows)
    model_groups: dict[str, list[ResultRow]] = {}
    for row in matrix.rows:
        model_groups.setdefault(row.model_fingerprint, []).append(row)
    ordered_models = sorted(
        model_groups.values(),
        key=lambda rows: (_model_label(rows[0]).casefold(), rows[0].model_fingerprint),
    )

    lines = [
        "# Results Matrix",
        "",
        f"Generated: `{_escape(matrix.generated)}`  ",
        f"Roots: {', '.join(f'`{_escape(root)}`' for root in matrix.roots) or 'none'}  ",
        f"Counts: {len(matrix.rows)} rows, {current} current, {len(model_groups)} models",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {_escape(warning)}" for warning in matrix.warnings)
    if not matrix.warnings:
        lines.append("None.")

    for rows in ordered_models:
        representative = sorted(rows, key=lambda row: row.row_id)[0]
        lines.extend(
            (
                "",
                f"## {_escape(_model_label(representative))}",
                "",
                (
                    f"Identity: quant `{_escape(representative.quant or 'unknown')}`, "
                    f"fingerprint `{representative.model_fingerprint[:16]}`, size `unavailable`"
                ),
                "",
                "### Best results",
                "",
            )
        )
        best_rows: list[tuple[Any, ...]] = []
        for name, kinds, suite, metric in _BEST_CASES:
            for winner, value in _best(rows, name, kinds, suite, metric):
                best_rows.append(
                    (
                        name,
                        winner.hardware_hash,
                        _config(winner),
                        _number(value),
                        "",
                        winner.evidence_dir,
                    )
                )
        if best_rows:
            lines.extend(
                _table(
                    ("Use case", "Hardware", "Config", "Result", "Compat", "Evidence"),
                    best_rows,
                )
            )
        else:
            lines.append("No current confirmed results.")

        operating = sorted(
            (
                row
                for row in rows
                if row.current and row.kind in {"operating_point", "context_envelope"}
            ),
            key=lambda row: (
                row.hardware_hash,
                row.ctx if row.ctx is not None else -1,
                row.depth if row.depth is not None else -1,
                row.kind,
                row.row_id,
            ),
        )
        if operating:
            lines.extend(("", "### Operating points", ""))
            lines.extend(
                _table(
                    ("Kind", "Hardware", "Context", "Depth", "PP", "TG", "Confirmed", "Evidence"),
                    [
                        (
                            row.kind,
                            row.hardware_hash,
                            row.ctx or "",
                            row.depth or "",
                            _number(row.metrics.get("perf.pp")),
                            _number(row.metrics.get("perf.tg")),
                            "yes" if row.confirmed else "no",
                            row.evidence_dir,
                        )
                        for row in operating
                    ],
                )
            )

        quality = sorted(
            (row for row in rows if row.current and row.kind == "quality_suite"),
            key=lambda row: (_suite(row) or "", row.hardware_hash, row.ts, row.row_id),
        )
        if quality:
            lines.extend(("", "### Quality", ""))
            lines.extend(
                _table(
                    ("Suite", "Hardware", "Score", "Overall", "Confirmed", "Evidence"),
                    [
                        (
                            _suite(row) or "unknown",
                            row.hardware_hash,
                            _number(
                                next(
                                    (
                                        value
                                        for key, value in sorted(row.metrics.items())
                                        if key.startswith("quality.") and key.endswith(".score")
                                    ),
                                    None,
                                )
                            ),
                            _number(row.metrics.get("quality.overall")),
                            "yes" if row.confirmed else "no",
                            row.evidence_dir,
                        )
                        for row in quality
                    ],
                )
            )

        superseded = sum(not row.current for row in rows)
        lines.extend(("", f"History: {superseded} superseded row(s)."))

    return "\n".join(lines) + "\n"


def render_csv(rows: tuple[ResultRow, ...]) -> str:
    """Render flat scalar row fields plus a sorted union of metrics."""
    metric_fields = tuple(sorted({name for row in rows for name in row.metrics}))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=(*_CSV_FIELDS, *metric_fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        document = row.to_dict()
        record = {field: document[field] for field in _CSV_FIELDS}
        record.update({name: row.metrics.get(name, "") for name in metric_fields})
        writer.writerow(record)
    return output.getvalue()
