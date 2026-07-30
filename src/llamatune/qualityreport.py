"""Deterministic quality-report rendering from quality.json content alone."""

from __future__ import annotations

import json
import shlex
from typing import Any


def _text(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _score(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.6g}"
    return "unavailable"


def _comparison_metrics(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return _score(value)


def _argv(value: Any) -> str:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return shlex.join(value)
    if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
        return shlex.join(value)
    return str(value) if value else "unavailable in schema v1"


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_text(value) for value in row) + " |" for row in rows)
    return lines


def _failed_details(task: dict[str, Any]) -> str:
    failures = [
        str(result.get("detail", "failed"))
        for result in task.get("graders", [])
        if isinstance(result, dict) and result.get("passed") is False and not result.get("skipped")
    ]
    return (
        "; ".join(failures) if failures else str(task.get("reason") or "score below pass threshold")
    )


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _array(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def render(summary: dict[str, Any]) -> str:
    """Render all required report sections without filesystem access."""
    model = _object(summary.get("model"))
    config = _object(summary.get("config"))
    warnings = _array(summary.get("warnings"))
    suites = _array(summary.get("suites"))
    raw_comparison = summary.get("comparison")
    comparison = _object(raw_comparison) if isinstance(raw_comparison, dict) else None

    lines = [
        "# Quality Evaluation Report (experimental)",
        "",
        "## Summary",
        "",
        f"- Model: `{_text(model.get('name') or model.get('path') or 'unknown')}`",
        f"- Fingerprint: `{_text(model.get('fingerprint', 'unknown'))}`",
        f"- Configuration source: `{_text(summary.get('config_source', 'unknown'))}`",
        f"- Configuration provenance: `{_text(summary.get('config_provenance', 'none'))}`",
        f"- Context: `{_text(summary.get('ctx', 'unknown'))}`",
        f"- Seed: `{_text(summary.get('seed', 'unknown'))}`",
        f"- Execution tier: `{'enabled' if summary.get('exec_enabled') else 'disabled'}`",
        f"- Overall score: `{_score(summary.get('overall'))}`",
        f"- Warnings: `{len(warnings)}`",
        "",
        "## Configuration",
        "",
    ]
    preferred = (
        "gpu_layers",
        "moe_cpu_layers",
        "flash_attn",
        "ubatch",
        "batch",
        "threads",
        "mmap",
        "no_kv_offload",
        "cache_type_k",
        "cache_type_v",
        "threads_batch",
        "ot_spec",
        "tensor_split",
        "split_mode",
    )
    ordered_keys = [key for key in preferred if key in config]
    ordered_keys.extend(sorted(set(config) - set(ordered_keys)))
    lines.extend(
        _table(
            ("Setting", "Value"),
            [(key, json.dumps(config[key], sort_keys=True)) for key in ordered_keys],
        )
    )
    lines.extend(
        (
            "",
            f"Source: `{_text(summary.get('config_source', 'unknown'))}`",
            f"Provenance: `{_text(summary.get('config_provenance', 'none'))}`",
        )
    )
    if comparison is not None:
        lossless = comparison.get("lossless_config")
        lines.append(f"Lossless configuration: `{_text(json.dumps(lossless, sort_keys=True))}`")
        deltas = _array(comparison.get("suites"))
        lines.extend(("", "Lossless suite comparison:", ""))
        lines.extend(
            _table(
                ("Suite", "Evaluated", "Lossless", "Delta score"),
                [
                    (
                        row.get("suite_id", "unknown"),
                        _comparison_metrics(row.get("evaluated")),
                        _comparison_metrics(row.get("lossless")),
                        _score(row.get("delta_score")),
                    )
                    for row in deltas
                    if isinstance(row, dict)
                ],
            )
        )
        if comparison.get("degradation_warning"):
            warning = comparison["degradation_warning"]
            if warning is True:
                warning = "lossy cache measurably degrades quality on this machine"
            lines.append(f"Degradation warning: {_text(warning)}")

    lines.extend(("", "## Per-suite results", ""))
    for suite in suites:
        if not isinstance(suite, dict):
            continue
        name = str(suite.get("name") or suite.get("kind") or "unknown")
        lines.extend((f"### {_text(name)}", ""))
        metrics = _object(suite.get("metrics"))
        lines.extend(
            _table(("Metric", "Value"), [(key, _score(metrics[key])) for key in sorted(metrics)])
        )
        tasks = _array(suite.get("tasks"))
        failed = [
            task
            for task in tasks
            if isinstance(task, dict)
            and task.get("status") not in {"skipped", "error"}
            and float(task.get("score", 0.0)) < 0.999
        ]
        lines.append("")
        lines.append("Failed tasks:")
        if failed:
            lines.extend(
                f"- `{_text(task.get('id', 'unknown'))}`: {_text(_failed_details(task))}"
                for task in failed
            )
        else:
            lines.append("- None.")
        lines.append("")

    lines.extend(("## Skipped and errored tasks", ""))
    exceptional = [
        (str(suite.get("name") or suite.get("kind") or "unknown"), task)
        for suite in suites
        if isinstance(suite, dict)
        for task in _array(suite.get("tasks"))
        if isinstance(task, dict) and task.get("status") in {"skipped", "error"}
    ]
    if exceptional:
        lines.extend(
            f"- `{_text(suite_name)}/{_text(task.get('id', 'unknown'))}` "
            f"({_text(task.get('status'))}): {_text(task.get('reason') or 'no reason recorded')}"
            for suite_name, task in exceptional
        )
    else:
        lines.append("None.")

    build = _object(summary.get("build"))
    launches = summary.get("server_launches", [])
    hardware = _text(json.dumps(summary.get("hardware_signature", []), sort_keys=True))
    discriminator = _text(build.get("bench_sha256") or build.get("help_sha256") or "unknown")
    launch_count: int | str
    if "server_launches" not in summary:
        launch_count = "unavailable in schema v1"
    else:
        launch_count = len(launches) if isinstance(launches, list) else _text(launches)
    lines.extend(
        (
            "",
            "## Environment",
            "",
            f"- Hardware: `{hardware}`",
            f"- Build discriminator: `{discriminator}`",
            f"- Build commit: `{_text(build.get('build_commit', 'unknown'))}`",
            f"- Server launches: `{launch_count}`",
        )
    )
    if warnings:
        lines.extend(("", "Warnings:"))
        lines.extend(f"- {_text(warning)}" for warning in warnings)

    server_argv = summary.get("server_argv")
    if server_argv is None and isinstance(launches, list) and launches:
        launch = launches[-1]
        if isinstance(launch, dict):
            server_argv = launch.get("argv")
    lines.extend(
        (
            "",
            "## Reproduce",
            "",
            "```console",
            f"{_argv(summary.get('argv'))}",
            "```",
            "",
            "Server argv:",
            "",
            "```console",
            f"{_argv(server_argv)}",
            "```",
            "",
        )
    )
    return "\n".join(lines)
