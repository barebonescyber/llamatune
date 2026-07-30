"""Pure Markdown rendering for a completed Night Shift summary."""

from __future__ import annotations

from typing import Any


def _pct(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "-"
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(float(value)) * 100:.1f}%"


def _calibration_text(calibration: dict[str, Any]) -> str:
    verdict = str(calibration.get("verdict") or "unknown")
    source = calibration.get("transfer_from")
    label = f"transfer-{verdict} vs {str(source)[:8]}" if source else verdict
    if verdict == "error":
        return f"failed: {calibration.get('reason') or 'unknown'}"
    changes = []
    if calibration.get("drift_pp") is not None:
        signed = calibration.get("drift_pp_signed")
        changes.append(
            f"pp {_pct(signed)}"
            if signed is not None
            else f"pp drift {_pct(calibration['drift_pp'])}"
        )
    if calibration.get("drift_tg") is not None:
        signed = calibration.get("drift_tg_signed")
        changes.append(
            f"tg {_pct(signed)}"
            if signed is not None
            else f"tg drift {_pct(calibration['drift_tg'])}"
        )
    return f"{label} ({', '.join(changes)})" if changes else label


def _item_result(item: dict[str, Any], *, retuned: bool = False) -> str:
    calibration = item.get("calibration")
    if isinstance(calibration, dict):
        text = _calibration_text(calibration)
    else:
        outcome = item.get("outcome") or item.get("result") or "unknown"
        reason = item.get("reason")
        text = f"failed: {reason}" if outcome in {"failed", "error"} and reason else str(outcome)
    if item.get("retuned") or retuned:
        text += " → retuned"
        improvement = item.get("winner_improvement")
        if improvement is not None:
            text += f", new winner {_pct(improvement)}"
    return text


def _identity(summary: dict[str, Any]) -> list[str]:
    hardware = summary.get("hardware") or {}
    llama = summary.get("llamacpp") or summary.get("llama") or {}
    gpu_names = ", ".join(str(gpu.get("name", "-")) for gpu in hardware.get("gpus") or [])
    return [
        f"- Machine: {hardware.get('cpu_model', '-')} / {gpu_names or 'no GPU'}",
        f"- llama.cpp build: {llama.get('build_commit') or '-'} "
        f"(help SHA-256 `{llama.get('help_sha256') or '-'}`)",
    ]


def render(summary: dict[str, Any]) -> str:
    """Render ``nightshift-report.md`` using summary content only."""
    window = summary.get("window") or {}
    counts = summary.get("counts") or {}
    items = summary.get("items") or []
    lines = [
        "# llamatune Night Shift report (experimental)",
        "",
        "## Shift summary",
        "",
        f"- Started: {window.get('started') or summary.get('started') or '-'}",
        f"- Ended: {window.get('ended') or summary.get('ended') or '-'}",
        f"- Deadline: {window.get('deadline') or 'none'}",
        f"- Deadline outcome: {window.get('outcome') or summary.get('outcome') or '-'}",
        f"- Total benchmark invocations: {summary.get('total_invocations', 0)}",
        *_identity(summary),
    ]
    if counts:
        lines.extend(
            ["- Counts by phase:", *(f"  - {key}: {value}" for key, value in counts.items())]
        )

    groups = summary.get("content_groups") or []
    if groups:
        lines.append("- Content groups:")
        for group in groups:
            if isinstance(group, dict):
                lines.append(
                    f"  - {group.get('key') or group.get('group_key') or '-'}: "
                    f"representative {group.get('representative') or '-'}"
                )
            else:
                lines.append(f"  - {group}")

    if any(
        isinstance(item, dict)
        and isinstance(item.get("calibration"), dict)
        and item["calibration"].get("build_changed")
        for item in items
    ):
        lines.append("- Warning: llama.cpp build changed since recorded tuning evidence.")

    lines.extend(
        [
            "",
            "## Per-model results",
            "",
            "| Model | Fingerprint | Action(s) | Verdict / result |",
            "|---|---|---|---|",
        ]
    )
    active = [
        item
        for item in items
        if isinstance(item, dict) and item.get("outcome") not in {"deferred", "skipped"}
    ]
    retuned_fingerprints = {
        item.get("fingerprint")
        for item in active
        if item.get("kind") == "retune" and item.get("outcome") == "succeeded"
    }
    if active:
        for item in active:
            model = item.get("model")
            if isinstance(model, dict):
                model_name = model.get("name") or model.get("path") or "-"
            else:
                model_name = model or item.get("model_path") or "-"
            action = item.get("kind") or item.get("action") or "-"
            was_retuned = item.get("fingerprint") in retuned_fingerprints and action == "calibrate"
            lines.append(
                f"| {model_name} | `{str(item.get('fingerprint') or '-')[:16]}` | "
                f"{action} | {_item_result(item, retuned=was_retuned)} |"
            )
    else:
        lines.append("| _None_ | - | - | - |")

    lines.extend(["", "## Deferred and skipped", ""])
    deferred = [
        item
        for item in items
        if isinstance(item, dict) and item.get("outcome") in {"deferred", "skipped"}
    ]
    if deferred:
        lines.extend(["| Item | Outcome | Reason | Estimate |", "|---|---|---|---|"])
        for item in deferred:
            label = item.get("model") or item.get("model_path") or item.get("kind") or "-"
            estimate = item.get("estimated_minutes")
            estimate_text = f"{estimate} min" if estimate is not None else "-"
            lines.append(
                f"| {label} | {item.get('outcome')} | {item.get('reason') or '-'} | "
                f"{estimate_text} |"
            )
    else:
        lines.append("_None._")

    lines.extend(["", "## Warnings", ""])
    warnings = summary.get("warnings") or []
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("_None._")
    return "\n".join(lines) + "\n"
