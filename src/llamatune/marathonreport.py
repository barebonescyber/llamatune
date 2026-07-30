"""Pure Markdown rendering for ``marathon.json`` summaries."""

from __future__ import annotations

from typing import Any


def _value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _config(config: Any) -> str:
    if not isinstance(config, dict):
        return "defaults"
    return ", ".join(f"{key}={value}" for key, value in sorted(config.items()))


def _row(*values: Any) -> str:
    return "| " + " | ".join(_value(value) for value in values) + " |"


def render(summary: dict[str, Any]) -> str:
    """Render a complete, absence-tolerant Marathon report from one dict."""
    lines = ["# llamatune Marathon report (experimental)", "", "## Summary", ""]
    champion = summary.get("champion") or {}
    verification = summary.get("ab") or summary.get("verification") or {}
    verdict = (
        verification.get("verdict", "not run") if isinstance(verification, dict) else "not run"
    )
    lines.extend(
        [
            f"- Stop reason: {_value(summary.get('stop_reason'))}",
            f"- Rounds: {_value(len(summary.get('rounds', [])))}",
            f"- Wall time: {_value(summary.get('wall_s'))} s",
            f"- Replication verdict: **{_value(verdict)}**",
        ]
    )
    if verdict not in ("a", "replicated", "not run"):
        lines.append(
            "- **NOT REPLICATED:** the round-confirmed improvement did not "
            "replicate in A/B testing."
        )
    lines.extend(
        [
            "",
            "## Champion",
            "",
            "Configuration: `"
            + _config(champion.get("config") if isinstance(champion, dict) else None)
            + "`",
            "",
        ]
    )
    if isinstance(champion, dict):
        lines.append(
            f"Confirmed pp/tg: {_value(champion.get('pp'))} / {_value(champion.get('tg'))}"
        )
        command = champion.get("command") or champion.get("reproduce_command")
        if command:
            lines.extend(["", "```sh", str(command), "```"])
    lines.extend(
        [
            "",
            "## Round history",
            "",
            "| Round | Budget | Outcome | Winner | Challenge | Wall |",
            "|---:|---:|---|---|---|---:|",
        ]
    )
    for row in summary.get("rounds", []):
        lines.append(
            _row(
                row.get("index"),
                row.get("budget_trials", row.get("budget")),
                row.get("outcome"),
                _config(row.get("winner_config", row.get("winner"))),
                row.get("challenge_verdict"),
                row.get("wall_s"),
            )
        )
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "| Tier | Enumerated | Executed | Pruned | Remaining |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    ledger = summary.get("ledger") or summary.get("coverage") or {}
    tiers = ledger.get("tiers", ledger) if isinstance(ledger, dict) else {}
    if isinstance(tiers, dict):
        for name, tier in sorted(tiers.items()):
            if isinstance(tier, dict):
                lines.append(
                    _row(
                        name,
                        tier.get("enumerated"),
                        tier.get("executed"),
                        tier.get("pruned"),
                        tier.get("remaining"),
                    )
                )
    responsive = ledger.get("responsive", []) if isinstance(ledger, dict) else []
    lines.extend(
        [
            "",
            f"Responsive dimensions: {', '.join(map(str, responsive)) or 'none'}",
            "",
            "## Operating points",
            "",
            "| Context | Depth | Status | pp | tg | Refined | Configuration |",
            "|---:|---:|---|---:|---:|---|---|",
        ]
    )
    overrides: list[dict[str, Any]] = []
    for row in summary.get("matrix", []):
        lines.append(
            _row(
                row.get("ctx"),
                row.get("depth"),
                row.get("status"),
                row.get("pp"),
                row.get("tg"),
                row.get("refined", False),
                _config(row.get("config")),
            )
        )
        if (
            row.get("config")
            and isinstance(champion, dict)
            and row.get("config") != champion.get("config")
        ):
            overrides.append(row)
    if overrides:
        lines.extend(["", "### Per-context/depth overrides", ""])
        for row in overrides:
            lines.append(
                f"- `ctx={row.get('ctx')} depth={row.get('depth')}`: `{_config(row.get('config'))}`"
            )
    lines.extend(["", "## Environment", ""])
    reference = summary.get("reference") or {}
    lines.append(
        "Reconnaissance trend: "
        + _value(reference.get("trend") if isinstance(reference, dict) else None)
    )
    for bracket in summary.get("brackets", []):
        lines.append(
            "- "
            + _value(bracket.get("timestamp", bracket.get("index")))
            + ": "
            + _value(bracket.get("verdict"))
            + " (pp "
            + _value(bracket.get("drift_pp"))
            + ", tg "
            + _value(bracket.get("drift_tg"))
            + ")"
        )
    lines.extend(
        [
            "",
            "## A/B verification",
            "",
            "| Block | A pp | A tg | B pp | B tg | Verdict |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    blocks = (
        verification.get("block_results", verification.get("blocks_detail", []))
        if isinstance(verification, dict)
        else []
    )
    for block in blocks if isinstance(blocks, list) else []:
        lines.append(
            _row(
                block.get("block"),
                block.get("a_pp"),
                block.get("a_tg"),
                block.get("b_pp"),
                block.get("b_tg"),
                block.get("verdict"),
            )
        )
    lines.extend(["", f"Verdict: **{_value(verdict)}**", "", "## Deferred and failed", ""])
    items = list(summary.get("deferred", [])) + list(summary.get("failed", []))
    lines.extend(
        f"- {_value(item.get('item', item.get('kind')))}: {_value(item.get('reason'))}"
        for item in items
    )
    if not items:
        lines.append("None.")
    lines.extend(["", "## Warnings", ""])
    warnings = summary.get("warnings", [])
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("None.")
    return "\n".join(lines) + "\n"
