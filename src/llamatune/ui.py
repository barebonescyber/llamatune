"""Ephemeral stderr progress renderers for tuning runs."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import TextIO

from llamatune.sanitize import strip_control_chars
from llamatune.types import ProgressEvent, Reporter

_verbose = False


def set_verbose(enabled: bool) -> None:
    """Enable or disable extra stderr diagnostics."""
    global _verbose
    _verbose = enabled


def is_verbose() -> bool:
    """Return True when extra stderr diagnostics are enabled."""
    return _verbose


def emit_diagnostic(message: str, *, err: TextIO | None = None) -> None:
    """Write one diagnostic line to stderr in verbose mode."""
    if not _verbose:
        return
    stream = sys.stderr if err is None else err
    stream.write(f"[verbose] {message}\n")
    stream.flush()


def make_reporter(mode: str, *, err: TextIO | None = None) -> Reporter | None:
    """Construct a progress reporter; ``auto`` chooses Rich only on a TTY."""
    if err is None:
        err = sys.stderr
    if mode == "none":
        return None
    if mode == "auto":
        mode = "rich" if err.isatty() else "plain"
    if mode == "json":
        return JsonReporter(err)
    if mode == "rich":
        try:
            return RichReporter(err)
        except Exception:
            err.write("warning: Rich progress unavailable; using plain progress\n")
            err.flush()
            return PlainReporter(err)
    return PlainReporter(err)


@dataclass(slots=True)
class JsonReporter:
    err: TextIO

    def emit(self, event: ProgressEvent) -> None:
        self.err.write(json.dumps({"kind": event.kind, "ts": event.ts, **event.payload}) + "\n")
        self.err.flush()


@dataclass(slots=True)
class PlainReporter:
    err: TextIO
    _heartbeat_at: dict[str, float] = field(default_factory=dict)

    def emit(self, event: ProgressEvent) -> None:
        p = event.payload
        line: str | None = None
        if event.kind == "session_start":
            line = (
                f"[session] {p['model_name']} — {p['session_dir']} "
                f"({p.get('stop_hint', 'Ctrl-C twice to abort')})"
            )
        elif event.kind == "stage":
            line = f"[stage] {p.get('stage', 'unknown')}"
        elif event.kind == "exec_start":
            line = f"[{p.get('kind', 'run')}] {p.get('label', '')} …"
        elif event.kind == "exec_heartbeat":
            label = str(p.get("label", "run"))
            elapsed = float(p.get("elapsed_s", 0.0))
            previous = self._heartbeat_at.get(label, -30.0)
            if elapsed - previous >= 30.0 or previous < 0:
                self._heartbeat_at[label] = elapsed
                line = (
                    f"[running] {label} — {elapsed:.0f}s elapsed "
                    f"(timeout {float(p.get('timeout_s', 0.0)):.0f}s)"
                )
        elif event.kind == "exec_end":
            metrics = ""
            if p.get("pp") is not None:
                metrics = f" pp={float(p['pp']):.1f} tg={float(p['tg']):.1f}"
            line = (
                f"[result] {p.get('label', '')} — {p.get('status', 'unknown')}"
                f"{metrics} ({float(p.get('wall_s', 0.0)):.1f}s)"
            )
        elif event.kind == "incumbent":
            line = f"[incumbent] score={float(p.get('score', 0.0)):.3f}x"
        elif event.kind == "depth_profile":
            line = f"[depth profile] d={p.get('depth')}"
        elif event.kind == "envelope":
            line = f"[envelope] ctx={p.get('ctx')} — {p.get('status', 'running')}"
        elif event.kind == "warning":
            line = f"[warning] {p.get('message', '')}"
        elif event.kind == "session_end":
            line = f"[done] exit={p.get('exit_code')} winner={p.get('winner_trial_id') or 'none'}"
        if line is not None:
            self.err.write(strip_control_chars(line) + "\n")
            self.err.flush()


class RichReporter:
    """Compact live dashboard; state is updated only by progress events."""

    def __init__(self, err: TextIO) -> None:
        from rich.console import Console
        from rich.live import Live

        self._console = Console(file=err)
        self._lines: list[str] = ["llamatune starting…"]
        self._live = Live("\n".join(self._lines), console=self._console, refresh_per_second=4)
        self._live.start()

    def emit(self, event: ProgressEvent) -> None:
        p = event.payload
        if event.kind == "session_start":
            self._lines = [
                f"[bold]{strip_control_chars(str(p['model_name']))}[/bold]",
                strip_control_chars(str(p["session_dir"])),
                strip_control_chars(str(p.get("stop_hint", "Ctrl-C twice to abort"))),
            ]
        elif event.kind == "stage":
            self._replace("phase:", f"phase: {strip_control_chars(str(p.get('stage', 'unknown')))}")
        elif event.kind in ("exec_start", "exec_heartbeat"):
            elapsed = float(p.get("elapsed_s", 0.0))
            self._replace(
                "activity:",
                f"activity: {strip_control_chars(str(p.get('label', '')))} ({elapsed:.0f}s)",
            )
        elif event.kind == "exec_end":
            self._replace(
                "result:",
                "result: "
                f"{strip_control_chars(str(p.get('status')))} — "
                f"{strip_control_chars(str(p.get('label', '')))}",
            )
        elif event.kind == "incumbent":
            self._replace("incumbent:", f"incumbent: {float(p.get('score', 0.0)):.3f}x")
        elif event.kind == "warning":
            self._lines.append(
                f"[yellow]warning: {strip_control_chars(str(p.get('message', '')))}[/yellow]"
            )
            self._lines = self._lines[-12:]
        elif event.kind == "session_end":
            self._lines.append(f"done: exit {p.get('exit_code')}")
            self._live.update("\n".join(self._lines), refresh=True)
            self._live.stop()
            return
        self._live.update("\n".join(self._lines), refresh=True)

    def _replace(self, prefix: str, value: str) -> None:
        for index, line in enumerate(self._lines):
            if line.startswith(prefix):
                self._lines[index] = value
                return
        self._lines.append(value)
