"""Shared evidence IO foundation for llamatune orchestrators (DESIGN §12, §13).

One home for the primitives every evidence writer and reader needs:
deterministic JSON-ification, UTC timestamps, path confinement, deadline
resolution, one tolerant JSONL journal reader, confined run-writer behavior,
unique-directory allocation, and the two-stage interrupt protocol. Handlers
installed here never perform file IO; they only record events.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import secrets
import signal
import threading
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any, cast

CREATE_RETRIES = 5


class PathEscapeError(Exception):
    """Raised when a resolved evidence path would escape its root directory."""


def jsonable(value: Any) -> Any:
    """Return a JSON-safe structure; set and frozenset members are sorted."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [jsonable(item) for item in sorted(value)]
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def utc_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def resolve_deadline(
    start: datetime,
    until: str | None,
    max_hours: float | None,
    *,
    local_tz: tzinfo | None = None,
) -> datetime | None:
    """Resolve the earliest supplied deadline; past wall times mean tomorrow."""
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


def confined_path(root: Path, *parts: str, error: type[PathEscapeError] = PathEscapeError) -> Path:
    """Join ``parts`` onto ``root``, rejecting any result that escapes ``root``.

    The returned path is the joined candidate (not fully resolved), matching
    historical writer behavior; the containment check itself always resolves
    symlinks and ``..`` components before comparing against ``root``.
    """
    return confine_against(root.resolve(), root, *parts, error=error)


def confine_against(
    root_resolved: Path,
    root: Path,
    *parts: str,
    error: type[PathEscapeError] = PathEscapeError,
) -> Path:
    """Confine against an already-resolved ``root_resolved`` (hot-path variant).

    Callers that confine repeatedly against one unchanged directory can
    resolve the root once and pass it here, avoiding repeated ``resolve()``
    syscalls per journal append (#26 PERF-015).
    """
    candidate = root.joinpath(*parts)
    try:
        candidate.resolve().relative_to(root_resolved)
    except ValueError:
        msg = f"path {candidate} escapes evidence directory {root}"
        raise error(msg) from None
    return candidate


def read_journal_lines(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse a JSONL journal, skipping corrupt lines and reporting each one.

    One policy for every journal consumer: any unparseable line (including a
    torn final line) is skipped and named in the returned warnings; valid
    entries before and after it still load. Blank lines are ignored. A
    missing file yields no entries and no warnings. OSError still propagates.
    """
    entries: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not path.is_file():
        return entries, warnings
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            warnings.append(f"{path.name}: line {index + 1} is corrupt and was skipped")
            continue
        if not isinstance(value, dict):
            warnings.append(f"{path.name}: line {index + 1} is not a JSON object and was skipped")
            continue
        entries.append(value)
    return entries, warnings


def create_unique_dir(parent: Path, prefix: str, *, error: type[Exception]) -> Path:
    """Create ``<prefix>-<UTC stamp>-<6hex>`` under ``parent`` with retries."""
    parent.mkdir(parents=True, exist_ok=True)
    last_error: FileExistsError | None = None
    for _ in range(CREATE_RETRIES):
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        token = secrets.token_hex(3)
        name = "-".join(part for part in (prefix, stamp, token) if part)
        candidate = parent / name
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError as exc:
            last_error = exc
            continue
        return candidate
    msg = f"could not allocate a unique directory under {parent}"
    raise error(msg) from last_error


class EvidenceWriter:
    """Confined JSON/text/journal writing for one evidence directory.

    Subclasses assign ``dir`` in their initializer and may narrow
    ``path_error`` to their public confinement error type so existing
    exception mapping keeps working.
    """

    dir: Path
    path_error: type[PathEscapeError] = PathEscapeError
    _resolved_root: Path | None = None

    def _confined(self, *parts: str) -> Path:
        if self._resolved_root is None:
            self._resolved_root = self.dir.resolve()
        return confine_against(self._resolved_root, self.dir, *parts, error=self.path_error)

    def _journal_record(self, entry: dict[str, Any]) -> dict[str, Any]:
        record = cast(dict[str, Any], jsonable(dict(entry)))
        record.setdefault("ts", utc_iso())
        return record

    def _append_record(self, record: dict[str, Any]) -> None:
        path = self._confined("journal.jsonl")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def append(self, entry: dict[str, Any]) -> None:
        """Append one journal entry; adds a timestamp, flushes, and fsyncs."""
        self._append_record(self._journal_record(entry))

    def write_json(self, name: str, payload: dict[str, Any]) -> None:
        path = self._confined(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def write_text(self, name: str, text: str) -> None:
        path = self._confined(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


@dataclasses.dataclass(slots=True)
class InterruptState:
    """Signals observed by :func:`install_interrupt_handlers`."""

    count: int = 0
    stop_requested: bool = False
    second_signal: bool = False
    events: list[tuple[int, bool]] = dataclasses.field(default_factory=list)

    def drain_events(self) -> list[tuple[int, bool]]:
        """Return and clear recorded ``(signum, immediate)`` events."""
        drained, self.events = self.events, []
        return drained


@contextlib.contextmanager
def install_interrupt_handlers(
    signals: Sequence[signal.Signals] = (signal.SIGINT, signal.SIGTERM),
) -> Iterator[InterruptState]:
    """Install the two-stage interrupt protocol for the context duration.

    First delivery requests a graceful stop and records an event; second
    delivery records an immediate event and raises ``KeyboardInterrupt``
    so the process exits promptly. Handlers only mutate state; journaling
    belongs to the main flow via :meth:`InterruptState.drain_events`.
    Handlers are restored on exit. Outside the main thread nothing is
    installed and the returned state simply stays idle.
    """
    state = InterruptState()

    def handle(signum: int, frame: Any) -> None:
        del frame
        state.count += 1
        if state.stop_requested:
            state.second_signal = True
            state.events.append((signum, True))
            raise KeyboardInterrupt
        state.stop_requested = True
        state.events.append((signum, False))

    previous: dict[signal.Signals, Any] = {}
    if threading.current_thread() is threading.main_thread():
        for sig in signals:
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, handle)
    try:
        yield state
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


__all__ = [
    "CREATE_RETRIES",
    "EvidenceWriter",
    "InterruptState",
    "PathEscapeError",
    "confined_path",
    "create_unique_dir",
    "install_interrupt_handlers",
    "jsonable",
    "read_journal_lines",
    "resolve_deadline",
    "utc_iso",
]
