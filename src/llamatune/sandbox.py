"""Opt-in, POSIX-only accident barrier for generated Python code."""

from __future__ import annotations

import functools
import importlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

try:
    _resource: Any = importlib.import_module("resource")
except ImportError:  # pragma: no cover - exercised by platform simulation
    _resource = None

_OUTPUT_LIMIT = 64 * 1024
_WALL_LIMIT_S = 30.0
_RUN_GUARD = threading.Lock()
_ACTIVE_LOCK = threading.Lock()
_ACTIVE_PROCESS: subprocess.Popen[bytes] | None = None


@dataclass(frozen=True, slots=True)
class ExecVerdict:
    """Bounded evidence from one sandboxed Python execution."""

    passed: bool
    exit_code: int | None
    timed_out: bool
    stdout_tail: str
    stderr_tail: str


def _limits() -> tuple[tuple[int, int], ...]:
    if _resource is None:
        raise RuntimeError("Python execution sandbox requires POSIX resource limits")
    return (
        (_resource.RLIMIT_CPU, 10),
        (_resource.RLIMIT_AS, 512 * 1024 * 1024),
        (_resource.RLIMIT_FSIZE, 1024 * 1024),
        (_resource.RLIMIT_NOFILE, 32),
        (_resource.RLIMIT_CORE, 0),
    )


def _apply_limits() -> None:
    if _resource is None:
        raise RuntimeError("Python execution sandbox requires POSIX resource limits")
    for resource_id, value in _limits():
        # Darwin exposes RLIMIT_AS but rejects lowering it in a pre-exec child
        # on some Python/macOS combinations.  Keep the remaining accident
        # barriers usable and report this platform limitation in the verdict.
        if sys.platform == "darwin" and resource_id == _resource.RLIMIT_AS:
            continue
        _resource.setrlimit(resource_id, (value, value))


def _environment() -> dict[str, str]:
    return {"PATH": str(Path(sys.executable).resolve().parent)}


@functools.lru_cache(maxsize=1)
def _network_wrapper() -> tuple[str, ...]:
    if not sys.platform.startswith("linux"):
        return ()
    executable = shutil.which("unshare", path=os.defpath)
    if executable is None:
        return ()
    try:
        probe = subprocess.run(  # noqa: S603 - fixed trusted executable and argv
            (executable, "-rn", sys.executable, "-I", "-S", "-B", "-c", "pass"),
            cwd=str(Path(sys.executable).resolve().parent),
            env=_environment(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    return (executable, "-rn") if probe.returncode == 0 else ()


class _TailCapture:
    """A bounded output tail with synchronized writer and snapshot access."""

    def __init__(self) -> None:
        self._data = bytearray()
        self._lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        with self._lock:
            self._data.extend(chunk)
            if len(self._data) > _OUTPUT_LIMIT:
                del self._data[: len(self._data) - _OUTPUT_LIMIT]

    def decode(self) -> str:
        with self._lock:
            return bytes(self._data).decode("utf-8", errors="replace")


def _drain(stream: BinaryIO, tail: _TailCapture) -> None:
    try:
        while chunk := stream.read(8192):
            tail.append(chunk)
    finally:
        stream.close()


def _isolation_note(wrapper: tuple[str, ...]) -> str:
    notes: list[str] = []
    if sys.platform.startswith("linux") and not wrapper:
        notes.append("network namespace isolation unavailable")
    elif not sys.platform.startswith("linux"):
        notes.append("network namespace isolation unsupported on this platform")
    if sys.platform == "darwin":
        notes.append("RLIMIT_AS unsupported on this platform")
    return "; ".join(notes)


def _append_note(stderr: str, note: str) -> str:
    if not note:
        return stderr
    combined = (
        f"{stderr.rstrip()}\n[llamatune sandbox: {note}]\n"
        if stderr
        else f"[llamatune sandbox: {note}]\n"
    )
    encoded = combined.encode("utf-8", errors="replace")
    return encoded[-_OUTPUT_LIMIT:].decode("utf-8", errors="replace")


def _publish_active(process: subprocess.Popen[bytes]) -> None:
    global _ACTIVE_PROCESS
    with _ACTIVE_LOCK:
        _ACTIVE_PROCESS = process


def _clear_active(process: subprocess.Popen[bytes]) -> None:
    global _ACTIVE_PROCESS
    with _ACTIVE_LOCK:
        if _ACTIVE_PROCESS is process:
            _ACTIVE_PROCESS = None


def _kill_group(pid: int) -> None:
    killpg = getattr(os, "killpg", None)
    sigkill = getattr(signal, "SIGKILL", None)
    if killpg is None or sigkill is None:
        raise RuntimeError("sandbox process-group termination requires POSIX")
    killpg(pid, sigkill)


def terminate_active() -> bool:
    """Kill and reap the active sandbox group without blocking on its state lock."""
    global _ACTIVE_PROCESS
    if not _ACTIVE_LOCK.acquire(blocking=False):
        return False
    try:
        process = _ACTIVE_PROCESS
        if process is None or process.poll() is not None:
            if process is _ACTIVE_PROCESS:
                _ACTIVE_PROCESS = None
            return False
        with suppress(ProcessLookupError):
            _kill_group(process.pid)
        process.wait()
        if _ACTIVE_PROCESS is process:
            _ACTIVE_PROCESS = None
        return True
    finally:
        _ACTIVE_LOCK.release()


def run_python(code: str, *, timeout_s: float) -> ExecVerdict:
    """Execute Python with bounded POSIX resources and process-group cleanup."""
    if os.name != "posix" or _resource is None:
        raise RuntimeError("Python execution sandbox requires POSIX resource limits")
    if not _valid_timeout(timeout_s):
        raise ValueError("timeout_s must be finite and positive")
    if not _RUN_GUARD.acquire(blocking=False):
        raise RuntimeError("another Python sandbox run is already active")

    directory: Path | None = None
    process: subprocess.Popen[bytes] | None = None
    readers: list[threading.Thread] = []
    stdout_tail = _TailCapture()
    stderr_tail = _TailCapture()
    timed_out = False
    try:
        directory = Path(tempfile.mkdtemp(prefix="llamatune-quality-exec-"))
        (directory / "main.py").write_text(code, encoding="utf-8")
        wrapper = _network_wrapper()
        argv = (*wrapper, sys.executable, "-I", "-S", "-B", "main.py")
        process = subprocess.Popen(  # noqa: S603 - explicit opt-in sandbox boundary
            argv,
            cwd=directory,
            env=_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
            preexec_fn=_apply_limits,
        )
        _publish_active(process)
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("sandbox pipes were not created")
        readers = [
            threading.Thread(target=_drain, args=(process.stdout, stdout_tail), daemon=True),
            threading.Thread(target=_drain, args=(process.stderr, stderr_tail), daemon=True),
        ]
        for reader in readers:
            reader.start()
        try:
            process.wait(timeout=min(timeout_s, _WALL_LIMIT_S))
        except subprocess.TimeoutExpired:
            timed_out = True
            with suppress(ProcessLookupError):
                _kill_group(process.pid)
            process.wait()
        else:
            with suppress(ProcessLookupError):
                _kill_group(process.pid)
        for reader in readers:
            reader.join(timeout=2.0)
        stderr = _append_note(stderr_tail.decode(), _isolation_note(wrapper))
        return ExecVerdict(
            passed=process.returncode == 0 and not timed_out,
            exit_code=process.returncode,
            timed_out=timed_out,
            stdout_tail=stdout_tail.decode(),
            stderr_tail=stderr,
        )
    finally:
        if process is not None and process.poll() is None:
            with suppress(ProcessLookupError):
                _kill_group(process.pid)
            process.wait()
        if process is not None:
            _clear_active(process)
        for reader in readers:
            reader.join(timeout=2.0)
        try:
            if directory is not None:
                with suppress(FileNotFoundError):
                    shutil.rmtree(directory)
        finally:
            _RUN_GUARD.release()


def _valid_timeout(value: float) -> bool:
    """Keep timeout validation dependency-free and explicitly reject NaN."""
    return value > 0.0 and value != float("inf") and value == value
