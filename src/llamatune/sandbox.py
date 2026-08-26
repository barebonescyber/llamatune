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
from enum import StrEnum
from pathlib import Path
from typing import Any, BinaryIO

try:
    _resource: Any = importlib.import_module("resource")
except ImportError:  # pragma: no cover - exercised by platform simulation
    _resource = None

_OUTPUT_LIMIT = 64 * 1024
_WALL_LIMIT_S = 30.0
_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
_RUN_GUARD = threading.Lock()
_ACTIVE_LOCK = threading.Lock()
_ACTIVE_PROCESS: subprocess.Popen[bytes] | None = None

_ISOLATION_ACTIVE = "network-namespace"
_ISOLATION_ALLOWED = "none (allowed by flag)"
_ISOLATION_UNAVAILABLE = "none (unavailable)"
_FS_BWRAP = "bubblewrap"
_FS_NONE = "none"
_BWRAP_PYTHON_MOUNT = "/llamatune-py"


class IsolationStatus(StrEnum):
    """Probe result for unprivileged network-namespace isolation."""

    AVAILABLE = "available"
    UNAVAILABLE_TOOL = "unavailable-tool"
    UNAVAILABLE_PERMISSION = "unavailable-kernel-permission"
    UNSUPPORTED_PLATFORM = "unsupported-platform"


class SandboxIsolationError(RuntimeError):
    """Raised before spawn when network isolation is unconfirmed and not allowed."""


@dataclass(frozen=True, slots=True)
class ExecVerdict:
    """Bounded evidence from one sandboxed Python execution."""

    passed: bool
    exit_code: int | None
    timed_out: bool
    stdout_tail: str
    stderr_tail: str


@dataclass(frozen=True, slots=True)
class LimitPlan:
    """Planned rlimits per platform; memory limits are best-effort."""

    required: tuple[tuple[int, int], ...]
    memory: tuple[tuple[int, int], ...]
    memory_label: str


@dataclass(frozen=True, slots=True)
class IsolationReport:
    """Surfaced isolation posture for one --exec enabled quality run."""

    network_status: IsolationStatus
    network_active: bool
    filesystem_confinement: str
    memory_limit: str
    summary: str
    warnings: tuple[str, ...]


def limit_plan(platform: str) -> LimitPlan:
    """Return the planned rlimits for the given platform (pure selection)."""
    if _resource is None:
        raise RuntimeError("Python execution sandbox requires POSIX resource limits")
    required = (
        (_resource.RLIMIT_CPU, 10),
        (_resource.RLIMIT_FSIZE, 1024 * 1024),
        (_resource.RLIMIT_NOFILE, 32),
        (_resource.RLIMIT_CORE, 0),
    )
    if platform == "darwin":
        # Darwin exposes RLIMIT_AS but rejects lowering it in a pre-exec child
        # on some Python/macOS combinations.  Attempt RLIMIT_DATA (and RLIMIT_RSS
        # where the platform honors it) as best-effort heap bounds instead.
        memory = tuple(
            (limit_id, _MEMORY_LIMIT_BYTES)
            for limit_id in (
                getattr(_resource, "RLIMIT_DATA", None),
                getattr(_resource, "RLIMIT_RSS", None),
            )
            if limit_id is not None
        )
        label = "rlimit-data" if memory else "rlimit-none"
        return LimitPlan(required=required, memory=memory, memory_label=label)
    return LimitPlan(
        required=required,
        memory=((_resource.RLIMIT_AS, _MEMORY_LIMIT_BYTES),),
        memory_label="rlimit-as",
    )


def _apply_limits() -> None:
    if _resource is None:
        raise RuntimeError("Python execution sandbox requires POSIX resource limits")
    plan = limit_plan(sys.platform)
    for resource_id, value in plan.required:
        _resource.setrlimit(resource_id, (value, value))
    for resource_id, value in plan.memory:
        try:
            _resource.setrlimit(resource_id, (value, value))
        except OSError:
            # Some platforms reject individual memory limits in a pre-exec child.
            continue


def _environment() -> dict[str, str]:
    return {"PATH": str(Path(sys.executable).resolve().parent)}


@functools.lru_cache(maxsize=1)
def _network_probe() -> tuple[IsolationStatus, tuple[str, ...]]:
    """Verify that ``unshare -rn`` actually works; never assume it does."""
    if not sys.platform.startswith("linux"):
        return (IsolationStatus.UNSUPPORTED_PLATFORM, ())
    executable = shutil.which("unshare", path=os.defpath)
    if executable is None:
        return (IsolationStatus.UNAVAILABLE_TOOL, ())
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
        return (IsolationStatus.UNAVAILABLE_PERMISSION, ())
    status = (
        IsolationStatus.AVAILABLE
        if probe.returncode == 0
        else IsolationStatus.UNAVAILABLE_PERMISSION
    )
    wrapper = (executable, "-rn") if status is IsolationStatus.AVAILABLE else ()
    return (status, wrapper)


def detect_network_isolation() -> IsolationStatus:
    """Return the probed network-namespace isolation status."""
    return _network_probe()[0]


def _network_wrapper() -> tuple[str, ...]:
    return _network_probe()[1]


def _interpreter_visible_in_confinement() -> bool:
    resolved = str(Path(sys.executable).resolve())
    return any(
        resolved == prefix or resolved.startswith(prefix.rstrip("/") + "/")
        for prefix in ("/usr", "/bin", "/sbin", "/lib", "/lib64")
    )


@functools.lru_cache(maxsize=1)
def _bwrap_prefix_probe() -> tuple[tuple[str, ...], str] | None:
    """Probe bubblewrap with the exact confinement argv shape we would run."""
    if not sys.platform.startswith("linux"):
        return None
    executable = shutil.which("bwrap")
    if executable is None:
        return None
    visible = _interpreter_visible_in_confinement()
    child_python = (
        sys.executable if visible else f"{_BWRAP_PYTHON_MOUNT}/{Path(sys.executable).name}"
    )
    alt_bind = (
        ()
        if visible
        else ("--ro-bind", str(Path(sys.executable).resolve().parent), _BWRAP_PYTHON_MOUNT)
    )
    static = (
        executable,
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind-try",
        "/lib",
        "/lib",
        "--ro-bind-try",
        "/lib64",
        "/lib64",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",  # noqa: S108 - bubblewrap mount target, not a host temp file
        "--tmpfs",
        "/home",
        "--tmpfs",
        "/run",
        *alt_bind,
    )
    with tempfile.TemporaryDirectory(prefix="llamatune-bwrap-probe-") as scratch:
        argv = (
            *static,
            "--bind",
            scratch,
            scratch,
            "--die-with-parent",
            child_python,
            "-I",
            "-S",
            "-B",
            "-c",
            "pass",
        )
        try:
            probe = subprocess.run(  # noqa: S603 - fixed trusted executable and argv
                argv,
                cwd=scratch,
                env=_environment(),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=5.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
    return (static, child_python) if probe.returncode == 0 else None


def detect_filesystem_confinement() -> str:
    """Return ``bubblewrap`` when bubblewrap probes usable, else ``none``."""
    return _FS_BWRAP if _bwrap_prefix_probe() is not None else _FS_NONE


_ALLOW_DEGRADED_NETWORK = False


def set_allow_network_fallback(allowed: bool) -> None:
    """Record whether the operator accepted runs without network isolation."""
    global _ALLOW_DEGRADED_NETWORK
    _ALLOW_DEGRADED_NETWORK = allowed


def allow_network_fallback() -> bool:
    """Return whether degraded (unisolated) network execution was accepted."""
    return _ALLOW_DEGRADED_NETWORK


def describe_isolation(*, allow_network: bool | None = None) -> IsolationReport:
    """Summarize the isolation posture for an --exec run, with degradation warnings."""
    allowed = allow_network_fallback() if allow_network is None else allow_network
    status, wrapper = _network_probe()
    active = bool(wrapper)
    if active:
        summary = _ISOLATION_ACTIVE
    elif allowed:
        summary = _ISOLATION_ALLOWED
    else:
        summary = _ISOLATION_UNAVAILABLE
    warnings: list[str] = []
    if not active:
        warnings.append(
            f"network namespace isolation unavailable ({status.value}); "
            "model-generated code runs with network access"
        )
    filesystem = detect_filesystem_confinement()
    if filesystem == _FS_NONE:
        reason = (
            "unsupported on this platform"
            if not sys.platform.startswith("linux")
            else "bubblewrap missing or unusable"
        )
        warnings.append(f"filesystem confinement inactive ({reason})")
    memory = limit_plan(sys.platform).memory_label
    if memory == "rlimit-none":
        warnings.append("no memory rlimit could be applied")
    return IsolationReport(
        network_status=status,
        network_active=active,
        filesystem_confinement=filesystem,
        memory_limit=memory,
        summary=summary,
        warnings=tuple(warnings),
    )


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
        status = detect_network_isolation()
        notes.append(f"network namespace isolation unavailable ({status.value})")
    elif not sys.platform.startswith("linux"):
        notes.append("network namespace isolation unsupported on this platform")
    if sys.platform == "darwin":
        plan = limit_plan(sys.platform)
        memory = (
            "RLIMIT_DATA fallback attempted"
            if plan.memory_label != "rlimit-none"
            else "no memory rlimit applied"
        )
        notes.append(f"RLIMIT_AS unsupported on this platform; {memory}")
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


def run_python(code: str, *, timeout_s: float, allow_network: bool | None = None) -> ExecVerdict:
    """Execute Python with bounded POSIX resources and process-group cleanup.

    Fails closed before spawning anything unless network-namespace isolation is
    confirmed active, or the operator explicitly accepted degraded isolation.
    """
    if os.name != "posix" or _resource is None:
        raise RuntimeError("Python execution sandbox requires POSIX resource limits")
    if not _valid_timeout(timeout_s):
        raise ValueError("timeout_s must be finite and positive")
    allowed = allow_network_fallback() if allow_network is None else allow_network
    status, wrapper = _network_probe()
    if status is not IsolationStatus.AVAILABLE and not allowed:
        raise SandboxIsolationError(
            f"network namespace isolation unavailable ({status.value}); "
            "pass --exec-allow-network to accept reduced isolation"
        )
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
        confinement = _bwrap_prefix_probe()
        if confinement is not None:
            static, child_python = confinement
            head = (*static, "--bind", str(directory), str(directory), "--die-with-parent")
        else:
            head = ()
            child_python = sys.executable
        argv = (*head, *wrapper, child_python, "-I", "-S", "-B", "main.py")
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
