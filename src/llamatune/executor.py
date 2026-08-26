"""Bounded subprocess execution contract (DESIGN §7).

Executes argv lists with an allowlisted environment, bounded output capture,
and process-group timeout handling. This module never interprets benchmark
semantics (that belongs to ``bench.py``); it only runs a command and reports
exit status, wall time, and hashed/sized/truncated output artifacts.

Public supervision surface for long-lived children: :func:`spawn_supervised`,
:func:`terminate_group`, :func:`close_process_control`,
:func:`popen_platform_kwargs`, and their opaque :class:`ProcessControl` handle.
These are the supported way to run and stop a supervised child (such as
``llama-server``) that outlives a single :func:`run` call; the identical
SIGTERM/SIGKILL and Job Object semantics apply on every path.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Protocol, cast


class _ResourceApi(Protocol):
    RLIMIT_CORE: int

    def setrlimit(self, resource: int, limits: tuple[int, int]) -> None: ...


_resource: _ResourceApi | None
try:
    import resource as _resource_module
except ImportError:  # pragma: no cover - exercised on Windows in stage 4
    _resource = None
else:
    _resource = cast(_ResourceApi, _resource_module)

#: Output capture caps (DESIGN §7): 8 MiB stdout, 1 MiB stderr.
STDOUT_CAP_BYTES = 8 * 1024 * 1024
STDERR_CAP_BYTES = 1 * 1024 * 1024

#: Seconds between SIGTERM to the process group and the SIGKILL escalation.
_TERM_GRACE_S = 10.0
_HEARTBEAT_S = 2.0

_READ_CHUNK = 65536
_WINDOWS = os.name == "nt"
_CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

#: Default in-memory capture cap for pre-session capability probes.
PROBE_OUTPUT_CAP_BYTES = 256 * 1024

#: Exact-name environment allowlist (DESIGN §7).
_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        # Required by Windows process startup and executable resolution.
        "SYSTEMROOT",
        "COMSPEC",
        "PATHEXT",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
    }
)

#: Prefix allowlist; ``LLAMATUNE_FAKE_`` exists solely for the test fixture.
_ENV_PREFIXES: tuple[str, ...] = ("GGML_", "LLAMA_", "LLAMATUNE_FAKE_")


@dataclass(frozen=True, slots=True)
class CaptureInfo:
    """A captured output stream's artifact path, hash, size, and truncation."""

    path: Path
    sha256: str
    size_bytes: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class ExecResult:
    """The outcome of one bounded subprocess execution (DESIGN §7)."""

    exit_code: int | None
    wall_s: float
    timed_out: bool
    stdout: CaptureInfo
    stderr: CaptureInfo
    started: str
    ended: str
    env_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """The outcome of a bounded, in-memory external probe."""

    exit_code: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool


@dataclass(slots=True)
class ProcessControl:
    """Platform process-tree state retained for one child lifetime.

    Opaque handle returned by :func:`spawn_supervised`; pass it back to
    :func:`terminate_group` and :func:`close_process_control`. Callers never
    inspect its fields.
    """

    windows: bool
    job_handle: object | None = None


class _WindowsApi(Protocol):
    def create_kill_on_close_job(self) -> object | None: ...

    def assign(self, job: object, pid: int) -> bool: ...

    def terminate(self, job: object) -> bool: ...

    def close(self, handle: object) -> None: ...


class _CtypesWindowsApi:
    """Small Win32 Job Object adapter, loaded only on Windows."""

    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100

    def __init__(self) -> None:
        from ctypes import wintypes

        class _BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class _ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimitInformation),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self._info_type = _ExtendedLimitInformation
        win_dll = getattr(ctypes, "WinDLL", None)
        if win_dll is None:  # pragma: no cover - guarded by the Windows platform seam
            raise OSError("WinDLL is unavailable")
        self._kernel32 = win_dll("kernel32", use_last_error=True)
        self._kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        self._kernel32.OpenProcess.restype = wintypes.HANDLE

    def create_kill_on_close_job(self) -> object | None:
        job = self._kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = self._info_type()
        info.BasicLimitInformation.LimitFlags = self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = self._kernel32.SetInformationJobObject(
            job,
            self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            self.close(job)
            return None
        return cast(object, job)

    def assign(self, job: object, pid: int) -> bool:
        process = self._kernel32.OpenProcess(
            self._PROCESS_TERMINATE | self._PROCESS_SET_QUOTA, False, pid
        )
        if not process:
            return False
        try:
            return bool(self._kernel32.AssignProcessToJobObject(job, process))
        finally:
            self.close(process)

    def terminate(self, job: object) -> bool:
        return bool(self._kernel32.TerminateJobObject(job, 1))

    def close(self, handle: object) -> None:
        self._kernel32.CloseHandle(handle)


def _windows_api() -> _WindowsApi:
    return _CtypesWindowsApi()


def _create_windows_job(pid: int) -> object | None:
    try:
        api = _windows_api()
        job = api.create_kill_on_close_job()
        if job is None:
            return None
        if api.assign(job, pid):
            return job
        api.close(job)
    except (AttributeError, OSError):
        return None
    return None


def popen_platform_kwargs(preexec_fn: Callable[[], None] | None = None) -> dict[str, Any]:
    """Keyword arguments that give a child its own process group on this platform."""
    if _WINDOWS:
        return {"creationflags": _CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True, "preexec_fn": preexec_fn}


def spawn_supervised(
    argv: Sequence[str], **kwargs: Any
) -> tuple[subprocess.Popen[bytes], ProcessControl]:
    """Start ``argv`` and return its process plus platform :class:`ProcessControl`.

    Part of the public supervision surface used by long-lived children (such as
    ``llama-server``) that outlive a single :func:`run` call. Semantics are the
    DESIGN §7 spawn contract: argv list, no shell, caller-provided bounded
    streams and allowlisted environment.
    """
    proc = subprocess.Popen(  # noqa: S603
        list(argv),
        **kwargs,
    )
    control = ProcessControl(windows=_WINDOWS)
    if _WINDOWS:
        control.job_handle = _create_windows_job(proc.pid)
    return proc, control


def close_process_control(control: ProcessControl) -> None:
    """Release any platform job handle held by ``control`` (idempotent)."""
    if control.job_handle is None:
        return
    with contextlib.suppress(AttributeError, OSError):
        _windows_api().close(control.job_handle)
    control.job_handle = None


def build_child_env(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """A fresh allowlisted copy of the environment (names/values, DESIGN §7)."""
    src: Mapping[str, str] = os.environ if source is None else source
    return {
        key: value
        for key, value in src.items()
        if key in _ENV_ALLOWLIST or key.startswith(_ENV_PREFIXES)
    }


def _drain(stream: BinaryIO, path: Path, cap: int) -> CaptureInfo:
    """Stream ``stream`` to ``path``, writing at most ``cap`` bytes."""
    hasher = hashlib.sha256()
    written = 0
    truncated = False
    with path.open("wb") as handle:
        while True:
            chunk = stream.read(_READ_CHUNK)
            if not chunk:
                break
            if written < cap:
                room = cap - written
                take = chunk[:room]
                handle.write(take)
                hasher.update(take)
                written += len(take)
                if len(chunk) > room:
                    truncated = True
            else:
                truncated = True
    return CaptureInfo(
        path=path, sha256=hasher.hexdigest(), size_bytes=written, truncated=truncated
    )


def _drain_memory(stream: BinaryIO, cap: int) -> bytes:
    """Drain ``stream`` completely while retaining at most ``cap`` bytes."""
    captured = bytearray()
    while True:
        chunk = stream.read(_READ_CHUNK)
        if not chunk:
            break
        if len(captured) < cap:
            captured.extend(chunk[: cap - len(captured)])
    return bytes(captured)


def _terminate_posix_group(proc: subprocess.Popen[bytes]) -> None:
    """SIGTERM the child's POSIX process group, grace, then SIGKILL."""
    getpgid = cast(Callable[[int], int] | None, getattr(os, "getpgid", None))
    killpg = cast(Callable[[int, int], None] | None, getattr(os, "killpg", None))
    if getpgid is None or killpg is None:  # pragma: no cover - POSIX seam
        return
    try:
        pgid = getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=_TERM_GRACE_S)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            killpg(pgid, getattr(signal, "SIGKILL", 9))


def _taskkill_tree(pid: int) -> None:
    """Best-effort Windows process-tree fallback when Job Objects are unavailable.

    Like every other executor child this cleanup helper runs with a finite
    runtime and an allowlisted environment. It reuses the SIGTERM/SIGKILL
    termination grace as its wall bound. Startup errors and a ``TimeoutExpired``
    (a ``SubprocessError``) are suppressed: best-effort cleanup must never
    replace the original benchmark outcome.
    """
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run(  # noqa: S603
            ["taskkill", "/PID", str(pid), "/T", "/F"],  # noqa: S607
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=_TERM_GRACE_S,
            env=build_child_env(),
            creationflags=_CREATE_NO_WINDOW,
        )


def terminate_group(proc: subprocess.Popen[bytes], control: ProcessControl | None = None) -> None:
    """Terminate the child's platform process tree (DESIGN §7)."""
    if control is None or not control.windows:
        _terminate_posix_group(proc)
        return
    terminated = False
    if control.job_handle is not None:
        try:
            terminated = _windows_api().terminate(control.job_handle)
        except (AttributeError, OSError):
            terminated = False
        close_process_control(control)
    if not terminated:
        _taskkill_tree(proc.pid)


def core_dump_control_status(disable_core_dumps: bool) -> str:
    """Describe whether the POSIX core-dump limit was applied to a child."""
    if not disable_core_dumps:
        return "inherited"
    return "disabled" if _resource is not None and not _WINDOWS else "unsupported"


def run_probe(
    argv: Sequence[str],
    *,
    timeout_s: float = 10.0,
    max_output_bytes: int = PROBE_OUTPUT_CAP_BYTES,
) -> ProbeResult | None:
    """Run a pre-session probe under the executor contract.

    Output is retained in memory up to ``max_output_bytes`` per stream. The
    process is still drained completely so excess output cannot block it.
    Returns ``None`` only when the process cannot be started.
    """
    try:
        proc, control = spawn_supervised(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=build_child_env(),
            **popen_platform_kwargs(),
        )
    except OSError:
        return None

    stdout_stream = proc.stdout
    stderr_stream = proc.stderr
    if stdout_stream is None or stderr_stream is None:  # pragma: no cover - PIPE guarantees streams
        msg = "subprocess pipes were not created"
        raise RuntimeError(msg)

    captures: dict[str, bytes] = {}
    capture_errors: dict[str, BaseException] = {}
    capture_lock = threading.Lock()

    def _worker(name: str, stream: BinaryIO) -> None:
        try:
            captured = _drain_memory(stream, max_output_bytes)
        except BaseException as exc:
            with capture_lock:
                capture_errors[name] = exc
        else:
            with capture_lock:
                captures[name] = captured
        finally:
            with contextlib.suppress(OSError):
                stream.close()

    out_thread = threading.Thread(target=_worker, args=("stdout", stdout_stream))
    err_thread = threading.Thread(target=_worker, args=("stderr", stderr_stream))
    out_thread.start()
    err_thread.start()

    timed_out = False
    try:
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_group(proc, control)
            proc.wait()
    except BaseException:
        with contextlib.suppress(Exception):
            terminate_group(proc, control)
            proc.wait()
        out_thread.join()
        err_thread.join()
        raise
    else:
        out_thread.join()
        err_thread.join()
    finally:
        close_process_control(control)
    if capture_errors:
        name = sorted(capture_errors)[0]
        error = capture_errors[name]
        raise RuntimeError(f"failed to capture child {name}: {error}") from error
    missing = {"stdout", "stderr"} - captures.keys()
    if missing:
        name = sorted(missing)[0]
        raise RuntimeError(f"child {name} capture did not complete")
    return ProbeResult(
        exit_code=proc.poll(),
        stdout=captures["stdout"],
        stderr=captures["stderr"],
        timed_out=timed_out,
    )


def run(
    argv: Sequence[str],
    *,
    timeout_s: float,
    stdout_path: Path,
    stderr_path: Path,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    disable_core_dumps: bool = True,
    on_heartbeat: Callable[[float], None] | None = None,
) -> ExecResult:
    """Run ``argv`` under the DESIGN §7 execution contract.

    argv-only (never ``shell=True``), an allowlisted child environment, a new
    session/process-group, bounded output capture to the given paths, and
    SIGTERM/SIGKILL group termination on timeout.

    ``env`` is a *source* mapping, not a trusted final environment: its allowed
    names are selected through :func:`build_child_env` exactly like the process
    environment, so passing ``env`` can never bypass the §7 allowlist.
    """
    child_env = build_child_env(env)
    env_names = tuple(sorted(child_env))

    started_dt = datetime.now(UTC)
    start = time.monotonic()
    preexec_fn = None
    if disable_core_dumps and _resource is not None:

        def _disable_core_dumps() -> None:
            # The engine spawns benchmark children sequentially, so this
            # pre-exec hook never forks from a concurrently mutating thread.
            _resource.setrlimit(_resource.RLIMIT_CORE, (0, 0))

        preexec_fn = _disable_core_dumps

    proc, control = spawn_supervised(
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_env,
        cwd=str(cwd) if cwd is not None else None,
        **popen_platform_kwargs(preexec_fn),
    )
    stdout_stream = proc.stdout
    stderr_stream = proc.stderr
    if stdout_stream is None or stderr_stream is None:  # pragma: no cover - PIPE guarantees streams
        msg = "subprocess pipes were not created"
        raise RuntimeError(msg)

    captures: dict[str, CaptureInfo] = {}
    capture_errors: dict[str, BaseException] = {}
    capture_lock = threading.Lock()

    def _worker(name: str, stream: BinaryIO, path: Path, cap: int) -> None:
        try:
            captured = _drain(stream, path, cap)
        except BaseException as exc:
            with capture_lock:
                capture_errors[name] = exc
        else:
            with capture_lock:
                captures[name] = captured
        finally:
            with contextlib.suppress(OSError):
                stream.close()

    out_thread = threading.Thread(
        target=_worker, args=("stdout", stdout_stream, Path(stdout_path), STDOUT_CAP_BYTES)
    )
    err_thread = threading.Thread(
        target=_worker, args=("stderr", stderr_stream, Path(stderr_path), STDERR_CAP_BYTES)
    )
    out_thread.start()
    err_thread.start()

    timed_out = False
    deadline = start + timeout_s
    try:
        while proc.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                terminate_group(proc, control)
                proc.wait()  # reap; platform termination guarantees completion
                break
            try:
                proc.wait(timeout=min(_HEARTBEAT_S, remaining))
            except subprocess.TimeoutExpired:
                if on_heartbeat is not None:
                    # Liveness reporting must never alter child execution.
                    with contextlib.suppress(Exception):
                        on_heartbeat(time.monotonic() - start)
    except BaseException:
        with contextlib.suppress(Exception):
            terminate_group(proc, control)
            proc.wait()
        out_thread.join()
        err_thread.join()
        raise
    else:
        out_thread.join()
        err_thread.join()
    finally:
        close_process_control(control)
    exit_code = proc.poll()
    wall_s = time.monotonic() - start
    ended_dt = datetime.now(UTC)

    if capture_errors:
        name = sorted(capture_errors)[0]
        error = capture_errors[name]
        destination = stdout_path if name == "stdout" else stderr_path
        raise RuntimeError(
            f"failed to capture child {name} output to {destination}: {error}"
        ) from error
    missing = {"stdout", "stderr"} - captures.keys()
    if missing:
        name = sorted(missing)[0]
        destination = stdout_path if name == "stdout" else stderr_path
        raise RuntimeError(f"child {name} capture to {destination} did not complete")

    return ExecResult(
        exit_code=exit_code,
        wall_s=wall_s,
        timed_out=timed_out,
        stdout=captures["stdout"],
        stderr=captures["stderr"],
        started=started_dt.isoformat(),
        ended=ended_dt.isoformat(),
        env_names=env_names,
    )
