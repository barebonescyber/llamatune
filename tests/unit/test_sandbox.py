from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, cast

import pytest

from llamatune import sandbox

_POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="POSIX sandbox only")


@pytest.fixture(autouse=True)
def _disable_unshare(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_network_wrapper", lambda: ())
    monkeypatch.setattr(sandbox, "require_exec_isolation", lambda: None)


@_POSIX_ONLY
def test_limits_and_environment_are_exact() -> None:
    assert [value for _, value in sandbox._limits()] == [
        10,
        512 * 1024 * 1024,
        1024 * 1024,
        32,
        0,
    ]
    assert sandbox._environment() == {"PATH": str(Path(sys.executable).resolve().parent)}


@_POSIX_ONLY
def test_run_python_exact_flags_empty_env_cwd_and_rlimits() -> None:
    resource_api = cast(Any, sandbox._resource)
    code = """import json, os, resource, sys
print(json.dumps({
    "argv": sys.argv,
    "isolated": sys.flags.isolated,
    "no_site": sys.flags.no_site,
    "dont_write_bytecode": sys.flags.dont_write_bytecode,
    "env": dict(os.environ),
    "cwd": os.getcwd(),
    "limits": {
        "cpu": resource.getrlimit(resource.RLIMIT_CPU),
        "as": resource.getrlimit(resource.RLIMIT_AS),
        "fsize": resource.getrlimit(resource.RLIMIT_FSIZE),
        "nofile": resource.getrlimit(resource.RLIMIT_NOFILE),
        "core": resource.getrlimit(resource.RLIMIT_CORE),
    },
}, sort_keys=True))
"""
    verdict = sandbox.run_python(code, timeout_s=2.0)
    assert verdict.passed is True
    evidence = json.loads(verdict.stdout_tail)
    assert evidence["argv"] == ["main.py"]
    assert evidence["isolated"] == 1
    assert evidence["no_site"] == 1
    assert evidence["dont_write_bytecode"] == 1
    assert evidence["env"]["PATH"] == sandbox._environment()["PATH"]
    os_injected = {"__CF_USER_TEXT_ENCODING"} if sys.platform == "darwin" else set()
    assert set(evidence["env"]) <= {"PATH", "LC_CTYPE", *os_injected}
    assert Path(evidence["cwd"]).name.startswith("llamatune-quality-exec-")
    expected_as = (
        list(resource_api.getrlimit(resource_api.RLIMIT_AS))
        if sys.platform == "darwin"
        else [512 * 1024 * 1024, 512 * 1024 * 1024]
    )
    assert evidence["limits"] == {
        "as": expected_as,
        "core": [0, 0],
        "cpu": [10, 10],
        "fsize": [1024 * 1024, 1024 * 1024],
        "nofile": [32, 32],
    }
    assert sandbox._isolation_note(()) in verdict.stderr_tail


@_POSIX_ONLY
def test_darwin_omits_only_unreliable_address_space_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource_api = cast(Any, sandbox._resource)
    applied: list[tuple[int, tuple[int, int]]] = []
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(resource_api, "setrlimit", lambda key, value: applied.append((key, value)))

    sandbox._apply_limits()

    assert {key for key, _ in applied} == {
        resource_api.RLIMIT_CPU,
        resource_api.RLIMIT_FSIZE,
        resource_api.RLIMIT_NOFILE,
        resource_api.RLIMIT_CORE,
    }
    assert "RLIMIT_AS unsupported" in sandbox._isolation_note(())


@_POSIX_ONLY
def test_success_failure_and_verdict_fields() -> None:
    passed = sandbox.run_python(
        "import sys\nprint('out')\nprint('err', file=sys.stderr)\n", timeout_s=2.0
    )
    assert passed.passed is True
    assert passed.exit_code == 0
    assert passed.timed_out is False
    assert passed.stdout_tail == "out\n"
    assert "err\n" in passed.stderr_tail

    failed = sandbox.run_python("raise RuntimeError('boom')\n", timeout_s=2.0)
    assert failed.passed is False
    assert failed.exit_code not in {None, 0}
    assert failed.timed_out is False
    assert "RuntimeError: boom" in failed.stderr_tail


@_POSIX_ONLY
def test_timeout_kills_group_reaps_and_returns() -> None:
    verdict = sandbox.run_python("while True:\n    pass\n", timeout_s=0.05)
    assert verdict.passed is False
    assert verdict.timed_out is True
    assert verdict.exit_code is not None


@_POSIX_ONLY
def test_stdout_and_stderr_are_each_capped_at_64_kib() -> None:
    verdict = sandbox.run_python(
        "import sys\nsys.stdout.write('a' * 100000)\nsys.stderr.write('b' * 100000)\n",
        timeout_s=2.0,
    )
    assert len(verdict.stdout_tail.encode()) <= 64 * 1024
    assert len(verdict.stderr_tail.encode()) <= 64 * 1024
    assert verdict.stdout_tail.endswith("a" * 100)
    assert "llamatune sandbox" in verdict.stderr_tail


def test_tail_capture_mutation_and_snapshot_share_one_lock() -> None:
    class _RecordingLock:
        def __init__(self) -> None:
            self.entries = 0

        def __enter__(self) -> None:
            self.entries += 1

        def __exit__(self, *args: object) -> None:
            return None

    capture = sandbox._TailCapture()
    lock = _RecordingLock()
    cast(Any, capture)._lock = lock

    capture.append(b"a" * (sandbox._OUTPUT_LIMIT + 10))
    assert capture.decode() == "a" * sandbox._OUTPUT_LIMIT
    assert lock.entries == 2


@_POSIX_ONLY
def test_temp_directory_cleanup_occurs_on_success_and_spawn_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = tmp_path / "sandbox-success"

    def create_success(**_kwargs: object) -> str:
        created.mkdir()
        return str(created)

    monkeypatch.setattr(tempfile, "mkdtemp", create_success)
    assert sandbox.run_python("pass\n", timeout_s=2.0).passed is True
    assert not created.exists()

    broken = tmp_path / "sandbox-broken"

    def create_broken(**_kwargs: object) -> str:
        broken.mkdir()
        return str(broken)

    monkeypatch.setattr(tempfile, "mkdtemp", create_broken)

    def fail_spawn(*_args: object, **_kwargs: object) -> None:
        raise OSError("spawn failed")

    monkeypatch.setattr(subprocess, "Popen", fail_spawn)
    with pytest.raises(OSError, match="spawn failed"):
        sandbox.run_python("pass\n", timeout_s=2.0)
    assert not broken.exists()


@_POSIX_ONLY
def test_unavailable_platform_and_invalid_timeout_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_resource", None)
    with pytest.raises(RuntimeError, match="POSIX resource limits"):
        sandbox.run_python("pass\n", timeout_s=1.0)
    monkeypatch.undo()
    monkeypatch.setattr(sandbox, "require_exec_isolation", lambda: None)
    for value in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and positive"):
            sandbox.run_python("pass\n", timeout_s=value)


def test_network_namespace_probe_absence_is_cached_and_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.undo()
    sandbox._network_wrapper.cache_clear()
    monkeypatch.setattr(shutil, "which", lambda *_args, **_kwargs: None)
    assert sandbox._network_wrapper() == ()
    assert sandbox._network_wrapper() == ()
    sandbox._network_wrapper.cache_clear()


def test_network_namespace_probe_success_failure_and_non_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.undo()
    sandbox._network_wrapper.cache_clear()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", lambda *_args, **_kwargs: "/usr/bin/unshare")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, b"", b""),
    )
    assert sandbox._network_wrapper() == ("/usr/bin/unshare", "-rn")

    sandbox._network_wrapper.cache_clear()

    def probe_error(*_args: object, **_kwargs: object) -> None:
        raise OSError("probe failed")

    monkeypatch.setattr(subprocess, "run", probe_error)
    assert sandbox._network_wrapper() == ()

    sandbox._network_wrapper.cache_clear()
    monkeypatch.setattr(sys, "platform", "darwin")
    assert sandbox._network_wrapper() == ()
    assert sandbox._isolation_note(()) == (
        "network namespace isolation unsupported on this platform; "
        "RLIMIT_AS unsupported on this platform"
    )
    assert sandbox._isolation_note(("unshare", "-rn")) == sandbox._isolation_note(())
    assert sandbox._append_note("original", "") == "original"
    sandbox._network_wrapper.cache_clear()


def test_limit_and_process_group_helpers_reject_missing_platform_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sandbox, "_resource", None)
    with pytest.raises(RuntimeError, match="POSIX resource limits"):
        sandbox._limits()
    with pytest.raises(RuntimeError, match="POSIX resource limits"):
        sandbox._apply_limits()
    monkeypatch.delattr(os, "killpg", raising=False)
    with pytest.raises(RuntimeError, match="requires POSIX"):
        sandbox._kill_group(1)


def test_terminate_active_no_active_busy_lock_and_stale_identity() -> None:
    assert sandbox.terminate_active() is False
    sandbox._ACTIVE_LOCK.acquire()
    try:
        assert sandbox.terminate_active() is False
    finally:
        sandbox._ACTIVE_LOCK.release()

    class FakeProcess:
        pid = 999_999

        def poll(self) -> int | None:
            return None

    first = FakeProcess()
    second = FakeProcess()
    sandbox._ACTIVE_PROCESS = cast(Any, second)
    sandbox._clear_active(cast(Any, first))
    assert sandbox._ACTIVE_PROCESS is cast(Any, second)
    sandbox._ACTIVE_PROCESS = None

    class ExitedProcess(FakeProcess):
        def poll(self) -> int | None:
            return 0

    sandbox._ACTIVE_PROCESS = cast(Any, ExitedProcess())
    assert sandbox.terminate_active() is False
    assert sandbox._ACTIVE_PROCESS is None


@_POSIX_ONLY
def test_terminate_active_kills_reaps_and_overlap_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published = threading.Event()
    original_publish = sandbox._publish_active

    def publish(process: subprocess.Popen[bytes]) -> None:
        original_publish(process)
        published.set()

    monkeypatch.setattr(sandbox, "_publish_active", publish)
    result: list[sandbox.ExecVerdict] = []

    def execute() -> None:
        result.append(sandbox.run_python("while True:\n    pass\n", timeout_s=5.0))

    worker = threading.Thread(target=execute)
    worker.start()
    assert published.wait(timeout=2.0)
    with pytest.raises(RuntimeError, match="already active"):
        sandbox.run_python("pass\n", timeout_s=1.0)
    assert sandbox.terminate_active() is True
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert result[0].passed is False
    assert result[0].exit_code is not None
    assert sandbox._ACTIVE_PROCESS is None
