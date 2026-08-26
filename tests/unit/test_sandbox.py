from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from llamatune import sandbox

_POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="POSIX sandbox only")

_AVAILABLE = sandbox.IsolationStatus.AVAILABLE
_UNAVAILABLE_TOOL = sandbox.IsolationStatus.UNAVAILABLE_TOOL
_UNAVAILABLE_PERMISSION = sandbox.IsolationStatus.UNAVAILABLE_PERMISSION
_UNSUPPORTED = sandbox.IsolationStatus.UNSUPPORTED_PLATFORM

_ORIGINAL_NETWORK_PROBE = sandbox._network_probe
_ORIGINAL_BWRAP_PROBE = sandbox._bwrap_prefix_probe


@pytest.fixture(autouse=True)
def _hermetic_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decouple every test from host unshare/bwrap availability."""
    monkeypatch.setattr(
        sandbox, "_network_probe", lambda: (_AVAILABLE, ("/usr/bin/unshare", "-rn"))
    )
    monkeypatch.setattr(sandbox, "_bwrap_prefix_probe", lambda: None)
    monkeypatch.setattr(sandbox, "_ALLOW_DEGRADED_NETWORK", False)


@_POSIX_ONLY
def test_limit_plan_linux_exact_and_environment() -> None:
    resource_api = cast(Any, sandbox._resource)
    plan = sandbox.limit_plan("linux")
    assert plan.required == (
        (resource_api.RLIMIT_CPU, 10),
        (resource_api.RLIMIT_FSIZE, 1024 * 1024),
        (resource_api.RLIMIT_NOFILE, 32),
        (resource_api.RLIMIT_CORE, 0),
    )
    assert plan.memory == ((resource_api.RLIMIT_AS, 512 * 1024 * 1024),)
    assert plan.memory_label == "rlimit-as"
    assert sandbox._environment() == {"PATH": str(Path(sys.executable).resolve().parent)}


@_POSIX_ONLY
def test_limit_plan_darwin_replaces_as_with_data_rss_fallback() -> None:
    resource_api = cast(Any, sandbox._resource)
    plan = sandbox.limit_plan("darwin")
    memory_ids = {limit_id for limit_id, _ in plan.memory}
    assert memory_ids == {resource_api.RLIMIT_DATA, resource_api.RLIMIT_RSS}
    assert all(value == 512 * 1024 * 1024 for _, value in plan.memory)
    assert resource_api.RLIMIT_AS not in memory_ids
    assert plan.required == (
        (resource_api.RLIMIT_CPU, 10),
        (resource_api.RLIMIT_FSIZE, 1024 * 1024),
        (resource_api.RLIMIT_NOFILE, 32),
        (resource_api.RLIMIT_CORE, 0),
    )
    assert plan.memory_label == "rlimit-data"


def test_limit_plan_platform_variants_with_synthetic_resource_module() -> None:
    minimal = SimpleNamespace(
        RLIMIT_CPU=7,
        RLIMIT_FSIZE=9,
        RLIMIT_NOFILE=11,
        RLIMIT_CORE=13,
        RLIMIT_AS=15,
    )
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(sandbox, "_resource", minimal)
        linux = sandbox.limit_plan("linux")
        assert linux.memory == ((15, 512 * 1024 * 1024),)
        assert linux.memory_label == "rlimit-as"

        darwin_bare = sandbox.limit_plan("darwin")
        assert darwin_bare.memory == ()
        assert darwin_bare.memory_label == "rlimit-none"

        darwin_data_only = SimpleNamespace(
            RLIMIT_CPU=7,
            RLIMIT_FSIZE=9,
            RLIMIT_NOFILE=11,
            RLIMIT_CORE=13,
            RLIMIT_AS=15,
            RLIMIT_DATA=17,
        )
        monkey.setattr(sandbox, "_resource", darwin_data_only)
        plan = sandbox.limit_plan("darwin")
        assert plan.memory == ((17, 512 * 1024 * 1024),)
        assert plan.memory_label == "rlimit-data"
    finally:
        monkey.undo()


@_POSIX_ONLY
def test_apply_limits_darwin_tolerates_rejected_memory_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource_api = cast(Any, sandbox._resource)
    applied: list[int] = []
    rejected = {resource_api.RLIMIT_DATA, resource_api.RLIMIT_RSS}

    def setrlimit(resource_id: int, value: tuple[int, int]) -> None:
        del value
        if resource_id in rejected:
            raise OSError("rejected by platform")
        applied.append(resource_id)

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(resource_api, "setrlimit", setrlimit)

    sandbox._apply_limits()

    assert applied == [
        resource_api.RLIMIT_CPU,
        resource_api.RLIMIT_FSIZE,
        resource_api.RLIMIT_NOFILE,
        resource_api.RLIMIT_CORE,
    ]


@_POSIX_ONLY
def test_apply_limits_required_failures_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    resource_api = cast(Any, sandbox._resource)

    def setrlimit(_resource_id: int, _value: tuple[int, int]) -> None:
        raise OSError("hard failure")

    monkeypatch.setattr(resource_api, "setrlimit", setrlimit)
    with pytest.raises(OSError, match="hard failure"):
        sandbox._apply_limits()


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
    # Isolation is confirmed active in the hermetic fixture, so no note is appended.
    assert "[llamatune sandbox:" not in verdict.stderr_tail


@_POSIX_ONLY
def test_isolation_note_covers_degraded_postures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_network_probe", lambda: (_UNAVAILABLE_TOOL, ()))
    assert sandbox._isolation_note(()) == (
        "network namespace isolation unavailable (unavailable-tool)"
    )
    assert sandbox._isolation_note(("/usr/bin/unshare", "-rn")) == ""
    monkeypatch.setattr(sys, "platform", "darwin")
    assert sandbox._isolation_note(()) == (
        "network namespace isolation unsupported on this platform; "
        "RLIMIT_AS unsupported on this platform; RLIMIT_DATA fallback attempted"
    )

    class BareResource:
        RLIMIT_CPU = 1
        RLIMIT_FSIZE = 2
        RLIMIT_NOFILE = 3
        RLIMIT_CORE = 4
        RLIMIT_AS = 5

    monkeypatch.setattr(sandbox, "_resource", BareResource())
    assert "no memory rlimit applied" in sandbox._isolation_note(())
    assert sandbox._append_note("original", "") == "original"


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
    for value in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and positive"):
            sandbox.run_python("pass\n", timeout_s=value)


def test_network_isolation_probe_reports_all_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unsupported platform.
    with monkeypatch.context() as context:
        context.setattr(sys, "platform", "darwin")
        _ORIGINAL_NETWORK_PROBE.cache_clear()
        assert _ORIGINAL_NETWORK_PROBE()[0] is _UNSUPPORTED
        _ORIGINAL_NETWORK_PROBE.cache_clear()

    with monkeypatch.context() as context:
        context.setattr(sys, "platform", "linux")
        # Tool missing.
        context.setattr(shutil, "which", lambda *_args, **_kwargs: None)
        _ORIGINAL_NETWORK_PROBE.cache_clear()
        assert _ORIGINAL_NETWORK_PROBE()[0] is _UNAVAILABLE_TOOL
        _ORIGINAL_NETWORK_PROBE.cache_clear()

        # Probe rejects the namespace (kernel permission).
        context.setattr(
            subprocess,
            "run",
            lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, b"", b"denied"),
        )
        context.setattr(shutil, "which", lambda *_args, **_kwargs: "/usr/bin/unshare")
        _ORIGINAL_NETWORK_PROBE.cache_clear()
        assert _ORIGINAL_NETWORK_PROBE()[0] is _UNAVAILABLE_PERMISSION
        assert _ORIGINAL_NETWORK_PROBE()[1] == ()
        _ORIGINAL_NETWORK_PROBE.cache_clear()

        # Probe crashes or times out.
        def probe_error(*_args: object, **_kwargs: object) -> None:
            raise subprocess.TimeoutExpired(cmd="unshare", timeout=2.0)

        context.setattr(subprocess, "run", probe_error)
        _ORIGINAL_NETWORK_PROBE.cache_clear()
        assert _ORIGINAL_NETWORK_PROBE()[0] is _UNAVAILABLE_PERMISSION
        _ORIGINAL_NETWORK_PROBE.cache_clear()

        # Probe succeeds and reports the wrapper argv.
        context.setattr(
            subprocess,
            "run",
            lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, b"", b""),
        )
        _ORIGINAL_NETWORK_PROBE.cache_clear()
        assert _ORIGINAL_NETWORK_PROBE()[0] is _AVAILABLE
        assert _ORIGINAL_NETWORK_PROBE()[1] == ("/usr/bin/unshare", "-rn")
        _ORIGINAL_NETWORK_PROBE.cache_clear()


@_POSIX_ONLY
def test_run_python_fails_closed_before_spawn_without_confirmed_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def exploding_spawn(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("child was spawned without confirmed isolation")

    monkeypatch.setattr(subprocess, "Popen", exploding_spawn)
    for status in (_UNAVAILABLE_TOOL, _UNAVAILABLE_PERMISSION, _UNSUPPORTED):
        monkeypatch.setattr(sandbox, "_network_probe", lambda s=status: (s, ()))
        with pytest.raises(sandbox.SandboxIsolationError) as excinfo:
            sandbox.run_python("pass\n", timeout_s=1.0)
        message = str(excinfo.value)
        assert "--exec-allow-network" in message
        assert status.value in message


@_POSIX_ONLY
def test_opt_in_flag_runs_unwrapped_when_isolation_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sandbox, "_network_probe", lambda: (_UNAVAILABLE_PERMISSION, ()))
    recorded: list[tuple[str, ...]] = []
    original_popen = subprocess.Popen

    def record_popen(argv: Any, *args: Any, **kwargs: Any) -> Any:
        recorded.append(tuple(argv))
        return original_popen(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", record_popen)

    verdict = sandbox.run_python("print('ok')\n", timeout_s=2.0, allow_network=True)
    assert verdict.passed is True
    assert Path(recorded[0][0]).name == Path(sys.executable).name

    recorded.clear()
    sandbox.set_allow_network_fallback(True)
    try:
        verdict = sandbox.run_python("print('ok')\n", timeout_s=2.0)
    finally:
        sandbox.set_allow_network_fallback(False)
    assert verdict.passed is True
    assert Path(recorded[0][0]).name == Path(sys.executable).name


@_POSIX_ONLY
def test_explicit_false_overrides_module_level_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_network_probe", lambda: (_UNAVAILABLE_PERMISSION, ()))
    sandbox.set_allow_network_fallback(True)
    try:
        with pytest.raises(sandbox.SandboxIsolationError):
            sandbox.run_python("pass\n", timeout_s=1.0, allow_network=False)
    finally:
        sandbox.set_allow_network_fallback(False)


@_POSIX_ONLY
def test_bwrap_confinement_argv_shape_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    static = (
        "/usr/bin/bwrap",
        "--ro-bind",
        "/usr",
        "/usr",
        "--tmpfs",
        "/home",
    )
    child = "/usr/bin/python3"
    monkeypatch.setattr(sandbox, "_bwrap_prefix_probe", lambda: (static, child))
    monkeypatch.setattr(
        sandbox, "_network_probe", lambda: (_AVAILABLE, ("/usr/bin/unshare", "-rn"))
    )
    recorded: list[tuple[str, ...]] = []

    class FakeProcess:
        pid = 4_242_421

        def __init__(self) -> None:
            self.stdout = io.BytesIO(b"")
            self.stderr = io.BytesIO(b"")
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            self.returncode = 0
            return 0

    def fake_popen(argv: Any, *_args: object, **_kwargs: object) -> FakeProcess:
        recorded.append(tuple(argv))
        return FakeProcess()

    kills: list[int] = []
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sandbox, "_kill_group", lambda pid: kills.append(pid))

    verdict = sandbox.run_python("print('confined')\n", timeout_s=2.0)
    assert verdict.passed is True
    argv = recorded[0]
    head = argv[: len(static)]
    assert head == static
    bind_dir = argv[len(static) + 1]
    assert argv[len(static) : len(static) + 4] == (
        "--bind",
        bind_dir,
        bind_dir,
        "--die-with-parent",
    )
    rest = argv[len(static) + 4 :]
    assert rest[:2] == ("/usr/bin/unshare", "-rn")
    assert rest[2:] == (child, "-I", "-S", "-B", "main.py")
    assert "$HOME" not in argv and str(Path.home()) not in argv
    assert Path(bind_dir).name.startswith("llamatune-quality-exec-")


def test_filesystem_confinement_detection_maps_probe_to_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert sandbox.detect_filesystem_confinement() == "none"
    monkeypatch.setattr(sandbox, "_bwrap_prefix_probe", lambda: (("/bin/bwrap",), sys.executable))
    assert sandbox.detect_filesystem_confinement() == "bubblewrap"


def test_bwrap_prefix_probe_builds_static_args_and_alt_mount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = subprocess.CompletedProcess([], 0, b"", b"")

    with monkeypatch.context() as context:
        context.setattr(sys, "platform", "linux")
        context.setattr(shutil, "which", lambda name, *_a, **_k: f"/usr/bin/{name}")
        context.setattr(subprocess, "run", lambda *_a, **_k: completed)
        _ORIGINAL_BWRAP_PROBE.cache_clear()
        probed = _ORIGINAL_BWRAP_PROBE()
        assert probed is not None
        static, child = probed
        assert static[0] == "/usr/bin/bwrap"
        assert (static[1], static[2], static[3]) == ("--ro-bind", "/usr", "/usr")
        for flag in ("--proc", "--dev", "--tmpfs"):
            assert flag in static
        assert (
            static[static.index("--tmpfs") + 1],
            static[static.index("--tmpfs") + 3],
            static[static.index("--tmpfs") + 5],
        ) == ("/tmp", "/home", "/run")  # noqa: S108 - bwrap mount targets
        expected_child = (
            sys.executable
            if sandbox._interpreter_visible_in_confinement()
            else f"/llamatune-py/{Path(sys.executable).name}"
        )
        assert child == expected_child
        _ORIGINAL_BWRAP_PROBE.cache_clear()

    with monkeypatch.context() as context:
        context.setattr(sys, "platform", "linux")
        context.setattr(shutil, "which", lambda name, *_a, **_k: f"/usr/bin/{name}")
        context.setattr(subprocess, "run", lambda *_a, **_k: completed)
        context.setattr(sys, "executable", "/home/dev/.venv/bin/pythonX")
        _ORIGINAL_BWRAP_PROBE.cache_clear()
        probed = _ORIGINAL_BWRAP_PROBE()
        assert probed is not None
        static, child = probed
        assert child == "/llamatune-py/pythonX"
        assert static[-3:] == ("--ro-bind", "/home/dev/.venv/bin", "/llamatune-py")
        _ORIGINAL_BWRAP_PROBE.cache_clear()

    with monkeypatch.context() as context:
        context.setattr(sys, "platform", "linux")
        context.setattr(shutil, "which", lambda name, *_a, **_k: None)
        _ORIGINAL_BWRAP_PROBE.cache_clear()
        assert _ORIGINAL_BWRAP_PROBE() is None

    with monkeypatch.context() as context:
        context.setattr(sys, "platform", "linux")
        context.setattr(shutil, "which", lambda name, *_a, **_k: "/usr/bin/bwrap")
        context.setattr(
            subprocess,
            "run",
            lambda *_a, **_k: subprocess.CompletedProcess([], 32, b"", b"boom"),
        )
        _ORIGINAL_BWRAP_PROBE.cache_clear()
        assert _ORIGINAL_BWRAP_PROBE() is None

    with monkeypatch.context() as context:
        context.setattr(sys, "platform", "linux")
        context.setattr(shutil, "which", lambda name, *_a, **_k: "/usr/bin/bwrap")

        def probe_crash(*_a: object, **_k: object) -> None:
            raise OSError("bwrap unusable")

        context.setattr(subprocess, "run", probe_crash)
        _ORIGINAL_BWRAP_PROBE.cache_clear()
        assert _ORIGINAL_BWRAP_PROBE() is None

    with monkeypatch.context() as context:
        context.setattr(sys, "platform", "darwin")
        _ORIGINAL_BWRAP_PROBE.cache_clear()
        assert _ORIGINAL_BWRAP_PROBE() is None
        assert sandbox.detect_filesystem_confinement() == "none"
        _ORIGINAL_BWRAP_PROBE.cache_clear()


def test_describe_isolation_summaries_warnings_and_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bwrap_probe = (("/usr/bin/bwrap", "--ro-bind", "/usr", "/usr"), sys.executable)
    monkeypatch.setattr(sandbox, "_bwrap_prefix_probe", lambda: bwrap_probe)

    active = sandbox.describe_isolation()
    assert active.summary == "network-namespace"
    assert active.network_active is True
    assert active.network_status is _AVAILABLE
    assert active.filesystem_confinement == "bubblewrap"
    assert active.warnings == ()

    monkeypatch.setattr(sandbox, "_network_probe", lambda: (_UNAVAILABLE_PERMISSION, ()))
    allowed_by_param = sandbox.describe_isolation(allow_network=True)
    assert allowed_by_param.summary == "none (allowed by flag)"
    assert any("runs with network access" in warning for warning in allowed_by_param.warnings)
    assert any("filesystem confinement inactive" in w for w in allowed_by_param.warnings) is False

    sandbox.set_allow_network_fallback(True)
    try:
        allowed_by_state = sandbox.describe_isolation()
        assert allowed_by_state.summary == "none (allowed by flag)"
    finally:
        sandbox.set_allow_network_fallback(False)

    blocked = sandbox.describe_isolation(allow_network=False)
    assert blocked.summary == "none (unavailable)"

    monkeypatch.setattr(sandbox, "_bwrap_prefix_probe", lambda: None)
    degraded_fs = sandbox.describe_isolation(allow_network=True)
    assert degraded_fs.filesystem_confinement == "none"
    assert any(
        "filesystem confinement inactive (bubblewrap missing or unusable)" in warning
        for warning in degraded_fs.warnings
    )


def test_describe_isolation_flags_missing_memory_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BareResource:
        RLIMIT_CPU = 1
        RLIMIT_FSIZE = 2
        RLIMIT_NOFILE = 3
        RLIMIT_CORE = 4
        RLIMIT_AS = 5

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sandbox, "_resource", BareResource())
    report = sandbox.describe_isolation(allow_network=True)
    assert report.memory_limit == "rlimit-none"
    assert any("no memory rlimit could be applied" in warning for warning in report.warnings)


def test_limit_and_process_group_helpers_reject_missing_platform_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sandbox, "_resource", None)
    with pytest.raises(RuntimeError, match="POSIX resource limits"):
        sandbox.limit_plan("linux")
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
