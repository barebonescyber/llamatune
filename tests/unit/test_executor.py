"""Unit tests for llamatune.executor (bounded subprocess execution)."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from llamatune import executor


class _FakeWindowsApi:
    def __init__(
        self, *, create_ok: bool = True, assign_ok: bool = True, terminate_ok: bool = True
    ) -> None:
        self.create_ok = create_ok
        self.assign_ok = assign_ok
        self.terminate_ok = terminate_ok
        self.assigned: list[tuple[object, int]] = []
        self.terminated: list[object] = []
        self.closed: list[object] = []

    def create_kill_on_close_job(self) -> object | None:
        return "job" if self.create_ok else None

    def assign(self, job: object, pid: int) -> bool:
        self.assigned.append((job, pid))
        return self.assign_ok

    def terminate(self, job: object) -> bool:
        self.terminated.append(job)
        return self.terminate_ok

    def close(self, handle: object) -> None:
        self.closed.append(handle)


def _run(
    argv: list[str], tmp_path: Path, *, timeout_s: float = 30.0, **kwargs: object
) -> executor.ExecResult:
    return executor.run(
        argv,
        timeout_s=timeout_s,
        stdout_path=tmp_path / "stdout.bin",
        stderr_path=tmp_path / "stderr.bin",
        **kwargs,  # type: ignore[arg-type]
    )


class TestBuildChildEnv:
    def test_exact_allowlist_kept(self) -> None:
        source = {
            "PATH": "/usr/bin",
            "HOME": "/users/test",
            "SECRET_TOKEN": "hunter2",  # pragma: allowlist secret
        }
        env = executor.build_child_env(source)
        assert env == {"PATH": "/usr/bin", "HOME": "/users/test"}

    def test_windows_process_environment_names_kept(self) -> None:
        source = {
            "SYSTEMROOT": r"C:\Windows",
            "COMSPEC": r"C:\Windows\System32\cmd.exe",
            "PATHEXT": ".EXE;.CMD;.BAT",
        }
        assert executor.build_child_env(source) == source

    def test_prefix_allowlist_kept(self) -> None:
        source = {
            "GGML_METAL_DEBUG": "1",
            "LLAMA_ARG_THREADS": "4",
            "LLAMATUNE_FAKE_VRAM_MB": "2500",
            "LLAMATUNEX_NOT_ALLOWED": "x",
            "AWS_SECRET_ACCESS_KEY": "nope",  # pragma: allowlist secret
        }
        env = executor.build_child_env(source)
        assert set(env) == {"GGML_METAL_DEBUG", "LLAMA_ARG_THREADS", "LLAMATUNE_FAKE_VRAM_MB"}

    def test_defaults_to_os_environ(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLAMATUNE_FAKE_MARKER", "yes")
        monkeypatch.setenv("DEFINITELY_NOT_ALLOWED_VAR", "no")
        env = executor.build_child_env()
        assert env.get("LLAMATUNE_FAKE_MARKER") == "yes"
        assert "DEFINITELY_NOT_ALLOWED_VAR" not in env


class TestRun:
    def test_heartbeat_runs_and_callback_failure_is_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(executor, "_HEARTBEAT_S", 0.01)
        calls: list[float] = []

        def heartbeat(elapsed: float) -> None:
            calls.append(elapsed)
            raise RuntimeError("renderer failed")

        result = _run(
            [sys.executable, "-c", "import time; time.sleep(0.05)"],
            tmp_path,
            on_heartbeat=heartbeat,
        )
        assert result.exit_code == 0
        assert calls

    def test_captures_output_and_exit_code(self, tmp_path: Path) -> None:
        code = "import sys; sys.stdout.write('out-data'); sys.stderr.write('err-data')"
        result = _run([sys.executable, "-c", code], tmp_path)

        assert result.exit_code == 0
        assert result.timed_out is False
        assert result.stdout.path.read_bytes() == b"out-data"
        assert result.stderr.path.read_bytes() == b"err-data"
        assert result.stdout.sha256 == hashlib.sha256(b"out-data").hexdigest()
        assert result.stdout.size_bytes == len(b"out-data")
        assert result.stdout.truncated is False
        assert result.wall_s > 0
        assert result.started <= result.ended

    def test_nonzero_exit_code(self, tmp_path: Path) -> None:
        result = _run([sys.executable, "-c", "raise SystemExit(7)"], tmp_path)
        assert result.exit_code == 7

    def test_env_is_allowlisted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SUPER_SECRET_VALUE", "leak-me")
        monkeypatch.setenv("LLAMATUNE_FAKE_OK", "fine")
        code = "import os, sys; sys.stdout.write(','.join(sorted(os.environ)))"
        result = _run([sys.executable, "-c", code], tmp_path)
        child_env_names = result.stdout.path.read_text().split(",")
        assert "SUPER_SECRET_VALUE" not in child_env_names
        assert "LLAMATUNE_FAKE_OK" in child_env_names
        assert "SUPER_SECRET_VALUE" not in result.env_names
        assert "LLAMATUNE_FAKE_OK" in result.env_names

    def test_explicit_env_is_allowlisted_not_bypassed(self, tmp_path: Path) -> None:
        # `env` is a source mapping filtered through the same allowlist, not a
        # trusted final environment: an allowed name survives, while an adjacent
        # secret-like name is dropped from the child and from env_names.
        code = (
            "import os, sys; "
            "allowed = os.environ.get('LLAMATUNE_FAKE_ONLY', 'missing'); "
            # pragma: allowlist nextline secret
            "secret = 'leaked' if 'AWS_SECRET_ACCESS_KEY' in os.environ else 'absent'; "
            "sys.stdout.write(allowed + '|' + secret)"
        )
        result = _run(
            [sys.executable, "-c", code],
            tmp_path,
            env={
                "LLAMATUNE_FAKE_ONLY": "value",
                "AWS_SECRET_ACCESS_KEY": "leak-me",  # pragma: allowlist secret
            },
        )
        assert result.stdout.path.read_text() == "value|absent"
        assert result.env_names == ("LLAMATUNE_FAKE_ONLY",)
        assert "AWS_SECRET_ACCESS_KEY" not in result.env_names

    def test_stdout_truncation_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(executor, "STDOUT_CAP_BYTES", 16)
        code = "import sys; sys.stdout.write('x' * 100)"
        result = _run([sys.executable, "-c", code], tmp_path)
        assert result.stdout.truncated is True
        assert result.stdout.size_bytes == 16
        assert result.stdout.path.read_bytes() == b"x" * 16
        # The hash covers exactly the bytes that were written to disk.
        assert result.stdout.sha256 == hashlib.sha256(b"x" * 16).hexdigest()

    def test_stderr_truncation_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(executor, "STDERR_CAP_BYTES", 8)
        code = "import sys; sys.stderr.write('e' * 100)"
        result = _run([sys.executable, "-c", code], tmp_path)
        assert result.stderr.truncated is True
        assert result.stderr.size_bytes == 8

    def test_timeout_kills_process_group(self, tmp_path: Path) -> None:
        start = time.monotonic()
        result = _run([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path, timeout_s=1)
        elapsed = time.monotonic() - start

        assert result.timed_out is True
        assert result.exit_code != 0
        assert elapsed < 30  # far below the child's 60 s sleep

    def test_timeout_wall_seconds_recorded(self, tmp_path: Path) -> None:
        result = _run(
            [sys.executable, "-c", "import time; time.sleep(60)"], tmp_path, timeout_s=0.5
        )
        assert result.timed_out is True
        assert result.wall_s >= 0.5

    def test_cwd_is_respected(self, tmp_path: Path) -> None:
        workdir = tmp_path / "work"
        workdir.mkdir()
        code = "import os, sys; sys.stdout.write(os.getcwd())"
        result = _run([sys.executable, "-c", code], tmp_path, cwd=workdir)
        assert result.stdout.path.read_text() == str(workdir.resolve())

    def test_capture_open_failure_surfaces_cause(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stdout_path = tmp_path / "stdout.bin"
        original_open = Path.open

        def fail_stdout_open(path: Path, *args: Any, **kwargs: Any) -> Any:
            if path == stdout_path:
                raise OSError("capture storage unavailable")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", fail_stdout_open)

        with pytest.raises(RuntimeError, match=r"failed to capture child stdout output") as caught:
            _run([sys.executable, "-c", "pass"], tmp_path)

        assert isinstance(caught.value.__cause__, OSError)
        assert str(stdout_path) in str(caught.value)

    @pytest.mark.skipif(importlib.util.find_spec("resource") is None, reason="resource unavailable")
    def test_core_dumps_disabled_by_default(self, tmp_path: Path) -> None:
        code = "import resource; print(resource.getrlimit(resource.RLIMIT_CORE))"
        result = _run([sys.executable, "-c", code], tmp_path)
        assert result.stdout.path.read_text().strip() == "(0, 0)"

    @pytest.mark.skipif(importlib.util.find_spec("resource") is None, reason="resource unavailable")
    def test_abort_does_not_leave_a_core_file(self, tmp_path: Path) -> None:
        result = _run([sys.executable, "-c", "import os; os.abort()"], tmp_path, cwd=tmp_path)
        assert result.exit_code != 0
        assert not list(tmp_path.glob("core*"))

    @pytest.mark.skipif(importlib.util.find_spec("resource") is None, reason="resource unavailable")
    def test_core_dump_limit_can_be_inherited(self, tmp_path: Path) -> None:
        import resource

        getrlimit = cast(
            Callable[[int], tuple[int, int]] | None,
            getattr(resource, "getrlimit", None),
        )
        rlimit_core = cast(int | None, getattr(resource, "RLIMIT_CORE", None))
        assert getrlimit is not None and rlimit_core is not None
        code = "import resource; print(resource.getrlimit(resource.RLIMIT_CORE))"
        result = _run([sys.executable, "-c", code], tmp_path, disable_core_dumps=False)
        assert result.stdout.path.read_text().strip() == str(getrlimit(rlimit_core))


class TestRunProbe:
    def test_child_does_not_receive_sensitive_parent_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_API_TOKEN", "must-not-leak")
        code = "import os; print(os.environ.get('FAKE_API_TOKEN', 'absent'))"

        result = executor.run_probe([sys.executable, "-c", code])

        assert result is not None
        assert result.stdout.splitlines() == [b"absent"]

    def test_env_is_allowlisted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SUPER_SECRET_VALUE", "leak-me")
        code = "import os; print(os.environ.get('SUPER_SECRET_VALUE', 'absent'))"

        result = executor.run_probe([sys.executable, "-c", code])

        assert result is not None
        assert result.stdout.splitlines() == [b"absent"]

    def test_output_is_truncated_at_cap(self) -> None:
        code = "import sys; sys.stdout.write('x' * 100); sys.stderr.write('e' * 100)"

        result = executor.run_probe([sys.executable, "-c", code], max_output_bytes=7)

        assert result is not None
        assert result.stdout == b"x" * 7
        assert result.stderr == b"e" * 7

    def test_timeout_kills_process_group(self) -> None:
        result = executor.run_probe(
            [sys.executable, "-c", "import time; time.sleep(60)"], timeout_s=0.1
        )

        assert result is not None
        assert result.timed_out is True
        assert result.exit_code != 0

    def test_missing_binary_returns_none(self) -> None:
        assert executor.run_probe(["definitely-not-a-real-binary-xyz"]) is None

    def test_capture_worker_failure_is_not_reported_as_key_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            executor,
            "_drain_memory",
            lambda stream, cap: (_ for _ in ()).throw(OSError("pipe read failed")),
        )

        with pytest.raises(RuntimeError, match="failed to capture child") as caught:
            executor.run_probe([sys.executable, "-c", "pass"])

        assert isinstance(caught.value.__cause__, OSError)


class TestTerminateGroup:
    def test_terminate_already_exited_process(self, tmp_path: Path) -> None:
        import subprocess

        proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
        proc.wait()
        # Must not raise even though the process (and group) is gone.
        executor.terminate_group(proc)

    def test_windows_job_is_assigned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        api = _FakeWindowsApi()
        monkeypatch.setattr(executor, "_windows_api", lambda: api)

        assert executor._create_windows_job(42) == "job"
        assert api.assigned == [("job", 42)]
        assert api.closed == []

    def test_windows_job_creation_failure_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        api = _FakeWindowsApi(create_ok=False)
        monkeypatch.setattr(executor, "_windows_api", lambda: api)

        assert executor._create_windows_job(42) is None
        assert api.assigned == []

    def test_failed_windows_job_assignment_closes_handle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        api = _FakeWindowsApi(assign_ok=False)
        monkeypatch.setattr(executor, "_windows_api", lambda: api)

        assert executor._create_windows_job(42) is None
        assert api.closed == ["job"]

    def test_windows_termination_uses_job_and_closes_handle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        api = _FakeWindowsApi()
        monkeypatch.setattr(executor, "_windows_api", lambda: api)
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        control = executor.ProcessControl(windows=True, job_handle="job")

        executor.terminate_group(proc, control)

        assert api.terminated == ["job"]
        assert api.closed == ["job"]
        assert control.job_handle is None

    def test_windows_termination_falls_back_to_taskkill(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        api = _FakeWindowsApi(terminate_ok=False)
        killed: list[int] = []
        monkeypatch.setattr(executor, "_windows_api", lambda: api)
        monkeypatch.setattr(executor, "_taskkill_tree", killed.append)
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()

        executor.terminate_group(proc, executor.ProcessControl(windows=True, job_handle="job"))

        assert killed == [proc.pid]

    def test_windows_termination_without_job_falls_back_to_taskkill(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        killed: list[int] = []
        monkeypatch.setattr(executor, "_taskkill_tree", killed.append)
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()

        executor.terminate_group(proc, executor.ProcessControl(windows=True))

        assert killed == [proc.pid]


class TestTaskkillFallback:
    def test_taskkill_argv_timeout_env_and_devnull(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_run(argv: object, **kwargs: object) -> None:
            captured["argv"] = argv
            captured.update(kwargs)

        monkeypatch.setattr(subprocess, "run", fake_run)

        executor._taskkill_tree(4321)

        assert captured["argv"] == ["taskkill", "/PID", "4321", "/T", "/F"]
        timeout = captured["timeout"]
        assert isinstance(timeout, (int, float)) and timeout > 0
        assert captured["env"] == executor.build_child_env()
        assert captured["stdout"] is subprocess.DEVNULL
        assert captured["stderr"] is subprocess.DEVNULL
        assert captured["check"] is False

    def test_taskkill_timeout_expired_is_suppressed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(argv: object, **kwargs: object) -> None:
            raise subprocess.TimeoutExpired(cast(list[str], argv), cast(float, kwargs["timeout"]))

        monkeypatch.setattr(subprocess, "run", fake_run)

        # Best-effort cleanup: a timeout must not escape or replace the outcome.
        executor._taskkill_tree(4321)

    def test_taskkill_startup_error_is_suppressed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(argv: object, **kwargs: object) -> None:
            raise OSError("taskkill is unavailable")

        monkeypatch.setattr(subprocess, "run", fake_run)

        executor._taskkill_tree(4321)

    def test_job_success_avoids_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        api = _FakeWindowsApi()  # terminate_ok=True by default
        killed: list[int] = []
        monkeypatch.setattr(executor, "_windows_api", lambda: api)
        monkeypatch.setattr(executor, "_taskkill_tree", killed.append)
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()

        executor.terminate_group(proc, executor.ProcessControl(windows=True, job_handle="job"))

        assert api.terminated == ["job"]
        assert killed == []


def test_windows_spawn_attaches_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor, "_WINDOWS", True)
    monkeypatch.setattr(executor, "_create_windows_job", lambda pid: f"job-{pid}")

    proc, control = executor.spawn_supervised(
        [sys.executable, "-c", "pass"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc.wait()

    assert control.windows is True
    assert control.job_handle == f"job-{proc.pid}"


def test_drain_continues_after_file_capture_cap(tmp_path: Path) -> None:
    data = b"x" * (executor._READ_CHUNK + 1)

    capture = executor._drain(io.BytesIO(data), tmp_path / "capture.bin", 1)

    assert capture.size_bytes == 1
    assert capture.truncated is True
    assert capture.path.read_bytes() == b"x"


def test_memory_drain_continues_after_capture_cap() -> None:
    data = b"x" * (executor._READ_CHUNK + 1)

    assert executor._drain_memory(io.BytesIO(data), 1) == b"x"


def test_platform_spawn_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor, "_WINDOWS", True)
    assert executor.popen_platform_kwargs() == {"creationflags": executor._CREATE_NEW_PROCESS_GROUP}
    monkeypatch.setattr(executor, "_WINDOWS", False)

    def marker() -> None:
        pass

    assert executor.popen_platform_kwargs(marker) == {
        "start_new_session": True,
        "preexec_fn": marker,
    }


def test_core_dump_control_status(monkeypatch: pytest.MonkeyPatch) -> None:
    assert executor.core_dump_control_status(False) == "inherited"
    monkeypatch.setattr(executor, "_WINDOWS", True)
    assert executor.core_dump_control_status(True) == "unsupported"
