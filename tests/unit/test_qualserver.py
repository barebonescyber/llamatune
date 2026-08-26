"""Unit coverage for supervised literal-loopback server lifecycle."""

from __future__ import annotations

import contextlib
import http.client
import io
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest

from llamatune import executor, qualserver
from llamatune.quality import QualityRun


def _run(tmp_path: Path) -> QualityRun:
    root = tmp_path / "run"
    root.mkdir()
    return QualityRun(root)


def _fake_server_argv() -> tuple[str, ...]:
    script = Path(__file__).parents[1] / "fixtures" / "fake_llama_server.py"
    return (sys.executable, str(script), "-m", "fixture.gguf", "-c", "1024", "--seed", "7")


class _FakeTiming:
    """Injected :class:`qualserver.Timing` double with optional virtual advance."""

    def __init__(self, *, advance_on_sleep: bool = True) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []
        self._advance_on_sleep = advance_on_sleep

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        if self._advance_on_sleep:
            self.now += seconds


def test_endpoint_is_literal_loopback_and_replaces_port() -> None:
    argv = qualserver._endpoint_argv(
        (
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--api-key",
            "caller-secret",
        ),
        42,
        "our-key",
    )
    assert argv == ("server", "--host", "127.0.0.1", "--port", "42", "--api-key", "our-key")

    with pytest.raises(ValueError, match=r"literal 127\.0\.0\.1"):
        qualserver._endpoint_argv(("server", "--host", "localhost"), 42, "k")

    with pytest.raises(ValueError, match=r"--api-key requires a value"):
        qualserver._endpoint_argv(("server", "--api-key"), 42, "k")


def test_redaction_hides_every_api_key_value() -> None:
    command = ("srv", "--api-key", "a", "x", "--api-key", "b")
    assert qualserver._redact_api_key(command) == [
        "srv",
        "--api-key",
        qualserver._REDACTED_API_KEY,
        "x",
        "--api-key",
        qualserver._REDACTED_API_KEY,
    ]


def test_real_fake_server_chat_and_idempotent_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "script.json"
    script.write_text(json.dumps({"*": "known response"}), encoding="utf-8")
    monkeypatch.setenv("LLAMATUNE_FAKE_SRV_SCRIPT", str(script))
    monkeypatch.setattr(qualserver, "_READINESS_POLL_S", 0.01)
    run = _run(tmp_path)

    handle = qualserver.start(run, _fake_server_argv(), start_timeout_s=3.0)
    assert handle.alive
    assert (
        handle.chat(
            [{"role": "user", "content": "hello"}],
            max_tokens=32,
            seed=7,
            timeout_s=2.0,
        )
        == "known response"
    )

    handle.stop()
    handle.stop()
    assert not handle.alive
    assert (run.dir / "server" / "1" / "command.json").is_file()
    assert (run.dir / "server" / "1" / "stderr.log").is_file()


def test_malformed_and_oversized_responses_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLAMATUNE_FAKE_SRV_MALFORMED", "1")
    monkeypatch.setattr(qualserver, "_READINESS_POLL_S", 0.01)
    handle = qualserver.start(_run(tmp_path), _fake_server_argv(), start_timeout_s=3.0)
    try:
        with pytest.raises(qualserver.ServerProtocolError, match="malformed"):
            handle.chat(
                [{"role": "user", "content": "hello"}],
                max_tokens=32,
                seed=7,
                timeout_s=2.0,
            )
    finally:
        handle.stop()

    class _OversizedResponse:
        def read1(self, amount: int) -> bytes:
            return b"x" * amount

    # A frozen injected clock keeps the read well within its deadline so the
    # 8 MiB size cap — not the wall deadline — is what rejects the body.
    with pytest.raises(qualserver.ServerProtocolError, match="8 MiB"):
        qualserver._read_bounded_body(
            None, cast(Any, _OversizedResponse()), deadline=1.0, timing=_FakeTiming()
        )

    stream = io.BytesIO(b"a" * 8192 + b"TAIL")
    capture = qualserver._Capture(stream, 16)
    capture.start()
    capture.join()
    assert capture.data == b"a" * 16
    assert capture.truncated is True
    assert capture.tail.endswith(b"TAIL")
    assert len(capture.tail) == 4096


def test_capture_mutation_and_snapshots_share_one_lock() -> None:
    class _RecordingLock:
        def __init__(self) -> None:
            self.entries = 0

        def __enter__(self) -> None:
            self.entries += 1

        def __exit__(self, *args: object) -> None:
            return None

    capture = qualserver._Capture(io.BytesIO(b"x" * 32), 16)
    lock = _RecordingLock()
    cast(Any, capture)._lock = lock

    capture._drain()
    assert capture.data == b"x" * 16
    assert capture.tail == b"x" * 32
    assert capture.truncated is True
    assert lock.entries == 4


class _FakeProcess:
    def __init__(self) -> None:
        self.stdout = io.BytesIO()
        self.stderr = io.BytesIO()
        self.returncode: int | None = None
        self.waited = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.waited = True
        self.returncode = -15
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


def test_readiness_uses_injected_clock_without_sleep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = _FakeProcess()
    timing = _FakeTiming()
    monkeypatch.setattr(qualserver, "_free_loopback_port", lambda: 12345)
    monkeypatch.setattr(executor, "spawn_supervised", lambda *args, **kwargs: (process, object()))
    monkeypatch.setattr(executor, "terminate_group", lambda proc, control: proc.kill())
    monkeypatch.setattr(executor, "close_process_control", lambda control: None)
    monkeypatch.setattr(qualserver, "_health", lambda port, timeout, key, timing: None)

    with pytest.raises(qualserver.ServerStartError, match="startup timeout"):
        qualserver.start(_run(tmp_path), ("server",), start_timeout_s=5.0, timing=timing)

    assert timing.sleeps == [2.0, 2.0, 1.0]
    assert process.waited


def test_start_retries_on_interloper_rejection_then_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A peer that answers /health with 401 burns attempts on fresh ports."""
    processes = [_FakeProcess() for _ in range(qualserver._MAX_START_ATTEMPTS)]
    spawned: list[int] = []
    timing = _FakeTiming()
    ports = iter(range(20000, 20100))

    def spawn(argv: Any, **kwargs: Any) -> tuple[subprocess.Popen[bytes], object]:
        del kwargs
        spawned.append(int(argv[argv.index("--port") + 1]))
        process = processes[len(spawned) - 1]
        return cast(subprocess.Popen[bytes], process), object()

    monkeypatch.setattr(qualserver, "_free_loopback_port", lambda: next(ports))
    monkeypatch.setattr(executor, "spawn_supervised", spawn)
    monkeypatch.setattr(executor, "terminate_group", lambda proc, control: proc.kill())
    monkeypatch.setattr(executor, "close_process_control", lambda control: None)

    statuses = iter([401] * qualserver._MAX_START_ATTEMPTS)
    monkeypatch.setattr(
        qualserver,
        "_health",
        lambda port, timeout, key, timing: next(statuses),
    )

    run = _run(tmp_path)
    with pytest.raises(qualserver.ServerStartError, match="credentials"):
        qualserver.start(run, ("server",), start_timeout_s=30.0, timing=timing)

    assert len(spawned) == qualserver._MAX_START_ATTEMPTS
    assert len(set(spawned)) == len(spawned)  # every attempt used a fresh port
    assert all(process.waited for process in processes)  # interlopers reaped between tries


def test_start_recovers_from_lost_bind_race_with_fresh_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First child dies instantly (bind race lost); the second becomes ready."""
    loser = _FakeProcess()
    winner = _FakeProcess()
    spawned: list[list[str]] = []
    timing = _FakeTiming()

    def spawn(argv: Any, **kwargs: Any) -> tuple[subprocess.Popen[bytes], object]:
        del kwargs
        spawned.append([str(part) for part in argv])
        process = loser if len(spawned) == 1 else winner
        if len(spawned) == 1:
            process.returncode = 1  # exited during startup before first poll
            process.poll = lambda: 1  # type: ignore[method-assign]
        return cast(subprocess.Popen[bytes], process), object()

    monkeypatch.setattr(qualserver, "_free_loopback_port", lambda: 12345)
    monkeypatch.setattr(executor, "spawn_supervised", spawn)
    monkeypatch.setattr(executor, "terminate_group", lambda proc, control: proc.kill())
    monkeypatch.setattr(executor, "close_process_control", lambda control: None)
    monkeypatch.setattr(qualserver, "_health", lambda port, timeout, key, timing: 200)

    run = _run(tmp_path)
    handle = qualserver.start(run, ("server",), start_timeout_s=5.0, timing=timing)
    try:
        assert handle.alive
        assert len(spawned) == 2
        ready = [entry for entry in run.entries if entry.get("type") == "server_ready"]
        assert [(entry["launch"], entry["port"]) for entry in ready] == [(1, 12345)]
    finally:
        handle.stop()


def test_start_evidence_redacts_the_per_launch_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = _FakeProcess()
    commands: list[tuple[str, ...]] = []
    timing = _FakeTiming()

    def spawn(argv: Any, **kwargs: Any) -> tuple[subprocess.Popen[bytes], object]:
        del kwargs
        commands.append(tuple(str(part) for part in argv))
        return cast(subprocess.Popen[bytes], process), object()

    monkeypatch.setattr(qualserver, "_free_loopback_port", lambda: 12345)
    monkeypatch.setattr(executor, "spawn_supervised", spawn)
    monkeypatch.setattr(executor, "terminate_group", lambda proc, control: proc.kill())
    monkeypatch.setattr(executor, "close_process_control", lambda control: None)
    monkeypatch.setattr(qualserver, "_health", lambda port, timeout, key, timing: 200)

    run = _run(tmp_path)
    handle = qualserver.start(run, ("server",), start_timeout_s=5.0, timing=timing)
    handle.stop()

    command_json = json.loads((run.dir / "server" / "1" / "command.json").read_text())
    journal_argv = next(
        entry["argv"] for entry in run.entries if entry.get("type") == "server_start"
    )
    secret = commands[0][commands[0].index("--api-key") + 1]
    assert secret and "--api-key" in commands[0]
    for recorded in (command_json["argv"], journal_argv):
        assert "--api-key" in recorded
        assert recorded.index("--api-key") < len(recorded) - 1
        assert recorded[recorded.index("--api-key") + 1] == qualserver._REDACTED_API_KEY
        assert secret not in recorded


def test_post_spawn_journal_failure_always_reaps_and_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = _FakeProcess()
    closed: list[object] = []
    monkeypatch.setattr(qualserver, "_free_loopback_port", lambda: 12345)
    monkeypatch.setattr(executor, "spawn_supervised", lambda *args, **kwargs: (process, object()))
    monkeypatch.setattr(executor, "terminate_group", lambda proc, control: proc.kill())
    monkeypatch.setattr(
        executor,
        "close_process_control",
        lambda control: closed.append(control),
    )
    run = _run(tmp_path)
    monkeypatch.setattr(run, "append", lambda entry: (_ for _ in ()).throw(OSError("journal")))

    with pytest.raises(OSError, match="journal"):
        qualserver.start(run, ("server",), start_timeout_s=1.0)

    assert process.waited
    assert len(closed) == 1


def test_stop_reaps_after_wait_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    process = _FakeProcess()
    calls = 0

    def wait(timeout: float | None = None) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.TimeoutExpired("server", timeout or 0.0)
        process.waited = True
        return -9

    process.wait = wait  # type: ignore[method-assign]
    capture_out = qualserver._Capture(io.BytesIO(), 10)
    capture_err = qualserver._Capture(io.BytesIO(), 10)
    capture_out.start()
    capture_err.start()
    monkeypatch.setattr(executor, "terminate_group", lambda proc, control: None)
    monkeypatch.setattr(executor, "close_process_control", lambda control: None)
    handle = qualserver.ServerHandle(
        run=_run(tmp_path),
        launch_number=1,
        argv=("server",),
        port=1,
        process=cast(Any, process),
        control=object(),
        stdout_capture=capture_out,
        stderr_capture=capture_err,
    )

    handle.stop()

    assert process.returncode == -9
    assert process.waited


def test_stop_can_retry_after_interrupted_log_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = _FakeProcess()
    capture_out = qualserver._Capture(io.BytesIO(b"out"), 10)
    capture_err = qualserver._Capture(io.BytesIO(b"err"), 10)
    capture_out.start()
    capture_err.start()
    monkeypatch.setattr(executor, "terminate_group", lambda proc, control: proc.kill())
    monkeypatch.setattr(executor, "close_process_control", lambda control: None)
    run = _run(tmp_path)
    original_write = run.write_text
    failures = 0

    def interrupt_once(relative: str, content: str) -> None:
        nonlocal failures
        if failures == 0:
            failures += 1
            raise KeyboardInterrupt
        original_write(relative, content)

    monkeypatch.setattr(run, "write_text", interrupt_once)
    handle = qualserver.ServerHandle(
        run=run,
        launch_number=1,
        argv=("server",),
        port=1,
        process=cast(Any, process),
        control=object(),
        stdout_capture=capture_out,
        stderr_capture=capture_err,
    )

    with pytest.raises(KeyboardInterrupt):
        handle.stop()
    handle.stop()

    assert not handle.alive
    assert (run.dir / "server" / "1" / "stdout.log").read_text() == "out"
    assert (run.dir / "server" / "1" / "stderr.log").read_text() == "err"


def test_body_read_rejects_trickle_and_deadline_is_not_reset() -> None:
    timing = _FakeTiming()

    class _Trickle:
        def __init__(self) -> None:
            self.reads = 0

        def read1(self, amount: int) -> bytes:
            self.reads += 1
            timing.now += 0.05  # each chunk advances the fake wall clock
            return b"x"  # a byte per chunk, forever — never EOF

    response = _Trickle()

    with pytest.raises(qualserver.ServerUnavailableError, match="deadline"):
        qualserver._read_bounded_body(None, cast(Any, response), deadline=1.0, timing=timing)

    # The absolute 1 s deadline is not reset by frequent chunks: ~20 chunks of
    # 0.05 s advance the clock past it, far below the 8 MiB size cap.
    assert response.reads == 20


def test_body_read_tightens_socket_to_shrinking_budget() -> None:
    timing = _FakeTiming()

    class _RecordingSock:
        def __init__(self) -> None:
            self.timeouts: list[float] = []

        def settimeout(self, value: float) -> None:
            self.timeouts.append(value)

    class _Body:
        def __init__(self) -> None:
            self._chunks = [b"aa", b"bb", b""]

        def read1(self, amount: int) -> bytes:
            timing.now += 0.25  # each recv consumes wall-clock budget
            return self._chunks.pop(0)

    sock = _RecordingSock()
    body = qualserver._read_bounded_body(
        cast(Any, sock), cast(Any, _Body()), deadline=2.0, timing=timing
    )

    assert body == b"aabb"
    # The socket owned by the response is tightened to the decreasing budget
    # before every recv, so no single blocking read can outlast the deadline.
    assert sock.timeouts == [2.0, 1.75, 1.5]


def test_body_read_returns_full_body_within_deadline() -> None:
    timing = _FakeTiming()  # frozen: always within deadline

    class _Finite:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read1(self, amount: int) -> bytes:
            head, self._body = self._body[:amount], self._body[amount:]
            return head

    body = qualserver._read_bounded_body(
        None, cast(Any, _Finite(b"hello world")), deadline=1.0, timing=timing
    )
    assert body == b"hello world"


class _DeadlineConnection:
    """Minimal HTTPConnection stand-in whose body read trickles the fake clock."""

    instances: ClassVar[list[_DeadlineConnection]] = []

    def __init__(self, clock: _FakeTiming) -> None:
        self.sock = None
        self.closed = False
        self.response: Any = None
        self.captured_headers: dict[str, str] | None = None
        self._clock = clock
        _DeadlineConnection.instances.append(self)

    def connect(self) -> None:
        pass

    def request(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        del method, url, body
        self.captured_headers = dict(headers or {})

    def getresponse(self) -> Any:
        clock = self._clock
        self.response = _TricklingResponse(clock)
        return self.response

    def close(self) -> None:
        self.closed = True


class _TricklingResponse:
    def __init__(self, clock: _FakeTiming) -> None:
        self.status = 200
        self.closed = False
        self._clock = clock

    def read1(self, amount: int) -> bytes:
        self._clock.now += 0.5
        return b"x"  # trickle forever

    def close(self) -> None:
        self.closed = True


def _patch_deadline_connection(monkeypatch: pytest.MonkeyPatch, clock: _FakeTiming) -> None:
    _DeadlineConnection.instances = []
    monkeypatch.setattr(
        http.client, "HTTPConnection", lambda *args, **kwargs: _DeadlineConnection(clock)
    )


def test_health_rejects_trickling_body_within_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    timing = _FakeTiming()
    _patch_deadline_connection(monkeypatch, timing)

    # A trickling /health body advances the clock past the 1 s poll budget;
    # readiness treats it as not-ready (None) instead of blocking indefinitely.
    assert qualserver._health(12345, 1.0, "secret", timing) is None
    connection = _DeadlineConnection.instances[-1]
    assert connection.closed is True
    assert connection.response.closed is True  # response-owned socket released


def test_health_sends_bearer_key_and_reports_status(monkeypatch: pytest.MonkeyPatch) -> None:
    timing = _FakeTiming()

    class _FiniteResponse:
        status = 200
        closed = False

        def read1(self, amount: int) -> bytes:
            del amount
            return b""

        def close(self) -> None:
            self.closed = True

    class _FiniteConnection(_DeadlineConnection):
        def __init__(self) -> None:
            super().__init__(timing)

        def getresponse(self) -> Any:
            self.response = _FiniteResponse()
            return self.response

    _DeadlineConnection.instances = []
    monkeypatch.setattr(http.client, "HTTPConnection", lambda *args, **kwargs: _FiniteConnection())

    assert qualserver._health(12345, 1.0, "secret", timing) == 200
    connection = _DeadlineConnection.instances[-1]
    assert connection.captured_headers is not None
    assert connection.captured_headers["Authorization"] == "Bearer secret"


def test_chat_deadline_closes_connection_and_leaves_no_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timing = _FakeTiming()
    _patch_deadline_connection(monkeypatch, timing)

    process = _FakeProcess()  # returncode None -> handle.alive is True
    handle = qualserver.ServerHandle(
        run=_run(tmp_path),
        launch_number=1,
        argv=("server",),
        port=1,
        process=cast(Any, process),
        control=object(),
        stdout_capture=qualserver._Capture(io.BytesIO(), 10),
        stderr_capture=qualserver._Capture(io.BytesIO(), 10),
        api_key="secret",
        timing=timing,
    )
    threads_before = threading.active_count()

    with pytest.raises(qualserver.ServerUnavailableError, match="deadline"):
        handle.chat(
            [{"role": "user", "content": "hello"}],
            max_tokens=8,
            seed=1,
            timeout_s=1.0,
        )

    connection = _DeadlineConnection.instances[-1]
    assert connection.closed is True
    assert connection.response.closed is True
    assert connection.captured_headers is not None
    assert connection.captured_headers["Authorization"] == "Bearer secret"
    # The Connection: close response owns the socket and cannot be reached via
    # connection.close(); it must be closed explicitly on the deadline path.
    assert threading.active_count() == threads_before


def test_request_within_deadline_closes_response_on_failure() -> None:
    timing = _FakeTiming()
    response = _TricklingResponse(timing)

    class _Connection:
        sock = None

        def connect(self) -> None:
            pass

        def request(self, *args: object, **kwargs: object) -> None:
            pass

        def getresponse(self) -> Any:
            return response

    with pytest.raises(qualserver.ServerUnavailableError, match="deadline"):
        qualserver._request_within_deadline(
            cast(Any, _Connection()), "GET", "/health", 1.0, timing=timing
        )

    assert response.closed is True


def _serve_connection_close_stall(
    server: socket.socket, *, trickle_bytes: int, interval_s: float, stall_s: float
) -> None:
    try:
        conn, _ = server.accept()
    except OSError:
        return
    with conn:
        conn.settimeout(3.0)
        with contextlib.suppress(OSError):
            conn.recv(65536)  # consume the request (best effort)
        headers = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/json\r\n"
            b"Connection: close\r\n"
            b"Content-Length: 4096\r\n"
            b"\r\n"
        )
        with contextlib.suppress(OSError):
            conn.sendall(headers)
            for _ in range(trickle_bytes):
                conn.sendall(b"x")
                time.sleep(interval_s)
            time.sleep(stall_s)  # never deliver the promised body


def test_chat_enforces_hard_deadline_against_connection_close_stall(tmp_path: Path) -> None:
    # A real loopback peer that sends `Connection: close`, trickles a few body
    # bytes, then stalls. http.client hands the live socket to the response and
    # clears connection.sock, so the reader must tighten the *captured* socket or
    # a single blocking read runs for the full static timeout past the deadline.
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    thread = threading.Thread(
        target=_serve_connection_close_stall,
        args=(server,),
        kwargs={"trickle_bytes": 15, "interval_s": 0.02, "stall_s": 1.5},
        daemon=True,
    )
    thread.start()

    handle = qualserver.ServerHandle(
        run=_run(tmp_path),
        launch_number=1,
        argv=("server",),
        port=port,
        process=cast(Any, _FakeProcess()),  # returncode None -> alive
        control=object(),
        stdout_capture=qualserver._Capture(io.BytesIO(), 10),
        stderr_capture=qualserver._Capture(io.BytesIO(), 10),
    )

    timeout_s = 0.5
    start = time.monotonic()
    try:
        with pytest.raises(qualserver.ServerUnavailableError):
            handle.chat(
                [{"role": "user", "content": "hello"}],
                max_tokens=8,
                seed=1,
                timeout_s=timeout_s,
            )
        elapsed = time.monotonic() - start
    finally:
        server.close()
        thread.join(timeout=5)

    # The hard deadline holds: elapsed stays near timeout_s, well below the
    # timeout_s + static-socket-timeout overrun the unbounded read would incur,
    # and comfortably under one second.
    assert elapsed < timeout_s + 0.15
