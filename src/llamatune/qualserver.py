"""Supervised literal-loopback llama-server process and bounded HTTP client."""

from __future__ import annotations

import contextlib
import http.client
import json
import subprocess
import threading
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, BinaryIO, cast

from llamatune import executor

if TYPE_CHECKING:
    from llamatune.quality import QualityRun

_HOST = "127.0.0.1"
_HTTP_BODY_CAP = 8 * 1024 * 1024
_STDOUT_CAP = executor.STDOUT_CAP_BYTES
_STDERR_CAP = executor.STDERR_CAP_BYTES
_READ_CHUNK = 64 * 1024
_READINESS_POLL_S = 2.0

# Module seams keep readiness tests deterministic without changing the frozen API.
_monotonic = time.monotonic
_sleep = time.sleep


def _swap_monotonic(clock: Any) -> Any:
    global _monotonic
    previous = _monotonic
    _monotonic = clock
    return previous


class ServerError(RuntimeError):
    """Base class for supervised server failures."""


class ServerStartError(ServerError):
    """Raised when a server child exits or cannot become ready."""


class ServerUnavailableError(ServerError):
    """Raised when the live server exits or cannot be reached."""


class ServerProtocolError(ServerError):
    """Raised for malformed, unsuccessful, or oversized HTTP replies."""


class _Capture:
    def __init__(self, stream: BinaryIO, cap: int) -> None:
        self._stream = stream
        self._cap = cap
        self._data = bytearray()
        self._tail = bytearray()
        self._truncated = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._started = False

    def start(self) -> None:
        self._thread.start()
        self._started = True

    def _drain(self) -> None:
        while True:
            chunk = self._stream.read(_READ_CHUNK)
            if not chunk:
                return
            with self._lock:
                room = self._cap - len(self._data)
                if room > 0:
                    self._data.extend(chunk[:room])
                if len(chunk) > room:
                    self._truncated = True
                self._tail.extend(chunk)
                if len(self._tail) > 4096:
                    del self._tail[:-4096]

    def join(self) -> None:
        if self._started:
            self._thread.join()

    @property
    def data(self) -> bytes:
        with self._lock:
            return bytes(self._data)

    @property
    def tail(self) -> bytes:
        with self._lock:
            return bytes(self._tail)

    @property
    def truncated(self) -> bool:
        with self._lock:
            return self._truncated


def _endpoint_argv(argv: Sequence[str], port: int) -> tuple[str, ...]:
    result: list[str] = []
    index = 0
    host_seen = False
    while index < len(argv):
        value = argv[index]
        if value == "--host":
            if index + 1 >= len(argv) or argv[index + 1] != _HOST:
                raise ValueError("llama-server host must be literal 127.0.0.1")
            result.extend(("--host", _HOST))
            host_seen = True
            index += 2
            continue
        if value == "--port":
            if index + 1 >= len(argv):
                raise ValueError("--port requires a value")
            index += 2
            continue
        result.append(value)
        index += 1
    if not host_seen:
        result.extend(("--host", _HOST))
    result.extend(("--port", str(port)))
    return tuple(result)


def _free_loopback_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_HOST, 0))
        return int(sock.getsockname()[1])


def _tighten_socket(sock: Any, deadline: float) -> None:
    """Shrink ``sock``'s timeout to the time left before ``deadline``.

    Raises :class:`ServerUnavailableError` once the deadline has elapsed so no
    blocking socket operation can start past it. ``sock`` may be ``None`` (no
    live socket yet), in which case only the deadline is checked.
    """
    remaining = deadline - _monotonic()
    if remaining <= 0:
        raise ServerUnavailableError("llama-server response exceeded its deadline")
    if sock is not None:
        with contextlib.suppress(OSError):
            sock.settimeout(remaining)


def _read_bounded_body(sock: Any, response: http.client.HTTPResponse, deadline: float) -> bytes:
    """Read a size- and time-bounded response body against a hard deadline.

    ``sock`` is the socket the response actually reads from — captured before
    ``getresponse`` because ``http.client`` hands the socket to the response and
    clears ``connection.sock`` for ``Connection: close`` replies. Each iteration
    performs at most one recv (``read1``) after shrinking that socket's timeout
    to the remaining budget, so a single blocking read cannot exceed the
    deadline, and the decreasing monotonic deadline is re-checked between chunks.
    A peer that trickles bytes within each socket-inactivity window therefore
    cannot extend the total elapsed time; once the deadline passes the read is
    rejected and the caller closes the connection.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        _tighten_socket(sock, deadline)
        chunk = response.read1(_READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > _HTTP_BODY_CAP:
            raise ServerProtocolError("server response exceeded 8 MiB")
        chunks.append(chunk)
    return b"".join(chunks)


def _request_within_deadline(
    connection: http.client.HTTPConnection,
    method: str,
    url: str,
    deadline: float,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    """Perform one request/response cycle bounded by a single monotonic deadline.

    Connect, request write, header reception, and the complete bounded body read
    all run under the same shrinking deadline. The live socket is captured before
    ``getresponse`` (which clears ``connection.sock`` for ``Connection: close``
    replies while the response keeps the socket) so its timeout can be tightened
    across every phase, including each body-read chunk.

    Returns ``(status, body)``. The ``HTTPResponse`` — which owns the socket for
    ``Connection: close`` replies and is not reachable through
    ``connection.close()`` — is closed on every path, including a deadline or
    size-cap failure, so no response-owned descriptor survives the exception.
    """
    connection.connect()
    sock = connection.sock
    _tighten_socket(sock, deadline)
    connection.request(method, url, body=body, headers=headers or {})
    _tighten_socket(sock, deadline)
    response = connection.getresponse()
    try:
        raw = _read_bounded_body(sock, response, deadline)
        status = response.status
    finally:
        response.close()
    return status, raw


class ServerHandle:
    """One supervised llama-server child with an idempotent stop operation."""

    def __init__(
        self,
        *,
        run: QualityRun,
        launch_number: int,
        argv: tuple[str, ...],
        port: int,
        process: subprocess.Popen[bytes],
        control: Any,
        stdout_capture: _Capture,
        stderr_capture: _Capture,
    ) -> None:
        self._run = run
        self.launch_number = launch_number
        self.argv = argv
        self.port = port
        self._process = process
        self._control = control
        self._stdout_capture = stdout_capture
        self._stderr_capture = stderr_capture
        self._stopped = False
        self._lock = threading.Lock()

    @property
    def alive(self) -> bool:
        return not self._stopped and self._process.poll() is None

    @property
    def stderr_tail(self) -> str:
        return self._stderr_capture.tail.decode("utf-8", errors="replace")

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_tokens: int,
        seed: int,
        timeout_s: float,
    ) -> str:
        """Return one deterministic chat completion or raise a bounded failure."""
        if not self.alive:
            raise ServerUnavailableError("llama-server is not running")
        payload = {
            "messages": list(messages),
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "seed": seed,
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        # One hard monotonic deadline spans connect, request write, header
        # reception, and the complete bounded body read (never a per-recv
        # inactivity timeout that a trickling peer could reset indefinitely).
        deadline = _monotonic() + timeout_s
        connection = http.client.HTTPConnection(_HOST, self.port, timeout=max(0.01, timeout_s))
        try:
            status, raw = _request_within_deadline(
                connection,
                "POST",
                "/v1/chat/completions",
                deadline,
                body=body,
                headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            )
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            raise ServerUnavailableError(f"llama-server request failed: {exc}") from exc
        finally:
            connection.close()
        if status != 200:
            raise ServerProtocolError(f"llama-server returned HTTP {status}")
        try:
            document = json.loads(raw)
            choices = document["choices"]
            content = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ServerProtocolError("llama-server returned a malformed chat response") from exc
        if not isinstance(content, str):
            raise ServerProtocolError("llama-server response content is not a string")
        return content

    def stop(self) -> None:
        """Terminate the whole process group, reap it, and persist bounded logs once."""
        with self._lock:
            if self._stopped:
                return
            try:
                if self._process.poll() is None:
                    executor._terminate_group(self._process, self._control)
                try:
                    self._process.wait(timeout=15.0)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(OSError):
                        self._process.kill()
                    self._process.wait()
            finally:
                executor._close_process_control(self._control)
                self._stdout_capture.join()
                self._stderr_capture.join()
                base = f"server/{self.launch_number}"
                self._run.write_text(
                    f"{base}/stdout.log",
                    self._stdout_capture.data.decode("utf-8", errors="replace"),
                )
                self._run.write_text(
                    f"{base}/stderr.log",
                    self._stderr_capture.data.decode("utf-8", errors="replace"),
                )
            self._stopped = True


def _health(port: int, timeout_s: float) -> bool:
    # A non-positive budget means the enclosing startup deadline has no time
    # left: report not-ready rather than inflating it to a floor value.
    if timeout_s <= 0:
        return False
    # Same hard deadline as chat(): a trickling /health body cannot overrun the
    # per-poll budget and so cannot escape the enclosing startup deadline.
    deadline = _monotonic() + timeout_s
    connection = http.client.HTTPConnection(_HOST, port, timeout=timeout_s)
    try:
        status, _ = _request_within_deadline(connection, "GET", "/health", deadline)
        return status == 200
    except (OSError, TimeoutError, http.client.HTTPException, ServerError):
        return False
    finally:
        connection.close()


def start(
    run: QualityRun,
    argv: tuple[str, ...],
    *,
    start_timeout_s: float,
) -> ServerHandle:
    """Launch one server on a released ephemeral loopback port and await readiness."""
    if start_timeout_s <= 0:
        raise ValueError("start_timeout_s must be positive")
    port = _free_loopback_port()
    command = _endpoint_argv(argv, port)
    launch_number = run._allocate_server_launch()
    env = executor.build_child_env()
    run.write_json(
        f"server/{launch_number}/command.json",
        {"argv": list(command), "env_names": sorted(env)},
    )
    try:
        process, control = executor._spawn(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            **executor._popen_platform_kwargs(),
        )
    except OSError as exc:
        raise ServerStartError(f"could not start llama-server: {exc}") from exc
    if process.stdout is None or process.stderr is None:  # pragma: no cover - PIPE invariant
        try:
            executor._terminate_group(process, control)
            process.wait()
        finally:
            executor._close_process_control(control)
        raise ServerStartError("llama-server pipes were not created")
    stdout_capture = _Capture(cast(BinaryIO, process.stdout), _STDOUT_CAP)
    stderr_capture = _Capture(cast(BinaryIO, process.stderr), _STDERR_CAP)
    handle: ServerHandle | None = None
    try:
        stdout_capture.start()
        stderr_capture.start()
        handle = ServerHandle(
            run=run,
            launch_number=launch_number,
            argv=command,
            port=port,
            process=process,
            control=control,
            stdout_capture=stdout_capture,
            stderr_capture=stderr_capture,
        )
        run.append({"type": "server_start", "launch": launch_number, "argv": list(command)})
        deadline = _monotonic() + start_timeout_s
        while _monotonic() < deadline:
            if process.poll() is not None:
                raise ServerStartError(
                    f"llama-server exited during startup with code {process.returncode}"
                )
            remaining = deadline - _monotonic()
            if _health(port, min(2.0, remaining)):
                run.append({"type": "server_ready", "launch": launch_number, "port": port})
                return handle
            _sleep(min(_READINESS_POLL_S, max(0.0, deadline - _monotonic())))
        raise ServerStartError("llama-server did not become ready before the startup timeout")
    except BaseException:
        if handle is not None:
            handle.stop()
        else:
            try:
                if process.poll() is None:
                    executor._terminate_group(process, control)
                process.wait()
            finally:
                executor._close_process_control(control)
                stdout_capture.join()
                stderr_capture.join()
                base = f"server/{launch_number}"
                with contextlib.suppress(Exception):
                    run.write_text(
                        f"{base}/stdout.log",
                        stdout_capture.data.decode("utf-8", errors="replace"),
                    )
                    run.write_text(
                        f"{base}/stderr.log",
                        stderr_capture.data.decode("utf-8", errors="replace"),
                    )
        raise
