#!/usr/bin/env python3
"""Standalone stdlib-only fake llama-server for quality integration tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socketserver
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class _LiteralLoopbackServer(ThreadingHTTPServer):
    """HTTP server that never performs reverse DNS for its literal host."""

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)


def _float_env(name: str, default: float = 0.0) -> float:
    try:
        return max(0.0, float(os.environ.get(name, default)))
    except ValueError:
        return default


def _int_env(name: str, default: int = 0) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except ValueError:
        return default


def _script() -> dict[str, str]:
    path = os.environ.get("LLAMATUNE_FAKE_SRV_SCRIPT")
    if not path:
        return {"*": "fixture response"}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError("LLAMATUNE_FAKE_SRV_SCRIPT must be a JSON string map")
    return value


def _last_user(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("-m", dest="model")
    parser.add_argument("-c", dest="ctx", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--api-key")
    # Accept the complete quality config-to-server flag surface.  These
    # values do not alter fixture behavior, but spelling them out prevents
    # argparse from treating multi-character short options such as ``-ctk``
    # as ``-c tk`` before ``parse_known_args`` can ignore them.
    parser.add_argument("-ngl")
    parser.add_argument("--n-cpu-moe")
    parser.add_argument("-fa")
    parser.add_argument("-b")
    parser.add_argument("-ub")
    parser.add_argument("-t")
    parser.add_argument("-ctk")
    parser.add_argument("-ctv")
    parser.add_argument("-tb")
    parser.add_argument("-ot")
    parser.add_argument("-ts")
    parser.add_argument("-sm")
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--no-kv-offload", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def main() -> int:
    args = _arguments()
    script = _script()
    started = time.monotonic()
    ready_delay = _float_env("LLAMATUNE_FAKE_SRV_READY_DELAY_S")
    hang_s = _float_env("LLAMATUNE_FAKE_SRV_HANG_S")
    die_after = _int_env("LLAMATUNE_FAKE_SRV_DIE_AFTER_N")
    never_ready = os.environ.get("LLAMATUNE_FAKE_SRV_NEVER_READY") == "1"
    malformed = os.environ.get("LLAMATUNE_FAKE_SRV_MALFORMED") == "1"
    completions = 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, message_format: str, *values: object) -> None:
            del message_format, values

        def _authorized(self) -> bool:
            if not args.api_key:
                return True
            header = self.headers.get("Authorization", "")
            return header == f"Bearer {args.api_key}"

        def _reject(self) -> None:
            body = b'{"error":{"message":"Invalid API key"}}'
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict[str, Any] | None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(min(length, 8 * 1024 * 1024 + 1))
                value = json.loads(raw)
            except (ValueError, json.JSONDecodeError):
                return None
            return value if isinstance(value, dict) else None

        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_error(404)
                return
            if not self._authorized():
                self._reject()
                return
            if never_ready or time.monotonic() - started < ready_delay:
                self.send_error(503)
                return
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            nonlocal completions
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            if not self._authorized():
                self._reject()
                return
            payload = self._body()
            if payload is None:
                self.send_error(400)
                return
            if hang_s:
                time.sleep(hang_s)
            completions += 1
            if malformed:
                body = b"not-json"
            else:
                user = _last_user(payload)
                key = hashlib.sha256(user.encode()).hexdigest()[:12]
                response = script.get(key, script.get("*", "fixture response"))
                body = json.dumps(
                    {"choices": [{"message": {"role": "assistant", "content": response}}]}
                ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            if die_after and completions >= die_after:
                os._exit(17)

    server = _LiteralLoopbackServer((args.host, args.port), Handler)
    try:
        server.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"fake llama-server: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
