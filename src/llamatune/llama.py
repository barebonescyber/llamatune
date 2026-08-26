"""llama.cpp binary discovery and capability probing.

Resolves `llama-bench` (required) and the optional `llama-cli` /
`llama-server` binaries from `--llama-bin` or PATH, then probes
`llama-bench --help` through the bounded, allowlisted executor contract
(DESIGN §7).
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path

from llamatune import executor
from llamatune.types import LlamaCppReport

_HELP_TIMEOUT_S = 10.0
_HELP_MAX_BYTES = 256 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024

#: Capability key -> the llama-bench flag whose presence in `--help` output
#: proves the flag is supported by this build.
_CAPABILITY_FLAGS: dict[str, str] = {
    "fa": "-fa",
    "mmp": "-mmp",
    "nkvo": "-nkvo",
    "ctk": "-ctk",
    "ctv": "-ctv",
    "ncmoe": "-ncmoe",
    "ot": "-ot",
    "tb": "-tb",
    "r": "-r",
    "o": "-o",
    "d": "-d",
    "ts": "-ts",
    "sm": "-sm",
}


class LlamaDiscoveryError(Exception):
    """Raised when llama-bench cannot be located or probed."""


def hash_binary(path: Path) -> str:
    """Return a streaming SHA-256 identity for a resolved binary."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _resolve_binary(name: str, llama_bin: Path | None) -> Path | None:
    if llama_bin is not None:
        if os.name == "nt":
            suffixes = [
                suffix.lower() for suffix in os.environ.get("PATHEXT", ".EXE;.CMD;.BAT").split(";")
            ]
            names = [f"{name}{suffix}" for suffix in suffixes if suffix]
        else:
            names = [name]
        for candidate_name in names:
            candidate = llama_bin / candidate_name
            if candidate.is_file() and (os.name == "nt" or os.access(candidate, os.X_OK)):
                return candidate
        return None
    found = shutil.which(name)
    return Path(found) if found else None


def _flag_present(help_text: str, flag: str) -> bool:
    """True if `flag` (e.g. "-fa") appears as its own token in `help_text`."""
    pattern = r"(?<![\w-])" + re.escape(flag) + r"(?![\w-])"
    return re.search(pattern, help_text) is not None


def discover_llama(llama_bin: Path | None = None) -> LlamaCppReport:
    """Resolve llama.cpp binaries and probe llama-bench's capabilities.

    Raises LlamaDiscoveryError if llama-bench cannot be found or `--help`
    cannot be run.
    """
    bench_path = _resolve_binary("llama-bench", llama_bin)
    if bench_path is None:
        location = str(llama_bin) if llama_bin is not None else "PATH"
        if llama_bin is None:
            msg = (
                f"llama-bench not found in {location}. Install llama.cpp, or pass "
                "--llama-bin DIR (the directory containing the binaries, "
                "not the binary itself)."
            )
        else:
            msg = (
                f"llama-bench not found in {location}. Pass --llama-bin DIR as the "
                "directory containing the binaries, not the binary itself."
            )
        raise LlamaDiscoveryError(msg)

    cli_path = _resolve_binary("llama-cli", llama_bin)
    server_path = _resolve_binary("llama-server", llama_bin)
    perplexity_path = _resolve_binary("llama-perplexity", llama_bin)

    result = executor.run_probe(
        [str(bench_path), "--help"],
        timeout_s=_HELP_TIMEOUT_S,
        max_output_bytes=_HELP_MAX_BYTES,
    )
    if result is None or result.timed_out:
        msg = (
            f"failed to run '{bench_path} --help'. Check that the file is executable, "
            "or pass --llama-bin DIR (the directory containing the binaries)."
        )
        raise LlamaDiscoveryError(msg)

    # Usage text normally goes to stdout; fall back to stderr for builds
    # that print --help there instead.
    help_bytes = result.stdout or result.stderr
    help_text = help_bytes.decode("utf-8", errors="replace")
    help_sha256 = hashlib.sha256(help_bytes).hexdigest()

    capabilities = frozenset(
        cap for cap, flag in _CAPABILITY_FLAGS.items() if _flag_present(help_text, flag)
    )

    return LlamaCppReport(
        bench_path=bench_path,
        cli_path=cli_path,
        server_path=server_path,
        perplexity_path=perplexity_path,
        capabilities=capabilities,
        help_sha256=help_sha256,
        build_commit=None,
        build_number=None,
        backends=None,
        bench_sha256=hash_binary(bench_path),
    )
