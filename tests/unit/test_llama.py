"""Unit tests for llamatune.llama (binary discovery and capability probing)."""

from __future__ import annotations

import hashlib
import os
import stat
import sys
from pathlib import Path

import pytest

from llamatune import executor
from llamatune.llama import LlamaDiscoveryError, _flag_present, discover_llama

_EXPECTED_CAPABILITIES = frozenset({"fa", "mmp", "nkvo", "ctk", "ctv", "ncmoe", "r", "o"})


def _binary_path(directory: Path, name: str) -> Path:
    suffix = ".cmd" if sys.platform == "win32" else ""
    return directory / f"{name}{suffix}"


def test_discover_llama_via_llama_bin(fake_bin_dir: Path, fake_bench_path: Path) -> None:
    report = discover_llama(fake_bin_dir)
    assert report.bench_path == fake_bench_path
    assert report.cli_path is None
    assert report.server_path is None
    assert report.perplexity_path is None
    assert report.build_commit is None
    assert report.build_number is None
    assert report.capabilities >= _EXPECTED_CAPABILITIES
    assert "ot" not in report.capabilities  # fake bench does not support -ot


def test_discover_llama_help_sha256_matches_captured_bytes(
    fake_bin_dir: Path, fake_bench_path: Path
) -> None:
    report = discover_llama(fake_bin_dir)
    import subprocess

    proc = subprocess.run(  # noqa: S603
        [str(fake_bench_path), "--help"], capture_output=True, timeout=10, check=False
    )
    assert report.help_sha256 == hashlib.sha256(proc.stdout).hexdigest()
    assert report.bench_sha256 == hashlib.sha256(report.bench_path.read_bytes()).hexdigest()


def test_hash_binary_streams_full_file(tmp_path: Path) -> None:
    from llamatune.llama import hash_binary

    path = tmp_path / "binary"
    path.write_bytes(b"a" * (1024 * 1024 + 17))
    assert hash_binary(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_hash_binary_propagates_read_failure(tmp_path: Path) -> None:
    from llamatune.llama import hash_binary

    with pytest.raises(FileNotFoundError):
        hash_binary(tmp_path / "missing-binary")


def test_discover_llama_reads_help_from_stderr(
    fake_bin_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    help_bytes = b"usage: llama-bench -fa -nkvo -d"
    monkeypatch.setattr(
        executor,
        "run_probe",
        lambda argv, **kwargs: executor.ProbeResult(
            exit_code=1, stdout=b"", stderr=help_bytes, timed_out=False
        ),
    )

    report = discover_llama(fake_bin_dir)

    assert {"fa", "nkvo", "d"} <= report.capabilities
    assert report.help_sha256 == hashlib.sha256(help_bytes).hexdigest()


def test_discover_llama_finds_optional_binaries(fake_bin_dir: Path) -> None:
    for name in ("llama-cli", "llama-server", "llama-perplexity"):
        path = _binary_path(fake_bin_dir, name)
        path.write_text("@echo off\r\n" if sys.platform == "win32" else "#!/usr/bin/env python3\n")
        if sys.platform != "win32":
            path.chmod(path.stat().st_mode | stat.S_IXUSR)

    report = discover_llama(fake_bin_dir)
    assert report.cli_path == _binary_path(fake_bin_dir, "llama-cli")
    assert report.server_path == _binary_path(fake_bin_dir, "llama-server")
    assert report.perplexity_path == _binary_path(fake_bin_dir, "llama-perplexity")


def test_discover_llama_missing_bench_raises(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(LlamaDiscoveryError):
        discover_llama(empty_dir)


def test_discover_llama_non_executable_is_not_found(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    bench = bin_dir / "llama-bench"
    bench.write_text("#!/usr/bin/env python3\n")
    bench.chmod(0o644)  # not executable
    with pytest.raises(LlamaDiscoveryError):
        discover_llama(bin_dir)


def test_discover_llama_via_path(
    fake_bin_dir: Path, fake_bench_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", f"{fake_bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    report = discover_llama(None)
    assert report.bench_path == fake_bench_path


def test_flag_present_no_false_positive_inside_longer_flag() -> None:
    help_text = "-nkvo, --no-kv-offload <0|1>    disable KV cache offload\n"
    # "-o" must not match inside "--no-kv-offload"
    assert _flag_present(help_text, "-o") is False
    assert _flag_present(help_text, "-nkvo") is True


def test_flag_present_matches_standalone_token() -> None:
    help_text = "-o, --output <json>             output format\n"
    assert _flag_present(help_text, "-o") is True


def test_flag_present_absent_flag() -> None:
    assert _flag_present("no relevant flags here", "-ot") is False


def test_resolve_binary_windows_pathext(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bare = tmp_path / "llama-bench"
    bare.write_text("#!/usr/bin/env python3\n")
    wrapper = tmp_path / "llama-bench.cmd"
    wrapper.write_text("@echo off\r\n")
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setenv("PATHEXT", ".EXE;.CMD;.BAT")
    from llamatune.llama import _resolve_binary

    assert _resolve_binary("llama-bench", tmp_path) == wrapper
