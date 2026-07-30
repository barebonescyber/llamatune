"""Direct contract tests for the standalone fake llama-bench fixture."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _run(
    fake_bench_path: Path, model: Path, *args: str, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [str(fake_bench_path), "-m", str(model), *args, "-o", "json"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        timeout=5,
    )


def test_ambiguous_load_failure_only_affects_high_offload(
    fake_bench_path: Path, tiny_gguf: Path
) -> None:
    env = {
        "LLAMATUNE_FAKE_VRAM_MB": "1000",
        "LLAMATUNE_FAKE_MODEL_VRAM_MB": "4000",
        "LLAMATUNE_FAKE_FAIL_STYLE": "load",
    }
    failed = _run(fake_bench_path, tiny_gguf, "-ngl", "33", env=env)
    safe = _run(fake_bench_path, tiny_gguf, "-ngl", "0", env=env)
    assert failed.returncode == 1
    assert "failed to load model" in failed.stderr
    assert "allocate" not in failed.stderr.lower()
    assert "out of memory" not in failed.stderr.lower()
    assert safe.returncode == 0


def test_cuda_failure_style_emits_classifiable_abort_signature(
    fake_bench_path: Path, tiny_gguf: Path
) -> None:
    result = _run(
        fake_bench_path,
        tiny_gguf,
        "-ngl",
        "33",
        env={"LLAMATUNE_FAKE_VRAM_MB": "1000", "LLAMATUNE_FAKE_FAIL_STYLE": "cuda"},
    )
    assert result.returncode != 0
    assert "ggml-cuda.cu:106: CUDA error" in result.stderr


def test_kv_pressure_distinguishes_small_and_large_context(
    fake_bench_path: Path, tiny_gguf: Path
) -> None:
    env = {
        "LLAMATUNE_FAKE_VRAM_MB": "2000",
        "LLAMATUNE_FAKE_MODEL_VRAM_MB": "1000",
        "LLAMATUNE_FAKE_KV_MB_PER_1K": "200",
    }
    small = _run(fake_bench_path, tiny_gguf, "-ngl", "20", "-p", "512", "-n", "16", env=env)
    large = _run(fake_bench_path, tiny_gguf, "-ngl", "20", "-p", "8192", "-n", "16", env=env)
    assert small.returncode == 0
    assert large.returncode == 1


def test_genuine_failure_also_fails_cpu_placement(fake_bench_path: Path, tiny_gguf: Path) -> None:
    result = _run(
        fake_bench_path,
        tiny_gguf,
        "-ngl",
        "0",
        env={"LLAMATUNE_FAKE_GENUINE_FAIL": "1"},
    )
    assert result.returncode == 1
    assert "failed to load model" in result.stderr


def test_comma_list_emits_one_pair_per_combo(fake_bench_path: Path, tiny_gguf: Path) -> None:
    result = _run(fake_bench_path, tiny_gguf, "-t", "4,8,12", "-tb", "6", env={})
    assert result.returncode == 0
    entries = json.loads(result.stdout)
    assert len(entries) == 6
    assert {entry["n_threads"] for entry in entries} == {4, 8, 12}
    assert {entry["n_threads_batch"] for entry in entries} == {6}


def test_speed_scale_multiplies_prompt_and_generation_throughput(
    fake_bench_path: Path, tiny_gguf: Path
) -> None:
    normal = _run(fake_bench_path, tiny_gguf, "-r", "3", env={})
    scaled = _run(
        fake_bench_path,
        tiny_gguf,
        "-r",
        "3",
        env={"LLAMATUNE_FAKE_SPEED_SCALE": "0.5"},
    )
    assert normal.returncode == scaled.returncode == 0
    normal_entries = json.loads(normal.stdout)
    scaled_entries = json.loads(scaled.stdout)
    assert len(normal_entries) == len(scaled_entries) == 2
    for original, changed in zip(normal_entries, scaled_entries, strict=True):
        assert changed["avg_ts"] == original["avg_ts"] * 0.5
        assert changed["stddev_ts"] == original["stddev_ts"] * 0.5
