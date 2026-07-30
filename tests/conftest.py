"""Shared pytest fixtures for llamatune's test suite."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import gguf
import numpy as np
import pytest


def _write_tiny_gguf(path: Path, *, moe: bool, kv_heads: int | list[int] | None = None) -> None:
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_block_count(32)
    writer.add_name("tiny-test-model")
    if moe:
        writer.add_expert_count(8)
    if kv_heads is not None:
        writer.add_head_count(32)
        writer.add_embedding_length(4096)
        if isinstance(kv_heads, list):
            writer.add_array("llama.attention.head_count_kv", kv_heads)
        else:
            writer.add_head_count_kv(kv_heads)
    tensor = np.zeros((4,), dtype=np.float32)
    writer.add_tensor("dummy.weight", tensor)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture
def tiny_gguf(tmp_path: Path) -> Path:
    """A minimal, valid, non-MoE GGUF file (arch llama, 32 layers)."""
    path = tmp_path / "tiny.gguf"
    _write_tiny_gguf(path, moe=False)
    return path


@pytest.fixture
def tiny_moe_gguf(tmp_path: Path) -> Path:
    """A minimal, valid, MoE GGUF file (arch llama, 32 layers, 8 experts)."""
    path = tmp_path / "tiny-moe.gguf"
    _write_tiny_gguf(path, moe=True)
    return path


@pytest.fixture
def tiny_moe_gguf_kv_scalar(tmp_path: Path) -> Path:
    """A tiny MoE GGUF with scalar GQA KV metadata."""
    path = tmp_path / "tiny-moe-kv-scalar.gguf"
    _write_tiny_gguf(path, moe=True, kv_heads=8)
    return path


@pytest.fixture
def tiny_moe_gguf_kv_array(tmp_path: Path) -> Path:
    """A tiny hybrid-like GGUF with alternating KV-bearing layers."""
    path = tmp_path / "tiny-moe-kv-array.gguf"
    _write_tiny_gguf(path, moe=True, kv_heads=[8 if i % 2 == 0 else 0 for i in range(32)])
    return path


@pytest.fixture
def fake_bin_dir(tmp_path: Path) -> Path:
    """A directory containing a platform-native fake `llama-bench`."""
    source = Path(__file__).parent / "fixtures" / "fake_llama_bench.py"
    if sys.platform == "win32":
        script = tmp_path / "fake_llama_bench.py"
        shutil.copy2(source, script)
        wrapper = tmp_path / "llama-bench.cmd"
        wrapper.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    else:
        dest = tmp_path / "llama-bench"
        shutil.copy2(source, dest)
        dest.chmod(0o755)
    return tmp_path


@pytest.fixture
def fake_bench_path(fake_bin_dir: Path) -> Path:
    """The platform-native executable path for the fake benchmark."""
    name = "llama-bench.cmd" if sys.platform == "win32" else "llama-bench"
    return fake_bin_dir / name


@pytest.fixture(autouse=True)
def isolate_search_thermal_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep fake-benchmark tests independent of the host GPU temperature."""
    from llamatune import search

    monkeypatch.setattr(search, "_sample_gpu_state", lambda: None)
