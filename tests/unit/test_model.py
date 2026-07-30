"""Unit tests for llamatune.model (GGUF inspection and fingerprinting)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import gguf
import numpy as np
import pytest

from llamatune.model import ModelInspectionError, inspect_model


def _write_shard(
    path: Path,
    *,
    split_no: int,
    split_count: int = 2,
    tensor_count: int = 2,
) -> None:
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_block_count(2)
    writer.add_name("tiny-sharded-model")
    writer.add_uint16("split.no", split_no)
    writer.add_uint16("split.count", split_count)
    writer.add_uint64("split.tensors.count", tensor_count)
    writer.add_tensor(f"blk.{split_no}.weight", np.full((4,), split_no, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _shards(tmp_path: Path) -> tuple[Path, Path]:
    first = tmp_path / "tiny-00001-of-00002.gguf"
    second = tmp_path / "tiny-00002-of-00002.gguf"
    _write_shard(first, split_no=0)
    _write_shard(second, split_no=1)
    return first, second


def test_inspect_model_basic_fields(tiny_gguf: Path) -> None:
    report = inspect_model(tiny_gguf)
    assert report.architecture == "llama"
    assert report.n_layer == 32
    assert report.ngl_all == 33
    assert report.expert_count == 0
    assert report.moe is False
    assert report.name == "tiny-test-model"
    assert report.path == tiny_gguf
    assert report.size_bytes == tiny_gguf.stat().st_size
    assert report.full_sha256 is None


def test_inspect_model_moe(tiny_moe_gguf: Path) -> None:
    report = inspect_model(tiny_moe_gguf)
    assert report.expert_count == 8
    assert report.moe is True
    assert report.expert_bytes == 0
    assert report.dense_bytes is not None
    assert report.kv_bytes_per_token_f16 is None
    assert report.n_kv_layers is None


def test_inspect_model_scalar_kv_geometry(tiny_moe_gguf_kv_scalar: Path) -> None:
    report = inspect_model(tiny_moe_gguf_kv_scalar)
    # 32 layers * 8 KV heads * (128 K + 128 V) * 2 bytes.
    assert report.kv_bytes_per_token_f16 == 32 * 8 * 256 * 2
    assert report.n_kv_layers == 32


def test_inspect_model_per_layer_kv_geometry(tiny_moe_gguf_kv_array: Path) -> None:
    report = inspect_model(tiny_moe_gguf_kv_array)
    # Only the 16 even-numbered layers use conventional KV attention.
    assert report.kv_bytes_per_token_f16 == 16 * 8 * 256 * 2
    assert report.n_kv_layers == 16


def test_inspect_model_full_hash(tiny_gguf: Path) -> None:
    report = inspect_model(tiny_gguf, full_hash=True)
    expected = hashlib.sha256(tiny_gguf.read_bytes()).hexdigest()
    assert report.full_sha256 == expected


def test_fingerprint_is_deterministic(tiny_gguf: Path) -> None:
    first = inspect_model(tiny_gguf)
    second = inspect_model(tiny_gguf)
    assert first.fingerprint == second.fingerprint


def test_fingerprint_differs_between_models(tiny_gguf: Path, tiny_moe_gguf: Path) -> None:
    a = inspect_model(tiny_gguf)
    b = inspect_model(tiny_moe_gguf)
    assert a.fingerprint != b.fingerprint


def test_inspect_model_aggregates_ordered_shards(tmp_path: Path) -> None:
    first, second = _shards(tmp_path)

    report = inspect_model(first, full_hash=True)

    assert report.path == first
    assert report.size_bytes == first.stat().st_size + second.stat().st_size
    assert report.dense_bytes == 32
    assert report.expert_bytes == 0
    assert report.full_sha256 is not None


def test_sharded_fingerprint_changes_when_later_shard_changes(tmp_path: Path) -> None:
    first, second = _shards(tmp_path)
    before = inspect_model(first)
    size_before = second.stat().st_size

    with second.open("r+b") as handle:
        handle.seek(-1, 2)
        last = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([last[0] ^ 0xFF]))

    after = inspect_model(first)
    assert second.stat().st_size == size_before
    assert after.fingerprint != before.fingerprint


def test_inspect_model_rejects_missing_shard(tmp_path: Path) -> None:
    first, _second = _shards(tmp_path)
    (tmp_path / "tiny-00002-of-00002.gguf").unlink()

    with pytest.raises(ModelInspectionError, match="missing or out-of-directory"):
        inspect_model(first)


def test_inspect_model_rejects_inconsistent_shard_metadata(tmp_path: Path) -> None:
    first = tmp_path / "tiny-00001-of-00002.gguf"
    second = tmp_path / "tiny-00002-of-00002.gguf"
    _write_shard(first, split_no=0)
    _write_shard(second, split_no=0)

    with pytest.raises(ModelInspectionError, match="inconsistent GGUF split metadata"):
        inspect_model(first)


def test_inspect_model_rejects_out_of_directory_shard(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    first = models / "tiny-00001-of-00002.gguf"
    linked_second = models / "tiny-00002-of-00002.gguf"
    outside = tmp_path / "outside.gguf"
    _write_shard(first, split_no=0)
    _write_shard(outside, split_no=1)
    try:
        linked_second.symlink_to(outside)
    except OSError:
        pytest.skip("file symlinks unavailable")

    with pytest.raises(ModelInspectionError, match="missing or out-of-directory"):
        inspect_model(first)


def test_inspect_model_requires_first_shard_path(tmp_path: Path) -> None:
    _first, second = _shards(tmp_path)

    with pytest.raises(ModelInspectionError, match="invalid GGUF split metadata"):
        inspect_model(second)


def test_inspect_model_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ModelInspectionError):
        inspect_model(tmp_path / "nope.gguf")


def test_inspect_model_not_a_gguf_file(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.gguf"
    bogus.write_bytes(b"not a gguf file" * 100)
    with pytest.raises(ModelInspectionError):
        inspect_model(bogus)


def test_inspect_model_missing_block_count(tmp_path: Path) -> None:
    path = tmp_path / "no-block-count.gguf"
    writer = gguf.GGUFWriter(str(path), "llama")
    tensor = np.zeros((4,), dtype=np.float32)
    writer.add_tensor("dummy.weight", tensor)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    with pytest.raises(ModelInspectionError, match="block_count"):
        inspect_model(path)


def test_inspect_model_no_name_is_none(tmp_path: Path) -> None:
    path = tmp_path / "no-name.gguf"
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_block_count(4)
    tensor = np.zeros((4,), dtype=np.float32)
    writer.add_tensor("dummy.weight", tensor)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    report = inspect_model(path)
    assert report.name is None


def test_inspect_model_splits_expert_dense_and_shared_expert_tensors(tmp_path: Path) -> None:
    path = tmp_path / "expert-split.gguf"
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_block_count(2)
    writer.add_expert_count(4)
    writer.add_tensor("blk.0.ffn_gate_exps.weight", np.zeros((8,), dtype=np.float32))
    writer.add_tensor("blk.0.ffn_up_exps.weight", np.zeros((4,), dtype=np.float32))
    writer.add_tensor("blk.0.ffn_down_shexp.weight", np.zeros((3,), dtype=np.float32))
    writer.add_tensor("blk.0.attn_q.weight", np.zeros((2,), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    report = inspect_model(path)
    assert report.expert_bytes == 48
    assert report.dense_bytes == 20
