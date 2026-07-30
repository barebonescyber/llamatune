"""GGUF model inspection and fingerprinting (DESIGN §5).

Reads only GGUF header/key-value metadata with the official `gguf` package;
tensor data is never loaded. A model file is data -- it is never executed.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gguf

from llamatune.types import ModelReport

if TYPE_CHECKING:
    from gguf import ReaderField

_MIB = 1024 * 1024
_EXPERT_TENSOR_MARKERS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
_SHARD_RE = re.compile(
    r"^(?P<stem>.+?)(?P<separator>[-_])(?P<index>\d{1,5})-of-(?P<total>\d{1,5})\.gguf$",
    re.IGNORECASE,
)


class ModelInspectionError(Exception):
    """Raised when a model file cannot be read or is missing required metadata."""


def _field_str(field: ReaderField | None) -> str | None:
    if field is None:
        return None
    value = field.contents()
    return None if value is None else str(value)


def _field_int(field: ReaderField | None) -> int | None:
    if field is None:
        return None
    value = field.contents()
    return None if value is None else int(value)


def _field_ints(field: ReaderField | None) -> tuple[int, ...] | None:
    """Return a scalar or GGUF array metadata value as integers."""
    if field is None:
        return None
    value: Any = field.contents()
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        try:
            return tuple(int(item) for item in value)
        except (TypeError, ValueError, OverflowError):
            return None
    # numpy arrays are iterable but do not implement collections.abc.Sequence.
    if hasattr(value, "__iter__"):
        try:
            return tuple(int(item) for item in value)
        except (TypeError, ValueError, OverflowError):
            return None
    try:
        return (int(value),)
    except (TypeError, ValueError, OverflowError):
        return None


def _per_layer(values: tuple[int, ...] | None, n_layer: int) -> tuple[int, ...] | None:
    if values is None or not values:
        return None
    if len(values) == 1:
        return values * n_layer
    if len(values) != n_layer:
        return None
    return values


def _kv_geometry(
    reader: gguf.GGUFReader, architecture: str, n_layer: int
) -> tuple[int, int] | None:
    """Derive total f16 KV bytes/token and the number of KV-bearing layers."""
    prefix = f"{architecture}.attention"
    kv_heads = _per_layer(_field_ints(reader.get_field(f"{prefix}.head_count_kv")), n_layer)
    if kv_heads is None:
        return None

    head_counts = _per_layer(_field_ints(reader.get_field(f"{prefix}.head_count")), n_layer)
    embedding_lengths = _per_layer(
        _field_ints(reader.get_field(f"{architecture}.embedding_length")), n_layer
    )
    key_lengths = _per_layer(_field_ints(reader.get_field(f"{prefix}.key_length")), n_layer)
    value_lengths = _per_layer(_field_ints(reader.get_field(f"{prefix}.value_length")), n_layer)

    total_elements = 0
    kv_layers = 0
    for layer, n_kv_heads in enumerate(kv_heads):
        if n_kv_heads <= 0:
            continue
        key_length = key_lengths[layer] if key_lengths is not None else None
        value_length = value_lengths[layer] if value_lengths is not None else None
        if key_length is None or value_length is None:
            if head_counts is None or embedding_lengths is None or head_counts[layer] <= 0:
                return None
            default_length = embedding_lengths[layer] // head_counts[layer]
            key_length = default_length if key_length is None else key_length
            value_length = default_length if value_length is None else value_length
        if key_length <= 0 or value_length <= 0:
            return None
        total_elements += n_kv_heads * (key_length + value_length)
        kv_layers += 1
    return total_elements * 2, kv_layers


def _fast_fingerprint(
    path: Path,
    *,
    size_bytes: int,
    architecture: str,
    n_layer: int,
    expert_count: int,
    name: str | None,
) -> str:
    """SHA-256 over (first 1 MiB + last 1 MiB + file size + key metadata values)."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        hasher.update(fh.read(_MIB))
        if size_bytes > _MIB:
            fh.seek(max(0, size_bytes - _MIB))
            hasher.update(fh.read(_MIB))
    hasher.update(str(size_bytes).encode("utf-8"))
    hasher.update(architecture.encode("utf-8"))
    hasher.update(str(n_layer).encode("utf-8"))
    hasher.update(str(expert_count).encode("utf-8"))
    hasher.update((name or "").encode("utf-8"))
    return hasher.hexdigest()


def _fast_sharded_fingerprint(
    paths: tuple[Path, ...],
    *,
    sizes: tuple[int, ...],
    architecture: str,
    n_layer: int,
    expert_count: int,
    name: str | None,
) -> str:
    """Hash bounded content from every shard without depending on its location."""
    hasher = hashlib.sha256(b"llamatune-sharded-fast-v1\0")
    hasher.update(len(paths).to_bytes(4, "big"))
    for index, (path, size) in enumerate(zip(paths, sizes, strict=True)):
        hasher.update(index.to_bytes(4, "big"))
        hasher.update(size.to_bytes(16, "big"))
        with path.open("rb") as fh:
            hasher.update(fh.read(_MIB))
            if size > _MIB:
                fh.seek(max(0, size - _MIB))
                hasher.update(fh.read(_MIB))
    hasher.update(str(sum(sizes)).encode("utf-8"))
    hasher.update(architecture.encode("utf-8"))
    hasher.update(str(n_layer).encode("utf-8"))
    hasher.update(str(expert_count).encode("utf-8"))
    hasher.update((name or "").encode("utf-8"))
    return hasher.hexdigest()


def _full_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_MIB), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _full_sharded_sha256(paths: tuple[Path, ...], sizes: tuple[int, ...]) -> str:
    """Hash all shard bytes with explicit boundaries and stable numeric order."""
    hasher = hashlib.sha256(b"llamatune-sharded-full-v1\0")
    hasher.update(len(paths).to_bytes(4, "big"))
    for index, (path, size) in enumerate(zip(paths, sizes, strict=True)):
        hasher.update(index.to_bytes(4, "big"))
        hasher.update(size.to_bytes(16, "big"))
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(_MIB), b""):
                hasher.update(chunk)
    return hasher.hexdigest()


def _tensor_weight_split(
    readers: Sequence[gguf.GGUFReader],
) -> tuple[int | None, int | None]:
    """Return expert and non-expert tensor bytes without touching tensor payloads."""
    try:
        expert_bytes = 0
        dense_bytes = 0
        for reader in readers:
            for tensor in reader.tensors:
                size = int(tensor.n_bytes)
                name = str(tensor.name).lower()
                if any(marker in name for marker in _EXPERT_TENSOR_MARKERS):
                    expert_bytes += size
                else:
                    # Shared-expert ``*_shexp`` tensors intentionally land here.
                    dense_bytes += size
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None, None
    return expert_bytes, dense_bytes


def _sharded_readers(
    path: Path, first_reader: gguf.GGUFReader
) -> tuple[tuple[Path, ...], tuple[gguf.GGUFReader, ...]]:
    """Resolve and validate an ordered GGUF shard set from its first member."""
    split_count = _field_int(first_reader.get_field("split.count"))
    if split_count is None:
        return (path,), (first_reader,)

    split_no = _field_int(first_reader.get_field("split.no"))
    tensor_count = _field_int(first_reader.get_field("split.tensors.count"))
    if split_count < 1 or split_no != 0 or tensor_count is None or tensor_count < 0:
        msg = f"invalid GGUF split metadata in {path}"
        raise ModelInspectionError(msg)

    match = _SHARD_RE.match(path.name)
    if match is None or int(match.group("index")) != 1 or int(match.group("total")) != split_count:
        msg = (
            f"sharded GGUF must use its first '<name>-00001-of-{split_count:05d}.gguf' "
            f"member: {path}"
        )
        raise ModelInspectionError(msg)

    index_width = len(match.group("index"))
    total_label = match.group("total")
    prefix = f"{match.group('stem')}{match.group('separator')}"
    parent = path.parent.resolve()
    paths = tuple(
        path.parent / f"{prefix}{index:0{index_width}d}-of-{total_label}.gguf"
        for index in range(1, split_count + 1)
    )

    readers: list[gguf.GGUFReader] = []
    for expected_no, member in enumerate(paths):
        try:
            if not member.is_file() or member.resolve().parent != parent:
                msg = f"missing or out-of-directory GGUF shard: {member}"
                raise ModelInspectionError(msg)
            reader = first_reader if expected_no == 0 else gguf.GGUFReader(member)
        except (OSError, ValueError) as exc:
            msg = f"failed to read GGUF shard {member}: {exc}"
            raise ModelInspectionError(msg) from exc
        if (
            _field_int(reader.get_field("split.no")) != expected_no
            or _field_int(reader.get_field("split.count")) != split_count
            or _field_int(reader.get_field("split.tensors.count")) != tensor_count
        ):
            msg = f"inconsistent GGUF split metadata in {member}"
            raise ModelInspectionError(msg)
        readers.append(reader)

    actual_tensors = sum(len(reader.tensors) for reader in readers)
    if actual_tensors != tensor_count:
        msg = (
            f"inconsistent GGUF shard tensor count: expected {tensor_count}, found {actual_tensors}"
        )
        raise ModelInspectionError(msg)
    return paths, tuple(readers)


def inspect_model(path: Path, *, full_hash: bool = False) -> ModelReport:
    """Read GGUF metadata for `path` and return a ModelReport.

    Raises ModelInspectionError if the file is missing, unreadable, not a
    valid GGUF file, or lacks the required `general.architecture` /
    `<arch>.block_count` metadata keys.
    """
    if not path.is_file():
        msg = f"model file not found: {path}"
        raise ModelInspectionError(msg)

    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        msg = f"cannot stat model file: {path}"
        raise ModelInspectionError(msg) from exc

    try:
        reader = gguf.GGUFReader(path)
    except (OSError, ValueError) as exc:
        msg = f"failed to read GGUF file {path}: {exc}"
        raise ModelInspectionError(msg) from exc

    paths, readers = _sharded_readers(path, reader)
    reader = readers[0]

    architecture = _field_str(reader.get_field("general.architecture"))
    if architecture is None:
        msg = f"GGUF file {path} is missing required key 'general.architecture'"
        raise ModelInspectionError(msg)

    n_layer = _field_int(reader.get_field(f"{architecture}.block_count"))
    if n_layer is None:
        msg = f"GGUF file {path} is missing required key '{architecture}.block_count'"
        raise ModelInspectionError(msg)

    expert_count = _field_int(reader.get_field(f"{architecture}.expert_count")) or 0
    name = _field_str(reader.get_field("general.name"))

    try:
        sizes = tuple(member.stat().st_size for member in paths)
    except OSError as exc:
        msg = f"cannot stat model shard: {exc}"
        raise ModelInspectionError(msg) from exc
    size_bytes = sum(sizes)
    if len(paths) == 1:
        fingerprint = _fast_fingerprint(
            path,
            size_bytes=size_bytes,
            architecture=architecture,
            n_layer=n_layer,
            expert_count=expert_count,
            name=name,
        )
        full_sha256 = _full_file_sha256(path) if full_hash else None
    else:
        fingerprint = _fast_sharded_fingerprint(
            paths,
            sizes=sizes,
            architecture=architecture,
            n_layer=n_layer,
            expert_count=expert_count,
            name=name,
        )
        full_sha256 = _full_sharded_sha256(paths, sizes) if full_hash else None
    expert_bytes, dense_bytes = _tensor_weight_split(readers)
    kv_geometry = _kv_geometry(reader, architecture, n_layer)

    return ModelReport(
        path=path,
        size_bytes=size_bytes,
        architecture=architecture,
        n_layer=n_layer,
        ngl_all=n_layer + 1,
        expert_count=expert_count,
        moe=expert_count > 0,
        name=name,
        fingerprint=fingerprint,
        full_sha256=full_sha256,
        expert_bytes=expert_bytes,
        dense_bytes=dense_bytes,
        kv_bytes_per_token_f16=kv_geometry[0] if kv_geometry is not None else None,
        n_kv_layers=kv_geometry[1] if kv_geometry is not None else None,
    )
