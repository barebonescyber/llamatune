"""Night Shift GGUF discovery and content-duplicate grouping."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import gguf

from llamatune.model import ModelInspectionError, inspect_model
from llamatune.types import DiscoveredModel, ModelReport

_SHARD_RE = re.compile(
    r"^(?P<stem>.+?)(?P<separator>[-_])(?P<index>\d{1,5})-of-(?P<total>\d{1,5})\.gguf$",
    re.I,
)


@dataclass(frozen=True, slots=True)
class _Candidate:
    path: Path
    report: ModelReport
    shard_paths: tuple[Path, ...]
    tensor_sha: str | None
    payload_size: int


def _paths(models_dir: Path) -> tuple[Path, ...]:
    found: list[Path] = []
    seen_dirs: set[Path] = set()
    for root, dirs, files in os.walk(models_dir, followlinks=True):
        root_path = Path(root)
        try:
            real_root = root_path.resolve()
        except OSError:
            dirs[:] = []
            continue
        if real_root in seen_dirs:
            dirs[:] = []
            continue
        seen_dirs.add(real_root)
        kept: list[str] = []
        for name in dirs:
            try:
                if (root_path / name).resolve() not in seen_dirs:
                    kept.append(name)
            except OSError:
                continue
        dirs[:] = kept
        found.extend(root_path / name for name in files if Path(name).suffix.lower() == ".gguf")
    return tuple(sorted(found, key=lambda path: str(path)))


def _selected(path: Path, include: tuple[str, ...], exclude: tuple[str, ...]) -> bool:
    name = path.name
    return (not include or any(fnmatch.fnmatchcase(name, glob) for glob in include)) and not any(
        fnmatch.fnmatchcase(name, glob) for glob in exclude
    )


def _tensor_table_sha(paths: tuple[Path, ...]) -> str | None:
    rows: list[tuple[str, tuple[int, ...], str]] = []
    try:
        for path in paths:
            reader = gguf.GGUFReader(str(path), "r")
            for tensor in reader.tensors:
                rows.append(
                    (
                        str(tensor.name),
                        tuple(int(value) for value in tensor.shape),
                        str(tensor.tensor_type),
                    )
                )
    except (OSError, TypeError, ValueError, AttributeError):
        return None
    if not rows:
        return None
    encoded = json.dumps(sorted(rows), separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _metadata_key(candidate: _Candidate) -> tuple[object, ...]:
    report = candidate.report
    return (
        report.architecture,
        report.name,
        report.n_layer,
        report.expert_count,
        candidate.tensor_sha,
    )


def discover_models(
    models_dir: Path,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    *,
    duplicates: str = "one",
    full_hash: bool = False,
) -> tuple[DiscoveredModel, ...]:
    """Discover deterministic benchmark entries below ``models_dir``."""
    if duplicates not in {"one", "both"}:
        raise ValueError("duplicates must be 'one' or 'both'")
    paths = tuple(path for path in _paths(models_dir) if _selected(path, include, exclude))
    shard_groups: dict[tuple[Path, str], list[tuple[int, int, Path, str, str, str]]] = {}
    singles: list[Path] = []
    for path in paths:
        match = _SHARD_RE.match(path.name)
        if match is None:
            singles.append(path)
            continue
        index_text = match.group("index")
        total_text = match.group("total")
        key = (path.parent, match.group("stem"))
        shard_groups.setdefault(key, []).append(
            (
                int(index_text),
                int(total_text),
                path,
                match.group("separator"),
                index_text,
                total_text,
            )
        )

    entries: list[tuple[Path, tuple[Path, ...]]] = [(path, ()) for path in singles]
    for (_parent, stem), shards in sorted(shard_groups.items(), key=lambda item: str(item[0])):
        totals = {total for _index, total, _path, _separator, _index_text, _total_text in shards}
        if len(totals) != 1:
            warnings.warn(
                f"skipping inconsistent shard group {stem}; mixed declared totals: "
                + ", ".join(str(total) for total in sorted(totals)),
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        total = totals.pop()
        if total < 1 or any(index < 1 or index > total for index, *_rest in shards):
            warnings.warn(
                f"skipping inconsistent shard group {stem}; shard index outside declared total",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        members: dict[int, Path] = {}
        duplicate_index = False
        for index, _total, path, _separator, _index_text, _total_text in shards:
            if index in members:
                duplicate_index = True
                break
            members[index] = path
        if duplicate_index:
            warnings.warn(
                f"skipping inconsistent shard group {stem}; duplicate numeric shard index",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        missing = [index for index in range(1, total + 1) if index not in members]
        if missing:
            separator = shards[0][3]
            index_widths = {
                len(index_text)
                for _index, _total, _path, _separator, index_text, _total_text in shards
            }
            padded_index_widths = {
                len(index_text)
                for _index, _total, _path, _separator, index_text, _total_text in shards
                if len(index_text) > len(str(int(index_text)))
            }
            index_width = (
                next(iter(index_widths))
                if len(index_widths) == 1 and padded_index_widths == index_widths
                else None
            )
            total_texts = {
                total_text for _index, _total, _path, _separator, _index_text, total_text in shards
            }
            total_label = next(iter(total_texts)) if len(total_texts) == 1 else str(total)
            names = ", ".join(
                f"{stem}{separator}{index:0{index_width}d}-of-{total_label}.gguf"
                if index_width is not None
                else f"{stem}{separator}{index}-of-{total_label}.gguf"
                for index in missing
            )
            warnings.warn(
                f"skipping incomplete shard group; missing: {names}", RuntimeWarning, stacklevel=2
            )
            continue
        shard_paths = tuple(members[index] for index in range(1, total + 1))
        entries.append((members[1], shard_paths))

    candidates: list[_Candidate] = []
    seen_fingerprints: dict[str, Path] = {}
    for path, shard_paths in sorted(entries, key=lambda item: str(item[0])):
        try:
            report = inspect_model(path, full_hash=full_hash)
            physical_paths = shard_paths or (path,)
            payload_size = sum(member.stat().st_size for member in physical_paths)
        except (ModelInspectionError, OSError, ValueError) as exc:
            warnings.warn(f"skipping unreadable GGUF {path}: {exc}", RuntimeWarning, stacklevel=2)
            continue
        previous = seen_fingerprints.get(report.fingerprint)
        if previous is not None:
            warnings.warn(
                f"deduplicating identical GGUF {path}; kept {previous}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        seen_fingerprints[report.fingerprint] = path
        candidates.append(
            _Candidate(
                path=path,
                report=report,
                shard_paths=shard_paths,
                tensor_sha=_tensor_table_sha(physical_paths),
                payload_size=payload_size,
            )
        )

    groups: dict[tuple[object, ...], list[_Candidate]] = {}
    for candidate in candidates:
        if candidate.tensor_sha is not None:
            groups.setdefault(_metadata_key(candidate), []).append(candidate)
    grouped: dict[Path, tuple[str, bool]] = {}
    for group_members in groups.values():
        if len(group_members) < 2:
            continue
        compatible = [
            member
            for member in group_members
            if all(
                abs(member.payload_size - other.payload_size)
                / max(member.payload_size, other.payload_size)
                <= 0.01
                for other in group_members
            )
        ]
        if len(compatible) < 2:
            continue
        tensor_sha = compatible[0].tensor_sha
        if tensor_sha is None:
            continue
        group_key = tensor_sha[:16]
        merged = sorted(
            (member for member in compatible if not member.shard_paths),
            key=lambda member: str(member.path),
        )
        representative = merged[0] if merged else min(compatible, key=lambda m: str(m.path))
        for member in compatible:
            grouped[member.path] = (group_key, duplicates == "both" or member is representative)

    return tuple(
        DiscoveredModel(
            path=candidate.path,
            report=candidate.report,
            shard_paths=candidate.shard_paths,
            group_key=grouped.get(candidate.path, (None, True))[0],
            representative=grouped.get(candidate.path, (None, True))[1],
        )
        for candidate in sorted(candidates, key=lambda item: str(item.path))
    )
