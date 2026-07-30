"""llama-bench command construction and output parsing (DESIGN §6, §8).

This module builds argv lists and parses bytes; it never executes a process
(that is ``executor.py``'s job). It may interpret llama-bench's own output
semantics (JSON shape, OOM stderr signatures) since that is measurement
parsing, not process control.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llamatune.types import TrialConfig

#: Case-insensitive OOM stderr signatures (DESIGN §8). The matched pattern
#: string is recorded on an ``oom`` outcome.
_OOM_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"failed to allocate",
        r"out of memory",
        r"cudaMalloc",
        r"kIOGPUCommandBuffer.*OutOfMemory",
        r"ggml_backend.*alloc.*fail",
        r"ErrorOutOfDeviceMemory",
        r"vk::OutOfDeviceMemoryError",
        r"hipMalloc",
    )
)

_CUDA_ERROR_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ggml-cuda\.cu:\d+",
        r"CUDA error",
        r"cuBLAS error",
        r"CUBLAS_STATUS_",
        r"HIP error",
    )
)

_GPU_RESOURCE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"failed to load model",
        r"failed to create context",
        r"unable to load model",
        r"PI_ERROR_OUT_OF_RESOURCES",
    )
)

_PPL_NUMBER = r"([0-9]+(?:\.[0-9]+)?)"
_PERPLEXITY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"Final\s+estimate\s*:\s*PPL\s*=\s*{_PPL_NUMBER}", re.IGNORECASE),
    re.compile(rf"\bPPL\s*=\s*{_PPL_NUMBER}", re.IGNORECASE),
    re.compile(rf"\bperplexity\s*[:=]\s*{_PPL_NUMBER}", re.IGNORECASE),
)


class BenchParseError(Exception):
    """Raised when llama-bench output cannot be parsed into a valid sample."""


def normalize_build_info(commit: str | None, number: int | None) -> tuple[str | None, int | None]:
    """Normalize llama.cpp's missing-version sentinel values."""
    normalized_commit = str(commit).strip() if commit is not None else ""
    if not normalized_commit or normalized_commit.lower() == "unknown":
        normalized_commit = ""
    normalized_number = number if number not in (None, 0) else None
    return (normalized_commit or None, normalized_number)


@dataclass(frozen=True, slots=True)
class BenchSample:
    """The pp/tg measurement extracted from one llama-bench JSON array."""

    pp_avg: float
    pp_stddev: float
    tg_avg: float
    tg_stddev: float
    pp_entry: dict[str, Any]
    tg_entry: dict[str, Any]
    config_fields: dict[str, Any]


def build_bench_argv(
    *,
    bench_path: Path,
    model_path: Path,
    pp: int,
    tg: int,
    reps: int,
    config: TrialConfig,
    capabilities: frozenset[str],
    depth: int | None = None,
) -> tuple[str, ...]:
    """Full argv for one trial: ``-m/-p/-n/-r/-o json`` plus the trial flags."""
    args = _base_argv(bench_path, model_path, pp, tg, reps)
    if depth is not None and "d" in capabilities:
        args.extend(["-d", str(depth)])
    args.extend(config.bench_args(capabilities))
    return tuple(args)


def build_bench_batch_argv(
    *,
    bench_path: Path,
    model_path: Path,
    pp: int,
    tg: int,
    reps: int,
    configs: Sequence[TrialConfig],
    capabilities: frozenset[str],
    depth: int | None = None,
) -> tuple[str, ...]:
    """Build one same-dimension comma-list llama-bench invocation."""
    if not configs:
        raise ValueError("batch requires at least one config")
    if len(configs) > 1 and any(config.tensor_split is not None for config in configs):
        raise ValueError("tensor-split configs cannot use comma-list batching")
    dictionaries = [config.to_dict() for config in configs]
    varying = {
        key
        for key in set().union(*(dictionary.keys() for dictionary in dictionaries))
        if len({dictionary.get(key) for dictionary in dictionaries}) > 1
    }
    if len(varying) > 1:
        raise ValueError("batch configs must vary in exactly one dimension")
    flag_maps: list[dict[str, str]] = []
    flag_order: list[str] = []
    for config in configs:
        config_args = config.bench_args(capabilities)
        mapping = dict(zip(config_args[::2], config_args[1::2], strict=True))
        flag_maps.append(mapping)
        if not flag_order:
            flag_order = list(config_args[::2])
    if any(set(mapping) != set(flag_order) for mapping in flag_maps):
        raise ValueError("batch configs emit different flag sets")
    batch_args = _base_argv(bench_path, model_path, pp, tg, reps)
    if depth is not None and "d" in capabilities:
        batch_args.extend(["-d", str(depth)])
    for flag in flag_order:
        values = [mapping[flag] for mapping in flag_maps]
        batch_args.extend([flag, values[0] if len(set(values)) == 1 else ",".join(values)])
    return tuple(batch_args)


def build_baseline_argv(
    *,
    bench_path: Path,
    model_path: Path,
    pp: int,
    tg: int,
    reps: int,
    ngl: int | None = None,
    capabilities: frozenset[str] = frozenset(),
    depth: int | None = None,
) -> tuple[str, ...]:
    """Baseline argv: no tuning flags (DESIGN §6), except an optional ``-ngl``
    used only for the OOM CPU fallback (``ngl=0``)."""
    args = _base_argv(bench_path, model_path, pp, tg, reps)
    if depth is not None and "d" in capabilities:
        args.extend(["-d", str(depth)])
    if ngl is not None:
        args.extend(["-ngl", str(ngl)])
    return tuple(args)


def build_context_probe_argv(
    *,
    bench_path: Path,
    model_path: Path,
    ctx: int,
    config: TrialConfig,
    capabilities: frozenset[str],
) -> tuple[str, ...]:
    """Build a one-repetition probe that forces allocation for ``ctx`` tokens."""
    args = _base_argv(bench_path, model_path, ctx, 16, 1)
    args.extend(config.bench_args(capabilities))
    return tuple(args)


def build_cli_context_argv(
    *,
    cli_path: Path,
    model_path: Path,
    config: TrialConfig,
    ctx: int,
    n_predict: int = 8,
    moe: bool,
) -> tuple[str, ...]:
    """Build an argv-only final context probe for ``llama-cli``."""
    args = [
        str(cli_path),
        "-m",
        str(model_path),
        "-ngl",
        str(config.gpu_layers),
        "-b",
        str(config.batch),
        "-ub",
        str(config.ubatch),
        "-t",
        str(config.threads),
    ]
    if moe and config.moe_cpu_layers > 0:
        args.extend(["--n-cpu-moe", str(config.moe_cpu_layers)])
    if config.flash_attn:
        args.extend(["-fa", "on"])
    if not config.mmap:
        args.append("--no-mmap")
    if config.no_kv_offload:
        args.append("--no-kv-offload")
    if config.cache_type_k != "f16":
        args.extend(["-ctk", config.cache_type_k])
    if config.cache_type_v != "f16":
        args.extend(["-ctv", config.cache_type_v])
    args.extend(
        [
            "-c",
            str(ctx),
            "-n",
            str(n_predict),
            "-p",
            "llamatune context probe",
            "--no-display-prompt",
        ]
    )
    return tuple(args)


def build_perplexity_argv(
    *,
    perplexity_path: Path,
    model_path: Path,
    config: TrialConfig | None,
    corpus: Path,
    ctx: int,
    capabilities: frozenset[str],
) -> tuple[str, ...]:
    """Build an argv-only llama-perplexity quality probe."""
    args = [str(perplexity_path), "-m", str(model_path)]
    if config is not None:
        args.extend(
            [
                "-ngl",
                str(config.gpu_layers),
                "-b",
                str(config.batch),
                "-ub",
                str(config.ubatch),
                "-t",
                str(config.threads),
            ]
        )
        if "ncmoe" in capabilities and config.moe_cpu_layers > 0 and config.ot_spec is None:
            args.extend(["--n-cpu-moe", str(config.moe_cpu_layers)])
        if "fa" in capabilities and config.flash_attn:
            args.extend(["-fa", "on"])
        if "mmp" in capabilities and not config.mmap:
            args.append("--no-mmap")
        if "nkvo" in capabilities and config.no_kv_offload:
            args.append("--no-kv-offload")
        if "ctk" in capabilities and config.cache_type_k != "f16":
            args.extend(["-ctk", config.cache_type_k])
        if "ctv" in capabilities and config.cache_type_v != "f16":
            args.extend(["-ctv", config.cache_type_v])
        if "tb" in capabilities and config.threads_batch is not None:
            args.extend(["-tb", str(config.threads_batch)])
        if "ot" in capabilities and config.ot_spec is not None:
            args.extend(["-ot", config.ot_spec])
    args.extend(["-c", str(ctx), "-f", str(corpus)])
    return tuple(args)


def parse_perplexity_output(text: bytes | str) -> float | None:
    """Extract llama-perplexity's final PPL estimate."""
    decoded = text.decode("utf-8", errors="replace") if isinstance(text, bytes) else text
    for pattern in _PERPLEXITY_PATTERNS:
        matches = pattern.findall(decoded)
        if matches:
            return float(matches[-1])
    return None


def _base_argv(bench_path: Path, model_path: Path, pp: int, tg: int, reps: int) -> list[str]:
    return [
        str(bench_path),
        "-m",
        str(model_path),
        "-p",
        str(pp),
        "-n",
        str(tg),
        "-r",
        str(reps),
        "-o",
        "json",
    ]


def _no_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            msg = f"duplicate key in bench JSON object: {key!r}"
            raise BenchParseError(msg)
        result[key] = value
    return result


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_positive_number(value: Any) -> bool:
    return _is_number(value) and value > 0


def _require_number(entry: dict[str, Any], key: str) -> float:
    if key not in entry:
        msg = f"bench entry missing required field {key!r}"
        raise BenchParseError(msg)
    value = entry[key]
    if not _is_number(value):
        msg = f"bench field {key!r} is not numeric: {value!r}"
        raise BenchParseError(msg)
    return float(value)


def parse_bench_output(raw: bytes | str) -> BenchSample:
    """Parse a llama-bench JSON array into a :class:`BenchSample`.

    The array must contain an entry with ``n_prompt > 0`` (pp sample) and one
    with ``n_gen > 0`` (tg sample); each must carry numeric ``avg_ts`` and
    ``stddev_ts``. Duplicate object keys are rejected. Anything else raises
    :class:`BenchParseError`.
    """
    samples = parse_bench_output_multi(raw)
    if len(samples) != 1:
        raise BenchParseError(f"bench output contains {len(samples)} configurations; expected one")
    return samples[0]


_CONFIG_ENTRY_FIELDS: tuple[str, ...] = (
    "n_gpu_layers",
    "n_cpu_moe",
    "flash_attn",
    "n_ubatch",
    "n_batch",
    "n_threads",
    "n_threads_batch",
    "use_mmap",
    "no_kv_offload",
    "type_k",
    "type_v",
    "tensor_buft_overrides",
)


def parse_bench_output_multi(raw: bytes | str) -> list[BenchSample]:
    """Parse one or more config-attributed pp/tg pairs from llama-bench JSON."""
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    try:
        data = json.loads(text, object_pairs_hook=_no_duplicate_keys)
    except json.JSONDecodeError as exc:
        msg = f"bench output is not valid JSON: {exc}"
        raise BenchParseError(msg) from exc

    if not isinstance(data, list):
        msg = "bench output is not a JSON array"
        raise BenchParseError(msg)

    grouped: dict[tuple[tuple[str, Any], ...], dict[str, dict[str, Any]]] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        config_fields = {key: entry[key] for key in _CONFIG_ENTRY_FIELDS if key in entry}
        key = tuple(sorted(config_fields.items()))
        pair = grouped.setdefault(key, {})
        if _is_positive_number(entry.get("n_prompt")):
            pair.setdefault("pp", entry)
        if _is_positive_number(entry.get("n_gen")):
            pair.setdefault("tg", entry)

    if not grouped:
        msg = "bench output missing a pp (n_prompt>0) or tg (n_gen>0) entry"
        raise BenchParseError(msg)
    samples: list[BenchSample] = []
    for key, pair in grouped.items():
        pp_entry = pair.get("pp")
        tg_entry = pair.get("tg")
        if pp_entry is None or tg_entry is None:
            msg = "bench output missing a pp (n_prompt>0) or tg (n_gen>0) entry"
            raise BenchParseError(msg)
        samples.append(
            BenchSample(
                pp_avg=_require_number(pp_entry, "avg_ts"),
                pp_stddev=_require_number(pp_entry, "stddev_ts"),
                tg_avg=_require_number(tg_entry, "avg_ts"),
                tg_stddev=_require_number(tg_entry, "stddev_ts"),
                pp_entry=pp_entry,
                tg_entry=tg_entry,
                config_fields=dict(key),
            )
        )
    return samples


def classify_failure(stderr_text: str) -> str | None:
    """Return the factual stderr classification, if a signature matches.

    OOM signatures take precedence over CUDA and ambiguous GPU resource signatures. Whether
    an ambiguous resource failure was actually caused by GPU placement is established
    behaviorally by the search engine, not here.
    """
    for pattern in _OOM_PATTERNS:
        if pattern.search(stderr_text) is not None:
            return "oom"
    for pattern in _CUDA_ERROR_PATTERNS:
        if pattern.search(stderr_text) is not None:
            return "cuda_error"
    for pattern in _GPU_RESOURCE_PATTERNS:
        if pattern.search(stderr_text) is not None:
            return "gpu_resource"
    return None


def failure_pattern(stderr_text: str, classification: str) -> str | None:
    """Return the first matching pattern for a known failure classification."""
    patterns = {
        "oom": _OOM_PATTERNS,
        "cuda_error": _CUDA_ERROR_PATTERNS,
        "gpu_resource": _GPU_RESOURCE_PATTERNS,
    }.get(classification, ())
    for pattern in patterns:
        if pattern.search(stderr_text) is not None:
            return pattern.pattern
    return None


def detect_oom(stderr_text: str) -> str | None:
    """Return the matched OOM pattern string, or None (DESIGN §8)."""
    if classify_failure(stderr_text) != "oom":
        return None
    return failure_pattern(stderr_text, "oom")


def resolved_config(
    entry: dict[str, Any], *, force_gpu_layers: int | None = None, multi_gpu: bool = False
) -> TrialConfig:
    """Reconstruct the resolved default TrialConfig from a baseline JSON entry.

    ``force_gpu_layers`` overrides the parsed ``n_gpu_layers`` (used when no
    GPU is present, where the default offload must be pinned to 0).
    """
    gpu_layers = (
        force_gpu_layers if force_gpu_layers is not None else int(entry.get("n_gpu_layers", 0))
    )
    return TrialConfig(
        gpu_layers=gpu_layers,
        moe_cpu_layers=0,
        flash_attn=bool(entry.get("flash_attn", False)),
        ubatch=int(entry.get("n_ubatch", 512)),
        batch=int(entry.get("n_batch", 2048)),
        threads=int(entry.get("n_threads", 1)),
        mmap=bool(entry.get("use_mmap", True)),
        no_kv_offload=bool(entry.get("no_kv_offload", False)),
        cache_type_k=str(entry.get("type_k", "f16")),
        cache_type_v=str(entry.get("type_v", "f16")),
        tensor_split=_parse_tensor_split(entry.get("tensor_split")) if multi_gpu else None,
        split_mode=(
            str(entry["split_mode"]) if multi_gpu and entry.get("split_mode") is not None else None
        ),
    )


def _parse_tensor_split(value: object) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts: tuple[object, ...] = tuple(value.split(","))
    elif isinstance(value, (list, tuple)):
        parts = tuple(value)
    else:
        return None
    try:
        return tuple(float(str(part)) for part in parts)
    except (TypeError, ValueError):
        return None
