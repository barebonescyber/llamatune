"""Shared runtime flags for recommendation reference commands and exports."""

from __future__ import annotations

from llamatune.types import TrialConfig


def runtime_flags(config: TrialConfig, *, moe: bool = True) -> list[str]:
    flags = [
        "-ngl",
        str(config.gpu_layers),
        "-b",
        str(config.batch),
        "-ub",
        str(config.ubatch),
        "-t",
        str(config.threads),
    ]
    if config.ot_spec is not None:
        flags += ["-ot", config.ot_spec]
    elif moe and config.moe_cpu_layers > 0:
        flags += ["--n-cpu-moe", str(config.moe_cpu_layers)]
    if config.threads_batch is not None:
        flags += ["-tb", str(config.threads_batch)]
    if config.flash_attn:
        flags += ["-fa", "on"]
    if not config.mmap:
        flags.append("--no-mmap")
    if config.no_kv_offload:
        flags.append("--no-kv-offload")
    if config.cache_type_k != "f16":
        flags += ["-ctk", config.cache_type_k]
    if config.cache_type_v != "f16":
        flags += ["-ctv", config.cache_type_v]
    if config.tensor_split is not None:
        flags += ["-ts", ",".join(f"{value:g}" for value in config.tensor_split)]
    if config.split_mode is not None:
        flags += ["-sm", config.split_mode]
    return flags
