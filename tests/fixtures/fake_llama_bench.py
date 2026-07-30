#!/usr/bin/env python3
"""Deterministic fake `llama-bench` used by llamatune's test suite.

Implements the CLI surface, environment knobs, and performance/OOM model
specified in DESIGN.md section 15.1. Stdlib only, and intentionally
independent of the `llamatune` package: tests copy this file standalone into
a temp directory as an executable named `llama-bench`.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import statistics
import sys
import time

_HELP_TEXT = """\
usage: llama-bench [options]

fake llama-bench (llamatune test fixture)

options:
  -m, --model FNAME              model path (required)
  -p, --n-prompt N                number of prompt tokens
  -n, --n-gen N                   number of generated tokens
  -d, --n-depth N                 KV depth before measurement
  -b, --batch-size N              logical batch size
  -ub, --ubatch-size N            physical batch size
  -t, --threads N                 number of threads
  -tb N                           prompt-processing threads
  -ngl, --n-gpu-layers N          number of layers to offload to the GPU
  -fa, --flash-attn <0|1>         use flash attention
  -mmp, --mmap <0|1>              use mmap for model loading
  -nkvo, --no-kv-offload <0|1>    disable KV cache offload
  -ctk, --cache-type-k TYPE       KV cache data type for K
  -ctv, --cache-type-v TYPE       KV cache data type for V
  -ncmoe, --n-cpu-moe N           number of MoE layers to keep on the CPU
  -r, --repetitions N             number of repetitions per test
  -o, --output <json>             output format (only json supported)
  --delay N                       fake-bench only: sleep N seconds first
  -h, --help                      show this help and exit
"""

_UB_FACTORS: dict[int, float] = {
    128: 0.80,
    256: 0.92,
    512: 1.00,
    1024: 0.97,
    2048: 0.95,
    4096: 0.94,
}
_B_FACTORS: dict[int, float] = {512: 0.97, 1024: 0.99, 2048: 1.0, 4096: 1.0}

_INT_FLAGS: dict[str, str] = {
    "-p": "p",
    "-n": "n",
    "-d": "d",
    "-b": "b",
    "-ub": "ub",
    "-t": "t",
    "-tb": "tb",
    "-ngl": "ngl",
    "-fa": "fa",
    "-mmp": "mmp",
    "-nkvo": "nkvo",
    "-ncmoe": "ncmoe",
    "-r": "r",
}


def _die(message: str) -> int:
    sys.stderr.write(message + "\n")
    return 2


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    return int(raw)


def _nearest_factor(value: int, table: dict[int, float]) -> float:
    nearest_key = min(table, key=lambda k: abs(k - value))
    return table[nearest_key]


def _canonical_argv(params: dict[str, object]) -> str:
    return ",".join(f"{key}={params[key]}" for key in sorted(params))


def _jittered_samples(base_value: float, canonical: str, rep_count: int) -> list[float]:
    samples = []
    for rep in range(rep_count):
        payload = f"{canonical}|{rep}".encode()
        digest = hashlib.sha256(payload).digest()
        u = int.from_bytes(digest[:4], "big") / 2**32
        jitter = 1 + (u - 0.5) * 0.02
        samples.append(base_value * jitter)
    return samples


def _entry(
    *,
    n_prompt: int,
    n_gen: int,
    avg_ts: float,
    stddev_ts: float,
    model_filename: str,
    n_batch: int,
    n_ubatch: int,
    n_threads: int,
    n_threads_batch: int | None,
    n_gpu_layers: int,
    flash_attn: bool,
    use_mmap: bool,
    no_kv_offload: bool,
    type_k: str,
    type_v: str,
) -> dict[str, object]:
    return {
        "build_commit": "fake1234",
        "build_number": 9999,
        "backends": os.environ.get("LLAMATUNE_FAKE_BACKEND", "CUDA"),
        "model_filename": model_filename,
        "model_type": "fake 7B",
        "n_batch": n_batch,
        "n_ubatch": n_ubatch,
        "n_threads": n_threads,
        "n_threads_batch": n_threads_batch,
        "n_gpu_layers": n_gpu_layers,
        "flash_attn": flash_attn,
        "use_mmap": use_mmap,
        "no_kv_offload": no_kv_offload,
        "type_k": type_k,
        "type_v": type_v,
        "n_prompt": n_prompt,
        "n_gen": n_gen,
        "avg_ts": avg_ts,
        "stddev_ts": stddev_ts,
    }


def main(argv: list[str]) -> int:
    # Unconditional crash injection: no output at all, ahead of everything
    # else, including argument parsing.
    if os.environ.get("LLAMATUNE_FAKE_CRASH") == "1":
        return 134

    if "--help" in argv or "-h" in argv:
        help_text = _HELP_TEXT
        if os.environ.get("LLAMATUNE_FAKE_NO_DEPTH") == "1":
            help_text = "\n".join(
                line for line in help_text.splitlines() if "--n-depth" not in line
            )
        sys.stdout.write(help_text)
        return 0

    model_path: str | None = None
    values: dict[str, int | list[int]] = {}
    ctk = "f16"
    ctv = "f16"
    output_format: str | None = None
    delay_s: float | None = None

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-m", "--model"):
            i += 1
            if i >= len(argv):
                return _die(f"missing value for {arg}")
            model_path = argv[i]
        elif arg in _INT_FLAGS:
            i += 1
            if i >= len(argv):
                return _die(f"missing value for {arg}")
            try:
                parsed = [int(value) for value in argv[i].split(",")]
                values[_INT_FLAGS[arg]] = parsed if len(parsed) > 1 else parsed[0]
            except ValueError:
                return _die(f"invalid integer value for {arg}: {argv[i]!r}")
        elif arg == "-ctk":
            i += 1
            if i >= len(argv):
                return _die(f"missing value for {arg}")
            ctk = argv[i]
        elif arg == "-ctv":
            i += 1
            if i >= len(argv):
                return _die(f"missing value for {arg}")
            ctv = argv[i]
        elif arg in ("-o", "--output"):
            i += 1
            if i >= len(argv):
                return _die(f"missing value for {arg}")
            output_format = argv[i]
        elif arg == "--delay":
            i += 1
            if i >= len(argv):
                return _die(f"missing value for {arg}")
            try:
                delay_s = float(argv[i])
            except ValueError:
                return _die(f"invalid value for --delay: {argv[i]!r}")
        else:
            return _die(f"unknown argument: {arg}")
        i += 1

    if model_path is None:
        return _die("missing required -m/--model")
    if not os.path.isfile(model_path):  # noqa: PTH113
        return _die(f"model file not found: {model_path}")

    if output_format is None or output_format != "json":
        return _die(f"unsupported output format: {output_format!r} (only 'json' is supported)")
    if os.environ.get("LLAMATUNE_FAKE_BATCH_CRASH") == "1" and any(
        isinstance(value, list) for value in values.values()
    ):
        return 134

    n_layer = _env_int("LLAMATUNE_FAKE_N_LAYER", 32)
    cores = _env_int("LLAMATUNE_FAKE_CORES", 8)
    vram_mb = _env_int("LLAMATUNE_FAKE_VRAM_MB", 999999)
    model_vram_mb = _env_int("LLAMATUNE_FAKE_MODEL_VRAM_MB", 4000)
    kv_mb_per_1k = _env_int("LLAMATUNE_FAKE_KV_MB_PER_1K", 0)
    speed_scale = float(os.environ.get("LLAMATUNE_FAKE_SPEED_SCALE", "1.0"))
    speed_ramp = float(os.environ.get("LLAMATUNE_FAKE_SPEED_RAMP", "0.0"))
    counter_file = os.environ.get("LLAMATUNE_FAKE_COUNTER_FILE")
    invocation = 0
    if counter_file:
        try:
            with open(counter_file, encoding="utf-8") as counter_fh:  # noqa: PTH123
                invocation = int(counter_fh.read().strip() or "0")
        except (FileNotFoundError, ValueError):
            invocation = 0
        with open(counter_file, "w", encoding="utf-8") as counter_fh:  # noqa: PTH123
            counter_fh.write(str(invocation + 1))
    speed_scale *= (1 + speed_ramp) ** invocation
    depth_penalty = float(os.environ.get("LLAMATUNE_FAKE_DEPTH_PENALTY", "0.0"))

    def scalar(name: str, default: int) -> int:
        value = values.get(name, default)
        return value[0] if isinstance(value, list) else value

    def choices(name: str, default: int) -> list[int]:
        value = values.get(name, default)
        return value if isinstance(value, list) else [value]

    ngl = scalar("ngl", n_layer + 1)
    fa = scalar("fa", 0)
    nkvo = scalar("nkvo", 0)
    ncmoe = scalar("ncmoe", 0)
    p = scalar("p", 512)
    n = scalar("n", 128)
    r = scalar("r", 5)

    hang_s = (
        delay_s if delay_s is not None else float(os.environ.get("LLAMATUNE_FAKE_HANG_S", 0) or 0)
    )
    if hang_s > 0:
        time.sleep(hang_s)

    if os.environ.get("LLAMATUNE_FAKE_MALFORMED_OUTPUT") == "1":
        sys.stdout.write("{not-valid-json")
        return 0
    if os.environ.get("LLAMATUNE_FAKE_OVERSIZED_OUTPUT") == "1":
        sys.stdout.write("x" * (8 * 1024 * 1024 + 1))
        return 0
    if os.environ.get("LLAMATUNE_FAKE_HOST_MEMORY_FAIL") == "1":
        sys.stderr.write("std::bad_alloc: cannot allocate host memory\n")
        return 1

    if os.environ.get("LLAMATUNE_FAKE_GENUINE_FAIL") == "1":
        sys.stderr.write("llama_model_load: error: failed to load model\n")
        return 1

    model_term = model_vram_mb * (ngl / (n_layer + 1)) * (1 - 0.6 * ncmoe / n_layer)
    kv_term = kv_mb_per_1k * (p + n) / 1024
    vram_needed = model_term + kv_term
    if vram_needed > vram_mb:
        fail_style = os.environ.get("LLAMATUNE_FAKE_FAIL_STYLE", "oom")
        messages = {
            "oom": "ggml_backend_cuda_buffer_type_alloc_buffer: failed to allocate\n",
            "load": "llama_model_load: error: failed to load model\n",
            "ctx": "llama_init_from_model: failed to create context with model\n",
            "cuda": "ggml-cuda.cu:106: CUDA error\n",
        }
        sys.stderr.write(messages.get(fail_style, messages["oom"]))
        return 1

    f_ngl = 0.3 + 0.7 * ngl / (n_layer + 1)
    fully_offloaded = ngl >= n_layer + 1
    results: list[dict[str, object]] = []
    batch_choices = itertools.product(
        choices("t", cores),
        choices("tb", cores),
        choices("ub", 512),
        choices("b", 2048),
        choices("mmp", 1),
    )
    for t, tb, ub, b, mmp in batch_choices:
        pp_base = (
            600
            * f_ngl
            * (1.15 if fa else 1.0)
            * _nearest_factor(ub, _UB_FACTORS)
            * _nearest_factor(b, _B_FACTORS)
            * (1 - 0.15 * ncmoe / n_layer)
            * speed_scale
        )
        if fully_offloaded and ncmoe == 0:
            f_t = 1.0
        else:
            t_safe = t if t > 0 else cores
            f_t = max(0.6, 1 - 0.05 * abs(math.log2(t_safe / cores)))
        tg_base = (
            40
            * f_ngl
            * (1.05 if fa else 1.0)
            * f_t
            * (1 - 0.30 * ncmoe / n_layer)
            * (0.99 if mmp == 0 else 1.0)
            * speed_scale
        )
        if "d" in values:
            tg_base /= 1 + depth_penalty * scalar("d", 0) / 8192
        canonical = _canonical_argv(
            {
                "m": model_path,
                "p": p,
                "n": n,
                "b": b,
                "ub": ub,
                "t": t,
                "tb": tb,
                "ngl": ngl,
                "fa": fa,
                "mmp": mmp,
                "nkvo": nkvo,
                "ctk": ctk,
                "ctv": ctv,
                "ncmoe": ncmoe,
                "r": r,
            }
        )
        pp_samples = _jittered_samples(pp_base, canonical, r)
        tg_samples = _jittered_samples(tg_base, canonical, r)
        results.extend(
            [
                _entry(
                    n_prompt=p,
                    n_gen=0,
                    avg_ts=statistics.fmean(pp_samples),
                    stddev_ts=statistics.pstdev(pp_samples),
                    model_filename=model_path,
                    n_batch=b,
                    n_ubatch=ub,
                    n_threads=t,
                    n_threads_batch=tb if "tb" in values else None,
                    n_gpu_layers=ngl,
                    flash_attn=bool(fa),
                    use_mmap=bool(mmp),
                    no_kv_offload=bool(nkvo),
                    type_k=ctk,
                    type_v=ctv,
                ),
                _entry(
                    n_prompt=0,
                    n_gen=n,
                    avg_ts=statistics.fmean(tg_samples),
                    stddev_ts=statistics.pstdev(tg_samples),
                    model_filename=model_path,
                    n_batch=b,
                    n_ubatch=ub,
                    n_threads=t,
                    n_threads_batch=tb if "tb" in values else None,
                    n_gpu_layers=ngl,
                    flash_attn=bool(fa),
                    use_mmap=bool(mmp),
                    no_kv_offload=bool(nkvo),
                    type_k=ctk,
                    type_v=ctv,
                ),
            ]
        )

    sys.stdout.write(json.dumps(results))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
