# llamatune report

## Summary

Winner `t00000000000042` for target `balanced`: pp 5200.00 t/s (+4.00%), tg 830.00 t/s (+3.50%), score +3.75% vs baseline.
Config: `ngl=999 ncmoe=0 fa=1 ub=512 b=2048 t=8 mmap=1 nkvo=0 ctk=f16 ctv=f16`

## Hardware

- OS: Linux (x86_64)
- CPU: AMD Ryzen 9 7950X
- Physical cores: 16, logical cores: 32
- RAM: 131072 MiB
- GPU: NVIDIA GeForce RTX 4090 (nvidia), VRAM: 24576 MiB, method: nvidia-smi
- Warnings:
  - VRAM estimate is heuristic

## Model

- Path: `/home/user/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`
- Name: Llama 3.1 8B Instruct (Q4_K_M)
- Architecture: llama
- Layers: 32 (ngl_all=33)
- MoE: False (expert_count=0)
- Size: 4,921,000,000 bytes
- Fingerprint: `0123456789abcdef0123456789abcdef`

## llama.cpp build

- llama-bench: `/opt/llama.cpp/llama-bench`
- llama-cli: `/opt/llama.cpp/llama-cli`
- llama-server: `(not found)`
- Capabilities: ctk, ctv, fa, mmp, ncmoe, nkvo
- Build commit: b4202
- Build number: (unknown)
- llama-bench SHA-256: `cafebabecafebabecafebabecafebabecafebabecafebabecafebabecafebabe`

## Baseline

- Runs: 3
- pp: mean=5000.00 t/s, stdev=50.00, cv=0.0100, n=3
- tg: mean=800.00 t/s, stdev=8.00, cv=0.0100, n=3
- Noise floor (cv): 0.0100
- Fallback: (none)
- Provenance: measured

## Default probe

- Status: ok
- Classification: completed
- Pattern: (none)
- Evidence: `exit=0`

## Feasibility and placement

- Workload: pp=512 tg=128
- Context validated: 4096
- VRAM reserve: 1024 MiB (default)
- Boundaries: not evaluated
- Maximum fitting placement: `ngl=64 ncmoe=0` — fits with reserve
- Best measured placement: not evaluated
- Safety-adjusted recommendation: `ngl=48 ncmoe=0` — safety margin
- Estimated memory pressure: weights=4700.00 MiB, KV=512.00 MiB, compute=300.00 MiB, total=5512.00 MiB, budget=23552.00 MiB
- Warning: KV estimate used fallback heuristic
- Estimate calibration: calibrated from observed sessions

## Context validation

- Context: 4096
- Status: ok
- Evidence: `probe`

## Runtime validation

- llama-cli: ok (`help`)
- Estimate vs observed: not evaluated

## Method

- Dimensions searched (order): gpu_layers -> moe_cpu_layers -> flash_attn -> ubatch -> batch -> threads -> threads_batch -> mmap -> kv_offload -> cache_type_k -> cache_type_v
- Target: balanced
- Budget: 60 trials, unlimited minutes
- Repetitions: search=3, confirm=5
- Baseline runs: 3
- Workload: pp=512, tg=128
- Lossy dimensions allowed: False
- Cooldown: 0.0 s
- Noise floor (cv): 0.0100

## Search coverage

| dimension | candidates | executed | pruned | skipped |
|---|---|---|---|---|

## Top trials

_None recorded._

## Pareto set

_None recorded._

## Failures and prunes

| outcome | count |
|---|---|
| executed | 12 |
| ok | 10 |
| unstable | 0 |
| oom | 0 |
| cuda_error | 0 |
| gpu_resource | 0 |
| timeout | 0 |
| crash | 0 |
| parse_error | 0 |
| pruned | 2 |

## Warnings

- shard group kept first copy
- calibration reused prior evidence

## Reproduce

```sh
/opt/llama.cpp/llama-bench -m /home/user/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -p 512 -n 128 -r 5 -o json -ngl 999 -ub 512 -b 2048 -t 8 -ncmoe 0 -fa 1 -mmp 1 -nkvo 0 -ctk f16 -ctv f16
```

## Recommended runtime

```sh
/opt/llama.cpp/llama-cli -m /home/user/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -ngl 48 -b 2048 -ub 512 -t 8 -fa on -c 4096
```
