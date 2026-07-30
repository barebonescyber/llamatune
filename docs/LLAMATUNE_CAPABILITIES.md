# llamatune Capabilities and Command Guide

## Overview

`llamatune` is a local, evidence-driven optimizer for GGUF models running through
`llama.cpp`. It benchmarks the installed `llama.cpp` binaries on the current machine,
discovers feasible CPU/GPU memory placements, searches runtime settings, validates the
winner at requested context sizes, and emits reproducible recommendations.

It supports three levels of work:

| Mode | Scope | Best use |
|---|---|---|
| `tune` | One model, one bounded search | Normal interactive tuning |
| `nightshift` | A directory of models, one unattended window | Overnight fleet testing |
| `marathon` | One model, repeated exhaustive rounds | Maximum-confidence characterization |

llamatune does not alter GPU clocks, CPU governors, drivers, model files, or system
configuration. Benchmark processes run serially, and every important decision is backed
by stored command output and measurements.

## What llamatune can test

### Model placement and memory

- Number of model layers offloaded to the GPU (`-ngl`).
- MoE layers or experts assigned to CPU RAM (`--n-cpu-moe`/`-ncmoe`).
- Joint GPU-layer and CPU-MoE placement boundaries.
- VRAM feasibility with a configurable safety reserve.
- CPU RAM and VRAM estimates derived from GGUF metadata.
- KV-cache offload when supported by the installed `llama.cpp`.
- Multi-GPU tensor placement when `--multi-gpu` is enabled and the necessary
  `llama.cpp` capabilities are present.
- Optional tensor-override placement search with `--ot-search`.

MoE tuning is especially useful for models larger than VRAM. llamatune can keep the
transformer layer graph GPU-offloaded while placing selected MoE data in system RAM,
then measure the actual throughput tradeoff instead of assuming that maximum GPU
offload is always fastest.

### Throughput settings

- Prompt-processing batch size.
- Micro-batch size.
- CPU generation thread count.
- Batch thread count when supported.
- Flash attention on and off.
- Memory mapping on and off.
- Prompt-processing and token-generation workloads.
- Prompt-focused, generation-focused, or balanced optimization targets.

### Context and KV behavior

- A required context size plus ascending stretch contexts.
- Full-context feasibility probes for the exact recommended configuration.
- Safer placement fallbacks when a stretch context does not fit.
- A fixed KV depth for the complete tuning workload.
- Post-tuning depth profiles.
- Marathon context-by-depth matrices with bounded per-cell micro-batch refinement.

For `--ctx-size 8192,16384,32768`, 8192 is the required production context. Later
values are stretch validations. A winner is not confirmed unless its exact configuration
passes the required context test.

### Quality-sensitive settings

With `--allow-lossy`, llamatune may test supported quantized KV-cache types. These
dimensions are disabled by default because they can affect output quality. Supplying
`--quality-corpus` enables a perplexity-based quality gate when the installed tools
support it.

### Stability and environmental behavior

- Independent baseline invocations.
- Repeated search and confirmation measurements.
- Measurement variance and noise floors.
- GPU memory, utilization, temperature, and clock telemetry when available.
- Configurable fixed cooldowns.
- Adaptive high-temperature settling without treating cool idle clock drops as thermal
  throttling.
- Host-load warnings and optional waiting for a quieter machine.
- Revalidation drift against previously recorded results.
- Marathon calibration brackets and interleaved ABBA comparisons to reduce time-order
  bias.

## How normal tuning works

A normal `tune` run follows this general process:

1. Inspect the GGUF and fingerprint it.
2. Assess CPU, RAM, GPUs, available VRAM, and relevant telemetry.
3. Discover `llama-bench`, `llama-cli`, `llama-server`, and supported flags.
4. Measure the installed defaults. If the default placement does not fit, establish a
   conservative CPU baseline.
5. Discover GPU/MoE feasibility boundaries using measured success and failure.
6. Search applicable dimensions using bounded coordinate passes and joint refinements.
7. Prune configurations proven infeasible by monotone memory boundaries.
8. Recheck memory-sensitive candidates at the requested context.
9. Confirm the best candidate with repeated measurements.
10. Validate the final exact recommendation at the required context and test stretch
    contexts when budget permits.
11. Write analysis, commands, raw evidence, a report, and exportable recommendations.

The search score compares prompt and generation throughput with the measured baseline.
The selected target controls the relative importance of those metrics. A candidate must
beat the baseline beyond the measured noise floor; a numerically faster but statistically
unconvincing result is not confirmed.

## Command guide

### `llamatune scan`

Assesses the machine and installed `llama.cpp` without loading a model.

It reports CPU topology, RAM, detected GPUs, VRAM, `llama-bench` identity, compiled
backends, and detected command-line capabilities. Use it first to catch a CPU-only build,
incorrect `--llama-bin`, or missing feature flag.

```bash
uv run llamatune scan --llama-bin /path/to/llama.cpp/build/bin
```

Add `--json` for machine-readable output.

### `llamatune tune`

Runs one bounded tuning search for one GGUF. This is the normal mode for finding a good
production configuration.

Important option groups:

- Search budget: `--budget-trials`, `--budget-minutes`.
- Statistical depth: `--baseline-runs`, `--reps-search`, `--reps-confirm`.
- Workload: `--pp`, `--tg`, `--target`.
- Memory: `--vram-reserve-mb`, `--initial-gpu-layers`, `--max-gpu-layers`,
  `--initial-cpu-moe`.
- Context: `--ctx-size`, `--depth`, `--depth-profile`.
- Safety and stability: `--cooldown`, `--thermal-threshold-c`,
  `--thermal-wait-cap-s`, `--quiet-wait-s`, `--quiet-load`.
- Advanced search: `--allow-lossy`, `--quality-corpus`, `--multi-gpu`,
  `--ot-search`, `--batched-trials`.
- Output: `--json`, `--progress`, `--tui`, `--quiet`.

`--dry-run` prints the planned search without launching benchmarks.
`--baseline-only` measures defaults and stops.

### `llamatune resume`

Continues an interrupted or incomplete tuning session. It revalidates model, hardware,
and binary identity, reloads the append-only journal, skips already executed or pruned
trials, and continues from the best recorded state.

```bash
uv run llamatune resume ./llamatune-sessions/MODEL-RUN-ID
```

### `llamatune report`

Regenerates `report.md` from a session's recorded evidence. It does not benchmark the
model. This is useful after report-rendering improvements or when the original report was
removed.

### `llamatune export`

Converts a confirmed recommendation into a ready-to-use format. Supported formats are
`llama-server`, `llama-cli`, `systemd`, `llama-swap`, and `json`.

```bash
uv run llamatune export SESSION_DIR --format llama-server
```

### `llamatune nightshift`

Runs unattended work across a directory of GGUF models. It recursively discovers models,
recognizes sharded files, groups content duplicates, and creates a deadline-aware serial
work plan.

Night Shift can schedule:

- `resume`: continue one of its incomplete sessions.
- `tune`: benchmark a model with no compatible completed session.
- `calibrate`: replay a known recommendation and check current performance drift.
- `retune`: search again when calibration finds meaningful drift or an error.
- `deepen`: spend remaining time on one larger-budget follow-up per eligible model.

Profiles:

| Setting | `standard` | `deep` | Spare-time deepening |
|---|---:|---:|---:|
| Trial budget | 60 | 120 | 240 |
| Search repetitions | 3 | 5 | 5 |
| Confirmation repetitions | 5 | 8 | 8 |
| Baseline runs | 3 | 5 | 5 |
| Fixed cooldown | 0 seconds | 5 seconds | 5 seconds |

Explicit budget options override profile values. If those overrides make the initial and
deepening profiles identical, the redundant deepening phase is skipped. Each model is
deepened at most once per shift.

With no deadline, Night Shift performs one bounded pass. `--until` and `--max-hours`
create an unattended window; if both are present, the earlier deadline wins. Use
`--dry-run` to inspect the plan.

Night Shift is the right choice when breadth across several models matters more than
exhaustive work on one model.

### `llamatune marathon`

Concentrates an entire bounded or convergence-limited window on one model. Marathon uses
the normal tuning engine for each round, then adds cross-round coverage, drift control,
operating-point characterization, and final replicated verification.

Its phases are:

1. Resume an incomplete round belonging to the same Marathon.
2. Run ten independent default reconnaissance measurements.
3. Execute escalating tuning rounds, warm-started from the current champion.
4. Challenge a differing contender against the champion in interleaved ABBA blocks.
5. Recompute the coverage ledger over executed and pruned candidates.
6. Stop on convergence, round cap, deadline, or a failure circuit breaker.
7. Measure the champion across the context-by-depth matrix.
8. Run final interleaved A/B verification against defaults.
9. Write `marathon.json` and `marathon-report.md`.

Default round settings begin at 240 trials, 8 search repetitions, 12 confirmation
repetitions, 5 per-round baseline runs, and a 10-second cooldown. Trial budgets escalate
by 1.5× per round up to 1000. The default maximum is six rounds.

Coverage is divided into auditable tiers:

- Tier A: high-value placement × batch × micro-batch × flash-attention combinations.
- Tier B: CPU-thread and MoE placement interactions.
- Tier C: remaining single-dimension sweeps.
- Tier D: a capped factorial over dimensions proven responsive by earlier evidence.

Marathon only declares statistical replication when the champion wins the final A/B
comparison using the current reconnaissance or re-baselined noise threshold. It is the
right mode for deep characterization of an important model, not routine testing of a
large directory.

### `llamatune sessions`

Lists complete, in-progress, and corrupt tuning sessions, including exit status, winner,
and confirmation state. Add `--json` for programmatic inspection.

### `llamatune best`

Looks up the best compatible confirmed recommendation for a model from
`registry.jsonl`. Compatibility considers model identity, `llama.cpp`, hardware, and an
optional requested context size. It does not simply return the numerically fastest result
from an incompatible machine or unvalidated context.

### `llamatune revalidate`

Re-runs confirmation for a recorded winner under current conditions and reports whether
the result reproduced within expected noise. This is useful after driver, operating
system, `llama.cpp`, cooling, or hardware changes.

### `llamatune calibrate`

Fits correction factors for the VRAM estimator using observed tuning sessions. It writes
`calibration.json`, which later tuning plans can load to improve weight, KV-cache, and
compute-memory estimates. Actual benchmark outcomes remain authoritative over estimates.

## Evidence and outputs

Normal sessions contain:

| Artifact | Purpose |
|---|---|
| `session.json` | Options, identity, and workload |
| `hardware.json` | CPU, RAM, GPU, and telemetry capabilities |
| `model.json` | GGUF metadata and fingerprint |
| `llamacpp.json` | Binary identity, build, backends, and flags |
| `journal.jsonl` | Append-only stages, probes, trials, pruning, and outcomes |
| `analysis.json` | Baseline, feasibility, winner, context validation, and warnings |
| `report.md` | Human-readable analysis |
| `recommended.json` | Machine-readable confirmed recommendation |
| `recommended.sh` | Commented `llama-server` and `llama-cli` examples |
| `baseline/`, `probes/`, `trials/` | Commands, captured output, hashes, and timings |

Night Shift writes its own run journal and `nightshift-report.md` under
`SESSION_DIR/nightshift/`. Marathon writes `marathon.json` and
`marathon-report.md` under `SESSION_DIR/marathon/RUN-ID/`, while its individual rounds
remain ordinary resumable tuning sessions.

## Exit codes

| Code | Meaning |
|---:|---|
| `0` | Successful command or confirmed/replicated improvement |
| `1` | Valid completed measurement without a confirmed improvement, or a completed orchestrated run containing a nonfatal failure/non-replication |
| `2` | Invalid option, unsupported requested capability, or malformed evidence request |
| `3` | Environment, model, binary, baseline, or repeated-run failure |
| `4` | Interrupted with resumable work remaining |

Exit code 1 is often a scientifically valid result rather than a software failure: the
defaults may be optimal within measured noise, or a promising winner may not have passed
the required safety or replication gate.

## Choosing a mode

Use `tune` when you need a strong configuration for one model in a predictable budget.
Use `nightshift` when you want several local models triaged, calibrated, and tuned while
the machine is unattended. Use `marathon` when one model justifies repeated rounds,
coverage accounting, context/depth mapping, drift brackets, and head-to-head replication.

For the authoritative current option list, run:

```bash
uv run llamatune COMMAND --help
```
