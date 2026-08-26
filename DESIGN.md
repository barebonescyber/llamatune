# llamatune — Design, Requirements, and Specification (v1)

Status: v1 specification, approved for implementation.
Platforms: Linux (x86_64/aarch64) and Windows 10/11 (x86_64) are Tier 1;
macOS (Apple Silicon and Intel) is Tier 2 until the v1.0-rc validation gate.
Prior art: adapts the evidence-first methodology of the infer-tune project
(M1) into a narrower, single-purpose, shippable tool. This document is
authoritative; implementation deviations must be listed in the handoff.

## 1. Product definition

llamatune finds the fastest combination of llama.cpp **runtime** settings for
one GGUF model on the workstation it runs on, proves the result against a
measured baseline, and emits it as a ready-to-use llama.cpp configuration.

Pipeline of the `tune` command:

1. **Scan**: assess and record the hardware and the llama.cpp installation.
2. **Inspect**: read the GGUF model (size, architecture, layer count, MoE
   expert count) and fingerprint it.
3. **Baseline**: 3 independent `llama-bench` invocations at llama.cpp default
   settings; establishes reference pp/tg throughput and the session noise
   floor.
4. **Search**: staged search over the feasible flag space (Sections 9–11) —
   GPU layer offload, MoE CPU offload, flash attention, batch/ubatch sizes,
   threads, mmap, KV-cache offload, and (opt-in) KV-cache quantization. One
   bounded `llama-bench` invocation per trial; every outcome, including
   failures and pruned candidates, is journaled as evidence.
5. **Confirm**: re-measure the winner with baseline-grade repetitions; it
   must beat the baseline beyond the noise floor or the tool honestly reports
   that defaults are already optimal.
6. **Analyze and emit**: `analysis.json`, human-readable `report.md`,
   `recommended.json`, and `recommended.sh` (llama-server / llama-cli command
   lines).

Metrics of record: prompt-processing throughput (**pp**, tokens/s) and
token-generation throughput (**tg**, tokens/s) as reported by `llama-bench`
(`avg_ts`), with `stddev_ts` retained.

### 1.1 Non-goals (v1)

WSL-specific integration; building llama.cpp from source; model quantization/conversion;
mutating system state (governors, clocks, caches, drivers, fan curves);
fine-grained multi-GPU tensor-ratio optimization (the opt-in `--multi-gpu` mode pools
known capacity and runs a bounded `-ts`/`-sm` placement pre-sweep);
speculative decoding; any hosted service; network access at runtime.

## 2. Prerequisites

- llama.cpp already installed or built by the user. `llama-bench` is
  required; `llama-cli` / `llama-server` are optional (used only to render
  example command lines). Binaries are located via `--llama-bin DIR` or PATH.
- CPython >= 3.11 (development uses `uv`).
- A locally readable GGUF model file.

## 3. CLI specification

Typer application, program name `llamatune`. All commands accept `--json`
for machine-readable output on stdout (human text otherwise).

- `llamatune scan [--llama-bin DIR] [--json]`
  Hardware + llama.cpp assessment only; no model needed.
- `llamatune tune MODEL.gguf [options]`
  - `--llama-bin DIR` directory containing llama.cpp binaries
  - `--sessions-dir DIR` (default `./llamatune-sessions`)
  - `--target balanced|prompt|generation` (default `balanced`)
  - `--budget-trials N` (default 60) — counted benchmark measurements,
    including baseline, probes, measured batch members, individual trials,
    pair checks, stability and thermal reruns, depth/quality/runtime validation,
    and confirmation; pruned candidates consume no budget. This is a hard
    execution ceiling. Search reserves capacity for the exact winner's required
    context probe (when requested) and one complete confirmation batch; if
    insufficient budget remains for either, that evidence is skipped and no
    unconfirmed candidate is promoted.
  - `--budget-minutes M` (default: unlimited)
  - `--baseline-runs N` (default 3, minimum 3)
  - `--reps-search N` (default 3) / `--reps-confirm N` (default 5) —
    `llama-bench -r` values
  - `--pp N` (default 512) / `--tg N` (default 128) — workload sizes
  - `--depth N` — KV depth used uniformly by baseline, probes, trials, and confirmation
  - `--depth-profile CSV` — profile the confirmed winner at ascending KV depths
  - `--ctx-size CSV` — required context followed by optional ascending stretch rungs
  - `--vram-reserve-mb N` (optional, non-negative; auto resolves to 1536 MiB)
  - `--initial-gpu-layers N` (optional, non-negative first boundary probe)
  - `--max-gpu-layers N|auto` (optional hard boundary cap)
  - `--initial-cpu-moe N|auto` (optional initial MoE-to-RAM placement)
  - `--allow-lossy` include KV-quantization dimensions (§9.2)
  - `--cooldown SECONDS` (default 0) sleep between invocations
  - `--baseline-only` stop after the baseline phase
  - `--full-hash` compute the full SHA-256 of the model file
  - `--progress auto|plain|rich|json|none`, `--tui`, `--quiet` — ephemeral
    progress on stderr (`auto` selects Rich on a TTY and plain otherwise)
  - `--allow-core-dumps` keep inherited child core limits (disabled by default)
  - `--quiet-wait-s S`, `--quiet-load N` — bounded, non-mutating load gate
  - `--observe-vram/--no-observe-vram` — best-effort GPU telemetry
  - `--validate-with-cli` — final recommendation check through llama-cli
  - `--dry-run` print a capability-gated plan without creating a session
  - `--quality-corpus PATH` enable lossy-KV perplexity evidence
  - `--no-batched-trials` disable comma-list sweep batching
  - `--ot-search` enable expert tensor-override refinement
- `llamatune resume SESSION_DIR [--json] [--progress ...] [--tui] [--quiet]` — re-validates identity, skips
  journaled trials, continues within the recorded budgets.
- `llamatune report SESSION_DIR` — regenerate `report.md` from evidence.
- `export`, `sessions`, `best`, `revalidate`, and `calibrate` provide runtime
  export, registry lookup/reconfirmation, and advisory estimator calibration.

Exit codes: `0` success with a confirmed improvement (or scan/report
success); `1` tuning completed but no confirmed improvement over defaults,
or the nightshift circuit breaker stopped the shift after repeated tune
failures (evidence and a defaults-recommendation are still written); `2`
usage or configuration error; `3` environment error (missing/unusable
llama-bench, unreadable model); `4` interrupted by a user signal mid-run;
the session stays resumable.

An exit-3 tuning outcome is rendered as `tuning did not start`, with its
failure stage, reason, and session evidence path. It is never described as a
completed search with no confirmed improvement.

For `tune`, `resume`, and `revalidate`, a `--json` failure before a result
analysis exists emits an object with `status: "failed"`, `exit_code`,
`session_dir` (null when no session was created), `failure_stage`,
`failure_reason`, and `resumable`. Exit 4 is resumable; exits 2 and 3 are not.

## 4. Hardware and environment assessment (`hardware.py`)

Best-effort and layered; detection failure of any single probe must never
abort the tool. Every detected field carries a `method` provenance string.

- **CPU**: model name, physical cores, logical cores; macOS perf/efficiency
  core split. Linux: `/proc/cpuinfo`, `os.sched_getaffinity`,
  `os.cpu_count`. macOS: `sysctl` (`hw.physicalcpu`, `hw.logicalcpu`,
  `hw.perflevel0.physicalcpu`, `machdep.cpu.brand_string`).
- **Memory**: total and available. Linux `/proc/meminfo`; macOS
  `sysctl hw.memsize`.
- **GPU** (tiered, first hit wins per vendor, multiple GPUs recorded):
  NVIDIA `nvidia-smi --query-gpu=name,memory.total,memory.free
  --format=csv,noheader`; AMD `rocm-smi --showmeminfo vram` (fallback
  `/sys/class/drm/*/mem_info_vram_total`, with best-effort free VRAM derived
  from `mem_info_vram_used`);
  Apple `system_profiler SPDisplaysDataType -json` with unified-memory VRAM
  budget heuristic `min(0.75 × RAM)` recorded as an estimate.
- **Load**: `os.getloadavg()` recorded before every trial; warn when
  1-minute load > physical_cores / 2.
- Also recorded: `platform.uname()`, Python version, tool version, UTC start.

Total VRAM is durable capacity evidence. Free VRAM is a time-sensitive
observation with its own method and ISO-8601 UTC timestamp; a missing or
unparseable free-memory value never invalidates total capacity.

All external probes run through the executor contract (§7) with a 10-second
timeout and bounded capture; a missing probe binary yields `method: "absent"`.

## 5. Model assessment (`model.py`)

Read GGUF metadata with the official `gguf` package (header/KV only; never
load tensor data): `general.architecture`, `<arch>.block_count` → `n_layer`,
`<arch>.expert_count` → `expert_count` (`moe = expert_count > 0`),
`general.name` when present. `ngl_all = n_layer + 1` (offload-everything
value). Fingerprint (fast, default): SHA-256 over (first 1 MiB + last 1 MiB +
file size + the metadata keys above); `--full-hash` adds a whole-file
SHA-256. A model file is data — it is never executed.

For a standard `<name>-00001-of-000NN.gguf` sharded input, the supplied path
must be shard 1 and the `split.no`, `split.count`, and `split.tensors.count`
metadata must agree across a complete, same-directory numeric shard set.
Inspection rejects missing, out-of-directory, duplicate, or inconsistent
members before benchmarking. Reported size and tensor byte totals are
aggregate values. The fast fingerprint samples the first and last 1 MiB of
every shard in numeric order with explicit shard sizes and boundaries;
`--full-hash` hashes every shard byte with the same deterministic boundaries.
Single-file fingerprint and full-hash values retain their existing contract.

When the GGUF tensor table is cheaply available, tensor byte sizes are split
into expert weights (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) and
dense weights (all other tensors, including shared-expert `_shexp` tensors).
If that table cannot be interpreted reliably, both split values are unknown;
model inspection still succeeds and the memory estimator uses its legacy
proportional fallback.

## 6. Measurement methodology

- **One `llama-bench` invocation per trial**:
  `llama-bench -m MODEL -p PP -n TG -r REPS -o json` plus the trial's flags
  (§9). Stdout must parse as a JSON array; the entry with `n_prompt > 0` is
  the pp sample and the entry with `n_gen > 0` is the tg sample
  (`avg_ts`, `stddev_ts`). Unknown extra fields are ignored; a missing
  required field is a `parse_error` outcome.
- **Baseline**: `--baseline-runs` (default 3) invocations with *no* tuning
  flags (only `-m/-p/-n/-r/-o`), `-r = reps-confirm`. `pp0`/`tg0` = mean of
  the per-run means. **Noise floor** `cv_nf` = max over {pp, tg} of
  stdev(run means)/mean(run means), floored at 0.01. The resolved default
  configuration (n_batch, n_ubatch, n_threads, n_gpu_layers, flash_attn,
  use_mmap, type_k, type_v, no_kv_offload) is parsed from the baseline JSON
  and becomes the search starting point. llama.cpp's `n_gpu_layers = -1`
  full-offload sentinel is normalized to `ngl_all` so the internal
  `0 <= gpu_layers <= ngl_all` contract remains true. Build identity (`build_commit`,
  `build_number`) and backend identity (`backends`, falling back to the older
  `backend` key) are captured from the first successful run and appended to
  `llamacpp.json`.
  - **Safe baseline fallback**: the first defaults probe is recorded separately.
    Any failure retries at `-ngl 0`; success establishes a `safe_fallback`
    baseline without discarding the defaults evidence. If both attempts fail,
    exit 3 with both artifact paths. Ambiguous GPU-resource wording is only
    treated as placement pressure after this behavioral CPU-load check.
- **Feasibility probes** use one repetition and are separate from scored trials.
  Boundary probes use the search pp/tg workload. With `--ctx-size`, context
  probes use `-p CTX -n 16 -r 1`, forcing full prompt/KV allocation without
  using that result as throughput evidence.
- **Search trials** use `-r = reps-search`; **confirmation** re-measures the
  winner (and the runner-up when one exists) with `--baseline-runs`
  independent invocations at `-r = reps-confirm`.
- **Stability**: a trial with internal `stddev_ts/avg_ts > 0.10` on either
  metric is re-run once; if still unstable it is kept but flagged
  `unstable`.
- The tool never modifies system state to improve numbers; it records
  conditions and warns (load, cooldown suggestions).

## 7. Execution contract (`executor.py`)

- argv lists only; `shell=True` is forbidden repository-wide.
- Child environment is a fresh allowlist copy: `PATH HOME USER TMPDIR LANG
  LC_ALL CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES` plus
  any variable starting with `GGML_`, `LLAMA_`, or `LLAMATUNE_FAKE_`
  (the last one exists for the test fixture). Environment *values* are never
  written into evidence; only the names of passed-through variables are.
- On POSIX, children use `start_new_session=True`; timeout or exceptional
  unwinding sends SIGTERM to the process group, waits up to 10 s, then sends
  SIGKILL. On Windows, children use `CREATE_NEW_PROCESS_GROUP` and are assigned
  to a kill-on-close Job Object; timeout or exceptional unwinding terminates
  the Job, with `taskkill /T /F` as a best-effort fallback when Job assignment
  is unavailable. Handles and child processes are reaped on every exit path.
- Per-trial total timeout: `clamp(6 × median baseline wall time, 120 s,
  3600 s)`; baseline and probe runs use fixed 1800 s and 10 s respectively.
- stdout captured to a file, max 8 MiB; stderr max 1 MiB; each with SHA-256,
  byte size, and truncation flag. Truncated stdout ⇒ `parse_error`.
- Result: exit code, wall seconds, timed_out, artifact paths/hashes/sizes,
  UTC start/end.
- Benchmark children set `RLIMIT_CORE=0` by default where supported, and
  command evidence records whether control was `disabled`, `inherited`, or
  `unsupported`. The
  executor emits guarded liveness callbacks about every two seconds; callback
  failures never affect child execution. Unsupported platform features degrade
  explicitly rather than aborting.

## 8. Trial outcome classification

`ok | unstable | oom | cuda_error | gpu_resource | timeout | crash | parse_error | pruned` (pruned =
never executed, carries the pruning ancestor trial id). OOM detection:
nonzero exit AND stderr matching any of the case-insensitive regexes
`failed to allocate`, `out of memory`, `cudaMalloc`, `kIOGPUCommandBuffer.*OutOfMemory`,
`ggml_backend.*alloc.*fail`; the matched pattern is recorded.
Ambiguous case-insensitive resource signatures (`failed to load model`,
`failed to create context`, `unable to load model`) classify factually as
`gpu_resource`; only a successful retry at CPU placement may behaviorally
confirm that GPU placement caused the failure. OOM takes precedence when
both signature classes appear.
CUDA-runtime signatures (`ggml-cuda.cu:<line>`, `CUDA error`, cuBLAS errors)
classify as `cuda_error` after OOM precedence. They are feasibility failures but
do not establish a monotone prune ancestor.

## 9. Search space

### 9.1 Dimensions

Candidates are deduplicated, clamped to feasibility, and centered on the
resolved baseline defaults. `n_layer`, core counts, and capabilities come
from Sections 4–5 and the `llama-bench --help` capability probe.

| dim | bench flag | candidate values | applies when | class |
|---|---|---|---|---|
| gpu_layers | `-ngl` | {0, ¼, ½, ¾, 1} × ngl_all (rounded, dedup) | GPU present | lossless |
| moe_cpu_layers | `-ncmoe` | {0, ¼, ½, ¾, 1} × n_layer | model.moe and cap `ncmoe` and gpu_layers > 0 | lossless |
| flash_attn | `-fa` | {0, 1} | cap `fa` | lossless¹ |
| ubatch | `-ub` | {128, 256, 512, 1024, 2048} | always | lossless |
| batch | `-b` | {512, 1024, 2048, 4096} | always | lossless |
| threads | `-t` | {physical, physical/2, logical, perf-cores} dedup | not fully offloaded, or CPU-only, or moe_cpu_layers > 0 | lossless |
| mmap | `-mmp` | {1, 0} | cap `mmp` | lossless |
| kv_offload | `-nkvo` | {0, 1} | GPU present and (estimated memory pressure > 80% (§9.3) or incumbent no_kv_offload = 1) | lossless |
| cache_type_k | `-ctk` | {f16, q8_0, q4_0} | `--allow-lossy` and cap `ctk` | lossy |
| cache_type_v | `-ctv` | {f16, q8_0, q4_0} | `--allow-lossy` and cap `ctv` and flash_attn = 1 | lossy |

¹ flash attention may change numerics at rounding level; treated as
lossless per upstream guidance and noted in the report.

Constraints (enforced by `config.py`, pure functions, unit-tested):
`ubatch <= batch`; `cache_type_v != f16` requires `flash_attn = 1`;
`moe_cpu_layers <= n_layer`; `0 <= gpu_layers <= ngl_all`; `1 <= threads <=
logical`; no GPU backend ⇒ gpu_layers fixed at 0 and GPU-dependent dims
inapplicable.

### 9.2 Quality policy

Lossy dimensions are excluded unless `--allow-lossy`. Their precision order is
`f16` > `q8_0` > `q4_0`; the advisory KV-memory scales are respectively 1.0,
0.5, and 0.25. `q4_0` is the most aggressive tier and carries no implied quality
equivalence to either higher-precision tier.

When a lossy setting wins, the report labels the winner **quality-affecting —
validate output quality before adoption** and additionally reports the best
all-lossless configuration. When `--quality-corpus` is supplied, the perplexity
comparison against the otherwise-identical `f16` configuration remains evidence,
not an automatic acceptance gate; a delta above 1% emits a warning. Lossy context
envelope alternates are allocation-validity evidence only and must identify their
cache types plus carry the same validate-before-adoption warning.

### 9.3 Memory feasibility estimate

The advisory estimate is the sum of three deliberately conservative terms:

- weights: `dense_bytes × gpu_layers/ngl_all + expert_bytes ×
  gpu_layers/ngl_all × (1 − moe_cpu_layers/n_layer)`; when the GGUF split
  is unknown, use the legacy `model_size × gpu_layers/ngl_all ×
  (1 − 0.6 × moe_cpu_layers/n_layer)`;
- KV: when GGUF attention geometry is available, `tokens × sum_per_layer(
  n_kv_heads × (key_length + value_length) × 2 bytes) × cache-type scale ×
  gpu_layers/ngl_all`; otherwise the explicitly warned legacy heuristic
  `n_layer × tokens/32 MiB × cache-type scale × gpu_layers/ngl_all` is used.
  `-nkvo 1` makes this GPU term zero;
- compute buffers: for GPU placement, `128 + batch/16 + ubatch/8 MiB`.

The budget is detected total VRAM minus the configured reserve. Unknown
capacity produces an unknown budget, never a rejection. Estimates may order
or defer candidates and must carry a human-readable reason when over budget;
runtime probes remain ground truth. The formulas expose their approximation
instead of implying false byte-level precision.

## 10. Search algorithm (`search.py`)

The staged, evidence-first algorithm is:

1. Measure defaults, falling back to the known-safe CPU placement when needed.
2. Verify the observed backend before attempting GPU placement.
3. For each MoE CPU-residency rung (descending full, ¾, ½, ¼, zero),
   grow the GPU-layer lower bound and binary-search the first failure to an
   adjacent `max_ok_ngl`/`min_fail_ngl` pair. Explicit initial/max bounds are
   honored. Boundary probes have deterministic IDs over purpose, full canonical
   config, and context, so completed work is resume-skipped and budget-counted.
4. Measure the boundary and two lower neighbors normally, then refine the joint
   `(gpu_layers, moe_cpu_layers)` neighborhood by score rather than residency,
   continuing a bounded (12-execution) MoE hill climb in an improving direction.
   Boundaries record whether their cap came from the model, CLI, or a warm start.
5. When requested, validate the best pair at full context and walk toward safer
   placement until one passes. Without it, emit a warning. After confirmation,
   optional upper context-envelope rungs interleave `q8_0`, then `q4_0`, K/V
   cache variants with each placement fallback when `--allow-lossy` and the
   corresponding llama.cpp capability are present. For an `f16` base placement,
   the deterministic compression order is: base; K=`q8_0`; K/V=`q8_0`;
   K=`q4_0`; K/V=`q4_0`. If K quantization is unavailable, V alone follows
   `f16` → `q8_0` → `q4_0`; if V quantization is unavailable or flash attention
   is disabled, only the K variants remain. A base that is already lossy starts
   at its current tier and is never upgraded to a higher-memory tier. Invalid,
   unsupported, and duplicate candidates are omitted. Per-rung and total probe
   caps remain authoritative.
6. Sweep flash attention, batch/ubatch, threads, mmap, KV offload, and optional
   lossy cache types. The mmap sweep uses a fresh paired incumbent measurement
   to remove ordering bias. Memory-relevant mutations must pass an exact-config
   revalidation before replacing the incumbent. Repeat up to three passes.
   Eligible same-dimension candidates at or below the incumbent memory
   footprint are batched into one comma-list invocation. Any batch failure
   falls back to individual runs; budgets count measured combinations.
7. **Monotone pruning**: `oom` and behaviorally confirmed `gpu_resource` at
   `(g,c)` prune `gpu_layers >= g, moe_cpu_layers <= c` only when every other
   memory-relevant field is identical. Runtime probes remain ground truth;
   estimates only explain/order work.
8. **Confirmation** (§6). The winner is confirmed when its confirmed mean
   score > 1 + max(2 × cv_nf, 0.03) versus baseline. An unconfirmed winner
   falls back to the runner-up (one attempt); if none confirms, the tool
   reports "defaults optimal within noise" and exits 1.
9. Close local coverage with the minimum-feasible full-offload MoE frontier,
   thread midpoints, bounded ubatch×batch and threads×ncmoe interactions, and
   optional `-ot` expert placement. Coverage evidence distinguishes executed,
   cached, pruned, invalid, and budget-skipped candidates.

Before baseline and confirmation, an opt-in quiet gate may wait in bounded
five-second intervals; this time counts against the wall budget and unsupported
load sampling is journaled. With GPU observation enabled, a successful individual
trial, stability rerun, or confirmation whose own samples indicate clock
throttling receives at most one replacement measurement after the normal bounded
cooldown. The retry consumes the normal budget and never recurses; an unavailable
or still-contaminated replacement is retained as `unstable` evidence but excluded
from scoring. Batched trials remain unchanged because one shared sample window
cannot identify which measured combination was contaminated. An optional final
llama-cli context check is evidence only and never changes winner selection.
Progress is represented by ephemeral `ProgressEvent` values; the session journal
remains the evidence of record. The first SIGINT finishes the current child and
confirms best-so-far; a second SIGINT preserves immediate resumable exit 4.
Confirmed recommendations append to `<sessions-dir>/registry.jsonl`. Perplexity
and GPU telemetry otherwise remain evidence. `calibration.json` adjusts the
advisory estimator but never the authority of runtime probes.

## 11. Scoring, analysis, and recommendation (`stats.py`, `recommend.py`)

`score(c) = (pp_c / pp0)^w_pp × (tg_c / tg0)^w_tg`, weights by `--target`:
balanced (0.5, 0.5), prompt (0.8, 0.2), generation (0.2, 0.8). The Pareto
front over (pp, tg) across all `ok` trials is always computed and reported.

### 11.1 `analysis.json` schema (version 2 — `report.py` reads additions tolerantly)

```json
{
  "schema_version": 2,
  "target": "balanced",
  "baseline": {
    "runs": 3,
    "pp": {"mean": 0.0, "stdev": 0.0, "cv": 0.0, "n": 3},
    "tg": {"mean": 0.0, "stdev": 0.0, "cv": 0.0, "n": 3},
    "noise_floor_cv": 0.01,
    "fallback": null,
    "resolved_defaults": {}, "kind": "defaults"
  },
  "default_probe": null,
  "baseline_kind": "defaults",
  "feasibility": null,
  "context_validation": null,
  "cli_validation": null,
  "estimate_vs_observed": null,
  "coverage": {},
  "quality_gate": null,
  "telemetry": null,
  "counts": {"budget_consumed": 0, "executed": 0, "ok": 0, "unstable": 0,
             "oom": 0, "cuda_error": 0, "gpu_resource": 0, "timeout": 0,
             "crash": 0, "parse_error": 0, "pruned": 0},
  "pareto": [],
  "top": [],
  "winner": null,
  "lossless_winner": null,
  "warnings": []
}
```

`pareto`/`top` (≤ 10) contain TrialSummary objects:
`{"trial_id", "status", "config", "pp_mean", "tg_mean", "score", "flags"}`.
`winner` (null when unconfirmed): `{"trial_id", "config", "pp", "tg",
"score", "improvement_pct": {"pp", "tg", "score"}, "confirmed": true,
"confirmation": {"runs", "pp": {...}, "tg": {...}}}`.
`counts.executed` is the number of non-pruned final search-candidate records;
the status counters classify those records. `counts.budget_consumed` is the
resume-authoritative total of all countable benchmark measurements described by
`--budget-trials`, so it may be larger than `counts.executed`.

When requested and measured, schema-v2 documents additionally carry
`depth_profile: {"trial_id", "rows": [{"d", "pp", "tg"}]}` and
`context_envelope: [{"ctx", "status", "config", "fallback_config", "evidence"}]`.
These keys are omitted when their stages are not requested. Lossy envelope fallback
configs retain their exact `cache_type_k`/`cache_type_v` values.
`feasibility` distinguishes runtime-observed per-MoE boundaries, maximum
fitting placement, best measured placement, and the context-safe recommended
placement. It also records reserve value/provenance and the advisory estimate.

### 11.2 Emission

- `recommended.json`: config, bench flag list, expected pp/tg and
  improvement, model path+fingerprint, llama.cpp build identity.
- `recommended.sh`: commented, non-executable-by-default reference snippet
  with `llama-server` and `llama-cli` example lines using the flag mapping:
  `-ngl/-b/-ub/-t/-ctk/-ctv/--n-cpu-moe` map 1:1; `-mmp 0` → `--no-mmap`;
  `-nkvo 1` → `--no-kv-offload`; `-fa 1` → `-fa on` (comment noting older
  builds use bare `-fa`). Context-envelope alternates are emitted as commented
  commands; every lossy alternate has an adjacent quality-affecting warning.
- `report.md` sections: Summary (winner vs baseline), Hardware, Model,
  llama.cpp build, Baseline, Method (dimensions searched, budgets, noise
  floor), Top trials table, Pareto set, Failures and prunes, Warnings,
  Reproduce (exact `llama-bench` command for the winner).

## 12. Evidence, session layout, and resume (`session.py`)

```
<sessions-dir>/<model-stem>-<UTCyyyymmdd-hhmmss>-<6hex>/
  session.json        # schema_version 2, tool version, argv, options, created
  hardware.json
  model.json
  llamacpp.json       # probe result; build identity appended after baseline
  journal.jsonl       # append-only, one JSON object per line
  baseline/run-<n>/   # command.json, stdout.json, stderr.log
  trials/<trial-id>/  # primary plus optional rerun/thermal-retry/confirmation evidence
  analysis.json
  report.md
  recommended.json
  recommended.sh
```

`journal.jsonl` entry types: `session_start`, `baseline_run`, `probe`, `trial`,
`stability_rerun`, `thermal_retry`, `pair_check`, `depth_profile_run`,
`quality_gate_run`, `stage`, `confirmation_run`, `analysis_written`, and
`session_end`; every entry carries a UTC timestamp. Each budget-consuming
execution has a countable entry, while a batched benchmark contributes one
`trial` entry per measured configuration. Writes are flushed and fsynced per
line. A torn final line is tolerated on resume: discarded with a recorded
warning. (v1 deliberately does not hash-chain the journal — it is
honest-operator evidence, not tamper-proof; documented divergence from
infer-tune.)

**Journal corruption policy** (`evidence.read_journal_lines`, shared by all
orchestrator readers): unparseable lines are skipped and reported as
warnings. Valid entries before and after a corrupt line still load.
Nothing is silently truncated or hidden. A torn final line follows the
same rule and is named in a warning. Session resume stays stricter:
mid-file corruption stops resume with a corruption error.

Thermally observed `trial` and `confirmation_run` records carry contamination,
retry, and replacement-contamination state. A final trial rejected for thermal
provenance additionally carries `thermal_rejected: true` and has `score: null`.
`thermal_retry` entries link the original `run_id` to a distinct `retry_run_id`;
progress emits one terminal event for each identity, while the later trial or
confirmation record states which measurement was selected.

All session writes go through `Session` methods that resolve paths and
verify they remain under the session directory (reject traversal/links
outside it).

Schema v2 adds the context, reserve, and explicit placement-bound options plus
time-sensitive free-VRAM evidence. An absent schema version or version 1 is
migrated in memory with all new option/evidence fields set to `None`. Versions
newer than 2 are rejected as session corruption with the unsupported version
named in the error.

**Resume**: re-scan hardware, re-probe llama-bench, re-inspect the model.
The model fingerprint and the `llama-bench --help` SHA-256 must match the
session records (mismatch ⇒ exit 3); hardware drift is a warning. Journaled
trial and probe ids are skipped. The remaining budget is reconstructed from
every countable execution entry, including stability reruns, thermal retries,
pair checks, validation runs, and confirmation; restarting the engine therefore
does not restore budget consumed by auxiliary measurements.

## 13. Architecture

```
pyproject.toml
src/llamatune/
  __init__.py  _version.py  __main__.py
  types.py      # frozen dataclasses shared by all modules (below)
  config.py     # search-space construction, constraints, feasibility (pure)
  hardware.py   # §4      model.py  # §5      llama.py  # binary discovery/probe
  executor.py   # §7      bench.py  # command build + JSON parse (never executes)
  evidence.py   # shared evidence IO: jsonable, timestamps, confinement,
                 # journal reader, unique dirs, two-stage interrupt protocol
  session.py    # §12     search.py # §10 orchestration incl. baseline/confirm
  stats.py      # §6/§11 statistics
  recommend.py  # §11 scoring, pareto, analysis.json, recommended.*
  report.py     # report.md rendering from analysis.json content
  ui.py         # stderr-only progress renderers; no execution or session writes
  cli.py        # Typer app; thin; heavy imports lazy inside commands
tests/
  conftest.py  fixtures/fake_llama_bench.py  unit/  integration/
```

Layering rules: `types`/`config`/`stats` import nothing above stdlib;
`bench` builds argv and parses bytes but never executes; `executor` executes
but never parses benchmark semantics; only `session` writes inside the
session directory; `ui` consumes ephemeral events and writes only stderr;
`cli` contains no logic beyond argument handling and
output formatting.

### 13.1 `types.py` dataclasses (frozen, slots, kw_only — exact fields)

- `GPUInfo(vendor, name, vram_mb: int | None, method, vram_free_mb: int | None,
  vram_free_method: str | None, vram_free_at: str | None)`
- `HardwareReport(os_name, arch, cpu_model, physical_cores, logical_cores,
  perf_cores: int | None, ram_mb, gpus: tuple[GPUInfo, ...],
  warnings: tuple[str, ...])`
- `ModelReport(path, size_bytes, architecture, n_layer, ngl_all,
  expert_count, moe: bool, name: str | None, fingerprint,
  full_sha256: str | None, expert_bytes: int | None, dense_bytes: int | None)`
- `LlamaCppReport(bench_path, cli_path: Path | None,
  server_path: Path | None, capabilities: frozenset[str], help_sha256,
  build_commit: str | None, build_number: int | None, backends: str | None)`
- `TrialConfig(gpu_layers, moe_cpu_layers, flash_attn, ubatch, batch,
  threads, mmap, no_kv_offload, cache_type_k, cache_type_v,
  threads_batch=None, ot_spec=None)` with
  `trial_id` property (sha256 of canonical JSON, first 16 hex),
  `to_dict/from_dict`, and `bench_args(capabilities) -> tuple[str, ...]`
  (emits only flags present in the capability set; omits values equal to
  llama-bench semantics only when the dimension is inapplicable — applicable
  dims are always emitted explicitly).
- `MetricStats(mean, stdev, cv, n)`
- `TrialResult(trial_id, config, status, pp: MetricStats | None,
  tg: MetricStats | None, wall_s, exit_code: int | None,
  oom_pattern: str | None, artifact_dir: Path | None,
  flags: tuple[str, ...])`
- `BaselineResult(runs, pp: MetricStats, tg: MetricStats, noise_floor_cv,
  fallback: str | None, resolved_defaults: dict, kind="defaults")`
- `TuneOptions(target, budget_trials, budget_minutes: float | None,
  reps_search, reps_confirm, baseline_runs, pp, tg, allow_lossy,
  cooldown_s, baseline_only, llama_bin: Path | None, sessions_dir,
  full_hash, ctx_size: int | None, vram_reserve_mb: int | None,
  initial_gpu_layers: int | None, max_gpu_layers: int | None,
  initial_cpu_moe: int | None, quality_corpus=None, batched_trials=True,
  ot_search=False)`
- `TuneOutcome(session_dir, analysis: dict, exit_code: int,
  failure_stage: str | None, failure_reason: str | None)`
- `VramEstimate(weights_mb, kv_mb, compute_mb, total_mb, reserve_mb,
  budget_mb: float | None)`
- `FeasibilityBoundary(moe_cpu_layers, max_ok_ngl,
  min_fail_ngl: int | None, probes)`

### 13.2 `Session` API (consumed by `search.py`)

`Session.create(sessions_root, *, model, hardware, llama, options, argv)`;
`Session.load(session_dir)`; properties `dir`, `model`, `hardware`, `llama`,
`options`, `journaled_trial_ids: frozenset[str]`,
`baseline_runs_completed: int`, `entries: tuple[dict, ...]`; methods
`append(entry: dict)` (adds timestamp, flush+fsync), `trial_dir(trial_id)`,
`baseline_dir(n)`, `write_analysis(analysis)`, `write_text(name, text)`,
`record_build_info(commit, number, backends)`.

### 13.3 Entry points (implemented by `search.py`)

`run_tuning(session, hardware, model, llama, options) -> TuneOutcome` and
`resume_tuning(session_dir) -> TuneOutcome`.

## 14. Safety invariants

Never `shell=True`; bounded reads of all child output; writes confined to
the session directory; model files and benchmark output are data, never
code; no system-state mutation; no runtime network access; environment
values never recorded (names only); no credentials anywhere in argv,
config, or evidence.

## 15. Testing strategy

- **Fake llama-bench** (`tests/fixtures/fake_llama_bench.py`): executable
  fixture emulating the llama-bench CLI surface with a deterministic
  performance model and env-controlled failure injection
  (`LLAMATUNE_FAKE_*`). Tests copy it to a temp dir as an executable named
  `llama-bench`. Its normative behavioral contract is §15.1.
- **Tiny GGUF fixture**: `conftest.py` builds a real minimal GGUF via
  `gguf.GGUFWriter` (arch `llama`, `block_count` 32, one small tensor; MoE
  variant adds `expert_count` 8).
- Unit tests: config constraints/feasibility, hardware parsing with injected
  fake runners (macOS paths included), model inspection, bench argv/JSON
  parsing, executor timeout/output caps (tiny scripts), session journal +
  confinement + torn-tail tolerance, stats, pareto/scoring, report
  rendering, CLI `scan`.
- Integration tests (fake bench + tiny GGUF): unconstrained tune finds the
  known optimum; VRAM-constrained MoE scenario exercises baseline OOM
  fallback, monotone pruning, and MoE offload discovery; no-GPU scenario;
  resume skips journaled trials; all emission files exist and parse.
- No GPU, no network, no real model in CI.

### 15.1 Fake llama-bench behavioral contract

The fake benchmark is a standalone, standard-library-only executable. It
accepts the llama-bench dimensions used by the test suite: model, prompt and
generation lengths, KV depth, batch and microbatch, generation and batch
threads, GPU layers, flash attention, mmap, KV offload, K/V cache types, CPU
MoE layers, repetitions, and JSON output. Thread, batch, microbatch, and mmap
values may be comma-separated; the fixture emits the corresponding
cross-product.

Omitted dimensions use llama-bench-like defaults: 32 model layers, 8 CPU
cores, all 33 layers offloaded, flash attention off, mmap on, KV offload on,
8 threads, batch 2048, microbatch 512, f16 K/V, zero CPU MoE layers, prompt
512, generation 128, and five repetitions.

The deterministic memory model is:

```
model_vram = model_vram_mb * (ngl / (n_layer + 1))
             * (1 - 0.6 * ncmoe / n_layer)
kv_vram = kv_mb_per_1k * (prompt + generation) / 1024
vram_needed = model_vram + kv_vram
```

The invocation fails with the selected resource-failure signature when
`vram_needed` exceeds available fake VRAM. The default fake model size is
4000 MiB and default fake VRAM is effectively unconstrained.

The deterministic throughput model is:

```
f_ngl = 0.3 + 0.7 * ngl / (n_layer + 1)
pp = 600 * f_ngl * flash_factor * ubatch_factor * batch_factor
     * (1 - 0.15 * ncmoe / n_layer) * speed_scale
tg = 40 * f_ngl * flash_factor * thread_factor
     * (1 - 0.30 * ncmoe / n_layer) * mmap_factor * speed_scale
```

Microbatch factors for 128, 256, 512, 1024, 2048, and 4096 are 0.80, 0.92,
1.00, 0.97, 0.95, and 0.94. Batch factors for 512, 1024, 2048, and 4096 are
0.97, 0.99, 1.00, and 1.00. Flash factors are 1.15 for prompt processing and
1.05 for generation. The generation thread factor is 1.0 for a fully
offloaded non-MoE configuration; otherwise it is
`max(0.6, 1 - 0.05 * abs(log2(threads / cores)))`. Disabling mmap applies a
0.99 generation factor. When depth is explicitly supplied, generation is
divided by `1 + depth_penalty * depth / 8192`.

Every repetition receives deterministic ±1% jitter derived from SHA-256 of
the canonical invocation and repetition index. Successful output is a JSON
array containing one prompt-processing and one generation object for every
cross-product configuration, including build identity, resolved dimensions,
and mean and standard deviation.

`LLAMATUNE_FAKE_*` controls capability removal, layer/core/VRAM sizing, KV
growth, speed scaling and drift, depth penalty, invocation counting, hanging,
crashing, malformed or oversized output, host-memory failure, and selectable
resource-failure signatures. These controls must remain deterministic and
must not import `llamatune`.

The known unconstrained optimum is flash attention enabled, microbatch 512,
and all layers offloaded. With 2500 MiB fake VRAM and the default 4000 MiB MoE
model, the baseline OOM fallback is exercised and the best feasible region is
all layers offloaded with at least 20 MoE layers on the CPU.

## 16. Quality gates

```
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests        # strict
uv run pytest                # branch coverage >= 85%
uv run python .github/scripts/check_module_coverage.py coverage.json
```

The aggregate branch-coverage floor remains 85%. In addition, every module in
the beta-supported core must have at least 80% branch coverage, as measured by
the `coverage.json` emitted by the full test run. The core module allowlist is
versioned in `check_module_coverage.py`; experimental orchestration and quality
evaluation modules are not used to weaken or average away this per-module
gate.

Dependencies: `typer<1`, `gguf<1` (numpy transitively). Dev: `pytest`,
`pytest-cov`, `ruff`, `mypy`. Python >= 3.11.

## 17. Future work (post-v1)

Fine-grained multi-GPU tensor-ratio optimization beyond the bounded placement
pre-sweep; automatic quality rejection thresholds for lossy dimensions;
llama-server concurrent-throughput mode; richer vendor thermal telemetry;
hash-chained journal; speculative-decoding dimensions; `-ot` regex offload
fallback for builds without `-ncmoe`; estimator-driven context-size-aware KV
budgeting.
Depth is workload identity, not a `TrialConfig` dimension: it never changes trial
ids. Context probes omit depth because their prompt already forces full-context
allocation. After confirmation, an optional context envelope probes stretch rungs,
records monotone pruning, and searches boundary-derived fallback placements plus
opted-in `q8_0`/`q4_0` KV-cache tiers under per-rung and total caps. Optional
winner depth profiles and envelope probes are journaled evidence and consume the
normal trial budget.

Build sentinels such as `unknown`/`0` normalize to missing values. The streamed
`llama-bench` SHA-256 is the primary rebuild discriminator; CPU-only backends on a
GPU host and aggregate load contention are reported as warnings.
