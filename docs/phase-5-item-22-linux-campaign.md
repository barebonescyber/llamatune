# Phase 5 Item 22 — Native Linux reference campaign

Status date: 2026-07-26.

Status: **native Linux campaign scenarios are complete through the optional
32,768-token stretch context**.

## Scope

Roadmap Item 22 covers the native Linux campaigns for CPU-only and NVIDIA CUDA
llama.cpp, full and partial GPU offload, MoE CPU placement, required and stretch
contexts, interrupt/resume, Night Shift, matrix ingestion, and recommendation
lookup.

This record now covers:

- stable pinned llama.cpp build `b9637`;
- CPU-only and NVIDIA CUDA backends;
- dense, CPU-feasible Qwen2.5 1.5B Q4_K_M model;
- primary memory-bound Qwen3.6 35B-A3B UD-Q4_K_M model with partial MoE CPU
  placement;
- stable b9637 and recent b10107 comparison;
- genuine two-file sharded input;
- externally interrupted and natively resumed tuning;
- a bounded Night Shift tune/deepen cycle;
- the 512-token prompt / 128-token generation workload;
- required 8,192-token plus stretch 16,384- and 32,768-token context
  validation; and
- scan, baseline, tune, report, export, cross-root matrix ingestion, and
  recommendation lookup behavior.

The first 16K-led ladder conservatively recorded its optional 32K envelope row
as skipped when its hard budget was exhausted. A separately budgeted follow-up
then passed 32K and produced an independently reproduced recommendation; the
matrix retains both facts.

## Exact identities

| Component | Recorded identity |
|---|---|
| Host | Fedora Linux 44, x86_64, AMD Ryzen 7 9850X3D, 8 physical / 16 logical cores |
| llama.cpp | `b9637`, commit `aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3`, build 9637 |
| CPU `llama-bench` SHA-256 | `d1c988eaeb9a2105c6cc8847f10f986ce78a7bad1a3e74fb7c2a78ab82aa6b87` |
| CUDA `llama-bench` SHA-256 | `fbf82593f6c470fa0c53e5cc0ca7587f526d8e04396cae4509debb7f999965e0` |
| Recent llama.cpp | `b10107`, commit `c0bc8591e8815c63cb01dd3f051a8b0df02501c9`, build 10107 |
| Recent CUDA `llama-bench` SHA-256 | `00ae37cdf3e0769f41265007f87bb5040002d2d34a64bd8bcd86945047f2e695` |
| CUDA device | NVIDIA GeForce RTX 5080, compute capability 12.0, driver 610.43.03 |
| Model | `qwen2.5-1.5b-instruct-q4_k_m.gguf`, Q4_K_M, dense `qwen2`, 28 layers |
| Model fast fingerprint | `54dbd5eb4d61d99fc131d1374d3f65ad6fd9becd0573beafb9c9b37ece26b2bf` |
| Model full SHA-256 | `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e` |
| Final CPU session | `models/phase5-item22-final-sessions/qwen2.5-1.5b-instruct-q4_k_m-20260726-035816-ed508b` |
| Final CUDA session | `models/phase5-item22-cuda-dense-normalized-sessions/qwen2.5-1.5b-instruct-q4_k_m-20260726-051437-ddfe43` |
| MoE model | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, Q4_K_M, `qwen35moe`, 40 layers / 256 experts |
| MoE model fast fingerprint | `12f6fcd83a6df5928afad8adabee6b255b003ad6c33107b4324963a930a86a4b` |
| Final MoE session | `models/phase5-item22-cuda-moe-final-sessions/Qwen3.6-35B-A3B-UD-Q4_K_M-20260726-070432-f00928` |
| Recent dense CUDA session | `models/phase5-item22-b10107-cuda-dense-sessions/qwen2.5-1.5b-instruct-q4_k_m-20260726-072851-d201ec` |
| Recent MoE session | `models/phase5-item22-b10107-cuda-moe-sessions/Qwen3.6-35B-A3B-UD-Q4_K_M-20260726-073016-bcdb2d` |
| Sharded model | `qwen2.5-1.5b-instruct-q4_k_m-00001-of-00002.gguf` plus `-00002-of-00002.gguf` |
| Shard SHA-256 values | `16e585b421871a28c7a99a20409b116ce54616751873743296c2673807d5d903`, `d1a18dd500d195eb1d72cca36ffa0e82d52d9e999cf3aa09c3151f456de92ebf` |
| Sharded aggregate fingerprint / full SHA-256 | `922400bcaaa41f7323cc566412c1f88638d80c3d2a334f9721fe33ff73e2b39d` / `097ccff779c79117a76acab8266b1fb66f8fe9a0edd2d2d20b2f765bdd45ea26` |
| Sharded session | `models/phase5-item22-b10107-sharded-sessions/qwen2.5-1.5b-instruct-q4_k_m-00001-of-00002-20260726-161121-d06352` |
| Interrupted/resumed session | `models/phase5-item22-b10107-stretch-sessions/Qwen3.6-35B-A3B-UD-Q4_K_M-20260726-161211-7c5037` |
| 16K stretch session | `models/phase5-item22-b10107-stretch16k-sessions/Qwen3.6-35B-A3B-UD-Q4_K_M-20260726-232809-531432` |
| 32K stretch session | `models/phase5-item22-b10107-stretch32k-sessions/Qwen3.6-35B-A3B-UD-Q4_K_M-20260727-021603-41e9be` |
| Night Shift run | `models/phase5-item22-nightshift-sessions/nightshift/20260727-014732-776773` |
| Cross-build matrix | `models/phase5-item22-cross-build-matrix/results-matrix.json` |

The model and session roots are ignored local campaign assets. They are not
tracked repository content.

## Commands

The final tune invocation on the remediated tree was:

```bash
timeout 720s .venv/bin/llamatune tune \
  models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --llama-bin $HOME/llama.cpp-reference-b9637/build-cpu/bin \
  --sessions-dir models/phase5-item22-final-sessions \
  --budget-trials 18 \
  --budget-minutes 10 \
  --ctx-size 8192 \
  --full-hash \
  --no-observe-vram \
  --progress plain \
  --json
```

The corresponding CUDA invocation used the same limits and model with:

```bash
timeout 720s .venv/bin/llamatune tune \
  models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --llama-bin $HOME/llama.cpp-reference-b9637/build-cuda/bin \
  --sessions-dir models/phase5-item22-cuda-dense-normalized-sessions \
  --budget-trials 18 \
  --budget-minutes 10 \
  --ctx-size 8192 \
  --full-hash \
  --progress plain \
  --json
```

The bounded memory-bound MoE campaign used:

```bash
timeout 1920s .venv/bin/llamatune tune \
  models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --llama-bin $HOME/llama.cpp-reference-b9637/build-cuda/bin \
  --sessions-dir models/phase5-item22-cuda-moe-final-sessions \
  --budget-trials 40 \
  --budget-minutes 30 \
  --ctx-size 8192 \
  --progress plain \
  --json
```

The recent-build comparison repeated the same dense CUDA limits with:

```bash
timeout 720s .venv/bin/llamatune tune \
  models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --llama-bin $HOME/llama.cpp-reference-b10107/build-cuda/bin \
  --sessions-dir models/phase5-item22-b10107-cuda-dense-sessions \
  --budget-trials 18 \
  --budget-minutes 10 \
  --ctx-size 8192 \
  --full-hash \
  --progress plain \
  --json
```

The recent-build MoE invocation was:

```bash
.venv/bin/llamatune tune \
  models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --llama-bin $HOME/llama.cpp-reference-b10107/build-cuda/bin \
  --sessions-dir models/phase5-item22-b10107-cuda-moe-sessions \
  --budget-trials 40 \
  --budget-minutes 30 \
  --ctx-size 8192 \
  --progress plain \
  --json
```

The genuine-shard case passed the first shard; discovery grouped both files:

```bash
.venv/bin/llamatune tune \
  models/qwen2.5-1.5b-instruct-q4_k_m-sharded/qwen2.5-1.5b-instruct-q4_k_m-00001-of-00002.gguf \
  --llama-bin $HOME/llama.cpp-reference-b10107/build-cuda/bin \
  --sessions-dir models/phase5-item22-b10107-sharded-sessions \
  --budget-trials 18 \
  --budget-minutes 10 \
  --ctx-size 8192 \
  --full-hash \
  --progress plain \
  --json
```

The completed stretch run used a hard 84-measurement ceiling and requested a
16K required rung followed by an optional 32K rung:

```bash
.venv/bin/llamatune tune \
  models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --llama-bin $HOME/llama.cpp-reference-b10107/build-cuda/bin \
  --sessions-dir models/phase5-item22-b10107-stretch16k-sessions \
  --budget-trials 84 \
  --budget-minutes 30 \
  --ctx-size 16384,32768 \
  --initial-gpu-layers 40 \
  --max-gpu-layers 40 \
  --initial-cpu-moe 20 \
  --progress plain \
  --json
```

The separately budgeted 32K follow-up used the same pinned warm start with 32K
as the required context:

```bash
.venv/bin/llamatune tune \
  models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --llama-bin $HOME/llama.cpp-reference-b10107/build-cuda/bin \
  --sessions-dir models/phase5-item22-b10107-stretch32k-sessions \
  --budget-trials 84 \
  --budget-minutes 30 \
  --ctx-size 32768 \
  --initial-gpu-layers 40 \
  --max-gpu-layers 40 \
  --initial-cpu-moe 20 \
  --progress plain \
  --json
```

The earlier externally stopped run was continued in place with:

```bash
.venv/bin/llamatune resume \
  models/phase5-item22-b10107-stretch-sessions/Qwen3.6-35B-A3B-UD-Q4_K_M-20260726-161211-7c5037 \
  --progress plain \
  --json
```

Night Shift first passed a dry-run plan check, then ran the same bounded plan
without `--dry-run`:

```bash
.venv/bin/llamatune nightshift \
  models/qwen2.5-1.5b-instruct-q4_k_m-sharded \
  --llama-bin $HOME/llama.cpp-reference-b10107/build-cuda/bin \
  --sessions-dir models/phase5-item22-nightshift-sessions \
  --max-hours 0.5 \
  --profile standard \
  --budget-trials 18 \
  --ctx-size 8192 \
  --full-hash \
  --json
```

The downstream checks used:

```bash
.venv/bin/llamatune report SESSION --json
.venv/bin/llamatune export SESSION --format json
.venv/bin/llamatune matrix build \
  --sessions-dir models/phase5-item22-final-sessions \
  --output models/phase5-item22-final-sessions/matrix \
  --json
.venv/bin/llamatune matrix query \
  --sessions-dir models/phase5-item22-final-sessions \
  --sort perf.pp \
  --include-unconfirmed \
  --json
.venv/bin/llamatune sessions models/phase5-item22-final-sessions --json
.venv/bin/llamatune best MODEL \
  --llama-bin $HOME/llama.cpp-reference-b9637/build-cpu/bin \
  --sessions-dir models/phase5-item22-final-sessions \
  --ctx-size 8192 \
  --json
.venv/bin/llamatune revalidate SESSION --json
```

`SESSION` and `MODEL` above denote the exact paths recorded in the identity
table; they are documentation placeholders, not shell variables used during
the campaign.

## CPU result

The final tune completed with exit code 1, the documented no-confirmed-
improvement outcome:

- baseline: 712.30 prompt-processing tokens/s and 58.30
  text-generation tokens/s;
- baseline noise floor: 3.17% CV;
- budget consumed: exactly 18 of 18 counted measurements;
- search trials: 7 `ok`, with zero crash, timeout, parse, CUDA, GPU-resource,
  OOM, or unstable outcomes;
- required 8,192-token context validation: `ok`;
- three confirmation runs completed within the ceiling;
- winner: none, because no candidate cleared the noise-aware promotion gate;
  and
- one explicit contention warning: load exceeded the threshold before 1 of 16
  invocations, so affected measurements may be pessimistic.

The absence of a winner is expected, valid evidence. It is not a failed core
workflow and no default recommendation was falsely labeled confirmed.

## CUDA result

The final CUDA tune also completed with the valid no-confirmed-improvement exit
code 1:

- baseline: 27,656.61 prompt-processing tokens/s and 477.98
  text-generation tokens/s;
- baseline noise floor: the 1% minimum;
- explicit resolved full offload: `gpu_layers = ngl_all = 29`;
- budget consumed: 15 of the 18-measurement hard ceiling;
- search trials: 8 `ok`, with zero crash, timeout, parse, CUDA, GPU-resource,
  OOM, or unstable outcomes;
- required 8,192-token context validation: `ok`;
- full-offload estimate: 1,603.89 MiB against a 13,772 MiB observed-free
  budget;
- peak telemetry: 203.82 W and 54 C, with no throttling; and
- winner: none, because the default full-offload configuration remained best
  within the noise-aware gate.

An earlier CUDA session exposed the `n_gpu_layers = -1` defect described below
and is retained only as defect evidence. Its apparent thread-count winner is
not acceptance evidence.

## Memory-bound MoE result

The stable-build MoE campaign completed at the exact 40-measurement ceiling
with exit code 0:

- the default full-offload probe failed conservatively as `gpu_resource` with
  pattern `failed to load model`;
- three explicit `-ngl 0` fallback baselines averaged 588.28 pp and 18.49 tg,
  with a 1.67% noise floor;
- full layer offload with 40, 30, and 20 CPU-resident MoE layers passed
  progressively, without a CUDA/OOM/crash/timeout/parse/unstable result;
- `ngl=41, ncmoe=20` passed the required 8,192-token context;
- confirmation averaged 1,152.17 pp and 86.11 tg, a balanced score of 3.020
  and a confirmed improvement of 202.00%;
- estimated VRAM was 12,727.65 MiB and observed used delta was 11,903 MiB;
  and
- peak telemetry was 129.11 W / 49 C with no throttling.

The confirmation score was below the best search sample by more than twice the
noise floor, and the report warns accordingly. It still clears the promotion
threshold by a very large margin.

### Independent direct reproduction

Pinned b9637 `llama-bench` was invoked directly, outside the LlamaTune search,
with five repetitions and `--no-warmup` for each exact configuration:

| Configuration | pp tokens/s | tg tokens/s |
|---|---:|---:|
| CPU fallback `-ngl 0` | 573.77 | 18.47 |
| Recommendation `-ngl 41 -ncmoe 20 -fa 1` plus recorded flags | 1,174.31 | 98.16 |

The direct reproduction improved pp by approximately 104.7% and tg by 431.6%,
well above `max(2 × 1.67%, 3%) = 3.34%`. This closes Item 23 verification for
this specific claimed winner; it does not verify future builds or sessions.

## Recent-build comparison

### Dense CUDA

The b10107 dense CUDA run completed with the valid no-confirmed-improvement
exit code 1:

- the default full-offload baseline averaged 30,328.13 pp and 524.29 tg, with
  the 1% minimum noise floor;
- throughput was 9.66% higher for pp and 9.69% higher for tg than the b9637
  dense CUDA baseline;
- the run consumed 15 of the 18-measurement ceiling and passed the required
  8,192-token context;
- six search trials were `ok`; one flash-attention-off sample was conservatively
  marked unstable for internal CV above 10%, with no crash, timeout, parse,
  CUDA, GPU-resource, or OOM outcome;
- peak telemetry was 204.28 W / 56 C with no throttling; and
- no candidate cleared the promotion gate, so the explicit full-offload
  default `ngl=29` remains the safe configuration.

### Memory-bound MoE

The b10107 MoE run completed at the exact 40-measurement ceiling with exit code
0:

- the default probe failed conservatively as `gpu_resource`, followed by three
  successful CPU fallback samples averaging 531.71 pp and 17.20 tg with a
  2.01% noise floor;
- full layer offload passed at 40, 30, and 20 CPU-resident MoE layers with no
  execution failure;
- the best sample was `ngl=40, ncmoe=20` at 1,210.14 pp and 93.65 tg;
- that placement passed the required 8,192-token context and confirmation
  averaged 1,137.50 pp and 86.61 tg, a balanced score of 3.282 and a confirmed
  improvement of 228.21%;
- compared with the corrected b9637 confirmation, absolute pp was 1.27% lower
  and tg was 0.59% higher; the larger relative score primarily reflects the
  slower b10107 CPU fallback baseline;
- estimated VRAM was 12,425.03 MiB and observed used delta was 11,864 MiB; and
- peak telemetry was 123.63 W / 48 C with no throttling.

The confirmation score was below the best search sample by more than twice the
noise floor, and the report warns accordingly.

### Independent direct reproduction

Pinned b10107 `llama-bench` was invoked directly with five repetitions and
`--no-warmup`:

| Configuration | pp tokens/s | tg tokens/s |
|---|---:|---:|
| CPU fallback `-ngl 0` | 557.72 | 17.71 |
| Recommendation `-ngl 40 -ncmoe 20 -fa 1` plus recorded flags | 1,093.30 | 83.85 |

The direct reproduction improved pp by 96.03% and tg by 373.44%, well above
`max(2 × 2.01%, 3%) = 4.02%`. This closes Item 23 verification for the b10107
winner.

## Remaining scenario results

### Genuine sharded input

The b10107 CUDA run discovered and hashed both qualified shards as one logical
model. It completed with the valid no-confirmed-improvement exit code 1:

- the aggregate fast fingerprint and full SHA-256 match the identities above;
- the full-offload baseline averaged 30,192.03 pp and 525.38 tg, with a 1.32%
  noise floor;
- seven trials completed `ok`, with no execution failure or warning;
- 15 of 18 counted measurements were consumed;
- the required 8,192-token context passed; and
- no candidate cleared the promotion gate.

This is actual multi-file GGUF evidence, not a renamed or synthetic single
file. The aggregate identity remains distinct from the equivalent monolithic
model in the cross-build matrix.

### Interrupt and native resume

The first long MoE stretch attempt was externally terminated after it had
persisted search progress. `sessions` reported it as `in_progress`; `resume`
then validated identity, reconstructed the incumbent and feasibility
boundaries, and finished the same directory with exit code 1:

- the journal has one `session_start`, one `resumed` stage, and one terminal
  `session_end`;
- its four baseline evidence entries are the failed default probe plus three
  CPU fallback samples and were not repeated after resume;
- the completed session consumed 78 measurements and retained 37 `ok`, seven
  `gpu_resource`, and two `cuda_error` search outcomes;
- the required 8,192-token context passed; and
- no winner was promoted because the candidate failed conservative
  confirmation under GPU pressure.

The interruption was an external `SIGTERM`, so the pre-resume journal correctly
had no fabricated terminal event.

### Stretch context

The revised b10107 MoE run completed at its exact 84-measurement ceiling with
exit code 0:

- its CPU fallback baseline averaged 575.73 pp and 18.80 tg with a 2.33% noise
  floor;
- `ngl=40, ncmoe=20`, flash attention on, 2,048 batch, 512 ubatch, eight
  threads, f16 KV cache, KV offload enabled, and mmap disabled passed the
  required 16,384-token context;
- confirmation averaged 1,568.88 pp and 95.84 tg, a balanced score of 3.727
  and a confirmed improvement of 272.71%;
- the optional 32,768-token envelope row is explicitly `skipped` because the
  hard budget was exhausted;
- peak telemetry was 124.96 W / 54 C with no throttling; and
- the report warns that system load exceeded the contention threshold before
  18 of 77 invocations and records the GPU-resource fallback accurately.

Pinned b10107 `llama-bench` independently reproduced the exact winner with five
repetitions and `--no-warmup`:

| Configuration | pp tokens/s | tg tokens/s |
|---|---:|---:|
| Fresh CPU fallback `-ngl 0` | 583.13 | 18.69 |
| Exact 16K recommendation plus recorded flags | 1,510.70 | 91.56 |

That direct run improved pp by 159.07%, tg by 389.77%, and the balanced ratio
by 256.21%, all far above `max(2 × 2.33%, 3%) = 4.67%`.

### Standalone 32K stretch

The separately budgeted b10107 run closed the optional rung with exit code 0:

- its CPU fallback baseline averaged 577.85 pp and 18.34 tg with the 1% minimum
  noise floor;
- `ngl=40, ncmoe=20`, flash attention on, 4,096 batch, 1,024 ubatch, ten
  threads, f16 KV cache, KV offload enabled, and mmap disabled passed the
  required 32,768-token context;
- confirmation averaged 1,549.74 pp and 90.21 tg, a balanced score of 3.632
  and a confirmed improvement of 263.21%;
- 83 of 84 counted measurements were consumed, with all 41 counted search
  outcomes `ok`;
- estimated allocation was 14,490.20 MiB, while observed peak GPU use was
  13,663 MiB and observed used delta was 11,894 MiB;
- peak telemetry was 127.83 W / 54 C with no throttling; and
- the report accurately warns about the CPU fallback and the confirmation
  score falling more than twice the noise floor below the best search sample.

Pinned b10107 `llama-bench` independently reproduced the exact recommendation:

| Configuration | pp tokens/s | tg tokens/s |
|---|---:|---:|
| Fresh CPU fallback `-ngl 0`, five repetitions | 575.92 | 18.08 |
| Exact 32K recommendation, five repetitions | 1,481.80 | 85.81 |
| Exact 32K allocation, 32,768 pp / 16 tg | 2,222.27 | 84.36 |

The normal-workload direct run improved pp by 157.29%, tg by 374.65%, and the
balanced ratio by 249.46%, far above `max(2 × 1%, 3%) = 3%`. Native
GPU-visible `best --ctx-size 32768` returned `status: hit` with no stale reason.

### Night Shift

The dry run planned exactly one tune for the two-shard logical model and no
duplicate. The bounded live window then completed before its deadline:

- one standard tune and one spare-time deepen action both succeeded;
- their underlying tuning exits were the accepted no-winner exit code 1;
- 21 total benchmark invocations were recorded;
- the window ran from 01:47:31 to 01:53:24 UTC against a 02:17:31 deadline;
  and
- the Night Shift summary has no warning.

The run therefore validates planning, shard grouping, tune dispatch, deepen
dispatch, artifact persistence, and clean bounded-window termination.

### Cross-build Results Matrix

The combined matrix ingested ten session roots spanning b9637, b10107, dense,
MoE, sharded, resumed, stretch, and Night Shift evidence. It contains 17 rows
for three distinct model identities: 11 baselines, two context-envelope rows,
and four confirmed recommendations, with no ingestion warning. The earlier
explicit 32K skip and the later confirmed 32K recommendation are both retained.

## Downstream disposition

- `report --json` regenerated the recorded final analyses and retained their
  exact measurement counts.
- Dense `export --format json` returned `confirmed: false`, the safe default
  configuration, build `aedb2a5e9` / 9637, the backend-specific binary
  SHA-256, model fast fingerprint, and full model SHA-256. CUDA export records
  explicit `gpu_layers: 29`.
- Each matrix build produced one current baseline row with no warning. Query
  retained the matching build discriminator, model fingerprint, and baseline
  metrics.
- `sessions` returned one complete session and did not misclassify the
  colocated `matrix/` artifact directory as corrupt.
- `best` returned exit 1 and `status: miss`, correctly refusing an unconfirmed
  recommendation.
- `revalidate` returned exit 3 with `session has no confirmed winner`, which is
  the correct inapplicable result for this no-improvement session.
- The MoE matrix contains one baseline and one recommendation row. Export
  retains `ngl=41`, `ncmoe=20`, the pinned build/binary identity, and the model
  fingerprint.
- Native GPU-visible `best` lookup returned `status: hit` with no stale reason.
  A device-isolated lookup correctly returned `hardware changed`, demonstrating
  conservative hardware matching rather than a registry defect.
- The b10107 dense matrix contains one baseline row with no warning; native
  `best` returned a clean miss and `revalidate` returned the expected
  inapplicable exit 3 because the session has no confirmed winner.
- The b10107 MoE matrix contains one baseline and one recommendation row with no
  warning. Export retained build 10107, the recent CUDA binary hash, and
  `ngl=40, ncmoe=20`; native `best` returned a clean hit.
- The sharded export retained the aggregate fingerprint and full hash, while
  the combined matrix kept it distinct from the monolithic dense identity.
- The combined matrix retained both historical stretch envelope rows and all
  four confirmed recommendations, including the standalone 32K pass, without
  warning.
- `sessions` on the Night Shift root now returns only the two complete tune
  sessions, not the materialized `nightshift/` artifact container.

## Defects found and remediated

The campaign exposed eight defects:

1. `--budget-trials 18` consumed 22 measurements because required context and
   confirmation were allowed to overrun the advertised counted budget.
2. Recommendation export used the discovery report held by the engine and
   lost the model-backed llama.cpp commit/build identity recorded during the
   baseline.
3. `sessions` treated its own colocated Results Matrix output directory as a
   corrupt tuning session.
4. If budget exhaustion occurred while revalidating an already-executed
   batched candidate, the candidate could disappear from the coverage summary.
5. b9637 reports its default full-offload placement as
   `n_gpu_layers = -1`. Passing that sentinel into the internal configuration
   model produced a negative VRAM estimate and caused ordinary CUDA sweep
   candidates to be rejected as invalid.
6. After a failed default probe, the three successful fallback samples were
   labeled `baseline 2/3`, `baseline 3/3`, and `baseline 4/3`.
7. Every CPU fallback warning said the default failed from OOM even when the
   recorded conservative classification was `gpu_resource` or `cuda_error`.
8. `sessions` treated the valid top-level Night Shift artifact container as a
   corrupt tuning session.

The narrow remediation:

- makes the trial count a hard ceiling;
- reserves one exact-winner context probe, when requested, plus one complete
  confirmation batch during search;
- rejects a trial budget smaller than the mandatory baseline batch;
- permits the mandatory baseline before applying a zero-minute search limit;
- merges persisted build identity with the recorded baseline backend on
  resume;
- excludes materialized Results Matrix and Night Shift artifact containers
  without hiding other incomplete/corrupt session directories;
- persists executed coverage before a revalidation probe can exhaust budget;
- normalizes the llama.cpp `-1` full-offload sentinel to the model's explicit
  `ngl_all` value, while making the estimator robust to legacy sentinel
  evidence;
- labels the fallback evidence as `CPU fallback 1/3` through `3/3` while
  retaining its actual evidence run IDs; and
- derives the fallback warning from the recorded default-probe classification.

Focused regression coverage includes hard context/confirmation ceilings,
baseline/trial-budget validation, zero-minute behavior, resumed backend
preservation, build identity in recommendations, matrix-directory filtering,
executed batch coverage at revalidation exhaustion, and full-offload sentinel
normalization/estimation, fallback progress and warning accuracy, and
Night Shift container filtering.

## Local verification

The final tracked tree passed:

```text
uv run ruff format --check .  -> 89 files already formatted
uv run ruff check .           -> All checks passed!
uv run mypy src tests         -> Success: no issues found in 86 source files
uv lock --check               -> resolved successfully
uv run pytest                 -> 927 passed, 2 skipped; 89.24% coverage
git diff --check              -> clean
```

## Remaining Item 22 work

The native Linux Item 22 campaign is complete through the optional 32,768-token
stretch rung, and every claimed winner has an independent Item 23 throughput
reproduction. No campaign execution remains; the next repository action is a
consolidated review and intentional commit.

No GitHub Actions run is required for native campaign execution. Full hosted CI
should be reserved for a consolidated reviewed code commit.

## Ownership deviation

The original implementation ownership did not define a Phase 5 campaign
owner. Item 22 owns this campaign record plus the narrowly necessary changes
in `DESIGN.md`, `config.py`, `cli.py`, `search.py`, `session.py`, and their
directly corresponding tests. No workflow, dependency, benchmark parser,
executor, or unrelated file is changed.
