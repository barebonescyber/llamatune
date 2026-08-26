# llamatune

`llamatune` benchmarks a local GGUF model with an existing llama.cpp build and finds a
faster, validated set of runtime flags for the machine where it runs. It measures the
installed build's defaults, discovers safe GPU/CPU placement boundaries, tunes supported
dimensions, confirms improvements against measured noise, and writes reproducible evidence
plus ready-to-use commands.

The beta-supported core covers dense and Mixture-of-Experts (MoE) models on native Linux
and Windows with CPU-only or NVIDIA CUDA llama.cpp builds. Implemented multi-GPU,
unattended orchestration, quality evaluation, lossy-KV, calibration, and additional
platform paths remain experimental during the initial beta.

> [!IMPORTANT]
> llamatune tunes an installation you already control. It does not download models, build
> llama.cpp, quantize GGUF files, access hosted services, or change clocks, fan curves,
> governors, drivers, or system caches.

See [DESIGN.md](DESIGN.md) for the full behavior and evidence contract.

## Features

- Measures prompt-processing (`pp`) and token-generation (`tg`) throughput with
  `llama-bench`.
- Starts from measured defaults and conservative feasibility probes instead of assuming
  full GPU offload is safe.
- Tunes GPU layers, CPU MoE layers, batch/ubatch, CPU threads, flash attention, mmap, KV
  offload, and other flags supported by the detected llama.cpp build.
- Experimentally searches expert tensor overrides and lossy KV-cache types with a
  perplexity quality gate.
- Validates required context sizes and profiles operating KV depths.
- Uses bounded thermal cooldowns to reduce heat-related benchmark noise.
- Experimentally supports opt-in pooled multi-GPU capacity with coarse `-ts`/`-sm`
  placement search and per-device VRAM observations.
- Journals every run, survives interruption, and resumes without repeating completed work.
- Maintains a registry of confirmed recommendations and rejects stale matches when the
  requested context is larger than the recorded validation context.
- Includes experimental Night Shift orchestration for serial, unattended tuning of a
  directory of GGUF models.

## Requirements

- CPython 3.11 or newer.
- A readable local `.gguf` model.
- An installed or locally built llama.cpp containing `llama-bench`.
- Optional `llama-cli`, `llama-server`, and `llama-perplexity` binaries for validation,
  exports, and quality gating.

| v1 design tier | Platform | Status |
|---|---|---|
| Tier 1 | Linux x86_64 / aarch64 | v1 target; the beta claim covers x86_64 only |
| Tier 1 | Windows 10/11 x86_64 | Required CI lane; unavailable OS telemetry degrades with warnings |
| Tier 2 | macOS arm64 / x86_64 | Advisory CI lane pending the v1.0 release gate |

These are the design targets for v1. The initial public beta makes a narrower validation
claim: native Linux x86_64 and native Windows 10/11 x86_64, using NVIDIA CUDA or CPU-only
llama.cpp builds. See the [public beta contract](docs/beta-contract.md) for the supported,
experimental, and excluded surfaces.

The GPU backend comes from your llama.cpp build. For example, CUDA tuning requires a
CUDA-enabled `llama-bench`; installing llamatune does not add CUDA support to llama.cpp.

Verify the binary first:

```bash
/path/to/llama.cpp/build/bin/llama-bench --help
```

On Windows, use the directory containing `llama-bench.exe` in the commands below.

### Choosing a llama.cpp implementation on Windows

The official [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) project is the
reference implementation for LlamaTune's beta acceptance work. Alternative implementations
may expose compatible `llama-bench.exe` binaries, but they are not bundled, endorsed, or
beta-validated by LlamaTune. The descriptions below summarize claims made by their
maintainers rather than performance or compatibility claims made by this project.

| Implementation | Maintainer's claimed focus | LlamaTune status |
|---|---|---|
| [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) | Official upstream implementation with CPU and NVIDIA CUDA Windows builds | Recommended beta reference |
| [`thecodacus/llama.cpp`](https://github.com/thecodacus/llama.cpp) | Opt-in pinned-host-memory and expert-prefetch optimizations for faster prompt processing when large MoE models offload experts to system RAM | Experimental; fork-specific modes use `GGML_*` environment variables |
| [`ikawrakow/ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp) | Additional quantization types and CPU, hybrid CPU/GPU, fused-MoE, and specialized attention optimizations | Experimental; its CLI and defaults differ from upstream and some dimensions may not be detected or tuned |

Before testing an alternative implementation:

1. Confirm that its Windows build contains `llama-bench.exe` and that
   `llama-bench.exe --help` succeeds.
2. Give each implementation—and each environment-controlled mode—a separate
   `--sessions-dir`. Environment-variable values are not recommendation compatibility
   dimensions, so enabled and disabled fork modes must not share reusable results.
3. Run `llamatune scan`, then perform a fresh tune after changing the executable. LlamaTune
   fingerprints `llama-bench`, and recommendations from a different binary require new
   evidence.
4. Compare implementations with the same GGUF content, workload, context, and machine.
   Different quantizations or defaults are different experiments.
5. Expect capability gating: unsupported or unrecognized flags are omitted. A basic
   benchmark may work even when a fork-specific optimization is outside LlamaTune's search
   space.

This list is intentionally limited to implementations that provide or can build a
`llama-bench.exe`-compatible command. Applications that expose only a server or interactive
CLI are not drop-in LlamaTune benchmark targets.

## Installation

### Clone from GitHub with `uv` (recommended)

Install [`uv`](https://docs.astral.sh/uv/), clone the repository, and create the locked
environment:

```bash
git clone https://github.com/barebonescyber/llamatune.git
cd llamatune
uv sync
uv run llamatune --help
```

`uv run` automatically uses the checkout's environment; manual activation is unnecessary.

### Clone from GitHub with `venv` and pip

```bash
git clone https://github.com/barebonescyber/llamatune.git
cd llamatune
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
llamatune --help
```

Windows PowerShell:

```powershell
git clone https://github.com/barebonescyber/llamatune.git
Set-Location llamatune
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
llamatune --help
```

### Install directly from GitHub

Install the current default branch without keeping a checkout:

```bash
python3 -m pip install "git+https://github.com/barebonescyber/llamatune.git"
```

Or install it as an isolated command with `uv`:

```bash
uv tool install "llamatune @ git+https://github.com/barebonescyber/llamatune.git"
llamatune --help
```

To update either installation, repeat the command with pip's `--upgrade` or run
`uv tool upgrade llamatune`.

### Development environment and packages

```bash
git clone https://github.com/barebonescyber/llamatune.git
cd llamatune
uv sync --dev
uv run pytest
uv build
```

`uv build` writes a source distribution and wheel to `dist/`.

## Quick start

### 1. Inspect the machine and llama.cpp build

Pass the directory containing the binaries, not the `llama-bench` executable itself:

```bash
uv run llamatune scan \
  --llama-bin /path/to/llama.cpp/build/bin
```

If llama.cpp is already on `PATH`, omit `--llama-bin`. Add `--json` for machine-readable
output.

### 2. Preview the search without benchmarking

```bash
uv run llamatune tune /path/to/model.gguf \
  --llama-bin /path/to/llama.cpp/build/bin \
  --sessions-dir ./llamatune-sessions \
  --ctx-size 8192 \
  --vram-reserve-mb 1536 \
  --dry-run
```

The plan is capability-gated: unsupported llama.cpp flags are not included.

### 3. Tune the model

```bash
uv run llamatune tune /path/to/model.gguf \
  --llama-bin /path/to/llama.cpp/build/bin \
  --sessions-dir ./llamatune-sessions \
  --ctx-size 8192 \
  --vram-reserve-mb 1536 \
  --budget-trials 100 \
  --budget-minutes 60
```

For an installed package, remove `uv run` and begin with `llamatune`.

## Tuning guide

### Conservative CUDA and MoE tuning

This is a practical starting point for a model close to the GPU's VRAM limit:

```bash
uv run llamatune tune /models/model.gguf \
  --llama-bin /home/user/llama.cpp/build-cuda/bin \
  --sessions-dir ./llamatune-sessions \
  --ctx-size 8192 \
  --vram-reserve-mb 1536 \
  --initial-gpu-layers 0 \
  --initial-cpu-moe auto \
  --budget-trials 100 \
  --budget-minutes 60
```

llamatune probes safe GPU-layer boundaries and tunes CPU MoE placement when the installed
llama.cpp exposes the relevant capability. A resource failure or OOM is recorded as search
evidence and constrains later candidates; it does not automatically invalidate the session.

Useful controls:

- `--initial-gpu-layers N` chooses the first offload probe.
- `--max-gpu-layers N|auto` caps the search.
- `--initial-cpu-moe N|auto` seeds MoE-to-RAM placement.
- `--vram-reserve-mb N` preserves display/runtime headroom.
- `--validate-with-cli` checks the final recommendation with `llama-cli` when available.
- `--ot-search` enables expert tensor-override refinement when llama.cpp supports `-ot`.

### Context and depth

Validate one required context and additional larger contexts with an ascending CSV list:

```bash
uv run llamatune tune model.gguf \
  --ctx-size 32768,49152,65536
```

The first value is the required validation context. Larger values form a stretch ladder.
Recommendations are recorded with their validated context, so `best --ctx-size` will not
silently reuse a result validated only at a smaller context.

Tune at a particular KV depth and profile the confirmed winner across depths:

```bash
uv run llamatune tune model.gguf \
  --depth 32768 \
  --depth-profile 0,16384,32768
```

Depth flags are emitted only when the detected `llama-bench` supports them.

### Optimization targets and workloads

The default target is `balanced`. Choose a different objective with:

```bash
uv run llamatune tune model.gguf --target prompt
uv run llamatune tune model.gguf --target generation
```

Adjust the benchmark workloads with `--pp` and `--tg`; defaults are 512 prompt tokens and
128 generated tokens. Use `--baseline-only` to measure defaults without searching.

### Thermal stability and telemetry

llamatune can sample VRAM and GPU telemetry around trials. After the fixed `--cooldown`, it
waits in bounded cycles when temperature exceeds the threshold or clock telemetry indicates
throttling:

```bash
uv run llamatune tune model.gguf \
  --cooldown 2 \
  --thermal-threshold-c 75 \
  --thermal-wait-cap-s 60
```

Use `--no-observe-vram` where telemetry is unavailable or undesired. Missing telemetry
degrades gracefully and does not prevent tuning.

With telemetry enabled, a successful individual trial, stability rerun, or confirmation
that shows clock throttling receives at most one bounded replacement measurement after
cooldown. Both attempts remain journaled and consume budget. If the replacement cannot run
or is also contaminated, the measurement remains available as unstable evidence but is
excluded from winner scoring. Batched trials are not retried because their shared telemetry
cannot be attributed safely to one configuration.

### Multi-GPU tuning

> [!WARNING]
> Multi-GPU tuning is experimental during the initial public beta. Independently validate
> every resulting placement before adoption.

Multi-GPU behavior is opt-in:

```bash
uv run llamatune tune model.gguf \
  --llama-bin /path/to/llama.cpp/build/bin \
  --multi-gpu \
  --ctx-size 8192 \
  --vram-reserve-mb 1536
```

When multiple devices and the required llama.cpp capabilities are detected, llamatune:

- pools known free/total VRAM for the feasibility budget;
- tests a bounded grid of proportional, even, and single-device tensor splits;
- compares `layer` and `row` split modes;
- records per-device VRAM peaks and advisory capacity estimates; and
- keeps runtime results authoritative when an estimate disagrees with the benchmark.

With `--multi-gpu` off, the largest known device remains the capacity basis and the legacy
candidate/evidence behavior is preserved. Enabling it on a single-GPU system reduces to the
ordinary single-GPU path. This is a coarse placement search, not fine-grained ratio
optimization.

### Lossy KV-cache quality gating

> [!WARNING]
> Lossy KV-cache tuning is experimental and quality-affecting during the initial public
> beta. It is not covered by the beta acceptance claim.

Lossy cache candidates are excluded by default. To include them and compare perplexity
against an f16 reference:

```bash
uv run llamatune tune model.gguf \
  --allow-lossy \
  --quality-corpus /path/to/corpus.txt
```

Review the quality-gate result in `analysis.json` and `report.md` before deploying a lossy
recommendation. Cache precision descends conservatively from `f16` to `q8_0`, then `q4_0`.
At expanded context rungs, llamatune tries K-cache compression before matching K/V
compression when the installed llama.cpp capabilities permit it. Every lossy alternate is
identified by its cache types and marked for independent quality validation.

### Budgets, batching, and progress

- `--budget-trials N` caps counted benchmark measurements (default: 60), including
  baselines, probes, measured batch members, reruns/retries, validation, and confirmation;
  pruned candidates consume no budget.
- `--budget-minutes N` adds a wall-clock limit.
- `--reps-search`, `--reps-confirm`, and `--baseline-runs` control repetitions.
- `--no-batched-trials` disables compatible comma-list sweep batching.
- `--progress auto|plain|rich|json|none`, `--tui`, and `--quiet` control display output.
- `--quiet-wait-s` and `--quiet-load` can wait for lower host load before trials.
- `--allow-core-dumps` preserves inherited core-dump limits; core dumps are disabled for
  benchmark children by default.

### Windows example

```powershell
llamatune tune "C:\Models\model.gguf" `
  --llama-bin "C:\llama.cpp\build\bin" `
  --sessions-dir ".\llamatune-sessions" `
  --ctx-size 8192 `
  --vram-reserve-mb 1536
```

## Sessions and recommendations

### Resume an interrupted run

```bash
uv run llamatune resume ./llamatune-sessions/SESSION_DIRECTORY
```

Resume revalidates model, hardware, and llama.cpp identity, then skips journaled work and
continues with the recorded options and remaining budgets. Budget reconstruction includes
auxiliary measurements such as stability reruns, thermal retries, pair checks, validation,
and confirmation—not only final trial records.

### List sessions

```bash
uv run llamatune sessions ./llamatune-sessions
uv run llamatune sessions ./llamatune-sessions --json
```

### Regenerate a report

```bash
uv run llamatune report ./llamatune-sessions/SESSION_DIRECTORY
```

### Export a recommendation

```bash
uv run llamatune export ./llamatune-sessions/SESSION_DIRECTORY --format llama-server
uv run llamatune export ./llamatune-sessions/SESSION_DIRECTORY --format llama-cli
uv run llamatune export ./llamatune-sessions/SESSION_DIRECTORY --format systemd
uv run llamatune export ./llamatune-sessions/SESSION_DIRECTORY --format llama-swap
uv run llamatune export ./llamatune-sessions/SESSION_DIRECTORY --format json
```

### Look up the best compatible result

```bash
uv run llamatune best /path/to/model.gguf \
  --llama-bin /path/to/llama.cpp/build/bin \
  --sessions-dir ./llamatune-sessions \
  --ctx-size 8192
```

The lookup checks the model, hardware, llama.cpp identity, and requested context. An older
or undersized result is reported as stale instead of being returned as current.

### Revalidate and calibrate

Re-run confirmation for a recorded winner:

```bash
uv run llamatune revalidate ./llamatune-sessions/SESSION_DIRECTORY
```

Fit conservative VRAM correction factors from completed sessions:

```bash
uv run llamatune calibrate --sessions-dir ./llamatune-sessions
```

Calibration is experimental during the initial public beta. It writes `calibration.json`;
later searches load it automatically when valid.

## Night Shift (experimental)

Night Shift recursively discovers GGUF files, groups sharded models, avoids duplicate
layouts, and spends an unattended window tuning or revalidating models serially:

```bash
uv run llamatune nightshift ./models \
  --llama-bin /path/to/llama.cpp/build/bin \
  --sessions-dir ./llamatune-sessions \
  --until 07:00 \
  --profile deep
```

Night Shift accepts the same ascending context ladder as `tune`. The first value is
required; later values are stretch validations for each confirmed recommendation:

```bash
uv run llamatune nightshift ./models \
  --llama-bin /path/to/llama.cpp/build/bin \
  --sessions-dir ./llamatune-sessions \
  --ctx-size 8192,16384,32768 \
  --max-hours 8
```

Preview the deterministic plan first:

```bash
uv run llamatune nightshift ./models \
  --llama-bin /path/to/llama.cpp/build/bin \
  --sessions-dir ./llamatune-sessions \
  --max-hours 8 \
  --dry-run
```

- `--until HH:MM` uses local time; a time already passed means tomorrow.
- `--max-hours` sets a duration cap; when both are supplied, the earlier deadline wins.
- With neither deadline, Night Shift makes one bounded pass and exits.
- `--profile standard|deep` controls search depth.
- `--include` and `--exclude` filter model names.
- `--duplicates one|both` controls duplicate-layout handling.
- `--drift-threshold` and `--calibration-runs` control revalidation/calibration behavior.
- Spare-time deepening runs at most once per eligible model and is skipped when explicit
  overrides make its resolved profile identical to the initial tune.

Night Shift writes its run summary and `nightshift-report.md` beneath
`SESSION_DIR/nightshift/`. Use `llamatune nightshift --help` for all budget overrides.

## Marathon (experimental)

Marathon concentrates an unattended benchmarking window on one GGUF model. It combines
repeated full tuning rounds with an auditable coverage ledger, drift brackets, a complete
context-by-KV-depth matrix, and interleaved A/B replication of the final recommendation.

```bash
uv run llamatune marathon ./models/model.gguf \
  --llama-bin /path/to/llama.cpp/build/bin \
  --sessions-dir ./llamatune-sessions \
  --until 07:00 \
  --ctx-size 8192,16384,32768 \
  --depth-grid 0,8192,32768
```

`--until` and `--max-hours` use the same deadline rules as Night Shift; when both are
given, the earlier deadline wins. Without either, Marathon stops after coverage converges
or `--rounds-max` is reached, so it is never unbounded. Night Shift distributes a window
across a model directory, while Marathon spends the window exhaustively on one model.
Use `--dry-run` to inspect the resolved work without benchmarking.

Each run writes `marathon.json` and `marathon-report.md` beneath
`SESSION_DIR/marathon/<model-and-run-id>/`; ordinary tuning-round evidence remains in the
top-level sessions directory.

## Results Matrix

The Results Matrix is the single authoritative aggregation of measured tuning, Night
Shift, Marathon, and quality results. Session and run directories remain the evidence of
record; the matrix is a deterministic, rebuildable view whose rows retain provenance back
to that evidence. Successful evidence-producing commands refresh it opportunistically.

```bash
uv run llamatune matrix build --sessions-dir ./llamatune-sessions
uv run llamatune matrix show --sessions-dir ./llamatune-sessions
uv run llamatune matrix query --sessions-dir ./llamatune-sessions --use-case max-tg
uv run llamatune matrix export --sessions-dir ./llamatune-sessions --format csv
```

To select a result compatible with the machine and llama.cpp build being used now, probe
that build and request only current matches:

```bash
uv run llamatune matrix query \
  --sessions-dir ./llamatune-sessions \
  --use-case max-tg \
  --llama-bin /path/to/llama.cpp/build/bin \
  --compat current \
  --json
```

Agents can invoke `matrix query --json` for a freshly harvested, filtered answer or read
`<sessions-dir>/matrix/results-matrix.json` directly when the materialized view is
appropriate. `matrix build` writes that JSON artifact and `results-matrix.md` beneath the
first sessions root by default; `--output` selects another artifact directory. Repeating
`--sessions-dir` merges multiple roots without mixing hardware groups in rankings. Terminal
`tune`, `resume`, `revalidate`, NightShift, Marathon, and quality runs refresh the matrix
best-effort; when the owning root already contains a multi-root matrix, its configured root
set is preserved and re-harvested. If that artifact is corrupt, unreadable, or invalid, the
refresh warns and safely rebuilds from the owning sessions root alone.

## Quality evaluation (experimental)

`llamatune quality` measures a model and exact runtime configuration with deterministic,
versioned smoke suites. `coding` checks small self-contained programs, `tooluse` grades a
prompt-embedded tool protocol, `agentic` exercises declarative simulated environments,
`ifollow` checks output constraints and long-context needle retrieval, and `perplexity`
wraps the existing corpus measurement. Scores are meaningful only relative to the same
suite id; they are not claims of equivalence to public benchmarks.

```bash
uv run llamatune quality ./models/model.gguf \
  --llama-bin /path/to/llama.cpp/build/bin \
  --sessions-dir ./llamatune-sessions \
  --config best \
  --suite coding --suite tooluse --suite agentic --suite ifollow
```

`--config best` uses a current compatible registry recommendation and otherwise warns and
falls back to llama.cpp defaults; `--strict-config` rejects that fallback.
`--config-session DIR` evaluates the winner from a specific matching session. For a lossy
KV-cache recommendation, add `--compare-lossless` to run the evaluated and lossless
configurations serially and report per-suite deltas. Select `perplexity` only with a local
`--quality-corpus PATH`.

### `quality --exec` trust boundary

Generated Python is always data unless the operator explicitly enables `--exec`. With
`--exec`, llamatune extracts fenced Python blocks from model responses and executes them
in a child process. Each child runs one suite assertion in a fresh temporary directory.
The child gets an allowlisted environment, bounded runtime, and bounded captured output.

What the sandbox confines:

| Limit | Linux | macOS | Windows |
|---|---|---|---|
| CPU time | 10 s rlimit | 10 s rlimit | not available (`--exec` refuses) |
| File size | 1 MiB rlimit | 1 MiB rlimit | not available |
| Open files | 32 rlimit | 32 rlimit | not available |
| Core dumps | disabled | disabled | not available |
| Memory | 512 MiB address-space rlimit | data-segment and RSS rlimits where honored; no address-space cap | not available |
| Network | confirmed network namespace (`unshare -rn`) | none | none |
| Filesystem | bubblewrap sandbox when available | none | none |

Before any execution, llamatune probes network-namespace support with a real `unshare -rn`
test process. The probe must confirm isolation. If it does not, `--exec` refuses to run
and llamatune exits with code `3`. Pass `--exec-allow-network` to run anyway. Isolation
is still applied when available, even with `--exec-allow-network`.

When bubblewrap (`bwrap`) is installed and works, the child sees `/usr`, `/lib`, and
`/lib64` read-only, gets private tmpfs mounts for `/tmp`, `/home`, and `/run`, and can
write only its own run directory. Without bubblewrap, there is no filesystem confinement.

What the sandbox does NOT confine:

- Without a confirmed namespace: the child can open network connections.
- Without bubblewrap: the child reads and writes files with your account's permissions,
  including `$HOME`.
- On macOS and Windows: no network or filesystem confinement.
- A confirmed namespace does not confine the filesystem. Bubblewrap does not confine
  the network by itself here.

The run summary records `exec_isolation`: `network-namespace` when confirmed active,
or `none (allowed by flag)` after opt-in. Degraded runs add entries to the summary
`warnings` list and print `WARNING:` lines to stderr.

> WARNING: With `--exec-allow-network`, model-generated code runs with network access.
> That code can exfiltrate files, credentials, and session evidence to remote systems.
> Do not use it with an adversarial model.

Each run writes under `<sessions-dir>/quality/<run>/`, including identity/config metadata,
an fsynced journal, bounded request/response evidence, `quality.json`, and
`quality-report.md`. Interrupted runs can continue with `llamatune quality --resume RUN`.
Completed evidence refreshes the Results Matrix, where quality use cases such as `coding`,
`tool-use`, and `quality-overall` can be ranked per hardware group.

## Session artifacts

Each tuning session is stored below `--sessions-dir`. Important files include:

| Path | Purpose |
|---|---|
| `session.json` | Recorded options and session identity |
| `hardware.json` | Detected CPU, RAM, GPU, and driver information |
| `model.json` | GGUF metadata and fingerprint |
| `llamacpp.json` | Binary identity and detected capabilities |
| `journal.jsonl` | Append-only stages, probes, trials, reruns/retries, pruning, validation, and confirmation evidence |
| `analysis.json` | Baseline, budget consumption, boundaries, Pareto results, winner, validation, and warnings |
| `report.md` | Human-readable methodology and result report |
| `recommended.json` | Machine-readable recommendation |
| `recommended.sh` | Commented llama.cpp command examples |
| `baseline/`, `probes/`, `trials/` | Commands, output captures, hashes, and timings |

Confirmed recommendations are also appended to `registry.jsonl` under the sessions root.
Session evidence is designed to be auditable and resumable; avoid editing it manually.

## Exit codes

| Code | Meaning |
|---:|---|
| `0` | Confirmed improvement, or successful non-tuning command |
| `1` | Tuning completed but no improvement was confirmed over measured defaults |
| `2` | Usage or configuration error |
| `3` | Environment/model/baseline error; tuning could not start or complete normally |
| `4` | Interrupted or failed mid-run with a resumable session |

Exit code `1` is a valid result: defaults were optimal within measured noise, and the
session still contains recommendation and evidence files.

## Troubleshooting

### CUDA is detected but no GPU layers are used

Run `scan` and verify that `backends` and capabilities describe the expected build. Ensure
`--llama-bin` points at the CUDA build directory rather than a CPU-only build elsewhere on
`PATH`.

### A high GPU-layer trial fails or llama-bench aborts

VRAM availability depends on context, batch sizes, KV types, display usage, and MoE
placement. Preserve headroom with `--vram-reserve-mb`, validate the production context,
and let the boundary search back off. A failed aggressive probe can be expected evidence;
inspect its stderr and classification under the session directory.

### `bash: --ctx-size: command not found`

A multiline shell command is missing a trailing backslash on the preceding line. Every
continued line except the last must end in `\`:

```bash
uv run llamatune tune model.gguf \
  --sessions-dir ./llamatune-sessions \
  --ctx-size 8192
```

### `uv` warns that hardlinking failed

The cache and environment are on different filesystems. This affects installation speed,
not tuning correctness. Suppress the warning with:

```bash
UV_LINK_MODE=copy uv run llamatune --help
```

### No confirmed improvement

This is not necessarily an error. The installed llama.cpp defaults may already be optimal
for the selected workload, or measured gains may be within the baseline noise floor. Read
`report.md` and check exit code `1` before increasing budgets.

## Command reference

```text
llamatune scan        Assess hardware and llama.cpp
llamatune tune        Tune one GGUF model
llamatune resume      Continue an interrupted session
llamatune report      Regenerate a session report
llamatune export      Export a confirmed recommendation
llamatune nightshift  Experimental: tune and verify a model directory unattended
llamatune marathon    Experimental: benchmark one model over repeated rounds
llamatune sessions    List session health and status
llamatune best        Find the best compatible confirmed result
llamatune revalidate  Reconfirm a recorded winner
llamatune matrix      Build, query, show, or export the Results Matrix
llamatune quality     Experimental: evaluate deterministic quality suites
llamatune calibrate   Experimental: fit VRAM estimator corrections
```

Run `llamatune COMMAND --help` for the authoritative option list.

## Contributing

Install the development environment and run the same local quality gates used by the
project:

```bash
uv sync --dev
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

The test suite includes integration scenarios with fake llama.cpp binaries; it does not
require a GPU for ordinary development validation.

## License

llamatune is distributed under the [Apache License 2.0](LICENSE).
