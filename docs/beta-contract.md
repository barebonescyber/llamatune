# Public beta contract

## Status and purpose

This document defines the support boundary for the initial `llamatune` public beta. It is
deliberately narrower than the complete v1 target in [`DESIGN.md`](../DESIGN.md). The design
remains authoritative for product behavior; this contract states which implemented paths
have enough validation to receive a beta support claim.

The beta is intended to find, confirm, record, and export faster llama.cpp runtime settings
for one local GGUF model on the machine where the measurements are made. It is an
evidence-producing tuning tool, not a universal performance guarantee or a substitute for
validating the resulting configuration in the user's production workload.

## Support classifications

The beta uses three classifications:

- **Beta-supported** — included in the release acceptance campaign. Defects are accepted
  as beta bugs, and compatibility is actively maintained throughout the beta series.
- **Experimental** — available for evaluation but not part of the beta acceptance claim.
  Behavior, schemas, and recommendations may receive additional validation or change before
  v1. Reports remain valuable evidence, but users must independently verify results.
- **Excluded** — outside the beta contract. The project makes no correctness,
  reproducibility, performance, or support claim for these environments.

## Beta-supported surface

### Platforms

- Native Linux x86_64.
- Native Windows 10/11 x86_64.
- CPython 3.11 or newer, subject to the versions exercised by the release's required CI
  lanes and recorded in its release notes.

“Native” means that `llamatune`, Python, and llama.cpp run directly under the named
operating system. A container or WSL2 Linux guest does not qualify as native Windows under
this contract.

### Reference execution backends

- CPU-only llama.cpp builds.
- NVIDIA CUDA llama.cpp builds on supported native Linux and Windows systems.
- A single visible GPU is the reference GPU configuration.

The user supplies llama.cpp and the local GGUF model. `llamatune` does not install GPU
drivers, add CUDA support to llama.cpp, download models, build llama.cpp, or alter the
machine's clocks, governors, caches, drivers, or fan controls.

Upstream flag availability is capability-gated. A feature that is absent from the detected
llama.cpp build is omitted rather than forced. The exact llama.cpp build identity and binary
hash recorded in session evidence remain part of result compatibility.

### Core commands and workflows

The following command paths are beta-supported:

- `scan` — assess native hardware and discover the user-provided llama.cpp installation.
- `tune` — inspect one GGUF model, measure a baseline, search supported runtime settings,
  confirm a result, and emit evidence and recommendations.
- `resume` — validate session identity and continue a resumable interrupted or bounded
  tuning session without silently repeating completed evidence.
- `report` — regenerate the human-readable report from recorded session evidence.
- `export` — render a confirmed recommendation in a supported export format.
- `sessions` — enumerate complete, incomplete, and corrupt sessions conservatively.
- `best` — look up a compatible confirmed recommendation without silently reusing stale
  model, hardware, build, workload, depth, or context evidence.
- `revalidate` — re-measure a recorded recommendation under current conditions.
- Results Matrix ingestion — automatically refresh the canonical matrix after terminal
  session-producing commands and explicitly build, query, show, or export the matrix from
  supported session evidence.

The supported tuning path includes conservative feasibility probes, CPU and GPU layer
placement, supported MoE CPU placement, flash attention, batch and ubatch sizes, threads,
mmap, KV-cache offload, bounded execution, statistical confirmation, session journaling,
and recommendation emission as defined by `DESIGN.md`.

### Result semantics

A successful process invocation does not necessarily mean a faster configuration exists:

- Exit code `0` means the command succeeded and, for tuning, a statistically confirmed
  improvement was found.
- Exit code `1` means tuning completed normally but no candidate cleared the confirmation
  threshold; defaults remain the honest recommendation.
- Exit codes `2` through `4` retain the usage, environment, and resumability meanings in
  `DESIGN.md`.

Only confirmation measurements support a “faster” claim. Search peaks, advisory estimates,
context allocation probes, telemetry, and unconfirmed candidates are evidence but are not
promoted to confirmed recommendations.

Recommendations apply only to the recorded combination of model identity, hardware,
llama.cpp build, workload, tuning target, depth, and validated context. They do not promise
equivalent application latency, concurrent-server throughput, output quality, or behavior
after any of those inputs change.

## Experimental surface

The following implemented surfaces are experimental during the initial public beta:

- AMD/ROCm GPU detection and tuning.
- Linux aarch64.
- macOS arm64 and x86_64, which remain Tier 2 design targets.
- The opt-in `--multi-gpu` path, especially heterogeneous GPU configurations and tensor
  placement.
- `nightshift` unattended multi-model orchestration.
- `marathon` extended single-model orchestration.
- `quality` evaluation and quality-row ingestion into the Results Matrix.
- `calibrate` advisory estimator calibration.
- Lossy KV-cache tuning, including `q8_0` and `q4_0`; these settings remain explicitly
  quality-affecting and require independent output-quality validation.

Experimental results must remain labeled by their actual environment and evidence. They
must not be merged with beta-supported results merely because model or nominal hardware
names appear similar.

## Excluded surface

The following are outside the initial public beta contract:

- Docker and other container runtimes.
- WSL1, WSL2, and Docker Desktop running through WSL2.
- Hosted tuning or benchmark services.
- Runtime network access or remote model acquisition.
- Building, converting, merging, sharding, quantizing, or modifying model files.
- Building or updating llama.cpp.
- Fine-grained exhaustive multi-GPU tensor-ratio optimization.
- Speculative decoding and production server concurrency optimization.
- Mutating system state to improve scores, including governors, clocks, caches, drivers,
  persistence modes, fan curves, or power limits.

Users may experiment in an excluded environment, but those sessions cannot establish a
native Linux or native Windows beta-support result. Container and WSL2 measurements must be
treated as distinct experimental environments if they are retained locally.

## Safety and evidence commitments

Within the beta-supported surface, `llamatune` commits to the following behavior:

- Commands are passed as argument lists; shell execution is never used.
- Every child process has a bounded runtime and bounded captured output.
- Benchmark output and model files are treated only as data.
- Child environments are allowlisted and evidence does not record credential values.
- Only the session layer writes inside a session directory, with path confinement.
- Runtime paths do not require network access.
- Missing hardware probes or telemetry degrade with warnings when safe operation can
  continue.
- Failures and pruning decisions are retained as evidence rather than silently discarded.
- Counted benchmark measurements, including reruns and retries, remain budget-consumed after
  resume; pruned candidates consume no measurement budget.
- A failed matrix refresh does not change the outcome of the session-producing command. An
  invalid existing artifact is discarded as root configuration and rebuilt from its owning
  sessions root.

No beta software can promise absence of defects. Users should retain the complete session
directory for diagnosis and review generated commands before using them in another runtime.

## Compatibility during beta

- `DESIGN.md` remains the authority when this document is silent.
- Release notes will identify supported CI versions, reference validation environments,
  schema changes, and known limitations for each beta release.
- Evidence is never silently rewritten to imitate a newer schema. Unsupported or corrupt
  versions are rejected or reported conservatively.
- Recommendations must be revalidated after changing the model, llama.cpp binary,
  operating system, material hardware configuration, workload identity, depth, or required
  context.
- Promotion of an experimental platform or feature requires its own acceptance evidence;
  implementation or a green synthetic test alone is insufficient.

## Release acceptance and defect policy

The initial beta may be published only after the candidate satisfies the tracked
[`beta acceptance checklist`](beta-acceptance-checklist.md). That checklist also defines
the P0, P1, and P2 defect classes, required triage evidence, and release-blocking rules.

No open P0 or P1 defect may be waived for the initial public beta. Every open P2 must have
an explicit disposition, owner, documented impact, and user-facing workaround or known
limitation before release approval.

## Reporting beta defects

A useful beta report should include:

- `llamatune` version.
- Native operating system, architecture, and Python version.
- CPU and GPU model.
- llama.cpp build commit or version and the relevant binary identity when available.
- The exact `llamatune` command with credentials and sensitive paths removed.
- Exit code and the affected session directory's non-sensitive evidence.
- Whether the result is repeatable on an otherwise idle machine.

Do not attach model files, credentials, complete environment dumps, or unrelated private
session content. Model fingerprints and the evidence already emitted by `llamatune` are
preferred over distributing the model itself.
