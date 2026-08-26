# Changelog

All notable changes to llamatune will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases use [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Tag spelling note

Released tags use the spelling `v0.1.0-beta.N`. This equals the PEP 440 canonical
spelling `0.1.0bN`. For example, tag `v0.1.0-beta.3` and version `0.1.0b3` name
the same release.

## [Unreleased]

### Added

- A continuous-integration job that resolves the lowest allowed direct dependency
  versions into a clean environment and smoke-tests the installed command line
  against them.

### Changed

- Explicit dependency floors: `typer>=0.16,<1` and `gguf>=0.18,<1`. Older typer
  releases fail with current click releases, and older gguf releases either fail
  to import or require an undeclared extra.
- The security audit workflow now runs its local dependency, source, and secret
  audits on every push to main, every pull request, and a weekly schedule.
  CodeQL runs by default on the schedule and on pushes to main; it stays opt-in
  through manual dispatch elsewhere.

## [0.1.0-beta.3] - 2026-08-01

### Fixed

- Enforce an explicit `--max-gpu-layers` hard cap across planning, boundary discovery,
  search, resumed evidence, confirmation, recommendation emission, and auxiliary
  validation.
- Treat cap zero as a CPU-only search constraint while preserving the unmodified
  llama.cpp defaults measurement as the baseline of record.
- Return a controlled failure instead of fabricating a recommendation when no measured
  cap-compliant configuration succeeds, and remove stale derived outputs on resume.

### Known limitations

- The initial beta support claim is limited to native Linux x86_64 and native Windows
  10/11 x86_64 using CPU-only or NVIDIA CUDA llama.cpp builds; the other implemented
  platforms and workflows named in the beta contract remain experimental or excluded.
- Recommendations are valid only for their recorded model, hardware, llama.cpp build,
  workload, depth, and context identity and require revalidation after any material change.
- llamatune does not download models, install or build llama.cpp, install GPU drivers, or
  modify clocks, governors, caches, drivers, fan controls, or other system state.

## 0.1.0-beta.2 (rejected candidate) - 2026-07-27

### Added

- Complete package metadata, public security/support/compatibility policies, and structured
  issue templates.
- Explicit source-distribution contents, retained release archives and checksums, hosted
  Windows wheel installation, opt-in artifact provenance attestations, and per-module
  coverage enforcement for the beta-supported core.

### Changed

- Project licensing changed from MIT to Apache-2.0, with PEP 639 package metadata and
  contributor sign-off requirements.
- Experimental beta surfaces are labeled consistently in documentation, command help, and
  generated report titles.
- Release archives exclude tests, local reports, planning documents, and other files
  outside the public source-distribution allowlist.
- GitHub Actions checkouts do not persist workflow credentials, and privileged artifact
  attestation permissions are isolated to the explicitly requested attestation job.

### Fixed

- A nominally successful baseline result that omits its parsed benchmark entry now becomes
  a controlled baseline failure instead of relying on an optimization-removable assertion.

### Known limitations

- The initial beta support claim is limited to native Linux x86_64 and native Windows
  10/11 x86_64 using CPU-only or NVIDIA CUDA llama.cpp builds; the other implemented
  platforms and workflows named in the beta contract remain experimental or excluded.
- Recommendations are valid only for their recorded model, hardware, llama.cpp build,
  workload, depth, and context identity and require revalidation after any material change.
- llamatune does not download models, install or build llama.cpp, install GPU drivers, or
  modify clocks, governors, caches, drivers, fan controls, or other system state.

## 0.1.0-beta.1 (private candidate) - 2026-07-27

### Added

- Evidence-first tuning with measured baselines, bounded feasibility probes,
  staged search, confirmation, resumable journals, and reproducible
  recommendations.
- Dense and Mixture-of-Experts placement tuning, context/depth validation,
  thermal handling, bounded multi-GPU placement, and opt-in lossy KV quality
  evaluation.
- Night Shift, Marathon, result-matrix, registry, revalidation, and quality
  evaluation workflows.
- Tier-1 Linux and Windows execution contracts with bounded child runtime and
  output, process-tree cleanup, and allowlisted environments.
- Local dependency, static-analysis, and secret-audit tooling plus manual
  CodeQL and grouped Dependabot configuration.

### Fixed

- Synchronized live server and Python-sandbox output snapshots with their
  capture threads.
- Night Shift discovery now warns and skips a model removed between inspection
  and payload sizing instead of aborting the directory scan.
- Executor capture-worker failures now retain and surface their original cause
  instead of becoming an unrelated missing-result error.

[Unreleased]: https://github.com/barebonescyber/llamatune/compare/v0.1.0-beta.3...HEAD
[0.1.0-beta.3]: https://github.com/barebonescyber/llamatune/compare/5f63164d3a392a0e628e63448a8203b4dbe0d2f0...v0.1.0-beta.3
