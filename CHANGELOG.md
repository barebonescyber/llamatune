# Changelog

All notable changes to llamatune will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases use [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

No changes yet.

## [0.1.0-beta.2] - 2026-07-27

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

[Unreleased]: https://github.com/barebonescyber/llamatune/compare/v0.1.0-beta.2...HEAD
[0.1.0-beta.2]: https://github.com/barebonescyber/llamatune/releases/tag/v0.1.0-beta.2
