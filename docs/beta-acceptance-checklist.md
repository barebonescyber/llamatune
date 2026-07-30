# Public beta acceptance checklist

## Purpose

This is the auditable go/no-go checklist for the initial `llamatune` public beta. It applies
to one exact candidate commit and its immutable release artifacts. Evidence from another
commit may inform investigation but cannot satisfy a candidate gate unless the gate says
otherwise.

The support boundary is defined by [`beta-contract.md`](beta-contract.md). `DESIGN.md`
remains authoritative for product behavior.

## Candidate record

- Candidate version: `0.1.0b2` (SemVer release identity `0.1.0-beta.2`)
- Candidate state: **UNFROZEN**
- Candidate tag: _unset_
- Git commit SHA: _unset_
- Candidate build/run URL: _unset_
- Candidate artifacts and SHA-256 file: _unset_
- Acceptance start date: _unset_
- Acceptance owner: [`barebonescyber`](https://github.com/barebonescyber)
- Final decision: **NOT EVALUATED**
- Decision date and approver: _unset_

## How to use this checklist

- `[ ]` means the gate is open, failed, or lacks acceptable evidence.
- `[x]` means the gate passed for the exact candidate identified above.
- Every checked environment-sensitive gate must link to or name its CI run, session,
  acceptance report, or release artifact.
- A later code or packaging change invalidates affected checks. Re-run every gate that the
  change could influence.
- “Works in synthetic tests” does not substitute for a required native-hardware gate.
- Experimental and excluded surfaces cannot block the beta unless they regress a
  beta-supported path, violate a safety invariant, or are presented as beta-supported.

## Review governance

This is a solo-maintainer project. It does not claim independent third-party
review when none occurred. The release owner is the accountable human reviewer
and final approver; an AI-assisted technical second pass may inspect changes,
identify findings, and verify their disposition, but it does not replace or
misrepresent human approval. The PR review record and candidate evidence must
state this review model explicitly.

## Defect severity policy

Every candidate defect must be classified as P0, P1, or P2. Classification is based on the
highest credible impact within the beta-supported surface, not on how easy the fix appears.

### P0 — Critical integrity or safety failure

A P0 includes any of the following:

- Loss, overwrite, or corruption of user data outside the explicitly owned session output.
- Command execution outside the constructed argv contract, use of a shell, execution of
  model or benchmark output, credential disclosure, or session path escape.
- An incorrect recommendation caused by model, hardware, build, workload, or session
  identity confusion.
- Evidence represented as measured or confirmed when the measurement did not occur, came
  from a different identity, or was materially altered without an explicit schema process.
- An unbounded child runtime or output path that can make the supported workflow unsafe.
- A broadly reproducible failure that makes the release unsuitable for public testing on
  both supported operating systems.

P0 response:

- Stop release work and distribution of the affected candidate immediately.
- Mark affected recommendations and artifacts invalid until their provenance is proven.
- Fix the defect, add regression coverage, and repeat every affected acceptance campaign.
- Start a new release candidate. P0 defects cannot be waived.

### P1 — Release-blocking correctness or reliability failure

A P1 includes any of the following when no P0 impact is present:

- A beta-supported command cannot complete its documented primary workflow.
- Resume repeats completed work, loses required evidence, or cannot safely continue a
  session that is documented as resumable.
- An unsupported llama.cpp flag is passed rather than capability-gated.
- Search or statistical logic produces a false confirmed performance winner for the
  correct model and environment identity.
- A generated recommendation is unusable for the recorded configuration even though it
  does not create a safety or data-integrity risk.
- Timeout, interruption, or process-tree cleanup leaves benchmark children running.
- Linux or Windows required CI, artifact installation, or native acceptance fails
  reproducibly.
- Matrix or registry behavior silently returns an incompatible or stale result.

P1 response:

- Block the release.
- Fix the defect and add a focused regression test.
- Re-run all directly affected gates and the full required CI matrix.
- Reset the private release-candidate observation period when real-session correctness or
  reliability could have been affected. P1 defects cannot be waived.

### P2 — Non-blocking only after explicit disposition

A P2 includes defects with limited impact and a safe, documented workaround, such as:

- Missing or inaccurate non-authoritative telemetry that degrades with a warning.
- Report formatting, progress display, or help-text defects that do not change evidence or
  command behavior.
- Packaging metadata or documentation defects that do not prevent installation or safe
  use.
- A localized usability problem that does not prevent the supported workflow.
- An experimental-feature defect that does not affect a beta-supported or safety path.

P2 response:

- Triage before release; P2 does not mean “ignore.”
- Record impact, affected platform or command, workaround, owner, and target milestone.
- Document user-visible limitations in release notes when unresolved.
- Promote the issue to P1 if the workaround is unsafe, unreliable, undiscoverable, or
  prevents ordinary use of a beta-supported workflow.

### Defect gate

- [x] P0/P1/P2 definitions and response rules are documented here.
- [ ] All candidate issues have been reviewed and assigned a severity.
- [ ] Zero open P0 defects.
- [ ] Zero open P1 defects.
- [ ] Every open P2 has an owner, disposition, milestone, and user-facing documentation.
- [ ] No closed P0/P1 fix lacks a regression test and affected-gate rerun.

Evidence or issue-query URL: _unset_

## A. Contract and scope

- [x] The tracked public beta contract exists at `docs/beta-contract.md`.
- [x] The README distinguishes v1 design targets from the narrower beta claim.
- [x] Release notes repeat the supported, experimental, and excluded surfaces.
- [ ] Installation and usage examples do not imply support beyond the beta contract.
- [x] Docker and WSL2 are explicitly excluded from native Windows acceptance evidence.
- [ ] Experimental command output and documentation are labeled consistently.

Evidence: `docs/beta-contract.md`; `README.md`; candidate release notes pending

## B. Candidate source integrity

- [ ] Candidate version, tag, commit, build URL, and checksums are recorded above.
- [ ] The candidate commit is on the intended protected release branch.
- [ ] The repository is clean at tag creation.
- [ ] All intended changes have completed review.
- [ ] `DESIGN.md`, implementation, tests, and user documentation agree.
- [ ] `git diff --check` passes before the tag is created.
- [ ] The dependency lock is current and `uv lock --check` passes.
- [ ] No model, session, credential, local roadmap, or workstation-specific file is tracked
  or included unintentionally.

Evidence: _unset_

## C. Required automated validation

- [ ] `uv run ruff format --check .` passes.
- [ ] `uv run ruff check .` passes.
- [ ] `uv run mypy src tests` passes.
- [ ] `uv run pytest` passes with the required coverage threshold.
- [ ] Core-supported modules meet their per-module coverage target.
- [ ] Linux x86_64 / Python 3.11 required CI passes.
- [ ] Linux x86_64 / Python 3.12 required CI passes.
- [ ] Linux x86_64 / Python 3.14 required CI passes.
- [ ] Windows x86_64 / Python 3.11 required CI passes.
- [ ] Windows x86_64 / Python 3.12 required CI passes.
- [ ] Windows x86_64 / Python 3.14 required CI passes.
- [ ] macOS / Python 3.14 advisory CI passes.
- [ ] Three consecutive full required CI runs pass without test changes between runs.
- [ ] Test-order or repeated-test validation finds no reproducible flake.

Evidence: _unset_

## D. Safety and failure handling

- [ ] The no-shell, bounded-runtime, bounded-output, allowlisted-environment, and path-
  confinement invariants have been re-audited for the candidate.
- [ ] Hang, crash, malformed output, and oversized output scenarios produce bounded,
  documented outcomes.
- [ ] CUDA OOM and host-memory failure paths are classified conservatively.
- [ ] Missing model, missing binary, and unsupported capability paths return the documented
  exit codes.
- [ ] First interruption preserves a controlled best-so-far path and second interruption
  preserves resumability as designed.
- [ ] Corrupt or incomplete sessions are reported without unsafe writes or false results.
- [ ] Unwritable output and insufficient-disk scenarios fail without corrupting unrelated
  data.
- [ ] Matrix corruption cannot change the owning command's result or poison valid evidence.
- [ ] Dependency, static-analysis, secret-scanning, and CodeQL checks have acceptable
  results.

Evidence: _unset_

## E. Native Linux acceptance

- [ ] Acceptance uses native Linux x86_64, not a container or compatibility layer.
- [ ] CPU-only llama.cpp completes `scan`, baseline, tune, report, export, and matrix
  ingestion.
- [ ] NVIDIA CUDA llama.cpp completes the same core workflow.
- [ ] Stable and recent pinned llama.cpp builds are identified by commit and binary hash.
- [ ] Dense, MoE, memory-bound, sharded, and CPU-feasible reference models are covered.
- [ ] Full offload, partial offload, CPU MoE placement, and no-improvement outcomes are
  represented.
- [ ] Required and stretch context behavior is reported accurately.
- [ ] Interrupt/resume completes without duplicate executed evidence.
- [ ] Every claimed winner is independently reproduced above `max(2 × CV, 3%)`.
- [ ] Important cases remain consistent across two independent sessions and a cold reboot.
- [ ] No false confirmed winner, corrupt session, or missing required artifact is observed.

Acceptance report and session evidence: _unset_

## F. Native Windows acceptance

- [ ] Acceptance uses a physical native Windows 10/11 x86_64 installation, not WSL2 or
  Docker Desktop.
- [ ] The documented PowerShell installation succeeds in a clean environment.
- [ ] Real CPU-only `llama-bench.exe` completes the core workflow.
- [ ] Real CUDA-enabled `llama-bench.exe` completes the core workflow.
- [ ] Stable and recent pinned llama.cpp builds are identified by commit and binary hash.
- [ ] Paths with spaces, long components, another drive letter, non-ASCII text, read-only
  locations, and relative/absolute session roots behave conservatively.
- [ ] Timeout and interruption terminate a real descendant process tree without orphans.
- [ ] Resume, report regeneration, export, revalidation, registry lookup, and automatic
  matrix refresh complete successfully.
- [ ] GPU name and VRAM are recorded; unavailable telemetry degrades with explicit warnings.
- [ ] CUDA OOM is classified without Linux-only assumptions.
- [ ] Every claimed winner is independently reproduced above `max(2 × CV, 3%)`.
- [ ] No false confirmed winner, corrupt session, or missing required artifact is observed.

Acceptance report and session evidence: _unset_

## G. Evidence, registry, and Results Matrix

- [ ] Validated Linux and Windows sessions occupy distinct hardware groups.
- [ ] Different llama.cpp binaries remain distinct through the build discriminator.
- [ ] Model fingerprints deduplicate only compatible content.
- [ ] Registry lookup rejects incompatible model, hardware, build, workload, depth, and
  context requests.
- [ ] Baseline, recommendation, context-envelope, calibration, and supported matrix rows
  retain their source provenance.
- [ ] Corrupt or stale sessions are reported without contaminating valid rankings.
- [ ] Automatic refresh uses the canonical root unless a valid explicit multi-root matrix
  configuration already owns the artifact.
- [ ] A clean rebuild matches the incrementally refreshed matrix deterministically.

Evidence: _unset_

## H. Packaging, installation, and documentation

- [x] Candidate version follows the intended beta version scheme.
- [ ] Wheel and source distribution build from the immutable candidate tag.
- [ ] Artifact contents include required package data and exclude internal/local material.
- [ ] Wheel installation succeeds in clean Linux and Windows environments.
- [ ] Installed `llamatune --help` and `llamatune scan --json` succeed.
- [ ] An installed-wheel fake-benchmark end-to-end tune succeeds.
- [ ] Package metadata contains repository, issue tracker, documentation, Python, platform,
  maintainer, license, and keyword fields.
- [ ] Changelog describes the actual beta features and known limitations.
- [ ] `SECURITY.md`, support policy, compatibility policy, and issue templates are present.
- [ ] Every documented beta-supported command example has been exercised on its claimed
  operating system.
- [ ] Generated artifact SHA-256 checksums and attestations are available.

Evidence: _unset_

## I. Private release-candidate observation

- [ ] The exact built candidate was tested on another Linux/NVIDIA machine.
- [ ] The exact built candidate was tested on native Windows/NVIDIA.
- [ ] The exact built candidate was tested on a CPU-only machine.
- [ ] At least 20 successful independent tuning sessions were collected.
- [ ] At least five real interruption/resume scenarios completed.
- [ ] Defaults-optimal and OOM/fallback outcomes were observed and correctly represented.
- [ ] More than one hardware group was ingested without cross-group ranking contamination.
- [ ] The observation period completed without a P0 or P1 reset.

Evidence and observation dates: _unset_

## J. Final go/no-go decision

- [ ] All sections required for the beta-supported surface are complete.
- [ ] Zero open P0 and P1 defects are confirmed.
- [ ] Every remaining P2 is explicitly accepted and documented.
- [ ] Required CI is green on the exact candidate commit.
- [ ] Native Linux and native Windows acceptance reports are approved.
- [ ] Release artifacts, checksums, attestations, changelog, and known limitations are final.
- [ ] The candidate commit is frozen and no validation-invalidating change remains.
- [ ] Final approval is recorded in the candidate record.

If any item above is false, missing evidence, or invalidated by a later change, the final
decision remains **NO-GO**.
