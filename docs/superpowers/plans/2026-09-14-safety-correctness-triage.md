# Safety and Correctness Triage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver small, independently reviewed safety and correctness repairs without importing the architecture and performance changes in PR #33.

**Architecture:** Split this campaign into safety containment, evidence correctness, and a separately gated GPU reliability lane. Reuse existing orchestration and artifact writers. Introduce shared helpers only when two repaired consumers need the same contract.

**Tech Stack:** Python >=3.11, uv, pytest, Ruff, mypy, Typer, GGUF, Rich, GitHub Actions. Use Python 3.12 for the local verification commands.

## Global Constraints

- `DESIGN.md` is authoritative. Feature-specific documents own only their named features.
- `bench` builds commands and parses output. It never executes processes.
- `executor` executes. It never interprets benchmark semantics.
- Only `session` writes inside a session directory, with path confinement. Preserve the existing feature-specific confined writers for Quality, Marathon, and Night Shift.
- `cli` stays thin. Heavy imports are lazy inside command bodies.
- Never `shell=True`. Use argv lists and allowlisted environments only.
- Bound every child's runtime and captured output.
- Model files and benchmark output are data, never executed.
- Never mutate system state (governors, clocks, caches, drivers).
- No network access in runtime code paths. Preserve the documented Quality loopback exception without adding network access.
- No credentials or environment values in evidence. Names only.
- No new runtime dependencies, no em dash characters, no unrelated formatting or renaming.
- No automatic PyPI publication, tag movement, release creation, workflow approval, issue closure, or native hardware run.
- Multi-GPU acceptance stays deferred. Preserve the published beta-3 CPU evidence rather than rewriting it for a new candidate.

---

## Evidence and execution baseline

Planning snapshot: 2026-09-14, America/Chicago. GitHub heads were re-read during planning.

| Source | Exact identity | Use |
| --- | --- | --- |
| Public main | `12485e742b12fe9ca0b9d296e04de427f077b09e` | Base for the first-wave repair plans |
| PR #33 | `72f92fc12fd2812d14d701d383ab95f00e6c8964` | Reference material, not an integration base |
| PR #42 | `ab9b0e7c9bf83eac838c32cda621cfe497fd8d44` | Separate unmerged GPU branch |
| Local checkout | `agent/enforce-gpu-layer-hard-cap` at `f545e2810777b4b7778a2db44014422279472486` | Planning documents only |

The local checkout is the private development repository, not public main. Its `public` remote points at `barebonescyber/llamatune`. Preserve existing `.gitignore` and `.claude/` changes. The owner authorized aligning `AGENTS.md` model definitions with Hermes-wide routing before execution.

Source line references in the subplans refer to the pinned public commit, not the current private branch. After execution approval, create an isolated worktree using the using-git-worktrees skill. Start from the public commit. Do not implement against the private branch or blindly cherry-pick either large PR.

The earlier exact-main audit passed 950 tests with two skips, 89.50% coverage, and all 21 module gates. Those are baseline results, not evidence that the proposed patches work. This planning run does not claim a new full-suite run.

Public sources: [issues](https://github.com/barebonescyber/llamatune/issues), [PR #33](https://github.com/barebonescyber/llamatune/pull/33), [PR #42](https://github.com/barebonescyber/llamatune/pull/42).

## Decisions and independently testable work packages

1. **Safety containment:** temporarily reject generated-code execution through CLI, Quality run/resume, and the sandbox entry point. This is a mitigation for #5/#28, not a claim that filesystem confinement now exists. No unsafe override flag.
2. **Evidence correctness:** repair Marathon's ignored middle records, recommendation flags, confirmation reuse, MoE patience, and Night Shift stop semantics in separate test cycles.
3. **Dependency security:** use already locked dependency versions as conservative lower bounds. Run the existing audit on PRs without changing its permissions or introducing `pull_request_target`.
4. **GPU reliability:** retain PR #42 as a separate lane until the merge blockers and admission-control contract have acceptance evidence. Do not call these PR-only defects released beta-3 regressions.
5. **Refactoring:** defer file splits, generalized caches, executor API changes, serialization consolidation, and micro-optimizations. A fix is not contingent on any of them.

Executable first-wave subplans:

- [Safety containment and dependency gates](2026-09-14-safety-containment.md): tasks S1-S3.
- [Evidence and resume correctness](2026-09-14-evidence-correctness.md): tasks E1-E5.

These plans intentionally do not claim to implement every reported backlog item. The next section assigns every open public issue a disposition. The later safety and GPU lanes have explicit acceptance contracts and remain open. They are not hidden inside a broad refactor.

## Complete issue triage

Priority here is maintainer triage, not a copy of the issue-title severity. P0 means immediate containment on the exposed opt-in path. P1 means repair before relying on the affected workflow. P2 is an independently scoped follow-up. Refactor is outside the repair campaign.

Evidence labels: **reproduced** means the previous audit exercised the failure offline. **source** means source inspection supports the finding. **reported** means this work has no independent native reproduction.

| Issue | Priority / evidence | Disposition and closure boundary |
| --- | --- | --- |
| #5 | P0, source | S1 contains generated-code execution. Keep open until mandatory network and filesystem confinement have adversarial tests. PR #33's optional filesystem confinement is insufficient. |
| #6 | Refactor | Dead-code cleanup in its own PR after repairs. No safety dependency. |
| #7 | Refactor | No universal utilities extraction. Only the runtime-flags helper in E2 is required now. |
| #8 | P1, reproduced | E1 stops silent loss of later Marathon records. Do not describe it as physical file truncation. Shared-reader unification remains out of scope. |
| #9 | P1, source | Separate output-boundary repair, contract O1. Raw captures remain unchanged. Do not claim renderer-dependent Markdown injection proves RCE. |
| #10 | P1, reproduced | E5 uses explicit user-interruption state. Breaker failure exits 1, user interruption exits 4. |
| #11 | Split | Exception sites that mask real failure belong in bug-specific PRs. Cast cleanup and typed-dict conversion wait. No blanket closure. |
| #12 | P1 for quality identity, P2 for traversal | Separate contracts O2/O3. Do not combine server authentication with discovery redesign. |
| #13 | P1, reproduced | E2 gives recommendation and export one runtime flag emitter, with exact argv parity tests. |
| #14 | Refactor | Defer executor API and clock abstraction redesign. Use existing monkeypatch/clock hooks for repair tests. |
| #15 | P2 | Context-probe caching is a separate performance change. Require context/config/build identity and unchanged budget accounting. |
| #16 | P1 prevention, source | S2 tests dependency floors. S3 enables the existing PR audit. Required-check policy needs separate maintainer authorization. |
| #17 | P1, reproduced | E3 reuses identity-bound confirmation records after disk reload. Revalidation remains fresh. CLI/perplexity stage caching is excluded. |
| #18 | Tracker | Keep open as a regression-evidence index. Link exact merged patches and tests, not contributor pass counts. |
| #19 | Split | Fix machine-output and exit-code defects independently. Naming and UX uniformity are not repair dependencies. |
| #20 | Refactor | Defer search-engine split. PR #33 explicitly leaves it incomplete. |
| #21 | P2 | Narrow discovery/version diagnostics after failures have explicit contracts. Do not add a broad catch-all handler. |
| #22 | P2 / architecture | Keep cache invalidation and matrix provenance separate from E1. Incremental caches require their own design and parity tests. |
| #23 | P2 | Help text and a CLI completeness test can land independently. |
| #24 | P2, reproduced | E2 removes duplicate `tb`/`ot` display tokens while touching the same formatter. |
| #25 | P2 presentation | Progress UI is not on the repair path. Preserve clean JSON stdout. |
| #26 | Refactor / performance | Keep partial micro-optimization work open. No blanket completion via PR #33. |
| #27 | Re-triage, reproduced | Public `apply()` filters missing metrics before ranking. No demonstrated CLI defect. Private-helper hardening is optional and separate. |
| #28 | P1, source | S1 prevents Darwin generated-code execution. Keep platform-enablement work open. Planned rlimits must never be reported as applied. |
| #29 | Documentation | Update the contributor module map after contracts settle. No helper-renaming dependency. |
| #30 | P2, source | Separate small calibration validation patch. Reject nonfinite or nonpositive reference pp/tg before a child starts, return existing `error` verdict with runs=0, and test zero/negative/NaN/infinity. |
| #31 | P1, reproduced | E4 excludes cached candidates from miss accounting and bounds revisits. Test both executed and replayed candidates. |
| #32 | Split | Scan failure exit semantics deserve a narrow patch. Table formatting and version spelling wait. |
| #35 | Re-triage, reported | A low observed VRAM delta does not prove total residency or spill. Keep telemetry-only anomalies advisory. See G1/G2. |
| #36 | P1, reproduced on PR #42 | G1 blocks merge: partial baselines must not be demoted as full offload, and bisection must not publish spilled placements as fitting. |
| #37 | P1 for affected resume, source on PR #42 | G3 requires stored custom build-directory discovery on tune/resume/revalidation. |
| #38 | P1 for affected hardware, source | G4 checks scaled timeouts plus remaining deadline and each retry budget. The PR's per-call change alone is not acceptance. |
| #39 | P1 evidence, source | G4 separates execution success, context validation, and budget cause. Preserve existing machine outcome compatibility. |
| #41 | P1, source on PR #42 | G2 blocks merge until journal reload restores divergence and placement demotions. Not an established beta-3 defect. |
| #43 | P1 before unattended GPU use, reported + source | G5 is fresh device-memory admission control. Do not reproduce by exhausting host GPU memory. |

## Later narrow safety contracts

These are separate work packages requiring implementation plans before coding, not refactor prerequisites. Keep #9 and #12 open when the first wave lands.

**O1, output rendering (#9):** add a pure display module with separate `terminal_text(value: str) -> str` and `markdown_text(value: str) -> str` functions. Terminal output must neutralize C0/C1 control sequences, including OSC links and carriage-return rewriting. Markdown must neutralize raw HTML, link/image syntax, backticks, table pipes, and line breaks in inline values. Cover `report.py`, `matrixreport.py`, `nightreport.py`, `marathonreport.py`, `qualityreport.py`, discovery warnings, and CLI display sinks. Preserve raw bounded capture files and structured values. Hostile-value tests must exercise public renderers as well as the sanitizer. Do not claim cross-renderer XSS prevention from HTML escaping alone.

**O2, local server identity (#12):** isolate changes to `qualserver.py`, Quality server-argv construction, and their tests. A successful `/health` response alone cannot authenticate the launched child. Test a competing loopback server, child exit during readiness, bounded retry exhaustion, and authenticated requests against the owned process. Never journal the API key or full secret-bearing argv. Resolve the exact llama-server credential transport before implementation, since copying a random key into recorded command evidence would violate this repository's safety contract.

**O3, discovery traversal (#12):** preserve documented symlink behavior until the maintainer selects an explicit root policy. Test file links, directory links, links outside the root, cycles, and split-model shards. Do not silently turn a read-discovery policy change into a path-confinement rewrite.

**S1 follow-up, execution re-enablement (#5/#28):** require an approved dedicated confinement design. Network isolation and filesystem read confinement must both be mandatory. Missing tools, denied namespaces, unknown memory enforcement, or unsupported platforms refuse execution. No global fallback switch. Verify host sentinel unreadability/unwritability, no external network, bounded children, and actual memory-limit application. Use synthetic sentinel files, never real secrets. A preflight label alone is not applied-isolation evidence.

## GPU lane acceptance contracts

Do not merge PR #42 or start unattended GPU acceptance until G1-G5 have explicit dispositions. These contracts constrain the revised PR, but do not authorize importing the current branch wholesale.

**G1, spill classification (#36):**

- Owned files: `src/llamatune/search.py:1150-1256,1356-1402,1404-1542`, `tests/unit/test_search.py` on the pinned PR head.
- Guard `_check_baseline_spill()` with both `_spill_applicable()` and `is_fully_offloaded(defaults.gpu_layers, self.model.ngl_all)` before any CPU reference or demotion.
- Pass every minimum-spill bisection success through the same `_probe_measured`, `_boundary_spill_suspected`, and `_handle_spill_suspected` decision path as ordinary boundary discovery.
- A `host_spill` frontier cannot create a fitting boundary before the measured frontier is checked. Preserve the last measured safe boundary. Do not fabricate `max_ok_ngl` by subtracting a layer.
- Test partial offload with CPU-class throughput, full-offload speed demotion, and fast GPU work with a tiny delta. Also test bisection with an `ok` raw probe and a `host_spill` measured frontier.

**G2, resume state (#35/#41):**

- Owned files: `search.py:439-470,2602-2644,2720-2758,2829-2853` and disk-backed resume tests.
- Restore divergence and demotion evidence before choosing candidates. Keep state bound to complete placement, workload/context, and build identity. Add missing context/build fields to new records with a conservative legacy policy.
- A later low-confidence observation cannot erase a measured failure. Do not infer spill from `vram_delta_anomaly`.
- Test interrupt, flush, `Session.load`, `resume_tuning`, candidate list parity, unchanged counted budget, and no inflated `max_fitting`. An in-memory engine-reconstruction test alone does not close this issue.

**G3, custom build discovery (#37):** pass `options.llama_bin` to `assess_hardware(llama_bin: Path | None = None) -> HardwareReport` on resume, and `session.options.llama_bin` on revalidation. Test a custom build absent from PATH, empty vendor probes, and a valid `--list-devices` response. The resumed search must retain its stored hardware identity.

**G4, deadline and provenance (#38/#39):** use one authoritative stop-cause contract with trial, time, and user-interruption reasons. Scale context timeout from measured prompt throughput. Cap every invocation and retry by its per-call ceiling and remaining deadline. Reserve cooldown/shutdown margin, count every executed retry once, and do not launch when the remaining budget is nonpositive. A halved-placement retry is a distinct configuration identity. Test minutes-only exhaustion, trial-only exhaustion, both exhausted, retry exhaustion, and interrupted retry reload. Night Shift must show context status separately from legacy execution outcomes. Only an `ok` context record linked to the recommended configuration and requested context permits the word `verified`.

**G5, admission control (#43):** fresh observation is an admission input, not a replacement for stored identity. Before automatic resume, compare the pending placement's conservative total requirement with `max(0, free_mb - reserve_mb)`. Reserve reduces capacity. Never add it to free memory. Do not substitute a device-used delta for total need. CPU-only sessions remain eligible. Unknown/stale/incompatible GPU observations or unknown requirement defer GPU resume. Journal an explicit deferred reason, continue other eligible items, and do not stop resident services. Bound retries by the existing deadline and avoid busy loops. Unit tests cover enough/insufficient/unknown headroom and changed device identity without allocating GPU memory. Native acceptance is a later owner-approved activity.

## Merge and model assignment

| Lane | Implementation | Review | Scheduling |
| --- | --- | --- | --- |
| S1 containment | Terra `high` | Sol `high` safety contract, Luna `high` CLI/docs | First |
| S2 dependency floors | Terra `high` | Sol `high` security/packaging | After S1 |
| S3 PR audit | Luna `medium` | Sol `high` workflow security | After S2, serialize workflow edits with #34 |
| E1 journal, E3 confirmation, E4 patience, E5 stop causes | Terra `high` | Sol `high` correctness, Luna `high` visible errors/reports | E3 then E4, one owner of `search.py` at a time |
| E2 flags/report parity | Terra `high` | Sol `high` data fidelity, Luna `high` reference-command presentation | Separate patch |
| Revised GPU lane | Terra `high` | Sol `high` placement/budget contract, Luna `high` evidence wording | Outside this first-wave execution |

The owner approved subagent-driven execution on 2026-09-14 and requested Hermes-wide model alignment, including GPT-6 Astra 900k XHigh. Astra (`openai-codex/gpt-6-astra-900k`, `xhigh`) now owns campaign orchestration, integration, and escalation. A fresh Astra session conducts final review. Sol retains independent safety/correctness review, Terra implements backend repairs, and Luna handles bounded workflow work and presentation review. The S3 assignment to Luna reflects its small, explicit patch.

`AGENTS.md` is the canonical repository routing definition. It includes the seven configured Hermes aliases and the explicit Astra 900k selector. Model, provider, context selection, and reasoning effort remain separate recorded fields. Use independent processes with explicit routes and effort, not profile-default `delegate_task` as a named-model substitute. No global Hermes configuration change is part of this approval.

Hotspots: `search.py`, `tests/unit/test_search.py`, `cli.py`, `report.py`, `types.py`, `DESIGN.md`, and `.github/workflows/security.yml`. PRs #33/#42 overlap in 21 files. Give one worker ownership of each hotspot during integration. Defer broad changes rather than repeatedly resolving overlapping edits.

PR #34 remains a standalone workflow-pin update. Coordinate its pins with S3, then test the exact integrated head. PR #3 is a separate low-urgency Hatchling floor refresh. Neither is permission to merge or approve fork workflows automatically.

## Gates and handoff

- [ ] At execution start, re-read public main and all referenced PR heads. If changed, rebase the plan's line anchors and reassess findings before coding.
- [ ] Carry this complete plan pack into each execution worktree. Uncommitted parent files are not inherited by a new worktree.
- [ ] Follow each subplan's RED, minimal patch, GREEN, focused review, and commit sequence. Commit commands are for the later authorized execution phase only.
- [ ] Run focused checks at each task boundary. At final integration, run `uv run ruff format --check .`, `uv run ruff check .`, `uv run mypy src tests`, and `uv run pytest`. On this host prefix with `env -u PYTHONPATH -u VIRTUAL_ENV` and use `uv run --locked --python 3.12`. This follows `AGENTS.md`: full gates once at the end.
- [ ] Run `uv run --locked --python 3.12 python .github/scripts/check_module_coverage.py coverage.json` after the full suite. Retain the 85% total and 80% per-module gates. Never waive them to make a subset pass.
- [ ] Review `git diff --check` and the exact file list. No unrelated files, dependency churn, changed runtime boundaries, or ignored regression failures.
- [ ] Keep partial issue dispositions explicit. A narrow E1 repair does not complete shared-reader unification. S1 containment does not complete secure execution support.
- [ ] Request exact-head CI only after fork workflow inspection and maintainer approval. `action_required` is neither PASS nor FAIL.
- [ ] Stop for review before release actions. A new candidate needs new CPU acceptance, then authorized single-GPU CUDA acceptance on Asgard. Multi-GPU remains deferred.

The owner authorized first-wave implementation with subagents, including its local task commits, on 2026-09-14. Stop after implementation and independent review. GitHub metadata writes, pushes, merges, releases, and hardware runs still require separate authorization.

## Planning verification

The refreshed inventory and this table contain the same 35 unique public issue numbers. All referenced existing files and relative plan links resolve. The eight first-wave tasks have explicit interfaces, code, RED/GREEN commands, and patch boundaries.

The plan's proposed test snippets ran against an unchanged disposable copy of public main: 31 expected failures, one passing signal-control case, and 191 deselected existing cases. Failures include missing planned APIs and policy guards as well as reproduced defects. They are RED-phase evidence, not 31 distinct bugs or a completed repair suite. Both confirmation budget variants failed as predicted. Production-source comparison verified all 36 Python files unchanged in the disposable copy.

Validation records remain under `/tmp/llamatune-status-927ckpuq/plan-red-validation/`: `plan-red-results.log`, `plan-test-selection.json`, and `verification-summary.json`. The plan pack passed code-fence parsing with its stated insertion contexts, file/link checks, placeholder checks, no-em-dash checks, and whitespace checks. Full GREEN, type, coverage, and hosted CI gates remain execution work.
