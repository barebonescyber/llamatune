# llamatune Night Shift — Design, Architecture, and Requirements (v1)

Status: v1 specification, approved for implementation.
Audience: users, contributors, and implementers. This document is
authoritative for the Night Shift feature;
[DESIGN.md](../DESIGN.md) remains authoritative for everything it already
specifies. Implementation deviations must be documented and reviewed. Read
`AGENTS.md` before starting; its boundaries and safety invariants apply
unchanged.

## 1. Product definition

`llamatune nightshift` turns unattended machine time (overnight, workday)
into benchmark evidence. Given a directory of GGUF models, it:

1. **Discovers** every benchmarkable GGUF file in the directory.
2. **Classifies** each model against the existing session evidence:
   - never benchmarked → schedule a **full tune** (DESIGN §1 pipeline);
   - previously benchmarked → schedule a **verification calibration**
     (§6): re-measure the previously recommended configuration and compare
     against the recorded result;
   - a content-duplicate copy of an already-covered file (merged vs
     sharded, §4.1) → schedule a **transfer calibration** (§6.4) instead of
     a second full tune;
   - calibration drift beyond the noise-aware threshold → schedule a
     **full re-tune**.
3. **Iterates** until the shift deadline: resumes interrupted sessions,
   tunes new models, calibrates known ones, re-tunes drifted ones, and — when
   time remains — runs **deepening** re-tunes with escalated budgets so idle
   hours buy more trials, not idle silence.
4. **Reports**: one `nightshift-report.md` + `nightshift.json` summarizing
   what was done, what drifted, what was deferred, and why.

Night Shift is an orchestrator. It creates and consumes ordinary tuning
sessions through the existing `run_tuning`/`resume_tuning` entry points
(DESIGN §13.3); it invents no second benchmarking methodology. All
measurement semantics (baseline, noise floor, confirmation, classification)
remain exactly DESIGN §§6–11.

### 1.1 Non-goals (v1)

Concurrent benchmarking (execution is strictly serial — one `llama-bench`
process at a time, ever); daemonization, cron/systemd integration, or
scheduling itself (the user starts it; an external scheduler may start it);
per-model option overrides (one option set applies to every model in the
shift; listed as future work §15); notification delivery (email/push);
cross-night trend analysis; mutating, moving, or deleting model files or
prior sessions; network access.

## 2. Definitions

- **Model identity**: the DESIGN §5 fast fingerprint (sha256 over first
  1 MiB + last 1 MiB + size + metadata keys). Two files with equal
  fingerprints are the same model regardless of path. A file whose content
  changed has a new fingerprint and is a new model.
- **Previously benchmarked**: at least one *completed* session (§5.2) exists
  under `--sessions-dir` whose recorded `model.json` fingerprint equals the
  discovered file's current fingerprint.
- **Reference result**: from the latest completed session for a fingerprint —
  the confirmed winner's configuration and confirmation means when a winner
  exists, otherwise the bare-defaults baseline and its means
  (exit-1 "defaults optimal" sessions are valid references).
- **Verification calibration**: a short re-measurement of the reference
  configuration under current conditions, compared per-metric against the
  reference means (§6).
- **Content group**: files whose GGUF identity matches per §4.1 — the same
  weights in different file layouts, typically a sharded set and its merged
  single-file copy. One member is the **representative**; the rest are
  **siblings**.
- **Transfer calibration**: a verification calibration of a sibling file
  against its group representative's reference result (§6.4), recording
  per-fingerprint evidence that the recommendation transfers.
- **Drift**: relative per-metric deviation of the calibration means from the
  reference means, in either direction (§6.3).
- **Shift deadline**: the wall-clock instant Night Shift must stop starting
  new work (§7.1).
- **Work item**: one schedulable unit — `resume`, `tune`, `calibrate`,
  `retune`, or `deepen` (§7.2).

## 3. CLI specification

One new Typer command in `cli.py` (thin, lazy imports, per DESIGN §3):

```
llamatune nightshift MODELS_DIR [options]
```

- `MODELS_DIR` (positional, required) — directory scanned recursively for
  `*.gguf` files. Example on this workstation: `./models` or `/models`.
- `--llama-bin DIR` — as in `tune`.
- `--sessions-dir DIR` (default `./llamatune-sessions`) — shared with `tune`;
  Night Shift both reads prior sessions from here and creates new ones here.
- `--until HH:MM` — local-time deadline; a value at or earlier than the
  current time means the same time **tomorrow** (start 22:30 with
  `--until 07:00` → next morning).
- `--max-hours F` — duration cap. If both `--until` and `--max-hours` are
  given, the earlier deadline wins. If **neither** is given, the shift runs
  exactly one pass through phases 0–3 (§7.2) and exits (no deepening; never
  unbounded).
- `--profile deep|standard` (default `deep`) — budget profile, §8.
- `--drift-threshold F` (default `0.05`) — calibration drift floor, §6.3.
- `--calibration-runs N` (default `3`, minimum `2`) — independent
  invocations per calibration.
- `--duplicates one|both` (default `one`) — §4.1 policy for content
  duplicates: `one` fully tunes only each group's representative and gives
  the other members transfer calibrations; `both` benchmarks every member
  independently.
- `--include GLOB` / `--exclude GLOB` (repeatable) — filename filters applied
  to the GGUF basename during discovery; include filters (when any are
  given) are applied first, then excludes.
- `--dry-run` — print the plan (phases, items, estimates, reasons) and exit 0
  without executing any benchmark.
- Pass-through tuning options, applied uniformly to every tune-class item:
  `--target`, `--allow-lossy`, `--ctx-size N[,N...]`, `--vram-reserve-mb`,
  `--cooldown`, `--full-hash`, and explicit overrides of the profile's
  `--budget-trials`, `--reps-search`, `--reps-confirm`, `--baseline-runs`.
  The first context is required and later ascending values form the stretch
  validation ladder, matching the ordinary `tune` command.
- `--json` — machine-readable summary (the `nightshift.json` content) on
  stdout instead of human text.

Validation errors (bad `--until` format, negative thresholds,
`--calibration-runs < 2`) exit 2 with a one-line `error:` message, matching
existing CLI conventions.

### 3.1 Exit codes

- `0` — shift completed; every item that was started succeeded. Items
  deferred because the deadline arrived are normal and still exit 0 (they
  are listed in the report).
- `1`: shift completed but at least one started item failed (a tune exited
  3, a calibration errored, or the consecutive tune-failure circuit breaker
  tripped). Evidence and the report are still written.
- `2` — usage or configuration error.
- `3` — environment error: `llama-bench` missing/unusable, `MODELS_DIR`
  missing or unreadable, or no GGUF file survives discovery/filters.
- `4`: user interruption or an interrupted child session (§7.5). Re-running
  the same command resumes naturally (§7.6).

Night Shift exits 4 for user interruption or an interrupted child session.
The consecutive tune-failure circuit breaker exits 1 and records a failed window.
A stop caused only by the work window ending is not a user interruption.

## 4. Model discovery (`discovery.py`)

`discover_models(models_dir, include, exclude, *, duplicates="one",
full_hash=False) -> tuple[DiscoveredModel, ...]`, pure aside from
filesystem reads.

- Recursively glob `*.gguf` (case-insensitive suffix) under `MODELS_DIR`.
  Symlinks to files are followed; symlinked directory cycles must not hang
  (track resolved real paths).
- **Shard handling**: files matching the llama.cpp shard convention
  `-<idx>-of-<total>.gguf` (five-digit zero-padded, e.g.
  `Qwen3-Coder-Next-UD-Q5_K_XL-00001-of-00003.gguf`) are grouped; only the
  `00001` head is a benchmarkable entry (llama.cpp loads the rest
  automatically). Non-head shards are never inspected or listed as models.
  A shard group with a missing head or missing members is skipped with a
  warning naming the missing pieces.
- A merged single-file copy of a sharded model (e.g. `...-merged.gguf`) has
  a different fingerprint but identical weights; such **content duplicates**
  are grouped, and by default only one member per group is fully tuned
  (§4.1).
- Each surviving entry is inspected via the existing
  `model.inspect_model(path, full_hash=...)` (DESIGN §5). Inspection failure
  skips the file with a warning; it never aborts discovery.
- Entries with identical fingerprints (true duplicate files) are
  deduplicated to the lexicographically first path, with a warning.
- Deterministic order: sorted by path.

```python
DiscoveredModel(path: Path, report: ModelReport,
                shard_paths: tuple[Path, ...],   # empty for single files
                group_key: str | None,           # §4.1; None = ungrouped
                representative: bool)            # True when ungrouped or --duplicates both
```

### 4.1 Content groups and duplicate policy

The same weights often exist twice in a models directory — a sharded set
plus its merged single-file copy. Their fingerprints differ (different file
layout) but the tensors that reach RAM/VRAM are identical, so a second full
tune would spend hours re-learning a known result. Discovery therefore
groups **content duplicates** after include/exclude filtering (excluding a
member changes the representative choice).

Two entries belong to one content group when **all** of the following hold:

- `general.architecture`, `general.name`, `n_layer`, and `expert_count`
  are equal;
- their tensor tables match: identical multiset of (tensor name, shape,
  type) tuples, compared via `tensor_table_sha` = sha256 over the sorted
  tuples; `group_key` is its first 16 hex digits. `discovery.py` computes
  this itself via the `gguf` package (header/tensor table only — tensor
  data is never read; DESIGN §5 safety rules apply); `ModelReport` is not
  extended;
- total payload sizes agree within 1% (merged file vs the sum of the shard
  set's file sizes; header overhead accounts for the slack).

When the tensor table cannot be interpreted for an entry (the DESIGN §5
fallback case), that entry is never grouped — the conservative failure mode
is an extra full tune, never a wrong skip.

Policy (`--duplicates`, default `one`):

- `one`: exactly one **representative** per group is eligible for full
  tunes — the merged single file when the group has one, otherwise the
  lexicographically first path. Every other member is a **sibling**: it
  never gets a full tune by default; instead, each shift it receives a
  **transfer calibration** (§6.4) against the representative's reference
  result, which records cheap per-fingerprint evidence that the
  recommendation transfers. A user who prefers the sharded set as
  representative `--exclude`s the merged file.
- `both`: grouping is still computed and reported, but every member is
  treated as a representative (independent full tunes).

The known blind spot — two genuinely different fine-tunes exported with
identical metadata, quantization, and tensor table — is rare and covered by
`--duplicates both`. The report names every group and its chosen
representative.

## 5. Benchmark registry (`registry.py`)

The registry is **derived, never stored**: every invocation rebuilds it by
scanning `--sessions-dir`. There is no database and no cache file; the
session evidence remains the single source of truth (evidence-first,
DESIGN §12). This also makes Night Shift naturally re-entrant (§7.6).

### 5.1 Scan

For each child directory of `--sessions-dir` (skipping the `nightshift/`
subtree, §9): read `session.json`, `model.json`, `llamacpp.json`,
`analysis.json` when present. Unreadable or unparseable directories are
skipped with a warning, never fatal.

### 5.2 Completed vs incomplete

- **Completed**: `analysis.json` exists and parses, and `journal.jsonl`
  contains an `analysis_written` or `session_end` entry.
- **Incomplete**: `session.json` exists but the session is not completed.
  These are resume candidates (phase 0). The torn-tail rule of DESIGN §12
  applies when reading journals.

### 5.3 Records

`build_registry(sessions_dir) -> dict[str, RegistryRecord]` keeps only the
**latest** completed session per fingerprint (by `session.json` `created`).
`incomplete_sessions(sessions_dir) -> tuple[Path, ...]` returns incomplete
session directories, oldest first.

```python
RegistryRecord(
    fingerprint: str,
    session_dir: Path,
    created: str,                       # ISO-8601 UTC from session.json
    outcome: str,                       # "winner" | "defaults_optimal"
    reference_config: TrialConfig | None,  # None => bare llama-bench defaults
    reference_pp: float,                # confirmation (winner) or baseline mean
    reference_tg: float,
    noise_floor_cv: float,
    pp_workload: int, tg_workload: int, # recorded session options
    reps_confirm: int,
    target: str,
    build_commit: str | None,
    help_sha256: str,
    median_trial_wall_s: float | None,  # from journal `trial` entries
    session_wall_s: float | None,       # journal first→last timestamp span
)
```

`reference_config` for a winner comes from `analysis.json`
`winner.config` via `TrialConfig.from_dict`; reference means come from
`winner.confirmation`. For `defaults_optimal`, `reference_config` is `None`
and reference means are `baseline.pp.mean` / `baseline.tg.mean`.

## 6. Verification calibration (`calibrate.py`)

### 6.1 Method

Re-measure the reference under current conditions, exactly comparable to how
it was recorded:

- Workload and repetitions come from the **record**, not the shift options:
  `-p pp_workload -n tg_workload -r reps_confirm -o json`.
- Flags: `reference_config.bench_args(current_capabilities)` when a config
  exists; **no tuning flags at all** when `reference_config is None`
  (identical to the DESIGN §6 baseline invocation shape).
- `--calibration-runs` (default 3) independent invocations, executed through
  the existing `executor` contract with the baseline fixed timeout (1800 s)
  and `cooldown_s` honored between invocations. Load average is recorded
  before each run (DESIGN §4).
- Calibration means are the mean of per-run `avg_ts` means, mirroring the
  baseline aggregation.

### 6.2 Comparability gate

If the current capability probe no longer supports a flag that
`reference_config` requires (`bench_args` would silently drop it, DESIGN
§13.1), the calibration is **not comparable**: verdict `error`, reason
`capability_lost:<flag>`, and a full re-tune is enqueued. A changed
`build_commit` or `help_sha256` does *not* block calibration — behavioral
measurement is the point — but the change is recorded and surfaced in the
report as the likely drift explanation.

### 6.3 Drift verdict

For each metric `m ∈ {pp, tg}` with reference mean `ref_m` and calibration
mean `cal_m`:

```
drift_m = |cal_m − ref_m| / ref_m
τ       = max(drift_threshold, 2 × noise_floor_cv_recorded)
```

- Both metrics within `τ` → verdict **`consistent`**; the recorded
  recommendation stands.
- Either metric beyond `τ` → verdict **`drift`**; a full re-tune is
  enqueued. Drift is two-sided by design: a machine that got *faster*
  (driver/build upgrade) also invalidates the recommendation, because a
  better configuration may now exist.
- Any run failing (nonzero exit, parse error, timeout) → verdict
  **`error`** with the DESIGN §8 classification as the reason; a full
  re-tune is enqueued (a reference config that no longer runs is the
  strongest possible drift signal).

```python
CalibrationResult(
    fingerprint: str, reference_session: Path,
    verdict: str,                    # "consistent" | "drift" | "error"
    pp: MetricStats | None, tg: MetricStats | None,
    drift_pp: float | None, drift_tg: float | None,
    threshold: float, runs: int,
    reason: str | None,              # error reason / capability_lost:<flag>
    transfer_from: str | None,       # §6.4 representative fingerprint; None otherwise
    build_changed: bool,
    artifact_dir: Path | None,
)
```

Entry point (frozen interface):

```python
run_calibration(run: NightshiftRun, record: RegistryRecord,
                model: ModelReport, llama: LlamaCppReport,
                options: NightshiftOptions) -> CalibrationResult
```

`run_calibration` must not assume `record.fingerprint` equals the model's
fingerprint — for transfer calibrations (§6.4) they intentionally differ.

### 6.4 Transfer calibration (content-group siblings)

A sibling (§4.1) is calibrated with exactly the §6.1–6.3 method, except the
reference `RegistryRecord` is its group **representative's** while the
benchmarked file is the sibling's own. `transfer_from` records the
representative fingerprint. Verdict actions:

- `consistent` → the recommendation demonstrably transfers to the sibling
  file; recorded in `nightshift.json` and the report. The sibling still has
  no session of its own — it is re-verified against the representative each
  shift, which is cheap and is its ongoing calibration coverage.
- `drift` or `error` → the sibling behaves meaningfully differently from
  its twin (plausible causes: multi-file vs single-file mmap behavior, a
  damaged copy), so a full tune **of the sibling itself** is enqueued
  (§7.2 phase 3). Once that completes, the sibling has its own registry
  record and is treated as an ordinary benchmarked model on later shifts —
  a registry record always takes precedence over grouping policy.

## 7. Orchestration (`nightshift.py`)

### 7.1 Deadline

`deadline = min(until_resolved, start + max_hours)` over the options that
were provided; `None` when neither was. All scheduling decisions use an
injectable clock (`now_fn: Callable[[], datetime]`, defaulting to
`datetime.now(UTC)`) so tests never sleep.

### 7.2 Phases and the plan

Work items are processed strictly serially in phase order. The initial plan
(phases 0–2) is computed up front from discovery + registry and journaled;
phases 3–4 are populated dynamically.

- **Phase 0 — Resume**: every incomplete session whose fingerprint matches a
  discovered in-scope model, oldest first, via `resume_tuning` (which
  re-validates identity per DESIGN §12).
- **Phase 1 — Tune**: full tune for every discovered **representative**
  (§4.1) with no registry record, smallest `size_bytes` first (more models
  finished per night). A model with its own completed session is always an
  ordinary benchmarked model, even when it is a group sibling.
- **Phase 2 — Calibrate**: for every model with its own registry record, a
  verification calibration against that record, stalest `created` first;
  then a transfer calibration (§6.4) for every sibling without a record
  whose representative has one. A sibling whose representative is only
  being tuned in phase 1 of this shift has its transfer calibration
  enqueued dynamically when that tune completes with a usable result; if
  the representative's tune fails, the sibling is deferred with that
  reason.
- **Phase 3 — Retune**: full re-tune (or, for a sibling, first-ever own
  tune) for every model whose phase-2 verdict was `drift` or `error`, in
  verdict order.
- **Phase 4 — Deepen** (only with a deadline, and only while time remains):
  rotate over all in-scope models that have their own completed session
  (siblings without one are excluded), least-recently-tuned first, running
  at most one full re-tune per eligible model with the deepening budget (§8).
  Skip this phase when explicit overrides make its resolved profile identical
  to the initial tune. Each completed deepening session becomes that model's
  new registry reference on the next shift.

### 7.3 Time fitting

Tune-class items (`resume`, `tune`, `retune`, `deepen`) are launched with
`budget_minutes = remaining_minutes − SHUTDOWN_MARGIN_MIN`, so the existing
budget machinery concludes them gracefully before the deadline; they are
started only when `remaining_minutes ≥ MIN_TUNE_MINUTES`. Calibrations are
started only when the estimate
`calibration_runs × (median_trial_wall_s or 300 s) × 1.5` fits the remaining
time. Items that do not fit are recorded as `deferred` with the estimate
that excluded them, and the scheduler moves on (a smaller later item may
still fit). Module constants, documented in the report:
`MIN_TUNE_MINUTES = 20`, `SHUTDOWN_MARGIN_MIN = 5`.

### 7.4 Item execution

- One hardware assessment and one llama.cpp discovery run at shift start
  (recorded in the run dir); hardware is re-assessed before each tune-class
  item so per-session evidence stays honest about overnight conditions.
- Tune-class items construct `TuneOptions` from the profile (§8) plus
  pass-throughs and call the existing `Session.create` + `run_tuning`
  exactly as `cli.tune` does. Sessions land directly under `--sessions-dir`
  as ordinary sessions.
- A failed item (exit 3/4 from tuning, calibration `error`) is recorded and
  the loop **continues with the next item** — one broken model must never
  cost the night. Consecutive-failure circuit breaker: 3 consecutive
  tune-class failures across *different* models abort the shift with exit 1
  (the environment, not the models, is the likely fault).

### 7.5 Signals

First `SIGINT`/`SIGTERM`: set the stop flag; the in-flight `llama-bench`
invocation finishes (or the executor's own timeout handling applies), the
current item journals its state, the report is written, exit 4. Second
signal: the executor's process-group termination applies (DESIGN §7); the
run journal gets an `interrupted` entry; partial sessions remain resumable.

### 7.6 Re-entrancy

Night Shift keeps **no** cross-run state of its own. Because the registry is
derived from session evidence at startup, re-running the identical command
after any interruption naturally does the right thing: completed work is
found in the registry, the interrupted session appears in phase 0, and
untouched models appear in phase 1/2.

## 8. Budget profiles ("more iterations")

Night Shift exists to spend idle hours on statistical depth. Profile values
(overridable individually via pass-through flags):

| option | `standard` | `deep` (default) | deepening item (phase 4) |
|---|---|---|---|
| `budget_trials` | 60 | 120 | 240 |
| `reps_search` | 3 | 5 | 5 |
| `reps_confirm` | 5 | 8 | 8 |
| `baseline_runs` | 3 | 5 | 5 |
| `cooldown_s` | 0.0 | 5.0 | 5.0 |

`standard` equals the existing `tune` defaults (useful for a short evening
window). `deep` roughly doubles evidence per model. Deepening doubles the
trial budget again so phase 4 explores meaningfully rather than repeating
phase 1. `budget_minutes` is always scheduler-supplied (§7.3) and never part
of the profile.

## 9. Evidence: run directory, journal, report

```
<sessions-dir>/nightshift/<UTCyyyymmdd-hhmmss>-<6hex>/
  run.json            # schema_version 1, tool version, argv, options,
                      # resolved deadline, profile, created (UTC)
  hardware.json       # shift-start assessment
  llamacpp.json       # shift-start probe
  plan.json           # initial phases 0–2 with per-item reasons/estimates
  journal.jsonl       # append-only, one JSON object per line, fsync per line
  calibrations/<fingerprint16>/run-<n>/   # command.json, stdout.json, stderr.log
  nightshift.json     # final machine-readable summary (== --json stdout)
  nightshift-report.md
```

Journal entry types (every entry carries a UTC timestamp):
`nightshift_start`, `plan`, `item_start`, `item_end` (with outcome, session
dir or calibration verdict, wall seconds), `calibration`, `retune_enqueued`,
`deferred`, `interrupted`, `nightshift_end`.

Write confinement: all writes inside the run directory go through the
`NightshiftRun` writer, which enforces the same resolve-and-verify path
confinement as `Session` (DESIGN §12). The DESIGN §13 layering sentence
extends to: only `session` writes inside a session directory; only
`nightshift`'s `NightshiftRun` writes inside a nightshift run directory.

`nightshift-report.md` sections: **Shift summary** (window, deadline
outcome, counts per phase, total invocations, machine + build identity,
build-change warnings, content groups with their chosen representatives);
**Per-model table** (name, fingerprint prefix, action(s), verdict/result —
e.g. `consistent (pp −1.2%, tg +0.4%)`,
`drift (tg −9.8%) → retuned, new winner +6.1%`, `tuned, winner +12%`,
`transfer-consistent vs a1b2c3d4 (pp +0.3%, tg −0.9%)`, `failed: oom`);
**Deferred and skipped** (item, reason, estimate); **Warnings**. Rendering consumes only `nightshift.json` content (mirroring
`report.py`'s analysis-only rule) so the report is regenerable.

`nightshift.json` (schema_version 1): options, window, per-item records
(kind, model, fingerprint, session_dir/calibration data, outcome, wall_s),
aggregate counts, warnings. `NightshiftOutcome(run_dir, summary: dict,
exit_code: int)` is the entry-point return type.

## 10. Architecture

New modules and touched files:

```
src/llamatune/
  discovery.py    # §4  — GGUF enumeration, shard grouping (pure + fs reads)
  registry.py     # §5  — evidence scan → RegistryRecord (read-only)
  calibrate.py    # §6  — calibration execution + drift verdict
  nightreport.py  # §9  — nightshift-report.md rendering from nightshift.json
  nightshift.py   # §7  — NightshiftRun writer, planner, scheduler, signals,
                  #       run_nightshift(options, *, now_fn=None) -> NightshiftOutcome
  types.py        # additions only (§10.1)
  cli.py          # new `nightshift` command only
tests/
  fixtures/fake_llama_bench.py   # one new knob (§13.1)
  unit/test_discovery.py  test_registry.py  test_calibrate.py
  unit/test_nightreport.py  test_nightshift_plan.py  test_cli_nightshift.py
  integration/test_nightshift.py
```

Layering (extends DESIGN §13): `discovery` and `registry` read the
filesystem but never execute processes and never write; `calibrate` executes
only via `executor` and writes only via `NightshiftRun`; `nightreport`
renders strings from dicts, no I/O; `nightshift` orchestrates and owns the
only writer for its run directory; `cli` stays thin with lazy imports.
No new runtime dependencies.

### 10.1 `types.py` additions (frozen, slots, kw_only — exact fields)

`DiscoveredModel` (§4), `RegistryRecord` (§5.3), `CalibrationResult` (§6.3),
plus:

```python
NightshiftOptions(
    models_dir: Path, llama_bin: Path | None, sessions_dir: Path,
    until: str | None, max_hours: float | None,
    profile: str, drift_threshold: float, calibration_runs: int,
    include: tuple[str, ...], exclude: tuple[str, ...], dry_run: bool,
    target: str, allow_lossy: bool, ctx_size: int | None,
    vram_reserve_mb: int | None, cooldown_s: float | None, full_hash: bool,
    budget_trials: int | None, reps_search: int | None,
    reps_confirm: int | None, baseline_runs: int | None,   # None => profile value
)

WorkItem(kind: str,               # "resume"|"tune"|"calibrate"|"retune"|"deepen"
         model_path: Path | None, fingerprint: str | None,
         session_dir: Path | None,           # resume target
         reference_fingerprint: str | None,  # §6.4 transfer source; None otherwise
         estimated_minutes: float | None, reason: str)

NightshiftOutcome(run_dir: Path, summary: dict[str, Any], exit_code: int)
```

Existing dataclasses are **not** modified.

### 10.2 Frozen cross-module interfaces

These signatures are contracts; changing one requires a documented design
amendment and review of every affected module:

```python
discovery.discover_models(models_dir, include, exclude, *,
                          duplicates="one", full_hash=False)
    -> tuple[DiscoveredModel, ...]
registry.build_registry(sessions_dir) -> dict[str, RegistryRecord]
registry.incomplete_sessions(sessions_dir) -> tuple[Path, ...]
calibrate.run_calibration(run, record, model, llama, options) -> CalibrationResult
nightreport.render(summary: dict) -> str
nightshift.run_nightshift(options: NightshiftOptions, *, now_fn=None)
    -> NightshiftOutcome
NightshiftRun.create(sessions_dir, *, options, hardware, llama, argv)
NightshiftRun.load(run_dir); .dir; .append(entry: dict)
NightshiftRun.calibration_dir(fingerprint16: str, n: int)
NightshiftRun.write_json(name: str, payload: dict); .write_text(name, text)
```

## 11. Safety invariants

All of DESIGN §14 unchanged, plus: Night Shift never deletes or rewrites any
session, model file, or prior nightshift run; it is strictly additive on
disk. No prompts, no stdin reads — the run must complete unattended. Exactly
one child benchmark process at any moment. Signal handling never leaves a
child process orphaned (executor's process-group contract).

## 12. Requirements checklist

Functional:

- FR-1 `llamatune nightshift MODELS_DIR` exists with the §3 options; `--json`
  and human output both supported; validation errors exit 2.
- FR-2 Discovery finds nested GGUFs, benchmarks shard heads only, skips
  broken shard groups with warnings, dedups identical fingerprints, and
  groups content duplicates per §4.1.
- FR-3 A model with no completed matching-fingerprint session gets a full
  tune with profile budgets.
- FR-4 A model with a completed session gets a calibration that replays the
  recorded workload/reps/config through the executor.
- FR-5 Drift verdicts follow §6.3 exactly, two-sided, with
  `τ = max(drift_threshold, 2 × noise_floor_cv)`.
- FR-6 `drift` and `error` verdicts enqueue and (time permitting) execute a
  full re-tune in the same shift.
- FR-7 Incomplete sessions for in-scope models are resumed before new work.
- FR-8 With a deadline: no tune-class item starts inside the final
  `MIN_TUNE_MINUTES`; tune-class items receive
  `budget_minutes = remaining − SHUTDOWN_MARGIN_MIN`; non-fitting items are
  journaled as deferred; the process exits by the deadline plus the margin.
- FR-9 With no deadline: exactly one pass through phases 0–3, then exit.
- FR-10 With a deadline and spare time: deepening re-tunes at the escalated
  budget, least-recently-tuned first, at most once per eligible model per
  shift; an identical resolved profile is skipped.
- FR-11 One model's failure never aborts the shift (subject to the 3-strike
  circuit breaker); exit codes follow §3.1.
- FR-12 First signal → graceful conclusion, report written, exit 4;
  re-running the same command afterward resumes naturally with no
  nightshift-specific state.
- FR-13 Run directory, journal, `nightshift.json`, and
  `nightshift-report.md` per §9; report regenerable from `nightshift.json`
  alone.
- FR-14 `--dry-run` prints the full plan and executes nothing.
- FR-15 With `--duplicates one` (default), at most one full tune per
  content group; siblings receive transfer calibrations whose `consistent`
  verdict is recorded and whose `drift`/`error` verdict enqueues the
  sibling's own full tune; a sibling with its own completed session is
  treated as an ordinary benchmarked model; `--duplicates both` restores
  independent benchmarking of every member.

Non-functional:

- NFR-1 Serial execution; no daemons, threads only if signal handling
  requires none (prefer signal handlers + flags).
- NFR-2 Injectable clock for every scheduling decision; the test suite never
  sleeps for scheduling reasons and never runs a real model or GPU.
- NFR-3 All existing quality gates pass repo-wide: `ruff format --check`,
  `ruff check`, `mypy` (strict), `pytest` with branch coverage ≥ 85%.
- NFR-4 No new runtime dependencies; Python ≥ 3.11.
- NFR-5 Existing commands' behavior and outputs are byte-for-byte unchanged
  except for the added CLI command.

## 13. Testing strategy

### 13.1 Fixture extension (normative)

`fake_llama_bench.py` gains one knob: `LLAMATUNE_FAKE_SPEED_SCALE` (float,
default `1.0`) — multiplies every emitted `avg_ts` after the existing
performance model. This lets tests record a session at scale 1.0, then run a
calibration at 0.85 (drift) or 1.0 (consistent) without touching evidence.

### 13.2 Unit

- Discovery: shard grouping (head-only selection, missing-member warning),
  include/exclude, duplicate-fingerprint dedup, symlink cycle safety
  (tiny GGUFs from the existing `conftest.py` writer, plus zero-byte decoys
  that must be skipped-with-warning); content grouping (§4.1): sharded set
  + merged copy grouped with merged as representative; same name but
  different quantization not grouped (tensor types differ); tensor table
  unavailable → not grouped; size outside 1% tolerance → not grouped;
  `--exclude` of the merged file shifts the representative to the shard
  head; `--duplicates both` marks every member representative.
- Registry: synthetic session dirs — winner session, exit-1
  defaults-optimal session, incomplete session, corrupt `analysis.json` —
  yield correct records / resume candidates / warnings; latest-wins per
  fingerprint; `nightshift/` subtree ignored.
- Calibrate: drift math boundary cases (exactly at τ is `consistent`; τ
  driven by noise floor vs by `drift_threshold`), two-sided detection,
  `capability_lost`, error classification pass-through; transfer support:
  record fingerprint ≠ model fingerprint sets `transfer_from`, ordinary
  calibrations leave it `None`.
- Planner: phase ordering, deadline fitting (fake clock), deferral,
  `--until` next-day wrap, no-deadline single pass, deepening rotation
  order, circuit breaker.
- Report: golden-ish structural assertions on section presence and verdict
  strings from a canned `nightshift.json`.

### 13.3 Integration (fake bench + tiny GGUFs, no GPU, no network)

1. Fresh models dir, no sessions → phase 1 tunes every model; registry on a
   second run classifies all as benchmarked.
2. Pre-recorded session, `SPEED_SCALE=1.0` → calibration `consistent`, no
   retune, exit 0.
3. Pre-recorded session, `SPEED_SCALE=0.8` → `drift`, retune executes, new
   session appears, exit 0.
4. Deadline already nearly exhausted (fake clock) → items deferred, report
   lists them, exit 0.
5. SIGTERM mid-shift → exit 4; rerun resumes the interrupted session and
   completes.
6. One model's bench forced to fail (`LLAMATUNE_FAKE_GENUINE_FAIL`) → shift
   continues, exit 1, failure in report.
7. Sharded set + merged copy, no sessions → the first shift runs exactly
   one full tune (the representative) and one dynamically enqueued transfer
   calibration (`consistent`); a second shift runs one own-record
   calibration plus one transfer calibration and zero tunes.

## 14. Documentation

README gains a "Night Shift" section (command synopsis, one worked example
using `./models`, deadline semantics, where the report lands). DESIGN.md may
be updated only to align Night Shift exit-code documentation. This document
remains the Night Shift spec.

## 15. Future work (post-v1)

Per-model option overrides file (TOML: glob → tune options); thermal/idle
gating (pause when the machine is in use); cross-night trend report;
notification hooks; parallel calibration on multi-GPU hosts; registry cache
with mtime invalidation if sessions-dir scans ever get slow.

---
