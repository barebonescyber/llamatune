# llamatune Marathon — Design, Architecture, and Requirements (v1)

Status: v1 specification, approved for implementation.
Audience: users, contributors, and implementers. This document is
authoritative for the Marathon feature;
[DESIGN.md](../DESIGN.md) remains authoritative for everything it already
specifies, and [nightshift-design.md](nightshift-design.md) remains
authoritative for Night Shift. Implementation deviations must be documented
and reviewed. Read `AGENTS.md` before starting; its boundaries and safety
invariants apply unchanged.

## 1. Product definition

`llamatune marathon` is the single-model counterpart of Night Shift: where
Night Shift spreads an unattended window across a directory of models,
Marathon concentrates the entire window on **one** model and spends it on
exhaustive coverage, repetition, and cross-checked statistics. Its promise:
when a marathon completes, the recommended configuration is the best
combination of every tunable performance and context setting the installed
llama.cpp exposes, within measured feasibility, proven against drift-checked
baselines and head-to-head replication — not merely the best point a single
bounded hill-climb happened to visit.

Pipeline of the `marathon` command:

1. **Reconnaissance**: extended bare-defaults baseline (10 independent
   invocations) establishing high-confidence reference means, a
   marathon-grade noise floor, and a session-start drift trend check.
2. **Rounds**: repeated full tuning rounds through the existing
   `run_tuning` pipeline with escalating budgets, each round warm-started
   from the current champion and steered by a **coverage ledger** toward
   regions of the feasible configuration space no earlier round has
   executed or pruned (§6).
3. **Champion challenge**: whenever a round produces a winner that differs
   from the incumbent champion, an interleaved head-to-head A/B block
   (§9) decides succession under identical current conditions — cross-round
   comparisons never rely on comparing numbers measured hours apart.
4. **Context × depth matrix**: the champion is measured across the full
   cross-product of the context ladder and the KV-depth grid, with bounded
   per-cell refinement, producing an operating-point table (§8).
5. **Replicated verification**: the final champion must beat the
   drift-corrected defaults baseline in interleaved A/B blocks (§9), not
   just one confirmation pass.
6. **Report**: `marathon-report.md` + `marathon.json` — champion, coverage
   attained, per-round history, operating-point table, drift record, and
   every deferred or failed item with its reason.

Throughout, **calibration brackets** (§7) re-measure the bare-defaults
reference at phase boundaries so overnight environmental drift (thermal
state, background load, driver hiccups) is detected and quantified instead
of silently contaminating cross-phase comparisons.

Marathon is an orchestrator. Rounds are ordinary tuning sessions created
through `run_tuning`/`resume_tuning` (DESIGN §13.3); brackets reuse the
Night Shift calibration engine (`calibrate.run_calibration`); it invents no
second benchmarking methodology. All measurement semantics (baseline, noise
floor, confirmation, classification, monotone pruning) remain exactly
DESIGN §§6–11.

### 1.1 Relationship to `tune` and `nightshift`

| | `tune` | `nightshift` | `marathon` |
|---|---|---|---|
| scope | one model, one pass | many models, one window | one model, one window |
| search | bounded staged climb | same, per model | staged climb **plus** coverage-driven exhaustive tiers over rounds |
| repetition | 3/5 reps | 5/8 (deep) | 8/12, replicated A/B blocks |
| context/depth | single ladder / profile | pass-through | full ctx × depth matrix with per-cell refinement |
| stopping | budget | deadline | convergence, coverage, or deadline |

### 1.2 Non-goals (v1)

Concurrent benchmarking (strictly serial — one `llama-bench` process at a
time, ever); daemonization or scheduling itself; multiple models per run
(use Night Shift); modifying, moving, or deleting model files or prior
sessions; response-surface modeling or Bayesian optimization (the coverage
tiers are deterministic and auditable; smarter samplers are future work
§17); network access; new statistical dependencies (stdlib only, NFR-4).

## 2. Definitions

- **Champion**: the configuration currently believed best. Initialized to
  the resolved defaults; replaced only by winning a head-to-head A/B
  challenge (§9). `None`-config champion means "defaults".
- **Round**: one full tuning session executed by Marathon with a specific
  budget, warm start, and steering set (§5).
- **Reference**: the reconnaissance bare-defaults means (`pp_ref`,
  `tg_ref`) and marathon noise floor `cv_ref` (§7.1).
- **Bracket**: a 3-run re-measurement of the bare-defaults invocation
  compared against the reference, executed at phase boundaries (§7.2).
  Implemented as a Night Shift verification calibration with a synthesized
  reference record.
- **Coverage ledger**: the derived accounting of the enumerable feasible
  configuration space against every trial id executed, cached, pruned, or
  proven infeasible across all of this model's sessions (§6).
- **Responsive dimension**: a dimension where any single-dimension move
  observed so far changed the score by more than `τ` (§6.3).
- **Operating point**: one (ctx, depth) cell of the matrix with its
  measured pp/tg and per-cell best configuration (§8).
- **τ (decision threshold)**: `max(0.01, 2 × cv_ref)` — used uniformly for
  responsiveness, drift, challenge margins, and convergence.
- **Converged**: `--converge-rounds` consecutive completed rounds produced
  no champion change **and** the coverage ledger reports no remaining
  executable tier-A/B/C candidates (§10).

## 3. CLI specification

One new Typer command in `cli.py` (thin, lazy imports, per DESIGN §3):

```
llamatune marathon MODEL.gguf [options]
```

- `MODEL.gguf` (positional, required) — one readable GGUF file. Sharded
  models: pass the `-00001-of-…` head, as with `tune`.
- `--llama-bin DIR`, `--sessions-dir DIR` (default `./llamatune-sessions`)
  — as in `tune`. Rounds land as ordinary sessions under `--sessions-dir`;
  the marathon run directory lands under `<sessions-dir>/marathon/` (§12).
- `--until HH:MM` / `--max-hours F` — deadline semantics identical to
  Night Shift §3 (past `--until` means tomorrow; earlier of the two wins).
  If **neither** is given, the marathon runs until convergence (§10) or
  `--rounds-max`, whichever first — never unbounded.
- `--rounds-max N` (default `6`, minimum `1`) — hard cap on tuning rounds.
- `--converge-rounds K` (default `2`, minimum `1`) — consecutive
  no-champion-change rounds required for convergence.
- `--ab-blocks N` (default `5`, minimum `3`) — interleaved A/B blocks for
  the final verification; challenges use `max(3, N − 2)` blocks.
- `--depth-grid CSV` (default `0,8192,32768`) — ascending distinct
  non-negative KV depths for the matrix (§8). Requires the `d` capability
  when any value > 0; otherwise exit 2 naming the binary (per C16).
- `--ctx-size CSV` — as in `tune` (required rung + ascending stretch
  rungs); the ladder values are also the matrix context rungs. Default when
  omitted: the matrix uses the single resolved session context.
- `--matrix-refine/--no-matrix-refine` (default on) — per-cell refinement
  candidates (§8.2).
- `--drift-threshold F` (default `0.05`) — bracket drift floor, as Night
  Shift §6.3 (`τ_bracket = max(drift_threshold, 2 × cv_ref)`).
- Pass-through tuning options applied to every round: `--target`,
  `--allow-lossy`, `--vram-reserve-mb`, `--cooldown`, `--full-hash`,
  `--pp`, `--tg`, `--quality-corpus`, `--ot-search`, plus explicit
  overrides of the profile's `--budget-trials`, `--reps-search`,
  `--reps-confirm`, `--baseline-runs` (§11; an explicit `--budget-trials`
  fixes round 1 and still escalates).
- `--dry-run` — print the resolved plan (recon, round-1 budget and
  steering summary, matrix cells, A/B design, estimates) and exit 0
  without executing any benchmark.
- `--json` — machine-readable summary (the `marathon.json` content) on
  stdout instead of human text.

Validation errors (bad `--until`, `--rounds-max < 1`, non-ascending
`--depth-grid`, `--ab-blocks < 3`) exit 2 with a one-line `error:` message.

### 3.1 Exit codes

- `0` — marathon completed (converged, coverage-complete, or deadline)
  and the final champion is a confirmed improvement over defaults.
- `1` — marathon completed but the champion is the defaults configuration
  (no confirmed improvement), or completed with at least one failed item;
  evidence and the report are still written and distinguish the two.
- `2` — usage or configuration error.
- `3` — environment error (`llama-bench` missing/unusable, model
  unreadable, reconnaissance baseline failed both attempts), or the
  3-consecutive-round-failure circuit breaker (§5.4).
- `4` — interrupted by signal with work remaining; re-running the same
  command resumes naturally (§13.4).

## 4. Phase architecture (`marathon.py`)

Executed strictly serially, in order. All scheduling uses an injectable
clock (`now_fn`), as Night Shift §7.1. Every phase transition journals a
`phase` entry and runs a bracket (§7.2) unless one ran within the last
15 minutes.

- **Phase 0 — Resume**: if an incomplete session under `--sessions-dir`
  matches this model's fingerprint **and** is recorded in this marathon
  run's journal as one of its own rounds, resume it via `resume_tuning`
  before anything else. Foreign incomplete sessions (not started by this
  run) are listed as a warning and left alone — Marathon never adopts
  another command's session.
- **Phase 1 — Reconnaissance** (§7.1).
- **Phase 2 — Rounds** (§5): loop until convergence, `--rounds-max`, or
  the deadline no longer fits a round (Night Shift §7.3 fitting rules with
  `MIN_ROUND_MINUTES = 30`, `SHUTDOWN_MARGIN_MIN = 5`).
- **Phase 3 — Matrix** (§8): runs when a champion exists (defaults count)
  and at least `MATRIX_MIN_MINUTES = 15` remain; cells that do not fit are
  journaled `deferred`.
- **Phase 4 — Verification** (§9.2): final interleaved A/B versus
  drift-corrected defaults. Skipped (with a warning) only when the
  champion *is* defaults.
- **Phase 5 — Report**: write `marathon.json`, render
  `marathon-report.md`, append the confirmed final recommendation to
  `<sessions-dir>/registry.jsonl` exactly as a `tune` confirmation does
  (via the round session that produced the champion — Marathon itself
  never writes the registry file).

Deadline pressure degrades gracefully in reverse order of value: deepening
rounds are sacrificed first, then matrix refinement, then matrix breadth
(outer rungs first), and Phase 4 is preserved whenever
`AB_RESERVED_MINUTES = ceil(ab_blocks × 2 × est_invocation_minutes)` can be
protected; the plan and report state what was sacrificed.

## 5. Round engine

### 5.1 Construction

Round `r` (1-based) builds a `TuneOptions` from the escalation table (§11)
plus pass-throughs and calls `Session.create` + `run_tuning` exactly as
`cli.tune` does, with:

- **Warm start**: `initial_gpu_layers` / `initial_cpu_moe` seeded from the
  champion config (when not defaults), so boundary discovery re-verifies
  rather than re-learns placement.
- **Steering**: the round's coverage tier targets (§6.2) are recorded in
  the marathon journal. v1 steering is **budget-shaped**: Marathon cannot
  inject candidates into `search.py` (no engine changes, NFR-6); instead
  it sizes `budget_trials` so the engine's own passes (DESIGN §10 steps
  6 and 9) reach the tier candidates, and the ledger *measures* attained
  coverage after each round. The ledger is the auditor, not the driver;
  candidate injection is future work (§17).
- Round sessions are ordinary sessions; their evidence is the evidence of
  record. Marathon journals `round_start` / `round_end` with the session
  dir, outcome, wall seconds, and the post-round ledger snapshot.

### 5.2 Champion challenge

After a round completes with a confirmed winner whose `trial_id` differs
from the champion's: run a challenge A/B (§9.1) between champion and
contender. The contender succeeds only on an A/B win; ties retain the
incumbent (stability bias is deliberate). `challenge` journal entry records
blocks, per-block means, margin, and verdict. A round confirming the
incumbent's own config, or exiting 1 ("defaults optimal"), is a
no-champion-change round.

### 5.3 Escalation

`budget_trials(r) = min(round(budget_trials(1) × 1.5^(r−1)), 1000)`;
repetitions stay at profile values (§11). Escalation exists to push the
engine's coverage passes deeper each round, not to re-measure the same
points harder.

### 5.4 Failure

A round exiting 3 or 4 is journaled `round_failed`; the loop continues
with the next round. Three consecutive failed rounds abort the marathon
with exit 3 (the environment, not the search, is the likely fault). A
round exiting 1 is a *successful* round (evidence gained, defaults held).

## 6. Coverage model (`coverage.py`)

Pure functions; filesystem reads only through passed-in data. The ledger is
**derived, never stored** — recomputed from session evidence, mirroring the
Night Shift registry philosophy.

### 6.1 Enumerable space

`enumerate_space(model, llama, hardware, options) -> tuple[TrialConfig, ...]`
builds the bounded canonical candidate universe from DESIGN §9.1 dimension
values, capability-gated and constraint-filtered by the existing pure
helpers in `config.py` (which may be extended additively but not altered).
The universe is the union of the tiers below — never the full cross-product
of every dimension (which is combinatorially useless):

- **Tier A — placement × throughput core**: {top placement rungs from the
  §9.1 gpu_layers × moe_cpu_layers grids} × full `ubatch × batch`
  factorial × `flash_attn ∈ {0,1}`, constraints applied. "Top placement
  rungs" = the 2 highest-scoring feasible `(ngl, ncmoe)` pairs known so
  far (from the ledger; before any round: the §9.1 grid's maximal feasible
  estimate and its nearest lower rung).
- **Tier B — CPU interaction**: full `threads × moe_cpu_layers` factorial
  at the champion placement's gpu_layers.
- **Tier C — remaining singles**: every §9.1 dimension value not covered
  by A/B, swept one-at-a-time against the champion (mmap, kv_offload,
  lossy cache types when `--allow-lossy`, thread variants).
- **Tier D — responsive factorial**: full factorial over all dimensions
  currently classified responsive (§6.3), other dimensions fixed at
  champion values, capped at `TIER_D_CAP = 256` enumerated configs
  (deterministic truncation order: distance from champion ascending; the
  truncation is journaled — no silent caps).

### 6.2 Ledger

`build_ledger(space, sessions) -> CoverageLedger` classifies every
enumerated config by `trial_id` against all completed and incomplete
sessions for this fingerprint: `executed` (a journaled trial, any status),
`pruned` (covered by a journaled monotone prune), `infeasible`
(constraint-invalid at enumeration — excluded from the denominator),
`remaining`. Coverage % = (executed + pruned) / (executed + pruned +
remaining). The report presents per-tier coverage; FR-6 requires the
number to be honest, not 100%-by-construction.

### 6.3 Responsiveness

A dimension is responsive when any pair of executed `ok` trials differing
in that dimension alone has a score ratio beyond `τ`. Computed from the
ledger per round; drives Tier D membership. Deterministic: ties and
missing data classify as unresponsive (conservative — keeps Tier D small).

## 7. Noise and drift control

### 7.1 Reconnaissance

10 bare-defaults invocations (`-r reps_confirm`, baseline fixed 1800 s
timeout, cooldown honored), executed through the existing `executor` and
recorded under `recon/run-<n>/` in the run directory. Produces the
reference means, `cv_ref` (same formula as DESIGN §6 noise floor, over 10
runs, floored at 0.01), and a **trend check**: if the mean of runs 6–10
differs from runs 1–5 by more than `τ_bracket` on either metric, journal
and report a `warmup_drift` warning (the machine was not thermally settled;
results remain valid but the report says so). Reconnaissance failure
follows the DESIGN §6 safe-baseline fallback; both attempts failing is
exit 3.

### 7.2 Brackets

A bracket is `calibrate.run_calibration` invoked with a **synthesized**
`RegistryRecord`: fingerprint = the model's, `reference_config = None`,
reference means/noise floor from reconnaissance, workload/reps from the
marathon options. `run_calibration`'s `run` parameter is already a
structural `Protocol` (`dir`, `calibration_dir`, `write_json`), which
`MarathonRun` satisfies (§13.2); its `options` parameter is nominally
`NightshiftOptions` and reads exactly `drift_threshold`,
`calibration_runs`, `cooldown_s`, and `profile` — Marathon constructs a
minimal `NightshiftOptions` for bracket calls (`drift_threshold` and
`cooldown_s` from the marathon options, `calibration_runs = 3`,
`profile = "deep"`, remaining fields inert defaults). `calibrate.py` is
thus reused unmodified. Verdicts:

- `consistent` — proceed.
- `drift` — journal `environment_drift`; **re-baseline**: the bracket
  means become the new reference for subsequent comparisons (the old
  reference is retained in evidence), and any A/B phase uses the current
  reference. One re-baseline per phase boundary at most; a second
  consecutive drifting bracket adds a prominent report warning that the
  environment is unstable.
- `error` — treated as a failed item (counts toward nothing fatal by
  itself; three consecutive bracket errors abort with exit 3).

## 8. Context × depth matrix (`matrix.py`)

### 8.1 Cells

Cells = `ctx_rungs × depth_grid`, where `ctx_rungs` are the resolved
`--ctx-size` ladder values (single value → one rung). For each cell,
ascending ctx then depth:

1. Feasibility: the champion config at rung ctx via the existing context
   probe shape (`-p ctx -n 16 -r 1`, no `-d`, per C16). Monotone pruning
   across rungs applies (a failed rung prunes larger rungs for that
   config); the envelope fallback-placement walk (C17, the shared retreat
   helper) finds the cell's config when the champion itself fails.
2. Measurement: one invocation of the cell config with `-d depth`,
   `-r reps_search`, recorded under `matrix/ctx-<c>-d-<d>/`.
3. Depth 0 cells at the required rung reuse existing champion evidence
   when available (no re-execution; the cell records the source).

### 8.2 Per-cell refinement (`--matrix-refine`)

For each measured cell, up to 2 refinement candidates: champion `ubatch`
halved and doubled (constraint-clamped, deduplicated, skipped if equal to
champion). Best of the ≤ 3 measurements becomes the cell's operating
point; ties keep the champion config. Refinement never changes the global
champion — it produces the per-operating-point advice table only.

### 8.3 Output

`matrix` block in `marathon.json`: rows
`{ctx, depth, status: ok|failed|pruned|deferred, config, pp, tg,
refined: bool, evidence}`. The report renders the operating-point table
and, when any cell's best config differs from the champion, a
"per-context/depth overrides" section with commented `llama-server`
alternates (reusing the C17 `recommended.sh` alternate mechanism's flag
mapping).

## 9. Interleaved A/B verification (`abtest.py`)

### 9.1 Method

One **block** = 4 invocations in ABBA order (A, B, B, A), each
`-r reps_confirm`, cooldown between all four, same workload; block
score for each side = mean of its two invocations' `avg_ts` means per
metric. A side **wins a block** when its target-weighted score (DESIGN
§11 weights) exceeds the other's by more than `τ`; otherwise the block is
a tie. ABBA ordering cancels first-order thermal/cache trends within the
block; blocks are separated by cooldown so between-block variance is
represented.

`run_ab(run, a_config, b_config, *, blocks, model, llama, options,
label) -> ABResult` — configs may be `None` (bare defaults). Executes via
`executor`, writes under `run.ab_dir(label, block, slot)`, journals one
`ab_block` entry per block.

Verdict: side X **wins the A/B** when X wins a strict majority of blocks
and the pooled per-metric means do not show the *other* side ahead beyond
`τ` on the target's dominant metric. Anything else is a tie.

### 9.2 Final verification

`--ab-blocks` blocks of champion vs bare defaults. Champion wins → exit 0
path; tie or loss → the report states that the round-confirmed improvement
did not replicate under interleaved measurement, the recommendation is
still emitted with a **prominent `not replicated` label**, and the exit
code is 1. This is the honesty backstop that distinguishes Marathon from
running `tune` with big budgets.

## 10. Convergence and stopping

Stop Phase 2 at the first of:

1. **Converged**: `--converge-rounds` consecutive no-champion-change
   rounds **and** ledger `remaining` = 0 for tiers A–C (Tier D may remain
   when unresponsive dims never triggered it).
2. `--rounds-max` rounds completed.
3. Deadline: the next round no longer fits (§4 fitting).
4. Circuit breaker (§5.4).

The report states which criterion fired. Convergence with coverage
remaining is impossible by construction (criterion 1 requires both);
deadline stops record remaining coverage explicitly.

## 11. Budgets and escalation

Base profile (per-round; overridable individually via pass-throughs):

| option | round 1 | escalation |
|---|---|---|
| `budget_trials` | 240 | × 1.5 per round, cap 1000 |
| `reps_search` | 8 | fixed |
| `reps_confirm` | 12 | fixed |
| `baseline_runs` | 5 (per-session) | fixed |
| `cooldown_s` | 10.0 | fixed |

Marathon-level repetition (not per-session): reconnaissance 10 runs;
brackets 3 runs; A/B `4 × blocks` invocations. `budget_minutes` is always
scheduler-supplied (§4) and never part of the profile. For comparison:
`tune` defaults are 60/3/5/3/0 and Night Shift deep is 120/5/8/5/5 —
round 1 alone doubles Night Shift's deepening budget and adds the
marathon-level layers on top.

## 12. Evidence: run directory, journal, report

```
<sessions-dir>/marathon/<model-stem>-<UTCyyyymmdd-hhmmss>-<6hex>/
  run.json            # schema_version 1, tool version, argv, options,
                      # resolved deadline, created (UTC)
  hardware.json       # start-of-run assessment (re-assessed before each round)
  llamacpp.json       # start-of-run probe
  model.json          # inspection + fingerprint
  plan.json           # resolved phases, round-1 budget, tiers, matrix cells
  journal.jsonl       # append-only, one JSON object per line, fsync per line
  recon/run-<n>/      # command.json, stdout.json, stderr.log
  brackets/<n>/run-<k>/
  ab/<label>/block-<n>/<slot>/     # slot ∈ a1,b1,b2,a2
  matrix/ctx-<c>-d-<d>/[refine-<k>/]
  marathon.json       # final machine-readable summary (== --json stdout)
  marathon-report.md
```

Journal entry types (every entry carries a UTC timestamp):
`marathon_start`, `plan`, `phase`, `recon_run`, `bracket`,
`environment_drift`, `round_start`, `round_end`, `round_failed`,
`challenge`, `coverage_snapshot`, `matrix_cell`, `ab_block`, `deferred`,
`interrupted`, `marathon_end`.

Write confinement: all writes inside the run directory go through the
`MarathonRun` writer with the same resolve-and-verify confinement as
`Session` (DESIGN §12). The layering sentence extends to: only
`marathon`'s `MarathonRun` writes inside a marathon run directory. Round
sessions remain exclusively `Session`-written.

`marathon-report.md` sections: **Summary** (champion vs defaults with
replication verdict, stop reason, rounds, wall time, machine + build
identity); **Champion** (config, confirmed means, reproduce command);
**Round history** (table: round, budget, outcome, winner, challenge
verdict, wall); **Coverage** (per-tier enumerated/executed/pruned/
remaining, responsive dimensions); **Operating points** (ctx × depth
table + overrides); **Environment** (recon trend, bracket timeline,
re-baselines); **A/B verification** (per-block table, verdict);
**Deferred and failed**; **Warnings**. Rendering consumes only
`marathon.json` content (mirroring `report.py`'s analysis-only rule) so
the report is regenerable.

`marathon.json` (schema_version 1): options, window, reference (original
and any re-baselines), rounds, challenges, ledger summary, matrix rows,
ab results, aggregate counts, warnings. `MarathonOutcome(run_dir,
summary: dict, exit_code: int)` is the entry-point return type.

## 13. Architecture

New modules and touched files:

```
src/llamatune/
  coverage.py       # §6  — space enumeration + ledger (pure)
  matrix.py         # §8  — ctx × depth execution via executor
  abtest.py         # §9  — interleaved blocks + verdicts via executor
  marathonreport.py # §12 — marathon-report.md rendering from marathon.json
  marathon.py       # §§4–5,7,10 — MarathonRun writer, phases, rounds,
                    #   brackets, signals; run_marathon(options, *,
                    #   now_fn=None) -> MarathonOutcome
  types.py          # additions only (§13.1)
  cli.py            # new `marathon` command only
tests/
  fixtures/fake_llama_bench.py   # two new knobs (§15.1)
  unit/test_coverage.py  test_matrix.py  test_abtest.py
  unit/test_marathonreport.py  test_marathon_plan.py  test_cli_marathon.py
  integration/test_marathon.py
```

Layering (extends DESIGN §13): `coverage` imports nothing above stdlib +
`types`/`config` and never touches the filesystem or processes (session
evidence is passed in as parsed data); `matrix` and `abtest` execute only
via `executor`, parse only via `bench`, and write only via `MarathonRun`;
`marathonreport` renders strings from dicts, no I/O; `marathon`
orchestrates and owns the only writer for its run directory; `cli` stays
thin with lazy imports. `calibrate.run_calibration` is consumed as-is.
No new runtime dependencies.

### 13.1 `types.py` additions (frozen, slots, kw_only — exact fields)

```python
MarathonOptions(
    model_path: Path, llama_bin: Path | None, sessions_dir: Path,
    until: str | None, max_hours: float | None,
    rounds_max: int, converge_rounds: int, ab_blocks: int,
    depth_grid: tuple[int, ...], ctx_size: int | None,
    ctx_ladder: tuple[int, ...], matrix_refine: bool,
    drift_threshold: float, dry_run: bool,
    target: str, allow_lossy: bool, vram_reserve_mb: int | None,
    cooldown_s: float | None, full_hash: bool, pp: int, tg: int,
    quality_corpus: Path | None, ot_search: bool,
    budget_trials: int | None, reps_search: int | None,
    reps_confirm: int | None, baseline_runs: int | None,  # None => profile
)

CoverageLedger(tiers: dict[str, TierCoverage],
               responsive: tuple[str, ...])
TierCoverage(enumerated: int, executed: int, pruned: int, remaining: int,
             remaining_ids: tuple[str, ...])

RoundRecord(index: int, session_dir: Path, exit_code: int,
            winner_config: TrialConfig | None, champion_changed: bool,
            wall_s: float, coverage_pct: float)

MatrixCell(ctx: int, depth: int, status: str,   # ok|failed|pruned|deferred
           config: TrialConfig | None, pp: float | None, tg: float | None,
           refined: bool, evidence: Path | None)

ABResult(label: str, blocks: int, a_wins: int, b_wins: int, ties: int,
         a_pp: float, a_tg: float, b_pp: float, b_tg: float,
         margin: float, verdict: str)            # "a"|"b"|"tie"

MarathonOutcome(run_dir: Path, summary: dict[str, Any], exit_code: int)
```

Existing dataclasses are **not** modified.

### 13.2 Frozen cross-module interfaces

```python
coverage.enumerate_space(model, llama, hardware, options,
                         *, champion, known_placements)
    -> dict[str, tuple[TrialConfig, ...]]          # tier -> configs
coverage.build_ledger(space, executed_ids, pruned_ids, *, trials)
    -> CoverageLedger
matrix.run_matrix(run, champion, model, llama, options,
                  *, remaining_minutes_fn) -> tuple[MatrixCell, ...]
abtest.run_ab(run, a_config, b_config, *, blocks, model, llama,
              options, label) -> ABResult
marathonreport.render(summary: dict) -> str
marathon.run_marathon(options: MarathonOptions, *, now_fn=None)
    -> MarathonOutcome
MarathonRun.create(sessions_dir, *, options, hardware, llama, model, argv)
MarathonRun.load(run_dir); .dir; .append(entry: dict)
MarathonRun.recon_dir(n); .bracket_dir(n, k)
MarathonRun.ab_dir(label, block, slot); .matrix_dir(ctx, depth, refine=None)
MarathonRun.calibration_dir(fingerprint16, n)   # NightshiftRun-compatible,
                                                # maps into brackets/
MarathonRun.write_json(name, payload); .write_text(name, text)
```

`MarathonRun` must satisfy the writer surface `calibrate.run_calibration`
consumes. Structural compatibility with `NightshiftRun` is verified by tests;
any gap is fixed at the adapter layer rather than by changing `calibrate.py`.

### 13.3 Signals

First `SIGINT`/`SIGTERM`: stop flag; the in-flight invocation finishes,
the current item journals its state, `marathon.json` and the report are
written from work completed so far, exit 4. Second signal: executor
process-group termination; `interrupted` journal entry; round sessions
remain resumable.

### 13.4 Re-entrancy

Marathon detects its own prior interrupted run: if
`<sessions-dir>/marathon/` contains a run whose `run.json` options match
(model fingerprint, target, allow_lossy, workload, depth grid, ctx ladder)
and whose journal lacks `marathon_end`, the command **continues that run**
— completed rounds are found via the ledger, the interrupted round resumes
in Phase 0, completed matrix cells and A/B blocks are skipped via their
journal entries. A mismatch in any identity-bearing option starts a fresh
run and leaves the old one untouched, with a warning naming it.

## 14. Safety invariants

All of DESIGN §14 and Night Shift §11 unchanged: strictly additive on
disk; never deletes or rewrites sessions, model files, or prior runs; no
prompts or stdin; exactly one child benchmark process at any moment;
signal handling never orphans a child; no network.

## 15. Testing strategy

### 15.1 Fixture extension (normative)

`fake_llama_bench.py` gains two knobs, both layered after the existing
performance model and `LLAMATUNE_FAKE_SPEED_SCALE`:

- `LLAMATUNE_FAKE_SPEED_RAMP` (float, default `0.0`) — every emitted
  `avg_ts` is additionally multiplied by `(1 + ramp)^k`, where `k` is a
  per-process-tree invocation counter persisted in
  `LLAMATUNE_FAKE_COUNTER_FILE` (path env var; counter absent → `k = 0`).
  Enables drift-over-time tests (brackets, ABBA cancellation).
- `LLAMATUNE_FAKE_DEPTH_PENALTY` (float, default `0.0`) — when `-d N` is
  present, tg `avg_ts` is multiplied by `1 / (1 + penalty × N / 8192)`.
  Enables matrix-shape assertions.

### 15.2 Unit

- Coverage: tier enumeration is constraint-valid, capability-gated, and
  deterministic; Tier D appears only with responsive dims and truncates at
  the cap with the truncation recorded; ledger classification against
  synthetic executed/pruned id sets; coverage % excludes infeasible;
  responsiveness threshold boundary (exactly τ → unresponsive).
- Matrix: monotone rung pruning; fallback placement on rung failure;
  refinement clamp/dedup/skip; deferral on exhausted time (fake clock);
  depth-0 required-rung reuse.
- A/B: ABBA slot ordering in emitted commands; block win/tie math at the
  τ boundary; majority + pooled-mean verdict table (win/loss/tie cases);
  `None` config produces bare-defaults argv.
- Plan/orchestration (pure functions, fake clock): escalation series and
  cap; convergence requires both conditions; deadline fitting and the
  Phase 4 reservation; challenge-only-on-different-winner; circuit
  breaker; re-entrancy matching on identity-bearing options.
- Report: structural assertions on every §12 section from a canned
  `marathon.json`; absence-tolerance for skipped phases.
- CLI: validation exits, option assembly, `--dry-run` plan content.

### 15.3 Integration (fake bench + tiny GGUFs, no GPU, no network)

1. Fresh model, no deadline → recon, rounds until convergence, matrix,
   A/B; exit 0; champion beats defaults; report and `marathon.json`
   complete; coverage tiers A–C report 0 remaining.
2. `SPEED_RAMP` producing mid-run drift → bracket verdict `drift`,
   re-baseline journaled, final A/B still yields a verdict, report shows
   the environment timeline.
3. Fixture optimum equal to defaults → rounds converge with defaults as
   champion, Phase 4 skipped with warning, exit 1.
4. Deadline nearly exhausted (fake clock) → round deferred, matrix
   truncated outer-rung-first, Phase 4 preserved, exit 0, report lists
   sacrifices.
5. SIGTERM mid-round → exit 4; identical rerun continues the same run,
   resumes the round, skips completed matrix cells, completes.
6. `LLAMATUNE_FAKE_GENUINE_FAIL` on three consecutive rounds → exit 3.
7. `DEPTH_PENALTY` set, `--ctx-size 8192,32768 --depth-grid 0,8192` →
   4-cell matrix with decaying tg, one rung failure exercising fallback
   placement, refinement improving one cell.
8. Final A/B forced to tie (`SPEED_SCALE` flip between phases) → exit 1
   with the `not replicated` label present in report and summary.

## 16. Documentation

README gains a "Marathon" section (synopsis, one worked example, stopping
semantics, relationship to Night Shift, where the report lands). DESIGN.md
and nightshift-design.md are not modified; this document is the Marathon
spec.

## 17. Future work (post-v1)

Candidate injection into `search.py` so the ledger drives rounds directly
instead of budget-shaping; adaptive samplers (Bayesian/response-surface)
over the responsive subspace; per-cell full re-tunes in the matrix;
paired-difference statistics beyond mean-vs-τ; cross-marathon trend
reports; Night Shift integration (`nightshift --marathon-one MODEL` spare
time delegation); speculative-decoding dimensions when DESIGN adopts them.

---


## Appendix: Requirements checklist

Functional:

- FR-1 `llamatune marathon MODEL.gguf` exists with the §3 options; `--json`
  and human output; validation errors exit 2; depth grid without the `d`
  capability exits 2 before any run directory is created.
- FR-2 Reconnaissance produces reference means, `cv_ref`, and the trend
  check; double failure exits 3.
- FR-3 Rounds execute as ordinary sessions with §11 budgets, escalation,
  and champion warm start.
- FR-4 A differing round winner triggers a champion challenge; succession
  only on an A/B win; ties retain the incumbent.
- FR-5 Convergence requires both no-champion-change rounds and empty
  tier A–C remainder; the stop reason is reported.
- FR-6 The coverage ledger is derived from session evidence, honest
  (infeasible excluded, truncation journaled), and reported per tier.
- FR-7 Brackets run at phase boundaries; `drift` re-baselines with the old
  reference retained; consecutive instability is prominently warned.
- FR-8 The matrix covers ctx ladder × depth grid with monotone pruning,
  fallback placements, and evidence per cell.
- FR-9 Refinement is bounded to 2 candidates per cell and never changes
  the global champion.
- FR-10 A/B blocks are ABBA-ordered with per-block and pooled verdicts per
  §9.1.
- FR-11 Final verification runs champion vs defaults; non-replication
  yields exit 1 with the `not replicated` label.
- FR-12 Deadline fitting follows §4 with the Phase 4 reservation and
  reverse-value degradation, all sacrifices reported.
- FR-13 One failed round never aborts the marathon (3-strike breaker,
  exit 3); exit codes follow §3.1.
- FR-14 First signal → graceful conclusion, report written, exit 4;
  an identical rerun continues the same run per §13.4.
- FR-15 Run directory, journal, `marathon.json`, `marathon-report.md` per
  §12; report regenerable from `marathon.json` alone.
- FR-16 `--dry-run` prints the full resolved plan and executes nothing.

Non-functional:

- NFR-1 Strictly serial execution; prefer signal handlers + flags over
  threads.
- NFR-2 Injectable clock for every scheduling decision; the test suite
  never sleeps for scheduling and never runs a real model or GPU.
- NFR-3 All existing quality gates pass repo-wide: `ruff format --check`,
  `ruff check`, `mypy` (strict), `pytest` with branch coverage ≥ 85%.
- NFR-4 No new runtime dependencies; stdlib statistics only; Python ≥ 3.11.
- NFR-5 Existing commands' behavior and outputs are byte-for-byte
  unchanged except for the added CLI command.
- NFR-6 No modifications to `search.py`, `calibrate.py`, or any existing
  engine module — Marathon composes; it does not fork the methodology.
