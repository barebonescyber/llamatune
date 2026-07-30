# llamatune Results Matrix — Design, Architecture, and Requirements (v1)

Status: v1 specification, approved for implementation.
Audience: users, contributors, and implementers. This document is
authoritative for the Results Matrix feature;
[DESIGN.md](../DESIGN.md) remains authoritative for everything it already
specifies, [nightshift-design.md](nightshift-design.md) for Night Shift, and
[marathon-design.md](marathon-design.md) for Marathon. Implementation
deviations must be documented and reviewed. Read `AGENTS.md` before starting;
its boundaries and safety invariants apply unchanged.

The companion specification [quality-eval-design.md](quality-eval-design.md)
defines the quality-evaluation feature whose results this matrix ingests.
The Results Matrix lands first; its schema (§4) is deliberately generic so
quality rows require no schema change.

## 1. Product definition

Every llamatune mode produces evidence in its own place: `tune` sessions
under `--sessions-dir`, Night Shift runs under `<sessions-dir>/nightshift/`,
Marathon runs under `<sessions-dir>/marathon/`, confirmed recommendations in
`registry.jsonl`, and (once quality-eval lands) quality runs under
`<sessions-dir>/quality/`. Today a consumer who asks "which model, at which
configuration, is the best fit for my use case?" must read all of them.

The **Results Matrix** is the single consumer-facing aggregation of every
measured result across all modes. It answers, in one queryable place:

- which model and configuration delivers **maximum prompt-processing
  throughput** (pp t/s);
- which delivers **maximum token-generation throughput** (tg t/s);
- which delivers the best **balanced** throughput;
- which supports the **largest validated context**, and at what speed;
- how throughput degrades across **context × KV-depth operating points**;
- (with quality-eval) which model/configuration scores best on **coding
  correctness, tool-use, agentic, and instruction-following** suites;
- and, for each answer: on which hardware, against which llama.cpp build,
  how confident the number is, and where the raw evidence lives.

Precise meaning of "single source of truth": session and run directories
remain the **evidence of record** (they are never edited or replaced); the
matrix is the **sole authoritative aggregation** — the only artifact an
agent, agentic harness, or user needs to consult to select a model and
configuration. It is derived deterministically from the evidence, is
rebuildable at any time, and every row carries provenance pointing back to
the evidence that produced it. Nothing in the matrix exists that the
evidence does not support.

The feature consists of:

1. A **row model** (§4): one normalized `ResultRow` per measured result,
   keyed by model × hardware × build × configuration × operating point ×
   measurement kind, with metrics in an extensible namespace.
2. **Harvesters** (§5): tolerant, read-only extractors for each evidence
   kind (tuning sessions, Night Shift runs, Marathon runs, quality runs).
3. **Build semantics** (§6): deterministic materialization to
   `<root>/matrix/results-matrix.json` + `results-matrix.md`, multi-root
   merge, and an opportunistic post-command refresh.
4. **Query semantics** (§7): named use cases, filters, compatibility
   annotation against the current machine/build, and honest ranking rules.
5. A **CLI** (§3): `llamatune matrix build|query|show|export`.

### 1.1 Relationship to existing artifacts

- `registry.jsonl` (DESIGN C10) is unchanged and remains what `best`
  consults. The matrix is a superset view: the registry records only
  confirmed recommendations; the matrix additionally records baselines,
  lossless alternates, context envelopes, depth profiles, Marathon
  operating points and A/B verdicts, calibration history, and quality
  results. The matrix harvests sessions directly rather than reading
  `registry.jsonl`, so both are independently derived from the same
  ground truth and cannot disagree with the evidence.
- `report.md`/`marathon-report.md`/`nightshift-report.md` remain per-run
  narratives. `results-matrix.md` is the cross-run table.
- `llamatune best` is unchanged. `matrix query` is its many-model,
  many-objective generalization; `best` may adopt the matrix internally
  post-v1 (§14).

### 1.2 Non-goals (v1)

Executing benchmarks (the matrix only reads evidence; it never launches
`llama-bench`); modifying, migrating, or annotating any session, run, or
registry file; a daemon or file-watcher (refresh is command-driven);
cross-machine merging of matrices from different hosts (rows carry hardware
identity, so a later merge tool is possible — §14); SQLite or any query
language beyond the §7 filters; statistical re-analysis (the matrix reports
recorded statistics; it computes no new confidence intervals); network
access; new runtime dependencies.

## 2. Definitions

- **Root**: one sessions directory as passed via `--sessions-dir`
  (repeatable). A root contains ordinary sessions plus optional
  `nightshift/`, `marathon/`, and `quality/` subtrees and a
  `registry.jsonl`. This repository currently has three roots
  (`llamatune-sessions`, `llamatune-sessions-nightshift-12h`,
  `llamatune-sessions-context-ladder`) — multi-root build exists because
  real usage already produced multiple roots.
- **Row**: one `ResultRow` (§4) — a single measured result normalized out
  of the evidence.
- **Kind**: the measurement class of a row (§4.2). Kinds differ in what
  the number means and how much confidence it carries.
- **Metric namespace**: dotted metric names (`perf.pp`, `quality.coding.
  pass_rate`) in a flat `dict[str, float]`. Readers ignore namespaces they
  do not know; producers may add namespaces without a schema change.
- **Identity key / `row_id`**: sha256 (first 16 hex) of the canonical JSON
  of `(model_fingerprint, hardware_hash, build_discriminator,
  config_trial_id or "defaults", ctx, depth, pp_workload, tg_workload,
  kind, suite_id)`. Two measurements of the same thing at different times
  share a `row_id`.
- **Current row**: the newest row (by evidence timestamp, tie-broken by
  source directory name) among rows sharing a `row_id`. Older rows are
  retained with `current: false`.
- **Hardware hash**: sha256 (first 16 hex) of the canonical JSON of the
  DESIGN hardware signature (`config.hardware_signature` output). The full
  signature is also stored on the row.
- **Build discriminator**: `bench_sha256` when recorded, else
  `help_sha256` — the same precedence the registry lookup uses.
- **Compatibility**: a query-time annotation comparing a row's hardware
  hash and build discriminator against the current machine and the
  `--llama-bin` build: `current`, `build_changed`, `hardware_changed`, or
  `both` (§7.3).
- **Use case**: a named filter + ranking objective (§7.1), e.g. `max-tg`.

## 3. CLI specification

One new Typer sub-application in `cli.py` (thin, lazy imports, per
DESIGN §3), mounted as `llamatune matrix`:

```
llamatune matrix build  [--sessions-dir DIR]... [--output DIR] [--json]
llamatune matrix query  [--sessions-dir DIR]... [query options] [--json]
llamatune matrix show   [--sessions-dir DIR]... [--json]
llamatune matrix export [--sessions-dir DIR]... --format json|csv|md
                        [--output PATH]
```

Shared options:

- `--sessions-dir DIR` (repeatable; default: `./llamatune-sessions`) —
  roots to harvest. Every subcommand harvests fresh from the evidence on
  each invocation (§6.1); the materialized artifact is an export for
  direct file readers, never a cache consulted by the CLI.

`matrix build`:

- Harvests all roots and writes `results-matrix.json` and
  `results-matrix.md` under `--output DIR` (default:
  `<first-root>/matrix/`). Prints a one-line summary (rows, models,
  roots, warnings count); `--json` prints the §6.3 build summary instead.

`matrix query`:

- `--use-case NAME` — one of the §7.1 named use cases. Mutually exclusive
  with `--sort`.
- `--sort METRIC` — rank by any metric name, descending;
  `--ascending` flips.
- Filters (§7.2): `--model TEXT` (substring of name, path stem, or
  fingerprint prefix), `--quant TEXT`, `--kind NAME` (repeatable),
  `--suite NAME`, `--ctx N` (minimum), `--depth N` (exact),
  `--min-pp F`, `--min-tg F`, `--confirmed-only/--include-unconfirmed`
  (default `--confirmed-only`), `--current-only/--include-superseded`
  (default `--current-only`), `--compat any|current` (default `any`).
- `--llama-bin DIR` — probe the current build and scan hardware to
  compute compatibility annotations (§7.3). Without it, compatibility is
  `unknown` and `--compat current` is a usage error (exit 2).
- `--limit N` (default 10, `0` = unlimited).
- `--json` — machine-readable result rows (§7.4); otherwise a human
  table.

`matrix show`:

- Coverage summary: per model — kinds present, best pp/tg/balanced,
  max validated ctx, quality suites present, newest evidence timestamp,
  stale-identity flags. Human table or `--json`.

`matrix export`:

- `--format json|csv|md`; `--output PATH` (default stdout). `json` is the
  full §6.2 document; `csv` and `md` are the flattened row table (§8).

Validation errors (unknown use case, unknown kind, `--compat current`
without `--llama-bin`, no readable root) exit 2 with a one-line `error:`
message, matching existing CLI conventions.

### 3.1 Exit codes

- `0` — command completed. An empty matrix (no completed evidence found)
  is still exit 0 with an explicit "0 rows" message: absence of results is
  a valid answer.
- `2` — usage or configuration error.
- `3` — environment error: no `--sessions-dir` root exists/is readable, or
  `--output` is not writable, or `--llama-bin` was given but no usable
  `llama-bench` was found there.

Corrupt or unreadable individual sessions/runs never change the exit code;
they are skipped and surfaced in `warnings` (§5.4).

## 4. Row model and schema

### 4.1 Axes

Every row records all of:

| axis | fields | notes |
|---|---|---|
| model | `model_fingerprint`, `model_name`, `model_path`, `quant` | fingerprint per DESIGN §5; `quant` parsed best-effort from GGUF name/filename (e.g. `Q4_K_M`, `UD-Q5_K_XL`), else `null` |
| environment | `hardware_hash`, `hardware_signature`, `build_discriminator`, `build_commit` | §2 definitions |
| configuration | `config` (TrialConfig dict or `null` = defaults), derived `config_trial_id` | lossy configs are identifiable from `cache_type_*` |
| operating point | `ctx`, `depth`, `pp_workload`, `tg_workload` | any may be `null` when the source did not fix it |
| measurement | `kind`, `suite_id` (quality rows only, else `null`) | §4.2 |
| result | `metrics`, `status` | §4.3 |
| confidence | `confirmed`, `replicated`, `reps`, `noise_floor_cv` | §4.4 |
| provenance | `source_root`, `evidence_dir`, `ts` | `evidence_dir` is the session/run directory; `ts` is the evidence's own UTC timestamp |
| lifecycle | `row_id`, `current` | §2 |

### 4.2 Kinds

| kind | source | meaning |
|---|---|---|
| `baseline` | session `analysis.json` baseline | measured defaults (or safe fallback; `status` records which via the baseline `kind`) |
| `recommendation` | session winner | confirmed tuned configuration |
| `lossless_recommendation` | session `lossless_winner` | best all-lossless when the winner is lossy |
| `context_envelope` | session `context_envelope` rows | per-ctx-rung validation with fallback config |
| `depth_profile` | session depth-profile evidence | winner measured at profiled KV depths |
| `operating_point` | Marathon `marathon.json` matrix rows | per-(ctx, depth) best config and throughput |
| `ab_verification` | Marathon final A/B | champion-vs-defaults replication verdict |
| `calibration` | Night Shift `nightshift.json` calibrations | re-measurement of a reference config with drift verdict |
| `quality_suite` | quality run `quality.json` suites | one row per (config, suite) with the suite's metric set |

The kind list is closed for v1; harvesters must not invent kinds. Unknown
kinds encountered in a *newer-schema* `results-matrix.json` being read by
`export` are preserved untouched (forward tolerance).

### 4.3 Metrics

Flat `dict[str, float]`; only the names below are produced in v1. All
throughputs are t/s means as recorded in evidence — never re-derived.

- `perf.pp`, `perf.tg` — mean throughputs of the row's measurement.
- `perf.improvement_pp_pct`, `perf.improvement_tg_pct`,
  `perf.improvement_score_pct` — present on `recommendation`,
  `lossless_recommendation` (relative to that session's baseline).
- `perf.drift_pp_pct`, `perf.drift_tg_pct` — present on `calibration`.
- `ctx.validated` — present when the row's ctx passed validation (value =
  ctx, as float, for uniform sorting).
- `quality.<suite>.<name>` — copied verbatim from `quality.json` suite
  metrics (quality-eval §10 defines the names, e.g.
  `quality.coding.pass_rate`, `quality.tooluse.score`,
  `quality.agentic.completion_rate`, `quality.ifollow.score`,
  `quality.perplexity.ppl`).
- `quality.overall` — copied from `quality.json` when present.

Non-numeric facts (verdicts, statuses, flags) are row fields, never fake
numeric metrics. `status` values are the source's own vocabulary
(`ok|failed|pruned|deferred|consistent|drift|…`), passed through.

### 4.4 Confidence

- `confirmed` — true when the number comes from a DESIGN §6 confirmation
  (session winners) or better; false for search samples, envelope probes,
  and matrix cells measured at `reps_search`.
- `replicated` — true only when a Marathon final A/B verified the config
  (`ab_verification` verdict "a"-side win for the champion); `null` when
  no A/B applies.
- `reps` — repetitions behind the mean, when recorded; else `null`.
- `noise_floor_cv` — the producing session's noise floor, when available.

Ranking honesty (§7) is built on these fields; harvesters must populate
them conservatively (unknown ⇒ the weaker value).

## 5. Harvesters (`resultsmatrix.py`)

Pure functions over parsed JSON plus bounded filesystem reads (read-only,
never write, never execute). One harvester per evidence kind; each yields
zero or more rows.

### 5.1 Tuning sessions (includes Night Shift and Marathon round sessions)

For each completed session directory under a root (same completion
predicate as `registry.build_registry` — `analysis.json` present and
journal contains `analysis_written` or `session_end`):

- `baseline` row from `analysis.baseline` (+ resolved defaults config =
  `null` config; when `baseline_kind` is a safe fallback, `status`
  = `safe_fallback`, config = the `-ngl 0` fallback the evidence records).
- `recommendation` row from `analysis.winner` (skip when `null`):
  confirmation means, improvement percentages, `confirmed: true`,
  `ctx` = the session's validated ctx, workload from options.
- `lossless_recommendation` row from `analysis.lossless_winner` when
  present and distinct from the winner.
- `context_envelope` rows: one per envelope entry, `ctx` = rung,
  `status` from the entry, config = the rung's validated (possibly
  fallback) config, `confirmed: false`.
- `depth_profile` rows: one per profiled depth, `depth` set,
  `confirmed: false`.
- `quality_gate` (perplexity) evidence when present in `analysis.json`:
  a `quality_suite` row with `suite_id` `"perplexity-gate@legacy"` and
  metrics `quality.perplexity.ppl` / `quality.perplexity.delta_pct`, so
  pre-existing lossy-gate evidence is queryable alongside future quality
  runs.

Identity fields come from the session's `model.json`, `hardware.json`
(through `config.hardware_signature`), and `llamacpp.json`. `ts` is the
session `created` timestamp; for the winner row, the confirmation entry's
timestamp when recoverable from the journal, else `created`.

### 5.2 Marathon runs

For each `<root>/marathon/<run>/` with a readable `marathon.json`:

- `operating_point` rows from the matrix block (one per cell; `status`
  passed through; deferred/failed cells become rows with empty metrics —
  coverage gaps are visible, not hidden).
- one `ab_verification` row for the final verification: config = champion,
  metrics = pooled champion means, `replicated` = verdict == champion win,
  `status` = `replicated` / `not_replicated` / `tie`.
- Round sessions are already harvested by §5.1 (they are ordinary
  sessions); Marathon harvesting must not duplicate them.

### 5.3 Night Shift runs

For each `<root>/nightshift/<run>/` with a readable `nightshift.json`:
one `calibration` row per calibration item (verification and transfer):
config = the reference config, metrics = calibration means + drift
percentages, `status` = the calibration verdict. Tunes started by Night
Shift are ordinary sessions, harvested by §5.1.

### 5.4 Quality runs

For each `<root>/quality/<run>/` with a readable `quality.json`
(schema in quality-eval §10): one `quality_suite` row per suite entry.
`config` = the evaluated configuration (`null` for defaults), `ctx` = the
evaluated context, `suite_id` from the file, metrics copied verbatim,
`confirmed: true` (quality scores are exact grades, not noisy samples —
the field means "final", not "statistically confirmed"). Absence of a
`quality/` subtree is normal, not a warning.

### 5.5 Tolerance rules

Harvesting must never fail the command: a missing, torn, or unparsable
file, an unknown schema version, or a semantically incomplete record skips
that unit (session / run / row) and appends one human-readable string to
`warnings` naming the path and reason. Version tolerance follows each
source's own rules (e.g. `analysis.json` additions are read tolerantly per
DESIGN §11.1). Harvesters read only the files named in §§5.1–5.4 plus
`session.json`/`journal.jsonl` where specified — no directory-wide
scraping.

## 6. Build semantics

### 6.1 Determinism

`harvest(roots)` is deterministic: given identical evidence bytes it
produces an identical `ResultsMatrix` (row order: sorted by
`(model_name or "", model_fingerprint, kind, ctx or -1, depth or -1, ts,
row_id)`; JSON serialized with sorted keys). The document's `generated`
field is the **maximum evidence timestamp harvested** — not wall-clock —
so the artifact is byte-reproducible. `current` flags are computed at
build time per §2.

### 6.2 Artifact: `results-matrix.json` (schema_version 1)

```json
{
  "schema_version": 1,
  "generated": "<max evidence ts, UTC ISO>",
  "roots": ["<abs path>", "..."],
  "row_count": 0,
  "model_count": 0,
  "rows": [ { "...": "ResultRow fields per §4" } ],
  "warnings": ["<path>: <reason>", "..."]
}
```

Written atomically (temp file + rename) under `<output>/`, together with
`results-matrix.md` (§8). Writes are confined to the output directory with
the same resolve-and-verify discipline as `Session` (DESIGN §12); the
matrix writer never writes anywhere else, and never inside a session, run,
or `registry.jsonl`.

### 6.3 Build summary (`matrix build --json`)

`{"output": dir, "rows": n, "current_rows": n, "models": n, "roots": [...],
"kinds": {kind: count}, "warnings": [...]}`.

### 6.4 Post-command refresh

After `tune`, `resume`, `nightshift`, `marathon`, `revalidate`, and
`quality` complete (any exit code that produced/updated evidence: 0, 1, or
4), the owning command invokes `resultsmatrix.refresh(sessions_dir)`
best-effort. If `<root>/matrix/results-matrix.json` is an existing schema-v1
artifact whose configured roots include `root`, refresh re-harvests that full
root set; otherwise it harvests only `root`. This preserves an explicitly built
multi-root matrix across later runs in its owning sessions root. A foreign,
copied artifact that does not name the owner cannot redirect refresh writes. An
unreadable, corrupt, unsupported-schema, or otherwise invalid artifact is treated
as untrusted configuration: refresh warns that root preservation was discarded
and rebuilds from the owning root alone.

A refresh failure prints a one-line warning to stderr and never alters the
command's exit code, output files, or evidence. This keeps the materialized
artifact fresh for agents that read the JSON file directly instead of invoking
`matrix query`. Tune/resume/NightShift/Marathon/revalidate hooks live in CLI
epilogues; quality performs the same guarded Phase-4 refresh after writing its
frozen `quality.json` contract (NFR-5/NFR-6).

## 7. Query semantics (`matrixquery.py`)

Pure functions: `apply(matrix, spec) -> tuple[ResultRow, ...]` where
`spec` captures use case / sort, filters, and the optional current
identity. Default row pool: `current` rows only, `status` in
`{"ok", "replicated", "consistent"}` plus `recommendation`-class rows,
`confirmed` rows only. `--include-unconfirmed` widens to unconfirmed
kinds (envelope, depth profile, operating points measured at search reps);
the human output then labels those rows `unconfirmed`.

### 7.1 Named use cases

| name | pool filter | rank key (descending) | tie-break |
|---|---|---|---|
| `max-pp` | kinds `recommendation`, `baseline`, `operating_point` | `perf.pp` | `perf.tg`, then newer `ts` |
| `max-tg` | same | `perf.tg` | `perf.pp`, then newer `ts` |
| `balanced` | same | `sqrt(perf.pp × perf.tg)` | newer `ts` |
| `max-context` | rows with `ctx.validated` | `ctx.validated` | `perf.tg`, then newer `ts` |
| `coding` | `quality_suite`, suite `coding` | `quality.coding.score` | `perf.tg` of the same config's `recommendation` row when present, else newer `ts` |
| `tool-use` | `quality_suite`, suite `tooluse` | `quality.tooluse.score` | as `coding` |
| `agentic` | `quality_suite`, suite `agentic` | `quality.agentic.score` | as `coding` |
| `instruction` | `quality_suite`, suite `ifollow` | `quality.ifollow.score` | as `coding` |
| `quality-overall` | `quality_suite` rows carrying `quality.overall` | `quality.overall` | as `coding` |

Honesty rules (normative):

- Cross-model ranking uses **raw throughputs**, never improvement
  percentages (each session's improvement is relative to its own
  baseline; percentages are not comparable across models).
- Rows whose rank metric is absent are excluded from that use case, and
  the human output states how many rows were excluded and why (no silent
  drops).
- A use case never mixes hardware: when the pool contains more than one
  `hardware_hash`, rows are grouped and ranked per hardware group, with
  the current machine's group (when known) listed first. Numbers measured
  on different hardware are never presented as one ranking.
- Ties within a use case are ordered deterministically (table above);
  the query output is reproducible.

### 7.2 Filters

Applied before ranking, in this order: `current`/superseded, model, quant,
kind, suite, ctx (row `ctx >= N`; rows with `null` ctx are excluded when
the filter is given), depth (exact), min-pp/min-tg, confirmed, compat.
All filters are conjunctive.

### 7.3 Compatibility annotation

With `--llama-bin`: run the existing hardware assessment and llama.cpp
probe (same helpers `scan` uses), derive the current
`(hardware_hash, build_discriminator)`, and annotate every returned row
`compat: current | build_changed | hardware_changed | both`. Without
`--llama-bin`, `compat: unknown`. The annotation is always present in
output; `--compat current` additionally filters. Agents selecting a
configuration to run **now** are expected to pass `--llama-bin` and
`--compat current`; the README documents this idiom.

### 7.4 Query output

`--json`: `{"use_case"|"sort": ..., "filters": {...}, "identity":
{"hardware_hash": ..., "build_discriminator": ...} | null, "groups":
[{"hardware_hash": ..., "hardware_signature": [...], "rows": [ResultRow +
{"rank": n, "compat": ...}]}], "excluded": {"missing_metric": n, "...": n},
"warnings": [...]}`.

Human output: one table per hardware group — rank, model (name + quant),
config summary (`ngl/ncmoe/fa/ub/b/t` compact form), ctx, depth, the rank
metric, pp, tg, confirmed/replicated marks, compat, evidence dir. Below
the table: the reproduce pointer for the top row (its session's
`recommended.sh` when the row is a recommendation).

## 8. Rendering and export (`matrixreport.py`)

- `render_markdown(matrix) -> str` — `results-matrix.md`: header
  (generated, roots, counts, warnings), then one section per model
  ordered by name: identity line (name, quant, fingerprint16, size),
  a **Best results** table (best current row per use case that has data,
  with compat column blank — the file is identity-neutral), an
  **Operating points** table when `operating_point`/`context_envelope`
  rows exist, a **Quality** table when `quality_suite` rows exist, and a
  **History** count line (superseded rows are counted, not listed).
  Rendering consumes only the `ResultsMatrix` document (mirroring
  `report.py`'s analysis-only rule) so the file is regenerable.
- `render_csv(rows) -> str` — flat rows: all §4.1 scalar fields plus one
  column per metric name present anywhere in the export (absent ⇒ empty
  cell). Deterministic column order: fixed fields, then sorted metric
  names.
- Markdown export (`matrix export --format md`) reuses `render_markdown`.

## 9. Architecture

New modules and touched files:

```
src/llamatune/
  resultsmatrix.py  # §§4–6 — row model helpers, harvesters, build/refresh
  matrixquery.py    # §7  — use cases, filters, ranking (pure)
  matrixreport.py   # §8  — markdown/CSV rendering (pure)
  types.py          # additions only (§9.1)
  cli.py            # `matrix` sub-app + §6.4 refresh epilogues only
tests/
  unit/test_resultsmatrix.py  test_matrixquery.py  test_matrixreport.py
  unit/test_cli_matrix.py
  integration/test_matrix_e2e.py
```

Layering (extends DESIGN §13): `matrixquery` and `matrixreport` import
nothing above stdlib + `types` and never touch the filesystem;
`resultsmatrix` reads evidence files and writes only under the matrix
output directory via its own confined writer; no module executes
processes; `cli` stays thin with lazy imports. `registry.py`,
`session.py`, and every engine module are consumed read-only and
unmodified.

### 9.1 `types.py` additions (frozen, slots, kw_only — exact fields)

```python
ResultRow(
    row_id: str, kind: str, current: bool,
    model_fingerprint: str, model_name: str | None, model_path: str,
    quant: str | None,
    hardware_hash: str, hardware_signature: tuple[Any, ...],
    build_discriminator: str, build_commit: str | None,
    config: TrialConfig | None,
    ctx: int | None, depth: int | None,
    pp_workload: int | None, tg_workload: int | None,
    suite_id: str | None,
    metrics: dict[str, float],          # treated as immutable by convention
    status: str,
    confirmed: bool, replicated: bool | None,
    reps: int | None, noise_floor_cv: float | None,
    source_root: Path, evidence_dir: Path, ts: str,
)

ResultsMatrix(
    schema_version: int, generated: str,
    roots: tuple[Path, ...],
    rows: tuple[ResultRow, ...],
    warnings: tuple[str, ...],
)

MatrixQuerySpec(
    use_case: str | None, sort: str | None, ascending: bool,
    model: str | None, quant: str | None,
    kinds: tuple[str, ...], suite: str | None,
    ctx_min: int | None, depth: int | None,
    min_pp: float | None, min_tg: float | None,
    confirmed_only: bool, current_only: bool,
    compat: str,                          # "any" | "current"
    limit: int,
)
```

Existing dataclasses are **not** modified. `ResultRow.to_dict` /
`from_dict` round-trip exactly (configs via `TrialConfig.to_dict`).

### 9.2 Frozen cross-module interfaces

```python
resultsmatrix.harvest(roots: tuple[Path, ...]) -> ResultsMatrix
resultsmatrix.build(roots, output_dir: Path) -> dict[str, Any]   # §6.3 summary
resultsmatrix.refresh(root: Path) -> None                        # §6.4, best-effort
resultsmatrix.load_artifact(path: Path) -> ResultsMatrix         # read + validate
matrixquery.USE_CASES: dict[str, UseCaseDef]                     # §7.1 table
matrixquery.apply(matrix: ResultsMatrix, spec: MatrixQuerySpec,
                  identity: tuple[str, str] | None)
    -> dict[str, Any]                                            # §7.4 payload
matrixreport.render_markdown(matrix: ResultsMatrix) -> str
matrixreport.render_csv(rows: tuple[ResultRow, ...]) -> str
```

`UseCaseDef` is a small frozen dataclass local to `matrixquery`
(`name, description, kinds, suite, metric_fn, tiebreak_fn`) — not in
`types.py` because no other module constructs one.

## 10. Safety invariants

All of DESIGN §14 unchanged. Additionally: the matrix feature is
**strictly read-only over evidence** — it never writes, renames, or locks
any file inside a session, run, or registry; its only writes are the
atomic artifact writes under the matrix output directory; it executes no
child processes (the `--llama-bin` probe reuses the existing scan helpers,
which run `llama-bench --help` exactly as `scan` does — that is the sole
subprocess, and only in `matrix query --llama-bin`); no network; no
credentials or environment values in artifacts.

## 11. Testing strategy

### 11.1 Fixtures

A `tests/fixtures/matrix_evidence.py` helper builds synthetic evidence
trees in a temp root: minimal completed sessions (hand-written
`session.json`/`model.json`/`hardware.json`/`llamacpp.json`/
`analysis.json`/`journal.jsonl`), a Marathon run with `marathon.json`, a
Night Shift run with `nightshift.json`, and a quality run with
`quality.json` — all content deterministic and parameterized (fingerprint,
hardware, build, metrics). No fake bench needed for unit tests; the
integration test reuses the existing `fake_llama_bench` end-to-end.

### 11.2 Unit

- Harvesters: every §4.2 kind produced from canned evidence with exact
  field mapping; safe-fallback baseline status; lossless winner skipped
  when identical to winner; quality-gate legacy row; corrupt/missing files
  produce warnings and skip only the affected unit; no writes anywhere
  (assert via read-only tree).
- Identity: `row_id` stability across field ordering; `current`
  computation with ts ties broken by directory name; hardware hash
  matches `config.hardware_signature` canonicalization.
- Determinism: two harvests of the same tree are byte-identical after
  serialization; `generated` equals the max evidence ts.
- Query: each §7.1 use case's pool, rank, tie-break, and
  excluded-count against a constructed matrix; per-hardware grouping
  (never a mixed ranking); every §7.2 filter individually and combined;
  compat annotation for all four states plus `unknown`.
- Rendering: §8 structural assertions per section; CSV column
  determinism and metric union; regenerability (render twice, identical).
- CLI: sub-app validation exits, option assembly to `MatrixQuerySpec`,
  `--json` shapes, exit 0 on empty matrix, exit 3 on missing roots.

### 11.3 Integration (fake bench + tiny GGUFs, no GPU, no network)

1. Run a real `tune` with the fake bench, then `matrix build`:
   baseline + recommendation rows exist with the session's confirmed
   numbers; artifact files parse; `matrix query --use-case max-tg --json`
   ranks the recommendation first.
2. Two tunes of the same model (second improving on the first) →
   supersession: one `current` recommendation row, history retained,
   `--include-superseded` shows both.
3. Multi-root: two roots each with a session → merged matrix; rows carry
   the correct `source_root`; artifact written to the first root.
4. Post-command refresh: `tune` epilogue produces `<root>/matrix/`
   artifacts without being asked; a refresh failure (unwritable output
   dir) leaves `tune`'s exit code and evidence untouched.
5. Corrupt session mixed with valid ones → build succeeds, warning names
   the corrupt path, valid rows unaffected.

## 12. Documentation

README gains a "Results Matrix" section: what it aggregates, the
single-source-of-truth contract, `matrix build/query/show/export`
synopses, one worked `--use-case max-tg --llama-bin … --compat current`
example, the agent idiom (query with `--json`, or read
`<root>/matrix/results-matrix.json` directly), and where artifacts land.
DESIGN.md, nightshift-design.md, and marathon-design.md are not modified;
this document is the Results Matrix spec.

## 13. Compatibility and versioning

`results-matrix.json` carries `schema_version: 1`. Future versions may
add row fields and metric namespaces; readers ignore unknown fields and
namespaces. A reader encountering a *newer* major schema than it knows
reports it plainly instead of guessing. The harvesters' source tolerance
follows each producer's own schema rules and never blocks on unknown
additions.

## 14. Future work (post-v1)

`llamatune best` consulting the matrix; cross-machine matrix merge with
per-host namespacing; watch/daemon refresh; SQLite or DataFusion-style
query surface; confidence intervals recomputed from journals; Pareto
frontier views across models; matrix-driven Night Shift planning (spend
the shift where the matrix has gaps — coverage-aware scheduling); trend
lines across time per row_id; pruning/compaction policy for very large
histories.

---


## Appendix: Requirements checklist

Functional:

- FR-1 `harvest` produces every §4.2 kind from the corresponding evidence
  with the §4.1 field mapping, tolerantly (§5.5) and read-only.
- FR-2 Row identity (`row_id`, hardware hash, build discriminator) follows
  §2 exactly; supersession marks exactly one `current` row per `row_id`.
- FR-3 Confidence fields (`confirmed`, `replicated`, `reps`,
  `noise_floor_cv`) are populated conservatively per §4.4.
- FR-4 Harvest and serialization are deterministic; `generated` is the max
  evidence timestamp (§6.1).
- FR-5 `results-matrix.json` matches §6.2, is written atomically, and all
  writes are confined to the output directory.
- FR-6 Legacy perplexity quality-gate evidence appears as
  `quality_suite` rows (§5.1).
- FR-7 Every §7.1 use case filters, ranks, and tie-breaks as specified;
  cross-model rankings use raw throughput only.
- FR-8 Rankings are grouped per hardware hash; mixed-hardware rankings
  never occur.
- FR-9 Exclusions are counted and reported; nothing is silently dropped.
- FR-10 All §7.2 filters compose conjunctively in the specified order.
- FR-11 `render_markdown` and `render_csv` are pure, deterministic, and
  regenerable from the document alone.
- FR-12 `llamatune matrix build|query|show|export` exist with the §3
  options and §3.1 exit codes; `--json` shapes match §§6.3 and 7.4.
- FR-13 `--llama-bin` yields compat annotations (§7.3); `--compat current`
  without it exits 2.
- FR-14 Empty results exit 0 with an explicit message; corrupt evidence
  never changes an exit code.
- FR-15 Command epilogues refresh `<root>/matrix/` best-effort; a refresh
  failure never affects the command's exit code or evidence (§6.4).
- FR-16 README documents the feature and the agent query idiom.

Non-functional:

- NFR-1 The feature executes no benchmark process; the only subprocess is
  the existing scan probe in `matrix query --llama-bin`.
- NFR-2 Deterministic outputs; no wall-clock timestamps in artifacts.
- NFR-3 All existing quality gates pass repo-wide: `ruff format --check`,
  `ruff check`, `mypy` (strict), `pytest` with branch coverage ≥ 85%.
- NFR-4 No new runtime dependencies; Python ≥ 3.11.
- NFR-5 Existing commands' behavior and outputs are unchanged except the
  added sub-app and the §6.4 stderr-only refresh epilogue.
- NFR-6 No modifications to `search.py`, `registry.py`, `session.py`,
  `calibrate.py`, or any engine module — the matrix reads; it does not
  fork or wrap the methodology.
