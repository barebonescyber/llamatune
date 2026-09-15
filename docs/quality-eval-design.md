# llamatune Quality Evaluation — Design, Architecture, and Requirements (v1)

Status: v1 specification, approved for implementation.
Audience: users, contributors, and implementers. This document is
authoritative for the Quality Evaluation feature;
[DESIGN.md](../DESIGN.md) remains authoritative for everything it already
specifies, [nightshift-design.md](nightshift-design.md) for Night Shift,
[marathon-design.md](marathon-design.md) for Marathon, and
[results-matrix-design.md](results-matrix-design.md) for the Results
Matrix, which ingests this feature's results (§10 here is the normative
contract the matrix harvests). Implementation deviations must be documented
and reviewed. Read `AGENTS.md` before starting; its boundaries and safety
invariants apply as amended in §12.

## 1. Product definition

llamatune today measures **how fast** a model runs (pp/tg throughput,
context feasibility) but not **how well** it works. `llamatune quality`
adds the second axis: given a GGUF model and a specific runtime
configuration, it launches `llama-server` on loopback, drives a versioned
set of deterministic evaluation suites through the OpenAI-compatible API,
grades every response with auditable graders, and records a quality vector
alongside full request/response evidence:

- **coding** — coding correctness: self-contained programming tasks graded
  by static checks; declared execution graders are currently skipped;
- **tooluse** — tool-use quality: scripted single- and multi-turn tool-call
  scenarios grading call validity, tool selection, and argument accuracy;
- **agentic** — agentic evaluation: multi-step goal tasks against a
  deterministic simulated environment, grading completion, efficiency, and
  invalid-call behavior;
- **ifollow** — instruction following and output quality-of-life: format
  compliance (JSON, length, language, forbidden content) and long-context
  needle retrieval within the configuration's validated context;
- **perplexity** — the existing `llama-perplexity` corpus measurement,
  wrapped so its result lands in the same quality vector.

Why configuration matters: most quality behavior is a property of the model
and quantization, but **lossy KV-cache types and flash-attention numerics
can change outputs** — exactly the settings the tuner may recommend under
`--allow-lossy` (DESIGN §9.2 already labels such winners
"quality-affecting — validate output quality before adoption"). This
feature is that validation: every quality run records the exact evaluated
`TrialConfig`, and `--compare-lossless` (§4.4) measures the lossy winner
against the lossless alternate head-to-head.

Every completed run emits `quality.json` (the machine summary the Results
Matrix harvests), `quality-report.md`, and per-task evidence. Scores are
**suite-relative**: they compare models and configurations run against the
same `suite_id` on this machine; they are not claims about public
benchmark equivalence. Bundled suites are smoke-scale (§6.6) and
user-extensible via `--suite PATH`.

### 1.1 Relationship to existing machinery

`quality` is a new measurement mode, not a fork of the tuning methodology:
it reuses binary discovery (`llama.py` already resolves `llama-cli`,
`llama-server`, and `llama-perplexity`), the execution contract
(`executor.py` launches and reaps the server child), the registry lookup
(to resolve `--config best`), the session-writer discipline (a confined
`QualityRun` writer modeled on `Session`/`MarathonRun`), and the Results
Matrix refresh hook. Throughput semantics (DESIGN §§6–11) are untouched —
quality runs measure grades, not t/s, and never write throughput evidence.

### 1.2 Non-goals (v1)

Concurrent requests (strictly serial — one in-flight request, one child
process at a time, ever); sampling-parameter sweeps (temperature/seed are
fixed for determinism; sweeps are future work §15); native OpenAI `tools`
API usage (v1 uses the prompt-embedded protocol §6.2 for
build-independence); executing model-generated code (temporarily disabled
per §7.1); network access beyond the loopback
server (hard requirement, §12); downloading suites, models, or corpora;
judging by another LLM (all graders are deterministic code); Elo or
cross-run statistical modeling; re-enabling execution support on Windows
(§7.1).

## 2. Definitions

- **Suite**: a versioned set of tasks of one kind, identified by
  `suite_id = <name>@<version>+<sha256-12 of canonical task JSON>`. Equal
  `suite_id` ⇒ byte-identical tasks ⇒ comparable scores.
- **Task**: one evaluation unit (§6): prompt(s), constraints, graders,
  weight. Task ids are unique within a suite.
- **Turn**: one request/response exchange. Multi-turn tasks script the
  harness side deterministically.
- **Grader**: a pure, deterministic function from a response (and task
  data) to a grade contribution (§7). Graders never call the model.
- **Score**: per task, a float in [0, 1]; per suite, the weighted mean of
  task scores plus suite-specific rates (§7.2).
- **Evaluated configuration**: the exact `TrialConfig` (or defaults =
  `None`) mapped to `llama-server` flags for the run (§4.2).
- **Exec tier**: the retained, inactive resource-limit runner for
  model-generated code (§7.1). `--exec` is currently refused on every
  platform.
- **Simulated environment**: the deterministic state machine an agentic
  task defines (§6.3); tool calls mutate JSON state via declarative
  effects; nothing outside the harness process is touched.
- **Run directory**: `<sessions-dir>/quality/<model-stem>-<UTCyyyymmdd-
  hhmmss>-<6hex>/` (§9).

## 3. CLI specification

One new Typer command in `cli.py` (thin, lazy imports, per DESIGN §3):

```
llamatune quality MODEL.gguf [options]
```

- `MODEL.gguf` (positional, required) — one readable GGUF file; sharded
  models: pass the `-00001-of-…` head, as with `tune`.
- `--llama-bin DIR`, `--sessions-dir DIR` (default
  `./llamatune-sessions`) — as in `tune`. `llama-server` must be
  resolvable (exit 3 otherwise, naming the directory searched).
- `--config best|defaults` (default `best`) — the evaluated
  configuration. `best` performs the registry lookup exactly as
  `llamatune best` does (model fingerprint, current hardware signature,
  build identity); a hit uses the recorded config. A miss or stale result
  falls back to defaults **with a prominent warning** naming the stale
  reasons, unless `--strict-config` is given (then exit 3).
- `--config-session DIR` — evaluate the winner recorded in that session's
  `analysis.json` (fingerprint must match the model; exit 2 otherwise).
  Mutually exclusive with `--config`.
- `--compare-lossless` — when the evaluated config uses a lossy cache
  type and a lossless alternate exists (the config-source session's
  `lossless_winner`, else the same config with `f16` cache types), run
  every selected suite against **both** configs sequentially and report
  per-suite deltas (§4.4).
- `--suite NAME|PATH` (repeatable; default: all bundled suites except
  `perplexity`) — bundled suite name (`coding`, `tooluse`, `agentic`,
  `ifollow`, `perplexity`) or a path to a user task file validated
  against the §6 schema.
- `--list-suites`: print bundled suite ids, task counts, and exit 0 unless
  the disabled `--exec` option is also passed, which exits 2.
- `--tasks GLOB` (repeatable) — filter task ids within selected suites.
- `--exec`: Generated-code execution is temporarily disabled on every platform.
  Passing `--exec` returns exit 2 before model discovery or run creation.
  Resuming a run whose saved options enable execution also returns exit 2.
  Quality evaluation without `--exec` remains available and skips `exec_python` graders.
  The retained resource-limit runner is not a security boundary for untrusted code.
  Re-enablement requires mandatory filesystem and network confinement, verified memory
  limits, and adversarial tests. No reduced-isolation override is supported.
- `--ctx-size N` (default 8192) — server context. When the config source
  records a validated context smaller than N, warn; needle tasks
  auto-scale to fit (§6.4). Values below a suite's `ctx_min` skip those
  tasks with a recorded reason.
- `--quality-corpus PATH` — required when the `perplexity` suite is
  selected (exit 2 if selected without it); passed to `llama-perplexity`
  exactly as the existing quality gate does.
- `--reps N` (default 1, max 5) — repetitions per task; the recorded task
  score is the **worst** repetition (quality claims should be
  conservative); disagreement across reps sets an `unstable` flag.
- `--max-tokens N` (default 1024) — response cap; tasks may lower it,
  never raise it.
- `--request-timeout S` (default 300) / `--server-start-timeout S`
  (default 600) — per-request and server-readiness bounds.
- `--seed N` (default 42) — sampling seed passed on every request.
- `--resume RUN_DIR` — continue an interrupted run (§9.3). Mutually
  exclusive with every option above except `--llama-bin` (options are
  recorded in the run and reused).
- `--dry-run` — print the resolved plan (config + provenance, server
  argv, suites, task counts, exec tier state, estimates) and exit 0
  without launching anything.
- `--json` — machine-readable summary (the `quality.json` content) on
  stdout instead of human text.

Validation errors (unknown suite name, unreadable suite path, schema
violation in a user suite, `--reps` out of range, conflicting config
options) exit 2 with a one-line `error:` message.

### 3.1 Exit codes

- `0` — run completed; every selected task was graded (pass **and** fail
  are both valid grades — scores are data, not success criteria).
- `1` — run completed but at least one task ended `error` (harness-level:
  request timeout, malformed server reply, or another grader failure) or a suite
  was skipped after server relaunch failure (§5.3); evidence and reports
  are still written.
- `2` — usage or configuration error.
- `3` — environment error: `llama-server` missing/unusable, model
  unreadable, server never became ready (both §5.3 attempts), or
  `--strict-config` with no usable config.
- `4` — interrupted by signal with work remaining; `--resume RUN_DIR`
  continues (§9.3).

## 4. Evaluation pipeline (`quality.py`)

Executed strictly serially. All timing uses an injectable clock
(`now_fn`) and monotonic deadline math, as Night Shift §7.1.

- **Phase 0 — Resolve**: assess hardware, probe llama.cpp, inspect the
  model (existing helpers); resolve the evaluated config (§3 `--config`
  rules) and record its provenance (`registry`, `session`, `defaults`,
  plus the source path); load and validate suites; write `run.json`,
  `model.json`, `hardware.json`, `llamacpp.json`, `config.json`,
  `plan.json`.
- **Phase 1 — Serve**: launch `llama-server` (§5) with the evaluated
  config at `--ctx-size`; wait for readiness.
- **Phase 2 — Evaluate**: for each suite, for each task (deterministic
  order: suite order as selected, tasks in file order): run the task's
  turns (§6), grade (§7), journal `task_result`, write per-task evidence.
  The `perplexity` suite runs after HTTP suites and uses
  `llama-perplexity` instead of the server (§6.5) — the server is stopped
  first (one child at a time, always).
- **Phase 3 — Compare** (only with `--compare-lossless`): stop the
  server, relaunch with the lossless config, repeat Phase 2 for the same
  suites, compute deltas (§4.4).
- **Phase 4 — Report**: write `quality.json` (§10) and
  `quality-report.md` (§11.4); stop the server; refresh the Results
  Matrix for this root best-effort (results-matrix §6.4 semantics — a
  refresh failure is a stderr warning only).

### 4.1 Determinism

Every request sets `temperature 0`, `top_k 1`, `top_p 1`, `seed` from
`--seed`, and the task's token cap. Prompts are rendered by the server's
own chat template (`/v1/chat/completions`), so model-specific templating
is handled by llama.cpp, not re-implemented. Residual nondeterminism
(batching numerics, backend kernels) is acknowledged: `--reps > 1`
exists to detect it, and the `unstable` flag reports it honestly rather
than hiding it.

### 4.2 Config-to-server mapping

The evaluated `TrialConfig` maps to `llama-server` flags with the same
mapping `recommended.sh` uses (DESIGN §11.2): `-ngl`, `--n-cpu-moe`,
`-fa on|off`, `-b`, `-ub`, `-t`, `-ctk`, `-ctv`, `--no-mmap`,
`--no-kv-offload`, plus `-c CTX`, `--host 127.0.0.1`, `--port <ephemeral>`,
`--seed N`. Flags are emitted only when the corresponding dimension is
applicable, mirroring `TrialConfig.bench_args` discipline. Defaults
(`config = None`) emit only `-m`, `-c`, host/port/seed.

### 4.3 Repetitions

With `--reps N > 1`, each task's turns are re-run N times (server not
restarted between reps); per-rep scores are recorded; the task score is
the minimum; `unstable = (max_score − min_score) > 0`. Reps multiply
runtime and are bounded at 5.

### 4.4 Lossless comparison

`--compare-lossless` produces two full result sets, labeled `evaluated`
and `lossless`, plus a `comparison` block: per suite,
`delta = evaluated.score − lossless.score` and per-metric deltas. The
report renders the deltas prominently; a negative coding/tooluse delta
beyond 0.05 adds the warning **"lossy cache measurably degrades quality
on this machine"**. When the evaluated config is not lossy the flag is a
no-op with a note. This is the empirical backing for DESIGN §9.2's
"validate output quality before adoption" label.

## 5. Server lifecycle and loopback client (`qualserver.py`)

### 5.1 Launch and readiness

The server child is executed through `executor.run`-compatible process
management (argv list, allowlisted environment, process group,
stdout/stderr captured to files with caps) but as a **long-lived
supervised child**: `qualserver` extends the executor contract with
`start`/`stop` semantics (`ServerHandle`) while preserving every §7
safety property (bounded captures, group kill, reaping on all paths).
Port selection: bind a loopback socket to port 0, read the assigned
port, close it, and pass that port — collisions surface as startup
failure and retry once with a fresh port. Readiness = HTTP 200 from
`GET /health` polled every 2 s within `--server-start-timeout`; the model
load for large GGUFs dominates this window (the default of 600 s covers
the ~60 GB models this repository tests).

### 5.2 Client

Stdlib only (`http.client`). Hard rules: connections are made exclusively
to literal `127.0.0.1` (never a hostname, never configurable); proxy
environment variables are ignored by construction (no `urllib` opener
chain); request bodies are UTF-8 JSON; responses are bounded reads
(max 8 MiB, truncation ⇒ task `error`); one request in flight at a time;
per-request timeout from `--request-timeout`. The client records every
request and response body verbatim into task evidence (§9) — quality
grading must be auditable end to end.

### 5.3 Failure handling

Server exit or unreachability mid-suite: journal `server_exit` with the
captured stderr tail, relaunch once (fresh port), and re-run the
interrupted task. A second death within the same suite marks the
remaining tasks of that suite `error: server_unavailable`, skips to
Phase 4, and exits 1 — quality runs degrade to partial evidence, never to
hangs. Startup failure retries once; two failures exit 3.

## 6. Suites and task schema (`qualsuites.py`)

Task files are JSON documents:

```json
{
  "schema_version": 1,
  "name": "coding",
  "version": 1,
  "kind": "coding | tooluse | agentic | ifollow",
  "tasks": [ { "id": "...", "...": "kind-specific per §§6.1–6.4" } ]
}
```

Loading canonicalizes (sorted keys, no insignificant whitespace), hashes
to derive the `suite_id`, and validates structurally: unique ids,
required fields per kind, grader types known, weights positive, prompt
sizes bounded (≤ 8 KiB per message except `ifollow` needle tasks). A
schema violation in a bundled suite is a packaging bug (test-asserted);
in a user suite it is exit 2 naming the field.

Common task fields: `id`, `weight` (default 1.0), `max_tokens` (≤ CLI
cap), `ctx_min` (skip with reason when the run ctx is smaller),
`system` (optional system message), `graders` (kind-specific).

### 6.1 `coding` tasks

Fields: `prompt` (user message), `language` (`python` in v1's bundled
suite; the schema is language-open), `graders`: ordered list of

| type | fields | grade |
|---|---|---|
| `code_extract` | `language` | prerequisite: extract the first fenced code block of that language; absence fails the task (score 0) |
| `python_parses` | — | extracted code parses via `ast.parse` |
| `defines` | `name`, `kind: function\|class` | AST contains the definition |
| `signature` | `name`, `params: [..]` | definition's parameters match exactly |
| `contains_regex` / `absent_regex` | `pattern` | regex over the extracted code |
| `exec_python` | `tests` (assert lines), `requires_exec: true` | Execution disabled: skipped with a recorded reason |

Task score: all listed graders are conjunctive gates except those marked
`"partial": true`, which contribute fractionally (`score = passed_weight /
total_weight` over partial graders, gated by the conjunctive ones).
Grading never imports or evaluates model output in-process. No execution
path is active; `exec_python` is skipped while §7.1 remains disabled.

### 6.2 `tooluse` tasks and the embedded tool protocol

v1 uses a **prompt-embedded protocol** so grading is identical across
llama.cpp builds and models: the harness renders the task's tool catalog
(name, description, JSON-schema-like parameter spec) into the system
message together with this fixed instruction: reply with **exactly one**
fenced ` ```json ` block containing either
`{"tool": "<name>", "args": {…}}` or `{"final": "<answer>"}`.

Extraction rule (normative): the first fenced json block in the response;
if none, the first balanced `{…}` object containing a top-level `"tool"`
or `"final"` key; otherwise the turn is an invalid call.

Task fields: `tools` (catalog), `turns`: alternating
`{"user": "..."}` and expectation steps
`{"expect": {"tool": name, "args_equal": {...} | "args_subset": {...}},
"tool_result": {...}}` — when the model's call matches, the scripted
`tool_result` is appended as the tool reply and the conversation
continues; a final `{"expect_final": {"contains": [...], "regex": ...}}`
grades the closing answer.

Per-turn grades: `valid` (extraction succeeded), `tool_correct` (right
tool name, no hallucinated tool), `args_correct` (equality or subset
match after type-tolerant normalization: numbers compared numerically,
strings stripped). Task score = mean of turn grades × final-answer grade.
Suite metrics aggregate these (§7.2).

### 6.3 `agentic` tasks

A deterministic simulated environment, fully described in the task:

```json
{
  "id": "restock-01",
  "environment": {
    "initial_state": { "...": "JSON" },
    "tools": [ { "name": "...", "params": {...},
                 "effects": [ {"set": ["path.to.key", "<value|$arg>"]},
                              {"append": ["path.to.list", "$arg"]},
                              {"incr": ["path.to.num", 1]} ],
                 "result": "template with {state.path} and {args.x}" } ],
    "goal": { "state_subset": { "...": "JSON" } },
    "max_steps": 8, "optimal_steps": 3
  },
  "prompt": "..."
}
```

The harness owns the state (a JSON object inside the process); each
extracted tool call (same §6.2 protocol) applies the tool's declarative
effects and returns the rendered result string as the tool reply. Unknown
tools or malformed calls return a scripted error string and count as
invalid. The episode ends at `{"final": …}`, goal satisfaction, or
`max_steps`.

Grades: `completed` (goal subset satisfied), `efficiency`
(`optimal_steps / steps_used`, 0 when not completed, capped at 1),
`invalid_rate` (invalid calls / total calls). Task score =
`completed × (0.7 + 0.3 × efficiency)`. Nothing outside the harness
process is ever read, written, or executed by agentic tools — effects are
pure JSON transformations by construction (the schema admits nothing
else).

### 6.4 `ifollow` tasks

Two shapes:

- **Constraint tasks**: `prompt` + `constraints`, each graded
  independently and averaged: `json_object` (response parses as JSON;
  optional `schema_subset`), `max_words` / `max_chars`, `must_include` /
  `must_not_include` (case-insensitive literals), `regex_full`,
  `language` (ASCII-ratio heuristic for v1, documented as such),
  `starts_with` / `ends_with`.
- **Needle tasks**: `kind: "needle"`, `needle` (fact string),
  `question`, `haystack_ratio` (fraction of the run ctx to fill, e.g.
  0.6), `position` (0–1 relative placement). The haystack is generated
  deterministically from the bundled filler corpus (§6.6) and the task
  id as seed; token count is estimated at 4 chars/token (documented
  approximation). Grade: the response contains the needle's answer
  (`must_include` on the needle payload). Needle tasks auto-scale to
  `--ctx-size` and skip (with reason) below their `ctx_min`.

### 6.5 `perplexity` suite

Not HTTP: runs `llama-perplexity` through `executor.run` with the
evaluated config's applicable flags, the run ctx, and
`--quality-corpus`, exactly mirroring the existing lossy-gate invocation
shape. Metrics: `quality.perplexity.ppl` (final corpus perplexity) and,
under `--compare-lossless`, `quality.perplexity.delta_pct` versus the
lossless config's run. The server is never running concurrently.

### 6.6 Bundled suites (normative content requirements)

Shipped as package data under `src/llamatune/evaltasks/` (plus
`filler.txt`, ~64 KiB of original neutral English prose for haystacks).
Authored content must be: deterministic to grade, self-contained, free of
copyrighted text, free of network/filesystem references, and solvable
within the token caps. Minimum task counts (v1): `coding` 20 (≥ 8 with
`exec_python` graders), `tooluse` 15 (≥ 5 multi-turn), `agentic` 8,
`ifollow` 15 (≥ 5 needle). Suite versions bump on any task change; the
`suite_id` hash makes silent drift impossible.

## 7. Grading and scoring (`qualscore.py`)

All graders are pure functions `grade(task, responses, exec_runner |
None) -> TaskGrade`; the only impure collaborator is the injected exec
runner (§7.1), used solely by `exec_python`. Every grader records
`{"grader": type, "passed": bool, "detail": str}` — grades must be
explainable from evidence alone.

### 7.1 Execution sandbox (`sandbox.py`): disabled pending verified confinement

Generated-code execution is temporarily disabled on every platform.
Passing --exec returns exit 2 before model discovery or run creation.
Resuming a run whose saved options enable execution also returns exit 2.
Quality evaluation without --exec remains available and skips exec_python graders.
The retained resource-limit runner is not a security boundary for untrusted code.
Re-enablement requires mandatory filesystem and network confinement, verified memory
limits, and adversarial tests. No reduced-isolation override is supported.

Inactive implementation detail: The retained resource-limit runner writes
the extracted code plus the task's assert lines to `main.py` in a fresh temp
directory; it would spawn `sys.executable -I -S -B main.py` via a dedicated sandbox runner
with: argv-list execution (no shell), an **empty environment** except
`PATH` to the interpreter's directory, cwd = the temp dir,
`start_new_session` + group kill semantics (executor §7 discipline),
RLIMIT_CPU 10 s (wall cap 30 s), RLIMIT_AS 512 MiB, RLIMIT_FSIZE 1 MiB,
RLIMIT_NOFILE 32, RLIMIT_CORE 0, stdout/stderr capped at 64 KiB. Pass =
exit 0. The temp directory would be removed afterward. No shipped entry
point invokes this runner while execution is disabled.

Inactive historical limitation: POSIX rlimits do not block network
syscalls. The retained runner was an *accident barrier* for locally
generated code, not a security boundary against adversarial models. On
Linux, when `unshare` with user+network namespaces is available, the
retained runner wraps its child in `unshare -rn` (probed once; absence
degrades with a journal note). On Windows and platforms without `resource`,
the prior implementation returned exit 2 with a message naming the
limitation.

No execution tier currently amends the "model output is data, never
executed" invariant. See §12.

### 7.2 Aggregation

Per suite:

- `score` — weighted mean of task scores (skipped tasks excluded from
  the denominator; their ids and reasons are listed in evidence).
- `pass_rate` — fraction of graded tasks with score ≥ 0.999.
- kind-specific rates: `coding`: none beyond the above (plus
  `exec_enabled` flag); `tooluse`: `valid_call_rate`,
  `correct_tool_rate`, `args_accuracy` (means over turns);
  `agentic`: `completion_rate`, `efficiency` (mean over completed),
  `invalid_call_rate`; `ifollow`: `format_rate` (constraint tasks),
  `needle_accuracy` (needle tasks); `perplexity`: `ppl`
  (+ `delta_pct` under comparison).

Overall: `quality.overall` = unweighted mean of the selected suites'
`score` values (perplexity excluded — it has no [0, 1] score), reported
with the explicit caveat that it is an advisory scalar over
suite-relative numbers.

All metric names above, prefixed `quality.<suite>.`, are the exact names
the Results Matrix ingests (results-matrix §4.3).

## 8. Interruption and signals

First `SIGINT`/`SIGTERM`: stop flag; the in-flight request finishes or
times out, the current task journals its state,
`quality.json` and the report are written from graded work, the server is
stopped, exit 4. Second signal: process-group termination of the child
(server only), `interrupted` journal entry, evidence remains
resumable. No signal path may orphan the server: `stop` runs in a
`finally` on every exit route, and the process-group kill covers escapes.

## 9. Evidence: run directory, journal, resume

```
<sessions-dir>/quality/<model-stem>-<UTCyyyymmdd-hhmmss>-<6hex>/
  run.json            # schema_version 1, tool version, argv, options,
                      # resolved config + provenance, suite ids, created (UTC)
  hardware.json  llamacpp.json  model.json  config.json  plan.json
  journal.jsonl       # append-only, one JSON object per line, fsync per line
  server/<n>/         # command.json, stderr.log per server launch
  tasks/<suite-name>/<task-id>/[rep-<r>/][side-<evaluated|lossless>/]
                      # request-<turn>.json, response-<turn>.json, grade.json
  perplexity/[side-*/]# command.json, stdout, stderr per §6.5
  quality.json        # §10 — final machine-readable summary (== --json stdout)
  quality-report.md
```

Journal entry types (every entry carries a UTC timestamp):
`quality_start`, `plan`, `config_resolved`, `server_start`,
`server_ready`, `server_exit`, `task_start`, `task_result`,
`suite_summary`, `comparison`, `deferred`, `interrupted`, `quality_end`.

### 9.1 Writer

`QualityRun` follows the `MarathonRun` pattern: `create`/`load`, `dir`,
`append(entry)`, `write_json(name, payload)`, `write_text(name, text)`,
and path-confined directory helpers (§11.2). Only `QualityRun` writes
inside a quality run directory (the DESIGN §12 layering sentence extends
accordingly). Task and journal writes are flushed per entry so a crash
loses at most the in-flight task.

### 9.2 Task identity

A task execution is identified by
`(suite_id, task_id, rep, side)`. `task_result` journal entries carry
this tuple; grading is idempotent per tuple.

### 9.3 Resume

`--resume RUN_DIR`: reload `run.json` options; re-inspect the model and
re-probe the build; fingerprint or build-discriminator mismatch ⇒ exit 3
(quality numbers are identity-bound, exactly like session resume).
Journaled `task_result` tuples are skipped; the run continues from the
first ungraded task, relaunching the server as needed. A resumed run
appends to the same journal and overwrites only `quality.json` and the
report at the end.

## 10. `quality.json` (schema_version 1) — the matrix contract

```json
{
  "schema_version": 1,
  "run_dir": "<abs path>",
  "created": "<UTC ISO>",
  "model": { "fingerprint": "...", "name": "...", "path": "...",
             "size_bytes": 0 },
  "hardware_signature": [ "..." ],
  "build": { "bench_sha256": null, "help_sha256": "...",
             "build_commit": null },
  "ctx": 8192,
  "seed": 42,
  "reps": 1,
  "config": { "...": "TrialConfig.to_dict() or null" },
  "config_source": "registry | session | defaults",
  "config_provenance": "<path or null>",
  "exec_enabled": false,
  "suites": [
    { "suite_id": "coding@1+ab12cd34ef56", "name": "coding",
      "kind": "coding",
      "metrics": { "score": 0.0, "pass_rate": 0.0 },
      "tasks": [ { "id": "...", "score": 0.0,
                   "status": "graded | skipped | error",
                   "reason": null, "unstable": false,
                   "graders": [ {"grader": "...", "passed": true,
                                  "detail": "..."} ] } ] }
  ],
  "overall": 0.0,
  "comparison": null,
  "warnings": [ "..." ]
}
```

`comparison` (present under `--compare-lossless`):
`{"lossless_config": {...}, "suites": [{"suite_id": ..., "evaluated":
{...metrics}, "lossless": {...metrics}, "delta_score": 0.0}],
"degradation_warning": false}`.

The Results Matrix harvester (results-matrix §5.4) consumes exactly:
`model.fingerprint/name/path`, `hardware_signature`, `build`, `config`,
`ctx`, each suite's `suite_id` + `metrics` (namespaced as
`quality.<name>.<metric>`), `overall` (as `quality.overall`), and
`created`. Fields may be added in later versions; the harvester reads
tolerantly. This block is frozen: neither packet may rename or restructure
these fields without a spec change.

## 11. Architecture

New modules and touched files:

```
src/llamatune/
  qualsuites.py     # §6  — schema, loading, validation, suite_id, bundled
                    #       suite access, haystack generation (pure + pkg data)
  qualscore.py      # §7  — graders, extraction rules, aggregation (pure)
  sandbox.py        # §7.1: retained inactive exec runner (process mgmt only)
  qualserver.py     # §5  — ServerHandle lifecycle + loopback HTTP client
  qualityreport.py  # §11.4 — quality-report.md rendering from quality.json
  quality.py        # §§4, 8, 9 — QualityRun writer, phases, resume,
                    #   run_quality(options, *, now_fn=None) -> QualityOutcome
  types.py          # additions only (§11.1)
  cli.py            # new `quality` command only
src/llamatune/evaltasks/
  coding.json  tooluse.json  agentic.json  ifollow.json  filler.txt
tests/
  fixtures/fake_llama_server.py   # §13.1
  unit/test_qualsuites.py  test_qualscore.py  test_sandbox.py
  unit/test_qualserver.py  test_qualityreport.py  test_cli_quality.py
  integration/test_quality.py
```

Layering (extends DESIGN §13): `qualsuites` and `qualscore` import
nothing above stdlib + `types` and never touch processes or the network
(`qualsuites` reads only its package data and given task paths;
`qualscore` receives the retained exec runner as an injected callable);
the retained `sandbox` implementation manages at most one child process
with executor-grade discipline and no parsing of benchmark semantics, but
no shipped entry point invokes it; `qualserver` owns the
server child and the loopback client and writes only via `QualityRun`
handles passed in; `qualityreport` renders strings from dicts, no I/O;
`quality` orchestrates and owns the only writer for its run directory;
`cli` stays thin with lazy imports. No new runtime dependencies
(stdlib `http.client`, `json`, `ast`, `socket`, `resource`).

### 11.1 `types.py` additions (frozen, slots, kw_only — exact fields)

```python
QualityOptions(
    model_path: Path, llama_bin: Path | None, sessions_dir: Path,
    config_mode: str,                     # "best" | "defaults" | "session"
    config_session: Path | None, strict_config: bool,
    compare_lossless: bool,
    suites: tuple[str, ...],              # names or paths, as given
    task_filters: tuple[str, ...],
    exec_enabled: bool, ctx_size: int,
    quality_corpus: Path | None,
    reps: int, max_tokens: int,
    request_timeout_s: float, server_start_timeout_s: float,
    seed: int, dry_run: bool,
)

TaskGrade(task_id: str, score: float, status: str,   # graded|skipped|error
          reason: str | None, unstable: bool,
          grader_results: tuple[dict[str, Any], ...])

SuiteResult(suite_id: str, name: str, kind: str,
            metrics: dict[str, float],
            tasks: tuple[TaskGrade, ...])

QualityOutcome(run_dir: Path, summary: dict[str, Any], exit_code: int)
```

Existing dataclasses are **not** modified.

### 11.2 Frozen cross-module interfaces

```python
qualsuites.load_suite(name_or_path: str) -> SuiteSpec
qualsuites.bundled_suites() -> tuple[str, ...]
qualsuites.build_haystack(task_id: str, chars: int) -> str
qualscore.extract_code(text: str, language: str) -> str | None
qualscore.extract_call(text: str) -> dict[str, Any] | None   # §6.2 rule
qualscore.grade_task(task: dict, responses: tuple[str, ...],
                     exec_runner: Callable[[str], ExecVerdict] | None)
    -> TaskGrade
qualscore.aggregate(kind: str, grades: tuple[TaskGrade, ...],
                    *, exec_enabled: bool) -> dict[str, float]
sandbox.run_python(code: str, *, timeout_s: float) -> ExecVerdict  # inactive: raises
qualserver.start(run: QualityRun, argv: tuple[str, ...],
                 *, start_timeout_s: float) -> ServerHandle
ServerHandle.chat(messages, *, max_tokens, seed,
                  timeout_s) -> str            # response text or raises
ServerHandle.stop() -> None                    # idempotent
qualityreport.render(summary: dict) -> str
quality.run_quality(options: QualityOptions, *, now_fn=None)
    -> QualityOutcome
QualityRun.create(sessions_dir, *, options, hardware, llama, model, argv)
QualityRun.load(run_dir); .dir; .append(entry)
QualityRun.task_dir(suite, task_id, rep, side); .server_dir(n)
QualityRun.perplexity_dir(side); .write_json(name, payload)
QualityRun.write_text(name, text)
```

`SuiteSpec` and `ExecVerdict` are small frozen dataclasses local to their
defining modules (`SuiteSpec(suite_id, name, kind, tasks)`;
`ExecVerdict(passed, exit_code, timed_out, stdout_tail, stderr_tail)`).

### 11.3 The `quality.py` orchestrator

Owns phase sequencing (§4), the resume skip-set (§9.3), signal handling
(§8), the §5.3 relaunch policy, and the matrix refresh call. Pure
planning helpers (task ordering, skip predicates, deadline math) are
module-level functions testable without filesystem or processes.

### 11.4 `quality-report.md`

Sections: **Summary** (model + config + provenance, ctx, seed, exec tier,
overall score, warnings count); **Configuration** (evaluated config
table, source, lossless comparison verdict when run); **Per-suite
results** (metrics table + failed-task list with one-line grader
details); **Skipped and errored tasks** (with reasons); **Environment**
(hardware, build, server launches); **Reproduce** (the exact `llamatune
quality` invocation and server argv). Rendering consumes only
`quality.json` content, so the report is regenerable.

## 12. Safety invariants

All of DESIGN §14 remains unchanged except for the loopback exception:

1. **Loopback exception to "no network"**: `quality` communicates over
   HTTP exclusively with the `llama-server` child it launched, on literal
   `127.0.0.1`, on a port it selected. No other address, hostname, proxy,
   or interface is ever contacted; the client contains no redirect
   following and ignores proxy environment variables. The no-network
   invariant remains in force for everything else.
No generated-code execution exception is active. Passing `--exec` returns
exit 2 before model discovery or run creation, and Quality evaluation
without it skips `exec_python` graders. The retained runner in §7.1 is
inactive implementation detail, not a security boundary or an available
execution path. Graders parse, match, and AST-inspect; agentic "tools" are
pure JSON state transitions; Quality never executes model-authored output or
passes it to a shell or interpreter. Its confined QualityRun writer records
bounded response data as FR-12 evidence.

Unchanged and load-bearing: never `shell=True`; exactly one active child
process at any moment (server or perplexity); bounded
captures everywhere; writes confined to the run directory; no
system-state mutation; no credentials or environment values in evidence;
suites, corpora, and models are never downloaded.

## 13. Testing strategy

### 13.1 Fixture: `fake_llama_server.py` (normative)

An executable fixture emulating the minimal `llama-server` surface:
binds `--host --port`, serves `GET /health` (200 after
`LLAMATUNE_FAKE_SRV_READY_DELAY_S`, default 0) and
`POST /v1/chat/completions`. Response selection is scripted: the fixture
loads `LLAMATUNE_FAKE_SRV_SCRIPT` (JSON: sha256-12 of the last user
message content → response text; `"*"` → fallback text) so tests choose
pass/fail/malformed behavior per task without changing production
requests. Failure knobs: `LLAMATUNE_FAKE_SRV_DIE_AFTER_N` (exit after N
completions), `LLAMATUNE_FAKE_SRV_MALFORMED=1` (non-JSON body),
`LLAMATUNE_FAKE_SRV_NEVER_READY=1`, `LLAMATUNE_FAKE_SRV_HANG_S` (delay
each completion). A conftest helper generates script files from bundled
suites with known-correct and known-wrong answers.

### 13.2 Unit

- Suites: loading/validation errors per field; `suite_id` stability and
  change-on-edit; bundled suites satisfy §6.6 minimums and validate
  cleanly (packaging test); haystack determinism and length scaling.
- Scoring: `extract_code`/`extract_call` against the §6 normative rules
  (fenced, bare-object, absent, multiple blocks); every grader type
  pass/fail/detail; conjunctive-vs-partial composition; tooluse
  multi-turn flow with args equality/subset and hallucinated tool;
  agentic effects engine (`set`/`append`/`incr`, `$arg` substitution,
  goal subset, step caps, invalid-call accounting) as pure state tests;
  ifollow constraints incl. JSON schema subset; aggregation math and
  worst-of-reps; `exec_skipped` scoring path.
- Sandbox: retained rlimit assembly, empty-env allowlist, timeout kill and
  group reap (tiny scripts), verdict fields, unavailable-platform error
  path, and shipped entry-point refusal before any runner activity.
- Server: port selection, readiness polling with fake clock, §5.3
  relaunch-once policy, bounded response reads, loopback-literal
  assertion (the client refuses non-127.0.0.1 by construction), stop
  idempotency and finally-path coverage.
- Report: §11.4 structural assertions from a canned `quality.json`,
  including comparison and degradation-warning rendering.
- CLI: validation exits (unknown suite, corpus missing for perplexity,
  conflicting config flags, disabled `--exec` on every platform), option
  assembly, `--list-suites`, `--dry-run` plan content.

### 13.3 Integration (fake server + tiny GGUFs, no GPU, no network)

1. Full run, all HTTP suites, script = all-correct answers → exit 0;
   every suite `score == 1.0`; `quality.json` matches §10; report
   renders; matrix refresh produced `<root>/matrix/` artifacts.
2. Mixed script (some wrong, one malformed reply) → exit 1; wrong
   answers graded 0 with grader details; malformed reply is task
   `error`; aggregation and pass rates correct.
3. `--exec` → exit 2 before model discovery or run creation; without
   `--exec`, `exec_python` graders report `exec_skipped` and
   `exec_enabled: false`.
4. `DIE_AFTER_N` mid-suite → one relaunch, task retried, run completes;
   a second scripted death marks remaining suite tasks
   `server_unavailable`, exit 1.
5. SIGTERM mid-suite → exit 4; `--resume` skips graded tuples, completes,
   final `quality.json` covers all tasks exactly once.
6. `--config-session` pointing at a fake-bench-produced session with a
   lossy winner + `--compare-lossless` with a script that degrades one
   lossy-side answer → comparison block present, negative delta,
   degradation warning set.
7. `--ctx-size` below a needle task's `ctx_min` → task skipped with
   reason, excluded from denominators, listed in report.
8. `--dry-run` launches nothing and creates **no** run directory; it
   prints config provenance, suites, task counts, and the resolved
   server argv, then exits 0.

## 14. Documentation

README gains a "Quality evaluation" section: what each suite measures,
the suite-relative-score caveat, config sources, the lossy-comparison
workflow, the disabled `--exec` policy and inactive retained runner, one
worked example, evidence layout, and how results surface in the Results
Matrix. DESIGN.md is not modified; this document is the Quality Evaluation
spec, and §12's loopback exception is recorded here.

## 15. Future work (post-v1)

Native OpenAI `tools` API when capability-probed (`--jinja` builds);
Night Shift/Marathon integration (a quality pass per confirmed winner in
spare time); multi-model comparison runs driven by the Results Matrix;
sampling-parameter sweeps and temperature-robustness scoring;
container-grade sandboxing (namespaces/jobs) and Windows exec support;
more languages in `coding`; adjudicated free-form grading; concurrent
request throughput-under-load QoS suites; speculative-decoding quality
checks; suite difficulty calibration across model generations.

---


## Appendix: Requirements checklist

Functional:

- FR-1 `llamatune quality MODEL.gguf` exists with the §3 options,
  `--json`, `--list-suites`, and `--dry-run`; validation errors exit 2.
- FR-2 Suite loading validates the §6 schema, derives stable
  `suite_id`s, and rejects malformed user suites with the field named.
- FR-3 Bundled suites meet the §6.6 content minimums and validate
  cleanly under the packaging tests.
- FR-4 Needle haystacks are deterministic, seed-stable, and scale to the
  run context.
- FR-5 Extraction rules (§6.2) and every grader behave per §§6.1–6.4
  with explainable per-grader details.
- FR-6 Task, suite, and overall aggregation follow §7.2, including
  worst-of-reps, `unstable`, skip exclusion, and `exec_enabled`
  visibility.
- FR-7 No model output is ever executed, and `exec_python` graders are
  skipped with recorded reasons.
- FR-8 Passing `--exec` returns exit 2 before model discovery or run
  creation on every platform.
- FR-9 Agentic environments are pure JSON state machines; goal,
  efficiency, and invalid-call grading follow §6.3.
- FR-10 `quality-report.md` renders per §11.4 from `quality.json` alone.
- FR-11 The server runs on literal loopback with executor-grade child
  management; readiness, relaunch-once, and degradation follow §5.
- FR-12 Requests are deterministic per §4.1 and serially executed; all
  request/response bodies are recorded as evidence.
- FR-13 Config resolution (`best`/`defaults`/`session`,
  `--strict-config`) follows §3, records provenance, and warns on stale
  fallback.
- FR-14 `--compare-lossless` produces both result sets, deltas, and the
  degradation warning per §4.4.
- FR-15 The run directory, journal, and `quality.json` match §§9–10;
  `quality.json` satisfies the frozen matrix contract.
- FR-16 Signals conclude gracefully (exit 4) and `--resume` completes
  each task tuple exactly once, identity-checked per §9.3.
- FR-17 Exit codes follow §3.1; partial failures degrade to evidence,
  never hangs.
- FR-18 Completion refreshes the Results Matrix best-effort without ever
  affecting the run's exit code.

Non-functional:

- NFR-1 Strictly serial: at most one child process and one in-flight
  request at any moment.
- NFR-2 Injectable clock for every timeout/deadline decision; the test
  suite never sleeps for scheduling and never runs a real model, GPU, or
  external network endpoint.
- NFR-3 All existing quality gates pass repo-wide: `ruff format --check`,
  `ruff check`, `mypy` (strict), `pytest` with branch coverage ≥ 85%.
- NFR-4 No new runtime dependencies; stdlib HTTP client; Python ≥ 3.11.
- NFR-5 Existing commands' behavior and outputs are unchanged except the
  added CLI command.
- NFR-6 No modifications to `search.py`, `bench.py`, `stats.py`,
  `calibrate.py`, or any engine module; `executor.py` may gain additive
  public helpers only when required by the documented interface.
- NFR-7 The §12 loopback exception is the only relaxation of DESIGN §14
  and is inert unless the operator invokes `quality`. No Quality invocation
  activates a model-output execution exception.
