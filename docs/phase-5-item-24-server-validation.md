# Phase 5 Item 24 — OpenCode server validation

Status date: 2026-07-27.

Status: **complete with an unsupported-capability verdict for
`--cache-reuse` on the pinned Qwen3.6/b10107 combination**.

## Scope

Item 24 tested the 32,768-token Item 22 winner through a sanitized, read-only,
25-turn OpenCode session. The intended comparison changed only llama-server's
`--cache-reuse` value from `0` to `256`.

The production-shaped server configuration was:

```text
Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
ngl=40, n-cpu-moe=20
batch=4096, ubatch=1024, threads=10
flash attention on, f16 K/V, KV offload on
no mmap, context=32768, slots=1
jinja, reasoning=auto, cache-prompt
seed=424242
```

OpenCode 1.17.18 used one bounded read-only agent, a 2,048-token output limit,
automatic compaction, a five-step ceiling, and a 900-second per-turn timeout.
Whole-file reads, edits, network access, delegation, and unrestricted shell
commands were denied.

## Pinned identity

- llama-server: b10107 (`c0bc8591e`)
- llama-server SHA-256:
  `ccb7e4080a5dc7ee63fc24035936a8025017c0ab81c12f3d7e22786e09a37a31`
- model SHA-256:
  `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`
- model architecture recorded by Item 22: `qwen35moe`
- source commit under review: `413bc3d`

## Captures

### Requested `--cache-reuse 0`

- Sanitized durable evidence:
  `models/phase5-item24-server-validation/cache0`
- Raw temporary evidence: `/tmp/item24-overnight-20260727-133418`
- OpenCode session: `ses_05c37385cffeGONsRcCB2I8eSs`
- Result: 25 turns completed; sanitized export valid; no OpenCode error
  events; no non-empty stderr logs; capture worktree clean.
- Step ceiling reached: 13 turns.

### Requested `--cache-reuse 256`

- Sanitized durable evidence:
  `models/phase5-item24-server-validation/cache256-effective0`
- Raw temporary evidence:
  `/tmp/item24-overnight-cache256-20260727-140632`
- OpenCode session: `ses_05c19b5e9ffe0gQ8BZWBZ4nKqH`
- Result: 25 turns completed; sanitized export valid; capture worktree
  clean.
- OpenCode recovered from context overflow after compaction on turns 6 and
  19. Both recoveries are retained in
  `RECOVERED_CONTEXT_OVERFLOWS`; no other OpenCode errors occurred.
- Step ceiling reached: 12 turns.
- Effective cache-reuse value: **0**, because llama-server disabled the
  requested value during model initialization.

An earlier incomplete requested-256 capture remains at
`/tmp/item24-overnight-cache256-20260727-135850`. It stopped at turn 7 after
OpenCode recovered from a context overflow but retained exit status 1. It is
diagnostic evidence, not a completed comparison run.

## Capability verdict

The requested-256 server emitted this warning before accepting traffic:

```text
cache_reuse is not supported by this context, it will be disabled
```

The exact b10107 source explains the result:

1. `tools/server/server-context.cpp` sets `n_cache_reuse` to zero when
   `llama_memory_can_shift()` is false.
2. `src/llama-kv-cache.cpp` returns false from `get_can_shift()` when
   `hparams.n_pos_per_embd() > 1`.
3. `src/llama-hparams.cpp` returns four positions per embedding for MRoPE or
   IMRoPE.
4. `src/llama-model.cpp` selects IMRoPE for `LLM_ARCH_QWEN35` and
   `LLM_ARCH_QWEN35MOE`.

`--cache-reuse` reuses matching non-prefix chunks by shifting their KV entries.
That operation is unavailable for this model's four-position IMRoPE cache in
the pinned build. `--swa-full`, slot-count, batch, KV type, and prompt-cache
settings do not remove the four-position constraint. Enabling the option would
require a future llama.cpp implementation that safely shifts this cache; it is
not a runtime tuning choice.

Ordinary prompt caching remains active. Both captures show llama-server
selecting the slot by longest-common-prefix similarity and retaining the
common prefix. The unsupported verdict applies specifically to the additional
non-prefix chunk-shifting behavior controlled by `--cache-reuse`.

## Timing observations

These numbers are retained as repeatability observations only. They are **not**
an on/off cache-reuse comparison because both servers ran with effective
cache-reuse zero and model/tool behavior diverged between sessions.

Turn wall time is measured from creation of each redirected OpenCode JSONL
before process launch through its final write:

| Requested value | Effective value | Turn 1 | Turn 10 | Turn 25 |
|---|---:|---:|---:|---:|
| 0 | 0 | 24.382 s | 23.124 s | 27.188 s |
| 256 | 0 | 25.698 s | 8.387 s | 25.579 s |

Across all 25 turns:

| Requested value | Total | Mean | Median | P90 | Tool calls |
|---|---:|---:|---:|---:|---:|
| 0 | 742.681 s | 29.707 s | 24.382 s | 60.558 s | 183 |
| 256 (disabled) | 737.974 s | 29.519 s | 25.097 s | 61.726 s | 113 |

The raw total differed by only 0.6%, while the second session issued 38% fewer
tool calls. Turn 10 therefore cannot support a latency claim despite its lower
wall time. Turn 1 and turn 25 were similar, as expected from two effective-zero
runs.

Server counters reinforce that the workloads diverged:

| Requested value | Completed requests | Prompt tokens | Prompt time | Eval tokens | Eval time |
|---|---:|---:|---:|---:|---:|
| 0 | 98 | 361,961 | 203.986 s | 42,422 | 507.912 s |
| 256 (disabled) | 95 | 426,622 | 235.068 s | 39,702 | 471.906 s |

## Recommendation

For Qwen3.6-35B-A3B on pinned llama.cpp b10107:

- keep `--cache-prompt`;
- omit `--cache-reuse 256`, because the server disables it;
- retain one slot per endpoint for the validated 32K envelope;
- do not claim a cache-reuse latency improvement from these captures;
- re-open the experiment only after a pinned llama.cpp build reports that
  cache reuse remains enabled for `qwen35moe`/IMRoPE and startup evidence
  contains no disable warning.

The startup warning is the authoritative capability gate. A future replay
runner should fail before TURN 01 if a nonzero requested cache-reuse value is
disabled during model initialization.

## Ownership deviation

The original implementation ownership did not define a Phase 5
server-validation owner. Item 24 owns this campaign record and the historical
analysis correction made during validation. No runtime code, tests, workflows,
dependencies, benchmark parser, executor, or session evidence is changed.
