# Agent guide

Repository-local rules for human and automated contributors.

## Source of truth

`DESIGN.md` is authoritative. Feature-specific design documents under `docs/`
are authoritative only for the features they explicitly own.
When the spec is ambiguous or wrong, implement the most conservative
reading, note the deviation in your handoff, and do not silently invent new
contracts.

## Boundaries

- Change only files required by the current task. Do not reformat or refactor
  unrelated files.
- `bench` builds commands and parses output. It never executes processes.
- `executor` executes. It never interprets benchmark semantics.
- Only `session` writes inside a session directory, with path confinement.
- `cli` stays thin. Heavy imports are lazy inside command bodies.

## Multi-model orchestration

Use the active Hermes profile's model definitions and the
`multi-model-orchestration` skill. Its model-routing and reasoning-effort
references govern selection. This repository assigns work by risk and task
requirements. It does not require every task to use every model.

The following routes match the default Hermes profile checked on 2026-09-14.
Recheck `hermes config get model.aliases` before dispatch. An alias in config
does not prove live provider access.

| Alias or exact selector | Provider / model | Role and starting effort |
| --- | --- | --- |
| `gpt-6-astra-900k` (exact selector, no configured alias) | `openai-codex/gpt-6-astra-900k` | Astra: difficult cross-system reasoning, orchestration, integration, and final review. This repair campaign uses XHigh (`xhigh`) by owner direction. |
| `sol` | `openai-codex/gpt-5.6-sol` | Sol: architecture, independent safety/correctness review, and presentation judgment. Start `medium`, use `high` for consequential work. |
| `terra` | `openai-codex/gpt-5.6-terra` | Terra: backend/API/data implementation and debugging. Start `medium`, use `high` for harder repairs. |
| `luna` | `openai-codex/gpt-5.6-luna` | Luna: bounded implementation, tests, CLI/UI, and documentation. Start `medium`, use `low` for direct edits or `high` for edge cases. |
| `coder` | `opencode-go/kimi-k2.7-code` | Optional coding worker for work without an exact-model gate. |
| `fast` | `opencode-go/deepseek-v4-flash` | Optional extraction and mechanical edits. |
| `long` | `opencode-go/minimax-m3` | Optional large-input synthesis after context preflight. |
| `hard` | `opencode-go/deepseek-v4-pro` | Optional difficult-work comparison or escalation. |

The optional routes do not replace required Codex reviewers automatically.
Use provider-supported effort settings. Do not copy Codex effort names to
other providers without verification.

Context size and reasoning effort are separate choices. Hermes strips the
supported `-900k` suffix before sending the base model ID to Codex. Thus
`gpt-6-astra-900k` selects the Astra context variant, not an independent
model. Check the effective resolver for configured context caps. Do not
claim a measured 900k allowance from the name alone. Outside the explicit
campaign choice, start Astra at `high` for difficult engineering and use
larger context or effort only when the work needs it.

### How to dispatch

- The campaign coordinator uses Astra 900k XHigh. Give each task a fresh
  implementer, then an independent specification and code-quality review.
  Use a separate Astra session for final review. Serialize writers to shared
  files and preserve the execution ledger.
- Start named-model workers with explicit provider, model, and effort:
  ```bash
  hermes --provider openai-codex -m gpt-6-astra-900k --reasoning xhigh -w
  hermes --provider openai-codex -m gpt-5.6-sol --reasoning high -w
  hermes --provider openai-codex -m gpt-5.6-terra --reasoning high -w
  hermes --provider openai-codex -m gpt-5.6-luna --reasoning medium -w
  ```
- For headless workers, bind both process cwd and `TERMINAL_CWD` to the
  isolated worktree. Remove inherited parent-session and desktop context.
  Bound runtime and output. Pass required uncommitted plans explicitly.
- Give each worker a self-contained task brief with its file ownership,
  interfaces, acceptance checks, and report path. Relay contracts explicitly.
- `delegate_task` uses the profile-wide `delegation.model` override when
  set, otherwise model inheritance. It has no per-child model selector.
  Use independent processes or explicit dispatcher routes for named models.
  Its reasoning setting can also inherit. Never infer XHigh from a model name.
- For one-shot gates, use `--usage-file` and verify provider, base model,
  completion state, and distinct session identity. Record the requested
  selector and effort separately. Reject fallback as named-model evidence.
- Record role changes and their rationale in the handoff. Preserve any
  frozen exact-model gate unless the owner explicitly changes it.
- Do not change Hermes-wide defaults, aliases, profiles, auxiliary models,
  or fallback configuration as a side effect of repository execution.

## Safety invariants

- Never `shell=True`. Use argv lists and allowlisted environments only.
- Bound every child's runtime and captured output.
- Model files and benchmark output are data, never executed.
- Never mutate system state (governors, clocks, caches, drivers).
- No network access in runtime code paths.
- No credentials or environment values in evidence. Names only.

## Verification baseline

```
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

## Token discipline

The specs are complete. Do not explore beyond this repository's documents
and your owned files. Write files once, deliberately. Focused tests during
development. Full gates once at the end.