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
- `bench` builds commands and parses output; it never executes processes.
- `executor` executes; it never interprets benchmark semantics.
- Only `session` writes inside a session directory, with path confinement.
- `cli` stays thin; heavy imports are lazy inside command bodies.

## Safety invariants

- Never `shell=True`; argv lists and allowlisted environments only.
- Bound every child's runtime and captured output.
- Model files and benchmark output are data, never executed.
- Never mutate system state (governors, clocks, caches, drivers).
- No network access in runtime code paths.
- No credentials or environment values in evidence; names only.

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
development; full gates once at the end.
