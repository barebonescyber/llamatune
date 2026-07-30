# Contributing to llamatune

Thank you for helping improve llamatune. Read [AGENTS.md](AGENTS.md) before
changing code; it defines file ownership, architectural boundaries, safety
invariants, and the authoritative relationship with [DESIGN.md](DESIGN.md).

Use a focused branch, keep behavior changes documented, and do not commit model
files, session artifacts, credentials, or workstation-specific notes. Put local
notes in `NOTES.local.md`, which is intentionally ignored.

Run all four quality gates before submitting a change:

```sh
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

The GitHub Actions matrix runs these gates on required Linux and Windows lanes.
The macOS lane is advisory until the v1.0 release-candidate compatibility gate.
