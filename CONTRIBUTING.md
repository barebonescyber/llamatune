# Contributing to llamatune

Thank you for helping improve llamatune. Read [AGENTS.md](AGENTS.md) before
changing code; it defines file ownership, architectural boundaries, safety
invariants, and the authoritative relationship with [DESIGN.md](DESIGN.md).

Use a focused branch, keep behavior changes documented, and do not commit model
files, session artifacts, credentials, or workstation-specific notes. Put local
notes in `NOTES.local.md`, which is intentionally ignored.

Every commit must include a `Signed-off-by: Name <email>` trailer certifying
the contribution under the
[Developer Certificate of Origin 1.1](https://developercertificate.org/).
Create the trailer with `git commit -s`; never sign off on another contributor's
behalf.

Run all four quality gates before submitting a change:

```sh
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

The GitHub Actions matrix runs these gates on required Linux and Windows lanes.
The macOS lane is advisory until the v1.0 release-candidate compatibility gate.
