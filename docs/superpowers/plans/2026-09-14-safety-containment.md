# Safety Containment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop unsafe generated-code execution and add tested dependency floors and PR security checks without a sandbox or executor redesign.

**Architecture:** Put a fail-closed guard at the sandbox entry point and Quality resolution boundary, with a thin CLI diagnostic. Retain existing resource-control code for a later confinement design. Change only dependency metadata and the existing audit trigger for supply-chain prevention.

**Tech Stack:** Python >=3.11, uv, pytest, Typer, GitHub Actions. Local type checking uses Python 3.12.

## Global Constraints

- Base: public commit `12485e742b12fe9ca0b9d296e04de427f077b09e`, not the private working branch or PR #33.
- Follow `DESIGN.md`, `docs/quality-eval-design.md`, `AGENTS.md`, and the campaign's [global constraints](2026-09-14-safety-correctness-triage.md).
- Never `shell=True`. Use argv lists and allowlisted environments only.
- Bound every child's runtime and captured output.
- Model files and benchmark output are data, never executed.
- Never mutate system state (governors, clocks, caches, drivers).
- No credentials or environment values in evidence. Names only.
- No new runtime dependencies. Preserve `requires-python = ">=3.11"` and `rich>=15.0.0,<16`.
- No unsafe override, global network fallback, native sandbox trial, GPU run, publication, or workflow approval.
- Do not close #5/#28 as secure execution support. This plan delivers containment only.
- No em dash characters in generated prose or code.

---

## File responsibilities

| File | Responsibility |
| --- | --- |
| `src/llamatune/sandbox.py` | Reject execution before temporary files, locks, or subprocesses |
| `src/llamatune/quality.py` | Enforce the same policy for programmatic run and persisted resume options |
| `src/llamatune/cli.py` | Show an actionable usage error without creating a run |
| `tests/unit/test_exec_containment.py` | New public-entry containment tests, without legacy bypass fixtures |
| `tests/unit/test_sandbox.py` | Existing benign resource-control tests behind an explicit test-only guard bypass |
| `tests/unit/test_cli_quality.py` | CLI error and no-run-directory assertions |
| `tests/integration/test_quality.py` | Real persisted options and run/resume refusal |
| `docs/quality-eval-design.md`, `SECURITY.md` | Explicit temporary execution restriction and re-enablement criteria |
| `pyproject.toml`, `uv.lock` | Tested dependency support floors, no unrelated upgrades |
| `tests/unit/test_dependency_policy.py` | Conservative dependency-floor contract |
| `.github/workflows/security.yml` | Existing bounded audit on PR events with read-only permissions |
| `tests/unit/test_security_workflow.py` | Repository-local workflow contract assertions |

## Task S1: Contain generated-code execution (#5/#28)

**Files:**
- Modify: `src/llamatune/sandbox.py:183-201`, add the guard next to `ExecVerdict`.
- Modify: `src/llamatune/quality.py:455-459`.
- Modify: `src/llamatune/cli.py:779-781` and the `--exec` help string near line 692.
- Create: `tests/unit/test_exec_containment.py`.
- Modify: `tests/unit/test_cli_quality.py:206-216`.
- Modify: `tests/unit/test_sandbox.py:20-22`.
- Modify: `tests/integration/test_quality.py:284-364`, append new containment tests.
- Modify: `docs/quality-eval-design.md:134-136,437-460`, `SECURITY.md`.

**Interfaces:**
- Consumes: existing `run_python(code: str, *, timeout_s: float) -> ExecVerdict`, `run_quality(options: QualityOptions, *, now_fn: Callable[[], Any] | None = None) -> QualityOutcome`, and `resume_quality(run_dir: Path, *, llama_bin: Path | None = None, now_fn: Callable[[], Any] | None = None) -> QualityOutcome`.
- Produces: `sandbox.EXEC_DISABLED_REASON: str` and `sandbox.require_exec_isolation() -> None`, which raises `ValueError` until a separately reviewed confinement implementation replaces it.
- CLI and Quality outcomes use exit 2. `run_python` raises `ValueError`. No isolation field falsely claims enforcement.
- Decision: even `--dry-run --exec` refuses. Ordinary non-executing quality evaluation remains available.

- [ ] **Step 1: Write entry-point tests.** Create this complete new test file. The test supplies only a harmless literal and forbids any spawn.

```python
from pathlib import Path

import pytest

from llamatune import sandbox


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_run_python_refuses_before_spawn(
    platform: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sandbox.sys, "platform", platform)
    monkeypatch.setattr(
        sandbox.subprocess, "Popen", lambda *a, **k: pytest.fail("child started")
    )
    monkeypatch.setattr(
        sandbox.tempfile, "mkdtemp", lambda *a, **k: pytest.fail("scratch created")
    )
    with pytest.raises(ValueError, match="generated-code execution is disabled"):
        sandbox.run_python("pass", timeout_s=1.0)
    assert list(tmp_path.iterdir()) == []
```

In `tests/unit/test_cli_quality.py`, replace the platform-validation test with:

```python
def test_quality_exec_is_rejected_before_run_creation(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["quality", str(tmp_path / "model.gguf"), "--sessions-dir", str(tmp_path), "--exec"],
    )
    assert result.exit_code == 2
    assert "generated-code execution is disabled" in result.stderr
    assert not (tmp_path / "quality").exists()
```

- [ ] **Step 2: Run RED.**

```bash
env -u PYTHONPATH -u VIRTUAL_ENV uv run --locked --python 3.12 pytest --no-cov tests/unit/test_exec_containment.py tests/unit/test_cli_quality.py -k 'refuses_before_spawn or rejected_before_run_creation' -v
```

Expected: the current entry point attempts scratch creation, and the CLI lacks the containment message. A missing import or fixture is not the expected failure.

- [ ] **Step 3: Add the guard and its three call sites.** In `sandbox.py` define:

```python
EXEC_DISABLED_REASON = (
    "generated-code execution is disabled until filesystem, network, and memory "
    "confinement are verified; omit --exec to use non-executing graders"
)


def require_exec_isolation() -> None:
    raise ValueError(EXEC_DISABLED_REASON)
```

Make `require_exec_isolation()` the first statement after the `run_python` docstring. Leave its signature and existing bounded implementation intact. Do not add a production bypass.

At the beginning of `quality._resolve`, before `assess_hardware()`:

```python
if options.exec_enabled:
    from llamatune.sandbox import require_exec_isolation

    require_exec_isolation()
```

Both `run_quality` and `resume_quality` already translate this `ValueError` into exit 2. Resume passes persisted `exec_enabled` through `_resolve`, sharing the same policy.

Replace the CLI's `if exec_enabled and not _quality_exec_supported()` block with:

```python
if exec_enabled:
    from llamatune.sandbox import EXEC_DISABLED_REASON

    typer.echo(f"error: {EXEC_DISABLED_REASON}", err=True)
    raise typer.Exit(code=2)
```

Use this option help text: `Reserved: generated-code execution is disabled pending verified confinement.` Retain the existing private POSIX helper for now rather than bundling dead-code cleanup.

- [ ] **Step 4: Add disk-backed Quality run/resume coverage.** Append to `tests/integration/test_quality.py`, using its existing `_options` helper, `quality`, `sandbox`, `json`, `Path`, and `pytest` imports:

```python
@pytest.mark.parametrize("dry_run", [False, True])
def test_exec_run_refuses_before_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    options = _options(
        tmp_path / "unused.gguf", tmp_path / "bin", tmp_path / "sessions",
        ("coding",), exec_enabled=True, dry_run=dry_run,
    )
    monkeypatch.setattr(
        quality, "assess_hardware", lambda: pytest.fail("hardware discovery started")
    )
    result = quality.run_quality(options)
    assert result.exit_code == 2
    assert result.summary["error"] == sandbox.EXEC_DISABLED_REASON
    assert not options.sessions_dir.exists()


def test_exec_resume_refuses_persisted_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = _options(
        tmp_path / "unused.gguf", tmp_path / "bin", tmp_path / "sessions",
        ("coding",), exec_enabled=True,
    )
    run_dir = tmp_path / "previous-run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps({"schema_version": 1, "options": quality._options_dict(options)}),
        encoding="utf-8",
    )
    (run_dir / "journal.jsonl").write_bytes(b'{"type":"quality_start"}\n')
    before = {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()}
    monkeypatch.setattr(
        quality, "assess_hardware", lambda: pytest.fail("hardware discovery started")
    )
    result = quality.resume_quality(run_dir)
    assert result.exit_code == 2
    assert result.summary["error"] == sandbox.EXEC_DISABLED_REASON
    assert before == {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()}
```

For the legacy sandbox mechanics tests only, add the following line to their existing `_disable_unshare` autouse fixture:

```python
monkeypatch.setattr(sandbox, "require_exec_isolation", lambda: None)
```

Add that same explicit test-only patch inside `test_exec_and_no_exec_paths_never_overlap_server_and_sandbox`. That test runs only the repository's synthetic fixture responses. Do not put this patch in global `conftest.py` or in the new containment tests. This preserves cleanup/resource-control regression coverage while proving that shipped entry points cannot reach it.

- [ ] **Step 5: Update the policy documentation.** Replace the operative `--exec` description and the first paragraph of Quality design section 7.1 with this text. Keep the remaining mechanics description labeled as inactive implementation detail.

```text
Generated-code execution is temporarily disabled on every platform.
Passing --exec returns exit 2 before model discovery or run creation.
Resuming a run whose saved options enable execution also returns exit 2.
Quality evaluation without --exec remains available and skips exec_python graders.
The retained resource-limit runner is not a security boundary for untrusted code.
Re-enablement requires mandatory filesystem and network confinement, verified memory
limits, and adversarial tests. No reduced-isolation override is supported.
```

Add the same restriction to `SECURITY.md`. Identify this as mitigation for #5/#28. Do not claim this patch implements macOS memory limits or filesystem confinement.

- [ ] **Step 6: Run GREEN and the affected suite.**

```bash
env -u PYTHONPATH -u VIRTUAL_ENV uv run --locked --python 3.12 pytest --no-cov tests/unit/test_exec_containment.py tests/unit/test_sandbox.py tests/unit/test_cli_quality.py tests/integration/test_quality.py -v
```

Expected: containment cases pass without a child or run-directory creation. Non-exec Quality tests still pass. Test-only resource mechanics remain covered. Do not weaken assertions or skip platforms to hide regressions.

- [ ] **Step 7: Review and commit the bounded patch after execution authorization.** Run focused patch checks here and full gates at final integration, then:

```bash
git diff --check
git add src/llamatune/sandbox.py src/llamatune/quality.py src/llamatune/cli.py tests/unit/test_exec_containment.py tests/unit/test_sandbox.py tests/unit/test_cli_quality.py tests/integration/test_quality.py docs/quality-eval-design.md SECURITY.md
git commit -m "fix: contain generated-code execution pending confinement"
```

## Task S2: Test and declare conservative runtime floors (#16)

**Files:**
- Modify: `pyproject.toml:31-35`, `uv.lock` project metadata.
- Create: `tests/unit/test_dependency_policy.py`.

**Interfaces:**
- Consumes: locked versions Typer `0.27.0`, GGUF `0.19.0`, Rich `15.0.0`.
- Produces: install metadata `typer>=0.27.0,<1`, `gguf>=0.19.0,<1`, unchanged `rich>=15.0.0,<16`.
- This declares a conservative supported minimum, not the earliest version that might work. No unrelated package upgrade.

- [ ] **Step 1: Write the failing metadata test.**

```python
from pathlib import Path
import tomllib


def test_runtime_dependency_floors_match_supported_baseline() -> None:
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["requires-python"] == ">=3.11"
    assert set(project["project"]["dependencies"]) == {
        "typer>=0.27.0,<1", "gguf>=0.19.0,<1", "rich>=15.0.0,<16",
    }
```

- [ ] **Step 2: Run RED.**

```bash
env -u PYTHONPATH -u VIRTUAL_ENV uv run --locked --python 3.12 pytest --no-cov tests/unit/test_dependency_policy.py -v
```

Expected: dependency-set assertion fails because two lower bounds are absent.

- [ ] **Step 3: Edit the two dependency strings and refresh lock metadata.**

```toml
dependencies = [
    "typer>=0.27.0,<1",
    "gguf>=0.19.0,<1",
    "rich>=15.0.0,<16",
]
```

```bash
env -u PYTHONPATH -u VIRTUAL_ENV uv lock
git diff -- pyproject.toml uv.lock
```

Reject unrelated lock-version churn. Do not update Hatchling here. That belongs to PR #3.

- [ ] **Step 4: Exercise the declared floors in an isolated environment.** This is a development install, not runtime networking or package publication.

```bash
MIN_ENV=$(mktemp -d)
env -u PYTHONPATH -u VIRTUAL_ENV uv venv --python 3.11 "$MIN_ENV"
env -u PYTHONPATH -u VIRTUAL_ENV uv pip install --python "$MIN_ENV/bin/python" . 'typer==0.27.0' 'gguf==0.19.0' 'rich==15.0.0' pytest
env -u PYTHONPATH -u VIRTUAL_ENV "$MIN_ENV/bin/python" -m llamatune --help
env -u PYTHONPATH -u VIRTUAL_ENV "$MIN_ENV/bin/python" -m pytest -o addopts= tests/unit/test_cli_quality.py tests/unit/test_model.py -q
```

Expected: CLI imports, GGUF fixtures parse, and the selected suite passes on Python 3.11. If installation fails at the declared floor, record the resolver output. Revise the floor from evidence rather than claiming compatibility.

- [ ] **Step 5: Run GREEN and commit after focused patch checks.**

```bash
env -u PYTHONPATH -u VIRTUAL_ENV uv run --locked --python 3.12 pytest --no-cov tests/unit/test_dependency_policy.py -v
git add pyproject.toml uv.lock tests/unit/test_dependency_policy.py
git commit -m "fix: declare tested runtime dependency floors"
```

## Task S3: Run the existing security audit for PRs (#16)

**Files:**
- Modify: `.github/workflows/security.yml:3-17,24-25,49-51`.
- Create: `tests/unit/test_security_workflow.py`.

**Interfaces:**
- Consumes: existing `local-audit` job and pinned scanners. Its visible check name remains `Dependency, source, and secret audit`.
- Produces: `pull_request` trigger plus the existing `workflow_dispatch`, with top-level `contents: read`. PR runs have no secret, publish, or OIDC authority.
- Keep CodeQL manual-only with `github.event_name == 'workflow_dispatch' && inputs.run_codeql` until review resolves its separate permissions/platform contract.
- Required-check rules are remote policy. Do not edit them as a hidden part of this patch.

- [ ] **Step 1: Write the regression contract.** The standard library suffices. These assertions pin the intended small textual change and do not replace GitHub's workflow parser.

```python
from pathlib import Path


def test_security_audit_has_unprivileged_pr_trigger() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / ".github/workflows/security.yml").read_text(encoding="utf-8")
    assert "on:\n  pull_request:\n  workflow_dispatch:" in text
    assert "permissions:\n  contents: read\n" in text
    assert "pull_request_target" not in text
    assert "id-token: write" not in text
    assert "persist-credentials: false" in text
    assert "timeout-minutes: 15" in text
    assert "enable-cache: false" in text
    assert "github.event_name == 'workflow_dispatch' && inputs.run_codeql" in text
```

- [ ] **Step 2: Run RED.**

```bash
env -u PYTHONPATH -u VIRTUAL_ENV uv run --locked --python 3.12 pytest --no-cov tests/unit/test_security_workflow.py -v
```

Expected: the PR trigger assertion fails.

- [ ] **Step 3: Apply only the trigger, timeout, and cache changes.** Add `pull_request:` before `workflow_dispatch:`. Keep the existing dispatch inputs intact. Add `timeout-minutes: 15` to `local-audit`. Disable setup-uv cache writes for this audit. Restrict CodeQL explicitly:

```yaml
on:
  pull_request:
  workflow_dispatch:
    inputs:
      run_codeql:
        description: "Run CodeQL (requires GitHub Code Security for this private repository)"
        required: false
        default: false
        type: boolean
```

```yaml
timeout-minutes: 15
```

```yaml
enable-cache: false
```

```yaml
if: ${{ github.event_name == 'workflow_dispatch' && inputs.run_codeql }}
```

Retain every existing full-SHA action pin unless PR #34 has already changed that same pin in the approved integration base. Do not mix another action upgrade into this patch.

- [ ] **Step 4: Run GREEN and inspect workflow authority.**

```bash
env -u PYTHONPATH -u VIRTUAL_ENV uv run --locked --python 3.12 pytest --no-cov tests/unit/test_security_workflow.py -v
git diff -- .github/workflows/security.yml
```

Check that the checkout uses the event commit without credential persistence. Confirm the PR job has no secrets or write permissions. CodeQL must not run on PR events. After publication approval, require an actual successful exact-head audit run. A local string test does not prove hosted execution.

- [ ] **Step 5: Commit after patch gates and submit for review.**

```bash
git add .github/workflows/security.yml tests/unit/test_security_workflow.py
git commit -m "ci: run the read-only security audit on pull requests"
```

S2 and S3 together satisfy the code-side #16 scope only after the declared floor test and hosted audit pass. Enforcing the named check in repository rules remains a separate maintainer action.
