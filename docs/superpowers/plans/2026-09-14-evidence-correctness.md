# Evidence and Resume Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair ignored Marathon records, recommendation drift, confirmation replay, MoE patience, and Night Shift stop outcomes in independently reviewable patches.

**Architecture:** Keep existing readers, search stages, and writers. Share only runtime flag emission between its two consumers. Persist confirmation identity and allocate distinct capture directories when a new measurement is necessary.

**Tech Stack:** Python >=3.11, pytest fixtures, uv, Ruff, mypy. Local verification uses Python 3.12.

## Global Constraints

- Base: public commit `12485e742b12fe9ca0b9d296e04de427f077b09e`.
- Follow `DESIGN.md`, feature-specific design documents, `AGENTS.md`, and the [campaign constraints](2026-09-14-safety-correctness-triage.md).
- `bench` builds and parses. `executor` executes. Only the confined Session writer creates session artifacts.
- Never `shell=True`. Bound every child's runtime and captured output.
- No native models, GPU probes, runtime network additions, or system-state mutation.
- Preserve the GPU-layer hard cap, confirmation thresholds, raw evidence, and budget-counting rules.
- No search-engine split, universal journal reader, cache redesign, executor API change, or new dependency.
- No em dash characters. No issue closure for acceptance work this plan deliberately excludes.
- Source line numbers refer to the pinned public snapshot. The referenced test file defines each existing helper used here.

---

## File map and order

| Task | Production files | Boundary |
| --- | --- | --- |
| E1 | `marathon.py`, `docs/marathon-design.md`, `DESIGN.md` | Read-only, warned Marathon journal recovery |
| E2 | new `runtimeflags.py`, `recommend.py`, `report.py` | Pure flag emission and reference rendering |
| E3 | `search.py`, `session.py`, `DESIGN.md` | Identity-bound confirmation reuse and non-overwriting captures |
| E4 | `search.py` | Hill-climb cache semantics only |
| E5 | `nightshift.py`, `DESIGN.md`, `docs/nightshift-design.md` | Explicit interruption state and exit reporting |

Execute E3 before E4 because both own `search.py` and `tests/unit/test_search.py`. Use sequential implementation workers for E1-E5. Astra 900k XHigh owns final integration under `AGENTS.md`. Keep the later PR #42 lane out of these patches.

## Task E1: Preserve later Marathon journal records (#8)

**Files:**
- Modify: `src/llamatune/marathon.py:275-287`.
- Modify: `tests/unit/test_marathon_plan.py`.
- Modify: `docs/marathon-design.md`, `DESIGN.md:481-515`.

**Interfaces:**
- Consumes: `journal.jsonl` with one JSON object per nonblank line.
- Produces: unchanged `_entries(run_dir: Path) -> list[dict[str, Any]]` signature.
- Recovery policy: skip invalid JSON and non-object records with a line-numbered `RuntimeWarning`. Ignore blank lines and retain later valid objects. Never write the journal from this reader.
- Preserve strict `Session.load` corruption behavior. This repairs Marathon, not every reader.

- [ ] **Step 1: Add failing tests to `tests/unit/test_marathon_plan.py`.** Add `import json` beside existing imports.

```python
@pytest.mark.parametrize("bad", ["{broken", "[]", '"not an object"'])
def test_marathon_middle_corruption_keeps_later_records(
    tmp_path: Path, bad: str
) -> None:
    path = tmp_path / "journal.jsonl"
    original = ('{"type":"marathon_start"}\n' + bad + '\n{"type":"marathon_end"}\n').encode()
    path.write_bytes(original)
    with pytest.warns(RuntimeWarning, match=r"journal.jsonl line 2"):
        entries = marathon_module._entries(tmp_path)
    assert [entry["type"] for entry in entries] == ["marathon_start", "marathon_end"]
    assert path.read_bytes() == original


def test_completed_marathon_is_not_resumed_after_corruption(tmp_path: Path) -> None:
    opts = options()
    run_dir = tmp_path / "marathon" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"identity": identity(opts, "fp")}), encoding="utf-8"
    )
    (run_dir / "journal.jsonl").write_bytes(
        b'{"type":"marathon_start"}\n{broken\n{"type":"marathon_end"}\n'
    )
    with pytest.warns(RuntimeWarning, match="line 2"):
        candidate, mismatches = marathon_module.find_reentry(tmp_path, opts, "fp")
    assert candidate is None
    assert mismatches == ()


def test_marathon_blank_lines_and_torn_tail_do_not_change_bytes(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    original = b'\n{"type":"marathon_start"}\n\n{torn'
    path.write_bytes(original)
    with pytest.warns(RuntimeWarning, match="line 4"):
        assert marathon_module._entries(tmp_path) == [{"type": "marathon_start"}]
    assert path.read_bytes() == original
```

- [ ] **Step 2: Run RED.**

```bash
env -u PYTHONPATH -u VIRTUAL_ENV uv run --locked --python 3.12 pytest --no-cov tests/unit/test_marathon_plan.py -k 'corruption or torn_tail' -v
```

Expected: the current reader loses the final record and emits no warning. Non-object records also lack the required warning.

- [ ] **Step 3: Replace only `_entries` and add `import warnings`.**

```python
def _entries(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "journal.jsonl"
    if not path.is_file():
        return []
    result: list[dict[str, Any]] = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            warnings.warn(
                f"journal.jsonl line {number}: invalid JSON ignored", RuntimeWarning, stacklevel=2
            )
            continue
        if not isinstance(value, dict):
            warnings.warn(
                f"journal.jsonl line {number}: non-object record ignored",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        result.append(value)
    return result
```

Warnings identify location without echoing attacker-controlled record contents. Do not add a writer or share this policy with strict Session resume in this patch.

- [ ] **Step 4: Document the reader-specific contract.** Add this text to Marathon resume documentation and a short cross-reference in DESIGN section 12:

```text
Marathon recovery skips malformed or non-object journal records with a line-numbered
warning and retains later valid records. Blank lines are ignored. The reader does
not alter journal bytes. A later valid marathon_end prevents automatic re-entry.
This policy does not change the strict tuning-session resume reader or its warned
torn-final-line recovery.
```

- [ ] **Step 5: Run GREEN, review, and commit after focused patch checks.**

```bash
env -u PYTHONPATH -u VIRTUAL_ENV uv run --locked --python 3.12 pytest --no-cov tests/unit/test_marathon_plan.py tests/integration/test_marathon.py tests/unit/test_session.py -v
git diff --check
git add src/llamatune/marathon.py tests/unit/test_marathon_plan.py docs/marathon-design.md DESIGN.md
git commit -m "fix: retain later Marathon journal records with warnings"
```

#8 remains partially open if the maintainer retains its broader shared-reader acceptance criterion. Remove blanket closure language from any replacement PR description.

## Task E2: Give recommendation and export identical flags (#13/#24)

**Files:**
- Create: `src/llamatune/runtimeflags.py`.
- Modify: `src/llamatune/recommend.py:197-220,235-259` and alternate command rendering later in that function.
- Modify: `src/llamatune/report.py:37-85`.
- Modify: `tests/unit/test_recommend.py`.
- Test: `tests/unit/test_report.py`.

**Interfaces:**
- Consumes: existing frozen `TrialConfig` fields.
- Produces: `runtime_flags(config: TrialConfig, *, moe: bool = True) -> list[str]`.
- `ot_spec` wins over `--n-cpu-moe`. `moe=False` suppresses only `--n-cpu-moe`.
- `None` means omit optional threads-batch and placement flags. Do not change `TrialConfig.bench_args` or benchmark identity.

- [ ] **Step 1: Add the end-to-end parity test to `test_recommend.py`.** Add imports `dataclasses`, `shlex`, `pytest`, and `from llamatune import report`.

```python
@pytest.mark.parametrize("model_path", ["/models/plain.gguf", "/models/model one's.gguf"])
def test_recommended_commands_preserve_runtime_placement(model_path: str) -> None:
    cfg = _config(
        threads_batch=4, ot_spec="blk.0.ffn_.*_exps=CPU", moe_cpu_layers=3,
        tensor_split=(2.0, 1.0), split_mode="layer", mmap=False,
        no_kv_offload=True, cache_type_k="q8_0", cache_type_v="q4_0",
    )
    model = dataclasses.replace(_model(), path=Path(model_path))
    snippet = recommend.build_recommended_sh(
        config=cfg, model=model, expected={}, confirmed=True,
        target="balanced", moe=True, ctx_size=4096,
    )
    exported = report.render_export(
        {"config": cfg.to_dict(), "model": {"path": str(model.path)}},
        {"options": {"ctx_size": 4096}}, "llama-cli",
    )
    line = next(line[2:] for line in snippet.splitlines() if line.startswith("# llama-cli -m "))
    argv = shlex.split(line)
    assert argv == shlex.split(exported)
    for flag, value in (("-tb", "4"), ("-ot", cfg.ot_spec), ("-ts", "2,1"), ("-sm", "layer")):
        assert argv.count(flag) == 1
        assert argv[argv.index(flag) + 1] == value
    assert "--n-cpu-moe" not in argv


def test_runtime_flags_default_omission_and_moe_gate() -> None:
    from llamatune.runtimeflags import runtime_flags

    cfg = _config(moe_cpu_layers=3)
    assert "--n-cpu-moe" in runtime_flags(cfg, moe=True)
    assert "--n-cpu-moe" not in runtime_flags(cfg, moe=False)
    for flag in ("-tb", "-ot", "-ts", "-sm"):
        assert flag not in runtime_flags(cfg)


def test_config_text_has_no_duplicate_optional_fields() -> None:
    text = report._format_config(_config(threads_batch=4, ot_spec="x=CPU").to_dict())
    assert text.count("tb=") == 1
    assert text.count("ot=") == 1
```

- [ ] **Step 2: Run RED.**

```bash
env -u PYTHONPATH -u VIRTUAL_ENV uv run --locked --python 3.12 pytest --no-cov tests/unit/test_recommend.py -k 'runtime_placement or runtime_flags_default or duplicate_optional' -v
```

Expected: recommendation/export parity and duplicate-token checks fail. The helper test fails because the planned module does not exist yet.

- [ ] **Step 3: Create the pure emitter.**

```python
"""Shared runtime flags for recommendation reference commands and exports."""
from __future__ import annotations

from llamatune.types import TrialConfig


def runtime_flags(config: TrialConfig, *, moe: bool = True) -> list[str]:
    flags = [
        "-ngl", str(config.gpu_layers), "-b", str(config.batch),
        "-ub", str(config.ubatch), "-t", str(config.threads),
    ]
    if config.ot_spec is not None:
        flags += ["-ot", config.ot_spec]
    elif moe and config.moe_cpu_layers > 0:
        flags += ["--n-cpu-moe", str(config.moe_cpu_layers)]
    if config.threads_batch is not None:
        flags += ["-tb", str(config.threads_batch)]
    if config.flash_attn:
        flags += ["-fa", "on"]
    if not config.mmap:
        flags.append("--no-mmap")
    if config.no_kv_offload:
        flags.append("--no-kv-offload")
    if config.cache_type_k != "f16":
        flags += ["-ctk", config.cache_type_k]
    if config.cache_type_v != "f16":
        flags += ["-ctv", config.cache_type_v]
    if config.tensor_split is not None:
        flags += ["-ts", ",".join(f"{value:g}" for value in config.tensor_split)]
    if config.split_mode is not None:
        flags += ["-sm", config.split_mode]
    return flags
```

Import it in both consumers. Keep compatibility wrappers to avoid unrelated call-site churn:

```python
def _cli_flags(config: TrialConfig, *, moe: bool) -> list[str]:
    return runtime_flags(config, moe=moe)
```

```python
def _runtime_flags(config: TrialConfig) -> list[str]:
    return runtime_flags(config)
```

In `report._format_config`, remove the unconditional `tb=... ot=...` fragment from the initial `text` construction. Keep the two existing conditional appends.

- [ ] **Step 4: Quote reference commands with the same POSIX argv representation as export.** Add `import shlex` to `recommend.py`. Keep the local name `runtime_flags` used by `build_recommended_sh` for its list. Replace the two primary command expressions with:

```python
"# " + shlex.join(["llama-server", "-m", model_path, *runtime_flags])
```

```python
"# " + shlex.join(["llama-cli", "-m", model_path, *runtime_flags])
```

Remove the obsolete `flags = " ".join(runtime_flags)` when no longer used. Keep the existing `alternate_flags` construction, remove `alternate_text`, and replace the alternate command-list append with:

```python
lines += [
    "# " + shlex.join(["llama-server", "-m", model_path, *alternate_flags]),
    "# " + shlex.join(["llama-cli", "-m", model_path, *alternate_flags]),
    "",
]
```

Preserve the existing label and comment-only status. Do not alter executable bits or export formats. This task covers spaces and quotes, not the separate hostile-control-character contract O1.

- [ ] **Step 5: Run GREEN and commit after focused patch checks.**

```bash
env -u PYTHONPATH -u VIRTUAL_ENV uv run --locked --python 3.12 pytest --no-cov tests/unit/test_recommend.py tests/unit/test_report.py -v
git diff --check
git add src/llamatune/runtimeflags.py src/llamatune/recommend.py src/llamatune/report.py tests/unit/test_recommend.py
git commit -m "fix: preserve tuned runtime flags in recommendation commands"
```

## Task E3: Reuse identity-bound confirmations after disk reload (#17)

**Files:**
- Modify: `src/llamatune/search.py:3064-3172`.
- Modify: `src/llamatune/session.py:561-580`.
- Modify: `tests/unit/test_search.py`.
- Test: `tests/unit/test_session.py`.
- Modify: `DESIGN.md:481-515`.

**Interfaces:**
- Consumes: existing `_confirm(config: TrialConfig) -> _Confirm`, `Session.entries`, and `resume_tuning` identity validation.
- Produces: `_Engine._confirmation_key(config: TrialConfig) -> dict[str, Any]`, `_Engine._reusable_confirmations(config: TrialConfig) -> dict[int, tuple[float, float]]`.
- Produces: `Session.confirmation_dir(trial_id: str, run: int) -> Path`, an unused confined capture directory.
- New `confirmation_run` fields: `confirmation_key` (object), `purpose` (`confirmation` or `revalidation`), and `capture_dir` (session-relative string). Existing readers tolerate additive fields.
- Reuse only successful, finite, positive measurements with compatible thermal provenance, complete key equality, and an actual benchmark hash.
- Legacy records lacking the new key are not reusable. Retain their bytes and count their executions.
- Revalidation is intentionally fresh. `_tuning_budget_enforced=False` disables reuse and records `purpose="revalidation"`.

- [ ] **Step 1: Write the disk-backed interruption regression in `test_search.py`.** This file defines `_setup` and `_envelope_config` and imports `Session` and the required types. The fake benchmark is a repository fixture, not a native GPU binary.

```python
@pytest.mark.parametrize("budget", [6, 60])
def test_confirmation_interrupt_resume_does_not_repeat_completed_runs(
    tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch, budget: int,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path, fake_bin_dir, tiny_gguf, budget_trials=budget
    )
    engine = search._Engine(session, hw, model, llama, options)
    engine._establish_baseline()
    engine._finalizing = True
    cfg = _envelope_config(1)
    append = engine._append

    def interrupt_after_first(record: dict[str, Any]) -> None:
        append(record)
        if record.get("type") == "confirmation_run":
            raise KeyboardInterrupt

    monkeypatch.setattr(engine, "_append", interrupt_after_first)
    with pytest.raises(KeyboardInterrupt):
        engine._confirm(cfg)
    first = Session.load(session.dir)
    assert len([e for e in first.entries if e.get("type") == "confirmation_run"]) == 1
    before = search._count_executed(first.entries)
    captures = {
        str(p.relative_to(session.dir)): p.read_bytes()
        for p in (session.dir / "trials").rglob("*") if p.is_file()
    }
    monkeypatch.setattr("llamatune.hardware.assess_hardware", lambda: hw)

    def finish_confirmation(resumed: search._Engine) -> TuneOutcome:
        resumed._establish_baseline()
        resumed._finalizing = True
        resumed._confirm(cfg)
        return TuneOutcome(session_dir=resumed.session.dir, analysis={}, exit_code=0)

    monkeypatch.setattr(search._Engine, "run", finish_confirmation)
    assert search.resume_tuning(session.dir).exit_code == 0
    loaded = Session.load(session.dir)
    rows = [e for e in loaded.entries if e.get("type") == "confirmation_run"]
    assert [e["run"] for e in rows] == [1, 2, 3]
    assert search._count_executed(loaded.entries) == before + 2
    assert all((session.dir / name).read_bytes() == data for name, data in captures.items())
    assert search.resume_tuning(session.dir).exit_code == 0
    after_second_resume = Session.load(session.dir)
    assert [e for e in after_second_resume.entries if e.get("type") == "confirmation_run"] == rows
    assert search._count_executed(after_second_resume.entries) == before + 2
```

The test replaces only the resumed top-level stage runner to isolate confirmation. `resume_tuning`, identity validation, the journal load, and fake child measurements are real. Do not describe this as full native acceptance.

- [ ] **Step 2: Run RED.**

```bash
env -u PYTHONPATH -u VIRTUAL_ENV uv run --locked --python 3.12 pytest --no-cov tests/unit/test_search.py -k confirmation_interrupt_resume -v
```

Expected: at budget 60, current code records `[1, 1, 2, 3]` instead of `[1, 2, 3]`. At budget 6, it leaves only `[1]` because preflight incorrectly reserves a full batch again.

- [ ] **Step 3: Add the two identity/reuse methods to `_Engine`.** Use the existing `math` import or add it if absent. Preserve the last matching record for an index, even if that last record is unusable. Do not resurrect an earlier successful sample after a later failed sample.

```python
def _confirmation_key(self, config: TrialConfig) -> dict[str, Any]:
    return {
        "config": config.to_dict(),
        "model_fingerprint": self.model.fingerprint,
        "bench_sha256": self.llama.bench_sha256,
        "help_sha256": self.llama.help_sha256,
        "pp": self.options.pp,
        "tg": self.options.tg,
        "depth": self.options.depth,
        "reps_confirm": self.options.reps_confirm,
        "baseline_runs": self.options.baseline_runs,
    }


def _reusable_confirmations(self, config: TrialConfig) -> dict[int, tuple[float, float]]:
    if not self._tuning_budget_enforced or self.llama.bench_sha256 is None:
        return {}
    key = self._confirmation_key(config)
    latest: dict[int, dict[str, Any]] = {}
    for entry in self.session.entries:
        index = entry.get("run")
        if (
            entry.get("type") == "confirmation_run"
            and entry.get("trial_id") == config.trial_id
            and entry.get("purpose") == "confirmation"
            and entry.get("confirmation_key") == key
            and isinstance(index, int)
            and not isinstance(index, bool)
            and 1 <= index <= self.options.baseline_runs
        ):
            latest[index] = entry
    result: dict[int, tuple[float, float]] = {}
    for index, entry in latest.items():
        pp, tg = entry.get("pp_mean"), entry.get("tg_mean")
        if entry.get("status") != "ok":
            continue
        if entry.get("thermally_contaminated") and not (
            entry.get("thermal_retried") is True
            and entry.get("thermal_retry_contaminated") is False
        ):
            continue
        if (
            isinstance(pp, (int, float)) and not isinstance(pp, bool)
            and isinstance(tg, (int, float)) and not isinstance(tg, bool)
            and math.isfinite(pp) and math.isfinite(tg) and pp > 0 and tg > 0
        ):
            result[index] = (float(pp), float(tg))
    return result
```

- [ ] **Step 4: Allocate captures through Session without overwriting previous attempts.** `session.py` already imports `secrets` and defines `_confine`. Add this method next to `trial_dir`:

```python
def confirmation_dir(self, trial_id: str, run: int) -> Path:
    if run < 1:
        raise ValueError("confirmation run must be positive")
    for attempt in range(10):
        name = f"confirm-{run}"
        if attempt:
            name += f"-attempt-{secrets.token_hex(3)}"
        path = _confine(self._dir, "trials", trial_id, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return path
    raise OSError("could not allocate confirmation capture directory")
```

In `_confirm`, replace the current direct `confirm_dir.mkdir(...)` block and derive labels from the allocated path:

```python
confirm_dir = self.session.confirmation_dir(config.trial_id, index)
confirm_rel = str(confirm_dir.relative_to(self.session.dir))
run_id = f"{config.trial_id}-{confirm_dir.name}"
```

Use `confirm_rel` in `_write_command` and `f"{confirm_rel}/thermal-retry"` as `retry_rel_dir`. Keep `confirm_dir / "thermal-retry"` for its path. First-attempt names stay compatible. Subsequent attempts retain old captures.

- [ ] **Step 5: Integrate reuse and missing-count budgeting in `_confirm`.** After the hard-cap guard, compute:

```python
reusable = self._reusable_confirmations(config)
missing = self.options.baseline_runs - len(reusable)
if missing:
    self._quiet_gate("confirmation")
```

Replace the preflight budget condition with this condition, keeping its existing warning body:

```python
if (
    missing > 0
    and self._tuning_budget_enforced
    and self.executed_count + missing > self.options.budget_trials
):
    warning = "confirmation skipped because trial budget was exhausted"
    if warning not in self.extra_warnings:
        self.extra_warnings.append(warning)
        self._emit("warning", message=warning)
    return _Confirm(False, None, None, 0.0)
```

At the beginning of the existing index loop, before `_can_execute()`:

```python
if index in reusable:
    pp_mean, tg_mean = reusable[index]
    pp_means.append(pp_mean)
    tg_means.append(tg_mean)
    continue
```

Add these fields to each newly appended `confirmation_run`:

```python
"confirmation_key": self._confirmation_key(config),
"purpose": "confirmation" if self._tuning_budget_enforced else "revalidation",
"capture_dir": confirm_rel,
```

Keep existing thermal retry classification, aggregate statistics, threshold, and count increments. Reuse appends no countable measurement and emits no invented execution event.

- [ ] **Step 6: Add cache rejection and budget-boundary tests.** Append these tests to `test_search.py`:

```python
@pytest.mark.parametrize("change", [
    {"status": "timeout"}, {"pp_mean": float("nan")}, {"tg_mean": 0.0},
    {"thermally_contaminated": True}, {"confirmation_key": {}},
    {"purpose": "revalidation"},
])
def test_confirmation_reuse_rejects_invalid_or_foreign_evidence(
    tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path, change: dict[str, Any]
) -> None:
    session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf)
    engine = search._Engine(session, hw, model, llama, options)
    cfg = _envelope_config(1)
    row = {
        "type": "confirmation_run", "trial_id": cfg.trial_id, "run": 1,
        "confirmation_key": engine._confirmation_key(cfg), "purpose": "confirmation",
        "status": "ok", "pp_mean": 100.0, "tg_mean": 20.0,
        "thermally_contaminated": False,
    }
    session.append(row)
    assert engine._reusable_confirmations(cfg) == {1: (100.0, 20.0)}
    session.append({**row, **change})
    if "confirmation_key" in change or "purpose" in change:
        assert engine._reusable_confirmations(cfg) == {1: (100.0, 20.0)}
    else:
        assert engine._reusable_confirmations(cfg) == {}
    foreign = dataclasses.replace(llama, bench_sha256="different-build")
    assert search._Engine(session, hw, model, foreign, options)._reusable_confirmations(cfg) == {}
    engine._tuning_budget_enforced = False
    assert engine._reusable_confirmations(cfg) == {}


def test_confirmation_all_cached_needs_no_remaining_budget(
    tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf)
    engine = search._Engine(session, hw, model, llama, options)
    engine._establish_baseline()
    cfg = _envelope_config(1)
    for index in range(1, options.baseline_runs + 1):
        session.append({
            "type": "confirmation_run", "trial_id": cfg.trial_id, "run": index,
            "confirmation_key": engine._confirmation_key(cfg), "purpose": "confirmation",
            "status": "ok", "pp_mean": 100.0, "tg_mean": 20.0,
        })
    engine.executed_count = options.budget_trials
    monkeypatch.setattr(engine, "_run_child", lambda **k: pytest.fail("cached run executed"))
    result = engine._confirm(cfg)
    assert result.pp is not None and result.tg is not None
    assert engine.executed_count == options.budget_trials
```

In `test_search.py`, also exercise the new Session method without needing private Session constructor details:

```python
def test_confirmation_capture_allocation_preserves_previous_attempt(
    tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path,
) -> None:
    session, _, _, _, _ = _setup(tmp_path, fake_bin_dir, tiny_gguf)
    first = session.confirmation_dir("trial", 1)
    (first / "stdout.json").write_bytes(b"retained")
    second = session.confirmation_dir("trial", 1)
    assert second != first and second.is_dir()
    assert (first / "stdout.json").read_bytes() == b"retained"
    with pytest.raises(SessionPathError):
        session.confirmation_dir("../../../outside", 1)
```

Retain `test_revalidate_confirmed_session_returns_comparison` and thermal confirmation tests in the run selection. These prove revalidation still measures and thermal events remain counted.

- [ ] **Step 7: Document the additive evidence fields and run GREEN.** Add to DESIGN section 12:

```text
Confirmation resume reuses only successful measurements with matching configuration,
model fingerprint, benchmark/help hashes, workload, depth, and confirmation settings.
Each new confirmation_run carries confirmation_key, purpose, and capture_dir.
Records without that identity remain evidence and budget-counted but are not reused.
Revalidation always measures again. New attempts use distinct capture directories.
```

```bash
env -u PYTHONPATH -u VIRTUAL_ENV uv run --locked --python 3.12 pytest --no-cov tests/unit/test_search.py -k 'confirmation or revalidate or thermal' -v
env -u PYTHONPATH -u VIRTUAL_ENV uv run --locked --python 3.12 pytest --no-cov tests/unit/test_session.py -v
```

Expected: disk resume executes only missing indices, fully cached confirmation works at the trial limit, old capture bytes remain intact, and revalidation remains fresh. Review key equality when a benchmark hash is absent. Do not make absent hashes cache-compatible.

- [ ] **Step 8: Commit after focused patch checks.**

```bash
git diff --check
git add src/llamatune/search.py src/llamatune/session.py tests/unit/test_search.py DESIGN.md
git commit -m "fix: reuse identity-bound confirmations on resume"
```

## Task E4: Keep cached MoE candidates out of execution patience (#31)

**Files:**
- Modify: `src/llamatune/search.py:1352-1392`.
- Modify: `tests/unit/test_search.py`.

**Interfaces:**
- Consumes: `_evaluate(cfg: TrialConfig, dim: str) -> _Trial` and `executed_count`.
- Produces: unchanged `_hill_climb_moe(direction: int) -> None` signature.
- Improvements reset misses regardless of cache status. Only a newly executed non-improvement increments misses, once.
- Prevent infinite cached revisits with a per-invocation set of candidate trial IDs. Retain the 12-execution cap and existing context check.

- [ ] **Step 1: Add a cache-versus-execution regression.**

```python
@pytest.mark.parametrize("cached", [True, False])
def test_hill_climb_moe_reaches_improvement_after_cache_replay(
    tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch, cached: bool,
) -> None:
    session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf)
    model = dataclasses.replace(model, moe=True, n_layer=32, ngl_all=33, expert_count=8)
    engine = search._Engine(session, hw, model, llama, options)
    cfg = _envelope_config(1)
    metric = MetricStats(mean=100.0, stdev=0.0, cv=0.0, n=3)
    engine.baseline = BaselineResult(
        runs=3, pp=metric, tg=metric, noise_floor_cv=0.01,
        fallback=None, resolved_defaults=cfg.to_dict(),
    )
    engine.incumbent_config = cfg
    engine.incumbent_score = 1.0
    visited: list[int] = []

    def evaluate(candidate: TrialConfig, dim: str) -> search._Trial:
        del dim
        visited.append(candidate.moe_cpu_layers)
        if not cached:
            engine.executed_count += 1
        score = {1: 1.2, 3: 1.1, 2: 1.3}.get(candidate.moe_cpu_layers, 1.0)
        return search._Trial("ok", 100.0, 100.0, score)

    monkeypatch.setattr(engine, "_evaluate", evaluate)
    engine._hill_climb_moe(1)
    assert visited[:3] == [1, 3, 2]
    assert engine.incumbent_config.moe_cpu_layers == 2
    assert len(visited) == len(set(visited))
    assert engine.executed_count <= 12
```

- [ ] **Step 2: Run RED.**

```bash
env -u PYTHONPATH -u VIRTUAL_ENV uv run --locked --python 3.12 pytest --no-cov tests/unit/test_search.py -k hill_climb_moe_reaches -v
```

Expected: the cached case stops after candidates 1 and 3, before the improving candidate 2. The uncached case reaches candidate 2 but repeats candidate 3. Both failures expose behavior this patch repairs.

- [ ] **Step 3: Make the minimal loop correction.** Before the `while` loop:

```python
visited: set[str] = set()
```

After constructing `cfg`, before recording `before`:

```python
if cfg.trial_id in visited:
    break
visited.add(cfg.trial_id)
```

Replace the existing `else` and trailing cache increment with:

```python
else:
    if self.executed_count > before:
        misses += 1
    step_index = 0
```

Retain `misses = 0` and the step-index increase in the improving branch. Do not move the context-validation gate or remove budget checks. The visited set bounds replay loops without treating cached evidence as a new measurement.

- [ ] **Step 4: Run GREEN and commit after focused patch checks.**

```bash
env -u PYTHONPATH -u VIRTUAL_ENV uv run --locked --python 3.12 pytest --no-cov tests/unit/test_search.py -k 'moe or Budget or resume' -v
git add src/llamatune/search.py tests/unit/test_search.py
git commit -m "fix: exclude cached MoE candidates from execution patience"
```

## Task E5: Distinguish Night Shift failure from user interruption (#10)

**Files:**
- Modify: `src/llamatune/nightshift.py:651-663,965-989` and final outcome wording near line 506.
- Modify: `tests/integration/test_nightshift.py`.
- Modify: `DESIGN.md` exit-code paragraph, `docs/nightshift-design.md` exit-code and circuit-breaker paragraphs.

**Interfaces:**
- Consumes: existing `stop`, `failed`, child item outcomes, and signal handler.
- Produces: local `user_interrupted: bool`, distinct from scheduler stop.
- Exit 4 means a user signal/KeyboardInterrupt or a child journaled as interrupted. Exit 1 means a failure including the three-model breaker. Normal/deadline completion retains existing semantics.
- Report window outcome uses `interrupted` only for exit 4, `failed` for exit 1, and `completed` for exit 0. Finalization must not label failure as user interruption.

- [ ] **Step 1: Add a three-model breaker regression.** Use existing imports and helpers in `tests/integration/test_nightshift.py`.

```python
def test_three_model_failure_breaker_exits_one(
    tmp_path: Path, tiny_gguf: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = inspect_model(tiny_gguf)
    models = tuple(
        DiscoveredModel(
            path=tiny_gguf,
            report=dataclasses.replace(original, fingerprint=f"model-{index}"),
            shard_paths=(), group_key=None, representative=True,
        )
        for index in range(4)
    )
    _patch_foundation(monkeypatch, tmp_path, models[0])
    monkeypatch.setattr(discovery, "discover_models", lambda *a, **k: models)
    calls: list[Path] = []

    def failed(session: Any, *args: object, **kwargs: object) -> TuneOutcome:
        del args, kwargs
        calls.append(session.dir)
        return TuneOutcome(session_dir=session.dir, analysis={}, exit_code=3)

    monkeypatch.setattr(search, "run_tuning", failed)
    result = run_nightshift(_options(tmp_path, tmp_path))
    assert result.exit_code == 1
    assert len(calls) == 3
    assert any("circuit breaker" in value for value in result.summary["warnings"])
    assert result.summary["window"]["outcome"] == "failed"
```

Add this first-signal case. Add `import signal` to the test file.

```python
def test_first_user_signal_retains_exit_four(
    tmp_path: Path, tiny_gguf: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DiscoveredModel(
        path=tiny_gguf, report=inspect_model(tiny_gguf), shard_paths=(),
        group_key=None, representative=True,
    )
    _patch_foundation(monkeypatch, tmp_path, model)

    def signal_then_finish(session: Any, *args: object, **kwargs: object) -> TuneOutcome:
        del args, kwargs
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
        return TuneOutcome(session_dir=session.dir, analysis={}, exit_code=0)

    monkeypatch.setattr(search, "run_tuning", signal_then_finish)
    result = run_nightshift(_options(tmp_path, tmp_path))
    assert result.exit_code == 4
    assert result.summary["window"]["outcome"] == "interrupted"
```

- [ ] **Step 2: Run RED.**

```bash
env -u PYTHONPATH -u VIRTUAL_ENV uv run --locked --python 3.12 pytest --no-cov tests/integration/test_nightshift.py -k 'failure_breaker or user_signal or stopped_session' -v
```

Expected: the breaker test returns 4 instead of 1. Signal behavior is the control that must stay green.

- [ ] **Step 3: Track the actual interruption cause.** Initialize `user_interrupted = False` beside `stop` and `second_signal`. Include it in `handle_signal`'s `nonlocal` declaration and set it before either signal branch:

```python
nonlocal stop, second_signal, user_interrupted
user_interrupted = True
```

In the outer `except KeyboardInterrupt`, also set `user_interrupted = True`. Replace the always-truthy `or phase_queues` exit decision with:

```python
interrupted = user_interrupted or second_signal or any(
    item.get("outcome") == "interrupted" for item in items
)
exit_code = 4 if interrupted else 1 if failed else 0
```

Keep `stop` as loop control. In `_finalize`, replace the window's existing `"interrupted" if stopped else "completed"` expression with:

```python
"interrupted" if exit_code == 4 else "failed" if exit_code == 1 else "completed"
```

Do not change item execution outcome strings or queue structure. Failures and interruption remain different dimensions from the GPU context-validation contract G4.

- [ ] **Step 4: Align docs and run GREEN.** Use this exact rule in both exit-code documents:

```text
Night Shift exits 4 for user interruption or an interrupted child session.
The consecutive tune-failure circuit breaker exits 1 and records a failed window.
A stop caused only by the work window ending is not a user interruption.
```

```bash
env -u PYTHONPATH -u VIRTUAL_ENV uv run --locked --python 3.12 pytest --no-cov tests/integration/test_nightshift.py tests/unit/test_nightshift_plan.py tests/unit/test_cli_nightshift.py -v
```

Expected: breaker failure exits 1, first and second interruption paths retain exit 4, deadline deferral does not become failure, and reports agree with outcomes.

- [ ] **Step 5: Commit after focused patch checks.**

```bash
git diff --check
git add src/llamatune/nightshift.py tests/integration/test_nightshift.py DESIGN.md docs/nightshift-design.md
git commit -m "fix: distinguish Night Shift failure from interruption"
```

## Integration acceptance

- [ ] Run the complete baseline after integrating these patches with S1-S3. Focused `--no-cov` runs are not the final coverage gate.
- [ ] Re-run old GPU hard-cap and unmeasured-fallback regressions. No refactor may weaken these existing contracts.
- [ ] Review public commands and artifacts as well as private helpers. Confirm JSON stdout is still parseable and all derived commands preserve measured configuration.
- [ ] Keep O1-O3 and G1-G5 from the triage plan open. This first wave is not full secure-execution, Vulkan, or unattended-GPU acceptance.
