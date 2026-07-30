"""Integration scenario 4: resume skips journaled trials and continues.

The first run exhausts a small trial budget; the session's recorded budget
is then raised (the budget lives in session.json, DESIGN §12 "continues
within the recorded budgets") and the session is resumed. The second run
must execute only trial ids that were never journaled before.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from llamatune.llama import discover_llama
from llamatune.model import inspect_model
from llamatune.search import resume_tuning, run_tuning
from llamatune.session import Session
from llamatune.types import GPUInfo, HardwareReport, TuneOptions


def _gpu_hardware() -> HardwareReport:
    return HardwareReport(
        os_name="Linux",
        arch="x86_64",
        cpu_model="Fake CPU",
        physical_cores=8,
        logical_cores=16,
        perf_cores=None,
        ram_mb=32768,
        gpus=(GPUInfo(vendor="nvidia", name="Fake GPU", vram_mb=24000, method="test"),),
        warnings=(),
    )


def _options(sessions_dir: Path, llama_bin: Path, budget_trials: int) -> TuneOptions:
    return TuneOptions(
        target="balanced",
        budget_trials=budget_trials,
        budget_minutes=None,
        reps_search=3,
        reps_confirm=5,
        baseline_runs=3,
        pp=512,
        tg=128,
        allow_lossy=False,
        cooldown_s=0.0,
        thermal_wait_cap_s=0.0,
        baseline_only=False,
        llama_bin=llama_bin,
        sessions_dir=sessions_dir,
        full_hash=False,
    )


def _journal_entries(session_dir: Path) -> list[dict[str, Any]]:
    entries = []
    with (session_dir / "journal.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # torn tail injected by a test
    return entries


def _trial_ids(entries: list[dict[str, Any]]) -> list[str]:
    return [e["trial_id"] for e in entries if e.get("type") == "trial"]


def _raise_budget(session_dir: Path, budget_trials: int) -> None:
    """Raise the recorded trial budget (DESIGN §12: budgets are recorded)."""
    meta_path = session_dir / "session.json"
    meta = json.loads(meta_path.read_text())
    meta["options"]["budget_trials"] = budget_trials
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


@pytest.fixture()
def first_run(tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path) -> tuple[Any, Path, list[str]]:
    hardware = _gpu_hardware()
    llama = discover_llama(fake_bin_dir)
    model = inspect_model(tiny_gguf)
    options = _options(tmp_path / "sessions", fake_bin_dir, budget_trials=8)
    session = Session.create(
        options.sessions_dir,
        model=model,
        hardware=hardware,
        llama=llama,
        options=options,
        argv=["llamatune", "tune", str(tiny_gguf)],
    )
    outcome = run_tuning(session, hardware, model, llama, options)
    first_ids = _trial_ids(_journal_entries(session.dir))
    return outcome, session.dir, first_ids


class TestResume:
    def test_resume_executes_only_new_trial_ids(
        self, first_run: tuple[Any, Path, list[str]]
    ) -> None:
        _, session_dir, first_ids = first_run
        assert first_ids, "the first run should have journaled trials"
        stages = [e.get("stage") for e in _journal_entries(session_dir) if e.get("type") == "stage"]
        assert "budget_exhausted" in stages

        _raise_budget(session_dir, 40)
        outcome = resume_tuning(session_dir)

        assert outcome.exit_code == 0
        entries = _journal_entries(session_dir)
        all_ids = _trial_ids(entries)
        new_ids = all_ids[len(first_ids) :]

        assert new_ids, "the resumed run should have executed new trials"
        # No trial id is ever journaled twice: journaled trials are skipped.
        assert not (set(new_ids) & set(first_ids))
        assert all(count == 1 for count in Counter(all_ids).values())

    def test_resume_does_not_rerun_baseline(self, first_run: tuple[Any, Path, list[str]]) -> None:
        _, session_dir, _ = first_run
        baseline_before = [
            e for e in _journal_entries(session_dir) if e.get("type") == "baseline_run"
        ]

        _raise_budget(session_dir, 40)
        resume_tuning(session_dir)

        baseline_after = [
            e for e in _journal_entries(session_dir) if e.get("type") == "baseline_run"
        ]
        assert len(baseline_after) == len(baseline_before) == 3

    def test_resume_reaches_known_optimum_and_rewrites_emissions(
        self, first_run: tuple[Any, Path, list[str]]
    ) -> None:
        _, session_dir, _ = first_run
        _raise_budget(session_dir, 40)
        outcome = resume_tuning(session_dir)

        assert outcome.exit_code == 0
        winner = outcome.analysis["winner"]
        assert winner is not None
        assert winner["config"]["flash_attn"] is True
        assert winner["config"]["gpu_layers"] == 33
        analysis_on_disk = json.loads((session_dir / "analysis.json").read_text())
        assert analysis_on_disk["winner"]["trial_id"] == winner["trial_id"]
        for name in ("report.md", "recommended.json", "recommended.sh"):
            assert (session_dir / name).is_file()

    def test_resume_journals_resumed_stage_and_session_end(
        self, first_run: tuple[Any, Path, list[str]]
    ) -> None:
        _, session_dir, _ = first_run
        _raise_budget(session_dir, 40)
        outcome = resume_tuning(session_dir)

        entries = _journal_entries(session_dir)
        stages = [e for e in entries if e.get("type") == "stage" and e.get("stage") == "resumed"]
        assert stages
        end = entries[-1]
        assert end["type"] == "session_end"
        assert end["exit_code"] == outcome.exit_code

    def test_resume_tolerates_torn_final_journal_line(
        self, first_run: tuple[Any, Path, list[str]]
    ) -> None:
        _, session_dir, _ = first_run
        _raise_budget(session_dir, 40)
        with (session_dir / "journal.jsonl").open("a", encoding="utf-8") as fh:
            fh.write('{"type": "trial", "trial_id": "torn')  # no newline: torn write

        outcome = resume_tuning(session_dir)

        assert outcome.exit_code == 0
        assert any("torn final line" in w for w in outcome.analysis["warnings"])
        # The torn tail was truncated on load, so the journal (including the
        # resumed run's appended entries) parses cleanly on the next load.
        reloaded = Session.load(session_dir)
        assert reloaded.resume_warnings == ()
        assert reloaded.entries[-1]["type"] == "session_end"
