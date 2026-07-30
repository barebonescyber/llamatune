from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from llamatune.nightshift import (
    NightshiftPathError,
    NightshiftRun,
    build_initial_plan,
    deepen_order,
    deepening_changes_profile,
    item_fits,
    profile_values,
    resolve_deadline,
    tune_failure_breaker,
)
from llamatune.types import (
    DiscoveredModel,
    ModelReport,
    NightshiftOptions,
    RegistryRecord,
    WorkItem,
)


def _options(tmp_path: Path, **overrides: object) -> NightshiftOptions:
    values: dict[str, object] = {
        "models_dir": tmp_path / "models",
        "llama_bin": None,
        "sessions_dir": tmp_path / "sessions",
        "until": None,
        "max_hours": None,
        "profile": "deep",
        "drift_threshold": 0.05,
        "calibration_runs": 3,
        "duplicates": "one",
        "include": (),
        "exclude": (),
        "dry_run": False,
        "target": "balanced",
        "allow_lossy": False,
        "ctx_size": None,
        "vram_reserve_mb": None,
        "cooldown_s": None,
        "full_hash": False,
        "budget_trials": None,
        "reps_search": None,
        "reps_confirm": None,
        "baseline_runs": None,
    }
    values.update(overrides)
    return NightshiftOptions(**values)  # type: ignore[arg-type]


def _model(path: Path, fingerprint: str, size: int) -> DiscoveredModel:
    report = ModelReport(
        path=path,
        size_bytes=size,
        architecture="test",
        n_layer=2,
        ngl_all=3,
        expert_count=0,
        moe=False,
        name=path.stem,
        fingerprint=fingerprint,
        full_sha256=None,
    )
    return DiscoveredModel(
        path=path,
        report=report,
        shard_paths=(),
        group_key=None,
        representative=True,
    )


def _record(tmp_path: Path, fingerprint: str, created: str) -> RegistryRecord:
    return RegistryRecord(
        fingerprint=fingerprint,
        session_dir=tmp_path / fingerprint[:8],
        created=created,
        outcome="defaults_optimal",
        reference_config=None,
        reference_pp=100.0,
        reference_tg=20.0,
        noise_floor_cv=0.01,
        pp_workload=512,
        tg_workload=128,
        reps_confirm=5,
        target="balanced",
        build_commit=None,
        help_sha256="help",
        median_trial_wall_s=10.0,
        session_wall_s=60.0,
    )


def test_deadline_wraps_until_and_uses_earliest_cap() -> None:
    central = timezone(timedelta(hours=-5))
    start = datetime(2026, 7, 17, 3, 0, tzinfo=UTC)  # 22:00 local
    assert resolve_deadline(start, "07:00", None, local_tz=central) == datetime(
        2026, 7, 17, 12, 0, tzinfo=UTC
    )
    assert resolve_deadline(start, "07:00", 2, local_tz=central) == start + timedelta(hours=2)


def test_profiles_resolve_overrides_and_deepening(tmp_path: Path) -> None:
    options = _options(tmp_path, budget_trials=77, cooldown_s=1.5)
    assert profile_values(options)["budget_trials"] == 77
    assert profile_values(options)["reps_confirm"] == 8
    assert profile_values(options, deepen=True)["budget_trials"] == 77
    assert profile_values(options, deepen=True)["cooldown_s"] == 1.5
    assert not deepening_changes_profile(options)
    assert deepening_changes_profile(_options(tmp_path, profile="standard"))
    identical = _options(
        tmp_path,
        budget_trials=77,
        reps_search=5,
        reps_confirm=8,
        baseline_runs=5,
        cooldown_s=1.5,
    )
    assert not deepening_changes_profile(identical)


def test_initial_plan_resumes_before_smallest_new_tunes(tmp_path: Path) -> None:
    large = _model(tmp_path / "large.gguf", "a" * 64, 200)
    small = _model(tmp_path / "small.gguf", "b" * 64, 100)
    resumed = tmp_path / "old-session"
    plan = build_initial_plan(
        (large, small), {}, ((resumed, large.report.fingerprint),), calibration_runs=3
    )
    assert [item.kind for item in plan] == ["resume", "tune"]
    assert plan[1].model_path == small.path


def test_calibrations_are_stalest_first(tmp_path: Path) -> None:
    newer = _model(tmp_path / "new.gguf", "a" * 64, 100)
    older = _model(tmp_path / "old.gguf", "b" * 64, 100)
    records = {
        newer.report.fingerprint: _record(tmp_path, newer.report.fingerprint, "2026-02-01"),
        older.report.fingerprint: _record(tmp_path, older.report.fingerprint, "2026-01-01"),
    }
    plan = build_initial_plan((newer, older), records, (), calibration_runs=3)
    assert [item.model_path for item in plan] == [older.path, newer.path]


def test_fit_rules_distinguish_tunes_and_calibrations(tmp_path: Path) -> None:
    tune = WorkItem(
        kind="tune",
        model_path=None,
        fingerprint=None,
        session_dir=None,
        reference_fingerprint=None,
        estimated_minutes=None,
        reason="test",
    )
    calibration = WorkItem(
        kind="calibrate",
        model_path=None,
        fingerprint=None,
        session_dir=None,
        reference_fingerprint=None,
        estimated_minutes=8.0,
        reason="test",
    )
    assert not item_fits(tune, 19.9)
    assert item_fits(tune, 20.0)
    assert not item_fits(calibration, 7.9)
    assert item_fits(calibration, 8.0)


def test_deepening_rotates_least_recent_record_first(tmp_path: Path) -> None:
    recent = _model(tmp_path / "recent.gguf", "a" * 64, 100)
    stale = _model(tmp_path / "stale.gguf", "b" * 64, 100)
    records = {
        recent.report.fingerprint: _record(tmp_path, recent.report.fingerprint, "2026-02-01"),
        stale.report.fingerprint: _record(tmp_path, stale.report.fingerprint, "2026-01-01"),
    }
    assert deepen_order((recent, stale), records) == (stale, recent)


def test_failure_breaker_requires_three_distinct_consecutive_models() -> None:
    assert not tune_failure_breaker(("a", "b"))
    assert not tune_failure_breaker(("a", "a", "b"))
    assert tune_failure_breaker(("a", "b", "c"))


def test_writer_rejects_traversal(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text("{}", encoding="utf-8")
    run = NightshiftRun.load(run_dir)
    with pytest.raises(NightshiftPathError):
        run.write_text("../escape", "bad")
