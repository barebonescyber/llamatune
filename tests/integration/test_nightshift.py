from __future__ import annotations

import dataclasses
import signal
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from llamatune import calibrate, discovery, hardware, llama, registry, search
from llamatune.model import inspect_model
from llamatune.nightshift import resolve_deadline, run_nightshift
from llamatune.types import (
    CalibrationResult,
    DiscoveredModel,
    HardwareReport,
    LlamaCppReport,
    NightshiftOptions,
    RegistryRecord,
    TuneOutcome,
)


def _record(tmp_path: Path, model: DiscoveredModel) -> RegistryRecord:
    return RegistryRecord(
        fingerprint=model.report.fingerprint,
        session_dir=tmp_path / "reference",
        created="2026-01-01T00:00:00+00:00",
        outcome="defaults_optimal",
        reference_config=None,
        reference_pp=100.0,
        reference_tg=20.0,
        noise_floor_cv=0.01,
        pp_workload=512,
        tg_workload=128,
        reps_confirm=3,
        target="balanced",
        build_commit="test",
        help_sha256="fake",
        median_trial_wall_s=0.01,
        session_wall_s=1.0,
    )


def _calibration(model: DiscoveredModel, record: RegistryRecord, verdict: str) -> CalibrationResult:
    return CalibrationResult(
        fingerprint=model.report.fingerprint,
        reference_session=record.session_dir,
        verdict=verdict,
        pp=None,
        tg=None,
        drift_pp=0.2 if verdict == "drift" else 0.0,
        drift_tg=0.0,
        threshold=0.05,
        runs=3,
        reason=None,
        transfer_from=(
            record.fingerprint if record.fingerprint != model.report.fingerprint else None
        ),
        build_changed=False,
        artifact_dir=None,
    )


def _options(tmp_path: Path, model_dir: Path, **overrides: object) -> NightshiftOptions:
    values: dict[str, object] = {
        "models_dir": model_dir,
        "llama_bin": None,
        "sessions_dir": tmp_path / "sessions",
        "until": None,
        "max_hours": None,
        "profile": "standard",
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
        "cooldown_s": 0.0,
        "full_hash": False,
        "budget_trials": 1,
        "reps_search": 1,
        "reps_confirm": 1,
        "baseline_runs": 3,
    }
    values.update(overrides)
    return NightshiftOptions(**values)  # type: ignore[arg-type]


def _hardware() -> HardwareReport:
    return HardwareReport(
        os_name="test",
        arch="x86_64",
        cpu_model="fake",
        physical_cores=2,
        logical_cores=2,
        perf_cores=None,
        ram_mb=4096,
        gpus=(),
        warnings=(),
    )


def _llama(tmp_path: Path) -> LlamaCppReport:
    return LlamaCppReport(
        bench_path=tmp_path / "llama-bench",
        cli_path=None,
        server_path=None,
        capabilities=frozenset(),
        help_sha256="fake",
        build_commit="test",
        build_number=1,
    )


def _patch_foundation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model: DiscoveredModel
) -> None:
    monkeypatch.setattr(hardware, "assess_hardware", _hardware)
    monkeypatch.setattr(llama, "discover_llama", lambda _path: _llama(tmp_path))
    monkeypatch.setattr(discovery, "discover_models", lambda *_args, **_kwargs: (model,))
    monkeypatch.setattr(registry, "build_registry", lambda _path: {})
    monkeypatch.setattr(registry, "incomplete_sessions", lambda _path: ())


def test_no_window_means_one_pass_deadline_is_unbounded() -> None:
    assert resolve_deadline(datetime(2026, 1, 1, tzinfo=UTC), None, None) is None


def test_dry_run_exposes_full_plan_and_executes_no_benchmark(
    tmp_path: Path, tiny_gguf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = inspect_model(tiny_gguf)
    model = DiscoveredModel(
        path=tiny_gguf, report=report, shard_paths=(), group_key=None, representative=True
    )
    _patch_foundation(monkeypatch, tmp_path, model)
    monkeypatch.setattr(
        search,
        "run_tuning",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("benchmark executed")),
    )
    outcome = run_nightshift(_options(tmp_path, tiny_gguf.parent, dry_run=True))
    assert outcome.exit_code == 0
    assert outcome.summary["items"][0]["kind"] == "tune"
    assert outcome.summary["items"][0]["outcome"] == "planned"
    assert (outcome.run_dir / "plan.json").is_file()


def test_nearly_exhausted_deadline_defers_without_benchmark(
    tmp_path: Path, tiny_gguf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = inspect_model(tiny_gguf)
    model = DiscoveredModel(
        path=tiny_gguf, report=report, shard_paths=(), group_key=None, representative=True
    )
    _patch_foundation(monkeypatch, tmp_path, model)
    monkeypatch.setattr(
        search,
        "run_tuning",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("benchmark executed")),
    )
    outcome = run_nightshift(
        _options(tmp_path, tiny_gguf.parent, max_hours=0.1),
        now_fn=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert outcome.exit_code == 0
    assert outcome.summary["items"][0]["outcome"] == "deferred"


def test_llama_discovery_failure_writes_report_and_exits_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hardware, "assess_hardware", _hardware)
    monkeypatch.setattr(
        llama, "discover_llama", lambda _path: (_ for _ in ()).throw(RuntimeError("missing"))
    )
    outcome = run_nightshift(_options(tmp_path, tmp_path))
    assert outcome.exit_code == 3
    assert (outcome.run_dir / "nightshift.json").is_file()
    assert (outcome.run_dir / "nightshift-report.md").is_file()
    assert "llama.cpp discovery failed" in outcome.summary["warnings"][0]


def test_fresh_model_runs_an_ordinary_tuning_session(
    tmp_path: Path, tiny_gguf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = inspect_model(tiny_gguf)
    model = DiscoveredModel(
        path=tiny_gguf, report=report, shard_paths=(), group_key=None, representative=True
    )
    _patch_foundation(monkeypatch, tmp_path, model)

    def fake_tune(session: Any, *_args: object, **_kwargs: object) -> TuneOutcome:
        return TuneOutcome(session_dir=session.dir, analysis={"winner": None}, exit_code=1)

    monkeypatch.setattr(search, "run_tuning", fake_tune)
    outcome = run_nightshift(_options(tmp_path, tiny_gguf.parent))
    assert outcome.exit_code == 0
    assert [(item["kind"], item["outcome"]) for item in outcome.summary["items"]] == [
        ("tune", "succeeded")
    ]


def test_one_failed_model_does_not_prevent_the_next_model(
    tmp_path: Path, tiny_gguf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_report = inspect_model(tiny_gguf)
    second_path = tmp_path / "second.gguf"
    second_path.write_bytes(tiny_gguf.read_bytes() + b"different")
    second_report = dataclasses.replace(
        first_report, path=second_path, fingerprint="f" * 64, size_bytes=second_path.stat().st_size
    )
    models = (
        DiscoveredModel(
            path=tiny_gguf, report=first_report, shard_paths=(), group_key=None, representative=True
        ),
        DiscoveredModel(
            path=second_path,
            report=second_report,
            shard_paths=(),
            group_key=None,
            representative=True,
        ),
    )
    _patch_foundation(monkeypatch, tmp_path, models[0])
    monkeypatch.setattr(discovery, "discover_models", lambda *_args, **_kwargs: models)
    monkeypatch.setattr(
        search,
        "run_tuning",
        lambda session, *_args, **_kwargs: TuneOutcome(
            session_dir=session.dir, analysis={}, exit_code=3
        ),
    )
    outcome = run_nightshift(_options(tmp_path, tmp_path))
    assert outcome.exit_code == 1
    assert [item["outcome"] for item in outcome.summary["items"]] == ["failed", "failed"]


def test_three_model_failure_breaker_exits_one(
    tmp_path: Path, tiny_gguf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = inspect_model(tiny_gguf)
    models = tuple(
        DiscoveredModel(
            path=tiny_gguf,
            report=dataclasses.replace(original, fingerprint=f"model-{index}"),
            shard_paths=(),
            group_key=None,
            representative=True,
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


def test_first_user_signal_retains_exit_four(
    tmp_path: Path, tiny_gguf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = DiscoveredModel(
        path=tiny_gguf,
        report=inspect_model(tiny_gguf),
        shard_paths=(),
        group_key=None,
        representative=True,
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


def test_consistent_calibration_does_not_retune(
    tmp_path: Path, tiny_gguf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = DiscoveredModel(
        path=tiny_gguf,
        report=inspect_model(tiny_gguf),
        shard_paths=(),
        group_key=None,
        representative=True,
    )
    record = _record(tmp_path, model)
    _patch_foundation(monkeypatch, tmp_path, model)
    monkeypatch.setattr(registry, "build_registry", lambda _path: {record.fingerprint: record})
    monkeypatch.setattr(
        calibrate, "run_calibration", lambda *_args: _calibration(model, record, "consistent")
    )
    monkeypatch.setattr(
        search,
        "run_tuning",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("retune executed")),
    )
    outcome = run_nightshift(_options(tmp_path, tiny_gguf.parent))
    assert [(item["kind"], item["outcome"]) for item in outcome.summary["items"]] == [
        ("calibrate", "consistent")
    ]


def test_drift_calibration_enqueues_and_executes_retune(
    tmp_path: Path, tiny_gguf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = DiscoveredModel(
        path=tiny_gguf,
        report=inspect_model(tiny_gguf),
        shard_paths=(),
        group_key=None,
        representative=True,
    )
    record = dataclasses.replace(_record(tmp_path, model), depth_workload=32768)
    _patch_foundation(monkeypatch, tmp_path, model)
    monkeypatch.setattr(registry, "build_registry", lambda _path: {record.fingerprint: record})
    monkeypatch.setattr(
        calibrate, "run_calibration", lambda *_args: _calibration(model, record, "drift")
    )
    seen_depths: list[int | None] = []

    def tuned(session: Any, *_args: object, **_kwargs: object) -> TuneOutcome:
        seen_depths.append(session.options.depth)
        return TuneOutcome(session_dir=session.dir, analysis={}, exit_code=1)

    monkeypatch.setattr(search, "run_tuning", tuned)
    outcome = run_nightshift(_options(tmp_path, tiny_gguf.parent))
    assert [(item["kind"], item["outcome"]) for item in outcome.summary["items"]] == [
        ("calibrate", "drift"),
        ("retune", "succeeded"),
    ]
    assert seen_depths == [32768]


def test_nightshift_depth_requires_capability_and_writes_exit_three_report(
    tmp_path: Path, tiny_gguf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = DiscoveredModel(
        path=tiny_gguf,
        report=inspect_model(tiny_gguf),
        shard_paths=(),
        group_key=None,
        representative=True,
    )
    _patch_foundation(monkeypatch, tmp_path, model)
    outcome = run_nightshift(_options(tmp_path, tiny_gguf.parent, depth=8192))
    assert outcome.exit_code == 3
    assert any("does not support -d" in warning for warning in outcome.summary["warnings"])
    assert (outcome.run_dir / "nightshift-report.md").is_file()


def test_dynamic_transfer_after_representative_tune(
    tmp_path: Path, tiny_gguf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    representative = DiscoveredModel(
        path=tiny_gguf,
        report=inspect_model(tiny_gguf),
        shard_paths=(),
        group_key="group",
        representative=True,
    )
    sibling_report = dataclasses.replace(
        representative.report, path=tmp_path / "sibling.gguf", fingerprint="e" * 64
    )
    sibling = DiscoveredModel(
        path=sibling_report.path,
        report=sibling_report,
        shard_paths=(),
        group_key="group",
        representative=False,
    )
    _patch_foundation(monkeypatch, tmp_path, representative)
    monkeypatch.setattr(
        discovery, "discover_models", lambda *_args, **_kwargs: (representative, sibling)
    )
    record = _record(tmp_path, representative)
    tuned = False

    def build(_path: Path) -> dict[str, RegistryRecord]:
        return {record.fingerprint: record} if tuned else {}

    def tune(session: Any, *_args: object, **_kwargs: object) -> TuneOutcome:
        nonlocal tuned
        tuned = True
        return TuneOutcome(session_dir=session.dir, analysis={}, exit_code=1)

    monkeypatch.setattr(registry, "build_registry", build)
    monkeypatch.setattr(search, "run_tuning", tune)
    monkeypatch.setattr(
        calibrate, "run_calibration", lambda *_args: _calibration(sibling, record, "consistent")
    )
    outcome = run_nightshift(_options(tmp_path, tmp_path))
    assert [(item["kind"], item["outcome"]) for item in outcome.summary["items"]] == [
        ("tune", "succeeded"),
        ("calibrate", "consistent"),
    ]
    assert outcome.summary["items"][1]["calibration"]["transfer_from"] == record.fingerprint


def test_representative_failure_explicitly_defers_sibling(
    tmp_path: Path, tiny_gguf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    representative = DiscoveredModel(
        path=tiny_gguf,
        report=inspect_model(tiny_gguf),
        shard_paths=(),
        group_key="group",
        representative=True,
    )
    sibling_report = dataclasses.replace(
        representative.report, path=tmp_path / "sibling.gguf", fingerprint="e" * 64
    )
    sibling = DiscoveredModel(
        path=sibling_report.path,
        report=sibling_report,
        shard_paths=(),
        group_key="group",
        representative=False,
    )
    _patch_foundation(monkeypatch, tmp_path, representative)
    monkeypatch.setattr(
        discovery, "discover_models", lambda *_args, **_kwargs: (representative, sibling)
    )
    monkeypatch.setattr(
        search,
        "run_tuning",
        lambda session, *_args, **_kwargs: TuneOutcome(
            session_dir=session.dir, analysis={}, exit_code=3
        ),
    )
    outcome = run_nightshift(_options(tmp_path, tmp_path))
    assert [item["outcome"] for item in outcome.summary["items"]] == ["failed", "deferred"]
    assert outcome.summary["items"][1]["reason"] == "representative tune failed"


def test_stopped_session_journal_translates_to_nightshift_exit_four(
    tmp_path: Path, tiny_gguf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = DiscoveredModel(
        path=tiny_gguf,
        report=inspect_model(tiny_gguf),
        shard_paths=(),
        group_key=None,
        representative=True,
    )
    _patch_foundation(monkeypatch, tmp_path, model)

    def stopped(session: Any, *_args: object, **_kwargs: object) -> TuneOutcome:
        session.append({"type": "session_end", "exit_code": 1, "reason": "stopped_by_user"})
        return TuneOutcome(session_dir=session.dir, analysis={}, exit_code=1)

    monkeypatch.setattr(search, "run_tuning", stopped)
    outcome = run_nightshift(_options(tmp_path, tmp_path))
    assert outcome.exit_code == 4
    assert outcome.summary["items"][0]["outcome"] == "interrupted"


def test_identical_deepening_profile_is_skipped(
    tmp_path: Path, tiny_gguf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = DiscoveredModel(
        path=tiny_gguf,
        report=inspect_model(tiny_gguf),
        shard_paths=(),
        group_key=None,
        representative=True,
    )
    record = _record(tmp_path, model)
    _patch_foundation(monkeypatch, tmp_path, model)
    monkeypatch.setattr(registry, "build_registry", lambda _path: {record.fingerprint: record})
    monkeypatch.setattr(
        calibrate, "run_calibration", lambda *_args: _calibration(model, record, "consistent")
    )
    monkeypatch.setattr(
        search,
        "run_tuning",
        lambda session, *_args, **_kwargs: TuneOutcome(
            session_dir=session.dir, analysis={}, exit_code=1
        ),
    )
    fixed = datetime(2026, 1, 1, tzinfo=UTC)
    outcome = run_nightshift(_options(tmp_path, tmp_path, max_hours=1), now_fn=lambda: fixed)
    assert [item["kind"] for item in outcome.summary["items"]].count("deepen") == 0
    assert any("resolved profile matches" in warning for warning in outcome.summary["warnings"])


def test_spare_time_deepens_each_model_at_most_once(
    tmp_path: Path, tiny_gguf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = DiscoveredModel(
        path=tiny_gguf,
        report=inspect_model(tiny_gguf),
        shard_paths=(),
        group_key=None,
        representative=True,
    )
    record = _record(tmp_path, model)
    _patch_foundation(monkeypatch, tmp_path, model)
    monkeypatch.setattr(registry, "build_registry", lambda _path: {record.fingerprint: record})
    monkeypatch.setattr(
        calibrate, "run_calibration", lambda *_args: _calibration(model, record, "consistent")
    )
    monkeypatch.setattr(
        search,
        "run_tuning",
        lambda session, *_args, **_kwargs: TuneOutcome(
            session_dir=session.dir, analysis={}, exit_code=1
        ),
    )
    fixed = datetime(2026, 1, 1, tzinfo=UTC)
    outcome = run_nightshift(
        _options(
            tmp_path,
            tmp_path,
            max_hours=1,
            budget_trials=None,
            reps_search=None,
            reps_confirm=None,
            baseline_runs=None,
            cooldown_s=None,
        ),
        now_fn=lambda: fixed,
    )
    assert [item["kind"] for item in outcome.summary["items"]].count("deepen") == 1


def test_run_metadata_and_report_window_are_complete(
    tmp_path: Path, tiny_gguf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = DiscoveredModel(
        path=tiny_gguf,
        report=inspect_model(tiny_gguf),
        shard_paths=(),
        group_key=None,
        representative=True,
    )
    _patch_foundation(monkeypatch, tmp_path, model)
    fixed = datetime(2026, 1, 1, tzinfo=UTC)
    outcome = run_nightshift(
        _options(tmp_path, tmp_path, dry_run=True, max_hours=1), now_fn=lambda: fixed
    )
    run_meta = __import__("json").loads((outcome.run_dir / "run.json").read_text())
    assert run_meta["resolved_deadline"] == "2026-01-01T01:00:00+00:00"
    assert set(outcome.summary["window"]) == {"started", "ended", "deadline", "outcome"}
    report = (outcome.run_dir / "nightshift-report.md").read_text()
    assert "Started: 2026-01-01T00:00:00+00:00" in report
