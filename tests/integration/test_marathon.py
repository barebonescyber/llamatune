import json
import signal
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from llamatune import marathon as marathon_module
from llamatune.marathon import (
    MarathonPathError,
    MarathonRun,
    _entries,
    _trial_evidence,
    find_reentry,
    identity,
    run_marathon,
)
from llamatune.types import (
    ABResult,
    CalibrationResult,
    CoverageLedger,
    GPUInfo,
    HardwareReport,
    LlamaCppReport,
    MarathonOptions,
    MetricStats,
    ModelReport,
    TierCoverage,
    TrialConfig,
    TuneOutcome,
)


def options(tmp_path: Path) -> MarathonOptions:
    return MarathonOptions(
        model_path=tmp_path / "model.gguf",
        llama_bin=None,
        sessions_dir=tmp_path / "sessions",
        until=None,
        max_hours=None,
        rounds_max=2,
        converge_rounds=1,
        ab_blocks=3,
        depth_grid=(0,),
        ctx_size=8192,
        ctx_ladder=(8192,),
        matrix_refine=False,
        drift_threshold=0.05,
        dry_run=True,
        target="balanced",
        allow_lossy=False,
        vram_reserve_mb=None,
        cooldown_s=0,
        full_hash=False,
        pp=16,
        tg=8,
        quality_corpus=None,
        ot_search=False,
        budget_trials=1,
        reps_search=1,
        reps_confirm=1,
        baseline_runs=1,
    )


def reports(tmp_path: Path) -> tuple[HardwareReport, LlamaCppReport, ModelReport]:
    hardware = HardwareReport(
        os_name="linux",
        arch="x86_64",
        cpu_model="fake",
        physical_cores=1,
        logical_cores=1,
        perf_cores=None,
        ram_mb=1024,
        gpus=(GPUInfo(vendor="fake", name="fake", vram_mb=1024, method="fake"),),
        warnings=(),
    )
    llama = LlamaCppReport(
        bench_path=tmp_path / "llama-bench",
        cli_path=None,
        server_path=None,
        capabilities=frozenset(),
        help_sha256="help",
        build_commit=None,
        build_number=None,
    )
    model = ModelReport(
        path=tmp_path / "model.gguf",
        size_bytes=1,
        architecture="fake",
        n_layer=1,
        ngl_all=1,
        expert_count=0,
        moe=False,
        name="fake",
        fingerprint="a" * 64,
        full_sha256=None,
    )
    return hardware, llama, model


def config(*, flash_attn: bool = False) -> TrialConfig:
    return TrialConfig(
        gpu_layers=1,
        moe_cpu_layers=0,
        flash_attn=flash_attn,
        ubatch=512,
        batch=2048,
        threads=1,
        mmap=True,
        no_kv_offload=False,
        cache_type_k="f16",
        cache_type_v="f16",
    )


def metric(mean: float) -> MetricStats:
    return MetricStats(mean=mean, stdev=0.1, cv=0.01, n=3)


class AdvancingClock:
    def __init__(self, *, step_minutes: float = 20.0) -> None:
        self.value = datetime(2026, 1, 1, tzinfo=UTC)
        self.step = timedelta(minutes=step_minutes)

    def __call__(self) -> datetime:
        value = self.value
        self.value += self.step
        return value


def _coverage_ledger(remaining: int) -> CoverageLedger:
    tier = TierCoverage(
        enumerated=1,
        executed=1 - remaining,
        pruned=0,
        remaining=remaining,
        remaining_ids=("remaining",) if remaining else (),
    )
    return CoverageLedger(tiers=dict.fromkeys("ABC", tier), responsive=())


def _calibration(tmp_path: Path, verdict: str) -> CalibrationResult:
    measured = verdict == "drift"
    return CalibrationResult(
        fingerprint="a" * 64,
        reference_session=tmp_path,
        verdict=verdict,
        pp=metric(110.0) if measured else None,
        tg=metric(55.0) if measured else None,
        drift_pp=0.1 if measured else None,
        drift_tg=0.1 if measured else None,
        threshold=0.05,
        runs=3,
        reason="probe failed" if verdict == "error" else None,
        transfer_from=None,
        build_changed=False,
        artifact_dir=None,
    )


def _ab_result(label: str, verdict: str) -> ABResult:
    return ABResult(
        label=label,
        blocks=3,
        a_wins=0 if verdict == "b" else 3,
        b_wins=3 if verdict == "b" else 0,
        ties=0,
        a_pp=100.0,
        a_tg=50.0,
        b_pp=110.0,
        b_tg=55.0,
        margin=0.1,
        verdict=verdict,
    )


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    winners: list[TrialConfig | None] | None = None,
    tune_exit_codes: list[int] | None = None,
    calibration_verdicts: list[str] | None = None,
    ab_verdicts: list[str] | None = None,
    ledger_remaining: int = 0,
    warmup_drift: bool = False,
    on_tune: Any = None,
) -> dict[str, Any]:
    hardware, llama, model = reports(tmp_path)
    winner_values = list(winners or ())
    exit_values = list(tune_exit_codes or ())
    calibration_values = list(calibration_verdicts or ("consistent",))
    ab_values = list(ab_verdicts or ("b",))
    state: dict[str, Any] = {
        "hardware": hardware,
        "llama": llama,
        "model": model,
        "sessions": [],
        "matrix_calls": 0,
        "handlers": {},
    }

    monkeypatch.setattr("llamatune.hardware.assess_hardware", lambda: hardware)
    monkeypatch.setattr("llamatune.llama.discover_llama", lambda llama_bin: llama)
    monkeypatch.setattr(
        "llamatune.model.inspect_model",
        lambda model_path, *, full_hash: model,
    )
    monkeypatch.setattr(
        marathon_module,
        "_recon",
        lambda run, model, llama, options: {
            "pp": 100.0,
            "tg": 50.0,
            "cv_ref": 0.01,
            "runs": 10,
            "warmup_drift": warmup_drift,
            "trend": {"pp": 0.0, "tg": 0.0},
            "default_config": config().to_dict(),
        },
    )
    monkeypatch.setattr("llamatune.coverage.enumerate_space", lambda *args, **kwargs: ())
    monkeypatch.setattr(
        "llamatune.coverage.build_ledger",
        lambda *args, **kwargs: _coverage_ledger(ledger_remaining),
    )

    def create_session(sessions_dir: Path, **kwargs: Any) -> SimpleNamespace:
        session_dir = sessions_dir / f"round-{len(state['sessions']) + 1}"
        session_dir.mkdir(parents=True, exist_ok=True)
        state["sessions"].append(session_dir)
        return SimpleNamespace(dir=session_dir)

    monkeypatch.setattr(
        "llamatune.session.Session",
        SimpleNamespace(create=create_session),
    )

    def tune(session: Any, *args: Any, **kwargs: Any) -> TuneOutcome:
        if on_tune is not None:
            on_tune(state)
        exit_code = exit_values.pop(0) if exit_values else 0
        winner = winner_values.pop(0) if winner_values else None
        winner_payload = None if winner is None else {"config": winner.to_dict()}
        analysis = {"winner": winner_payload}
        if winner is not None:
            (session.dir / "analysis.json").write_text(
                json.dumps(
                    {
                        "winner": {
                            "config": winner.to_dict(),
                            "confirmed": True,
                            "pp": 110.0,
                            "tg": 55.0,
                        },
                        "feasibility": {"ctx_validated": 8192},
                    }
                ),
                encoding="utf-8",
            )
        return TuneOutcome(session_dir=session.dir, analysis=analysis, exit_code=exit_code)

    monkeypatch.setattr("llamatune.search.run_tuning", tune)
    monkeypatch.setattr(
        "llamatune.search.resume_tuning",
        lambda session_dir: TuneOutcome(
            session_dir=session_dir,
            analysis={"winner": None},
            exit_code=0,
        ),
    )

    def calibrate(*args: Any, **kwargs: Any) -> CalibrationResult:
        verdict = (
            calibration_values.pop(0) if len(calibration_values) > 1 else calibration_values[0]
        )
        return _calibration(tmp_path, verdict)

    monkeypatch.setattr("llamatune.calibrate.run_calibration", calibrate)

    def run_ab(*args: Any, **kwargs: Any) -> ABResult:
        verdict = ab_values.pop(0) if len(ab_values) > 1 else ab_values[0]
        return _ab_result(str(kwargs["label"]), verdict)

    monkeypatch.setattr("llamatune.abtest.run_ab", run_ab)

    def run_matrix(*args: Any, **kwargs: Any) -> list[Any]:
        state["matrix_calls"] += 1
        return []

    monkeypatch.setattr("llamatune.matrix.run_matrix", run_matrix)
    monkeypatch.setattr(signal, "getsignal", lambda sig: "original")

    def set_signal(sig: signal.Signals, handler: Any) -> None:
        state["handlers"][sig] = handler

    monkeypatch.setattr(signal, "signal", set_signal)
    return state


def test_writer_layout_confinement_and_reentry(tmp_path: Path) -> None:
    opts = options(tmp_path)
    hardware, llama, model = reports(tmp_path)
    run = MarathonRun.create(
        opts.sessions_dir,
        options=opts,
        hardware=hardware,
        llama=llama,
        model=model,
        argv=["llamatune", "marathon"],
    )
    meta = json.loads((run.dir / "run.json").read_text())
    meta["identity"] = identity(opts, model.fingerprint)
    run.write_json("run.json", meta)
    run.append({"type": "phase", "phase": "reconnaissance"})
    assert run.recon_dir(1).is_dir()
    assert run.bracket_dir(1, 1).is_dir()
    assert run.ab_dir("final", 1, "a1").is_dir()
    assert run.matrix_dir(8192, 0, 1).is_dir()
    assert run.matrix_dir(8192, 0).parts[-2:] == ("matrix", "ctx-8192-d-0")
    run.select_bracket(3)
    assert run.calibration_dir("abcdef0123456789", 2).parts[-2:] == ("3", "run-2")
    run.write_text("nested/note.txt", "ok")
    assert (run.dir / "nested/note.txt").read_text() == "ok"
    assert MarathonRun.load(run.dir).dir == run.dir
    assert _entries(run.dir)[-1]["phase"] == "reconnaissance"
    assert find_reentry(opts.sessions_dir, opts, model.fingerprint)[0] == run.dir
    with pytest.raises(MarathonPathError):
        run.ab_dir("final", 1, "invalid")
    with pytest.raises(MarathonPathError):
        run.calibration_dir("not-hex", 1)
    with pytest.raises(MarathonPathError):
        run.write_json("../escape.json", {})
    with pytest.raises(FileNotFoundError):
        MarathonRun.load(tmp_path / "missing")

    run.append({"type": "marathon_end"})
    assert find_reentry(opts.sessions_dir, opts, model.fingerprint)[0] is None


def test_reentry_retains_mismatched_unfinished_run(tmp_path: Path) -> None:
    opts = options(tmp_path)
    hardware, llama, model = reports(tmp_path)
    run = MarathonRun.create(
        opts.sessions_dir,
        options=opts,
        hardware=hardware,
        llama=llama,
        model=model,
        argv=["llamatune", "marathon"],
    )
    meta = json.loads((run.dir / "run.json").read_text())
    meta["identity"] = {"different": True}
    run.write_json("run.json", meta)
    broken = opts.sessions_dir / "marathon" / "broken"
    broken.mkdir()
    match, mismatches = find_reentry(opts.sessions_dir, opts, model.fingerprint)
    assert match is None
    assert mismatches == (run.dir,)


def test_journal_reader_ignores_non_records_and_stops_at_torn_tail(tmp_path: Path) -> None:
    (tmp_path / "journal.jsonl").write_text('1\n{"type":"phase"}\n{"torn"')
    assert _entries(tmp_path) == [{"type": "phase"}]
    assert _entries(tmp_path / "absent") == []


def test_trial_evidence_tolerates_unreadable_and_mixed_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unreadable = tmp_path / "unreadable"
    readable = tmp_path / "readable"

    def entries(path: Path) -> list[dict[str, Any]]:
        if path == unreadable:
            raise OSError("unreadable")
        return [
            {"type": "trial", "trial_id": "executed"},
            {"type": "trial", "trial_id": 7},
            {"type": "prune", "trial_ids": ["pruned"]},
            {"type": "other"},
        ]

    monkeypatch.setattr(marathon_module, "_entries", entries)
    executed, pruned, trials = _trial_evidence((unreadable, readable))
    assert executed == {"executed"}
    assert pruned == {"pruned"}
    assert [trial["trial_id"] for trial in trials] == ["executed"]


def test_run_marathon_success_rebaselines_converges_and_replicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contender = config(flash_attn=True)
    opts = replace(options(tmp_path), dry_run=False, rounds_max=2, converge_rounds=1)
    state = _patch_runtime(
        monkeypatch,
        tmp_path,
        winners=[contender, contender],
        calibration_verdicts=["drift", "drift", "consistent"],
        ab_verdicts=["b", "b"],
        ledger_remaining=0,
        warmup_drift=True,
    )

    outcome = run_marathon(opts, now_fn=AdvancingClock())

    assert outcome.exit_code == 0
    assert outcome.summary["stop_reason"] == "converged"
    assert outcome.summary["verification"]["status"] == "replicated"
    assert len(outcome.summary["reference"]["rebaselines"]) == 2
    assert any("environment unstable" in warning for warning in outcome.summary["warnings"])
    assert state["matrix_calls"] == 1


def test_run_marathon_defaults_reach_round_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opts = replace(options(tmp_path), dry_run=False, rounds_max=1, converge_rounds=2)
    state = _patch_runtime(
        monkeypatch,
        tmp_path,
        winners=[None],
        ledger_remaining=1,
    )

    outcome = run_marathon(opts, now_fn=AdvancingClock())

    assert outcome.exit_code == 1
    assert outcome.summary["stop_reason"] == "rounds_max"
    assert outcome.summary["champion"]["defaults"] is True
    assert outcome.summary["verification"]["status"] == "skipped"
    assert state["matrix_calls"] == 1


def test_run_marathon_defers_round_and_matrix_at_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opts = replace(options(tmp_path), dry_run=False, max_hours=0.1, rounds_max=1)
    state = _patch_runtime(monkeypatch, tmp_path, ledger_remaining=1)

    outcome = run_marathon(opts, now_fn=AdvancingClock())

    assert outcome.exit_code == 1
    assert outcome.summary["stop_reason"] == "deadline"
    assert [item["kind"] for item in outcome.summary["deferred"]] == ["round", "matrix"]
    assert state["matrix_calls"] == 0


def test_run_marathon_round_failure_circuit_breaker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opts = replace(options(tmp_path), dry_run=False, rounds_max=3)
    _patch_runtime(
        monkeypatch,
        tmp_path,
        tune_exit_codes=[3, 3, 3],
        ledger_remaining=1,
    )

    outcome = run_marathon(opts, now_fn=AdvancingClock())

    assert outcome.exit_code == 3
    assert outcome.summary["stop_reason"] == "circuit_breaker"
    assert len(outcome.summary["failed"]) == 3


def test_run_marathon_bracket_circuit_breaker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opts = replace(options(tmp_path), dry_run=False, rounds_max=1)
    _patch_runtime(
        monkeypatch,
        tmp_path,
        winners=[None],
        calibration_verdicts=["error", "error", "error"],
        ledger_remaining=1,
    )

    outcome = run_marathon(opts, now_fn=AdvancingClock())

    assert outcome.exit_code == 3
    assert outcome.summary["stop_reason"] == "bracket_circuit_breaker"
    assert len(outcome.summary["failed"]) == 3


def test_run_marathon_resumes_owned_incomplete_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opts = replace(options(tmp_path), dry_run=False, rounds_max=1)
    state = _patch_runtime(monkeypatch, tmp_path, ledger_remaining=1)
    run = MarathonRun.create(
        opts.sessions_dir,
        options=opts,
        hardware=state["hardware"],
        llama=state["llama"],
        model=state["model"],
        argv=["llamatune", "marathon"],
    )
    meta = json.loads((run.dir / "run.json").read_text())
    meta["identity"] = identity(opts, state["model"].fingerprint)
    run.write_json("run.json", meta)
    owned_session = tmp_path / "owned-round"
    run.append({"type": "round_start", "index": 1, "session_dir": str(owned_session)})

    outcome = run_marathon(opts, now_fn=AdvancingClock())

    assert outcome.run_dir == run.dir
    assert any(entry.get("resumed") is True for entry in _entries(run.dir))
    assert state["sessions"] == []


def test_run_marathon_second_signal_finalizes_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opts = replace(options(tmp_path), dry_run=False, rounds_max=1)

    def interrupt(state: dict[str, Any]) -> None:
        handler = state["handlers"][signal.SIGINT]
        handler(signal.SIGINT, None)
        handler(signal.SIGINT, None)

    _patch_runtime(
        monkeypatch,
        tmp_path,
        winners=[None],
        ledger_remaining=1,
        on_tune=interrupt,
    )

    outcome = run_marathon(opts, now_fn=AdvancingClock())

    assert outcome.exit_code == 4
    assert outcome.summary["stop_reason"] == "interrupted"
    assert _entries(outcome.run_dir)[-2]["immediate"] is True
