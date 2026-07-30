from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from llamatune import marathon as marathon_module
from llamatune.bench import BenchSample
from llamatune.marathon import (
    MarathonRun,
    _bracket_options,
    _plan,
    _profile,
    _recon,
    ab_reserved_minutes,
    coverage_complete,
    coverage_percent,
    has_converged,
    identity,
    identity_matches,
    remaining_minutes,
    resolve_deadline,
    round_budget,
    round_fits,
)
from llamatune.types import CoverageLedger, MarathonOptions, TierCoverage


def options() -> MarathonOptions:
    return MarathonOptions(
        model_path=Path("m.gguf"),
        llama_bin=None,
        sessions_dir=Path("sessions"),
        until=None,
        max_hours=None,
        rounds_max=6,
        converge_rounds=2,
        ab_blocks=5,
        depth_grid=(0, 8192),
        ctx_size=8192,
        ctx_ladder=(8192, 32768),
        matrix_refine=True,
        drift_threshold=0.05,
        dry_run=False,
        target="balanced",
        allow_lossy=False,
        vram_reserve_mb=None,
        cooldown_s=0,
        full_hash=False,
        pp=512,
        tg=128,
        quality_corpus=None,
        ot_search=False,
        budget_trials=None,
        reps_search=None,
        reps_confirm=None,
        baseline_runs=None,
    )


def ledger(remaining: int) -> CoverageLedger:
    tier = TierCoverage(
        enumerated=10,
        executed=10 - remaining,
        pruned=0,
        remaining=remaining,
        remaining_ids=tuple(str(n) for n in range(remaining)),
    )
    return CoverageLedger(tiers=dict.fromkeys("ABC", tier), responsive=())


def test_escalation_and_cap() -> None:
    assert [round_budget(240, n) for n in range(1, 6)] == [240, 360, 540, 810, 1000]
    with pytest.raises(ValueError):
        round_budget(240, 0)


def test_convergence_requires_coverage_and_streak() -> None:
    assert coverage_complete(ledger(0))
    assert has_converged(2, 2, ledger(0))
    assert not has_converged(1, 2, ledger(0))
    assert not has_converged(2, 2, ledger(1))


def test_deadline_and_reservation() -> None:
    assert round_fits(None)
    assert round_fits(55, ab_reserved=20)
    assert not round_fits(54.9, ab_reserved=20)
    start = datetime(2026, 1, 1, 23, 0, tzinfo=UTC)
    assert resolve_deadline(start, "22:00", None, local_tz=UTC) == start + timedelta(hours=23)
    assert resolve_deadline(start, "23:30", None, local_tz=UTC) == start + timedelta(minutes=30)
    assert resolve_deadline(start, None, 2) == start + timedelta(hours=2)


def test_reentry_identity_is_exact() -> None:
    value = identity(options(), "abc")
    assert identity_matches({"identity": value}, options(), "abc")
    changed = {**value, "tg": 64}
    assert not identity_matches({"identity": changed}, options(), "abc")
    assert not identity_matches({}, options(), "abc")


def test_plan_profile_and_remaining_budget() -> None:
    opts = options()
    plan = _plan(opts)
    assert plan["round_budgets"][:3] == [240, 360, 540]
    assert len(plan["matrix_cells"]) == 4
    profile = _profile(opts, 2, 90.0)
    assert profile.budget_trials == 360
    assert profile.budget_minutes == 85.0
    assert profile.reps_search == 8
    assert profile.reps_confirm == 12
    assert profile.baseline_runs == 5
    assert profile.cooldown_s == 0
    assert _profile(opts, 1, None).budget_minutes is None
    bracket = _bracket_options(opts)
    assert bracket.profile == "deep"
    assert bracket.calibration_runs == 3
    assert bracket.ctx_ladder == opts.ctx_ladder


def test_coverage_deadline_and_ab_helpers() -> None:
    assert coverage_percent(ledger(0)) == 100.0
    assert coverage_percent(CoverageLedger(tiers={}, responsive=())) == 100.0
    assert coverage_percent(ledger(4)) == 60.0
    start = datetime(2026, 1, 1, tzinfo=UTC)
    assert remaining_minutes(None, start) is None
    assert remaining_minutes(start + timedelta(minutes=7), start) == 7.0
    assert remaining_minutes(start - timedelta(minutes=1), start) == 0.0
    assert ab_reserved_minutes(5) == 50.0


def test_recon_success_and_failed_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from llamatune import bench, executor
    from llamatune.types import LlamaCppReport, ModelReport

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run = MarathonRun(run_dir)
    opts = options()
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
    llama = LlamaCppReport(
        bench_path=tmp_path / "llama-bench",
        cli_path=None,
        server_path=None,
        capabilities=frozenset(),
        help_sha256="help",
        build_commit=None,
        build_number=None,
    )
    sample = BenchSample(
        pp_avg=100.0,
        pp_stddev=1.0,
        tg_avg=50.0,
        tg_stddev=0.5,
        pp_entry={},
        tg_entry={},
        config_fields={},
    )
    monkeypatch.setattr(bench, "build_baseline_argv", lambda **kwargs: ("llama-bench",))
    monkeypatch.setattr(bench, "parse_bench_output", lambda raw: sample)
    monkeypatch.setattr(marathon_module, "COOLDOWN_S", 0.0)
    monkeypatch.setattr(
        bench, "resolved_config", lambda entry: SimpleNamespace(to_dict=config_dict)
    )

    def config_dict() -> dict[str, object]:
        return {
            "gpu_layers": 0,
            "moe_cpu_layers": 0,
            "flash_attn": False,
            "ubatch": 512,
            "batch": 2048,
            "threads": 1,
            "mmap": True,
            "no_kv_offload": False,
            "cache_type_k": "f16",
            "cache_type_v": "f16",
        }

    def successful_run(
        argv: object, *, stdout_path: Path, stderr_path: Path, **kwargs: object
    ) -> SimpleNamespace:
        stdout_path.write_bytes(b"[]")
        return SimpleNamespace(exit_code=0, timed_out=False)

    monkeypatch.setattr(executor, "run", successful_run)
    reference = _recon(run, model, llama, opts)
    assert reference is not None
    assert reference["runs"] == 10
    assert reference["default_config"]["threads"] == 1

    failed_dir = tmp_path / "failed"
    failed_dir.mkdir()
    monkeypatch.setattr(
        executor,
        "run",
        lambda *args, **kwargs: SimpleNamespace(exit_code=1, timed_out=False),
    )
    assert _recon(MarathonRun(failed_dir), model, llama, opts) is None
