from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import pytest

from llamatune import executor, hardware
from llamatune.calibrate import calibrate, run_calibration
from llamatune.executor import CaptureInfo, ExecResult
from llamatune.types import (
    LlamaCppReport,
    ModelReport,
    NightshiftOptions,
    RegistryRecord,
    TrialConfig,
)


def _analysis(path: Path, estimated: float, observed: float) -> None:
    path.mkdir()
    (path / "analysis.json").write_text(
        json.dumps(
            {
                "estimate_vs_observed": {
                    "estimated_total_mb": estimated,
                    "observed_used_delta_mb": observed,
                }
            }
        )
    )


def test_calibrate_uses_robust_total_ratio(tmp_path: Path) -> None:
    for index, observed in enumerate((200.0, 200.0, 300.0)):
        _analysis(tmp_path / f"s{index}", 100.0, observed)
    result = calibrate(tmp_path)
    assert result["samples"] == 3
    assert result["weights_scale"] == 2.0
    assert result["kv_scale"] == 2.0
    assert json.loads((tmp_path / "calibration.json").read_text()) == result


def test_calibrate_sparse_and_corrupt_use_neutral_scale(tmp_path: Path) -> None:
    _analysis(tmp_path / "one", 100.0, 200.0)
    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "analysis.json").write_text("{")
    result = calibrate(tmp_path)
    assert result["samples"] == 1
    assert result["compute_scale"] == 1.0


class _Run:
    def __init__(self, path: Path) -> None:
        self.dir = path
        self.writes: list[tuple[str, dict[str, Any]]] = []

    def calibration_dir(self, fingerprint16: str, n: int) -> Path:
        path = self.dir / "calibrations" / fingerprint16 / f"run-{n}"
        path.mkdir(parents=True)
        return path

    def write_json(self, name: str, payload: dict[str, Any]) -> None:
        self.writes.append((name, payload))


def _config(**updates: Any) -> TrialConfig:
    base = TrialConfig(
        gpu_layers=22,
        moe_cpu_layers=0,
        flash_attn=False,
        ubatch=512,
        batch=2048,
        threads=8,
        mmap=True,
        no_kv_offload=False,
        cache_type_k="f16",
        cache_type_v="f16",
    )
    return dataclasses.replace(base, **updates)


def _record(tmp_path: Path, config: TrialConfig | None = None, **updates: Any) -> RegistryRecord:
    base = RegistryRecord(
        fingerprint="a" * 64,
        session_dir=tmp_path / "session",
        created="2026-01-01T00:00:00+00:00",
        outcome="winner",
        reference_config=config,
        reference_pp=100.0,
        reference_tg=50.0,
        noise_floor_cv=0.01,
        pp_workload=512,
        tg_workload=128,
        reps_confirm=3,
        target="balanced",
        build_commit="old",
        help_sha256="help",
        median_trial_wall_s=1.0,
        session_wall_s=2.0,
    )
    return dataclasses.replace(base, **updates)


def _model(tmp_path: Path, fingerprint: str = "a" * 64) -> ModelReport:
    return ModelReport(
        path=tmp_path / "model.gguf",
        size_bytes=1,
        architecture="qwen",
        n_layer=40,
        ngl_all=41,
        expert_count=8,
        moe=True,
        name="model",
        fingerprint=fingerprint,
        full_sha256=None,
    )


def _llama(capabilities: frozenset[str] = frozenset()) -> LlamaCppReport:
    return LlamaCppReport(
        bench_path=Path("/bin/llama-bench"),
        cli_path=None,
        server_path=None,
        capabilities=capabilities,
        help_sha256="help",
        build_commit="old",
        build_number=1,
    )


def _options(**updates: Any) -> NightshiftOptions:
    base = NightshiftOptions(
        models_dir=Path("models"),
        llama_bin=None,
        sessions_dir=Path("sessions"),
        until=None,
        max_hours=None,
        profile="standard",
        drift_threshold=0.05,
        calibration_runs=2,
        include=(),
        exclude=(),
        duplicates="one",
        dry_run=False,
        target="balanced",
        allow_lossy=False,
        ctx_size=None,
        vram_reserve_mb=None,
        cooldown_s=0.0,
        full_hash=False,
        budget_trials=None,
        reps_search=None,
        reps_confirm=None,
        baseline_runs=None,
    )
    return dataclasses.replace(base, **updates)


def _bench_json(pp: float, tg: float) -> bytes:
    return json.dumps(
        [
            {"n_prompt": 512, "n_gen": 0, "avg_ts": pp, "stddev_ts": 1.0},
            {"n_prompt": 0, "n_gen": 128, "avg_ts": tg, "stddev_ts": 1.0},
        ]
    ).encode()


def _install_executor(
    monkeypatch: pytest.MonkeyPatch, outputs: list[bytes]
) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []

    def fake_run(argv: tuple[str, ...], **kwargs: Any) -> ExecResult:
        calls.append(argv)
        stdout = Path(kwargs["stdout_path"])
        stderr = Path(kwargs["stderr_path"])
        stdout.write_bytes(outputs[len(calls) - 1])
        stderr.write_text("")
        return ExecResult(
            exit_code=0,
            wall_s=1.0,
            timed_out=False,
            stdout=CaptureInfo(stdout, "out", stdout.stat().st_size, False),
            stderr=CaptureInfo(stderr, "err", 0, False),
            started="start",
            ended="end",
            env_names=("PATH",),
        )

    monkeypatch.setattr(executor, "run", fake_run)
    monkeypatch.setattr(hardware, "detect_load", lambda: (1.0, 2.0, 3.0))
    return calls


def test_run_calibration_baseline_aggregates_independent_means(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_executor(monkeypatch, [_bench_json(95.0, 52.0), _bench_json(105.0, 48.0)])
    run = _Run(tmp_path / "night")
    result = run_calibration(run, _record(tmp_path), _model(tmp_path), _llama(), _options())
    assert result.verdict == "consistent"
    assert result.pp is not None and result.pp.mean == 100.0 and result.pp.n == 2
    assert result.tg is not None and result.tg.mean == 50.0
    assert all("-ngl" not in call for call in calls)
    assert calls[0][3:] == ("-p", "512", "-n", "128", "-r", "3", "-o", "json")
    assert result.artifact_dir == tmp_path / "night" / "calibrations" / ("a" * 16)
    assert run.writes[0][1]["load_average"] == [1.0, 2.0, 3.0]


def test_run_calibration_boundary_transfer_and_build_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_executor(monkeypatch, [_bench_json(105.0, 50.0)] * 2)
    record = _record(tmp_path, noise_floor_cv=0.01)
    model = _model(tmp_path, "b" * 64)
    llama = LlamaCppReport(
        bench_path=Path("bench"),
        cli_path=None,
        server_path=None,
        capabilities=frozenset(),
        help_sha256="new",
        build_commit="new",
        build_number=2,
    )
    result = run_calibration(_Run(tmp_path / "night"), record, model, llama, _options())
    assert result.verdict == "consistent"
    assert result.drift_pp == pytest.approx(0.05)
    assert result.transfer_from == "a" * 64
    assert result.build_changed


def test_run_calibration_two_sided_drift_and_deep_cooldown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_executor(monkeypatch, [_bench_json(110.0, 50.0)] * 2)
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    result = run_calibration(
        _Run(tmp_path / "night"),
        _record(tmp_path),
        _model(tmp_path),
        _llama(),
        _options(profile="deep", cooldown_s=None),
    )
    assert result.verdict == "drift"
    assert sleeps == [5.0]


def test_depth_transfer_replay_and_binary_build_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_executor(monkeypatch, [_bench_json(100.0, 50.0)] * 2)
    record = _record(tmp_path, depth_workload=8192, bench_sha256="old-binary")
    model = _model(tmp_path, "b" * 64)
    llama = dataclasses.replace(_llama(frozenset({"d"})), bench_sha256="new-binary")

    result = run_calibration(_Run(tmp_path / "night"), record, model, llama, _options())

    assert result.verdict == "consistent"
    assert result.transfer_from == record.fingerprint
    assert result.build_changed
    assert all(call[call.index("-d") : call.index("-d") + 2] == ("-d", "8192") for call in calls)


def test_depth_capability_loss_prevents_calibration_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor, "run", lambda *_args, **_kwargs: pytest.fail("must not run"))
    result = run_calibration(
        _Run(tmp_path / "night"),
        _record(tmp_path, depth_workload=8192),
        _model(tmp_path),
        _llama(),
        _options(),
    )
    assert result.verdict == "error"
    assert result.reason == "capability_lost:d"
    assert result.runs == 0


def test_configured_depth_replay_emits_depth_and_tuning_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_executor(monkeypatch, [_bench_json(100.0, 50.0)] * 2)
    config = _config(flash_attn=True)
    result = run_calibration(
        _Run(tmp_path / "night"),
        _record(tmp_path, config, depth_workload=4096),
        _model(tmp_path),
        _llama(frozenset({"d", "fa"})),
        _options(),
    )
    assert result.verdict == "consistent"
    for argv in calls:
        assert argv[argv.index("-d") : argv.index("-d") + 2] == ("-d", "4096")
        assert argv[argv.index("-fa") : argv.index("-fa") + 2] == ("-fa", "1")


@pytest.mark.parametrize(
    ("record_hash", "current_hash", "record_help", "current_help", "changed"),
    [
        (None, None, "help", "help", False),
        (None, "new", "help", "help", False),
        ("old", None, "help", "changed-help", True),
        ("old", "new", "help", "help", True),
    ],
)
def test_calibration_build_change_hash_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_hash: str | None,
    current_hash: str | None,
    record_help: str,
    current_help: str,
    changed: bool,
) -> None:
    _install_executor(monkeypatch, [_bench_json(100.0, 50.0)] * 2)
    record = _record(tmp_path, bench_sha256=record_hash, help_sha256=record_help)
    llama = dataclasses.replace(_llama(), bench_sha256=current_hash, help_sha256=current_help)
    result = run_calibration(_Run(tmp_path / "night"), record, _model(tmp_path), llama, _options())
    assert result.build_changed is changed


@pytest.mark.parametrize(
    ("config", "missing"),
    [
        (_config(moe_cpu_layers=4), "ncmoe"),
        (_config(flash_attn=True), "fa"),
        (_config(mmap=False), "mmp"),
        (_config(no_kv_offload=True), "nkvo"),
        (_config(cache_type_k="q8_0"), "ctk"),
        (_config(cache_type_v="q8_0"), "ctv"),
        (_config(threads_batch=4), "tb"),
        (_config(ot_spec="blk.*=CPU"), "ot"),
    ],
)
def test_run_calibration_rejects_lost_required_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: TrialConfig, missing: str
) -> None:
    monkeypatch.setattr(
        executor,
        "run",
        lambda *_args, **_kwargs: pytest.fail("executor must not run"),
    )
    result = run_calibration(
        _Run(tmp_path / "night"), _record(tmp_path, config), _model(tmp_path), _llama(), _options()
    )
    assert result.verdict == "error"
    assert result.reason == f"capability_lost:{missing}"
    assert result.runs == 0


@pytest.mark.parametrize(
    ("timed_out", "truncated", "exit_code", "stderr", "reason"),
    [
        (True, False, None, "", "timeout"),
        (False, True, 0, "", "parse_error"),
        (False, False, 1, "CUDA error", "cuda_error"),
        (False, False, 6, "aborted", "crash"),
    ],
)
def test_run_calibration_classifies_execution_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timed_out: bool,
    truncated: bool,
    exit_code: int | None,
    stderr: str,
    reason: str,
) -> None:
    def fake_run(_argv: tuple[str, ...], **kwargs: Any) -> ExecResult:
        out_path, err_path = Path(kwargs["stdout_path"]), Path(kwargs["stderr_path"])
        out_path.write_bytes(_bench_json(100.0, 50.0))
        err_path.write_text(stderr)
        return ExecResult(
            exit_code=exit_code,
            wall_s=1.0,
            timed_out=timed_out,
            stdout=CaptureInfo(out_path, "o", out_path.stat().st_size, truncated),
            stderr=CaptureInfo(err_path, "e", len(stderr), False),
            started="s",
            ended="e",
            env_names=(),
        )

    monkeypatch.setattr(executor, "run", fake_run)
    result = run_calibration(
        _Run(tmp_path / "night"), _record(tmp_path), _model(tmp_path), _llama(), _options()
    )
    assert result.verdict == "error"
    assert result.reason == reason


@pytest.mark.parametrize(
    ("speed_scale", "expected_verdict", "transfer"),
    [
        (1.0, "consistent", True),
        (0.8, "drift", False),
        (1.2, "drift", False),
    ],
)
def test_run_calibration_executes_fake_bench_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_bench_path: Path,
    tiny_gguf: Path,
    speed_scale: float,
    expected_verdict: str,
    transfer: bool,
) -> None:
    """Exercise executor capture and bench parsing, not mocked drift arithmetic."""
    monkeypatch.setenv("LLAMATUNE_FAKE_SPEED_SCALE", str(speed_scale))
    fingerprint = "b" * 64 if transfer else "a" * 64
    model = dataclasses.replace(
        _model(tmp_path, fingerprint),
        path=tiny_gguf,
    )
    llama = dataclasses.replace(
        _llama(),
        bench_path=fake_bench_path,
    )
    run = _Run(tmp_path / "night")
    record = _record(tmp_path, reference_pp=600.0, reference_tg=40.0)

    result = run_calibration(run, record, model, llama, _options())

    assert result.verdict == expected_verdict
    assert result.runs == 2
    assert result.pp is not None and result.pp.n == 2
    assert result.tg is not None and result.tg.n == 2
    assert result.transfer_from == ("a" * 64 if transfer else None)
    assert len(run.writes) == 2
    for number in (1, 2):
        artifact = result.artifact_dir / f"run-{number}" if result.artifact_dir else None
        assert artifact is not None
        assert (artifact / "stdout.json").is_file()
        assert (artifact / "stderr.log").is_file()
    if speed_scale != 1.0:
        assert result.drift_pp is not None and result.drift_pp > result.threshold
        assert result.drift_tg is not None and result.drift_tg > result.threshold
