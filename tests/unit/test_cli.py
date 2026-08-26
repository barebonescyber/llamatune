"""Unit tests for llamatune.cli (Typer application)."""

from __future__ import annotations

import errno
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer
from typer.main import get_command
from typer.testing import CliRunner

from llamatune._version import __version__
from llamatune.cli import (
    _emit_tune_outcome,
    _guard_internal_errors,
    _parse_ctx_ladder,
    _parse_depth_profile,
    app,
)
from llamatune.session import Session
from llamatune.types import GPUInfo, HardwareReport, LlamaCppReport, TuneOutcome

runner = CliRunner()


def _fake_hardware() -> HardwareReport:
    return HardwareReport(
        os_name="Linux",
        arch="x86_64",
        cpu_model="Fake CPU",
        physical_cores=8,
        logical_cores=16,
        perf_cores=None,
        ram_mb=32768,
        gpus=(GPUInfo(vendor="nvidia", name="Fake GPU", vram_mb=24000, method="nvidia-smi"),),
        warnings=(),
    )


def _fake_llama(bench_path: Path) -> LlamaCppReport:
    return LlamaCppReport(
        bench_path=bench_path,
        cli_path=None,
        server_path=None,
        capabilities=frozenset({"fa", "mmp"}),
        help_sha256="deadbeef",
        build_commit=None,
        build_number=None,
    )


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "scan",
        "tune",
        "resume",
        "report",
        "export",
        "nightshift",
        "marathon",
        "sessions",
        "best",
        "revalidate",
        "matrix",
        "quality",
        "calibrate",
    ):
        assert command in result.output


@pytest.mark.parametrize("command", ["export", "sessions", "best", "revalidate", "calibrate"])
def test_stage3_command_help(command: str) -> None:
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0


def test_tune_help_lists_key_options() -> None:
    result = runner.invoke(app, ["tune", "--help"])
    assert result.exit_code == 0
    root: Any = get_command(app)
    tune_command = root.commands["tune"]
    option_names = {
        option
        for parameter in tune_command.params
        for option in (*getattr(parameter, "opts", ()), *getattr(parameter, "secondary_opts", ()))
    }
    for option in (
        "--llama-bin",
        "--sessions-dir",
        "--target",
        "--budget-trials",
        "--budget-minutes",
        "--baseline-runs",
        "--reps-search",
        "--reps-confirm",
        "--pp",
        "--tg",
        "--depth",
        "--depth-profile",
        "--ctx-size",
        "--vram-reserve-mb",
        "--initial-gpu-layers",
        "--max-gpu-layers",
        "--initial-cpu-moe",
        "--allow-lossy",
        "--cooldown",
        "--baseline-only",
        "--full-hash",
        "--json",
        "--progress",
        "--tui",
        "--quiet",
        "--allow-core-dumps",
        "--quiet-wait-s",
        "--quiet-load",
        "--observe-vram",
        "--validate-with-cli",
        "--dry-run",
        "--quality-corpus",
        "--batched-trials",
        "--no-batched-trials",
        "--ot-search",
    ):
        assert option in option_names


def test_scan_human_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llamatune.hardware.assess_hardware", _fake_hardware)
    monkeypatch.setattr(
        "llamatune.llama.discover_llama",
        lambda llama_bin: _fake_llama(Path("/usr/bin/llama-bench")),
    )

    result = runner.invoke(app, ["scan"])
    assert result.exit_code == 0
    assert "Fake CPU" in result.output
    assert "Fake GPU" in result.output
    assert "llama-bench" in result.output


def test_scan_json_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llamatune.hardware.assess_hardware", _fake_hardware)
    monkeypatch.setattr(
        "llamatune.llama.discover_llama",
        lambda llama_bin: _fake_llama(Path("/usr/bin/llama-bench")),
    )

    result = runner.invoke(app, ["scan", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["hardware"]["cpu_model"] == "Fake CPU"
    assert payload["llama"]["capabilities"]
    assert payload["llama_error"] is None


def test_scan_reports_missing_llama_bench_without_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    from llamatune.llama import LlamaDiscoveryError

    monkeypatch.setattr("llamatune.hardware.assess_hardware", _fake_hardware)

    def _raise(llama_bin: Path | None) -> Any:
        raise LlamaDiscoveryError("llama-bench not found in PATH")

    monkeypatch.setattr("llamatune.llama.discover_llama", _raise)

    result = runner.invoke(app, ["scan"])
    assert result.exit_code == 0
    assert "NOT FOUND" in result.output


def test_scan_json_reports_missing_llama_bench(monkeypatch: pytest.MonkeyPatch) -> None:
    from llamatune.llama import LlamaDiscoveryError

    monkeypatch.setattr("llamatune.hardware.assess_hardware", _fake_hardware)

    def _raise(llama_bin: Path | None) -> Any:
        raise LlamaDiscoveryError("llama-bench not found in PATH")

    monkeypatch.setattr("llamatune.llama.discover_llama", _raise)

    result = runner.invoke(app, ["scan", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["llama"] is None
    assert "not found" in payload["llama_error"]


def test_tune_rejects_baseline_runs_below_minimum() -> None:
    result = runner.invoke(app, ["tune", "model.gguf", "--baseline-runs", "1"])
    assert result.exit_code == 2


def test_tune_rejects_budget_smaller_than_baseline() -> None:
    result = runner.invoke(
        app,
        ["tune", "model.gguf", "--baseline-runs", "3", "--budget-trials", "2"],
    )
    assert result.exit_code == 2
    assert "--budget-trials must be >= --baseline-runs" in result.stderr


def test_context_and_depth_csv_parsing() -> None:
    assert _parse_ctx_ladder(None) == (None, ())
    assert _parse_ctx_ladder("32768") == (32768, ())
    assert _parse_ctx_ladder("32768,49152,65536") == (32768, (49152, 65536))
    assert _parse_depth_profile(None) is None
    assert _parse_depth_profile("0,8192,32768") == (0, 8192, 32768)
    for value in ("", "0", "8192,8192", "8192,4096"):
        with pytest.raises(ValueError):
            _parse_ctx_ladder(value)
    with pytest.raises(ValueError, match="comma-separated list of integers"):
        _parse_depth_profile("")
    with pytest.raises(ValueError, match="values must be >= 0"):
        _parse_depth_profile("-1,0")


def test_tune_rejects_tui_and_quiet() -> None:
    result = runner.invoke(app, ["tune", "model.gguf", "--tui", "--quiet"])
    assert result.exit_code == 2
    assert "mutually exclusive" in result.stderr


def test_resume_rejects_tui_and_quiet(tmp_path: Path) -> None:
    result = runner.invoke(app, ["resume", str(tmp_path), "--tui", "--quiet"])
    assert result.exit_code == 2
    assert "mutually exclusive" in result.stderr


def test_tune_missing_llama_bench_exits_3(tmp_path: Path) -> None:
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    result = runner.invoke(app, ["tune", "nonexistent.gguf", "--llama-bin", str(empty_bin)])
    assert result.exit_code == 3
    assert "llama-bench not found" in result.stderr
    assert (
        "Pass --llama-bin DIR as the directory containing the binaries, "
        "not the binary itself." in result.stderr
    )
    assert "Traceback" not in result.output


def test_tune_missing_model_exits_3(fake_bin_dir: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "tune",
            str(tmp_path / "nonexistent.gguf"),
            "--llama-bin",
            str(fake_bin_dir),
        ],
    )
    assert result.exit_code == 3
    assert "model file not found" in result.stderr
    assert "Traceback" not in result.output


def test_tune_startup_failures_have_machine_readable_contract(
    fake_bin_dir: Path, tmp_path: Path
) -> None:
    missing_executable = runner.invoke(
        app,
        [
            "tune",
            str(tmp_path / "missing.gguf"),
            "--llama-bin",
            str(tmp_path / "missing-bin"),
            "--json",
        ],
    )
    assert missing_executable.exit_code == 3
    executable_payload = json.loads(missing_executable.stdout)
    assert executable_payload["failure_stage"] == "llama_discovery"
    assert executable_payload["session_dir"] is None
    assert executable_payload["resumable"] is False

    missing_model = runner.invoke(
        app,
        [
            "tune",
            str(tmp_path / "missing.gguf"),
            "--llama-bin",
            str(fake_bin_dir),
            "--json",
        ],
    )
    assert missing_model.exit_code == 3
    model_payload = json.loads(missing_model.stdout)
    assert model_payload["failure_stage"] == "model_inspection"
    assert model_payload["session_dir"] is None
    assert model_payload["resumable"] is False


def test_tune_baseline_only_via_cli(fake_bin_dir: Path, tiny_gguf: Path, tmp_path: Path) -> None:
    # End-to-end CLI wiring check against the implemented engine: a
    # baseline-only run completes with exit 0 and writes its evidence.
    sessions = tmp_path / "sessions"
    result = runner.invoke(
        app,
        [
            "tune",
            str(tiny_gguf),
            "--llama-bin",
            str(fake_bin_dir),
            "--sessions-dir",
            str(sessions),
            "--baseline-only",
        ],
    )
    assert result.exit_code == 0
    session_dirs = [path for path in sessions.iterdir() if (path / "session.json").is_file()]
    assert len(session_dirs) == 1
    analysis = json.loads((session_dirs[0] / "analysis.json").read_text())
    assert analysis["baseline"]["runs"] == 3
    assert analysis["winner"] is None


def test_tune_new_options_round_trip(
    fake_bin_dir: Path,
    tiny_gguf: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("llamatune.hardware.assess_hardware", _fake_hardware)
    sessions = tmp_path / "sessions"
    result = runner.invoke(
        app,
        [
            "tune",
            str(tiny_gguf),
            "--llama-bin",
            str(fake_bin_dir),
            "--sessions-dir",
            str(sessions),
            "--baseline-only",
            "--ctx-size",
            "8192",
            "--vram-reserve-mb",
            "1536",
            "--initial-gpu-layers",
            "2",
            "--max-gpu-layers",
            "auto",
            "--initial-cpu-moe",
            "32",
            "--allow-core-dumps",
            "--quiet-wait-s",
            "30",
            "--quiet-load",
            "2.5",
            "--no-observe-vram",
            "--validate-with-cli",
            "--quality-corpus",
            str(tmp_path / "quality.txt"),
            "--no-batched-trials",
            "--ot-search",
            "--multi-gpu",
            "--thermal-wait-cap-s",
            "0",
        ],
    )
    assert result.exit_code == 0
    session_dir = next(path for path in sessions.iterdir() if (path / "session.json").is_file())
    session_meta = json.loads((session_dir / "session.json").read_text())
    options = session_meta["options"]
    assert options["ctx_size"] == 8192
    assert options["vram_reserve_mb"] == 1536
    assert options["initial_gpu_layers"] == 2
    assert options["max_gpu_layers"] is None
    assert options["initial_cpu_moe"] == 32
    assert options["allow_core_dumps"] is True
    assert options["quiet_wait_s"] == 30.0
    assert options["quiet_load"] == 2.5
    assert options["observe_vram"] is False
    assert options["validate_with_cli"] is True
    assert options["quality_corpus"] == str(tmp_path / "quality.txt")
    assert options["batched_trials"] is False
    assert options["ot_search"] is True
    assert options["multi_gpu"] is True


def test_export_command_and_unknown_format(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    recommended = {
        "config": {
            "gpu_layers": 10,
            "moe_cpu_layers": 0,
            "flash_attn": False,
            "ubatch": 512,
            "batch": 2048,
            "threads": 8,
            "mmap": True,
            "no_kv_offload": False,
            "cache_type_k": "f16",
            "cache_type_v": "f16",
        },
        "model": {"path": "/models/model.gguf"},
    }
    (session / "recommended.json").write_text(json.dumps(recommended))
    (session / "session.json").write_text(json.dumps({"options": {"ctx_size": 4096}}))
    result = runner.invoke(app, ["export", str(session), "--format", "llama-cli"])
    assert result.exit_code == 0
    assert "llama-cli -m /models/model.gguf" in result.output
    bad = runner.invoke(app, ["export", str(session), "--format", "unknown"])
    assert bad.exit_code == 2


def test_sessions_command_json_tolerates_empty_dir(tmp_path: Path) -> None:
    result = runner.invoke(app, ["sessions", str(tmp_path), "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == []


def test_sessions_command_human_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "llamatune.session.list_sessions",
        lambda path: [
            {
                "session_dir": "/sessions/a",
                "model": "model",
                "status": "complete",
                "exit_code": 0,
                "winner_trial_id": "winner",
                "confirmed": True,
            }
        ],
    )
    result = runner.invoke(app, ["sessions", str(tmp_path)])
    assert result.exit_code == 0
    assert "model=model status=complete exit=0 winner=winner confirmed=True" in result.output


def test_sessions_command_preserves_windows_drive_path(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def capture(path: Path) -> list[dict[str, Any]]:
        seen.append(str(path))
        return []

    monkeypatch.setattr("llamatune.session.list_sessions", capture)
    result = runner.invoke(app, ["sessions", r"C:\llamatune-sessions"])
    assert result.exit_code == 0
    assert seen == [r"C:\llamatune-sessions"]


@pytest.mark.parametrize("json_output", [False, True])
def test_best_command_hit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, json_output: bool
) -> None:
    record = {"session_dir": "/sessions/a", "config": {"threads": 8}}
    monkeypatch.setattr("llamatune.model.inspect_model", lambda path: SimpleNamespace())
    monkeypatch.setattr("llamatune.llama.discover_llama", lambda path: SimpleNamespace())
    monkeypatch.setattr("llamatune.hardware.assess_hardware", _fake_hardware)
    monkeypatch.setattr("llamatune.config.hardware_signature", lambda hardware: ("hardware",))
    monkeypatch.setattr(
        "llamatune.registry.lookup",
        lambda *args, **kwargs: {"status": "hit", "record": record, "stale_reasons": []},
    )
    args = ["best", str(tmp_path / "model.gguf")]
    if json_output:
        args.append("--json")
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    if json_output:
        assert json.loads(result.output)["status"] == "hit"
    else:
        assert "match: /sessions/a" in result.output


def test_best_command_stale_exits_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("llamatune.model.inspect_model", lambda path: SimpleNamespace())
    monkeypatch.setattr("llamatune.llama.discover_llama", lambda path: SimpleNamespace())
    monkeypatch.setattr("llamatune.hardware.assess_hardware", _fake_hardware)
    monkeypatch.setattr("llamatune.config.hardware_signature", lambda hardware: ())
    monkeypatch.setattr(
        "llamatune.registry.lookup",
        lambda *args, **kwargs: {
            "status": "stale",
            "record": {},
            "stale_reasons": ["llama.cpp build changed"],
        },
    )
    result = runner.invoke(app, ["best", str(tmp_path / "model.gguf")])
    assert result.exit_code == 1
    assert "stale: llama.cpp build changed" in result.output


def test_best_command_threads_requested_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    def capture(*args: Any, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"status": "miss", "record": None, "stale_reasons": []}

    monkeypatch.setattr("llamatune.model.inspect_model", lambda path: SimpleNamespace())
    monkeypatch.setattr("llamatune.llama.discover_llama", lambda path: SimpleNamespace())
    monkeypatch.setattr("llamatune.hardware.assess_hardware", _fake_hardware)
    monkeypatch.setattr("llamatune.config.hardware_signature", lambda hardware: ())
    monkeypatch.setattr("llamatune.registry.lookup", capture)

    result = runner.invoke(app, ["best", str(tmp_path / "model.gguf"), "--ctx-size", "65536"])
    assert result.exit_code == 1
    assert seen["ctx_size"] == 65536


@pytest.mark.parametrize("status,exit_code", [("reproduced", 0), ("regressed", 1)])
def test_revalidate_command_real_result_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str, exit_code: int
) -> None:
    outcome = TuneOutcome(
        session_dir=tmp_path,
        analysis={
            "status": status,
            "previous_score": 1.2,
            "current_score": 1.1,
            "noise_floor_cv": 0.02,
        },
        exit_code=exit_code,
    )
    monkeypatch.setattr("llamatune.search.revalidate_session", lambda path: outcome)
    result = runner.invoke(app, ["revalidate", str(tmp_path)])
    assert result.exit_code == exit_code
    assert f"revalidation: {status}" in result.output
    assert "previous=1.2000 current=1.1000" in result.output


def test_revalidate_command_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    outcome = TuneOutcome(session_dir=tmp_path, analysis={"status": "reproduced"}, exit_code=0)
    monkeypatch.setattr("llamatune.search.revalidate_session", lambda path: outcome)
    result = runner.invoke(app, ["revalidate", str(tmp_path), "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {"status": "reproduced"}


def test_revalidate_failure_has_human_and_json_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outcome = TuneOutcome(
        session_dir=tmp_path / "session",
        analysis={},
        exit_code=3,
        failure_stage="revalidation",
        failure_reason="session has no confirmed winner",
    )
    monkeypatch.setattr("llamatune.search.revalidate_session", lambda path: outcome)

    human = runner.invoke(app, ["revalidate", str(outcome.session_dir)])
    assert human.exit_code == 3
    assert "revalidation failed: revalidation: session has no confirmed winner" in human.stderr
    assert "resumable: no" in human.stderr
    assert "revalidation: unknown" not in human.output
    assert "revalidation failed" not in human.stdout

    machine = runner.invoke(app, ["revalidate", str(outcome.session_dir), "--json"])
    assert machine.exit_code == 3
    payload = json.loads(machine.stdout)
    assert payload["failure_stage"] == "revalidation"
    assert payload["failure_reason"] == "session has no confirmed winner"
    assert payload["resumable"] is False


def test_calibrate_command_outputs_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "samples": 3,
        "weights_scale": 1.1,
        "kv_scale": 1.2,
        "compute_scale": 1.3,
    }
    monkeypatch.setattr("llamatune.calibrate.calibrate", lambda path: payload)
    result = runner.invoke(app, ["calibrate", "--sessions-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert result.output.startswith("experimental calibration written:")
    assert "samples=3 weights=1.1000 kv=1.2000 compute=1.3000" in result.output


def test_dry_run_prints_plan_without_creating_session(
    fake_bin_dir: Path,
    tiny_gguf: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("llamatune.hardware.assess_hardware", _fake_hardware)
    sessions = tmp_path / "sessions"
    result = runner.invoke(
        app,
        [
            "tune",
            str(tiny_gguf),
            "--llama-bin",
            str(fake_bin_dir),
            "--sessions-dir",
            str(sessions),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "llamatune dry-run search plan" in result.output
    assert not sessions.exists()


def test_dry_run_respects_zero_gpu_layer_hard_cap(
    fake_bin_dir: Path,
    tiny_gguf: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("llamatune.hardware.assess_hardware", _fake_hardware)
    sessions = tmp_path / "sessions"
    result = runner.invoke(
        app,
        [
            "tune",
            str(tiny_gguf),
            "--llama-bin",
            str(fake_bin_dir),
            "--sessions-dir",
            str(sessions),
            "--initial-gpu-layers",
            "0",
            "--max-gpu-layers",
            "0",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "  gpu_layers:" not in result.output
    assert not sessions.exists()


def test_exit_3_outcome_renders_on_stderr_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _emit_tune_outcome(
        TuneOutcome(
            session_dir=tmp_path / "session",
            analysis={},
            exit_code=3,
            failure_stage="baseline",
            failure_reason="default probe and -ngl 0 fallback both failed",
        ),
        json_output=False,
    )
    captured = capsys.readouterr()
    assert "tuning did not start: baseline:" in captured.err
    assert "default probe and -ngl 0 fallback both failed" in captured.err
    assert "evidence:" in captured.err
    assert "resumable: no" in captured.err
    assert "no confirmed improvement over baseline" not in captured.err
    assert captured.out == ""


def test_exit_2_outcome_reports_unreadable_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _emit_tune_outcome(
        TuneOutcome(
            session_dir=tmp_path / "session",
            analysis={},
            exit_code=2,
            failure_stage="session",
            failure_reason="journal.jsonl is corrupt",
        ),
        json_output=False,
    )
    captured = capsys.readouterr()
    assert "tuning could not continue: session: journal.jsonl is corrupt" in captured.err
    assert "evidence: unavailable or unreadable" in captured.err
    assert "resumable: no" in captured.err
    assert "no confirmed improvement over baseline" not in captured.err
    assert captured.out == ""


def test_success_outcome_renders_on_stdout_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _emit_tune_outcome(
        TuneOutcome(
            session_dir=tmp_path / "session",
            analysis={"winner": {"trial_id": "w1"}},
            exit_code=0,
        ),
        json_output=False,
    )
    captured = capsys.readouterr()
    assert "winner: w1" in captured.out
    assert "exit_code: 0" in captured.out
    assert captured.err == ""


@pytest.mark.parametrize("exit_code,resumable", [(2, False), (3, False), (4, True)])
def test_failed_tune_json_has_machine_readable_contract(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    exit_code: int,
    resumable: bool,
) -> None:
    _emit_tune_outcome(
        TuneOutcome(
            session_dir=tmp_path / "session",
            analysis={},
            exit_code=exit_code,
            failure_stage="test-stage",
            failure_reason="test failure",
        ),
        json_output=True,
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "status": "failed",
        "exit_code": exit_code,
        "session_dir": str(tmp_path / "session"),
        "failure_stage": "test-stage",
        "failure_reason": "test failure",
        "resumable": resumable,
    }


def test_exit_4_evidence_failure_has_resumability_explanation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _emit_tune_outcome(
        TuneOutcome(
            session_dir=tmp_path / "session",
            analysis={},
            exit_code=4,
            failure_stage="evidence",
            failure_reason="[Errno 28] No space left on device",
        ),
        json_output=False,
    )
    captured = capsys.readouterr()
    assert "tuning stopped with a resumable session: evidence:" in captured.err
    assert "No space left on device" in captured.err
    assert "evidence:" in captured.err
    assert "resumable: yes" in captured.err
    assert captured.out == ""


def test_tune_invalid_sessions_dir_exits_2(
    fake_bin_dir: Path,
    tiny_gguf: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("llamatune.hardware.assess_hardware", _fake_hardware)
    regular_file = tmp_path / "f"
    regular_file.write_text("not a directory")

    result = runner.invoke(
        app,
        [
            "tune",
            str(tiny_gguf),
            "--llama-bin",
            str(fake_bin_dir),
            "--sessions-dir",
            str(regular_file / "sub"),
        ],
    )

    assert result.exit_code == 2
    assert "error: could not create session directory under" in result.stderr
    assert "Traceback" not in result.output


def test_tune_insufficient_disk_space_exits_2_without_traceback(
    fake_bin_dir: Path,
    tiny_gguf: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("llamatune.hardware.assess_hardware", _fake_hardware)

    def _disk_full(*_args: Any, **_kwargs: Any) -> Session:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(Session, "create", classmethod(_disk_full))
    sessions = tmp_path / "sessions"
    result = runner.invoke(
        app,
        [
            "tune",
            str(tiny_gguf),
            "--llama-bin",
            str(fake_bin_dir),
            "--sessions-dir",
            str(sessions),
        ],
    )

    assert result.exit_code == 2
    assert "No space left on device" in result.stderr
    assert "Traceback" not in result.output


def test_tune_insufficient_disk_space_json_has_no_evidence(
    fake_bin_dir: Path,
    tiny_gguf: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("llamatune.hardware.assess_hardware", _fake_hardware)

    def _disk_full(*_args: Any, **_kwargs: Any) -> Session:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(Session, "create", classmethod(_disk_full))
    result = runner.invoke(
        app,
        [
            "tune",
            str(tiny_gguf),
            "--llama-bin",
            str(fake_bin_dir),
            "--sessions-dir",
            str(tmp_path / "sessions"),
            "--json",
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["failure_stage"] == "session_creation"
    assert payload["session_dir"] is None
    assert payload["resumable"] is False
    assert "No space left on device" in payload["failure_reason"]


def test_resume_unreadable_session_exits_2_with_contract(tmp_path: Path) -> None:
    result = runner.invoke(app, ["resume", str(tmp_path / "no-such-session")])
    assert result.exit_code == 2
    assert "tuning could not continue: session:" in result.stderr
    assert "resumable: no" in result.stderr
    assert "no confirmed improvement over baseline" not in result.output
    assert "tuning could not continue" not in result.stdout

    json_result = runner.invoke(app, ["resume", str(tmp_path / "no-such-session"), "--json"])
    assert json_result.exit_code == 2
    payload = json.loads(json_result.stdout)
    assert payload["exit_code"] == 2
    assert payload["failure_stage"] == "session"
    assert payload["resumable"] is False


@pytest.mark.parametrize(
    "args",
    [
        ["--ctx-size", "0"],
        ["--vram-reserve-mb", "-1"],
        ["--initial-gpu-layers", "-1"],
        ["--max-gpu-layers", "nope"],
        ["--initial-cpu-moe", "-1"],
        ["--initial-gpu-layers", "25", "--max-gpu-layers", "24"],
        ["--quiet-wait-s", "-1"],
        ["--quiet-load", "0"],
        ["--depth", "-1"],
        ["--depth-profile", "0,8192,8192"],
        ["--depth-profile", "8192,0"],
        ["--depth-profile", "0,nope"],
        ["--thermal-threshold-c", "0"],
        ["--thermal-wait-cap-s", "-1"],
    ],
)
def test_tune_rejects_invalid_new_options(tiny_gguf: Path, args: list[str]) -> None:
    result = runner.invoke(app, ["tune", str(tiny_gguf), *args])
    assert result.exit_code == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.output


def test_depth_requires_capability_before_session_creation(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLAMATUNE_FAKE_NO_DEPTH", "1")
    sessions = tmp_path / "sessions"
    result = runner.invoke(
        app,
        [
            "tune",
            str(tiny_gguf),
            "--llama-bin",
            str(fake_bin_dir),
            "--sessions-dir",
            str(sessions),
            "--depth",
            "8192",
        ],
    )
    assert result.exit_code == 2
    assert "does not support -d/--n-depth" in result.stderr
    assert not sessions.exists()


def test_depth_profile_requires_capability_before_session_creation(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLAMATUNE_FAKE_NO_DEPTH", "1")
    sessions = tmp_path / "profile-sessions"
    result = runner.invoke(
        app,
        [
            "tune",
            str(tiny_gguf),
            "--llama-bin",
            str(fake_bin_dir),
            "--sessions-dir",
            str(sessions),
            "--depth-profile",
            "0,8192",
        ],
    )
    assert result.exit_code == 2
    assert "does not support -d/--n-depth" in result.stderr
    assert not sessions.exists()


def test_unsupported_depth_json_has_no_evidence_or_resumability(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLAMATUNE_FAKE_NO_DEPTH", "1")
    sessions = tmp_path / "json-depth-sessions"
    result = runner.invoke(
        app,
        [
            "tune",
            str(tiny_gguf),
            "--llama-bin",
            str(fake_bin_dir),
            "--sessions-dir",
            str(sessions),
            "--depth",
            "8192",
            "--json",
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["failure_stage"] == "capability"
    assert payload["session_dir"] is None
    assert payload["resumable"] is False
    assert "does not support -d/--n-depth" in payload["failure_reason"]
    assert not sessions.exists()


def _write_session_files(session_dir: Path) -> None:
    session_dir.mkdir(parents=True)
    (session_dir / "analysis.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "target": "balanced",
                "baseline": {
                    "runs": 3,
                    "pp": {"mean": 500.0, "stdev": 1.0, "cv": 0.002, "n": 3},
                    "tg": {"mean": 30.0, "stdev": 0.5, "cv": 0.016, "n": 3},
                    "noise_floor_cv": 0.01,
                    "fallback": None,
                    "resolved_defaults": {},
                },
                "counts": {
                    "executed": 0,
                    "ok": 0,
                    "unstable": 0,
                    "oom": 0,
                    "timeout": 0,
                    "crash": 0,
                    "parse_error": 0,
                    "pruned": 0,
                },
                "pareto": [],
                "top": [],
                "winner": None,
                "lossless_winner": None,
                "warnings": [],
            }
        )
    )
    (session_dir / "session.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tool_version": "0.1.0",
                "argv": ["llamatune", "tune", "tiny.gguf"],
                "created": "2026-07-14T00:00:00+00:00",
                "options": {
                    "target": "balanced",
                    "budget_trials": 60,
                    "budget_minutes": None,
                    "reps_search": 3,
                    "reps_confirm": 5,
                    "baseline_runs": 3,
                    "pp": 512,
                    "tg": 128,
                    "allow_lossy": False,
                    "cooldown_s": 0.0,
                    "baseline_only": False,
                    "llama_bin": None,
                    "sessions_dir": str(session_dir.parent),
                    "full_hash": False,
                },
            }
        )
    )
    (session_dir / "hardware.json").write_text(
        json.dumps(
            {
                "os_name": "Linux",
                "arch": "x86_64",
                "cpu_model": "Fake CPU",
                "physical_cores": 8,
                "logical_cores": 16,
                "perf_cores": None,
                "ram_mb": 32768,
                "gpus": [],
                "warnings": [],
            }
        )
    )
    (session_dir / "model.json").write_text(
        json.dumps(
            {
                "path": "/models/tiny.gguf",
                "size_bytes": 4,
                "architecture": "llama",
                "n_layer": 32,
                "ngl_all": 33,
                "expert_count": 0,
                "moe": False,
                "name": "tiny",
                "fingerprint": "abc",
                "full_sha256": None,
            }
        )
    )
    (session_dir / "llamacpp.json").write_text(
        json.dumps(
            {
                "bench_path": "/usr/bin/llama-bench",
                "cli_path": None,
                "server_path": None,
                "capabilities": ["fa"],
                "help_sha256": "deadbeef",
                "build_commit": None,
                "build_number": None,
            }
        )
    )
    (session_dir / "journal.jsonl").write_text("")


def test_report_command_writes_report_md(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    _write_session_files(session_dir)

    result = runner.invoke(app, ["report", str(session_dir)])
    assert result.exit_code == 0
    assert (session_dir / "report.md").is_file()
    assert "llamatune report" in (session_dir / "report.md").read_text()


def test_report_command_json_output(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    _write_session_files(session_dir)

    result = runner.invoke(app, ["report", str(session_dir), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema_version"] == 1


def test_report_command_missing_evidence_exits_2(tmp_path: Path) -> None:
    result = runner.invoke(app, ["report", str(tmp_path / "nonexistent-session")])
    assert result.exit_code == 2


def test_report_command_rejects_out_of_tree_analysis_symlink(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    _write_session_files(session_dir)
    victim = tmp_path / "victim.json"
    victim.write_text('{"schema_version": 1, "secret": "external"}\n')  # pragma: allowlist secret
    (session_dir / "analysis.json").unlink()
    try:
        (session_dir / "analysis.json").symlink_to(victim)
    except (OSError, NotImplementedError) as exc:  # pragma: no cover - platform seam
        pytest.skip(f"symlink creation unavailable on this platform: {exc}")

    result = runner.invoke(app, ["report", str(session_dir), "--json"])

    assert result.exit_code == 2
    assert "could not read session evidence" in result.stderr
    assert "external" not in result.output
    assert not (session_dir / "report.md").exists()


def test_report_command_non_object_analysis_exits_2(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    _write_session_files(session_dir)
    (session_dir / "analysis.json").write_text("[]\n")

    result = runner.invoke(app, ["report", str(session_dir)])

    assert result.exit_code == 2
    assert "could not read session evidence" in result.stderr
    assert "Traceback" not in result.output


def test_version_flag_prints_version_and_exits_zero() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout == f"llamatune {__version__}\n"
    assert "Traceback" not in result.output


def test_marathon_depth_capability_error_has_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _no_depth(llama_bin: Path | None) -> Any:
        return SimpleNamespace(bench_path=Path("/opt/llama-bench"), capabilities=frozenset())

    monkeypatch.setattr("llamatune.llama.discover_llama", _no_depth)
    result = runner.invoke(app, ["marathon", "model.gguf", "--depth-grid", "0,8192"])
    assert result.exit_code == 2
    assert "does not support -d/--n-depth" in result.stderr
    assert "Use --depth-grid 0, or update llama.cpp for depth support." in result.stderr


def test_calibrate_write_failure_exits_3_without_traceback(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("regular file, not a directory")

    result = runner.invoke(app, ["calibrate", "--sessions-dir", str(blocker)])

    assert result.exit_code == 3
    assert result.stderr.startswith("error: could not write calibration file under")
    assert str(blocker) in result.stderr
    assert "Traceback" not in result.output


def test_calibrate_value_error_exits_2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _boom(path: Path) -> dict[str, Any]:
        raise ValueError("bad session evidence")

    monkeypatch.setattr("llamatune.calibrate.calibrate", _boom)

    result = runner.invoke(app, ["calibrate", "--sessions-dir", str(tmp_path)])

    assert result.exit_code == 2
    assert "error: could not fit calibration factors: bad session evidence" in result.stderr
    assert "Traceback" not in result.output


def test_unexpected_engine_error_becomes_single_stderr_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _boom(path: Path) -> list[dict[str, Any]]:
        raise RuntimeError("boom")

    monkeypatch.setattr("llamatune.session.list_sessions", _boom)

    result = runner.invoke(app, ["sessions", str(tmp_path)])

    assert result.exit_code == 3
    assert result.stderr == (
        "internal error: boom; report at https://github.com/barebonescyber/llamatune/issues\n"
    )
    assert "Traceback" not in result.output


def test_internal_error_guard_reraises_keyboard_interrupt() -> None:
    def callback() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _guard_internal_errors(callback)()


def test_internal_error_guard_passes_through_typer_exit() -> None:
    def callback() -> None:
        raise typer.Exit(code=2)

    with pytest.raises(typer.Exit) as excinfo:
        _guard_internal_errors(callback)()

    assert excinfo.value.exit_code == 2


def test_verbose_flag_emits_startup_diagnostics_on_success(
    fake_bin_dir: Path, tiny_gguf: Path, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "--verbose",
            "tune",
            str(tiny_gguf),
            "--llama-bin",
            str(fake_bin_dir),
            "--sessions-dir",
            str(tmp_path / "sessions"),
            "--baseline-only",
        ],
    )
    assert result.exit_code == 0
    assert f"[verbose] model: {tiny_gguf}" in result.stderr
    assert "[verbose] llama-bench:" in result.stderr
    assert "[verbose] probe argv:" in result.stderr
    assert "[verbose] capabilities:" in result.stderr


def test_verbose_flag_emits_diagnostics_on_failing_command(tmp_path: Path) -> None:
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    result = runner.invoke(
        app,
        ["--verbose", "tune", "missing.gguf", "--llama-bin", str(empty_bin)],
    )
    assert result.exit_code == 3
    assert f"[verbose] llama-bin: {empty_bin}" in result.stderr
    assert "error: llama-bench not found" in result.stderr
    assert "[verbose]" not in result.stdout


def test_verbose_diagnostics_are_off_by_default(tmp_path: Path) -> None:
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    result = runner.invoke(
        app,
        ["tune", "missing.gguf", "--llama-bin", str(empty_bin)],
    )
    assert result.exit_code == 3
    assert "[verbose]" not in result.stderr
    assert "[verbose]" not in result.stdout


def test_llamatune_verbose_env_var_equals_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    monkeypatch.setenv("LLAMATUNE_VERBOSE", "1")
    result = runner.invoke(
        app,
        ["tune", "missing.gguf", "--llama-bin", str(empty_bin)],
    )
    assert result.exit_code == 3
    assert f"[verbose] llama-bin: {empty_bin}" in result.stderr


def test_verbose_resume_emits_session_diagnostics(tmp_path: Path) -> None:
    missing = tmp_path / "gone-session"
    result = runner.invoke(app, ["--verbose", "resume", str(missing)])
    assert result.exit_code == 2
    assert f"[verbose] session: {missing}" in result.stderr
