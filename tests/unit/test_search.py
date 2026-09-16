"""Unit tests for llamatune.search (orchestration edge cases and helpers)."""

from __future__ import annotations

import dataclasses
import errno
import json
import signal
import sys
from pathlib import Path
from typing import Any

import pytest

from llamatune import bench, config, executor, search
from llamatune.bench import BenchParseError
from llamatune.llama import discover_llama
from llamatune.model import inspect_model
from llamatune.session import Session, SessionPathError
from llamatune.types import (
    BaselineResult,
    GPUInfo,
    GpuSample,
    HardwareReport,
    LlamaCppReport,
    MetricStats,
    ModelReport,
    TrialConfig,
    TuneOptions,
    TuneOutcome,
)


def _write_fake_perplexity(tmp_path: Path, source: str) -> Path:
    if sys.platform == "win32":
        script = tmp_path / "fake_llama_perplexity.py"
        script.write_text(source)
        wrapper = tmp_path / "llama-perplexity.cmd"
        wrapper.write_text(f'@"{sys.executable}" "{script}" %*\r\n')
        return wrapper
    executable = tmp_path / "llama-perplexity"
    executable.write_text(f"#!/usr/bin/env python3\n{source}")
    executable.chmod(0o755)
    return executable


def _hardware(*, gpus: tuple[GPUInfo, ...] | None = None) -> HardwareReport:
    if gpus is None:
        gpus = (GPUInfo(vendor="nvidia", name="Fake GPU", vram_mb=24000, method="test"),)
    return HardwareReport(
        os_name="Linux",
        arch="x86_64",
        cpu_model="Fake CPU",
        physical_cores=8,
        logical_cores=16,
        perf_cores=None,
        ram_mb=32768,
        gpus=gpus,
        warnings=(),
    )


def _options(sessions_dir: Path, llama_bin: Path, **overrides: Any) -> TuneOptions:
    fields: dict[str, Any] = {
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
        "thermal_wait_cap_s": 0.0,
        "baseline_only": False,
        "llama_bin": llama_bin,
        "sessions_dir": sessions_dir,
        "full_hash": False,
    }
    fields.update(overrides)
    return TuneOptions(**fields)


def _setup(
    tmp_path: Path,
    fake_bin_dir: Path,
    gguf: Path,
    *,
    hardware: HardwareReport | None = None,
    **option_overrides: Any,
) -> tuple[Session, HardwareReport, ModelReport, LlamaCppReport, TuneOptions]:
    hw = hardware if hardware is not None else _hardware()
    llama = discover_llama(fake_bin_dir)
    model = inspect_model(gguf)
    options = _options(tmp_path / "sessions", fake_bin_dir, **option_overrides)
    session = Session.create(
        options.sessions_dir,
        model=model,
        hardware=hw,
        llama=llama,
        options=options,
        argv=["llamatune", "tune", str(gguf)],
    )
    return session, hw, model, llama, options


def _trial_entries(session_dir: Path) -> list[dict[str, Any]]:
    entries = []
    with (session_dir / "journal.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            entry = json.loads(line)
            if entry.get("type") == "trial":
                entries.append(entry)
    return entries


class TestBaselineOnly:
    def test_exit_0_and_no_trials(
        self, tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path
    ) -> None:
        session, hw, model, llama, options = _setup(
            tmp_path, fake_bin_dir, tiny_gguf, baseline_only=True, cooldown_s=0.01
        )
        outcome = search.run_tuning(session, hw, model, llama, options)

        assert outcome.exit_code == 0
        assert outcome.analysis["winner"] is None
        assert outcome.analysis["baseline"]["runs"] == 3
        assert outcome.analysis["baseline"]["noise_floor_cv"] >= 0.01
        assert _trial_entries(session.dir) == []
        assert (session.dir / "analysis.json").is_file()
        assert (session.dir / "recommended.json").is_file()
        assert (session.dir / "recommended.sh").is_file()
        assert (session.dir / "report.md").is_file()
        # Build identity captured from the fake bench output (DESIGN §6).
        llamacpp = json.loads((session.dir / "llamacpp.json").read_text())
        assert llamacpp["build_commit"] == "fake1234"
        assert llamacpp["build_number"] == 9999
        assert llamacpp["backends"] == "CUDA"
        recommended = json.loads((session.dir / "recommended.json").read_text())
        assert recommended["llamacpp"]["build_commit"] == "fake1234"
        assert recommended["llamacpp"]["build_number"] == 9999

    def test_resolved_defaults_recorded(
        self, tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path
    ) -> None:
        session, hw, model, llama, options = _setup(
            tmp_path, fake_bin_dir, tiny_gguf, baseline_only=True
        )
        outcome = search.run_tuning(session, hw, model, llama, options)
        defaults = outcome.analysis["baseline"]["resolved_defaults"]
        assert defaults["gpu_layers"] == 33
        assert defaults["flash_attn"] is False
        assert defaults["ubatch"] == 512
        assert defaults["batch"] == 2048

    def test_negative_one_default_is_normalized_to_full_offload(
        self, tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path
    ) -> None:
        session, hw, model, llama, options = _setup(
            tmp_path, fake_bin_dir, tiny_gguf, baseline_only=True
        )
        engine = search._Engine(session, hw, model, llama, options)
        sentinel = dataclasses.replace(_envelope_config(0), gpu_layers=-1)

        normalized = engine._normalize_resolved_config(sentinel)

        assert normalized.gpu_layers == model.ngl_all
        assert config.is_valid_config(
            normalized,
            hardware=hw,
            model=model,
            llama=llama,
        )

    def test_missing_successful_benchmark_entry_is_controlled_baseline_failure(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session, hw, model, llama, options = _setup(
            tmp_path, fake_bin_dir, tiny_gguf, baseline_only=True
        )
        metric = MetricStats(mean=1.0, stdev=0.0, cv=0.0, n=3)

        def missing_entry(
            _engine: search._Engine,
            _index: int,
            *,
            ngl: int | None,
            walls: list[float],
            display_index: int | None = None,
            display_prefix: str = "baseline",
        ) -> search._Measured:
            del ngl, display_index, display_prefix
            walls.append(1.0)
            return search._Measured("ok", metric, metric, None, None)

        monkeypatch.setattr(search._Engine, "_run_baseline_once", missing_entry)

        outcome = search.run_tuning(session, hw, model, llama, options)

        assert outcome.exit_code == 3
        assert outcome.failure_stage == "baseline"
        assert outcome.failure_reason == (
            "baseline/run-3 reported success without a benchmark entry"
        )


class TestBudget:
    def test_trial_budget_exhaustion_falls_back_to_defaults(
        self, tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path
    ) -> None:
        session, hw, model, llama, options = _setup(
            tmp_path, fake_bin_dir, tiny_gguf, budget_trials=4
        )
        outcome = search.run_tuning(session, hw, model, llama, options)

        # The three baseline runs consume the budget that remains after
        # reserving a complete confirmation batch. No unconfirmable search
        # trial is launched, so the default recommendation exits 1.
        assert outcome.exit_code == 1
        assert outcome.analysis["winner"] is None
        assert outcome.analysis["counts"]["executed"] == 0
        assert outcome.analysis["counts"]["budget_consumed"] == 3
        stages = [e.get("stage") for e in session.entries if e.get("type") == "stage"]
        assert "budget_exhausted" in stages
        coverage = outcome.analysis["coverage"]
        assert coverage
        for details in coverage.values():
            accounted = {
                item.get("value") if isinstance(item, dict) else item
                for bucket in ("executed", "cached_hit", "pruned", "skipped")
                for item in details[bucket]
            }
            assert accounted == set(details["candidates"])

    def test_minutes_budget_exhaustion(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session, hw, model, llama, options = _setup(
            tmp_path, fake_bin_dir, tiny_gguf, budget_minutes=0.0
        )
        outcome = search.run_tuning(session, hw, model, llama, options)
        assert outcome.exit_code == 1
        assert outcome.analysis["counts"]["executed"] == 0


class TestBaselineFailures:
    def test_cpu_fallback_progress_and_warning_match_the_default_failure(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class RecordingReporter:
            def __init__(self) -> None:
                self.events: list[Any] = []

            def emit(self, event: Any) -> None:
                self.events.append(event)

        monkeypatch.setenv("LLAMATUNE_FAKE_VRAM_MB", "2500")
        monkeypatch.setenv("LLAMATUNE_FAKE_FAIL_STYLE", "load")
        session, hw, model, llama, options = _setup(
            tmp_path,
            fake_bin_dir,
            tiny_gguf,
            baseline_only=True,
            budget_trials=4,
        )
        reporter = RecordingReporter()

        outcome = search.run_tuning(
            session,
            hw,
            model,
            llama,
            options,
            reporter=reporter,
        )

        labels = [
            event.payload["label"]
            for event in reporter.events
            if event.kind == "exec_start" and event.payload["kind"] == "baseline"
        ]
        assert labels == [
            "baseline 1/3",
            "CPU fallback 1/3",
            "CPU fallback 2/3",
            "CPU fallback 3/3",
        ]
        assert any(
            "after a GPU-resource failure" in warning for warning in outcome.analysis["warnings"]
        )
        assert all("out-of-memory" not in warning for warning in outcome.analysis["warnings"])

    @pytest.mark.parametrize("style", ["load", "ctx"])
    def test_ambiguous_default_failure_uses_safe_fallback(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
        monkeypatch: pytest.MonkeyPatch,
        style: str,
    ) -> None:
        monkeypatch.setenv("LLAMATUNE_FAKE_VRAM_MB", "2500")
        monkeypatch.setenv("LLAMATUNE_FAKE_FAIL_STYLE", style)
        session, hw, model, llama, options = _setup(
            tmp_path, fake_bin_dir, tiny_gguf, budget_trials=24
        )
        outcome = search.run_tuning(session, hw, model, llama, options)

        assert outcome.exit_code in (0, 1)
        assert outcome.analysis["baseline_kind"] == "safe_fallback"
        probe = outcome.analysis["default_probe"]
        assert probe["classification"] == "gpu_resource"
        assert probe["evidence"] == "baseline/run-1"

    def test_cuda_failure_is_recorded_and_never_recommended(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLAMATUNE_FAKE_VRAM_MB", "2500")
        monkeypatch.setenv("LLAMATUNE_FAKE_FAIL_STYLE", "cuda")
        session, hw, model, llama, options = _setup(
            tmp_path, fake_bin_dir, tiny_gguf, budget_trials=24
        )
        outcome = search.run_tuning(session, hw, model, llama, options)

        assert outcome.exit_code in (0, 1)
        assert outcome.analysis["default_probe"]["classification"] == "cuda_error"
        outcomes = [
            entry.get("status")
            for entry in session.entries
            if entry.get("type") in {"trial", "probe"}
        ]
        assert "cuda_error" in outcomes
        winner = outcome.analysis["winner"]
        if winner is not None:
            winner_records = [
                entry for entry in session.entries if entry.get("trial_id") == winner["trial_id"]
            ]
            assert winner_records
            assert all(entry.get("status") in {"ok", "unstable"} for entry in winner_records)

    def test_genuine_failure_names_both_baseline_paths(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLAMATUNE_FAKE_GENUINE_FAIL", "1")
        session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf)
        outcome = search.run_tuning(session, hw, model, llama, options)

        assert outcome.exit_code == 3
        assert outcome.failure_stage == "baseline"
        assert "baseline/run-1" in (outcome.failure_reason or "")
        assert "baseline/run-2" in (outcome.failure_reason or "")

    def test_crash_exits_3(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLAMATUNE_FAKE_CRASH", "1")
        session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf)
        outcome = search.run_tuning(session, hw, model, llama, options)

        assert outcome.exit_code == 3
        assert outcome.failure_stage == "baseline"
        assert "baseline/run-1" in (outcome.failure_reason or "")
        assert "baseline/run-2" in (outcome.failure_reason or "")
        end = session.entries[-1]
        assert end["type"] == "session_end"
        assert end["exit_code"] == 3
        assert "crash" in end["reason"]
        assert (session.dir / "baseline" / "run-1" / "command.json").is_file()
        assert not (session.dir / "analysis.json").exists()

    def test_timeout_exits_3(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLAMATUNE_FAKE_HANG_S", "10")
        monkeypatch.setattr(search, "_BASELINE_TIMEOUT_S", 0.5)
        session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf)
        outcome = search.run_tuning(session, hw, model, llama, options)

        assert outcome.exit_code == 3
        assert outcome.failure_stage == "baseline"
        assert "baseline/run-1" in (outcome.failure_reason or "")
        assert "baseline/run-2" in (outcome.failure_reason or "")
        assert "timeout" in session.entries[-1]["reason"]
        assert (session.dir / "baseline" / "run-1" / "command.json").is_file()
        assert not (session.dir / "analysis.json").exists()

    def test_parse_error_exits_3(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise(raw: bytes | str) -> Any:
            raise BenchParseError("injected")

        monkeypatch.setattr("llamatune.bench.parse_bench_output", _raise)
        session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf)
        outcome = search.run_tuning(session, hw, model, llama, options)

        assert outcome.exit_code == 3
        assert outcome.failure_stage == "baseline"
        assert "baseline/run-1" in (outcome.failure_reason or "")
        assert "baseline/run-2" in (outcome.failure_reason or "")
        assert "parse_error" in session.entries[-1]["reason"]
        assert (session.dir / "baseline" / "run-1" / "command.json").is_file()
        assert not (session.dir / "analysis.json").exists()

    @pytest.mark.parametrize(
        ("env_name", "expected_status"),
        [
            ("LLAMATUNE_FAKE_MALFORMED_OUTPUT", "parse_error"),
            ("LLAMATUNE_FAKE_OVERSIZED_OUTPUT", "parse_error"),
            ("LLAMATUNE_FAKE_HOST_MEMORY_FAIL", "crash"),
        ],
    )
    def test_real_child_failure_output_is_bounded_and_classified(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
        monkeypatch: pytest.MonkeyPatch,
        env_name: str,
        expected_status: str,
    ) -> None:
        monkeypatch.setenv(env_name, "1")
        session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf)

        outcome = search.run_tuning(session, hw, model, llama, options)

        assert outcome.exit_code == 3
        assert outcome.failure_stage == "baseline"
        assert "baseline/run-1" in (outcome.failure_reason or "")
        assert "baseline/run-2" in (outcome.failure_reason or "")
        assert session.entries[-1]["type"] == "session_end"
        assert session.entries[-1]["exit_code"] == 3
        assert not (session.dir / "analysis.json").exists()
        runs = [entry for entry in session.entries if entry.get("type") == "baseline_run"]
        assert [run["status"] for run in runs] == [expected_status, expected_status]
        commands = [
            json.loads((session.dir / "baseline" / f"run-{index}" / "command.json").read_text())
            for index in (1, 2)
        ]
        assert all(command["stdout"]["size_bytes"] <= 8 * 1024 * 1024 for command in commands)
        if env_name == "LLAMATUNE_FAKE_OVERSIZED_OUTPUT":
            assert all(command["stdout"]["truncated"] is True for command in commands)
            assert all(command["stdout"]["size_bytes"] == 8 * 1024 * 1024 for command in commands)
        if env_name == "LLAMATUNE_FAKE_HOST_MEMORY_FAIL":
            stderr = (session.dir / "baseline" / "run-1" / "stderr.log").read_text()
            assert "bad_alloc" in stderr
            assert all(run["status"] != "oom" for run in runs)

    def test_keyboard_interrupt_exits_4(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _interrupt(*args: Any, **kwargs: Any) -> Any:
            raise KeyboardInterrupt

        monkeypatch.setattr("llamatune.executor.run", _interrupt)
        session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf)
        outcome = search.run_tuning(session, hw, model, llama, options)

        assert outcome.exit_code == 4
        assert outcome.failure_stage == "interruption"
        assert outcome.failure_reason == "interrupted by user"
        end = session.entries[-1]
        assert end["type"] == "session_end"
        assert end["exit_code"] == 4

    def test_full_session_storage_returns_resumable_exit_4(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf)

        def _disk_full(_entry: dict[str, Any]) -> None:
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(session, "append", _disk_full)
        outcome = search.run_tuning(session, hw, model, llama, options)

        assert outcome.exit_code == 4
        assert outcome.failure_stage == "evidence"
        assert "No space left on device" in (outcome.failure_reason or "")
        assert outcome.session_dir == session.dir
        assert [entry["type"] for entry in session.entries] == ["session_start"]
        reloaded = Session.load(session.dir)
        assert [entry["type"] for entry in reloaded.entries] == ["session_start"]
        assert reloaded.model.fingerprint == model.fingerprint


class TestStability:
    def test_unstable_trials_rerun_once_and_flagged(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("llamatune.stats.is_unstable", lambda pp, tg: True)
        session, hw, model, llama, options = _setup(
            tmp_path, fake_bin_dir, tiny_gguf, budget_trials=10
        )
        outcome = search.run_tuning(session, hw, model, llama, options)

        trials = _trial_entries(session.dir)
        assert trials, "at least one trial should have executed"
        assert all(t["status"] == "unstable" for t in trials)
        assert any("unstable" in w for w in outcome.analysis["warnings"])
        assert outcome.analysis["counts"]["unstable"] == len(trials)
        # The rerun artifacts exist for at least one re-measured trial.
        rerun_dirs = list((session.dir / "trials").glob("*/rerun"))
        assert rerun_dirs


class TestLossy:
    def test_allow_lossy_explores_cache_types(
        self, tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path
    ) -> None:
        session, hw, model, llama, options = _setup(
            tmp_path, fake_bin_dir, tiny_gguf, allow_lossy=True
        )
        outcome = search.run_tuning(session, hw, model, llama, options)

        trials = _trial_entries(session.dir)
        assert any(t["config"]["cache_type_k"] == "q8_0" for t in trials)
        assert outcome.analysis["lossless_winner"] is not None
        winner = outcome.analysis["winner"]
        if winner is not None and (
            winner["config"]["cache_type_k"] != "f16" or winner["config"]["cache_type_v"] != "f16"
        ):
            assert any("lossy" in w for w in outcome.analysis["warnings"])

    def test_lossy_winner_warning_is_emitted_for_analysis_json(
        self, tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path
    ) -> None:
        session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf)
        engine = search._Engine(session, hw, model, llama, options)
        metric = MetricStats(mean=1.0, stdev=0.0, cv=0.0, n=3)
        engine.baseline = BaselineResult(
            runs=3,
            pp=metric,
            tg=metric,
            noise_floor_cv=0.01,
            fallback=None,
            resolved_defaults=_envelope_config(1).to_dict(),
        )
        winner = {
            "config": dataclasses.replace(_envelope_config(20), cache_type_k="q8_0").to_dict()
        }

        warnings = engine._collect_warnings(winner)

        assert (
            "winner uses lossy KV-cache quantization; validate output quality before adoption"
            in warnings
        )


class TestLoadWarning:
    def test_high_load_recorded(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(search, "_load_avg", lambda: 999.0)
        session, hw, model, llama, options = _setup(
            tmp_path, fake_bin_dir, tiny_gguf, baseline_only=True
        )
        outcome = search.run_tuning(session, hw, model, llama, options)
        assert any("system load" in w for w in outcome.analysis["warnings"])


class TestFeasibilitySearch:
    def test_hard_gpu_cap_prunes_defensively_before_execution(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session, hw, model, llama, options = _setup(
            tmp_path,
            fake_bin_dir,
            tiny_gguf,
            max_gpu_layers=8,
        )
        engine = search._Engine(session, hw, model, llama, options)
        monkeypatch.setattr(
            engine,
            "_execute_trial",
            lambda *_args, **_kwargs: pytest.fail("out-of-cap trial executed"),
        )
        monkeypatch.setattr(
            engine,
            "_run_child",
            lambda *_args, **_kwargs: pytest.fail("out-of-cap probe executed"),
        )

        result = engine._evaluate(_envelope_config(9), "defensive_cap")

        assert result.status == "pruned"
        pruned = next(entry for entry in session.entries if entry.get("dim") == "defensive_cap")
        assert pruned["pruned_from"] == "max_gpu_layers:8"
        assert engine._probe(_envelope_config(9), "context", 8192) == "pruned"
        probe = next(entry for entry in session.entries if entry.get("type") == "probe")
        assert probe["pruned_from"] == "max_gpu_layers:8"

        above = _envelope_config(9)
        below = _envelope_config(7)
        engine.known = {
            above.trial_id: {
                "trial_id": above.trial_id,
                "status": "ok",
                "score": 3.0,
                "config": above.to_dict(),
            },
            below.trial_id: {
                "trial_id": below.trial_id,
                "status": "ok",
                "score": 2.0,
                "config": below.to_dict(),
            },
        }
        assert engine._runner_up(_envelope_config(8)) == below
        assert [record["trial_id"] for record in engine._analysis_records()] == [below.trial_id]
        engine._validate_with_cli(above)

    def test_failed_bounded_probe_is_not_reported_as_a_fitting_cap(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session, hw, model, llama, options = _setup(
            tmp_path,
            fake_bin_dir,
            tiny_gguf,
            initial_gpu_layers=8,
            max_gpu_layers=8,
        )
        engine = search._Engine(session, hw, model, llama, options)
        engine.incumbent_config = _envelope_config(33)
        monkeypatch.setattr(engine, "_probe", lambda *_args, **_kwargs: "failed")
        monkeypatch.setattr(
            engine,
            "_evaluate",
            lambda *_args, **_kwargs: pytest.fail("unverified boundary measured"),
        )

        boundary = engine._discover_boundary(_envelope_config(33), 0)

        assert boundary.max_ok_ngl == 0
        assert boundary.min_fail_ngl == 0
        assert boundary.probes == 2
        engine.boundaries = [boundary]
        assert engine._context_candidates(_envelope_config(33)) == []

    def test_no_successful_capped_config_emits_no_fabricated_recommendation(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session, hw, model, llama, options = _setup(
            tmp_path,
            fake_bin_dir,
            tiny_gguf,
            budget_trials=40,
            initial_gpu_layers=0,
            max_gpu_layers=0,
        )
        monkeypatch.setattr(search._Engine, "_probe", lambda *_args, **_kwargs: "failed")
        monkeypatch.setattr(
            search._Engine,
            "_evaluate",
            lambda *_args, **_kwargs: search._Trial("failed", None, None, None),
        )

        outcome = search.run_tuning(session, hw, model, llama, options)

        assert outcome.exit_code == 3
        assert outcome.failure_stage == "recommendation"
        assert outcome.failure_reason == (
            "no successful measured configuration satisfies max_gpu_layers=0"
        )
        assert not (session.dir / "recommended.json").exists()
        stage = next(
            entry for entry in session.entries if entry.get("stage") == "no_feasible_config"
        )
        assert stage["reason"] == outcome.failure_reason

    def test_capped_baseline_only_emits_no_above_cap_recommendation(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
    ) -> None:
        session, hw, model, llama, options = _setup(
            tmp_path,
            fake_bin_dir,
            tiny_gguf,
            baseline_only=True,
            max_gpu_layers=0,
        )

        outcome = search.run_tuning(session, hw, model, llama, options)

        assert outcome.exit_code == 3
        assert outcome.failure_stage == "recommendation"
        assert outcome.failure_reason == (
            "baseline defaults exceed max_gpu_layers=0; no cap-compliant configuration was measured"
        )
        assert not (session.dir / "recommended.json").exists()

    def test_no_feasible_resume_invalidates_stale_derived_outputs(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
    ) -> None:
        session, hw, model, llama, options = _setup(
            tmp_path,
            fake_bin_dir,
            tiny_gguf,
            max_gpu_layers=0,
        )
        for name in ("analysis.json", "recommended.json", "recommended.sh", "report.md"):
            session.write_text(name, "stale\n")
        engine = search._Engine(session, hw, model, llama, options)
        engine.default_config = _envelope_config(33)

        outcome = engine._finalize_or_no_feasible(baseline_only=True)

        assert outcome.exit_code == 3
        assert not any(
            (session.dir / name).exists()
            for name in ("analysis.json", "recommended.json", "recommended.sh", "report.md")
        )

    def test_hard_gpu_cap_bounds_every_search_measurement(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
    ) -> None:
        cap = 8
        session, hw, model, llama, options = _setup(
            tmp_path,
            fake_bin_dir,
            tiny_gguf,
            budget_trials=100,
            initial_gpu_layers=cap,
            max_gpu_layers=cap,
        )
        outcome = search.run_tuning(session, hw, model, llama, options)

        assert outcome.exit_code in (0, 1)
        assert outcome.analysis["baseline"]["resolved_defaults"]["gpu_layers"] > cap
        boundary = outcome.analysis["feasibility"]["boundaries"][0]
        assert boundary["cap_ngl"] == cap
        assert boundary["cap_source"] == "cli"
        assert outcome.analysis["feasibility"]["recommended"]["gpu_layers"] <= cap
        assert outcome.analysis["feasibility"]["best_measured"]["gpu_layers"] <= cap
        recommended = json.loads((session.dir / "recommended.json").read_text())
        assert recommended["config"]["gpu_layers"] <= cap
        winner = outcome.analysis["winner"]
        if winner is not None:
            assert winner["config"]["gpu_layers"] <= cap
        measurements = [
            entry
            for entry in session.entries
            if entry.get("type") in {"trial", "probe"} and "config" in entry
        ]
        assert measurements
        assert all(entry["config"]["gpu_layers"] <= cap for entry in measurements)
        assert any(entry.get("dim") == "joint_refine" for entry in measurements)

    def test_zero_gpu_cap_skips_moe_ladder(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
    ) -> None:
        session, hw, model, llama, options = _setup(
            tmp_path,
            fake_bin_dir,
            tiny_gguf,
            budget_trials=40,
            initial_gpu_layers=0,
            max_gpu_layers=0,
        )
        outcome = search.run_tuning(session, hw, model, llama, options)

        assert outcome.analysis["feasibility"]["recommended"]["gpu_layers"] == 0
        assert len(outcome.analysis["feasibility"]["boundaries"]) == 1
        assert all(
            entry["config"]["moe_cpu_layers"] == 0
            for entry in session.entries
            if entry.get("type") in {"trial", "probe"} and "config" in entry
        )

    def test_exact_boundary_and_context_safe_recommendation(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLAMATUNE_FAKE_VRAM_MB", "2850")
        monkeypatch.setenv("LLAMATUNE_FAKE_FAIL_STYLE", "load")
        monkeypatch.setenv("LLAMATUNE_FAKE_KV_MB_PER_1K", "10")
        session, hw, model, llama, options = _setup(
            tmp_path, fake_bin_dir, tiny_gguf, budget_trials=100, ctx_size=8192
        )
        outcome = search.run_tuning(session, hw, model, llama, options)

        boundary = outcome.analysis["feasibility"]["boundaries"][0]
        assert boundary["max_ok_ngl"] == 23
        assert boundary["min_fail_ngl"] == 24
        assert outcome.analysis["context_validation"]["status"] == "ok"
        assert outcome.analysis["feasibility"]["recommended"]["gpu_layers"] <= 22
        winner = outcome.analysis["winner"]
        if winner is not None and winner["confirmed"]:
            successful_contexts = [
                entry
                for entry in session.entries
                if entry.get("type") == "probe"
                and entry.get("purpose") == "context"
                and entry.get("status") == "ok"
            ]
            assert any(entry["config"] == winner["config"] for entry in successful_contexts)
        probe_ids = [e["probe_id"] for e in session.entries if e.get("type") == "probe"]
        assert len(probe_ids) == len(set(probe_ids))

    def test_plan_records_reserve_provenance(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
    ) -> None:
        session, hw, model, llama, options = _setup(
            tmp_path, fake_bin_dir, tiny_gguf, baseline_only=True, vram_reserve_mb=512
        )
        search.run_tuning(session, hw, model, llama, options)
        plan = next(e for e in session.entries if e.get("stage") == "plan")
        assert plan["vram_reserve_mb"] == 512
        assert plan["reserve_provenance"] == "cli"
        assert plan["budget_mb"] == 23488
        assert plan["budget_basis"] == "total-reserve"

    def test_plan_pools_multi_gpu_capacity_only_when_enabled(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
    ) -> None:
        gpu = GPUInfo(
            vendor="nvidia",
            name="Second GPU",
            vram_mb=12000,
            method="test",
        )
        hw = _hardware(gpus=(*_hardware().gpus, gpu))
        session, _, model, llama, options = _setup(
            tmp_path,
            fake_bin_dir,
            tiny_gguf,
            hardware=hw,
            multi_gpu=True,
            vram_reserve_mb=512,
        )
        engine = search._Engine(session, hw, model, llama, options)
        engine._ensure_plan()
        plan = next(e for e in session.entries if e.get("stage") == "plan")
        assert plan["capacity_basis"] == "pooled-known-devices"
        assert plan["budget_mb"] == 35488


class TestResumeValidation:
    def test_unreadable_session_dir_exits_2(self, tmp_path: Path) -> None:
        outcome = search.resume_tuning(tmp_path / "does-not-exist")
        assert outcome.exit_code == 2
        assert outcome.analysis == {}
        assert outcome.failure_stage == "session"
        assert outcome.failure_reason

    def test_confined_path_failure_has_controlled_resume_and_revalidate_exits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _escape(_cls: type[Session], _session_dir: Path) -> Session:
            raise SessionPathError("artifact escapes session directory")

        monkeypatch.setattr(Session, "load", classmethod(_escape))

        resumed = search.resume_tuning(tmp_path / "session")
        assert resumed.exit_code == 2
        assert resumed.failure_stage == "session"
        assert resumed.failure_reason == "artifact escapes session directory"

        revalidated = search.revalidate_session(tmp_path / "session")
        assert revalidated.exit_code == 3
        assert revalidated.failure_stage == "revalidation"
        assert revalidated.failure_reason == "artifact escapes session directory"

    def test_non_object_journal_entry_exits_2(
        self, tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path
    ) -> None:
        session, _, _, _, _ = _setup(tmp_path, fake_bin_dir, tiny_gguf)
        with (session.dir / "journal.jsonl").open("a", encoding="utf-8") as fh:
            fh.write("123\n")

        outcome = search.resume_tuning(session.dir)

        assert outcome.exit_code == 2
        assert outcome.analysis == {}
        assert outcome.failure_stage == "session"
        assert "expected a JSON object" in (outcome.failure_reason or "")

    def test_null_gpus_in_hardware_json_exits_2(
        self, tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path
    ) -> None:
        session, _, _, _, _ = _setup(tmp_path, fake_bin_dir, tiny_gguf)
        hardware_path = session.dir / "hardware.json"
        hardware = json.loads(hardware_path.read_text())
        hardware["gpus"] = None
        hardware_path.write_text(json.dumps(hardware))

        outcome = search.resume_tuning(session.dir)

        assert outcome.exit_code == 2
        assert outcome.analysis == {}
        assert outcome.failure_stage == "session"
        assert outcome.failure_reason

    def test_identity_mismatch_exits_3(
        self, tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path
    ) -> None:
        import dataclasses

        llama = discover_llama(fake_bin_dir)
        doctored = dataclasses.replace(llama, help_sha256="0" * 64)
        model = inspect_model(tiny_gguf)
        options = _options(tmp_path / "sessions", fake_bin_dir)
        session = Session.create(
            options.sessions_dir,
            model=model,
            hardware=_hardware(),
            llama=doctored,
            options=options,
            argv=["llamatune"],
        )

        outcome = search.resume_tuning(session.dir)
        assert outcome.exit_code == 3
        assert outcome.failure_stage == "resume_validation"
        assert outcome.failure_reason == "model or llama.cpp identity mismatch"
        entries = Session.load(session.dir).entries
        assert entries[-1]["reason"] == "identity mismatch"

    def test_binary_hash_mismatch_is_a_soft_resume_warning(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session, hw, _model, llama, _options_value = _setup(tmp_path, fake_bin_dir, tiny_gguf)
        changed = dataclasses.replace(llama, bench_sha256="f" * 64)
        monkeypatch.setattr("llamatune.llama.discover_llama", lambda _path: changed)
        monkeypatch.setattr("llamatune.hardware.assess_hardware", lambda: hw)
        monkeypatch.setattr(
            search._Engine,
            "run",
            lambda engine: TuneOutcome(session_dir=engine.session.dir, analysis={}, exit_code=0),
        )

        outcome = search.resume_tuning(session.dir)

        assert outcome.exit_code == 0
        resumed = [
            entry for entry in Session.load(session.dir).entries if entry.get("stage") == "resumed"
        ][-1]
        assert any("binary changed" in warning for warning in resumed["warnings"])

    def test_discovery_failure_exits_3(
        self, tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path
    ) -> None:
        empty_bin = tmp_path / "empty-bin"
        empty_bin.mkdir()
        llama = discover_llama(fake_bin_dir)
        model = inspect_model(tiny_gguf)
        options = _options(tmp_path / "sessions", empty_bin)  # resume will look here
        session = Session.create(
            options.sessions_dir,
            model=model,
            hardware=_hardware(),
            llama=llama,
            options=options,
            argv=["llamatune"],
        )

        outcome = search.resume_tuning(session.dir)
        assert outcome.exit_code == 3
        assert outcome.failure_stage == "resume_validation"
        assert "llama-bench not found" in (outcome.failure_reason or "")


class TestHelpers:
    def test_order_candidates_gpu_ascending(self) -> None:
        assert search._order_candidates("gpu_layers", (17, 0, 33, 8)) == [0, 8, 17, 33]

    def test_order_candidates_moe_descending(self) -> None:
        assert search._order_candidates("moe_cpu_layers", (0, 16, 32)) == [32, 16, 0]

    def test_order_candidates_default_preserves(self) -> None:
        assert search._order_candidates("ubatch", (128, 256, 512)) == [128, 256, 512]

    def test_count_executed(self) -> None:
        entries = (
            {"type": "session_start"},
            {"type": "baseline_run"},
            {"type": "baseline_run"},
            {"type": "trial", "status": "ok"},
            {"type": "trial", "status": "pruned"},
            {"type": "confirmation_run"},
            {"type": "depth_profile_run", "status": "ok"},
            {"type": "depth_profile_run", "status": "pruned"},
            {"type": "stability_rerun", "status": "ok"},
            {"type": "pair_check"},
            {"type": "thermal_retry", "status": "ok"},
            {"type": "quality_gate_run", "status": "ok"},
            {"type": "stage", "stage": "cli_validation", "status": "ok"},
            {"type": "stage"},
        )
        assert search._count_executed(entries) == 10

    def test_other_fields_drops_pruning_dimensions(self) -> None:
        config = {"gpu_layers": 33, "moe_cpu_layers": 8, "flash_attn": True}
        assert search._other_fields(config) == {"flash_attn": True}

    def test_improvement_zero_baseline(self) -> None:
        assert search._improvement(10.0, 0.0) == 0.0
        assert search._improvement(110.0, 100.0) == pytest.approx(10.0)

    def test_median_empty(self) -> None:
        assert search._median([]) == 0.0

    def test_opt_int(self) -> None:
        assert search._opt_int(None) is None
        assert search._opt_int("9999") == 9999

    def test_hardware_signature_detects_gpu_change(self) -> None:
        base = _hardware()
        other = _hardware(gpus=())
        assert search._hardware_signature(base) != search._hardware_signature(other)
        assert search._hardware_signature(base) == search._hardware_signature(_hardware())

    def test_ot_specs_are_anchored_and_bounded(self) -> None:
        specs = search._ot_specs(40, 10)
        assert 1 <= len(specs) <= 6
        assert all(spec.startswith("^blk\\.") and spec.endswith("=CPU$") for spec in specs)
        component_layers = specs[-1].split("(", 1)[1].split(")", 1)[0].split("|")
        assert len(component_layers) == 20

    def test_sample_attribution_and_neighbors(self) -> None:
        config = TrialConfig(
            gpu_layers=12,
            moe_cpu_layers=3,
            flash_attn=True,
            ubatch=512,
            batch=2048,
            threads=8,
            mmap=True,
            no_kv_offload=False,
            cache_type_k="f16",
            cache_type_v="f16",
        )
        assert search._sample_matches({"n_gpu_layers": 12, "n_threads": 8}, config)
        assert not search._sample_matches({"n_gpu_layers": 11}, config)
        assert search._neighbor_values((4, 8, 16), 8) == (4, 8, 16)


def test_batched_failure_falls_back_to_individual_trials(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLAMATUNE_FAKE_BATCH_CRASH", "1")
    session, hw, model, llama, options = _setup(
        tmp_path, fake_bin_dir, tiny_gguf, budget_trials=40, observe_vram=False
    )
    outcome = search.run_tuning(session, hw, model, llama, options)
    assert outcome.exit_code in (0, 1)
    assert list((session.dir / "batches").glob("*/command.json"))
    assert any(entry.get("type") == "trial" for entry in session.entries)


def test_sweep_preserves_executed_coverage_when_revalidation_exhausts_budget(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path,
        fake_bin_dir,
        tiny_gguf,
        budget_trials=18,
        ctx_size=8192,
        observe_vram=False,
    )
    engine = search._Engine(session, hw, model, llama, options)
    engine.baseline = BaselineResult(
        runs=3,
        pp=MetricStats(mean=50.0, stdev=0.0, cv=0.0, n=3),
        tg=MetricStats(mean=5.0, stdev=0.0, cv=0.0, n=3),
        noise_floor_cv=0.01,
        fallback=None,
        resolved_defaults=_envelope_config(0).to_dict(),
    )
    engine.incumbent_config = _envelope_config(0)
    engine.incumbent_pp = 50.0
    engine.incumbent_tg = 5.0
    engine.incumbent_score = 1.0

    monkeypatch.setattr(
        engine,
        "_execute_batch",
        lambda configs, dim: {
            cfg.trial_id: search._Trial("ok", 51.0, 5.1, 1.02) for cfg in configs
        },
    )
    monkeypatch.setattr(
        engine,
        "_probe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(search._BudgetExhaustedError()),
    )

    with pytest.raises(search._BudgetExhaustedError):
        engine._sweep("batch")

    assert engine.coverage["batch"]["executed"] == [512]


def test_batch_admission_preserves_finalization_reserve(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path,
        fake_bin_dir,
        tiny_gguf,
        budget_trials=8,
        observe_vram=False,
    )
    engine = search._Engine(session, hw, model, llama, options)
    engine.executed_count = 4
    base = _envelope_config(0)
    configs = [
        dataclasses.replace(base, batch=512),
        dataclasses.replace(base, batch=1024),
    ]
    monkeypatch.setattr(
        engine,
        "_run_child",
        lambda **_kwargs: pytest.fail("batch must not consume reserved confirmation capacity"),
    )

    assert engine._execute_batch(configs, "batch") == {}


def test_first_sigint_requests_graceful_stop(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf)
    engine = search._Engine(session, hw, model, llama, options)

    def fake_run() -> TuneOutcome:
        signal.raise_signal(signal.SIGINT)
        return TuneOutcome(session_dir=session.dir, analysis={}, exit_code=1)

    monkeypatch.setattr(engine, "_run", fake_run)
    outcome = engine.run()
    assert outcome.exit_code == 1
    assert engine._stop_requested
    assert any("best-so-far" in warning for warning in engine.extra_warnings)


def test_second_sigint_aborts_with_exit_4(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf)
    engine = search._Engine(session, hw, model, llama, options)

    def fake_run() -> TuneOutcome:
        signal.raise_signal(signal.SIGINT)
        signal.raise_signal(signal.SIGINT)
        raise AssertionError("second SIGINT must abort")

    monkeypatch.setattr(engine, "_run", fake_run)
    outcome = engine.run()
    assert outcome.exit_code == 4
    assert outcome.failure_stage == "interruption"
    assert outcome.failure_reason == "interrupted by user"
    assert session.entries[-1]["type"] == "session_end"
    assert session.entries[-1]["reason"] == "interrupted"


@pytest.mark.parametrize(
    ("backends", "gpus", "warned"),
    [
        ("CPU", None, True),
        (None, None, False),
        ("CUDA", None, False),
        ("CPU", (), False),
    ],
)
def test_backend_mismatch_warning_and_journal(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    backends: str | None,
    gpus: tuple[GPUInfo, ...] | None,
    warned: bool,
) -> None:
    hw = _hardware(gpus=gpus)
    session, _, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf, hardware=hw)
    engine = search._Engine(
        session, hw, model, dataclasses.replace(llama, backends=backends), options
    )
    engine._check_backend_mismatch()
    mismatch = [
        entry
        for entry in session.entries
        if entry.get("type") == "stage" and entry.get("stage") == "backend_mismatch"
    ]
    assert bool(mismatch) is warned
    assert any("CPU-only backends" in warning for warning in engine.extra_warnings) is warned


def test_backend_mismatch_restored_stage_is_not_duplicated(
    tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path
) -> None:
    session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf)
    engine = search._Engine(session, hw, model, dataclasses.replace(llama, backends="CPU"), options)
    session.append(
        {
            "type": "stage",
            "stage": "backend_mismatch",
            "gpus": ["Fake GPU"],
            "backends": "CPU",
        }
    )
    engine._check_backend_mismatch()
    engine._check_backend_mismatch()
    stages = [entry for entry in session.entries if entry.get("stage") == "backend_mismatch"]
    assert len(stages) == 1
    assert sum("CPU-only backends" in warning for warning in engine.extra_warnings) == 1


def test_sentinel_build_info_is_normalized_and_warned(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = bench.parse_bench_output

    def sentinel(raw: bytes | str) -> Any:
        sample = original(raw)
        sample.pp_entry["build_commit"] = "unknown"
        sample.pp_entry["build_number"] = 0
        sample.tg_entry["build_commit"] = "unknown"
        sample.tg_entry["build_number"] = 0
        return sample

    monkeypatch.setattr(bench, "parse_bench_output", sentinel)
    session, hw, model, llama, options = _setup(
        tmp_path, fake_bin_dir, tiny_gguf, baseline_only=True
    )
    outcome = search.run_tuning(session, hw, model, llama, options)
    recorded = json.loads((session.dir / "llamacpp.json").read_text())
    assert recorded["build_commit"] is None
    assert recorded["build_number"] is None
    assert any("no embedded version info" in warning for warning in outcome.analysis["warnings"])
    assert "Build commit: (unknown)" in (session.dir / "report.md").read_text()


def test_lossy_quality_gate_records_delta_and_warning(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
) -> None:
    perplexity = _write_fake_perplexity(
        tmp_path,
        "import sys\n"
        "value = 10.0 if '-ctk' in sys.argv else 9.0\n"
        "print(f'Final estimate: PPL = {value}')\n",
    )
    corpus = tmp_path / "quality.txt"
    corpus.write_text("quality corpus")
    session, hw, model, llama, options = _setup(
        tmp_path,
        fake_bin_dir,
        tiny_gguf,
        quality_corpus=corpus,
        observe_vram=False,
    )
    llama = dataclasses.replace(llama, perplexity_path=perplexity)
    engine = search._Engine(session, hw, model, llama, options)
    config = TrialConfig(
        gpu_layers=0,
        moe_cpu_layers=0,
        flash_attn=True,
        ubatch=512,
        batch=2048,
        threads=8,
        mmap=True,
        no_kv_offload=False,
        cache_type_k="q8_0",
        cache_type_v="f16",
    )
    engine._run_quality_gate(config)
    assert engine.quality_gate is not None
    assert engine.quality_gate["delta_pct"] == pytest.approx(100 / 9)
    assert any("perplexity" in warning for warning in engine.extra_warnings)


def test_quality_gate_skips_f16_and_records_parse_failure(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
) -> None:
    perplexity = _write_fake_perplexity(
        tmp_path,
        "import sys\nprint('no perplexity here', file=sys.stderr)\n",
    )
    corpus = tmp_path / "quality.txt"
    corpus.write_text("quality corpus")
    session, hw, model, llama, options = _setup(
        tmp_path, fake_bin_dir, tiny_gguf, quality_corpus=corpus, observe_vram=False
    )
    engine = search._Engine(
        session, hw, model, dataclasses.replace(llama, perplexity_path=perplexity), options
    )
    f16 = TrialConfig(
        gpu_layers=0,
        moe_cpu_layers=0,
        flash_attn=True,
        ubatch=512,
        batch=2048,
        threads=8,
        mmap=True,
        no_kv_offload=False,
        cache_type_k="f16",
        cache_type_v="f16",
    )
    engine._run_quality_gate(f16)
    assert engine.quality_gate is None

    engine._run_quality_gate(dataclasses.replace(f16, cache_type_k="q8_0"))
    assert engine.quality_gate is not None
    assert engine.quality_gate["status"] == "failed"
    assert engine.quality_gate["delta_pct"] is None


def test_ot_candidate_is_rejected_when_context_probe_fails(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path, fake_bin_dir, tiny_gguf, ot_search=True, ctx_size=8192
    )
    model = dataclasses.replace(model, moe=True, n_layer=40, ngl_all=41, expert_count=8)
    llama = dataclasses.replace(llama, capabilities=llama.capabilities | {"ot"})
    engine = search._Engine(session, hw, model, llama, options)
    incumbent = TrialConfig(
        gpu_layers=20,
        moe_cpu_layers=10,
        flash_attn=True,
        ubatch=512,
        batch=2048,
        threads=8,
        mmap=True,
        no_kv_offload=False,
        cache_type_k="f16",
        cache_type_v="f16",
    )
    metric = MetricStats(mean=1.0, stdev=0.0, cv=0.0, n=3)
    engine.baseline = BaselineResult(
        runs=3,
        pp=metric,
        tg=metric,
        noise_floor_cv=0.01,
        fallback=None,
        resolved_defaults=incumbent.to_dict(),
    )
    engine._set_incumbent(incumbent, 1.0, 1.0, 1.0)
    monkeypatch.setattr(engine, "_evaluate", lambda _cfg, _dim: search._Trial("ok", 2, 2, 2))
    monkeypatch.setattr(engine, "_probe", lambda _cfg, _purpose, _ctx: "gpu_resource")

    engine._ot_refine()
    assert engine.incumbent_config == incumbent


def test_ot_search_prerequisites_gate_evaluation(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf)
    engine = search._Engine(session, hw, model, llama, options)
    engine.incumbent_config = TrialConfig(
        gpu_layers=1,
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
    monkeypatch.setattr(
        engine, "_evaluate", lambda *_args: pytest.fail("gated OT search evaluated a trial")
    )
    engine._ot_refine()


def test_revalidate_session_without_winner_exits_3(
    tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path
) -> None:
    session, _hw, _model, _llama, _options = _setup(tmp_path, fake_bin_dir, tiny_gguf)
    (session.dir / "analysis.json").write_text('{"winner": null}\n')
    outcome = search.revalidate_session(session.dir)
    assert outcome.exit_code == 3
    assert outcome.failure_stage == "revalidation"
    assert outcome.failure_reason == "session has no confirmed winner"


def test_revalidate_confirmed_session_returns_comparison(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path, fake_bin_dir, tiny_gguf, budget_trials=40, observe_vram=False
    )
    tuned = search.run_tuning(session, hw, model, llama, options)
    assert tuned.analysis["winner"] is not None
    metadata_path = session.dir / "session.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["options"]["budget_trials"] = tuned.analysis["counts"]["budget_consumed"]
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr("llamatune.hardware.assess_hardware", lambda: hw)
    outcome = search.revalidate_session(session.dir)
    assert outcome.exit_code == 0
    assert outcome.analysis["status"] == "reproduced"
    assert outcome.analysis["comparison"]["within_noise"] is True

    identity = json.loads((session.dir / "llamacpp.json").read_text())
    identity["help_sha256"] = "0" * 64
    (session.dir / "llamacpp.json").write_text(json.dumps(identity))
    assert search.revalidate_session(session.dir).exit_code == 3


def test_load_warning_aggregates_only_contention_beyond_expected_self_load(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf)
    engine = search._Engine(session, hw, model, llama, options)
    samples = iter((9.0, 11.0, 12.0, 13.0, 14.0))
    monkeypatch.setattr(search, "_load_avg", lambda: next(samples))

    assert engine._record_load(expected_threads=8) == 9.0
    assert engine._record_load(expected_threads=8) == 11.0
    assert engine._record_load(expected_threads=8) == 12.0
    assert engine.load_warnings == set()
    assert engine._record_load(expected_threads=8) == 13.0
    assert engine._record_load(expected_threads=8) == 14.0
    assert len(engine.load_warnings) == 1
    warning = next(iter(engine.load_warnings))
    assert "2 of 5 invocations" in warning
    assert "peak 14.00, threshold 12.0" in warning


def test_load_aggregation_handles_unsupported_and_one_exceedance(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf)
    engine = search._Engine(session, hw, model, llama, options)
    samples: Any = iter((None, 13.0, 10.0))
    monkeypatch.setattr(search, "_load_avg", lambda: next(samples))

    assert engine._record_load(8) is None
    assert engine.load_warnings == set()
    assert engine._record_load(8) == 13.0
    assert engine._record_load(8) == 10.0
    assert len(engine.load_warnings) == 1
    assert "1 of 2 invocations" in next(iter(engine.load_warnings))


def _envelope_config(ngl: int) -> TrialConfig:
    return TrialConfig(
        gpu_layers=ngl,
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


@pytest.mark.parametrize("cached", [True, False])
def test_hill_climb_moe_reaches_improvement_after_cache_replay(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
    cached: bool,
) -> None:
    session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf)
    model = dataclasses.replace(model, moe=True, n_layer=32, ngl_all=33, expert_count=8)
    engine = search._Engine(session, hw, model, llama, options)
    cfg = _envelope_config(1)
    metric = MetricStats(mean=100.0, stdev=0.0, cv=0.0, n=3)
    engine.baseline = BaselineResult(
        runs=3,
        pp=metric,
        tg=metric,
        noise_floor_cv=0.01,
        fallback=None,
        resolved_defaults=cfg.to_dict(),
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


def test_required_context_validation_respects_exhausted_budget(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path, fake_bin_dir, tiny_gguf, ctx_size=8192, budget_trials=1
    )
    engine = search._Engine(session, hw, model, llama, options)
    engine.incumbent_config = _envelope_config(20)
    engine.executed_count = options.budget_trials

    monkeypatch.setattr(
        engine,
        "_run_child",
        lambda **_kwargs: pytest.fail("budget exhaustion must prevent context execution"),
    )
    engine._validate_recommendation()
    assert engine.context_validation is not None
    assert engine.context_validation["status"] == "skipped"
    assert any("context validation skipped" in warning for warning in engine.extra_warnings)


def test_confirmation_respects_exhausted_budget(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf, budget_trials=3)
    engine = search._Engine(session, hw, model, llama, options)
    engine.executed_count = options.budget_trials
    monkeypatch.setattr(
        engine,
        "_run_child",
        lambda **_kwargs: pytest.fail("budget exhaustion must prevent confirmation execution"),
    )

    confirmation = engine._confirm(_envelope_config(0))

    assert confirmation.confirmed is False
    assert any("confirmation skipped" in warning for warning in engine.extra_warnings)


def test_envelope_primary_failure_fallback_pass_and_resume_dedup(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path,
        fake_bin_dir,
        tiny_gguf,
        ctx_size=8192,
        ctx_ladder=(32768, 65536),
    )
    engine = search._Engine(session, hw, model, llama, options)
    primary, fallback = _envelope_config(20), _envelope_config(10)
    engine.context_validation = {"ctx": 8192, "status": "ok", "evidence": "probes/required"}
    monkeypatch.setattr(engine, "_context_candidates", lambda _base: [primary, fallback])
    calls: list[tuple[str, int]] = []

    def probe(config: TrialConfig, purpose: str, ctx: int | None) -> str:
        assert purpose == "context" and ctx is not None
        probe_id = search._probe_id(purpose, config, ctx)
        if probe_id in engine.probes:
            return str(engine.probes[probe_id]["status"])
        status = "oom" if config == primary else "ok"
        engine.executed_count += 1
        calls.append((config.trial_id, ctx))
        engine.probes[probe_id] = {"status": status}
        return status

    monkeypatch.setattr(engine, "_probe", probe)
    engine._run_context_envelope(primary, None)
    assert engine.context_envelope is not None
    assert [row["status"] for row in engine.context_envelope] == ["ok", "failed", "pruned"]
    assert engine.context_envelope[1]["fallback_config"] == fallback.to_dict()
    assert engine.context_envelope[2]["fallback_config"] == fallback.to_dict()
    assert engine._max_validated_context() == 8192
    first_calls = list(calls)
    engine._run_context_envelope(primary, None)
    assert calls == first_calls


def test_context_envelope_candidates_follow_authoritative_lossy_tier_order(
    tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path
) -> None:
    session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf, allow_lossy=True)
    engine = search._Engine(session, hw, model, llama, options)
    base = dataclasses.replace(_envelope_config(20), flash_attn=True)

    candidates = engine._context_envelope_candidates(base)

    assert [(candidate.cache_type_k, candidate.cache_type_v) for candidate in candidates] == [
        ("f16", "f16"),
        ("q8_0", "f16"),
        ("q8_0", "q8_0"),
        ("q4_0", "f16"),
        ("q4_0", "q4_0"),
    ]


@pytest.mark.parametrize(
    ("caps", "flash_attn", "cache_k", "cache_v", "expected"),
    (
        (set(), False, "f16", "f16", [("f16", "f16")]),
        (set(), True, "f16", "f16", [("f16", "f16")]),
        (
            {"ctk"},
            False,
            "f16",
            "f16",
            [("f16", "f16"), ("q8_0", "f16"), ("q4_0", "f16")],
        ),
        (
            {"ctk"},
            True,
            "f16",
            "f16",
            [("f16", "f16"), ("q8_0", "f16"), ("q4_0", "f16")],
        ),
        (
            {"ctv"},
            True,
            "f16",
            "f16",
            [("f16", "f16"), ("f16", "q8_0"), ("f16", "q4_0")],
        ),
        ({"ctv"}, False, "f16", "f16", [("f16", "f16")]),
        (
            {"ctk", "ctv"},
            False,
            "f16",
            "f16",
            [("f16", "f16"), ("q8_0", "f16"), ("q4_0", "f16")],
        ),
        (
            {"ctk", "ctv"},
            True,
            "f16",
            "f16",
            [
                ("f16", "f16"),
                ("q8_0", "f16"),
                ("q8_0", "q8_0"),
                ("q4_0", "f16"),
                ("q4_0", "q4_0"),
            ],
        ),
        (
            {"ctk", "ctv"},
            True,
            "q8_0",
            "f16",
            [
                ("q8_0", "f16"),
                ("q8_0", "q8_0"),
                ("q4_0", "f16"),
                ("q4_0", "q4_0"),
            ],
        ),
        (
            {"ctk", "ctv"},
            True,
            "f16",
            "q8_0",
            [("f16", "q8_0"), ("q8_0", "q8_0"), ("q4_0", "q8_0"), ("q4_0", "q4_0")],
        ),
        (
            {"ctk", "ctv"},
            True,
            "q4_0",
            "f16",
            [("q4_0", "f16"), ("q4_0", "q8_0"), ("q4_0", "q4_0")],
        ),
        (
            {"ctk", "ctv"},
            True,
            "f16",
            "q4_0",
            [("f16", "q4_0"), ("q8_0", "q4_0"), ("q4_0", "q4_0")],
        ),
        (
            {"ctk", "ctv"},
            True,
            "q8_0",
            "q4_0",
            [("q8_0", "q4_0"), ("q4_0", "q4_0")],
        ),
        (
            {"ctk", "ctv"},
            True,
            "q4_0",
            "q8_0",
            [("q4_0", "q8_0"), ("q4_0", "q4_0")],
        ),
        (
            {"ctk", "ctv"},
            True,
            "q8_0",
            "q8_0",
            [("q8_0", "q8_0"), ("q4_0", "q8_0"), ("q4_0", "q4_0")],
        ),
        ({"ctk", "ctv"}, True, "q4_0", "q4_0", [("q4_0", "q4_0")]),
    ),
)
def test_context_envelope_candidates_respect_capabilities_and_never_upgrade(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    caps: set[str],
    flash_attn: bool,
    cache_k: str,
    cache_v: str,
    expected: list[tuple[str, str]],
) -> None:
    session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf, allow_lossy=True)
    engine = search._Engine(session, hw, model, llama, options)
    engine.caps = frozenset(caps)
    base = dataclasses.replace(
        _envelope_config(20),
        flash_attn=flash_attn,
        cache_type_k=cache_k,
        cache_type_v=cache_v,
    )

    candidates = engine._context_envelope_candidates(base)

    assert [
        (candidate.cache_type_k, candidate.cache_type_v) for candidate in candidates
    ] == expected


def test_context_envelope_uses_q8_fallback_at_upper_rung(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path,
        fake_bin_dir,
        tiny_gguf,
        allow_lossy=True,
        ctx_size=8192,
        ctx_ladder=(32768,),
    )
    engine = search._Engine(session, hw, model, llama, options)
    primary = dataclasses.replace(_envelope_config(20), flash_attn=True)
    engine.context_validation = {"ctx": 8192, "status": "ok", "evidence": "probes/required"}

    def probe(config: TrialConfig, purpose: str, ctx: int | None) -> str:
        assert purpose == "context" and ctx == 32768
        engine.executed_count += 1
        return "ok" if config.cache_type_k == "q8_0" else "oom"

    monkeypatch.setattr(engine, "_probe", probe)
    engine._run_context_envelope(primary, None)

    assert engine.context_envelope is not None
    fallback = engine.context_envelope[1]["fallback_config"]
    assert fallback is not None
    assert fallback["gpu_layers"] == primary.gpu_layers
    assert fallback["cache_type_k"] == "q8_0"
    assert fallback["cache_type_v"] == "f16"
    metric = MetricStats(mean=1.0, stdev=0.0, cv=0.0, n=3)
    engine.baseline = BaselineResult(
        runs=3,
        pp=metric,
        tg=metric,
        noise_floor_cv=0.01,
        fallback=None,
        resolved_defaults=primary.to_dict(),
    )
    assert any("lossy KV-cache fallback" in warning for warning in engine._collect_warnings(None))


def test_max_validated_context_tracks_only_exact_recommendation() -> None:
    engine = object.__new__(search._Engine)
    engine.context_validation = {"ctx": 8192, "status": "ok"}
    engine.context_envelope = [
        {"ctx": 8192, "status": "ok", "fallback_config": None},
        {"ctx": 16384, "status": "ok", "fallback_config": None},
        {"ctx": 32768, "status": "failed", "fallback_config": {"gpu_layers": 1}},
    ]
    assert engine._max_validated_context() == 16384


def test_envelope_honors_per_rung_and_total_caps(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path,
        fake_bin_dir,
        tiny_gguf,
        ctx_size=8192,
        ctx_ladder=(16384, 32768, 65536, 98304),
    )
    engine = search._Engine(session, hw, model, llama, options)
    primary = _envelope_config(30)
    candidates = [primary, *(_envelope_config(value) for value in range(29, 10, -1))]
    engine.context_validation = {"ctx": 8192, "status": "ok", "evidence": "probes/required"}
    monkeypatch.setattr(engine, "_context_candidates", lambda _base: candidates)
    calls: list[int] = []

    def probe(_config: TrialConfig, _purpose: str, ctx: int | None) -> str:
        assert ctx is not None
        calls.append(ctx)
        engine.executed_count += 1
        return "oom"

    monkeypatch.setattr(engine, "_probe", probe)
    engine._run_context_envelope(primary, None)
    assert {ctx: calls.count(ctx) for ctx in set(calls)} == {16384: 8, 32768: 8, 65536: 8}
    assert engine.executed_count == 24
    assert engine.context_envelope is not None
    assert engine.context_envelope[-1]["status"] == "skipped"
    assert any(entry.get("stage") == "envelope_truncated" for entry in session.entries)


def test_envelope_per_rung_cap_stops_before_later_q4_fallback(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path,
        fake_bin_dir,
        tiny_gguf,
        allow_lossy=True,
        ctx_size=8192,
        ctx_ladder=(32768,),
    )
    engine = search._Engine(session, hw, model, llama, options)
    primary = dataclasses.replace(_envelope_config(20), flash_attn=True)
    lower = dataclasses.replace(primary, gpu_layers=10)
    engine.context_validation = {"ctx": 8192, "status": "ok", "evidence": "probes/required"}
    monkeypatch.setattr(engine, "_context_candidates", lambda _base: [primary, lower])
    calls: list[TrialConfig] = []

    def probe(config: TrialConfig, purpose: str, ctx: int | None) -> str:
        assert purpose == "context" and ctx == 32768
        calls.append(config)
        engine.executed_count += 1
        return "oom"

    monkeypatch.setattr(engine, "_probe", probe)
    engine._run_context_envelope(primary, None)

    assert len(calls) == 8
    assert calls[-1] == dataclasses.replace(lower, cache_type_k="q8_0", cache_type_v="q8_0")
    assert dataclasses.replace(lower, cache_type_k="q4_0", cache_type_v="f16") not in calls
    assert any(entry.get("stage") == "envelope_truncated" for entry in session.entries)


def test_envelope_budget_exhaustion_marks_remaining_rows_skipped(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path,
        fake_bin_dir,
        tiny_gguf,
        ctx_size=8192,
        ctx_ladder=(16384, 32768, 65536),
    )
    engine = search._Engine(session, hw, model, llama, options)
    primary = _envelope_config(20)
    candidates = [primary, _envelope_config(10), _envelope_config(5)]
    engine.context_validation = {"ctx": 8192, "status": "ok", "evidence": "probes/required"}
    monkeypatch.setattr(engine, "_context_candidates", lambda _base: candidates)

    def probe(_config: TrialConfig, _purpose: str, _ctx: int | None) -> str:
        if engine.executed_count >= 2:
            raise search._BudgetExhaustedError
        engine.executed_count += 1
        return "oom"

    monkeypatch.setattr(engine, "_probe", probe)
    engine._run_context_envelope(primary, None)
    assert engine.context_envelope is not None
    assert [row["status"] for row in engine.context_envelope[2:]] == ["skipped", "skipped"]
    truncated = [entry for entry in session.entries if entry.get("stage") == "envelope_truncated"]
    assert len(truncated) == 1
    assert truncated[0]["reason"] == "budget"


def test_depth_profile_reuses_cached_rows_and_ignores_failures(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path, fake_bin_dir, tiny_gguf, depth_profile=(0, 1024)
    )
    config = _envelope_config(10)
    session.append(
        {
            "type": "depth_profile_run",
            "trial_id": config.trial_id,
            "depth": 0,
            "status": "ok",
            "pp_mean": 100.0,
            "tg_mean": 20.0,
        }
    )
    session.append(
        {
            "type": "depth_profile_run",
            "trial_id": config.trial_id,
            "depth": 1024,
            "status": "oom",
            "pp_mean": None,
            "tg_mean": None,
        }
    )
    engine = search._Engine(session, hw, model, llama, options)
    monkeypatch.setattr(engine, "_can_execute", lambda: pytest.fail("cached profile executed"))
    engine._run_depth_profile(config)
    assert engine.depth_profile == {
        "trial_id": config.trial_id,
        "rows": [{"d": 0, "pp": 100.0, "tg": 20.0}],
    }


def test_depth_profile_budget_skip_is_journaled_once(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path, fake_bin_dir, tiny_gguf, depth_profile=(0, 1024)
    )
    engine = search._Engine(session, hw, model, llama, options)
    monkeypatch.setattr(engine, "_can_execute", lambda: False)
    engine._run_depth_profile(_envelope_config(10))
    engine._run_depth_profile(_envelope_config(10))
    skipped = [entry for entry in session.entries if entry.get("stage") == "depth_profile_skipped"]
    assert len(skipped) == 1
    assert engine.depth_profile is not None and engine.depth_profile["rows"] == []


def test_lossless_context_checks_probe_required_and_highest_passing_rung(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf, allow_lossy=True)
    engine = search._Engine(session, hw, model, llama, options)
    lossy = dataclasses.replace(_envelope_config(20), cache_type_k="q8_0")
    lossless = _envelope_config(10)
    rows: list[dict[str, Any]] = [
        {"ctx": 8192, "status": "ok", "fallback_config": None},
        {"ctx": 32768, "status": "failed", "fallback_config": lossless.to_dict()},
        {"ctx": 65536, "status": "failed", "fallback_config": None},
    ]
    calls: list[int] = []

    def probe(_config: TrialConfig, _purpose: str, ctx: int | None) -> str:
        assert ctx is not None
        calls.append(ctx)
        return "ok"

    monkeypatch.setattr(engine, "_probe", probe)
    engine._run_lossless_context_checks(
        rows,
        lossy,
        {"config": lossless.to_dict()},
    )
    assert calls == [8192, 32768]
    assert rows[0]["lossless_check"]["status"] == "ok"
    assert rows[1]["lossless_check"]["config"] == lossless.to_dict()


def _gpu_sample(temp: float | None, clock: float = 2400.0) -> GpuSample:
    return GpuSample(
        ts="fixed",
        vram_used_mb=100,
        vram_free_mb=1000,
        utilization_pct=0.0,
        temperature_c=temp,
        clocks_sm_mhz=clock,
    )


def _exec_result(tmp_path: Path, name: str) -> executor.ExecResult:
    stdout = tmp_path / f"{name}.json"
    stderr = tmp_path / f"{name}.log"
    stdout.write_text("[]")
    stderr.write_text("")
    return executor.ExecResult(
        exit_code=0,
        wall_s=1.0,
        timed_out=False,
        stdout=executor.CaptureInfo(stdout, "hash", 2, False),
        stderr=executor.CaptureInfo(stderr, "hash", 0, False),
        started="start",
        ended="end",
        env_names=(),
    )


def _measured(pp: float = 100.0, tg: float = 10.0) -> search._Measured:
    return search._Measured(
        "ok",
        MetricStats(mean=pp, stdev=0.0, cv=0.0, n=3),
        MetricStats(mean=tg, stdev=0.0, cv=0.0, n=3),
        {},
        None,
    )


def _unstable_measured() -> search._Measured:
    return search._Measured(
        "ok",
        MetricStats(mean=100.0, stdev=20.0, cv=0.2, n=3),
        MetricStats(mean=10.0, stdev=2.0, cv=0.2, n=3),
        {},
        None,
    )


@pytest.mark.parametrize(
    ("budget_trials", "temperatures", "expected_status", "expected_retried"),
    (
        (2, (70.0, 90.0), "unstable", False),
        (3, (70.0, 90.0, 70.0), "ok", True),
    ),
)
def test_stability_thermal_retry_budget_and_clean_replacement_are_accounted(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
    budget_trials: int,
    temperatures: tuple[float, ...],
    expected_status: str,
    expected_retried: bool,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path,
        fake_bin_dir,
        tiny_gguf,
        observe_vram=True,
        budget_trials=budget_trials,
    )
    engine = search._Engine(session, hw, model, llama, options)
    engine._finalizing = True
    engine.baseline = BaselineResult(
        runs=3,
        pp=MetricStats(mean=50.0, stdev=0.0, cv=0.0, n=3),
        tg=MetricStats(mean=5.0, stdev=0.0, cv=0.0, n=3),
        noise_floor_cv=0.01,
        fallback=None,
        resolved_defaults=_envelope_config(1).to_dict(),
    )
    results = iter(
        (
            _exec_result(tmp_path, f"run-{index}"),
            search._RunObservation(None, None, (_gpu_sample(temperature),)),
        )
        for index, temperature in enumerate(temperatures)
    )
    measurements = iter([_unstable_measured(), *[_measured()] * (len(temperatures) - 1)])
    detections = iter([False, True, *([False] if expected_retried else [])])
    monkeypatch.setattr(engine, "_run_child", lambda **kwargs: next(results))
    monkeypatch.setattr(engine, "_classify", lambda result, reps: next(measurements))
    monkeypatch.setattr(search, "_detect_gpu_throttle", lambda samples: next(detections))
    monkeypatch.setattr(engine, "_write_command", lambda *args: None)
    monkeypatch.setattr(engine, "_record_load", lambda threads: None)
    monkeypatch.setattr(engine, "_cooldown", lambda: None)

    trial = engine._execute_trial(_envelope_config(20), "test")

    record = engine.known[_envelope_config(20).trial_id]
    assert trial.status == expected_status
    assert (trial.score is not None) is expected_retried
    assert record["thermal_retried"] is expected_retried
    assert record["thermal_rejected"] is not expected_retried
    assert engine.executed_count == budget_trials
    assert search._count_executed(tuple(session.entries)) == budget_trials


def test_stability_thermal_rejection_is_unscored_and_resume_count_is_exact(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path, fake_bin_dir, tiny_gguf, observe_vram=True, budget_trials=8
    )
    engine = search._Engine(session, hw, model, llama, options)
    baseline_metric = MetricStats(mean=50.0, stdev=0.0, cv=0.0, n=3)
    engine.baseline = BaselineResult(
        runs=3,
        pp=baseline_metric,
        tg=baseline_metric,
        noise_floor_cv=0.01,
        fallback=None,
        resolved_defaults=_envelope_config(1).to_dict(),
    )
    results = iter(
        (
            (
                _exec_result(tmp_path, "primary"),
                search._RunObservation(None, None, (_gpu_sample(70.0),)),
            ),
            (
                _exec_result(tmp_path, "rerun"),
                search._RunObservation(None, None, (_gpu_sample(90.0),)),
            ),
            (
                _exec_result(tmp_path, "retry"),
                search._RunObservation(None, None, (_gpu_sample(89.0),)),
            ),
        )
    )
    measurements = iter((_unstable_measured(), _measured(105.0, 10.5), _measured(106.0, 10.6)))
    detections = iter((False, True, True))
    monkeypatch.setattr(engine, "_run_child", lambda **kwargs: next(results))
    monkeypatch.setattr(engine, "_classify", lambda result, reps: next(measurements))
    monkeypatch.setattr(search, "_detect_gpu_throttle", lambda samples: next(detections))
    monkeypatch.setattr(engine, "_write_command", lambda *args: None)
    monkeypatch.setattr(engine, "_record_load", lambda threads: None)
    monkeypatch.setattr(engine, "_cooldown", lambda: None)

    trial = engine._execute_trial(_envelope_config(20), "test")

    assert trial.status == "unstable"
    assert trial.score is None
    record = engine.known[_envelope_config(20).trial_id]
    assert record["thermal_rejected"] is True
    assert record["thermally_contaminated"] is True
    assert [entry["type"] for entry in session.entries[-4:]] == [
        "stability_rerun",
        "stage",
        "thermal_retry",
        "trial",
    ]
    assert engine.executed_count == 3
    assert search._count_executed(tuple(session.entries)) == 3
    entries_before_resume = tuple(session.entries)
    resumed = search._Engine(session, hw, model, llama, options)
    assert resumed.executed_count == engine.executed_count
    monkeypatch.setattr(resumed, "_run_child", lambda **kwargs: pytest.fail("trial repeated"))
    cached = resumed._evaluate(_envelope_config(20), "resume")
    assert cached.status == "unstable"
    assert cached.score is None
    assert tuple(session.entries) == entries_before_resume


def test_thermal_retry_progress_preserves_original_and_retry_run_ids(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingReporter:
        def __init__(self) -> None:
            self.events: list[Any] = []

        def emit(self, event: Any) -> None:
            self.events.append(event)

    session, hw, model, llama, options = _setup(
        tmp_path, fake_bin_dir, tiny_gguf, observe_vram=True, budget_trials=5
    )
    reporter = RecordingReporter()
    engine = search._Engine(session, hw, model, llama, options, reporter=reporter)
    pp_metric = MetricStats(mean=50.0, stdev=0.0, cv=0.0, n=3)
    tg_metric = MetricStats(mean=5.0, stdev=0.0, cv=0.0, n=3)
    engine.baseline = BaselineResult(
        runs=3,
        pp=pp_metric,
        tg=tg_metric,
        noise_floor_cv=0.01,
        fallback=None,
        resolved_defaults=_envelope_config(1).to_dict(),
    )
    results = iter(
        (
            (
                _exec_result(tmp_path, "primary"),
                search._RunObservation(None, None, (_gpu_sample(90.0),)),
            ),
            (
                _exec_result(tmp_path, "retry"),
                search._RunObservation(None, None, (_gpu_sample(70.0),)),
            ),
        )
    )
    measurements = iter((_measured(), _measured(110.0, 11.0)))
    detections = iter((True, False))
    monkeypatch.setattr(engine, "_run_child", lambda **kwargs: next(results))
    monkeypatch.setattr(engine, "_classify", lambda result, reps: next(measurements))
    monkeypatch.setattr(search, "_detect_gpu_throttle", lambda samples: next(detections))
    monkeypatch.setattr(engine, "_write_command", lambda *args: None)
    monkeypatch.setattr(engine, "_record_load", lambda threads: None)
    monkeypatch.setattr(engine, "_cooldown", lambda: None)

    engine._execute_trial(_envelope_config(20), "test")

    ids = [event.payload["id"] for event in reporter.events if event.kind == "exec_end"]
    assert ids == [_envelope_config(20).trial_id, f"{_envelope_config(20).trial_id}-thermal-retry"]


def test_confirmation_thermal_retry_progress_has_no_duplicate_original_id(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingReporter:
        def __init__(self) -> None:
            self.events: list[Any] = []

        def emit(self, event: Any) -> None:
            self.events.append(event)

    session, hw, model, llama, options = _setup(
        tmp_path, fake_bin_dir, tiny_gguf, observe_vram=True, budget_trials=10
    )
    reporter = RecordingReporter()
    engine = search._Engine(session, hw, model, llama, options, reporter=reporter)
    pp_metric = MetricStats(mean=50.0, stdev=0.0, cv=0.0, n=3)
    tg_metric = MetricStats(mean=5.0, stdev=0.0, cv=0.0, n=3)
    engine.baseline = BaselineResult(
        runs=3,
        pp=pp_metric,
        tg=tg_metric,
        noise_floor_cv=0.01,
        fallback=None,
        resolved_defaults=_envelope_config(1).to_dict(),
    )
    results = iter(
        (
            (
                _exec_result(tmp_path, "confirm-1"),
                search._RunObservation(None, None, (_gpu_sample(90.0),)),
            ),
            (
                _exec_result(tmp_path, "confirm-1-retry"),
                search._RunObservation(None, None, (_gpu_sample(70.0),)),
            ),
            (
                _exec_result(tmp_path, "confirm-2"),
                search._RunObservation(None, None, (_gpu_sample(70.0),)),
            ),
            (
                _exec_result(tmp_path, "confirm-3"),
                search._RunObservation(None, None, (_gpu_sample(70.0),)),
            ),
        )
    )
    detections = iter((True, False, False, False))
    monkeypatch.setattr(engine, "_run_child", lambda **kwargs: next(results))
    monkeypatch.setattr(engine, "_classify", lambda result, reps: _measured(110.0, 11.0))
    monkeypatch.setattr(search, "_detect_gpu_throttle", lambda samples: next(detections))
    monkeypatch.setattr(engine, "_write_command", lambda *args: None)
    monkeypatch.setattr(engine, "_record_load", lambda threads: None)
    monkeypatch.setattr(engine, "_cooldown", lambda: None)
    monkeypatch.setattr(engine, "_quiet_gate", lambda stage: None)
    config = _envelope_config(20)

    confirmation = engine._confirm(config)

    assert confirmation.confirmed is True
    ids = [event.payload["id"] for event in reporter.events if event.kind == "exec_end"]
    assert ids == [
        f"{config.trial_id}-confirm-1",
        f"{config.trial_id}-confirm-1-thermal-retry",
        f"{config.trial_id}-confirm-2",
        f"{config.trial_id}-confirm-3",
    ]


def test_thermal_retest_requires_opted_in_successful_run_samples(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path, fake_bin_dir, tiny_gguf, observe_vram=False
    )
    engine = search._Engine(session, hw, model, llama, options)
    result = _exec_result(tmp_path, "primary")
    observation = search._RunObservation(None, None, (_gpu_sample(90.0),))
    monkeypatch.setattr(search, "_detect_gpu_throttle", lambda samples: pytest.fail("detector ran"))
    outcome = engine._thermal_retest(
        kind="trial",
        trial_id="trial",
        run_id="trial",
        label="trial",
        argv=("bench",),
        timeout_s=1.0,
        reps=3,
        threads=1,
        retry_dir=tmp_path / "retry",
        retry_rel_dir="trials/trial/thermal-retry",
        result=result,
        observation=observation,
        measured=_measured(),
    )
    assert outcome.contaminated is False
    assert outcome.retried is False


def test_thermal_retest_budget_rejects_contaminated_measurement(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path, fake_bin_dir, tiny_gguf, observe_vram=True, budget_trials=1
    )
    engine = search._Engine(session, hw, model, llama, options)
    engine.executed_count = 1
    monkeypatch.setattr(search, "_detect_gpu_throttle", lambda samples: True)
    monkeypatch.setattr(engine, "_cooldown", lambda: None)
    monkeypatch.setattr(engine, "_run_child", lambda **kwargs: pytest.fail("retry ran"))
    outcome = engine._thermal_retest(
        kind="trial",
        trial_id="trial",
        run_id="trial",
        label="trial",
        argv=("bench",),
        timeout_s=1.0,
        reps=3,
        threads=1,
        retry_dir=tmp_path / "retry",
        retry_rel_dir="trials/trial/thermal-retry",
        result=_exec_result(tmp_path, "primary"),
        observation=search._RunObservation(None, None, (_gpu_sample(90.0),)),
        measured=_measured(),
    )
    assert outcome.measured.status == "unstable"
    assert outcome.retried is False
    stage = session.entries[-1]
    assert stage["retry_scheduled"] is False
    assert stage["retry_reason"] == "budget"


@pytest.mark.parametrize(
    ("stop_point", "expected_retried", "expected_executions"),
    (("before", False, 1), ("during", True, 2)),
)
def test_thermal_retry_stop_before_or_during_replacement_preserves_evidence(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop_point: str,
    expected_retried: bool,
    expected_executions: int,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path, fake_bin_dir, tiny_gguf, observe_vram=True, budget_trials=5
    )
    engine = search._Engine(session, hw, model, llama, options)
    metric = MetricStats(mean=50.0, stdev=0.0, cv=0.0, n=3)
    engine.baseline = BaselineResult(
        runs=3,
        pp=metric,
        tg=metric,
        noise_floor_cv=0.01,
        fallback=None,
        resolved_defaults=_envelope_config(1).to_dict(),
    )
    calls: list[str] = []

    def run_child(**kwargs: Any) -> tuple[executor.ExecResult, search._RunObservation]:
        run_id = str(kwargs["run_id"])
        calls.append(run_id)
        is_retry = run_id.endswith("-thermal-retry")
        if stop_point == "during" and is_retry:
            engine._stop_requested = True
        temperature = 70.0 if is_retry else 90.0
        return (
            _exec_result(tmp_path, run_id),
            search._RunObservation(None, None, (_gpu_sample(temperature),)),
        )

    def cooldown() -> None:
        if stop_point == "before":
            engine._stop_requested = True

    monkeypatch.setattr(engine, "_run_child", run_child)
    monkeypatch.setattr(engine, "_classify", lambda result, reps: _measured())
    monkeypatch.setattr(
        search,
        "_detect_gpu_throttle",
        lambda samples: bool(samples and samples[0].temperature_c == 90.0),
    )
    monkeypatch.setattr(engine, "_write_command", lambda *args: None)
    monkeypatch.setattr(engine, "_record_load", lambda threads: None)
    monkeypatch.setattr(engine, "_cooldown", cooldown)

    trial = engine._execute_trial(_envelope_config(20), "test")

    record = engine.known[_envelope_config(20).trial_id]
    assert engine._stop_requested is True
    assert record["thermal_retried"] is expected_retried
    assert engine.executed_count == expected_executions
    assert search._count_executed(tuple(session.entries)) == expected_executions
    assert sum(entry.get("type") == "trial" for entry in session.entries) == 1
    if stop_point == "before":
        assert calls == [_envelope_config(20).trial_id]
        assert trial.status == "unstable"
        assert record["thermal_rejected"] is True
        contamination = next(
            entry for entry in session.entries if entry.get("stage") == "thermal_contamination"
        )
        assert contamination["retry_reason"] == "stop"
    else:
        assert calls == [
            _envelope_config(20).trial_id,
            f"{_envelope_config(20).trial_id}-thermal-retry",
        ]
        assert trial.status == "ok"
        assert record["thermal_rejected"] is False
        assert sum(entry.get("type") == "thermal_retry" for entry in session.entries) == 1


def test_thermal_retest_retries_once_and_rejects_second_contamination(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path, fake_bin_dir, tiny_gguf, observe_vram=True, budget_trials=5
    )
    engine = search._Engine(session, hw, model, llama, options)
    engine.executed_count = 1
    retry_result = _exec_result(tmp_path, "retry-result")
    retry_observation = search._RunObservation(None, None, (_gpu_sample(89.0),))
    detections = iter((True, True))
    calls: list[str] = []
    monkeypatch.setattr(search, "_detect_gpu_throttle", lambda samples: next(detections))
    monkeypatch.setattr(engine, "_cooldown", lambda: None)
    monkeypatch.setattr(engine, "_record_load", lambda threads: None)
    monkeypatch.setattr(engine, "_write_command", lambda *args: None)
    monkeypatch.setattr(engine, "_emit_exec_end", lambda *args: None)

    def run_child(**kwargs: Any) -> tuple[executor.ExecResult, search._RunObservation]:
        calls.append(kwargs["run_id"])
        return retry_result, retry_observation

    monkeypatch.setattr(engine, "_run_child", run_child)
    monkeypatch.setattr(engine, "_classify", lambda result, reps: _measured(110.0, 11.0))
    outcome = engine._thermal_retest(
        kind="trial",
        trial_id="trial",
        run_id="trial",
        label="trial",
        argv=("bench",),
        timeout_s=1.0,
        reps=3,
        threads=1,
        retry_dir=tmp_path / "retry",
        retry_rel_dir="trials/trial/thermal-retry",
        result=_exec_result(tmp_path, "primary"),
        observation=search._RunObservation(None, None, (_gpu_sample(90.0),)),
        measured=_measured(),
    )
    assert calls == ["trial-thermal-retry"]
    assert outcome.retried is True
    assert outcome.retry_contaminated is True
    assert outcome.measured.status == "unstable"
    retry_entry = session.entries[-1]
    assert retry_entry["type"] == "thermal_retry"
    assert retry_entry["thermally_contaminated"] is True
    assert retry_entry["status"] == "unstable"
    assert search._count_executed(tuple(session.entries)) == 1


def test_thermal_retest_accepts_clean_replacement(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path, fake_bin_dir, tiny_gguf, observe_vram=True, budget_trials=5
    )
    engine = search._Engine(session, hw, model, llama, options)
    engine.executed_count = 1
    detections = iter((True, False))
    monkeypatch.setattr(search, "_detect_gpu_throttle", lambda samples: next(detections))
    monkeypatch.setattr(engine, "_cooldown", lambda: None)
    monkeypatch.setattr(engine, "_record_load", lambda threads: None)
    monkeypatch.setattr(engine, "_write_command", lambda *args: None)
    monkeypatch.setattr(engine, "_emit_exec_end", lambda *args: None)
    monkeypatch.setattr(
        engine,
        "_run_child",
        lambda **kwargs: (
            _exec_result(tmp_path, "clean-retry"),
            search._RunObservation(None, None, (_gpu_sample(70.0),)),
        ),
    )
    monkeypatch.setattr(engine, "_classify", lambda result, reps: _measured(110.0, 11.0))

    outcome = engine._thermal_retest(
        kind="trial",
        trial_id="trial",
        run_id="trial",
        label="trial",
        argv=("bench",),
        timeout_s=1.0,
        reps=3,
        threads=1,
        retry_dir=tmp_path / "retry",
        retry_rel_dir="trials/trial/thermal-retry",
        result=_exec_result(tmp_path, "primary"),
        observation=search._RunObservation(None, None, (_gpu_sample(90.0),)),
        measured=_measured(),
    )

    assert outcome.retried is True
    assert outcome.retry_contaminated is False
    assert outcome.measured.status == "ok"
    assert outcome.measured.pp.mean == 110.0


def test_thermal_rejection_warning_is_not_mislabeled_as_internal_variance(
    tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path
) -> None:
    session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf)
    engine = search._Engine(session, hw, model, llama, options)
    metric = MetricStats(mean=1.0, stdev=0.0, cv=0.0, n=3)
    engine.baseline = BaselineResult(
        runs=3,
        pp=metric,
        tg=metric,
        noise_floor_cv=0.01,
        fallback=None,
        resolved_defaults=_envelope_config(1).to_dict(),
    )
    engine.known["trial"] = {
        "status": "unstable",
        "thermal_rejected": True,
    }

    warnings = engine._collect_warnings(None)

    assert any("thermally contaminated" in warning for warning in warnings)
    assert not any("internal cv" in warning for warning in warnings)


def test_multi_gpu_oom_pruning_isolated_by_placement() -> None:
    base = _envelope_config(10)
    left = dataclasses.replace(base, tensor_split=(1.0, 0.0), split_mode="layer")
    right = dataclasses.replace(base, tensor_split=(0.0, 1.0), split_mode="layer")
    assert search._other_fields(left.to_dict()) != search._other_fields(right.to_dict())
    sample = dataclasses.replace(_gpu_sample(None), device_index=1)
    assert "device_index" not in search._gpu_sample_dict(sample, include_device=False)
    assert search._gpu_sample_dict(sample, include_device=True)["device_index"] == 1


def test_multi_gpu_observation_reports_per_device_peaks(
    tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path
) -> None:
    session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf)
    engine = search._Engine(session, hw, model, llama, options)
    trial = _envelope_config(10)
    samples = tuple(
        dataclasses.replace(_gpu_sample(None), device_index=device, vram_used_mb=used)
        for device, used in ((0, 100), (1, 200), (0, 350), (1, 500))
    )
    engine._observations[trial.trial_id] = search._RunObservation(samples[0], samples[-1], samples)
    result = engine._estimate_observation(trial, 1000.0)
    assert result is not None
    assert result["per_device"] == {
        "0": {"observed_used_delta_mb": 250, "observed_peak_used_mb": 350, "samples": 2},
        "1": {"observed_used_delta_mb": 300, "observed_peak_used_mb": 500, "samples": 2},
    }


def test_multi_gpu_placement_sweep_runs_bounded_grid_and_accepts_best(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpus = (
        GPUInfo(vendor="nvidia", name="a", vram_mb=16000, method="test"),
        GPUInfo(vendor="nvidia", name="b", vram_mb=24000, method="test"),
    )
    session, hw, model, llama, options = _setup(
        tmp_path,
        fake_bin_dir,
        tiny_gguf,
        hardware=_hardware(gpus=gpus),
        multi_gpu=True,
    )
    llama = dataclasses.replace(llama, capabilities=llama.capabilities | {"ts", "sm"})
    engine = search._Engine(session, hw, model, llama, options)
    engine.incumbent_config = _envelope_config(10)
    engine.incumbent_score = 1.0
    seen: list[TrialConfig] = []

    def evaluate(candidate: TrialConfig, dim: str) -> search._Trial:
        seen.append(candidate)
        score = 1.0 + len(seen) / 100.0
        return search._Trial("ok", 100.0, 40.0, score)

    monkeypatch.setattr(engine, "_can_execute", lambda: True)
    monkeypatch.setattr(engine, "_evaluate", evaluate)
    engine._multi_gpu_placement_sweep()
    assert len(seen) == 8
    assert engine.incumbent_config == seen[-1]
    assert {candidate.split_mode for candidate in seen} == {"layer", "row"}
    estimates = [
        entry for entry in session.entries if entry.get("stage") == "multi_gpu_placement_estimate"
    ]
    assert len(estimates) == 8
    assert all(entry["advisory"] is True for entry in estimates)


def test_thermal_cooldown_none_and_threshold_equality_preserve_fixed_wait(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path, fake_bin_dir, tiny_gguf, cooldown_s=2.0, thermal_wait_cap_s=60.0
    )
    engine = search._Engine(session, hw, model, llama, options)
    sleeps: list[float] = []
    samples = iter((None, _gpu_sample(75.0)))
    monkeypatch.setattr(search, "_sleep", sleeps.append)
    monkeypatch.setattr(search, "_sample_gpu_state", lambda: next(samples))
    engine._cooldown()
    engine._cooldown()
    assert sleeps == [2.0, 2.0]
    assert not any(entry.get("stage") == "thermal_pause" for entry in session.entries)


def test_thermal_cooldown_hot_then_cool_journals_pause(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path, fake_bin_dir, tiny_gguf, thermal_wait_cap_s=60.0
    )
    engine = search._Engine(session, hw, model, llama, options)
    sleeps: list[float] = []
    samples = iter((_gpu_sample(80.0), _gpu_sample(70.0)))
    monkeypatch.setattr(search, "_sleep", sleeps.append)
    monkeypatch.setattr(search, "_sample_gpu_state", lambda: next(samples))
    engine._cooldown()
    pauses = [entry for entry in session.entries if entry.get("stage") == "thermal_pause"]
    assert sleeps == [10.0]
    assert len(pauses) == 1
    assert pauses[0]["reason"] == "temperature"
    assert pauses[0]["temperature_before_c"] == 80.0
    assert pauses[0]["temperature_after_c"] == 70.0
    assert engine.thermal_pause_count == 1
    assert engine.thermal_wait_s == 10.0
    resumed = search._Engine(session, hw, model, llama, options)
    assert resumed.thermal_pause_count == 1
    assert resumed.thermal_wait_s == 10.0
    telemetry = resumed._telemetry_summary(_envelope_config(10))
    assert telemetry is not None and telemetry["thermal_pause_count"] == 1


def test_thermal_cooldown_cap_and_budget_are_hard_bounds(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path, fake_bin_dir, tiny_gguf, thermal_wait_cap_s=25.0
    )
    engine = search._Engine(session, hw, model, llama, options)
    sleeps: list[float] = []
    monkeypatch.setattr(search, "_sleep", sleeps.append)
    monkeypatch.setattr(search, "_sample_gpu_state", lambda: _gpu_sample(90.0))
    engine._cooldown()
    assert sleeps == [10.0, 10.0, 5.0]
    assert session.entries[-1]["cap_reached"] is True

    budget_options = dataclasses.replace(options, budget_minutes=1.0, thermal_wait_cap_s=60.0)
    budget_engine = search._Engine(session, hw, model, llama, budget_options)
    budget_engine.start = 0.0
    sleeps.clear()
    monkeypatch.setattr(search, "_monotonic", lambda: 55.0)
    budget_engine._cooldown()
    assert sleeps == [5.0]
    assert session.entries[-1]["budget_exhausted"] is True


def test_thermal_cooldown_detects_throttle_and_cap_zero_disables(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path, fake_bin_dir, tiny_gguf, thermal_wait_cap_s=60.0
    )
    engine = search._Engine(session, hw, model, llama, options)
    engine._thermal_samples = [
        _gpu_sample(72.0, clock) for clock in (2400, 2400, 2300, 2200, 2100, 2000, 1900)
    ]
    samples = iter((_gpu_sample(72.0, 1800), _gpu_sample(72.0, 2400), _gpu_sample(72.0, 2400)))
    sleeps: list[float] = []
    monkeypatch.setattr(search, "_sleep", sleeps.append)
    monkeypatch.setattr(search, "_sample_gpu_state", lambda: next(samples))
    engine._cooldown()
    assert sleeps == [10.0, 10.0]
    assert session.entries[-1]["reason"] == "throttle"

    disabled = search._Engine(
        session, hw, model, llama, dataclasses.replace(options, thermal_wait_cap_s=0.0)
    )
    monkeypatch.setattr(
        search, "_sample_gpu_state", lambda: pytest.fail("disabled thermal sampler ran")
    )
    disabled._cooldown()


def test_thermal_cooldown_honors_clock_throttle_while_gpu_is_cool(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path, fake_bin_dir, tiny_gguf, thermal_wait_cap_s=60.0
    )
    engine = search._Engine(session, hw, model, llama, options)
    engine._thermal_samples = [
        _gpu_sample(45.0, clock) for clock in (2400, 2400, 2300, 2200, 2100, 2000, 1900)
    ]
    sleeps: list[float] = []
    monkeypatch.setattr(search, "_sleep", sleeps.append)
    monkeypatch.setattr(search, "_sample_gpu_state", lambda: _gpu_sample(45.0, 1800))
    engine._cooldown()
    assert sleeps == [10.0] * 6
    pauses = [entry for entry in session.entries if entry.get("stage") == "thermal_pause"]
    assert len(pauses) == 6
    assert all(entry["reason"] == "throttle" for entry in pauses)
    assert pauses[-1]["cap_reached"] is True


@pytest.mark.parametrize("budget", [6, 60])
def test_confirmation_interrupt_resume_does_not_repeat_completed_runs(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
    budget: int,
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
        for p in (session.dir / "trials").rglob("*")
        if p.is_file()
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


@pytest.mark.parametrize(
    "change",
    [
        {"status": "timeout"},
        {"pp_mean": float("nan")},
        {"tg_mean": 0.0},
        {"thermally_contaminated": True},
        {"confirmation_key": {}},
        {"purpose": "revalidation"},
    ],
)
def test_confirmation_reuse_rejects_invalid_or_foreign_evidence(
    tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path, change: dict[str, Any]
) -> None:
    session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf)
    engine = search._Engine(session, hw, model, llama, options)
    cfg = _envelope_config(1)
    row = {
        "type": "confirmation_run",
        "trial_id": cfg.trial_id,
        "run": 1,
        "confirmation_key": engine._confirmation_key(cfg),
        "purpose": "confirmation",
        "status": "ok",
        "pp_mean": 100.0,
        "tg_mean": 20.0,
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
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(tmp_path, fake_bin_dir, tiny_gguf)
    engine = search._Engine(session, hw, model, llama, options)
    engine._establish_baseline()
    cfg = _envelope_config(1)
    for index in range(1, options.baseline_runs + 1):
        session.append(
            {
                "type": "confirmation_run",
                "trial_id": cfg.trial_id,
                "run": index,
                "confirmation_key": engine._confirmation_key(cfg),
                "purpose": "confirmation",
                "status": "ok",
                "pp_mean": 100.0,
                "tg_mean": 20.0,
            }
        )
    engine.executed_count = options.budget_trials
    monkeypatch.setattr(engine, "_run_child", lambda **_kwargs: pytest.fail("cached run executed"))
    result = engine._confirm(cfg)
    assert result.pp is not None and result.tg is not None
    assert engine.executed_count == options.budget_trials


def test_confirmation_capture_allocation_preserves_previous_attempt(
    tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path
) -> None:
    session, _, _, _, _ = _setup(tmp_path, fake_bin_dir, tiny_gguf)
    first = session.confirmation_dir("trial", 1)
    (first / "stdout.json").write_bytes(b"retained")
    second = session.confirmation_dir("trial", 1)
    assert second != first and second.is_dir()
    assert (first / "stdout.json").read_bytes() == b"retained"
    with pytest.raises(SessionPathError):
        session.confirmation_dir("../../../outside", 1)


def _confirmation_rows(session_dir: Path) -> list[dict[str, Any]]:
    return [
        entry
        for entry in Session.load(session_dir).entries
        if entry.get("type") == "confirmation_run"
    ]


def _run_confirmed_session(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    *,
    budget_trials: int = 40,
) -> tuple[Session, HardwareReport, LlamaCppReport]:
    session, hw, model, llama, options = _setup(
        tmp_path,
        fake_bin_dir,
        tiny_gguf,
        budget_trials=budget_trials,
        observe_vram=False,
    )
    outcome = search.run_tuning(session, hw, model, llama, options)
    assert outcome.exit_code == 0
    assert outcome.analysis["winner"] is not None
    assert len(_confirmation_rows(session.dir)) == options.baseline_runs
    return session, hw, llama


def _rebuild_fake_bench(fake_bin_dir: Path) -> LlamaCppReport:
    bench_path = fake_bin_dir / ("llama-bench.cmd" if sys.platform == "win32" else "llama-bench")
    previous = discover_llama(fake_bin_dir)
    comment = (
        b"\r\nREM rebuilt without help changes\r\n"
        if sys.platform == "win32"
        else b"\n# rebuilt without help changes\n"
    )
    bench_path.write_bytes(bench_path.read_bytes() + comment)
    if sys.platform != "win32":
        bench_path.chmod(0o755)
    rebuilt = discover_llama(fake_bin_dir)
    assert rebuilt.bench_sha256 != previous.bench_sha256
    assert rebuilt.help_sha256 == previous.help_sha256
    return rebuilt


def test_public_resume_rebuild_replaces_complete_confirmation_cache(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, llama = _run_confirmed_session(tmp_path, fake_bin_dir, tiny_gguf)
    old_rows = _confirmation_rows(session.dir)
    rebuilt = _rebuild_fake_bench(fake_bin_dir)
    restored: list[tuple[str | None, str | None]] = []
    original_load = search._Engine._load_baseline_stage

    def tracked_load(engine: search._Engine, stage: dict[str, Any]) -> None:
        before = engine.llama.bench_sha256
        original_load(engine, stage)
        restored.append((before, engine.llama.bench_sha256))

    monkeypatch.setattr(search._Engine, "_load_baseline_stage", tracked_load)
    monkeypatch.setattr("llamatune.hardware.assess_hardware", lambda: hw)

    outcome = search.resume_tuning(session.dir)

    rows = _confirmation_rows(session.dir)
    new_rows = rows[len(old_rows) :]
    assert restored == [(rebuilt.bench_sha256, llama.bench_sha256)]
    assert outcome.exit_code == 0
    assert outcome.analysis["winner"]["confirmed"] is True
    assert len(new_rows) == 3
    assert all(row["confirmation_key"]["bench_sha256"] == rebuilt.bench_sha256 for row in new_rows)


def test_public_resume_rebuild_replaces_partial_confirmation_cache(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, model, llama, options = _setup(
        tmp_path, fake_bin_dir, tiny_gguf, budget_trials=40, observe_vram=False
    )
    original_append = search._Engine._append

    def interrupt_after_first_confirmation(engine: search._Engine, record: dict[str, Any]) -> None:
        original_append(engine, record)
        if record.get("type") == "confirmation_run":
            raise KeyboardInterrupt

    with monkeypatch.context() as interrupted:
        interrupted.setattr(search._Engine, "_append", interrupt_after_first_confirmation)
        assert search.run_tuning(session, hw, model, llama, options).exit_code == 4
    old_rows = _confirmation_rows(session.dir)
    assert len(old_rows) == 1
    rebuilt = _rebuild_fake_bench(fake_bin_dir)
    monkeypatch.setattr("llamatune.hardware.assess_hardware", lambda: hw)

    outcome = search.resume_tuning(session.dir)

    rows = _confirmation_rows(session.dir)
    new_rows = rows[len(old_rows) :]
    assert outcome.exit_code == 0
    assert outcome.analysis["winner"]["confirmed"] is True
    assert [row["run"] for row in rows] == [1, 1, 2, 3]
    assert len(new_rows) == 3
    assert all(row["confirmation_key"]["bench_sha256"] == rebuilt.bench_sha256 for row in new_rows)


def test_public_resume_unchanged_build_reuses_complete_confirmation_cache(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, _llama = _run_confirmed_session(tmp_path, fake_bin_dir, tiny_gguf)
    old_rows = _confirmation_rows(session.dir)
    monkeypatch.setattr("llamatune.hardware.assess_hardware", lambda: hw)

    outcome = search.resume_tuning(session.dir)

    assert outcome.exit_code == 0
    assert outcome.analysis["winner"]["confirmed"] is True
    assert _confirmation_rows(session.dir) == old_rows


def test_public_resume_without_current_hash_records_uncacheable_confirmation_rows(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, _llama = _run_confirmed_session(
        tmp_path, fake_bin_dir, tiny_gguf, budget_trials=60
    )
    old_rows = _confirmation_rows(session.dir)
    _rebuild_fake_bench(fake_bin_dir)
    original_discover = discover_llama

    def discover_without_bench_hash(path: Path | None = None) -> LlamaCppReport:
        return dataclasses.replace(original_discover(path), bench_sha256=None)

    monkeypatch.setattr("llamatune.llama.discover_llama", discover_without_bench_hash)
    monkeypatch.setattr("llamatune.hardware.assess_hardware", lambda: hw)

    first = search.resume_tuning(session.dir)
    first_new_rows = _confirmation_rows(session.dir)[len(old_rows) :]
    second = search.resume_tuning(session.dir)
    all_rows = _confirmation_rows(session.dir)

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert len(first_new_rows) == 3
    assert len(all_rows) == len(old_rows) + 6
    assert all(row["confirmation_key"]["bench_sha256"] is None for row in all_rows[len(old_rows) :])


def test_public_resume_rebuild_at_budget_does_not_promote_unconfirmed_winner(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, _llama = _run_confirmed_session(
        tmp_path, fake_bin_dir, tiny_gguf, budget_trials=33
    )
    old_rows = _confirmation_rows(session.dir)
    _rebuild_fake_bench(fake_bin_dir)
    monkeypatch.setattr("llamatune.hardware.assess_hardware", lambda: hw)

    outcome = search.resume_tuning(session.dir)

    assert (outcome.analysis["winner"] or {}).get("confirmed") is not True
    assert _confirmation_rows(session.dir) == old_rows
    assert any("confirmation skipped" in warning for warning in outcome.analysis["warnings"])


def test_revalidation_records_live_identity_after_baseline_restore(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, hw, _llama = _run_confirmed_session(tmp_path, fake_bin_dir, tiny_gguf)
    old_rows = _confirmation_rows(session.dir)
    rebuilt = _rebuild_fake_bench(fake_bin_dir)
    monkeypatch.setattr("llamatune.hardware.assess_hardware", lambda: hw)

    outcome = search.revalidate_session(session.dir)

    new_rows = _confirmation_rows(session.dir)[len(old_rows) :]
    assert outcome.exit_code == 0
    assert len(new_rows) == 3
    assert all(row["purpose"] == "revalidation" for row in new_rows)
    assert all(row["confirmation_key"]["bench_sha256"] == rebuilt.bench_sha256 for row in new_rows)
