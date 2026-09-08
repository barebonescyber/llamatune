"""Integration scenarios 1-3: full tune pipeline against the fake llama-bench.

Scenario 1 runs end-to-end through the real CLI (Typer runner); scenarios 2
and 3 drive the documented internal API (Session.create + run_tuning) so a
deterministic HardwareReport can be injected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from llamatune.cli import app
from llamatune.llama import discover_llama
from llamatune.model import inspect_model
from llamatune.search import _count_executed, resume_tuning, run_tuning
from llamatune.session import Session
from llamatune.types import GPUInfo, HardwareReport, TuneOptions

runner = CliRunner()

_ANALYSIS_KEYS = {
    "schema_version",
    "target",
    "baseline",
    "default_probe",
    "baseline_kind",
    "feasibility",
    "context_validation",
    "cli_validation",
    "estimate_vs_observed",
    "coverage",
    "quality_gate",
    "telemetry",
    "counts",
    "pareto",
    "top",
    "winner",
    "lossless_winner",
    "warnings",
}
_BASELINE_KEYS = {
    "runs",
    "pp",
    "tg",
    "noise_floor_cv",
    "fallback",
    "kind",
    "resolved_defaults",
}
_COUNT_KEYS = {
    "budget_consumed",
    "executed",
    "ok",
    "unstable",
    "oom",
    "cuda_error",
    "gpu_resource",
    "timeout",
    "crash",
    "parse_error",
    "pruned",
}
_METRIC_KEYS = {"mean", "stdev", "cv", "n"}
_SUMMARY_KEYS = {"trial_id", "status", "config", "pp_mean", "tg_mean", "score", "flags"}
_EMISSION_FILES = ("analysis.json", "report.md", "recommended.json", "recommended.sh")


def _gpu_hardware(vram_mb: int = 24000, llama_bin: Path | None = None) -> HardwareReport:
    return HardwareReport(
        os_name="Linux",
        arch="x86_64",
        cpu_model="Fake CPU",
        physical_cores=8,
        logical_cores=16,
        perf_cores=None,
        ram_mb=32768,
        gpus=(GPUInfo(vendor="nvidia", name="Fake GPU", vram_mb=vram_mb, method="test"),),
        warnings=(),
    )


def _no_gpu_hardware() -> HardwareReport:
    return HardwareReport(
        os_name="Linux",
        arch="x86_64",
        cpu_model="Fake CPU",
        physical_cores=8,
        logical_cores=16,
        perf_cores=None,
        ram_mb=32768,
        gpus=(),
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


def _run_internal(
    tmp_path: Path,
    fake_bin_dir: Path,
    gguf: Path,
    hardware: HardwareReport,
    **option_overrides: Any,
) -> tuple[Any, Session]:
    llama = discover_llama(fake_bin_dir)
    model = inspect_model(gguf)
    options = _options(tmp_path / "sessions", fake_bin_dir, **option_overrides)
    session = Session.create(
        options.sessions_dir,
        model=model,
        hardware=hardware,
        llama=llama,
        options=options,
        argv=["llamatune", "tune", str(gguf)],
    )
    outcome = run_tuning(session, hardware, model, llama, options)
    assert outcome.analysis["counts"]["budget_consumed"] == _count_executed(session.entries)
    return outcome, session


def _journal_trials(session_dir: Path) -> list[dict[str, Any]]:
    trials = []
    with (session_dir / "journal.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            entry = json.loads(line)
            if entry.get("type") == "trial":
                trials.append(entry)
    return trials


def _assert_analysis_schema(analysis: dict[str, Any]) -> None:
    assert set(analysis) == _ANALYSIS_KEYS
    assert analysis["schema_version"] == 2
    assert set(analysis["baseline"]) == _BASELINE_KEYS
    assert set(analysis["baseline"]["pp"]) == _METRIC_KEYS
    assert set(analysis["baseline"]["tg"]) == _METRIC_KEYS
    assert set(analysis["counts"]) == _COUNT_KEYS
    for summary in [*analysis["top"], *analysis["pareto"]]:
        assert set(summary) == _SUMMARY_KEYS
    assert len(analysis["top"]) <= 10


def test_depth_is_uniform_and_profile_is_isolated(
    tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path
) -> None:
    outcome, session = _run_internal(
        tmp_path,
        fake_bin_dir,
        tiny_gguf,
        _gpu_hardware(),
        depth=1024,
        depth_profile=(0, 1024),
        ctx_size=8192,
        observe_vram=False,
    )
    assert outcome.exit_code in (0, 1)
    commands = [
        (path, json.loads(path.read_text()))
        for path in session.dir.rglob("command.json")
        if "quality" not in path.parts and "cli-validation" not in path.parts
    ]
    assert commands
    for path, command in commands:
        argv = command["argv"]
        if "probes" in path.parts and "-p" in argv and argv[argv.index("-p") + 1] == "8192":
            assert "-d" not in argv
        else:
            expected_depth = (
                path.parent.name.removeprefix("depth-") if "depth-" in path.parent.name else "1024"
            )
            assert argv[argv.index("-d") + 1] == expected_depth
    if outcome.analysis.get("winner") is not None:
        assert [row["d"] for row in outcome.analysis["depth_profile"]["rows"]] == [0, 1024]


def test_context_ladder_emits_envelope_and_keeps_primary_context(
    tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path
) -> None:
    outcome, session = _run_internal(
        tmp_path,
        fake_bin_dir,
        tiny_gguf,
        _gpu_hardware(),
        ctx_size=8192,
        ctx_ladder=(16384,),
        observe_vram=False,
    )
    assert outcome.exit_code == 0
    envelope = outcome.analysis["context_envelope"]
    assert [row["ctx"] for row in envelope] == [8192, 16384]
    assert envelope[0]["status"] == "ok"
    assert "-c 8192" in (session.dir / "recommended.sh").read_text()


class TestScenario1UnconstrainedViaCli:
    """Scenario 1: unconstrained tune through the real CLI finds fa=1."""

    @pytest.fixture()
    def cli_result(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[Any, Path]:
        # The CLI assesses the host hardware; inject the deterministic
        # GPU-equipped report the fake bench's performance model assumes.
        monkeypatch.setattr("llamatune.hardware.assess_hardware", _gpu_hardware)
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
                "--budget-trials",
                "40",
                "--baseline-runs",
                "3",
                "--json",
            ],
        )
        (session_dir,) = (path for path in sessions.iterdir() if (path / "session.json").is_file())
        return result, session_dir

    def test_exit_0_and_winner_at_known_optimum(self, cli_result: tuple[Any, Path]) -> None:
        result, session_dir = cli_result
        assert result.exit_code == 0

        analysis = json.loads((session_dir / "analysis.json").read_text())
        winner = analysis["winner"]
        assert winner is not None
        assert winner["confirmed"] is True
        # Known fake-bench optimum: fa=1, ub=512, ngl=all (DESIGN §15.1).
        assert winner["config"]["flash_attn"] is True
        assert winner["config"]["ubatch"] == 512
        assert winner["config"]["gpu_layers"] == 33
        assert winner["improvement_pct"]["score"] > 0
        assert winner["improvement_pct"]["pp"] > 0

    def test_json_stdout_matches_analysis_file(self, cli_result: tuple[Any, Path]) -> None:
        result, session_dir = cli_result
        stdout_analysis = json.loads(result.output)
        file_analysis = json.loads((session_dir / "analysis.json").read_text())
        assert stdout_analysis == file_analysis

    def test_emission_files_and_schema(self, cli_result: tuple[Any, Path]) -> None:
        _, session_dir = cli_result
        for name in _EMISSION_FILES:
            assert (session_dir / name).is_file(), name

        analysis = json.loads((session_dir / "analysis.json").read_text())
        _assert_analysis_schema(analysis)
        assert analysis["baseline"]["fallback"] is None
        assert (session_dir / "batches").is_dir()
        assert any(entry.get("batch_id") for entry in _journal_trials(session_dir))
        for details in analysis["coverage"].values():
            accounted = sum(
                len(details.get(key) or [])
                for key in ("executed", "cached_hit", "pruned", "skipped")
            )
            assert accounted == len(details["candidates"])

        report_text = (session_dir / "report.md").read_text()
        assert report_text.strip()
        assert "# llamatune report" in report_text

        recommended = json.loads((session_dir / "recommended.json").read_text())
        assert recommended["confirmed"] is True
        assert recommended["config"] == analysis["winner"]["config"]
        assert "-fa" in recommended["bench_flags"]

        sh_text = (session_dir / "recommended.sh").read_text()
        assert "llama-server" in sh_text
        assert "-fa on" in sh_text


class TestScenario2VramConstrainedMoe:
    """Scenario 2: VRAM-constrained MoE exercises OOM fallback + pruning."""

    @pytest.fixture()
    def outcome_and_session(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_moe_gguf: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[Any, Session]:
        monkeypatch.setenv("LLAMATUNE_FAKE_VRAM_MB", "2500")
        return _run_internal(
            tmp_path,
            fake_bin_dir,
            tiny_moe_gguf,
            _gpu_hardware(vram_mb=2500),
            budget_trials=60,
        )

    def test_baseline_fallback_recorded(self, outcome_and_session: tuple[Any, Session]) -> None:
        outcome, session = outcome_and_session
        assert outcome.analysis["baseline"]["fallback"] == "cpu"
        assert any("fell back to CPU" in w for w in outcome.analysis["warnings"])
        # The failed default-settings probe left OOM evidence in the journal.
        baseline_entries = [e for e in session.entries if e.get("type") == "baseline_run"]
        assert baseline_entries[0]["status"] == "oom"
        assert all(e["status"] == "ok" for e in baseline_entries[1:])

    def test_oom_and_pruned_trials_journaled(
        self, outcome_and_session: tuple[Any, Session]
    ) -> None:
        _, session = outcome_and_session
        trials = _journal_trials(session.dir)
        probes = [e for e in session.entries if e.get("type") == "probe"]
        failed_probes = [p for p in probes if p["status"] == "oom"]
        pruned_probes = [p for p in probes if p["status"] == "pruned"]
        assert failed_probes
        # Pruning is opportunistic: an exact boundary search need not produce
        # a dominated follow-on candidate.
        # Boundary evidence is kept separate from scored trial measurements.
        failed_ids = {p["probe_id"] for p in failed_probes}
        assert all(p["pruned_from"] in failed_ids for p in pruned_probes)
        # OOM boundary lands exactly at ncmoe=20: everything strictly below
        # on the same context OOMs or is pruned, never measured ok.
        for trial in trials:
            cfg = trial["config"]
            if cfg["gpu_layers"] == 33 and cfg["moe_cpu_layers"] < 20:
                assert trial["status"] in ("oom", "pruned")

    def test_winner_in_moe_offload_region(self, outcome_and_session: tuple[Any, Session]) -> None:
        outcome, _ = outcome_and_session
        assert outcome.exit_code == 0
        winner = outcome.analysis["winner"]
        assert winner is not None
        assert winner["config"]["gpu_layers"] == 33
        assert winner["config"]["moe_cpu_layers"] >= 20
        assert winner["improvement_pct"]["score"] > 0
        assert outcome.analysis["counts"]["budget_consumed"] <= 60


class TestScenario3NoGpu:
    """Scenario 3: no-GPU hardware pins every trial at gpu_layers=0."""

    def test_completes_with_all_trials_on_cpu(
        self, tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path
    ) -> None:
        outcome, session = _run_internal(
            tmp_path, fake_bin_dir, tiny_gguf, _no_gpu_hardware(), budget_trials=60
        )

        assert outcome.exit_code in (0, 1)
        _assert_analysis_schema(outcome.analysis)
        for name in _EMISSION_FILES:
            assert (session.dir / name).is_file(), name

        trials = _journal_trials(session.dir)
        assert trials, "the search should have executed trials"
        assert all(t["config"]["gpu_layers"] == 0 for t in trials)
        assert outcome.analysis["baseline"]["resolved_defaults"]["gpu_layers"] == 0
        # Journal bookkeeping: session_end carries the exit code.
        end = session.entries[-1]
        assert end["type"] == "session_end"
        assert end["exit_code"] == outcome.exit_code


class TestScenario4CpuBackendOnGpuHost:
    """A CPU-only build pins GPU offload despite physical GPU presence."""

    def test_tune_and_resume_never_execute_gpu_trials(
        self,
        tmp_path: Path,
        fake_bin_dir: Path,
        tiny_gguf: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLAMATUNE_FAKE_BACKEND", "CPU")
        outcome, session = _run_internal(
            tmp_path,
            fake_bin_dir,
            tiny_gguf,
            _gpu_hardware(),
            budget_trials=8,
        )

        assert outcome.analysis["baseline"]["resolved_defaults"]["gpu_layers"] == 0
        assert all(t["config"]["gpu_layers"] == 0 for t in _journal_trials(session.dir))
        assert "-ngl 0" in (session.dir / "recommended.sh").read_text()
        baseline_stage = next(
            entry
            for entry in session.entries
            if entry.get("type") == "stage" and entry.get("stage") == "baseline_complete"
        )
        assert baseline_stage["backends"] == "CPU"

        meta_path = session.dir / "session.json"
        meta = json.loads(meta_path.read_text())
        meta["options"]["budget_trials"] = 40
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        # Prove resume restores backend identity from the baseline stage,
        # independently of the serialized report copy.
        llama_path = session.dir / "llamacpp.json"
        llama_data = json.loads(llama_path.read_text())
        llama_data.pop("backends")
        llama_path.write_text(json.dumps(llama_data, indent=2, sort_keys=True) + "\n")
        trials_before = len(_journal_trials(session.dir))
        resumed = resume_tuning(session.dir)

        assert resumed.exit_code in (0, 1)
        resumed_trials = _journal_trials(session.dir)
        assert len(resumed_trials) > trials_before
        assert all(t["config"]["gpu_layers"] == 0 for t in resumed_trials)
