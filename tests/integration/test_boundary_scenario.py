"""Recorded RTX-5080-style feasibility scenario against fake llama-bench."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llamatune.llama import discover_llama
from llamatune.model import inspect_model
from llamatune.search import run_tuning
from llamatune.session import Session
from llamatune.types import GPUInfo, HardwareReport, TuneOptions


def _run_scenario(
    tmp_path: Path, fake_bin_dir: Path, gguf: Path, *, budget_trials: int
) -> tuple[Any, Session]:
    hardware = HardwareReport(
        os_name="Linux",
        arch="x86_64",
        cpu_model="Fake Ryzen",
        physical_cores=8,
        logical_cores=16,
        perf_cores=None,
        ram_mb=65536,
        gpus=(GPUInfo(vendor="nvidia", name="Fake RTX 5080", vram_mb=2850, method="test"),),
        warnings=(),
    )
    llama = discover_llama(fake_bin_dir)
    model = inspect_model(gguf)
    options = TuneOptions(
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
        llama_bin=fake_bin_dir,
        sessions_dir=tmp_path / "sessions",
        full_hash=False,
        ctx_size=8192,
        vram_reserve_mb=0,
    )
    session = Session.create(
        options.sessions_dir,
        model=model,
        hardware=hardware,
        llama=llama,
        options=options,
        argv=["llamatune", "tune", str(gguf)],
    )
    return run_tuning(session, hardware, model, llama, options), session


@pytest.fixture(autouse=True)
def _recorded_memory_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLAMATUNE_FAKE_N_LAYER", "32")
    monkeypatch.setenv("LLAMATUNE_FAKE_VRAM_MB", "2850")
    monkeypatch.setenv("LLAMATUNE_FAKE_MODEL_VRAM_MB", "4000")
    monkeypatch.setenv("LLAMATUNE_FAKE_KV_MB_PER_1K", "10")
    monkeypatch.setenv("LLAMATUNE_FAKE_FAIL_STYLE", "load")


def test_safe_fallback_boundary_and_context_recommendation(
    tmp_path: Path, fake_bin_dir: Path, tiny_gguf: Path
) -> None:
    outcome, session = _run_scenario(tmp_path, fake_bin_dir, tiny_gguf, budget_trials=100)

    assert outcome.exit_code in (0, 1)
    assert outcome.analysis["baseline_kind"] == "safe_fallback"
    assert outcome.analysis["default_probe"]["classification"] == "gpu_resource"
    assert outcome.analysis["baseline"]["resolved_defaults"]["gpu_layers"] == 0

    boundary = outcome.analysis["feasibility"]["boundaries"][0]
    assert boundary == {
        "moe_cpu_layers": 0,
        "max_ok_ngl": 23,
        "min_fail_ngl": 24,
        "probes": boundary["probes"],
        "cap_ngl": 33,
        "cap_source": "model",
        "spill_suspected": False,
    }
    probes = [entry for entry in session.entries if entry.get("type") == "probe"]
    assert any(
        p["purpose"] == "boundary"
        and p["config"]["gpu_layers"] == 24
        and p["status"] == "gpu_resource"
        for p in probes
    )
    assert any(
        p["purpose"] == "context"
        and p["config"]["gpu_layers"] == 23
        and p["status"] == "gpu_resource"
        for p in probes
    )
    assert outcome.analysis["feasibility"]["max_fitting"]["gpu_layers"] == 23
    assert outcome.analysis["feasibility"]["recommended"]["gpu_layers"] == 22
    assert outcome.analysis["context_validation"]["ctx"] == 8192
    assert outcome.analysis["context_validation"]["status"] == "ok"


def test_moe_cpu_residency_expands_the_feasible_gpu_boundary(
    tmp_path: Path, fake_bin_dir: Path, tiny_moe_gguf: Path
) -> None:
    outcome, session = _run_scenario(tmp_path, fake_bin_dir, tiny_moe_gguf, budget_trials=140)

    boundaries = outcome.analysis["feasibility"]["boundaries"]
    by_ncmoe = {boundary["moe_cpu_layers"]: boundary for boundary in boundaries}
    assert 0 in by_ncmoe
    assert max(by_ncmoe) > 0
    assert by_ncmoe[max(by_ncmoe)]["max_ok_ngl"] > by_ncmoe[0]["max_ok_ngl"]
    assert any(
        entry.get("type") in ("probe", "trial")
        and entry.get("config", {}).get("moe_cpu_layers", 0) > 0
        for entry in session.entries
    )
    recommended = outcome.analysis["feasibility"]["recommended"]
    assert recommended is not None
    assert outcome.analysis["context_validation"]["status"] == "ok"
