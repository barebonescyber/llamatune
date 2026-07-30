"""Behavioral feasibility remains authoritative when GPU VRAM is unknown."""

from __future__ import annotations

from pathlib import Path

import pytest

from llamatune.llama import discover_llama
from llamatune.model import inspect_model
from llamatune.search import run_tuning
from llamatune.session import Session
from llamatune.types import GPUInfo, HardwareReport, TuneOptions


def test_unknown_vram_gpu_completes_behavioral_search(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLAMATUNE_FAKE_BACKEND", "Vulkan")
    monkeypatch.setenv("LLAMATUNE_FAKE_VRAM_MB", "2850")
    monkeypatch.setenv("LLAMATUNE_FAKE_MODEL_VRAM_MB", "4000")
    hardware = HardwareReport(
        os_name="Windows",
        arch="AMD64",
        cpu_model="Fake CPU",
        physical_cores=8,
        logical_cores=16,
        perf_cores=None,
        ram_mb=65536,
        gpus=(
            GPUInfo(vendor="intel", name="Unknown VRAM GPU", vram_mb=None, method="wmi:name-only"),
        ),
        warnings=("load monitoring is unavailable on Windows",),
    )
    llama = discover_llama(fake_bin_dir)
    model = inspect_model(tiny_gguf)
    options = TuneOptions(
        target="balanced",
        budget_trials=60,
        budget_minutes=None,
        reps_search=1,
        reps_confirm=2,
        baseline_runs=2,
        pp=512,
        tg=128,
        allow_lossy=False,
        cooldown_s=0,
        thermal_wait_cap_s=0.0,
        baseline_only=False,
        llama_bin=fake_bin_dir,
        sessions_dir=tmp_path / "sessions",
        full_hash=False,
        ctx_size=2048,
        vram_reserve_mb=1536,
        observe_vram=False,
    )
    session = Session.create(
        options.sessions_dir,
        model=model,
        hardware=hardware,
        llama=llama,
        options=options,
        argv=["llamatune", "tune", str(tiny_gguf)],
    )
    outcome = run_tuning(session, hardware, model, llama, options)

    assert outcome.exit_code in (0, 1)
    estimate = outcome.analysis["feasibility"]["estimate"]
    assert estimate["budget_mb"] is None
    assert all(
        entry.get("estimate_reason") is None
        for entry in session.entries
        if entry.get("type") == "probe"
    )
    assert outcome.analysis["context_validation"]["status"] == "ok"
    assert outcome.analysis["feasibility"]["recommended"] is not None
