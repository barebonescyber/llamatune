"""Deterministic synthetic evidence trees for Results Matrix tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config(*, gpu_layers: int = 33, cache_type_k: str = "f16") -> dict[str, Any]:
    return {
        "gpu_layers": gpu_layers,
        "moe_cpu_layers": 0,
        "flash_attn": True,
        "ubatch": 512,
        "batch": 2048,
        "threads": 8,
        "mmap": True,
        "no_kv_offload": False,
        "cache_type_k": cache_type_k,
        "cache_type_v": "f16",
    }


def _hardware(name: str = "Fake GPU") -> dict[str, Any]:
    return {
        "os_name": "Linux",
        "arch": "x86_64",
        "cpu_model": "Fake CPU",
        "physical_cores": 8,
        "logical_cores": 16,
        "perf_cores": None,
        "ram_mb": 32768,
        "gpus": [{"vendor": "nvidia", "name": name, "vram_mb": 24000, "method": "test"}],
        "warnings": [],
    }


def _llama(build: str = "bench-a") -> dict[str, Any]:
    return {
        "bench_path": "/bin/llama-bench",
        "cli_path": None,
        "server_path": None,
        "capabilities": [],
        "help_sha256": f"help-{build}",
        "bench_sha256": build,
        "build_commit": "commit-a",
        "build_number": 1,
    }


def write_matrix_evidence(
    root: Path,
    *,
    fingerprint: str = "fingerprint-a",
    model_name: str = "Fixture Q4_K_M",
    created: str = "2026-01-02T03:04:05+00:00",
    pp: float = 720.0,
    tg: float = 48.0,
    hardware_name: str = "Fake GPU",
    build: str = "bench-a",
    safe_fallback: bool = False,
) -> dict[str, Path]:
    """Write one conforming source of every matrix evidence kind."""
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / f"{model_name.replace(' ', '-')}.gguf"
    model = {
        "path": str(model_path),
        "size_bytes": 4_000_000_000,
        "architecture": "llama",
        "n_layer": 32,
        "ngl_all": 33,
        "expert_count": 0,
        "moe": False,
        "name": model_name,
        "fingerprint": fingerprint,
        "full_sha256": None,
    }
    hardware = _hardware(hardware_name)
    llama = _llama(build)
    session = root / "session-a"
    options = {"pp": 512, "tg": 128, "depth": 4096, "ctx_size": 8192, "reps_search": 3}
    _write(session / "session.json", {"schema_version": 2, "created": created, "options": options})
    _write(session / "model.json", model)
    _write(session / "hardware.json", hardware)
    _write(session / "llamacpp.json", llama)
    winner = {
        "trial_id": "winner",
        "config": _config(cache_type_k="q8_0"),
        "pp": pp,
        "tg": tg,
        "confirmed": True,
        "improvement_pct": {"pp": 20.0, "tg": 10.0, "score": 15.0},
        "confirmation": {
            "runs": 5,
            "pp": {"mean": pp, "stdev": 1.0, "cv": 0.01, "n": 5},
            "tg": {"mean": tg, "stdev": 0.5, "cv": 0.01, "n": 5},
        },
    }
    analysis = {
        "schema_version": 2,
        "baseline_kind": "safe_fallback" if safe_fallback else "defaults",
        "baseline": {
            "runs": 3,
            "pp": {"mean": 600.0, "stdev": 2.0, "cv": 0.01, "n": 3},
            "tg": {"mean": 40.0, "stdev": 1.0, "cv": 0.01, "n": 3},
            "noise_floor_cv": 0.01,
            "fallback": "-ngl 0" if safe_fallback else None,
            "resolved_defaults": _config(gpu_layers=0 if safe_fallback else 33),
            "kind": "safe_fallback" if safe_fallback else "defaults",
        },
        "winner": winner,
        "lossless_winner": {
            "trial_id": "lossless",
            "config": _config(),
            "pp_mean": pp - 10,
            "tg_mean": tg - 1,
            "score": 1.1,
            "status": "ok",
        },
        "feasibility": {"ctx_validated": 8192},
        "context_envelope": [
            {"ctx": 8192, "status": "ok", "config": _config(), "fallback_config": None},
            {
                "ctx": 16384,
                "status": "failed",
                "config": _config(),
                "fallback_config": _config(gpu_layers=24),
            },
        ],
        "depth_profile": {
            "trial_id": "winner",
            "rows": [{"d": 2048, "pp": pp + 5, "tg": tg + 1}],
        },
        "quality_gate": {"status": "ok", "ppl_lossy": 9.5, "ppl_f16": 9.4, "delta_pct": 1.06},
    }
    _write(session / "analysis.json", analysis)
    (session / "journal.jsonl").write_text(
        json.dumps({"type": "analysis_written", "ts": created})
        + "\n"
        + json.dumps({"type": "confirmation_run", "ts": "2026-01-02T03:05:00+00:00"})
        + "\n",
        encoding="utf-8",
    )

    marathon = root / "marathon" / "run-a"
    _write(marathon / "run.json", {"schema_version": 1, "created": "2026-01-03T00:00:00+00:00"})
    _write(marathon / "model.json", model)
    _write(marathon / "hardware.json", hardware)
    _write(marathon / "llamacpp.json", llama)
    _write(
        marathon / "marathon.json",
        {
            "schema_version": 1,
            "options": {"pp": 512, "tg": 128, "ctx_size": 8192, "reps_search": 3},
            "rounds": [{"winner_config": _config()}],
            "matrix": [
                {
                    "ctx": 8192,
                    "depth": 4096,
                    "status": "ok",
                    "config": _config(),
                    "pp": pp,
                    "tg": tg,
                    "refined": False,
                    "evidence": "matrix/ctx-8192-d-4096",
                },
                {
                    "ctx": 32768,
                    "depth": 4096,
                    "status": "deferred",
                    "config": None,
                    "pp": None,
                    "tg": None,
                    "refined": False,
                    "evidence": None,
                },
            ],
            "ab": {
                "label": "final",
                "blocks": 5,
                "a_wins": 1,
                "b_wins": 4,
                "ties": 0,
                "a_pp": 600.0,
                "a_tg": 40.0,
                "b_pp": pp,
                "b_tg": tg,
                "margin": 0.03,
                "verdict": "b",
            },
        },
    )

    nightshift = root / "nightshift" / "run-a"
    _write(nightshift / "run.json", {"schema_version": 1, "created": "2026-01-04T00:00:00+00:00"})
    _write(nightshift / "hardware.json", hardware)
    _write(nightshift / "llamacpp.json", llama)
    _write(
        nightshift / "nightshift.json",
        {
            "schema_version": 1,
            "hardware": hardware,
            "llamacpp": llama,
            "items": [
                {
                    "kind": "calibrate",
                    "model_path": str(model_path),
                    "fingerprint": fingerprint,
                    "depth_workload": 4096,
                    "calibration": {
                        "fingerprint": fingerprint,
                        "reference_session": str(session),
                        "verdict": "consistent",
                        "pp": {"mean": pp - 1, "stdev": 1.0, "cv": 0.01, "n": 3},
                        "tg": {"mean": tg - 0.5, "stdev": 0.5, "cv": 0.01, "n": 3},
                        "drift_pp": 0.01,
                        "drift_tg": 0.02,
                        "runs": 3,
                    },
                }
            ],
        },
    )

    quality = root / "quality" / "run-a"
    _write(
        quality / "quality.json",
        {
            "schema_version": 1,
            "run_dir": str(quality),
            "created": "2026-01-05T00:00:00+00:00",
            "model": {
                "fingerprint": fingerprint,
                "name": model_name,
                "path": str(model_path),
                "size_bytes": model["size_bytes"],
            },
            "hardware_signature": [
                "Linux",
                "x86_64",
                8,
                16,
                32768,
                [["nvidia", hardware_name, 24000]],
            ],
            "build": {
                "bench_sha256": build,
                "help_sha256": f"help-{build}",
                "build_commit": "commit-a",
            },
            "ctx": 8192,
            "seed": 42,
            "reps": 1,
            "config": _config(),
            "config_source": "session",
            "config_provenance": str(session),
            "exec_enabled": False,
            "suites": [
                {
                    "suite_id": "coding@1+abcdef123456",
                    "name": "coding",
                    "kind": "coding",
                    "metrics": {"score": 0.9, "pass_rate": 0.8},
                    "tasks": [],
                }
            ],
            "overall": 0.9,
            "comparison": None,
            "warnings": [],
        },
    )
    return {
        "root": root,
        "session": session,
        "marathon": marathon,
        "nightshift": nightshift,
        "quality": quality,
    }
