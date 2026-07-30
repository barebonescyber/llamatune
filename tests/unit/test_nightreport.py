from __future__ import annotations

from llamatune.nightreport import render


def test_render_contains_required_sections_and_verdicts() -> None:
    summary = {
        "window": {"started": "start", "ended": "end", "deadline": "07:00", "outcome": "completed"},
        "counts": {"calibrate": 2, "retune": 1},
        "total_invocations": 6,
        "hardware": {"cpu_model": "CPU", "gpus": [{"name": "GPU"}]},
        "llamacpp": {"build_commit": "abc", "help_sha256": "123"},
        "content_groups": [{"key": "qwen", "representative": "merged.gguf"}],
        "items": [
            {
                "kind": "calibrate",
                "model": "one.gguf",
                "fingerprint": "a" * 64,
                "outcome": "ok",
                "calibration": {
                    "verdict": "consistent",
                    "drift_pp": 0.012,
                    "drift_tg": 0.004,
                    "drift_pp_signed": -0.012,
                    "drift_tg_signed": 0.004,
                    "build_changed": True,
                },
            },
            {
                "kind": "calibrate",
                "model": "copy.gguf",
                "fingerprint": "b" * 64,
                "outcome": "ok",
                "calibration": {
                    "verdict": "consistent",
                    "transfer_from": "a1b2c3d4ffff",
                    "drift_pp": 0.003,
                    "drift_tg": 0.009,
                },
            },
            {
                "kind": "calibrate",
                "model": "bad.gguf",
                "fingerprint": "c" * 64,
                "outcome": "error",
                "calibration": {"verdict": "error", "reason": "oom"},
            },
            {
                "kind": "tune",
                "model": "later.gguf",
                "outcome": "deferred",
                "reason": "deadline",
                "estimated_minutes": 20,
            },
        ],
        "warnings": ["model skipped"],
    }
    text = render(summary)
    assert "## Shift summary" in text
    assert "## Per-model results" in text
    assert "## Deferred and skipped" in text
    assert "## Warnings" in text
    assert "consistent (pp -1.2%, tg +0.4%)" in text
    assert "transfer-consistent vs a1b2c3d4" in text
    assert "failed: oom" in text
    assert "deadline" in text and "20 min" in text
    assert "representative merged.gguf" in text
    assert "build changed" in text


def test_render_tolerates_minimal_summary() -> None:
    text = render({})
    assert "| _None_ |" in text
    assert text.count("_None._") == 2
