from __future__ import annotations

import json
from pathlib import Path

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


def _write_tune_session(
    tmp_path: Path,
    *,
    ctx_size: int | None,
    validation_status: str | None,
) -> str:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "session.json").write_text(
        json.dumps({"options": {"ctx_size": ctx_size}}), encoding="utf-8"
    )
    validation = None
    if validation_status is not None:
        validation = {"ctx": ctx_size, "status": validation_status, "evidence": None}
    (session_dir / "analysis.json").write_text(
        json.dumps({"context_validation": validation}), encoding="utf-8"
    )
    return str(session_dir)


def _tune_item(session_dir: str) -> dict[str, object]:
    return {
        "kind": "tune",
        "model": "model.gguf",
        "fingerprint": "d" * 64,
        "outcome": "succeeded",
        "session_dir": session_dir,
    }


def test_tune_without_required_context_renders_succeeded(tmp_path: Path) -> None:
    session_dir = _write_tune_session(tmp_path, ctx_size=None, validation_status=None)
    text = render({"items": [_tune_item(session_dir)]})
    assert "| model.gguf |" in text
    assert "succeeded" in text
    assert "tuned (context validation skipped/failed)" not in text


def test_tune_with_ok_context_validation_renders_succeeded(tmp_path: Path) -> None:
    session_dir = _write_tune_session(tmp_path, ctx_size=8192, validation_status="ok")
    text = render({"items": [_tune_item(session_dir)]})
    assert "succeeded" in text
    assert "tuned (context validation skipped/failed)" not in text


def test_tune_with_skipped_context_validation_is_not_succeeded(tmp_path: Path) -> None:
    session_dir = _write_tune_session(tmp_path, ctx_size=8192, validation_status="skipped")
    text = render({"items": [_tune_item(session_dir)]})
    assert "tuned (context validation skipped/failed)" in text
    assert "| model.gguf" in text


def test_tune_with_failed_context_validation_is_not_succeeded(tmp_path: Path) -> None:
    session_dir = _write_tune_session(tmp_path, ctx_size=8192, validation_status="failed")
    text = render({"items": [_tune_item(session_dir)]})
    assert "tuned (context validation skipped/failed)" in text


def test_tune_with_missing_evidence_keeps_succeeded(tmp_path: Path) -> None:
    item = _tune_item(str(tmp_path / "missing"))
    text = render({"items": [item]})
    assert "succeeded" in text


def test_calibrate_row_with_failed_validation_keeps_drift_text_and_qualifier(
    tmp_path: Path,
) -> None:
    session_dir = _write_tune_session(tmp_path, ctx_size=8192, validation_status="failed")
    item = {
        "kind": "calibrate",
        "model": "one.gguf",
        "fingerprint": "a" * 64,
        "outcome": "ok",
        "session_dir": session_dir,
        "calibration": {"verdict": "consistent", "drift_pp": 0.012, "drift_tg": 0.004},
    }
    text = render({"items": [item]})
    assert "consistent (pp drift +1.2%, tg drift +0.4%)" in text
    assert "tuned (context validation skipped/failed)" in text


def test_interrupted_retuned_row_keeps_suffix_when_gate_rewords(tmp_path: Path) -> None:
    session_dir = _write_tune_session(tmp_path, ctx_size=8192, validation_status="skipped")
    item = {
        "kind": "tune",
        "model": "model.gguf",
        "fingerprint": "d" * 64,
        "outcome": "interrupted",
        "session_dir": session_dir,
        "retuned": True,
        "winner_improvement": 0.02,
    }
    text = render({"items": [item]})
    assert "tuned (context validation skipped/failed) → retuned" in text
    assert "new winner +2.0%" in text


def test_calibrate_row_marked_retuned_keeps_suffix_and_qualifier(tmp_path: Path) -> None:
    session_dir = _write_tune_session(tmp_path, ctx_size=8192, validation_status="skipped")
    item = {
        "kind": "calibrate",
        "model": "one.gguf",
        "fingerprint": "a" * 64,
        "outcome": "ok",
        "session_dir": session_dir,
        "calibration": {"verdict": "consistent"},
        "retuned": True,
        "winner_improvement": 0.02,
    }
    text = render({"items": [item]})
    assert "consistent → retuned, new winner +2.0%" in text
    assert "tuned (context validation skipped/failed)" in text
