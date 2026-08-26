"""Derived Night Shift registry tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llamatune.registry import build_registry, incomplete_sessions
from llamatune.types import RegistryRecord


def _write_session(
    root: Path,
    name: str,
    *,
    fingerprint: str = "fp",
    created: str = "2026-01-01T00:00:00+00:00",
    winner: bool = True,
    complete: bool = True,
    ctx_size: int | None = None,
) -> Path:
    session = root / name
    session.mkdir(parents=True)
    (session / "session.json").write_text(
        json.dumps(
            {
                "created": created,
                "options": {
                    "pp": 512,
                    "tg": 128,
                    "reps_confirm": 5,
                    "target": "balanced",
                    "ctx_size": ctx_size,
                },
            }
        )
    )
    (session / "model.json").write_text(json.dumps({"fingerprint": fingerprint}))
    (session / "llamacpp.json").write_text(
        json.dumps({"build_commit": "abc", "help_sha256": "help"})
    )
    baseline = {
        "noise_floor_cv": 0.02,
        "pp": {"mean": 100.0},
        "tg": {"mean": 10.0},
    }
    winner_data = (
        {
            "config": {
                "gpu_layers": 0,
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
            "confirmation": {"pp": {"mean": 120.0}, "tg": {"mean": 11.0}},
        }
        if winner
        else None
    )
    if complete:
        (session / "analysis.json").write_text(
            json.dumps(
                {
                    "baseline": baseline,
                    "winner": winner_data,
                    "feasibility": {"ctx_validated": ctx_size},
                }
            )
        )
    entries = [
        {"type": "session_start", "ts": created},
        {"type": "trial", "wall_s": 4.0, "ts": "2026-01-01T00:00:04+00:00"},
    ]
    if complete:
        entries.append({"type": "session_end", "ts": "2026-01-01T00:00:10+00:00"})
    (session / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    return session


def test_registry_latest_winner_and_defaults_records(tmp_path: Path) -> None:
    _write_session(tmp_path, "old", created="2026-01-01T00:00:00+00:00")
    latest = _write_session(tmp_path, "latest", created="2026-01-02T00:00:00+00:00", winner=False)
    records = build_registry(tmp_path)
    record = records["fp"]
    assert record.session_dir == latest
    assert record.outcome == "defaults_optimal"
    assert record.reference_config is None
    assert (record.reference_pp, record.reference_tg) == (100.0, 10.0)
    assert record.median_trial_wall_s == 4.0


def test_winner_uses_confirmation_means(tmp_path: Path) -> None:
    _write_session(tmp_path, "winner")
    record = build_registry(tmp_path)["fp"]
    assert record.outcome == "winner"
    assert record.reference_config is not None
    assert (record.reference_pp, record.reference_tg) == (120.0, 11.0)


def test_registry_context_filter_uses_newest_compatible_session(tmp_path: Path) -> None:
    compatible = _write_session(
        tmp_path,
        "compatible",
        created="2026-01-01T00:00:00+00:00",
        ctx_size=32768,
    )
    _write_session(
        tmp_path,
        "newer-too-small",
        created="2026-01-02T00:00:00+00:00",
        ctx_size=8192,
    )

    record = build_registry(tmp_path, ctx_size=16384)["fp"]
    assert record.session_dir == compatible
    assert record.ctx_size == 32768


def test_registry_context_filter_rejects_legacy_session(tmp_path: Path) -> None:
    _write_session(tmp_path, "legacy")
    assert build_registry(tmp_path, ctx_size=8192) == {}


def test_registry_rejects_requested_but_unvalidated_context(tmp_path: Path) -> None:
    session = _write_session(tmp_path, "failed-context", ctx_size=8192)
    analysis = json.loads((session / "analysis.json").read_text())
    analysis["feasibility"]["ctx_validated"] = None
    analysis["context_validation"] = {"ctx": 8192, "status": "failed"}
    (session / "analysis.json").write_text(json.dumps(analysis))
    assert build_registry(tmp_path, ctx_size=8192) == {}


def test_registry_uses_largest_exact_context_envelope_rung(tmp_path: Path) -> None:
    session = _write_session(tmp_path, "envelope", ctx_size=8192)
    analysis = json.loads((session / "analysis.json").read_text())
    analysis["context_envelope"] = [
        {"ctx": 8192, "status": "ok", "fallback_config": None},
        {"ctx": 16384, "status": "ok", "fallback_config": None},
        {"ctx": 32768, "status": "failed", "fallback_config": {"gpu_layers": 0}},
    ]
    (session / "analysis.json").write_text(json.dumps(analysis))
    assert build_registry(tmp_path)["fp"].ctx_size == 16384


def test_incomplete_oldest_first_and_nightshift_ignored(tmp_path: Path) -> None:
    newer = _write_session(tmp_path, "newer", created="2026-01-02T00:00:00+00:00", complete=False)
    older = _write_session(tmp_path, "older", created="2026-01-01T00:00:00+00:00", complete=False)
    _write_session(tmp_path / "nightshift", "not-a-session", complete=False)
    assert incomplete_sessions(tmp_path) == (older, newer)
    assert build_registry(tmp_path) == {}


def test_corrupt_completed_session_warns_and_skips(tmp_path: Path) -> None:
    session = _write_session(tmp_path, "corrupt")
    (session / "analysis.json").write_text("{")
    with pytest.warns(RuntimeWarning, match="corrupt session"):
        assert build_registry(tmp_path) == {}


def test_torn_journal_tail_is_tolerated(tmp_path: Path) -> None:
    session = _write_session(tmp_path, "session")
    with (session / "journal.jsonl").open("a") as handle:
        handle.write('{"type":')
    assert build_registry(tmp_path)["fp"].session_dir == session


def test_incremental_absorb_converges_to_from_scratch_registry(tmp_path: Path) -> None:
    from llamatune.registry import absorb_session

    records: dict[str, RegistryRecord] = {}
    assert build_registry(tmp_path) == {}

    first = _write_session(tmp_path, "first", created="2026-01-01T00:00:00+00:00")
    assert absorb_session(records, first) is True
    assert records == build_registry(tmp_path)

    # A newer completed session replaces the fingerprint's record.
    second = _write_session(tmp_path, "second", created="2026-01-02T00:00:00+00:00")
    assert absorb_session(records, second) is True
    assert records["fp"].session_dir == second
    assert records == build_registry(tmp_path)

    # An older session never displaces the newer record.
    older = _write_session(tmp_path, "older", created="2026-01-01T06:00:00+00:00")
    assert absorb_session(records, older) is False
    assert records == build_registry(tmp_path)

    # An incomplete session is ignored, exactly as a full rescan would.
    incomplete = _write_session(
        tmp_path, "incomplete", created="2026-01-03T00:00:00+00:00", complete=False
    )
    assert absorb_session(records, incomplete) is False
    assert records == build_registry(tmp_path)


def test_incremental_absorb_applies_context_filter(tmp_path: Path) -> None:
    from llamatune.registry import absorb_session

    records: dict[str, RegistryRecord] = {}
    small = _write_session(tmp_path, "small", created="2026-01-01T00:00:00+00:00", ctx_size=8192)
    large = _write_session(tmp_path, "large", created="2026-01-02T00:00:00+00:00", ctx_size=32768)

    assert absorb_session(records, small, ctx_size=16384) is False
    assert records == {}
    assert absorb_session(records, large, ctx_size=16384) is True
    assert records == build_registry(tmp_path, ctx_size=16384)
    unfiltered: dict[str, RegistryRecord] = {}
    absorb_session(unfiltered, small)
    absorb_session(unfiltered, large)
    assert unfiltered == build_registry(tmp_path)


def test_incremental_absorb_warns_and_skips_corrupt_session(tmp_path: Path) -> None:
    from llamatune.registry import absorb_session

    records: dict[str, RegistryRecord] = {}
    good = _write_session(tmp_path, "good")
    corrupt = _write_session(tmp_path, "corrupt", created="2026-01-02T00:00:00+00:00")
    (corrupt / "analysis.json").write_text("{")
    with pytest.warns(RuntimeWarning, match="corrupt session"):
        assert absorb_session(records, corrupt) is False
    assert absorb_session(records, good) is True
    torn = _write_session(tmp_path, "torn", created="2026-01-03T00:00:00+00:00")
    with torn.joinpath("journal.jsonl").open("a") as handle:
        handle.write('{"type":')
    assert absorb_session(records, torn) is True
    with pytest.warns(RuntimeWarning, match="corrupt session"):
        assert records == build_registry(tmp_path)
