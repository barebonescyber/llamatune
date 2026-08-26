"""Unit tests for llamatune.session (evidence, layout, resume)."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest

from llamatune.session import Session, SessionCorruptionError, SessionPathError, _confine
from llamatune.types import (
    GPUInfo,
    HardwareReport,
    LlamaCppReport,
    ModelReport,
    TuneOptions,
)


def _hardware() -> HardwareReport:
    return HardwareReport(
        os_name="Linux",
        arch="x86_64",
        cpu_model="Fake CPU",
        physical_cores=8,
        logical_cores=16,
        perf_cores=None,
        ram_mb=32768,
        gpus=(GPUInfo(vendor="nvidia", name="Fake GPU", vram_mb=24000, method="nvidia-smi"),),
        warnings=("test warning",),
    )


def _model(tmp_path: Path) -> ModelReport:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"fake")
    return ModelReport(
        path=model_path,
        size_bytes=4,
        architecture="llama",
        n_layer=32,
        ngl_all=33,
        expert_count=0,
        moe=False,
        name="tiny",
        fingerprint="abc123",
        full_sha256=None,
    )


def _llama() -> LlamaCppReport:
    return LlamaCppReport(
        bench_path=Path("/usr/bin/llama-bench"),
        cli_path=None,
        server_path=None,
        capabilities=frozenset({"fa", "mmp"}),
        help_sha256="deadbeef",
        build_commit=None,
        build_number=None,
    )


def _options(sessions_dir: Path) -> TuneOptions:
    return TuneOptions(
        target="balanced",
        budget_trials=60,
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
        llama_bin=None,
        sessions_dir=sessions_dir,
        full_hash=False,
    )


def _create_session(tmp_path: Path) -> Session:
    sessions_root = tmp_path / "sessions"
    return Session.create(
        sessions_root,
        model=_model(tmp_path),
        hardware=_hardware(),
        llama=_llama(),
        options=_options(sessions_root),
        argv=["llamatune", "tune", "model.gguf"],
    )


def test_create_writes_expected_files_and_layout(tmp_path: Path) -> None:
    session = _create_session(tmp_path)

    assert (session.dir / "session.json").is_file()
    assert (session.dir / "hardware.json").is_file()
    assert (session.dir / "model.json").is_file()
    assert (session.dir / "llamacpp.json").is_file()
    assert (session.dir / "journal.jsonl").is_file()
    assert (session.dir / "baseline").is_dir()
    assert (session.dir / "trials").is_dir()

    session_json = json.loads((session.dir / "session.json").read_text())
    assert session_json["schema_version"] == 2
    assert session_json["argv"] == ["llamatune", "tune", "model.gguf"]
    assert session_json["options"]["target"] == "balanced"
    assert "created" in session_json


def test_create_directory_name_format(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    name = session.dir.name
    assert name.startswith("model-")
    parts = name.split("-")
    assert len(parts) >= 3
    assert len(parts[-1]) == 6  # 6 hex chars


def test_create_appends_session_start_entry(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    assert len(session.entries) == 1
    assert session.entries[0]["type"] == "session_start"
    assert "ts" in session.entries[0]


def test_load_round_trips_reports_and_options(tmp_path: Path) -> None:
    created = _create_session(tmp_path)
    loaded = Session.load(created.dir)

    assert loaded.hardware == created.hardware
    assert loaded.model == created.model
    assert loaded.llama == created.llama
    assert loaded.options == created.options
    assert loaded.entries == created.entries
    assert loaded.resume_warnings == ()


def test_build_hash_and_depth_options_round_trip_additively(tmp_path: Path) -> None:
    sessions_root = tmp_path / "sessions"
    session = Session.create(
        sessions_root,
        model=_model(tmp_path),
        hardware=_hardware(),
        llama=dataclasses.replace(_llama(), bench_sha256="a" * 64),
        options=dataclasses.replace(
            _options(sessions_root),
            depth=32768,
            depth_profile=(0, 8192, 32768),
            thermal_threshold_c=70.0,
            thermal_wait_cap_s=25.0,
            multi_gpu=True,
        ),
        argv=["llamatune"],
    )
    loaded = Session.load(session.dir)
    assert loaded.llama.bench_sha256 == "a" * 64
    assert loaded.options.depth == 32768
    assert loaded.options.depth_profile == (0, 8192, 32768)
    assert loaded.options.thermal_threshold_c == 70.0
    assert loaded.options.thermal_wait_cap_s == 25.0
    assert loaded.options.multi_gpu is True

    llama_path = session.dir / "llamacpp.json"
    llama_data = json.loads(llama_path.read_text())
    llama_data.pop("bench_sha256")
    llama_path.write_text(json.dumps(llama_data))
    session_path = session.dir / "session.json"
    session_data = json.loads(session_path.read_text())
    session_data["options"].pop("depth")
    session_data["options"].pop("depth_profile")
    session_data["options"].pop("thermal_threshold_c")
    session_data["options"].pop("thermal_wait_cap_s")
    session_data["options"].pop("multi_gpu")
    session_path.write_text(json.dumps(session_data))
    legacy = Session.load(session.dir)
    assert legacy.llama.bench_sha256 is None
    assert legacy.options.depth is None
    assert legacy.options.depth_profile is None
    assert legacy.options.ctx_ladder == ()
    assert legacy.options.thermal_threshold_c == 75.0
    assert legacy.options.thermal_wait_cap_s == 60.0
    assert legacy.options.multi_gpu is False


def test_append_adds_timestamp_and_persists(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    session.append({"type": "trial", "trial_id": "abc123", "status": "ok"})

    loaded = Session.load(session.dir)
    trial_entries = [e for e in loaded.entries if e.get("type") == "trial"]
    assert len(trial_entries) == 1
    assert trial_entries[0]["trial_id"] == "abc123"
    assert "ts" in trial_entries[0]


def test_journaled_trial_ids(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    session.append({"type": "trial", "trial_id": "aaa", "status": "ok"})
    session.append({"type": "trial", "trial_id": "bbb", "status": "pruned"})
    session.append({"type": "baseline_run", "run": 1})

    assert session.journaled_trial_ids == frozenset({"aaa", "bbb"})


def test_baseline_runs_completed(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    assert session.baseline_runs_completed == 0
    session.append({"type": "baseline_run", "run": 1})
    session.append({"type": "baseline_run", "run": 2})
    assert session.baseline_runs_completed == 2


def test_trial_dir_and_baseline_dir_create_directories(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    trial_dir = session.trial_dir("abc123")
    baseline_dir = session.baseline_dir(1)

    assert trial_dir == session.dir / "trials" / "abc123"
    assert trial_dir.is_dir()
    assert baseline_dir == session.dir / "baseline" / "run-1"
    assert baseline_dir.is_dir()


def test_write_analysis_and_write_text(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    session.write_analysis({"schema_version": 1, "winner": None})
    session.write_text("report.md", "# Report\n")

    analysis = json.loads((session.dir / "analysis.json").read_text())
    assert analysis["schema_version"] == 1
    assert (session.dir / "report.md").read_text() == "# Report\n"


def test_invalidate_derived_outputs_preserves_session_evidence(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    session.write_analysis({"schema_version": 1, "winner": None})
    for name in ("recommended.json", "recommended.sh", "report.md"):
        session.write_text(name, "stale\n")

    session.invalidate_derived_outputs()

    assert not any(
        (session.dir / name).exists()
        for name in ("analysis.json", "recommended.json", "recommended.sh", "report.md")
    )
    assert (session.dir / "session.json").is_file()
    assert (session.dir / "journal.jsonl").is_file()


def test_record_build_info_updates_llama_and_file(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    session.record_build_info("abc1234", 1234, "CUDA")

    assert session.llama.build_commit == "abc1234"
    assert session.llama.build_number == 1234
    assert session.llama.backends == "CUDA"

    on_disk = json.loads((session.dir / "llamacpp.json").read_text())
    assert on_disk["build_commit"] == "abc1234"
    assert on_disk["build_number"] == 1234
    assert on_disk["backends"] == "CUDA"

    reloaded = Session.load(session.dir)
    assert reloaded.llama.build_commit == "abc1234"
    assert reloaded.llama.build_number == 1234
    assert reloaded.llama.backends == "CUDA"


def test_load_legacy_llama_report_without_backends(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    llama_path = session.dir / "llamacpp.json"
    data = json.loads(llama_path.read_text())
    data.pop("backends")
    llama_path.write_text(json.dumps(data))

    assert Session.load(session.dir).llama.backends is None


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:  # pragma: no cover - platform seam
        pytest.skip(f"symlink creation unavailable on this platform: {exc}")


def test_probe_dir_rejects_traversal_and_creates_nothing(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    with pytest.raises(SessionPathError):
        session.probe_dir("../../outside")
    # The traversal resolves to <sessions>/outside; nothing must be created
    # there, and the probes/ directory itself must not be materialized.
    assert not (tmp_path / "sessions" / "outside").exists()
    assert not (session.dir / "probes").exists()


def test_probe_dir_rejects_out_of_tree_symlink(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    _symlink_or_skip(session.dir / "probes", outside)
    with pytest.raises(SessionPathError):
        session.probe_dir("p1")
    assert not (outside / "p1").exists()


def test_load_rejects_out_of_tree_journal_symlink_and_leaves_target_intact(
    tmp_path: Path,
) -> None:
    session = _create_session(tmp_path)
    # An external victim whose final line is torn: without confinement the
    # torn-journal repair would truncate it. The trailing "TORN" fragment is
    # not newline-terminated, exactly the shape the repair path would rewrite.
    victim = tmp_path / "victim.jsonl"
    victim.write_bytes(b'{"type":"ok"}\nTORN')
    original = victim.read_bytes()
    journal = session.dir / "journal.jsonl"
    journal.unlink()
    _symlink_or_skip(journal, victim)

    with pytest.raises(SessionPathError):
        Session.load(session.dir)

    assert victim.read_bytes() == original


def test_read_json_rejects_traversal(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    victim = tmp_path / "victim.json"
    victim.write_text('{"secret": true}\n')

    with pytest.raises(SessionPathError):
        session.read_json("../../victim.json")


def test_read_json_rejects_out_of_tree_symlink(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    victim = tmp_path / "victim.json"
    victim.write_text('{"secret": true}\n')
    _symlink_or_skip(session.dir / "analysis.json", victim)

    with pytest.raises(SessionPathError):
        session.read_json("analysis.json")


def test_path_confinement_rejects_traversal(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    with pytest.raises(SessionPathError):
        session.trial_dir("../../evil")


def test_path_confinement_rejects_absolute_escape(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    with pytest.raises(SessionPathError):
        session.write_text("../outside.txt", "nope")


@pytest.mark.skipif(os.name != "nt", reason="Windows path semantics")
def test_windows_drive_confinement_rejects_backslash_traversal() -> None:
    with pytest.raises(SessionPathError):
        _confine(Path(r"C:\sessions\run"), r"..\..\escape")


@pytest.mark.skipif(os.name != "nt", reason="Windows path semantics")
def test_windows_drive_confinement_rejects_other_drive() -> None:
    with pytest.raises(SessionPathError):
        _confine(Path(r"C:\sessions\run"), r"D:\escape")


def test_resume_tolerates_torn_final_journal_line(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    session.append({"type": "trial", "trial_id": "aaa", "status": "ok"})

    journal_path = session.dir / "journal.jsonl"
    with journal_path.open("a", encoding="utf-8") as fh:
        fh.write('{"type": "trial", "trial_id": "torn"')  # no closing brace/newline

    loaded = Session.load(session.dir)
    assert loaded.journaled_trial_ids == frozenset({"aaa"})
    assert len(loaded.resume_warnings) == 1
    assert "torn" in loaded.resume_warnings[0]


def test_resume_raises_on_non_final_corrupt_line(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    journal_path = session.dir / "journal.jsonl"
    with journal_path.open("a", encoding="utf-8") as fh:
        fh.write("not valid json at all\n")
        fh.write(json.dumps({"type": "trial", "trial_id": "zzz", "ts": "now"}) + "\n")

    with pytest.raises(SessionCorruptionError):
        Session.load(session.dir)


def test_resume_raises_on_valid_json_non_object_line(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    with (session.dir / "journal.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("123\n")

    with pytest.raises(SessionCorruptionError, match=r"line 2"):
        Session.load(session.dir)


def test_load_missing_journal_file_is_empty(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    (session.dir / "journal.jsonl").unlink()

    loaded = Session.load(session.dir)
    assert loaded.entries == ()


def test_v2_round_trip_and_legacy_defaults(tmp_path: Path) -> None:
    sessions_root = tmp_path / "sessions"
    options = dataclasses.replace(
        _options(sessions_root),
        ctx_size=8192,
        vram_reserve_mb=1536,
        initial_gpu_layers=2,
        max_gpu_layers=24,
        initial_cpu_moe=32,
        allow_core_dumps=True,
        quiet_wait_s=30.0,
        quiet_load=4.0,
        observe_vram=False,
        validate_with_cli=True,
        quality_corpus=tmp_path / "quality.txt",
        batched_trials=False,
        ot_search=True,
    )
    session = Session.create(
        sessions_root,
        model=_model(tmp_path),
        hardware=_hardware(),
        llama=_llama(),
        options=options,
        argv=["llamatune", "tune", "model.gguf"],
    )
    assert Session.load(session.dir).options == options

    meta_path = session.dir / "session.json"
    meta = json.loads(meta_path.read_text())
    meta.pop("schema_version")
    for key in (
        "ctx_size",
        "vram_reserve_mb",
        "initial_gpu_layers",
        "max_gpu_layers",
        "initial_cpu_moe",
        "allow_core_dumps",
        "quiet_wait_s",
        "quiet_load",
        "observe_vram",
        "validate_with_cli",
        "quality_corpus",
        "batched_trials",
        "ot_search",
    ):
        meta["options"].pop(key)
    meta_path.write_text(json.dumps(meta))
    legacy = Session.load(session.dir)
    assert legacy.options.ctx_size is None
    assert legacy.options.vram_reserve_mb is None
    assert legacy.options.allow_core_dumps is False
    assert legacy.options.quiet_wait_s == 0.0
    assert legacy.options.quiet_load is None
    assert legacy.options.observe_vram is True
    assert legacy.options.validate_with_cli is False
    assert legacy.options.quality_corpus is None
    assert legacy.options.batched_trials is True
    assert legacy.options.ot_search is False


def test_list_sessions_tolerates_complete_in_progress_and_corrupt(tmp_path: Path) -> None:
    from llamatune.session import list_sessions

    complete = _create_session(tmp_path)
    complete.write_analysis({"winner": {"trial_id": "winner", "confirmed": True}})
    complete.append({"type": "session_end", "exit_code": 0})
    in_progress = tmp_path / "sessions" / "in-progress"
    in_progress.mkdir()
    (in_progress / "session.json").write_text(json.dumps({"created": "now"}))
    (in_progress / "model.json").write_text(json.dumps({"name": "working"}))
    corrupt = tmp_path / "sessions" / "corrupt"
    corrupt.mkdir()
    (corrupt / "session.json").write_text("not-json")
    missing_metadata = tmp_path / "sessions" / "missing-metadata"
    missing_metadata.mkdir()
    matrix = tmp_path / "sessions" / "matrix"
    matrix.mkdir()
    (matrix / "results-matrix.json").write_text("{}")
    nightshift = tmp_path / "sessions" / "nightshift"
    nightshift_run = nightshift / "run-id"
    nightshift_run.mkdir(parents=True)
    (nightshift_run / "run.json").write_text("{}")
    (nightshift_run / "nightshift.json").write_text("{}")

    rows = list_sessions(tmp_path / "sessions")
    by_name = {Path(row["session_dir"]).name: row for row in rows}
    assert by_name[complete.dir.name]["winner_trial_id"] == "winner"
    assert by_name[complete.dir.name]["exit_code"] == 0
    assert by_name["in-progress"]["status"] == "in_progress"
    assert by_name["corrupt"]["status"] == "corrupt"
    assert by_name["missing-metadata"]["status"] == "corrupt"
    assert "matrix" not in by_name
    assert "nightshift" not in by_name


def test_future_session_schema_is_rejected(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    meta_path = session.dir / "session.json"
    meta = json.loads(meta_path.read_text())
    meta["schema_version"] = 3
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(SessionCorruptionError, match="version 3"):
        Session.load(session.dir)


def test_list_sessions_uses_last_session_end_and_falls_back_on_corrupt_tail(
    tmp_path: Path,
) -> None:
    from llamatune.session import list_sessions

    sessions_root = tmp_path / "sessions"
    superseded = _create_session(tmp_path)
    final = _create_session(tmp_path)
    superseded.write_analysis({"winner": {"trial_id": "w", "confirmed": False}})
    final.write_analysis({"winner": {"trial_id": "w", "confirmed": True}})

    journal = final.dir / "journal.jsonl"
    with journal.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "session_end", "exit_code": 3}) + "\n")
        handle.write(json.dumps({"type": "session_end", "exit_code": 0}) + "\n")
    with superseded.dir.joinpath("journal.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "session_end", "exit_code": 2}) + "\n")
        handle.write('{"torn tail')

    rows = list_sessions(sessions_root)
    by_dir = {Path(row["session_dir"]).name: row for row in rows}
    assert by_dir[final.dir.name]["exit_code"] == 0
    assert by_dir[final.dir.name]["status"] == "complete"
    assert by_dir[superseded.dir.name]["status"] == "corrupt"
    assert by_dir[superseded.dir.name]["exit_code"] is None


def test_list_sessions_tail_decision_aligns_with_shared_reader_tolerance(
    tmp_path: Path,
) -> None:
    """A decisive clean tail wins even if an earlier line is malformed.

    Mirrors the shared reader policy (skip corrupt lines, keep valid ones):
    the historical strict loop rejected the whole journal for any bad line;
    tail scanning decides from the newest complete entries instead.
    """
    from llamatune.session import list_sessions

    session = _create_session(tmp_path)
    session.write_analysis({"winner": {"trial_id": "w", "confirmed": False}})
    journal = session.dir / "journal.jsonl"
    original = journal.read_bytes()
    journal.write_bytes(b"\n" + original)
    with journal.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "session_end", "exit_code": 0}) + "\n")

    rows = list_sessions(session.dir.parent)

    assert len(rows) == 1
    assert rows[0]["exit_code"] == 0
    assert rows[0]["status"] == "complete"


def test_list_sessions_corrupt_tail_still_falls_back_and_marks_corrupt(
    tmp_path: Path,
) -> None:
    """Corruption at the tail forces the exact historical full-parse path."""
    from llamatune.session import list_sessions

    session = _create_session(tmp_path)
    session.write_analysis({"winner": {"trial_id": "w", "confirmed": False}})
    journal = session.dir / "journal.jsonl"
    with journal.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "session_end", "exit_code": 0}) + "\n")
        handle.write('{"torn')

    rows = list_sessions(session.dir.parent)

    assert rows[0]["status"] == "corrupt"
    assert rows[0]["exit_code"] is None


def test_list_sessions_large_journal_reads_only_the_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llamatune import session as session_module
    from llamatune.session import list_sessions

    session = _create_session(tmp_path)
    session.write_analysis({"winner": {"trial_id": "w", "confirmed": True}})
    journal = session.dir / "journal.jsonl"
    filler = json.dumps({"type": "trial", "trial_id": "filler", "note": "x" * 256})
    with journal.open("a", encoding="utf-8") as handle:
        for _ in range(2000):
            handle.write(filler + "\n")
        handle.write(json.dumps({"type": "session_end", "exit_code": 1}) + "\n")
    monkeypatch.setattr(session_module, "_TAIL_CHUNK_BYTES", 512)

    rows = list_sessions(session.dir.parent)

    assert rows[0]["exit_code"] == 1
    assert rows[0]["status"] == "complete"
