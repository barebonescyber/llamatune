"""Unit tests for the append-only recommendation registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llamatune import registry
from llamatune.types import LlamaCppReport, ModelReport, TrialConfig


def _model() -> ModelReport:
    return ModelReport(
        path=Path("model.gguf"),
        size_bytes=1,
        architecture="test",
        n_layer=1,
        ngl_all=2,
        expert_count=0,
        moe=False,
        name="model",
        fingerprint="fingerprint",
        full_sha256=None,
    )


def _llama(help_sha256: str = "build-a", bench_sha256: str | None = "binary-a") -> LlamaCppReport:
    return LlamaCppReport(
        bench_path=Path("llama-bench"),
        cli_path=None,
        server_path=None,
        capabilities=frozenset(),
        help_sha256=help_sha256,
        build_commit="abc",
        build_number=1,
        bench_sha256=bench_sha256,
    )


def _config() -> TrialConfig:
    return TrialConfig(
        gpu_layers=0,
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


def test_append_load_and_lookup_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    signature = ("Linux", "x86_64", 8)
    appended = registry.append_record(
        path,
        model_report=_model(),
        llama_report=_llama(),
        hardware_signature=signature,
        session_dir=tmp_path / "session",
        target="balanced",
        ctx_size=8192,
        config=_config(),
        expected={"score": 2.0},
    )
    assert registry.load_all(path) == [appended]
    result = registry.lookup(path, _model(), _llama(), signature)
    assert result["status"] == "hit"
    assert result["record"] == appended


def test_lookup_context_is_a_minimum_compatibility_constraint(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    signature = ("Linux", "x86_64", 8)
    appended = registry.append_record(
        path,
        model_report=_model(),
        llama_report=_llama(),
        hardware_signature=signature,
        session_dir=tmp_path / "session",
        target="balanced",
        ctx_size=32768,
        config=_config(),
        expected={},
    )

    assert registry.lookup(path, _model(), _llama(), signature, ctx_size=8192)["record"] == appended
    assert registry.lookup(path, _model(), _llama(), signature, ctx_size=32768)["status"] == "hit"
    stale = registry.lookup(path, _model(), _llama(), signature, ctx_size=65536)
    assert stale["status"] == "stale"
    assert stale["stale_reasons"] == ["context size increased"]
    assert registry.lookup(path, _model(), _llama(), signature)["status"] == "hit"


def test_lookup_legacy_record_without_context_is_stale_when_constrained(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    path.write_text(
        json.dumps(
            {
                "model_fingerprint": _model().fingerprint,
                "help_sha256": "build-a",
                "hardware_signature": ["Linux"],
            }
        )
        + "\n"
    )

    result = registry.lookup(path, _model(), _llama(bench_sha256=None), ("Linux",), ctx_size=8192)
    assert result["status"] == "stale"
    assert result["stale_reasons"] == ["context size increased"]


def test_lookup_matches_binary_hash_and_exact_depth_workload(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    signature = ("Linux", "x86_64", 8)
    appended = registry.append_record(
        path,
        model_report=_model(),
        llama_report=_llama(),
        hardware_signature=signature,
        session_dir=tmp_path / "deep-session",
        target="balanced",
        ctx_size=65536,
        pp=4096,
        tg=128,
        depth=32768,
        config=_config(),
        expected={},
    )
    hit = registry.lookup(path, _model(), _llama(), signature, pp=4096, tg=128, depth=32768)
    assert hit == {"status": "hit", "record": appended, "stale_reasons": []}

    wrong_depth = registry.lookup(path, _model(), _llama(), signature, pp=4096, tg=128, depth=0)
    assert wrong_depth["status"] == "stale"
    assert wrong_depth["stale_reasons"] == ["workload changed"]

    wrong_binary = registry.lookup(
        path,
        _model(),
        _llama(bench_sha256="binary-b"),
        signature,
        pp=4096,
        tg=128,
        depth=32768,
    )
    assert wrong_binary["status"] == "stale"
    assert wrong_binary["stale_reasons"] == ["llama.cpp build changed"]


def test_old_registry_record_cannot_false_hit_depth_workload(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    path.write_text(
        json.dumps(
            {
                "model_fingerprint": _model().fingerprint,
                "help_sha256": "build-a",
                "hardware_signature": ["Linux"],
            }
        )
        + "\n"
    )
    result = registry.lookup(
        path, _model(), _llama(bench_sha256=None), ("Linux",), pp=512, tg=128, depth=8192
    )
    assert result["status"] == "stale"
    assert result["stale_reasons"] == ["workload changed"]


def test_legacy_hash_fallback_and_malformed_workload_are_explicit(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    legacy = {
        "model_fingerprint": _model().fingerprint,
        "help_sha256": "build-a",
        "hardware_signature": ["Linux"],
        "workload": "not-an-object",
    }
    path.write_text(json.dumps(legacy) + "\n")

    # Without a requested workload, old records still use help identity.
    fallback_hit = registry.lookup(path, _model(), _llama(bench_sha256=None), ("Linux",))
    assert fallback_hit["status"] == "hit"

    malformed = registry.lookup(
        path,
        _model(),
        _llama(bench_sha256=None),
        ("Linux",),
        pp=512,
        tg=128,
        depth=None,
    )
    assert malformed["status"] == "stale"
    assert malformed["stale_reasons"] == ["workload changed"]


def test_registry_falls_back_to_help_when_only_one_binary_hash_is_known(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    registry.append_record(
        path,
        model_report=_model(),
        llama_report=_llama(bench_sha256=None),
        hardware_signature=("hardware",),
        session_dir=tmp_path / "session",
        target="balanced",
        ctx_size=None,
        config=_config(),
        expected={},
    )
    same_help = registry.lookup(path, _model(), _llama(bench_sha256="new-hash"), ("hardware",))
    changed_help = registry.lookup(
        path,
        _model(),
        _llama("changed-help", bench_sha256="new-hash"),
        ("hardware",),
    )
    assert same_help["status"] == "hit"
    assert changed_help["stale_reasons"] == ["llama.cpp build changed"]


def test_lookup_reports_stale_identities(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    registry.append_record(
        path,
        model_report=_model(),
        llama_report=_llama(),
        hardware_signature=("old-hardware",),
        session_dir=tmp_path / "session",
        target="balanced",
        ctx_size=None,
        config=_config(),
        expected={},
    )
    result = registry.lookup(
        path, _model(), _llama("build-b", bench_sha256="binary-b"), ("new-hardware",)
    )
    assert result["status"] == "stale"
    assert result["stale_reasons"] == ["llama.cpp build changed", "hardware changed"]


def test_load_tolerates_torn_tail_and_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    assert registry.load_all(path) == []
    path.write_text('{"model_fingerprint":"ok"}\n{"model_fingerprint":')
    assert registry.load_all(path) == [{"model_fingerprint": "ok"}]


def test_load_rejects_corrupt_nonfinal_line(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    path.write_text("broken\n{}\n")
    with pytest.raises(ValueError, match="line 1"):
        registry.load_all(path)


def test_load_skips_blank_lines_and_rejects_nonobject(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    path.write_text("\n{}\n")
    assert registry.load_all(path) == [{}]
    path.write_text("[]\n")
    with pytest.raises(ValueError, match="not an object"):
        registry.load_all(path)


def test_lookup_reports_each_stale_identity_independently(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    registry.append_record(
        path,
        model_report=_model(),
        llama_report=_llama(),
        hardware_signature=("hardware",),
        session_dir=tmp_path / "session",
        target="balanced",
        ctx_size=None,
        config=_config(),
        expected={},
    )
    build_only = registry.lookup(
        path, _model(), _llama("new-build", bench_sha256="binary-b"), ("hardware",)
    )
    hardware_only = registry.lookup(path, _model(), _llama(), ("new-hardware",))
    assert build_only["stale_reasons"] == ["llama.cpp build changed"]
    assert hardware_only["stale_reasons"] == ["hardware changed"]


def test_append_recovers_torn_final_line(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    path.write_text('{"model_fingerprint":"old"}\n{"torn"')
    registry.append_record(
        path,
        model_report=_model(),
        llama_report=_llama(),
        hardware_signature=("hardware",),
        session_dir=tmp_path / "session",
        target="balanced",
        ctx_size=8192,
        config=_config(),
        expected={"score": 2.0},
    )
    records = registry.load_all(path)
    assert len(records) == 2
    assert records[-1]["model_fingerprint"] == _model().fingerprint
