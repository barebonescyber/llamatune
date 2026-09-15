"""Unit tests for llamatune.recommend (analysis assembly and emission payloads)."""

from __future__ import annotations

import dataclasses
import shlex
from pathlib import Path
from typing import Any

import pytest

from llamatune import recommend, report
from llamatune.types import BaselineResult, LlamaCppReport, MetricStats, ModelReport, TrialConfig

_ALL_CAPS = frozenset({"fa", "mmp", "nkvo", "ctk", "ctv", "ncmoe", "r", "o"})


def _config(**overrides: Any) -> TrialConfig:
    fields: dict[str, Any] = {
        "gpu_layers": 33,
        "moe_cpu_layers": 0,
        "flash_attn": True,
        "ubatch": 512,
        "batch": 2048,
        "threads": 8,
        "mmap": True,
        "no_kv_offload": False,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
    }
    fields.update(overrides)
    return TrialConfig(**fields)


def _record(
    trial_id: str,
    status: str = "ok",
    *,
    pp: float | None = 600.0,
    tg: float | None = 40.0,
    score: float | None = 1.0,
    config: TrialConfig | None = None,
) -> dict[str, Any]:
    return {
        "type": "trial",
        "trial_id": trial_id,
        "status": status,
        "config": (config or _config()).to_dict(),
        "pp_mean": pp,
        "tg_mean": tg,
        "score": score,
        "flags": ["-ngl", "33"],
    }


def _baseline() -> BaselineResult:
    return BaselineResult(
        runs=3,
        pp=MetricStats(mean=500.0, stdev=1.0, cv=0.002, n=3),
        tg=MetricStats(mean=30.0, stdev=0.1, cv=0.003, n=3),
        noise_floor_cv=0.01,
        fallback=None,
        resolved_defaults=_config(flash_attn=False).to_dict(),
    )


def _model() -> ModelReport:
    return ModelReport(
        path=Path("/models/m.gguf"),
        size_bytes=4_000_000_000,
        architecture="llama",
        n_layer=32,
        ngl_all=33,
        expert_count=8,
        moe=True,
        name="m",
        fingerprint="fp",
        full_sha256=None,
    )


def _llama() -> LlamaCppReport:
    return LlamaCppReport(
        bench_path=Path("/bin/llama-bench"),
        cli_path=None,
        server_path=None,
        capabilities=_ALL_CAPS,
        help_sha256="deadbeef",
        build_commit="fake1234",
        build_number=9999,
    )


class TestComputeCounts:
    def test_tallies_and_executed(self) -> None:
        records = [
            _record("a", "ok"),
            _record("b", "unstable"),
            _record("c", "oom", pp=None, tg=None, score=None),
            _record("cc", "cuda_error", pp=None, tg=None, score=None),
            _record("d", "timeout", pp=None, tg=None, score=None),
            _record("e", "crash", pp=None, tg=None, score=None),
            _record("f", "parse_error", pp=None, tg=None, score=None),
            _record("g", "pruned", pp=None, tg=None, score=None),
        ]
        counts = recommend.compute_counts(records)
        assert counts == {
            "executed": 7,
            "ok": 1,
            "unstable": 1,
            "oom": 1,
            "cuda_error": 1,
            "gpu_resource": 0,
            "timeout": 1,
            "crash": 1,
            "parse_error": 1,
            "pruned": 1,
        }

    def test_empty(self) -> None:
        counts = recommend.compute_counts([])
        assert counts["executed"] == 0
        assert counts["ok"] == 0


class TestComputeTop:
    def test_sorted_by_score_descending_and_limited(self) -> None:
        records = [_record(f"t{i:02d}", score=float(i)) for i in range(12)]
        top = recommend.compute_top(records)
        assert len(top) == 10
        assert [t["score"] for t in top] == [float(i) for i in range(11, 1, -1)]

    def test_failed_trials_excluded(self) -> None:
        records = [
            _record("ok1", score=1.5),
            _record("oom1", "oom", pp=None, tg=None, score=None),
            _record("pr1", "pruned", pp=None, tg=None, score=None),
        ]
        top = recommend.compute_top(records)
        assert [t["trial_id"] for t in top] == ["ok1"]

    def test_summary_shape(self) -> None:
        (summary,) = recommend.compute_top([_record("a", score=1.2)])
        assert set(summary) == {
            "trial_id",
            "status",
            "config",
            "pp_mean",
            "tg_mean",
            "score",
            "flags",
        }


class TestComputePareto:
    def test_dominated_trial_excluded(self) -> None:
        records = [
            _record("both", pp=700.0, tg=50.0, score=1.4),
            _record("worse", pp=600.0, tg=40.0, score=1.2),
            _record("ppbest", pp=800.0, tg=30.0, score=1.1),
        ]
        pareto = recommend.compute_pareto(records)
        assert [t["trial_id"] for t in pareto] == ["ppbest", "both"]


class TestBestLossless:
    def test_prefers_highest_scoring_lossless(self) -> None:
        records = [
            _record("lossy", score=2.0, config=_config(cache_type_k="q8_0")),
            _record("clean", score=1.5),
            _record("clean2", score=1.4),
        ]
        best = recommend.best_lossless(records)
        assert best is not None
        assert best["trial_id"] == "clean"

    def test_none_when_no_lossless_measured(self) -> None:
        records = [_record("lossy", score=2.0, config=_config(cache_type_v="q8_0"))]
        assert recommend.best_lossless(records) is None


class TestBuildAnalysis:
    def test_schema_shape(self) -> None:
        analysis = recommend.build_analysis(
            target="balanced",
            baseline=_baseline(),
            records=[_record("a", score=1.2)],
            winner=None,
            lossless_winner=None,
            warnings=["w1"],
        )
        assert set(analysis) == {
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
        assert analysis["schema_version"] == 2
        assert set(analysis["baseline"]) == {
            "runs",
            "pp",
            "tg",
            "noise_floor_cv",
            "fallback",
            "resolved_defaults",
            "kind",
        }
        assert analysis["baseline"]["pp"] == {"mean": 500.0, "stdev": 1.0, "cv": 0.002, "n": 3}
        assert analysis["warnings"] == ["w1"]
        assert analysis["baseline_kind"] == "defaults"
        assert analysis["counts"]["budget_consumed"] == analysis["counts"]["executed"]

    def test_explicit_budget_consumption_is_distinct_from_trial_record_count(self) -> None:
        analysis = recommend.build_analysis(
            target="balanced",
            baseline=_baseline(),
            records=[_record("a", score=1.2)],
            winner=None,
            lossless_winner=None,
            warnings=[],
            budget_consumed=9,
        )

        assert analysis["counts"]["executed"] == 1
        assert analysis["counts"]["budget_consumed"] == 9


class TestRecommendedJson:
    def test_payload_contents(self) -> None:
        config = _config()
        payload = recommend.build_recommended_json(
            config=config,
            capabilities=_ALL_CAPS,
            expected={"pp": 690.0, "tg": 42.0, "improvement_pct": {"score": 9.9}},
            confirmed=True,
            target="balanced",
            model=_model(),
            llama=_llama(),
        )
        assert payload["config"] == config.to_dict()
        assert payload["bench_flags"] == list(config.bench_args(_ALL_CAPS))
        assert payload["model"]["fingerprint"] == "fp"
        assert payload["llamacpp"] == {"build_commit": "fake1234", "build_number": 9999}
        assert payload["confirmed"] is True

    def test_config_round_trips_through_trial_config(self) -> None:
        payload = recommend.build_recommended_json(
            config=_config(),
            capabilities=_ALL_CAPS,
            expected={},
            confirmed=False,
            target="prompt",
            model=_model(),
            llama=_llama(),
        )
        assert TrialConfig.from_dict(payload["config"]) == _config()


class TestRecommendedSh:
    def test_alternates_skip_missing_and_dedupe_identical_rows(self) -> None:
        fallback = _config(gpu_layers=8, moe_cpu_layers=4)
        row = {
            "ctx": 65536,
            "status": "failed",
            "fallback_config": fallback.to_dict(),
        }
        text = recommend.build_recommended_sh(
            config=_config(),
            model=_model(),
            expected={},
            confirmed=True,
            target="balanced",
            moe=True,
            ctx_size=8192,
            context_envelope=[{"ctx": 32768, "fallback_config": None}, row, dict(row)],
        )
        assert text.count("Alternate for 65536 context") == 1
        assert "--n-cpu-moe 4" in text
        assert "-fa on -c 65536" in text
        assert "QUALITY-AFFECTING" not in text

    def test_lossy_context_alternate_carries_quality_warning(self) -> None:
        fallback = _config(
            gpu_layers=8,
            moe_cpu_layers=4,
            flash_attn=True,
            cache_type_k="q4_0",
            cache_type_v="q4_0",
        )

        text = recommend.build_recommended_sh(
            config=_config(),
            model=_model(),
            expected={},
            confirmed=True,
            target="balanced",
            moe=True,
            ctx_size=8192,
            context_envelope=[
                {"ctx": 65536, "status": "failed", "fallback_config": fallback.to_dict()}
            ],
        )

        assert "QUALITY-AFFECTING: lossy KV-cache alternate" in text
        assert "-ctk q4_0 -ctv q4_0" in text

    def test_flag_mapping(self) -> None:
        config = _config(
            moe_cpu_layers=24,
            flash_attn=True,
            mmap=False,
            no_kv_offload=True,
            cache_type_k="q8_0",
            cache_type_v="q8_0",
        )
        text = recommend.build_recommended_sh(
            config=config,
            model=_model(),
            expected={"pp": 1.0, "tg": 2.0, "improvement_pct": {"pp": 5.0, "tg": 3.0}},
            confirmed=True,
            target="balanced",
            moe=True,
        )
        assert "--n-cpu-moe 24" in text
        assert "-fa on" in text
        assert "--no-mmap" in text
        assert "--no-kv-offload" in text
        assert "-ctk q8_0" in text
        assert "-ctv q8_0" in text
        assert f"llama-server -m {_model().path}" in text
        assert f"llama-cli -m {_model().path}" in text
        # Reference only: every content line is commented out.
        for line in text.splitlines():
            assert line == "" or line.startswith("#")

    def test_lossless_defaults_omit_optional_flags(self) -> None:
        text = recommend.build_recommended_sh(
            config=_config(flash_attn=False),
            model=_model(),
            expected={},
            confirmed=False,
            target="balanced",
            moe=False,
        )
        assert "--n-cpu-moe" not in text
        assert "-fa on" not in text
        assert "--no-mmap" not in text
        assert "-ctk" not in text
        assert "no confirmed improvement" in text

    def test_context_is_included_in_runtime_commands(self) -> None:
        text = recommend.build_recommended_sh(
            config=_config(),
            model=_model(),
            expected={},
            confirmed=True,
            target="balanced",
            moe=True,
            ctx_size=8192,
        )
        assert "-c 8192" in text


@pytest.mark.parametrize("model_path", ["/models/plain.gguf", "/models/model one's.gguf"])
def test_recommended_commands_preserve_runtime_placement(model_path: str) -> None:
    cfg = _config(
        threads_batch=4,
        ot_spec="blk.0.ffn_.*_exps=CPU",
        moe_cpu_layers=3,
        tensor_split=(2.0, 1.0),
        split_mode="layer",
        mmap=False,
        no_kv_offload=True,
        cache_type_k="q8_0",
        cache_type_v="q4_0",
    )
    model = dataclasses.replace(_model(), path=Path(model_path))
    snippet = recommend.build_recommended_sh(
        config=cfg,
        model=model,
        expected={},
        confirmed=True,
        target="balanced",
        moe=True,
        ctx_size=4096,
    )
    exported = report.render_export(
        {"config": cfg.to_dict(), "model": {"path": str(model.path)}},
        {"options": {"ctx_size": 4096}},
        "llama-cli",
    )
    line = next(line[2:] for line in snippet.splitlines() if line.startswith("# llama-cli -m "))
    argv = shlex.split(line)
    assert argv == shlex.split(exported)
    for flag, value in (("-tb", "4"), ("-ot", cfg.ot_spec), ("-ts", "2,1"), ("-sm", "layer")):
        assert argv.count(flag) == 1
        assert argv[argv.index(flag) + 1] == value
    assert "--n-cpu-moe" not in argv


def test_runtime_flags_default_omission_and_moe_gate() -> None:
    from llamatune.runtimeflags import runtime_flags

    cfg = _config(moe_cpu_layers=3)
    assert "--n-cpu-moe" in runtime_flags(cfg, moe=True)
    assert "--n-cpu-moe" not in runtime_flags(cfg, moe=False)
    for flag in ("-tb", "-ot", "-ts", "-sm"):
        assert flag not in runtime_flags(cfg)


def test_config_text_has_no_duplicate_optional_fields() -> None:
    text = report._format_config(_config(threads_batch=4, ot_spec="x=CPU").to_dict())
    assert text.count("tb=") == 1
    assert text.count("ot=") == 1
