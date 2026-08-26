"""Regression tests for output-boundary sanitizers (SEC-003, SEC-004).

Covers the shared helpers in ``llamatune.sanitize``, the hostile-payload
escaping behavior of every Markdown renderer, control-character stripping at
the discovery/quality/qualserver boundaries, and byte-stability of one full
benign report render (golden fixture).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, cast

import pytest

from llamatune import discovery as discovery_module
from llamatune import marathonreport, matrixreport, qualityreport, qualserver, report
from llamatune.model import ModelInspectionError
from llamatune.nightreport import render as render_night
from llamatune.sanitize import markdown_cell, markdown_text, strip_control_chars
from llamatune.types import ResultRow, ResultsMatrix, TaskGrade, TrialConfig

_HOSTILE_HTML = "<img src=x onerror=alert(1)>"
_HOSTILE_NAME_STRIPPED = "[31m(Bevilmodel"
_HOSTILE_MODEL_NAME = f"{_HOSTILE_HTML}\x1b[31m\x07\x1b(B\revil\u202emodel"

_BENIGN_HARDWARE: dict[str, Any] = {
    "os_name": "Linux",
    "arch": "x86_64",
    "cpu_model": "AMD Ryzen 9 7950X",
    "physical_cores": 16,
    "logical_cores": 32,
    "perf_cores": None,
    "ram_mb": 131072,
    "gpus": [
        {
            "vendor": "nvidia",
            "name": "NVIDIA GeForce RTX 4090",
            "vram_mb": 24576,
            "method": "nvidia-smi",
        }
    ],
    "warnings": ["VRAM estimate is heuristic"],
}

_BENIGN_MODEL: dict[str, Any] = {
    "path": "/home/user/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "size_bytes": 4_921_000_000,
    "architecture": "llama",
    "n_layer": 32,
    "ngl_all": 33,
    "expert_count": 0,
    "moe": False,
    "name": "Llama 3.1 8B Instruct (Q4_K_M)",
    "fingerprint": "0123456789abcdef0123456789abcdef",
    "full_sha256": None,
}

_BENIGN_LLAMACPP: dict[str, Any] = {
    "bench_path": "/opt/llama.cpp/llama-bench",
    "cli_path": "/opt/llama.cpp/llama-cli",
    "server_path": None,
    "capabilities": ["fa", "mmp", "nkvo", "ctk", "ctv", "ncmoe"],
    "help_sha256": "deadbeef" * 8,
    "build_commit": "b4202",
    "bench_sha256": "cafebabe" * 8,
}

_BENIGN_SESSION: dict[str, Any] = {
    "schema_version": 2,
    "tool_version": "0.1.0b3",
    "argv": ["llamatune", "tune", "model.gguf"],
    "created": "2026-08-26T00:00:00+00:00",
    "options": {
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
        "ctx_size": 4096,
        "depth": None,
        "multi_gpu": False,
    },
}


def _benign_config(**overrides: object) -> dict[str, Any]:
    base: dict[str, Any] = {
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
    base.update(overrides)
    return base


def _benign_analysis() -> dict[str, Any]:
    return {
        "target": "balanced",
        "winner": {
            "trial_id": "t00000000000042",
            "pp": 5200.0,
            "tg": 830.0,
            "improvement_pct": {"pp": 4.0, "tg": 3.5, "score": 3.75},
            "config": _benign_config(gpu_layers=999),
        },
        "lossless_winner": None,
        "baseline": {
            "runs": 3,
            "pp": {"mean": 5000.0, "stdev": 50.0, "cv": 0.01, "n": 3},
            "tg": {"mean": 800.0, "stdev": 8.0, "cv": 0.01, "n": 3},
            "noise_floor_cv": 0.01,
            "fallback": None,
            "kind": "measured",
            "resolved_defaults": {},
        },
        "default_probe": {
            "status": "ok",
            "classification": "completed",
            "pattern": None,
            "evidence": "exit=0",
        },
        "feasibility": {
            "workload": {"pp": 512, "tg": 128},
            "ctx_validated": 4096,
            "vram_reserve_mb": 1024,
            "reserve_provenance": "default",
            "boundaries": [],
            "max_fitting": {"gpu_layers": 64, "moe_cpu_layers": 0, "reason": "fits with reserve"},
            "best_measured": None,
            "recommended": {"gpu_layers": 48, "moe_cpu_layers": 0, "reason": "safety margin"},
            "estimate": {
                "weights_mb": 4700.0,
                "kv_mb": 512.0,
                "compute_mb": 300.0,
                "total_mb": 5512.0,
                "budget_mb": 23552.0,
                "kv_basis": "heuristic",
                "calibrated": True,
            },
        },
        "context_validation": {"ctx": 4096, "status": "ok", "evidence": "probe"},
        "cli_validation": {"status": "ok", "evidence": "help"},
        "estimate_vs_observed": None,
        "quality_gate": None,
        "telemetry": None,
        "coverage": {},
        "top": [],
        "pareto": [],
        "counts": {"executed": 12, "ok": 10, "pruned": 2},
        "warnings": ["shard group kept first copy", "calibration reused prior evidence"],
    }


class TestStripControlChars:
    def test_keeps_newline_tab_and_printable_text(self) -> None:
        value = "line one\nline two\ttabbed — ünïcode ✓"
        assert strip_control_chars(value) == value

    def test_removes_c0_and_carriage_return(self) -> None:
        assert strip_control_chars("a\x00b\x07c\rd\x0be") == "abcde"

    def test_removes_del_and_c1_range(self) -> None:
        assert strip_control_chars("a\x7fb\x85c\x9fd") == "abcd"

    def test_removes_ansi_escape_sequences(self) -> None:
        assert strip_control_chars("\x1b[31mred\x1b[0m") == "[31mred[0m"

    def test_removes_bidi_direction_overrides(self) -> None:
        assert strip_control_chars("safe\u202ename\u2066dir\u2069") == "safenamedir"

    def test_custom_keep_set(self) -> None:
        assert strip_control_chars("a\rb\nc", keep="\r") == "a\rbc"
        assert strip_control_chars("a\rb\nc", keep="") == "abc"

    def test_empty_string(self) -> None:
        assert strip_control_chars("") == ""


class TestMarkdownText:
    def test_benign_names_pass_through_unescaped(self) -> None:
        assert markdown_text("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf") == (
            "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        )
        assert markdown_text("Llama 3.1 8B Instruct (Q4_K_M)") == ("Llama 3.1 8B Instruct (Q4_K_M)")
        assert markdown_text("/models/qwen2.5/Qwen2.5-7B-Instruct [fp16].gguf") == (
            "/models/qwen2.5/Qwen2.5-7B-Instruct [fp16].gguf"
        )

    def test_escapes_html_special_characters(self) -> None:
        assert markdown_text("&") == "&amp;"
        assert markdown_text("<") == "&lt;"
        assert markdown_text(">") == "&gt;"
        assert markdown_text('"') == "&quot;"
        assert markdown_text("'") == "&#39;"

    def test_doubles_backslashes_before_other_escapes(self) -> None:
        assert markdown_text("a\\&b") == "a\\\\&amp;b"

    def test_hostile_html_renders_as_text(self) -> None:
        assert markdown_text(_HOSTILE_HTML) == "&lt;img src=x onerror=alert(1)&gt;"

    def test_strips_control_characters(self) -> None:
        assert markdown_text("a\x1b[31mb\rc") == "a[31mbc"

    def test_javascript_link_survives_as_inert_literal_text(self) -> None:
        # Brackets stay literal (benign names use parentheses), so scheme
        # neutralization is delegated to the rendering pipeline (GitHub strips
        # javascript: URLs). Our contract here: no raw HTML markup survives.
        assert markdown_text("[link](javascript:alert(1))") == "[link](javascript:alert(1))"
        assert "<" not in markdown_text("[link](javascript:alert(1))")

    def test_non_string_values_are_coerced(self) -> None:
        assert markdown_text(4096) == "4096"


class TestMarkdownCell:
    def test_escapes_pipes_and_flattens_newlines(self) -> None:
        assert markdown_cell("a|b\nc") == "a\\|b c"

    def test_full_hostile_cell(self) -> None:
        assert markdown_cell("x|y\nz\r\x1b[31m<img>") == "x\\|y z[31m&lt;img&gt;"


class TestReportRendererHostile:
    def test_model_section_escapes_hostile_fields(self) -> None:
        model = dict(_BENIGN_MODEL)
        model["name"] = _HOSTILE_MODEL_NAME
        model["path"] = "/models/<script>alert('x')</script>.gguf"
        model["architecture"] = "\x1b[32mllama"
        text = report.render(
            _benign_analysis(), _BENIGN_SESSION, _BENIGN_HARDWARE, model, _BENIGN_LLAMACPP
        )
        name_line = next(line for line in text.splitlines() if line.startswith("- Name: "))
        assert (
            f"- Name: {_HOSTILE_HTML.replace('<', '&lt;').replace('>', '&gt;')}"[:30] in name_line
        )
        assert "alert(1)" in name_line  # payload preserved as inert text
        assert "\x1b" not in name_line
        assert "\r" not in name_line
        assert "\x07" not in name_line
        assert "\u202e" not in name_line
        assert "`/models/&lt;script&gt;alert(&#39;x&#39;)&lt;/script&gt;.gguf`" in text
        assert "- Architecture: [32mllama" in text

    def test_warnings_section_escapes_entries(self) -> None:
        analysis = _benign_analysis()
        analysis["warnings"] = ["boom <b>bold</b>\x1b[31m"]
        text = report.render(
            analysis, _BENIGN_SESSION, _BENIGN_HARDWARE, _BENIGN_MODEL, _BENIGN_LLAMACPP
        )
        assert "- boom &lt;b&gt;bold&lt;/b&gt;[31m" in text

    def test_search_plan_terminal_echo_is_control_char_free(self) -> None:
        plan: dict[str, Any] = {"dimensions": {}, "ncmoe_ladder": [], "estimates": []}
        text = report.render_search_plan(plan, {}, {"name": "tiny \x1b[31mmodel\r"})
        assert "\x1b" not in text
        assert "\r" not in text


class TestMatrixReportHostile:
    def _matrix_with_hostile_label(self, label: str) -> ResultsMatrix:
        trial = TrialConfig(
            gpu_layers=33,
            moe_cpu_layers=0,
            flash_attn=True,
            ubatch=512,
            batch=2048,
            threads=8,
            mmap=True,
            no_kv_offload=False,
            cache_type_k="f16",
            cache_type_v="f16",
        )
        row = ResultRow(
            row_id="r-000001",
            kind="recommendation",
            current=True,
            model_fingerprint="0123456789abcdef0123456789abcdef",
            model_name=label,
            model_path="/models/hostile.gguf",
            quant="Q4|K|M",
            hardware_hash="hw-1234",
            hardware_signature=("nvidia", 24),
            build_discriminator="b4202-cafebabe",
            build_commit="b4202",
            config=trial,
            ctx=4096,
            depth=None,
            pp_workload=512,
            tg_workload=128,
            suite_id=None,
            metrics={"perf.pp": 5200.0, "perf.tg": 830.0},
            status="replicated",
            confirmed=True,
            replicated=True,
            reps=5,
            noise_floor_cv=0.01,
            source_root=Path("/sessions"),
            evidence_dir=Path("ev"),
            ts="2026-08-26T00:00:00+00:00",
        )
        return ResultsMatrix(
            schema_version=1,
            generated="2026-08-26T01:00:00+00:00",
            roots=(Path("/sessions"),),
            warnings=("stale \x1b[31mrow",),
            rows=(row,),
        )

    def test_model_heading_escapes_and_keeps_pipe_discipline(self) -> None:
        text = matrixreport.render_markdown(
            self._matrix_with_hostile_label(_HOSTILE_MODEL_NAME + "|pipe\njump")
        )
        heading = next(line for line in text.splitlines() if line.startswith("## &lt;"))
        assert heading.startswith("## &lt;img")
        assert "\\|pipe jump" in heading  # pipe escaped, newline flattened
        for forbidden in ("<img", "\x1b", "\r", "\x07", "\u202e"):
            assert forbidden not in heading

    def test_quant_and_warning_cells_are_escaped(self) -> None:
        text = matrixreport.render_markdown(self._matrix_with_hostile_label("plain-name"))
        assert "`Q4\\|K\\|M`" in text
        assert "- stale [31mrow" in text
        # best-results table stays a strict six-column grid
        best_header = next(line for line in text.splitlines() if line.startswith("| Use case"))
        assert best_header.count("|") == 7
        body = next(
            line
            for line in text.splitlines()
            if line.startswith("| max-pp") or line.startswith("| balanced")
        )
        assert body.count("|") == 7


class TestNightReportHostile:
    def test_model_reason_and_warning_cells_are_escaped(self) -> None:
        summary = {
            "items": [
                {
                    "kind": "calibrate",
                    "model": _HOSTILE_MODEL_NAME,
                    "fingerprint": "a" * 64,
                    "outcome": "error",
                    "reason": "server said <b>no</b>\x1b[31m",
                },
                {
                    "kind": "tune",
                    "model": "deferred|x.gguf",
                    "outcome": "deferred",
                    "reason": "deadline",
                    "estimated_minutes": 5,
                },
            ],
            "warnings": ["warn <i>me</i>"],
        }
        text = render_night(summary)
        for line in text.splitlines():
            if line.startswith("| ") and "---" not in line:
                for forbidden in ("<img", "\x1b", "\r"):
                    assert forbidden not in line
        assert "server said &lt;b&gt;no&lt;/b&gt;[31m" in text
        assert "deferred\\|x.gguf" in text
        assert "- warn &lt;i&gt;me&lt;/i&gt;" in text


class TestMarathonReportHostile:
    def test_stop_reason_warning_and_cells_are_escaped(self) -> None:
        summary: dict[str, Any] = {
            "stop_reason": f"done{_HOSTILE_HTML}",
            "rounds": [{"index": 1, "outcome": "ok|x"}],
            "champion": {},
            "warnings": ["w<arn\ring\x1b[31m"],
        }
        text = marathonreport.render(summary)
        assert (
            f"- Stop reason: done{_HOSTILE_HTML.replace('<', '&lt;').replace('>', '&gt;')}" in text
        )
        assert "- w&lt;arning[31m" in text
        round_row = next(line for line in text.splitlines() if line.startswith("| 1 "))
        assert "ok\\|x" in round_row


class TestQualityReportHostile:
    def test_model_name_failed_details_and_warnings_are_escaped(self) -> None:
        summary = {
            "model": {"name": _HOSTILE_MODEL_NAME, "path": "/m<x>.gguf"},
            "suites": [
                {
                    "suite_id": "coding@v1",
                    "name": "coding",
                    "kind": "coding",
                    "metrics": {"score": 0.5},
                    "tasks": [
                        {
                            "id": "task|broken",
                            "score": 0.0,
                            "status": "graded",
                            "reason": None,
                            "graders": [
                                {
                                    "type": "exec_python",
                                    "passed": False,
                                    "detail": "exit=1 stderr='\x1b[31mboom'",
                                }
                            ],
                        }
                    ],
                }
            ],
            "warnings": ["w<arn\ring"],
        }
        text = qualityreport.render(summary)
        assert f"- Model: `&lt;img src=x onerror=alert(1)&gt;{_HOSTILE_NAME_STRIPPED}`" in text
        assert "task\\|broken" in text
        assert "exit=1 stderr=&#39;[31mboom&#39;" in text
        assert "w&lt;arning" in text
        for forbidden in ("<img", "\x1b", "\r"):
            assert forbidden not in text

    def test_local_json_payloads_keep_literal_reading(self) -> None:
        summary = {
            "model": {"name": "tiny", "path": "/m.gguf"},
            "config": {"cache_type_v": "q8_0", "gpu_layers": 99},
            "hardware_signature": [{"vendor": "nvidia"}],
            "suites": [],
        }
        text = qualityreport.render(summary)
        assert '| cache_type_v | "q8_0" |' in text  # quotes NOT entity-mangled
        assert '`[{"vendor": "nvidia"}]`' in text


class TestDiscoveryWarningsStripped:
    @pytest.mark.parametrize(
        ("filenames", "expected_fragment"),
        [
            (("we\x1bi_rd-1-of-2.gguf", "we\x1bi_rd-3-of-3.gguf"), "shard group wei_rd"),
            (("solo\x07bad.gguf",), "solobad.gguf"),
        ],
    )
    def test_shard_and_unreadable_warnings_have_no_control_chars(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        filenames: tuple[str, ...],
        expected_fragment: str,
    ) -> None:
        def _raise_inspection(*args: Any, **kwargs: Any) -> None:
            raise ModelInspectionError("parse \x1b[31mfailed\r")

        for name in filenames:
            (tmp_path / name).write_bytes(b"not a gguf")
        monkeypatch.setattr(discovery_module, "inspect_model", _raise_inspection)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            discovery_module.discover_models(tmp_path, (), ())
        messages = [str(item.message) for item in caught]
        assert any(expected_fragment in message for message in messages)
        for message in messages:
            for forbidden in ("\x1b", "\x07", "\r"):
                assert forbidden not in message


class TestQualityBoundariesStrip:
    def test_error_and_skipped_grades_strip_reason(self) -> None:
        from llamatune.quality import _error_grade, _skipped_grade

        assert _error_grade("t", "boom\x1b[31m\r!").reason == "boom[31m!"
        assert _skipped_grade("t", "skip\x07ping").reason == "skipping"

    def test_grade_dict_serialization_strips_reason(self) -> None:
        from llamatune.quality import _grade_dict

        grade = TaskGrade(
            task_id="t",
            score=0.0,
            status="error",
            reason="srv\x1b[31merr\r",
            unstable=False,
            grader_results=(),
        )
        assert _grade_dict(grade)["reason"] == "srv[31merr"

    def test_stderr_tail_surface_strips_but_capture_stays_raw(self) -> None:
        class _FakeCapture:
            def __init__(self, tail: bytes) -> None:
                self._tail = tail

            @property
            def tail(self) -> bytes:
                return self._tail

        raw_tail = b"\x1b[31merr\r\nline two\ntab\tkept\n"
        handle = qualserver.ServerHandle(
            run=cast(Any, None),
            launch_number=1,
            argv=(),
            port=0,
            process=cast(Any, None),
            control=cast(Any, None),
            stdout_capture=cast(Any, None),
            stderr_capture=cast(Any, _FakeCapture(raw_tail)),
        )
        surfaced = handle.stderr_tail
        assert "\x1b" not in surfaced
        assert "\r" not in surfaced
        assert surfaced == "[31merr\nline two\ntab\tkept\n"
        assert handle._stderr_capture.tail == raw_tail


class TestBenignGoldenStability:
    _GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "golden_report_benign.md"

    def test_full_benign_report_render_matches_golden_fixture(self) -> None:
        golden = self._GOLDEN.read_text(encoding="utf-8")
        rendered = report.render(
            _benign_analysis(), _BENIGN_SESSION, _BENIGN_HARDWARE, _BENIGN_MODEL, _BENIGN_LLAMACPP
        )
        assert rendered == golden

    def test_render_is_deterministic_across_calls(self) -> None:
        args = (_benign_analysis(), _BENIGN_SESSION, _BENIGN_HARDWARE, _BENIGN_MODEL)
        assert report.render(*args, _BENIGN_LLAMACPP) == report.render(*args, _BENIGN_LLAMACPP)
