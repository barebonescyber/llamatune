from __future__ import annotations

from llamatune.qualityreport import render


def _summary() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model": {
            "name": "Fixture | Model",
            "path": "/models/fixture.gguf",
            "fingerprint": "abcdef0123456789",
        },
        "hardware_signature": ["Linux", "x86_64", ["GPU"]],
        "build": {"bench_sha256": "bench-a", "build_commit": "commit-a"},
        "ctx": 8192,
        "seed": 42,
        "exec_enabled": True,
        "overall": 0.75,
        "config_source": "session",
        "config_provenance": "/sessions/source",
        "config": {
            "threads": 8,
            "gpu_layers": 33,
            "flash_attn": True,
            "cache_type_k": "f16",
        },
        "comparison": {
            "lossless_config": {"gpu_layers": 0},
            "suites": [
                {
                    "suite_id": "coding@1+abc",
                    "evaluated": {"score": 0.75, "pass_rate": 0.5},
                    "lossless": {"score": 0.9, "pass_rate": 1.0},
                    "delta_score": -0.15,
                }
            ],
            "degradation_warning": "quality delta exceeded threshold",
        },
        "suites": [
            {
                "suite_id": "coding@1+abc",
                "name": "coding",
                "kind": "coding",
                "metrics": {"score": 0.75, "pass_rate": 0.5, "exec_enabled": 1.0},
                "tasks": [
                    {
                        "id": "failed-1",
                        "score": 0.5,
                        "status": "graded",
                        "reason": None,
                        "graders": [
                            {"grader": "defines", "passed": False, "detail": "function absent"}
                        ],
                    },
                    {
                        "id": "skipped-1",
                        "score": 0.0,
                        "status": "skipped",
                        "reason": "ctx below minimum",
                        "graders": [],
                    },
                    {
                        "id": "error-1",
                        "score": 0.0,
                        "status": "error",
                        "reason": "request failed",
                        "graders": [],
                    },
                ],
            }
        ],
        "warnings": ["sandbox network namespace unavailable"],
        "server_launches": [
            {"argv": ["llama-server", "--host", "127.0.0.1", "-m", "/models/a b.gguf"]}
        ],
        "argv": ["llamatune", "quality", "/models/a b.gguf", "--exec"],
    }


def test_report_contains_every_required_section_and_details() -> None:
    report = render(_summary())
    assert report.startswith("# Quality Evaluation Report (experimental)\n")
    for heading in (
        "## Summary",
        "## Configuration",
        "## Per-suite results",
        "## Skipped and errored tasks",
        "## Environment",
        "## Reproduce",
    ):
        assert heading in report
    assert "Fixture \\| Model" in report
    assert "Overall score: `0.75`" in report
    assert "Execution tier: `enabled`" in report
    assert "| gpu_layers | 33 |" in report
    assert report.index("| gpu_layers |") < report.index("| flash_attn |")
    assert "Lossless configuration: `{" in report
    assert (
        '| coding@1+abc | {"pass_rate":0.5,"score":0.75} | {"pass_rate":1.0,"score":0.9} | -0.15 |'
    ) in report
    assert "Degradation warning: quality delta exceeded threshold" in report
    assert "`failed-1`: function absent" in report
    assert "`coding/skipped-1` (skipped): ctx below minimum" in report
    assert "`coding/error-1` (error): request failed" in report
    assert "sandbox network namespace unavailable" in report
    assert "llamatune quality '/models/a b.gguf' --exec" in report
    assert "llama-server --host 127.0.0.1 -m '/models/a b.gguf'" in report


def test_report_is_deterministic_and_handles_minimal_summary() -> None:
    summary = _summary()
    assert render(summary) == render(summary)
    minimal = render({})
    assert "Overall score: `unavailable`" in minimal
    assert "Warnings: `0`" in minimal
    assert "## Per-suite results" in minimal
    assert "## Skipped and errored tasks\n\nNone." in minimal
    assert "Server launches: `unavailable in schema v1`" in minimal
    assert "unavailable in schema v1\n```" in minimal
    assert minimal.endswith("\n")


def test_report_comparison_without_degradation_has_no_warning() -> None:
    summary = _summary()
    summary["comparison"] = {"lossless_config": {}, "suites": [], "degradation_warning": None}
    report = render(summary)
    assert "Lossless configuration: `{}`" in report
    assert "Degradation warning:" not in report


def test_boolean_degradation_warning_uses_required_text() -> None:
    summary = _summary()
    comparison = summary["comparison"]
    assert isinstance(comparison, dict)
    comparison["degradation_warning"] = True
    report = render(summary)
    assert "Degradation warning: lossy cache measurably degrades quality on this machine" in report


def test_forward_compatible_optional_shapes_render_deterministically() -> None:
    summary: dict[str, object] = {
        "model": "invalid",
        "config": {"custom": [1, 2]},
        "warnings": "invalid",
        "suites": [
            "invalid",
            {
                "name": "custom",
                "metrics": {},
                "tasks": [
                    {
                        "id": "failed-with-reason",
                        "score": 0.0,
                        "status": "graded",
                        "reason": "explicit reason",
                        "graders": [],
                    }
                ],
            },
        ],
        "comparison": {
            "lossless_config": None,
            "suites": ["invalid"],
            "degradation_warning": False,
        },
        "server_launches": "unknown",
        "argv": "llamatune quality model.gguf",
        "server_argv": ("llama-server", "--host", "127.0.0.1"),
    }
    report = render(summary)
    assert "| custom | [1, 2] |" in report
    assert "`failed-with-reason`: explicit reason" in report
    assert "Server launches: `unknown`" in report
    assert "llamatune quality model.gguf" in report
    assert "llama-server --host 127.0.0.1" in report
