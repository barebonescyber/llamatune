"""End-to-end quality orchestration against the deterministic fake server."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import signal
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from llamatune import quality, qualserver, sandbox
from llamatune.model import inspect_model
from llamatune.qualsuites import load_suite
from llamatune.types import QualityOptions, TrialConfig


@pytest.fixture(autouse=True)
def _fast_server_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qualserver, "_READINESS_POLL_S", 0.01)


def _install_server(llama_bin: Path) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "fake_llama_server.py"
    if sys.platform == "win32":
        script = llama_bin / "fake_llama_server.py"
        shutil.copy2(source, script)
        (llama_bin / "llama-server.cmd").write_text(
            f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8"
        )
    else:
        destination = llama_bin / "llama-server"
        shutil.copy2(source, destination)
        destination.chmod(0o755)


def _install_perplexity(llama_bin: Path) -> None:
    script = llama_bin / "fake_llama_perplexity.py"
    script.write_text(
        "#!/usr/bin/env python3\nprint('Final estimate: PPL = 12.5')\n",
        encoding="utf-8",
    )
    if sys.platform == "win32":
        (llama_bin / "llama-perplexity.cmd").write_text(
            f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8"
        )
    else:
        destination = llama_bin / "llama-perplexity"
        shutil.copy2(script, destination)
        destination.chmod(0o755)


def _options(
    model: Path,
    llama_bin: Path,
    root: Path,
    suites: tuple[str, ...],
    *,
    filters: tuple[str, ...] = (),
    exec_enabled: bool = False,
    config_session: Path | None = None,
    compare_lossless: bool = False,
    ctx: int = 4096,
    dry_run: bool = False,
    quality_corpus: Path | None = None,
) -> QualityOptions:
    return QualityOptions(
        model_path=model,
        llama_bin=llama_bin,
        sessions_dir=root,
        config_mode="session" if config_session is not None else "defaults",
        config_session=config_session,
        strict_config=False,
        compare_lossless=compare_lossless,
        suites=suites,
        task_filters=filters,
        exec_enabled=exec_enabled,
        ctx_size=ctx,
        quality_corpus=quality_corpus,
        reps=1,
        max_tokens=512,
        request_timeout_s=3.0,
        server_start_timeout_s=3.0,
        seed=42,
        dry_run=dry_run,
    )


def _write_suite(root: Path, name: str, kind: str, tasks: list[dict[str, Any]]) -> Path:
    path = root / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": name,
                "version": 1,
                "kind": kind,
                "tasks": tasks,
            }
        ),
        encoding="utf-8",
    )
    return path


def _key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


def _write_script(root: Path, responses: dict[str, str]) -> Path:
    path = root / f"script-{len(list(root.glob('script-*.json')))}.json"
    path.write_text(json.dumps(responses), encoding="utf-8")
    return path


def _coding_response(task: dict[str, Any]) -> str:
    signature = next(grader for grader in task["graders"] if grader["type"] == "signature")
    params = ", ".join(signature["params"])
    return f"```python\ndef {signature['name']}({params}):\n    return None\n```"


def _bundled_correct_script(ctx: int) -> dict[str, str]:
    responses: dict[str, str] = {}
    coding = load_suite("coding")
    task = next(item for item in coding.tasks if item["id"] == "clamp")
    responses[_key(str(task["prompt"]))] = _coding_response(task)

    tooluse = load_suite("tooluse")
    task = next(item for item in tooluse.tasks if item["id"] == "weather-current")
    for index, step in enumerate(task["turns"]):
        if "user" not in step:
            continue
        following = task["turns"][index + 1]
        if "expect" in following:
            expected = following["expect"]
            args = expected.get("args_equal", expected.get("args_subset", {}))
            answer = {"tool": expected["tool"], "args": args}
        else:
            answer = {"final": "12 C"}
        responses[_key(step["user"])] = f"```json\n{json.dumps(answer)}\n```"

    agentic = load_suite("agentic")
    task = next(item for item in agentic.tasks if item["id"] == "restock")
    responses[_key(str(task["prompt"]))] = (
        '```json\n{"tool":"order","args":{"sku":"fixture","amount":5}}\n```'
    )

    ifollow = load_suite("ifollow")
    task = next(item for item in ifollow.tasks if item["id"] == "json-status")
    messages = quality._messages_for_task(task, ctx)
    responses[_key(str(messages[-1]["content"]))] = '{"status":"ready","count":2}'
    return responses


def _coding_tasks(count: int = 2, *, exec_graders: bool = False) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for index in range(count):
        graders: list[dict[str, Any]] = [
            {"type": "code_extract", "language": "python"},
            {"type": "python_parses"},
            {"type": "defines", "name": f"answer_{index}", "kind": "function"},
            {"type": "signature", "name": f"answer_{index}", "params": []},
        ]
        if exec_graders:
            graders.append(
                {
                    "type": "exec_python",
                    "requires_exec": True,
                    "tests": [f"assert answer_{index}() == 2"],
                }
            )
        tasks.append(
            {
                "id": f"task-{index}",
                "prompt": f"Write answer function {index}.",
                "language": "python",
                "graders": graders,
            }
        )
    return tasks


def _coding_script(tasks: list[dict[str, Any]], *, values: tuple[int, ...]) -> dict[str, str]:
    return {
        _key(str(task["prompt"])): (
            f"```python\ndef answer_{index}():\n    return {values[index]}\n```"
        )
        for index, task in enumerate(tasks)
    }


def test_all_http_suites_correct_emit_frozen_summary_and_matrix(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_server(fake_bin_dir)
    root = tmp_path / "quality-root"
    script = _write_script(tmp_path, _bundled_correct_script(4096))
    monkeypatch.setenv("LLAMATUNE_FAKE_SRV_SCRIPT", str(script))
    options = _options(
        tiny_gguf,
        fake_bin_dir,
        root,
        ("coding", "tooluse", "agentic", "ifollow"),
        filters=("clamp", "weather-current", "restock", "json-status"),
    )

    outcome = quality.run_quality(options)

    assert outcome.exit_code == 0
    assert all(suite["metrics"]["score"] == 1.0 for suite in outcome.summary["suites"])
    assert set(outcome.summary) == {
        "schema_version",
        "run_dir",
        "created",
        "model",
        "hardware_signature",
        "build",
        "ctx",
        "seed",
        "reps",
        "config",
        "config_source",
        "config_provenance",
        "exec_enabled",
        "suites",
        "overall",
        "comparison",
        "warnings",
    }
    assert (outcome.run_dir / "quality.json").is_file()
    assert (outcome.run_dir / "quality-report.md").is_file()
    assert (root / "matrix" / "results-matrix.json").is_file()
    request = json.loads(
        next((outcome.run_dir / "tasks" / "tooluse").rglob("request-0.json")).read_text()
    )
    assert request["messages"][0]["role"] == "system"
    assert "Available tools" in request["messages"][0]["content"]
    assert "exactly one fenced ```json block" in request["messages"][0]["content"]
    assert '"tool"' in request["messages"][0]["content"]


def test_wrong_grade_and_malformed_reply_degrade_to_exit_1(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_server(fake_bin_dir)
    tasks = _coding_tasks()
    suite = _write_suite(tmp_path, "mixed", "coding", tasks)
    monkeypatch.setenv(
        "LLAMATUNE_FAKE_SRV_SCRIPT",
        str(_write_script(tmp_path, {"*": "not python"})),
    )
    original = qualserver.ServerHandle.chat
    calls = 0

    def mixed(self: qualserver.ServerHandle, *args: Any, **kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise qualserver.ServerProtocolError("malformed fixture reply")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(qualserver.ServerHandle, "chat", mixed)
    outcome = quality.run_quality(
        _options(tiny_gguf, fake_bin_dir, tmp_path / "root", (str(suite),))
    )

    assert outcome.exit_code == 1
    grades = outcome.summary["suites"][0]["tasks"]
    assert grades[0]["score"] == 0.0 and grades[0]["status"] == "graded"
    assert grades[1]["status"] == "error"


def test_exec_and_no_exec_paths_never_overlap_server_and_sandbox(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_server(fake_bin_dir)
    tasks = _coding_tasks(exec_graders=True)
    suite = _write_suite(tmp_path, "exec-suite", "coding", tasks)
    monkeypatch.setenv(
        "LLAMATUNE_FAKE_SRV_SCRIPT",
        str(_write_script(tmp_path, _coding_script(tasks, values=(2, 3)))),
    )
    active_server = False
    original_start = qualserver.start
    original_stop = qualserver.ServerHandle.stop
    original_sandbox = sandbox.run_python

    def tracked_start(*args: Any, **kwargs: Any) -> qualserver.ServerHandle:
        nonlocal active_server
        handle = original_start(*args, **kwargs)
        active_server = True
        return handle

    def tracked_stop(self: qualserver.ServerHandle) -> None:
        nonlocal active_server
        original_stop(self)
        active_server = False

    def checked_sandbox(code: str, *, timeout_s: float, **_kwargs: Any) -> Any:
        assert active_server is False
        return original_sandbox(code, timeout_s=timeout_s)

    monkeypatch.setattr(quality, "start", tracked_start)
    monkeypatch.setattr(qualserver.ServerHandle, "stop", tracked_stop)
    monkeypatch.setattr(sandbox, "run_python", checked_sandbox)
    sandbox.set_allow_network_fallback(True)
    try:
        enabled = quality.run_quality(
            _options(
                tiny_gguf,
                fake_bin_dir,
                tmp_path / "enabled",
                (str(suite),),
                exec_enabled=True,
            )
        )
    finally:
        sandbox.set_allow_network_fallback(False)
    if os.name == "posix":
        assert enabled.exit_code == 0
        assert [task["score"] for task in enabled.summary["suites"][0]["tasks"]] == [1.0, 0.0]
    else:
        assert enabled.exit_code == 1
        assert all(task["status"] == "error" for task in enabled.summary["suites"][0]["tasks"])

    disabled = quality.run_quality(
        _options(tiny_gguf, fake_bin_dir, tmp_path / "disabled", (str(suite),))
    )
    assert disabled.exit_code == 0
    assert disabled.summary["exec_enabled"] is False
    assert "exec_isolation" not in disabled.summary
    assert any(
        grader.get("grader") == "exec_python"
        and grader.get("detail") == "execution disabled"
        and grader.get("skipped") is True
        for task in disabled.summary["suites"][0]["tasks"]
        for grader in task["graders"]
    )

    def unavailable_sandbox(code: str, *, timeout_s: float, **_kwargs: Any) -> Any:
        del code, timeout_s
        raise RuntimeError("sandbox limits unavailable")

    monkeypatch.setattr(sandbox, "run_python", unavailable_sandbox)
    sandbox.set_allow_network_fallback(True)
    try:
        unavailable = quality.run_quality(
            _options(
                tiny_gguf,
                fake_bin_dir,
                tmp_path / "unavailable",
                (str(suite),),
                exec_enabled=True,
            )
        )
    finally:
        sandbox.set_allow_network_fallback(False)
    assert unavailable.exit_code == 1
    assert all(task["status"] == "error" for task in unavailable.summary["suites"][0]["tasks"])


def test_exec_isolation_status_and_warnings_surface_in_summary(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_server(fake_bin_dir)
    tasks = _coding_tasks(exec_graders=True)
    suite = _write_suite(tmp_path, "exec-isolation", "coding", tasks)
    monkeypatch.setenv(
        "LLAMATUNE_FAKE_SRV_SCRIPT",
        str(_write_script(tmp_path, _coding_script(tasks, values=(2, 3)))),
    )
    calls: list[bool] = []

    def fake_run_python(code: str, *, timeout_s: float, **_kwargs: Any) -> Any:
        del code, timeout_s
        calls.append(sandbox.allow_network_fallback())
        return sandbox.ExecVerdict(
            passed=True,
            exit_code=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        )

    report = sandbox.IsolationReport(
        network_status=sandbox.IsolationStatus.AVAILABLE,
        network_active=True,
        filesystem_confinement="none",
        memory_limit="rlimit-as",
        summary="network-namespace",
        warnings=("filesystem confinement inactive (bubblewrap missing or unusable)",),
    )
    monkeypatch.setattr(sandbox, "run_python", fake_run_python)
    monkeypatch.setattr(sandbox, "describe_isolation", lambda **_kwargs: report)
    outcome = quality.run_quality(
        _options(
            tiny_gguf,
            fake_bin_dir,
            tmp_path / "isolation-root",
            (str(suite),),
            exec_enabled=True,
        )
    )

    assert outcome.exit_code == 0
    assert outcome.summary["exec_isolation"] == "network-namespace"
    assert any(
        "filesystem confinement inactive" in warning for warning in outcome.summary["warnings"]
    )
    assert all(call is False for call in calls)
    captured = capsys.readouterr()
    assert "WARNING: quality --exec degraded isolation" in captured.err


def test_exec_isolation_reports_allowed_fallback_summary(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_server(fake_bin_dir)
    tasks = _coding_tasks(exec_graders=True)
    suite = _write_suite(tmp_path, "exec-fallback", "coding", tasks)
    monkeypatch.setenv(
        "LLAMATUNE_FAKE_SRV_SCRIPT",
        str(_write_script(tmp_path, _coding_script(tasks, values=(2, 2)))),
    )

    def fake_run_python(code: str, *, timeout_s: float, **_kwargs: Any) -> Any:
        del code, timeout_s
        return sandbox.ExecVerdict(
            passed=True,
            exit_code=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        )

    monkeypatch.setattr(sandbox, "run_python", fake_run_python)
    sandbox.set_allow_network_fallback(True)
    try:
        outcome = quality.run_quality(
            _options(
                tiny_gguf,
                fake_bin_dir,
                tmp_path / "fallback-root",
                (str(suite),),
                exec_enabled=True,
            )
        )
        summary = outcome.summary
    finally:
        sandbox.set_allow_network_fallback(False)

    assert "exec_isolation" in summary
    assert summary["exec_isolation"] in {"none (allowed by flag)", "network-namespace"}


def test_server_death_relaunches_once_then_degrades_on_second_death(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_server(fake_bin_dir)
    tasks = _coding_tasks(3)
    suite = _write_suite(tmp_path, "death", "coding", tasks)
    monkeypatch.setenv(
        "LLAMATUNE_FAKE_SRV_SCRIPT",
        str(_write_script(tmp_path, _coding_script(tasks, values=(2, 2, 2)))),
    )

    with monkeypatch.context() as context:
        context.setenv("LLAMATUNE_FAKE_SRV_DIE_AFTER_N", "1")
        original_start = qualserver.start
        launches = 0

        def recover_once(*args: Any, **kwargs: Any) -> qualserver.ServerHandle:
            nonlocal launches
            launches += 1
            if launches == 2:
                context.delenv("LLAMATUNE_FAKE_SRV_DIE_AFTER_N")
            return original_start(*args, **kwargs)

        context.setattr(quality, "start", recover_once)
        recovered = quality.run_quality(
            _options(tiny_gguf, fake_bin_dir, tmp_path / "recovered", (str(suite),))
        )
    assert recovered.exit_code == 0
    assert launches == 2

    monkeypatch.setenv("LLAMATUNE_FAKE_SRV_DIE_AFTER_N", "1")
    with monkeypatch.context() as context:
        context.setattr(
            quality,
            "_perplexity_result",
            lambda *args, **kwargs: pytest.fail("Phase 2 ran after suite abort"),
        )
        degraded = quality.run_quality(
            _options(tiny_gguf, fake_bin_dir, tmp_path / "degraded", (str(suite),))
        )
    assert degraded.exit_code == 1
    degraded_tasks = degraded.summary["suites"][0]["tasks"]
    assert len(degraded_tasks) == 3
    assert [task["status"] for task in degraded_tasks[:2]] == ["graded", "graded"]
    assert degraded_tasks[2]["status"] == "error"
    assert degraded_tasks[2]["reason"] == "server_unavailable"
    results = [
        entry
        for entry in quality.QualityRun.load(degraded.run_dir).entries
        if entry.get("type") == "task_result"
    ]
    assert len(results) == 3
    assert (
        len(
            {
                (entry["suite_id"], entry["task_id"], entry["rep"], entry["side"])
                for entry in results
            }
        )
        == 3
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal integration")
def test_sigterm_then_resume_skips_exact_graded_tuples(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_server(fake_bin_dir)
    tasks = _coding_tasks()
    suite = _write_suite(tmp_path, "resume", "coding", tasks)
    monkeypatch.setenv(
        "LLAMATUNE_FAKE_SRV_SCRIPT",
        str(_write_script(tmp_path, _coding_script(tasks, values=(2, 2)))),
    )
    original = qualserver.ServerHandle.chat
    sent = False

    def interrupt_once(self: qualserver.ServerHandle, *args: Any, **kwargs: Any) -> str:
        nonlocal sent
        if not sent:
            sent = True
            os.kill(os.getpid(), signal.SIGTERM)
        return original(self, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(qualserver.ServerHandle, "chat", interrupt_once)
        interrupted = quality.run_quality(
            _options(tiny_gguf, fake_bin_dir, tmp_path / "root", (str(suite),))
        )
    assert interrupted.exit_code == 4
    before = [
        entry
        for entry in quality.QualityRun.load(interrupted.run_dir).entries
        if entry.get("type") == "task_result"
    ]
    assert len(before) == 1

    resumed = quality.resume_quality(interrupted.run_dir, llama_bin=fake_bin_dir)
    assert resumed.exit_code == 0
    after = [
        entry
        for entry in quality.QualityRun.load(interrupted.run_dir).entries
        if entry.get("type") == "task_result"
    ]
    assert len(after) == 2
    assert len({(e["suite_id"], e["task_id"], e["rep"], e["side"]) for e in after}) == 2


def test_session_lossless_comparison_and_fingerprint_validation(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_server(fake_bin_dir)
    task = _coding_tasks(1)[0]
    suite = _write_suite(tmp_path, "compare", "coding", [task])
    model = inspect_model(tiny_gguf)
    session = tmp_path / "session"
    session.mkdir()
    evaluated = TrialConfig(
        gpu_layers=4,
        moe_cpu_layers=0,
        flash_attn=True,
        ubatch=256,
        batch=512,
        threads=4,
        mmap=True,
        no_kv_offload=False,
        cache_type_k="q8_0",
        cache_type_v="q8_0",
    )
    preferred = dataclasses.replace(evaluated, gpu_layers=7, cache_type_k="f16", cache_type_v="f16")
    (session / "model.json").write_text(json.dumps({"fingerprint": "wrong"}), encoding="utf-8")
    (session / "recommended.json").write_text(
        json.dumps({"confirmed": True, "config": evaluated.to_dict()}), encoding="utf-8"
    )
    (session / "analysis.json").write_text(
        json.dumps(
            {
                "winner": {"confirmed": True, "config": evaluated.to_dict()},
                "lossless_winner": {"config": preferred.to_dict()},
                "feasibility": {"ctx_validated": 2048},
                "context_envelope": [
                    {
                        "ctx": 8192,
                        "status": "failed",
                        "config": evaluated.to_dict(),
                        "fallback_config": preferred.to_dict(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    mismatch = quality.run_quality(
        _options(
            tiny_gguf,
            fake_bin_dir,
            tmp_path / "mismatch",
            (str(suite),),
            config_session=session,
        )
    )
    assert mismatch.exit_code == 2
    assert not (tmp_path / "mismatch" / "quality").exists()

    (session / "model.json").write_text(
        json.dumps({"fingerprint": model.fingerprint}), encoding="utf-8"
    )
    (session / "recommended.json").write_text(
        json.dumps({"config": evaluated.to_dict()}), encoding="utf-8"
    )
    unconfirmed = quality.run_quality(
        _options(
            tiny_gguf,
            fake_bin_dir,
            tmp_path / "unconfirmed",
            (str(suite),),
            config_session=session,
            dry_run=True,
        )
    )
    assert unconfirmed.exit_code == 2
    (session / "recommended.json").write_text(
        json.dumps({"confirmed": True, "config": evaluated.to_dict()}), encoding="utf-8"
    )
    no_compare = quality.run_quality(
        _options(
            tiny_gguf,
            fake_bin_dir,
            tmp_path / "no-compare",
            (str(suite),),
            config_session=session,
            dry_run=True,
        )
    )
    assert no_compare.exit_code == 0
    assert any("validated context 2048" in warning for warning in no_compare.summary["warnings"])
    resolved_no_compare = quality._resolve(
        _options(
            tiny_gguf,
            fake_bin_dir,
            tmp_path / "resolved-no-compare",
            (str(suite),),
            config_session=session,
        )
    )
    assert resolved_no_compare.lossless_config is None
    compare_plan = quality.run_quality(
        _options(
            tiny_gguf,
            fake_bin_dir,
            tmp_path / "compare-plan",
            (str(suite),),
            config_session=session,
            compare_lossless=True,
            dry_run=True,
        )
    )
    assert compare_plan.exit_code == 0
    assert compare_plan.summary["estimated_requests"] == 2

    def side_response(
        run: quality.QualityRun,
        handle: qualserver.ServerHandle,
        suite_spec: Any,
        task_value: dict[str, Any],
        rep: int,
        side: str,
        options: QualityOptions,
    ) -> tuple[str, ...]:
        del run, handle, suite_spec, task_value, rep, options
        return (
            "not python"
            if side == "evaluated"
            else "```python\ndef answer_0():\n    return 2\n```",
        )

    monkeypatch.setattr(quality, "_drive_task", side_response)
    compared = quality.run_quality(
        _options(
            tiny_gguf,
            fake_bin_dir,
            tmp_path / "compared",
            (str(suite),),
            config_session=session,
            compare_lossless=True,
        )
    )
    assert compared.exit_code == 0
    comparison = compared.summary["comparison"]
    assert comparison["lossless_config"]["gpu_layers"] == 7
    assert comparison["suites"][0]["delta_score"] < -0.05
    assert comparison["degradation_warning"] is True
    assert set(comparison["suites"][0]) == {
        "suite_id",
        "evaluated",
        "lossless",
        "delta_score",
    }

    already_lossless = dataclasses.replace(evaluated, cache_type_k="f16", cache_type_v="f16")
    (session / "recommended.json").write_text(
        json.dumps({"confirmed": True, "config": already_lossless.to_dict()}),
        encoding="utf-8",
    )
    resolved_lossless = quality._resolve(
        _options(
            tiny_gguf,
            fake_bin_dir,
            tmp_path / "already-lossless",
            (str(suite),),
            config_session=session,
            compare_lossless=True,
        )
    )
    assert resolved_lossless.lossless_config is None
    assert any("no-op" in warning for warning in resolved_lossless.warnings)

    shutil.rmtree(session)
    resumed_without_source = quality.resume_quality(compared.run_dir, llama_bin=fake_bin_dir)
    assert resumed_without_source.exit_code == 0


def test_ctx_min_skips_needle_and_excludes_it_from_denominator(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
) -> None:
    _install_server(fake_bin_dir)
    suite = _write_suite(
        tmp_path,
        "needle",
        "ifollow",
        [
            {
                "id": "needle",
                "kind": "needle",
                "needle": "CODE-1",
                "question": "What is the code?",
                "haystack_ratio": 0.5,
                "position": 0.5,
                "ctx_min": 4096,
            }
        ],
    )
    outcome = quality.run_quality(
        _options(tiny_gguf, fake_bin_dir, tmp_path / "root", (str(suite),), ctx=1024)
    )

    assert outcome.exit_code == 0
    task = outcome.summary["suites"][0]["tasks"][0]
    assert task["status"] == "skipped"
    assert "ctx_min" in task["reason"]
    metrics = outcome.summary["suites"][0]["metrics"]
    assert metrics["score"] == 0.0
    assert metrics["pass_rate"] == 0.0
    report = (outcome.run_dir / "quality-report.md").read_text()
    assert "ctx_min" in report


def test_server_argv_capability_gating_and_tool_normalization(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
) -> None:
    _install_server(fake_bin_dir)
    resolved = quality._resolve(
        _options(tiny_gguf, fake_bin_dir, tmp_path / "root", ("coding",), dry_run=True)
    )
    config = TrialConfig(
        gpu_layers=7,
        moe_cpu_layers=3,
        flash_attn=True,
        ubatch=128,
        batch=256,
        threads=4,
        mmap=False,
        no_kv_offload=True,
        cache_type_k="q8_0",
        cache_type_v="q4_0",
        threads_batch=2,
        ot_spec="blk.0=CPU",
        tensor_split=(1.0, 2.5),
        split_mode="row",
    )
    all_caps = frozenset({"ncmoe", "fa", "mmp", "nkvo", "ctk", "ctv", "tb", "ot", "ts", "sm"})
    supported = dataclasses.replace(
        resolved, llama=dataclasses.replace(resolved.llama, capabilities=all_caps)
    )
    argv = quality._server_argv(supported, config)
    assert "-ot" in argv and "blk.0=CPU" in argv
    assert "--n-cpu-moe" not in argv
    assert "--no-mmap" in argv and "--no-kv-offload" in argv
    assert "-tb" in argv and "-ts" in argv and "-sm" in argv

    unsupported = dataclasses.replace(
        resolved,
        llama=dataclasses.replace(resolved.llama, capabilities=frozenset()),
    )
    unsupported_argv = quality._server_argv(unsupported, config)
    assert "-ngl" in unsupported_argv and "-ub" in unsupported_argv
    assert all(
        flag not in unsupported_argv
        for flag in (
            "--n-cpu-moe",
            "-fa",
            "--no-mmap",
            "--no-kv-offload",
            "-ctk",
            "-ctv",
            "-tb",
            "-ot",
            "-ts",
            "-sm",
        )
    )
    assert quality._normalized_equal(
        {"count": 2.0, "name": " Oslo ", "nested": [1, " x "]},
        {"count": 2, "name": "Oslo", "nested": [1.0, "x"]},
    )
    assert quality._normalized_subset({"count": 2.0, "extra": True}, {"count": 2})
    assert not quality._normalized_equal({"count": 2}, {"count": "2"})


def test_second_signal_terminates_sandbox_without_stopping_server_reentrantly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped = 0
    terminated = 0

    class Handle:
        def stop(self) -> None:
            nonlocal stopped
            stopped += 1

    def terminate() -> bool:
        nonlocal terminated
        terminated += 1
        return True

    state = quality._SignalState(handle=cast(qualserver.ServerHandle, Handle()))
    monkeypatch.setattr(sandbox, "terminate_active", terminate)
    state.receive(signal.SIGTERM, None)
    with pytest.raises(quality._ForcedInterrupt):
        state.receive(signal.SIGTERM, None)

    assert state.stop_requested is True
    assert state.forced is True
    assert terminated == 1
    assert stopped == 0


def test_perplexity_uses_public_parser_and_is_excluded_from_overall(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
) -> None:
    _install_perplexity(fake_bin_dir)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("deterministic fixture corpus\n", encoding="utf-8")
    outcome = quality.run_quality(
        _options(
            tiny_gguf,
            fake_bin_dir,
            tmp_path / "root",
            ("perplexity",),
            quality_corpus=corpus,
        )
    )

    assert outcome.exit_code == 0
    assert outcome.summary["overall"] == 0.0
    assert outcome.summary["suites"][0]["metrics"] == {"ppl": 12.5}
    command = json.loads(
        (outcome.run_dir / "perplexity" / "side-evaluated" / "command.json").read_text()
    )["argv"]
    assert command[-4:] == ["-c", "4096", "-f", str(corpus)]
    assert "-ngl" not in command


def test_invalid_perplexity_is_a_task_error_without_division(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
) -> None:
    _install_perplexity(fake_bin_dir)
    script = (
        fake_bin_dir / "fake_llama_perplexity.py"
        if sys.platform == "win32"
        else fake_bin_dir / "llama-perplexity"
    )
    script.write_text(
        "#!/usr/bin/env python3\nprint('Final estimate: PPL = 0')\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("fixture\n", encoding="utf-8")
    outcome = quality.run_quality(
        _options(
            tiny_gguf,
            fake_bin_dir,
            tmp_path / "root",
            ("perplexity",),
            quality_corpus=corpus,
        )
    )

    assert outcome.exit_code == 1
    assert outcome.summary["suites"][0]["tasks"][0]["status"] == "error"


def test_dry_run_has_estimates_and_creates_no_run_directory(
    tmp_path: Path,
    fake_bin_dir: Path,
    tiny_gguf: Path,
) -> None:
    _install_server(fake_bin_dir)
    root = tmp_path / "dry-root"
    outcome = quality.run_quality(
        _options(tiny_gguf, fake_bin_dir, root, ("coding",), filters=("clamp",), dry_run=True)
    )

    assert outcome.exit_code == 0
    assert outcome.summary["dry_run"] is True
    assert outcome.summary["exec_enabled"] is False
    assert outcome.summary["task_count"] == 1
    assert outcome.summary["estimated_requests"] == 1
    assert outcome.summary["suites"][0]["tasks"] == 1
    assert "<ephemeral>" in outcome.summary["server_argv"]
    assert not (root / "quality").exists()
