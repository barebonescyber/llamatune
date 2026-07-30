from __future__ import annotations

from typing import Any

import pytest

from llamatune.qualscore import (
    aggregate,
    combine_repetitions,
    extract_call,
    extract_code,
    grade_task,
    replay_agentic,
)
from llamatune.sandbox import ExecVerdict
from llamatune.types import TaskGrade


def _coding(*graders: dict[str, Any], weight: float = 1.0) -> dict[str, Any]:
    return {
        "id": "coding-1",
        "weight": weight,
        "language": "python",
        "graders": list(graders),
    }


def _metadata(weight: float = 1.0, **values: float) -> tuple[dict[str, Any], ...]:
    return (
        {
            "grader": "task_metrics",
            "passed": True,
            "detail": "aggregate inputs",
            "weight": weight,
            **values,
        },
    )


def _grade(
    task_id: str,
    score: float,
    *,
    status: str = "graded",
    reason: str | None = None,
    unstable: bool = False,
    metadata: tuple[dict[str, Any], ...] | None = None,
) -> TaskGrade:
    return TaskGrade(
        task_id=task_id,
        score=score,
        status=status,
        reason=reason,
        unstable=unstable,
        grader_results=metadata or _metadata(),
    )


def test_extract_code_first_matching_language_and_absence() -> None:
    text = """```json
{}
```
```PYTHON
print('first')
```
```python
print('second')
```"""
    assert extract_code(text, " python ") == "print('first')\n"
    assert extract_code(text, "ruby") is None
    assert extract_code("print('bare')", "python") is None


def test_extract_call_normative_fenced_bare_absent_and_multiple_rules() -> None:
    assert extract_call('```json\n{"tool":"lookup","args":{"x":1}}\n```') == {
        "tool": "lookup",
        "args": {"x": 1},
    }
    assert extract_call('prefix {"nested":{"brace":"}"},"final":" done "} suffix') == {
        "nested": {"brace": "}"},
        "final": " done ",
    }
    assert extract_call('ignore {"x":1} then {"tool":"go","args":{}}') == {
        "tool": "go",
        "args": {},
    }
    assert extract_call("no object") is None
    assert extract_call('```json\nnot json\n```\n```json\n{"final":"later"}\n```') is None
    assert extract_call('```json\n{"x":1}\n``` trailing {"final":"fallback"}') is None


def test_coding_every_grader_passes_and_exec_source_is_injected() -> None:
    task = _coding(
        {"type": "code_extract", "language": "python"},
        {"type": "python_parses"},
        {"type": "defines", "name": "add", "kind": "function"},
        {"type": "signature", "name": "add", "params": ["left", "right"]},
        {"type": "contains_regex", "pattern": r"return\s+left"},
        {"type": "absent_regex", "pattern": r"eval\("},
        {"type": "exec_python", "tests": ["assert add(2, 3) == 5"]},
        weight=2.0,
    )
    sources: list[str] = []

    def runner(source: str) -> ExecVerdict:
        sources.append(source)
        return ExecVerdict(True, 0, False, "", "")

    grade = grade_task(
        task,
        ("```python\ndef add(left, right):\n    return left + right\n```",),
        runner,
    )
    assert grade.score == 1.0
    assert grade.status == "graded"
    assert grade.reason is None
    assert len(sources) == 1
    assert "assert add(2, 3) == 5" in sources[0]
    assert all({"grader", "passed", "detail"} <= result.keys() for result in grade.grader_results)
    assert grade.grader_results[-1]["weight"] == 2.0


def test_coding_conjunctive_failure_partial_math_and_parse_details() -> None:
    partial = _coding(
        {"type": "code_extract", "language": "python"},
        {"type": "contains_regex", "pattern": "present", "partial": True, "weight": 3},
        {"type": "contains_regex", "pattern": "absent", "partial": True, "weight": 1},
    )
    grade = grade_task(partial, ("```python\npresent = True\n```",), None)
    assert grade.score == pytest.approx(0.75)

    gated = _coding(
        {"type": "code_extract", "language": "python"},
        {"type": "python_parses"},
        {"type": "defines", "name": "Missing", "kind": "class"},
        {"type": "signature", "name": "missing", "params": []},
        {"type": "contains_regex", "pattern": "["},
    )
    failed = grade_task(gated, ("```python\ndef broken(:\n```",), None)
    assert failed.score == 0.0
    details = " ".join(str(result["detail"]) for result in failed.grader_results)
    assert "syntax error" in details
    assert "invalid regex" in details

    class_task = _coding(
        {"type": "code_extract", "language": "python"},
        {"type": "defines", "name": "Example", "kind": "class"},
    )
    assert grade_task(class_task, ("```python\nclass Example:\n    pass\n```",), None).score == 1.0


def test_exec_python_failure_is_a_conjunctive_gate() -> None:
    task = _coding(
        {"type": "code_extract", "language": "python"},
        {"type": "exec_python", "tests": ["assert False"]},
    )

    def runner(_source: str) -> ExecVerdict:
        return ExecVerdict(False, 1, False, "", "AssertionError")

    grade = grade_task(task, ("```python\nx = 1\n```",), runner)
    assert grade.score == 0.0
    assert "exit=1" in grade.grader_results[1]["detail"]


def test_exec_python_is_skipped_without_runner_and_never_called() -> None:
    task = _coding(
        {"type": "code_extract", "language": "python"},
        {"type": "exec_python", "tests": ["assert True"]},
    )
    grade = grade_task(task, ("```python\nx = 1\n```",), None)
    assert grade.score == 1.0
    assert grade.reason == "exec_skipped"
    skipped = next(result for result in grade.grader_results if result["grader"] == "exec_python")
    assert skipped == {
        "grader": "exec_python",
        "passed": False,
        "detail": "execution disabled",
        "skipped": True,
    }


def test_absent_regex_fails_without_extracted_code() -> None:
    task = _coding({"type": "absent_regex", "pattern": "forbidden"})
    grade = grade_task(task, ("no fenced code",), None)
    assert grade.score == 0.0
    assert grade.grader_results[0]["detail"] == "no extracted code"


def _tool_task() -> dict[str, Any]:
    return {
        "id": "tools-1",
        "tools": [
            {"name": "lookup", "description": "look up", "params": {}},
            {"name": "save", "description": "save", "params": {}},
        ],
        "turns": [
            {"user": "find"},
            {"expect": {"tool": "lookup", "args_equal": {"id": 2}}, "tool_result": {}},
            {"user": "save"},
            {
                "expect": {"tool": "save", "args_subset": {"name": " Ada "}},
                "tool_result": {},
            },
            {"user": "finish"},
            {"expect_final": {"contains": ["done"], "regex": r"done\s+2"}},
        ],
    }


def test_tooluse_multi_turn_equality_subset_normalization_and_final() -> None:
    responses = (
        '```json\n{"tool":"lookup","args":{"id":2.0}}\n```',
        '{"tool":"save","args":{"name":"Ada","extra":true}}',
        '```json\n{"final":"done 2"}\n```',
    )
    grade = grade_task(_tool_task(), responses, None)
    assert grade.score == 1.0
    metrics = grade.grader_results[-1]
    assert metrics["valid_call_rate"] == 1.0
    assert metrics["correct_tool_rate"] == 1.0
    assert metrics["args_accuracy"] == 1.0


def test_tooluse_hallucinated_tool_malformed_call_and_bad_final_are_explainable() -> None:
    responses = (
        '{"tool":"invented","args":{"id":2}}',
        "not json",
        '{"final":"wrong"}',
    )
    grade = grade_task(_tool_task(), responses, None)
    assert grade.score == 0.0
    metrics = grade.grader_results[-1]
    assert metrics["valid_call_rate"] == 0.5
    assert metrics["correct_tool_rate"] == 0.0
    assert metrics["args_accuracy"] == 0.0
    assert any("actual='invented'" in result["detail"] for result in grade.grader_results)


def _agentic_task() -> dict[str, Any]:
    return {
        "id": "agent-1",
        "environment": {
            "initial_state": {"stock": 0, "events": [], "owner": "none"},
            "tools": [
                {
                    "name": "restock",
                    "params": {},
                    "effects": [
                        {"incr": ["stock", "$arg.amount"]},
                        {"append": ["events", "$arg.amount"]},
                        {"set": ["owner", "$arg.owner"]},
                    ],
                    "result": "stock={state.stock}; owner={args.owner}",
                }
            ],
            "goal": {"state_subset": {"stock": 3, "owner": "Ada"}},
            "max_steps": 3,
            "optimal_steps": 1,
        },
        "prompt": "restock",
    }


def test_agentic_incremental_replay_effects_reply_goal_and_final_termination() -> None:
    first = replay_agentic(
        _agentic_task(),
        ('{"tool":"restock","args":{"amount":3,"owner":"Ada"}}',),
    )
    assert first == {
        "state": {"stock": 3, "events": [3], "owner": "Ada"},
        "tool_reply": "stock=3; owner=Ada",
        "done": True,
        "completed": True,
        "steps_used": 1,
        "invalid_calls": 0,
        "total_calls": 1,
    }
    final = replay_agentic(_agentic_task(), ('{"final":"cannot complete"}',))
    assert final["done"] is True
    assert final["completed"] is False
    assert final["steps_used"] == 0
    grade = grade_task(
        _agentic_task(),
        ('{"tool":"restock","args":{"amount":3,"owner":"Ada"}}',),
        None,
    )
    assert grade.score == 1.0
    assert grade.grader_results[-1]["efficiency"] == 1.0


def test_agentic_invalid_unknown_and_step_cap_are_counted() -> None:
    replay = replay_agentic(
        _agentic_task(),
        ("bad", '{"tool":"unknown","args":{}}', "still bad", "ignored"),
    )
    assert replay["done"] is True
    assert replay["completed"] is False
    assert replay["steps_used"] == 3
    assert replay["invalid_calls"] == 3
    assert replay["total_calls"] == 3
    assert replay["tool_reply"] == "error: invalid tool call"

    grade = grade_task(_agentic_task(), ("bad", '{"final":"stop"}'), None)
    assert grade.score == 0.0
    assert grade.grader_results[-1]["invalid_call_rate"] == 1.0


def test_agentic_dollar_arg_single_value_and_invalid_effect_are_pure() -> None:
    task = _agentic_task()
    task["environment"]["tools"][0]["effects"] = [{"set": ["owner", "$arg"]}]
    replay = replay_agentic(task, ('{"tool":"restock","args":{"owner":"Ada"}}',))
    assert replay["state"]["owner"] == "Ada"
    assert task["environment"]["initial_state"]["owner"] == "none"

    task["environment"]["tools"][0]["effects"] = [{"append": ["owner", "bad"]}]
    failed = replay_agentic(task, ('{"tool":"restock","args":{}}',))
    assert failed["invalid_calls"] == 1
    assert str(failed["tool_reply"]).startswith("error: invalid effect")


def test_ifollow_all_constraints_average_and_fail_independently() -> None:
    task = {
        "id": "format-1",
        "constraints": [
            {"type": "json_object", "schema_subset": {"ok": True}},
            {"type": "max_words", "value": 3},
            {"type": "max_chars", "value": 20},
            {"type": "must_include", "values": ["OK"]},
            {"type": "must_not_include", "values": ["forbidden"]},
            {"type": "regex_full", "pattern": r'\{"ok":true\}'},
            {"type": "language", "value": "english"},
            {"type": "starts_with", "value": "{"},
            {"type": "ends_with", "value": "}"},
        ],
    }
    grade = grade_task(task, ('{"ok":true}',), None)
    assert grade.score == 1.0
    assert all(result["passed"] for result in grade.grader_results[:-1])

    failed = grade_task(task, (" forbidden text ",), None)
    assert 0.0 < failed.score < 1.0
    assert any(not result["passed"] and result["detail"] for result in failed.grader_results[:-1])


def test_ifollow_needle_and_unsupported_language() -> None:
    needle = {
        "id": "needle-1",
        "kind": "needle",
        "needle": "The payload is CERULEAN",
    }
    assert grade_task(needle, ("I found the payload is cerulean.",), None).score == 1.0
    assert grade_task(needle, ("missing",), None).score == 0.0
    unsupported = {
        "id": "language-1",
        "constraints": [{"type": "language", "value": "klingon"}],
    }
    result = grade_task(unsupported, ("hello",), None)
    assert result.score == 0.0
    assert "unsupported" in result.grader_results[0]["detail"]

    bounded = {
        "id": "bounds",
        "constraints": [
            {"type": "max_words", "value": 1},
            {"type": "max_chars", "value": 2},
            {"type": "language", "value": "english"},
        ],
    }
    bounded_grade = grade_task(bounded, ("ééé ééé",), None)
    assert bounded_grade.score == 0.0
    assert all(not result["passed"] for result in bounded_grade.grader_results[:-1])


def test_grade_task_errors_are_explicit() -> None:
    assert grade_task({"id": "empty", "constraints": []}, (), None).reason == "no responses"
    assert grade_task({"id": "unknown"}, ("response",), None).reason == "unknown task shape"


def test_combine_repetitions_worst_unstable_annotations_and_validation() -> None:
    combined = combine_repetitions((_grade("one", 1.0), _grade("one", 0.25)))
    assert combined.score == 0.25
    assert combined.unstable is True
    assert {result["rep"] for result in combined.grader_results} == {1, 2}

    error = _grade("one", 0.0, status="error", reason="request failed")
    assert combine_repetitions((_grade("one", 1.0), error)).status == "error"
    skipped = _grade("one", 0.0, status="skipped", reason="ctx below minimum")
    skipped_result = combine_repetitions((_grade("one", 1.0), skipped))
    assert skipped_result.status == "skipped"
    assert skipped_result.reason == "ctx below minimum"
    with pytest.raises(ValueError, match="at least one"):
        combine_repetitions(())
    with pytest.raises(ValueError, match="ids"):
        combine_repetitions((_grade("one", 1.0), _grade("two", 1.0)))


def test_aggregate_weighting_skip_exclusion_error_inclusion_and_short_shape() -> None:
    grades = (
        _grade("high", 1.0, metadata=_metadata(3.0)),
        _grade("low", 0.0, metadata=_metadata(1.0)),
        _grade("skip", 0.0, status="skipped", metadata=_metadata(100.0)),
        _grade("error", 0.0, status="error", metadata=_metadata(0.0)),
    )
    metrics = aggregate("coding", grades, exec_enabled=False)
    assert metrics == {"score": 0.75, "pass_rate": 0.5, "exec_enabled": 0.0}
    assert not any(key.startswith("quality.") for key in metrics)


def test_aggregate_all_kind_specific_metrics() -> None:
    tool = _grade(
        "tool",
        0.5,
        metadata=_metadata(
            valid_call_rate=0.5,
            correct_tool_rate=0.25,
            args_accuracy=0.75,
            valid_calls=2.0,
            correct_tools=1.0,
            correct_args=3.0,
            tool_turns=4.0,
        ),
    )
    tool_metrics = aggregate("tooluse", (tool,), exec_enabled=False)
    assert tool_metrics["valid_call_rate"] == 0.5
    assert tool_metrics["correct_tool_rate"] == 0.25
    assert tool_metrics["args_accuracy"] == 0.75

    completed = _grade(
        "done",
        0.85,
        metadata=_metadata(
            completed=1.0,
            efficiency=0.5,
            invalid_call_rate=0.25,
            invalid_calls=1.0,
            total_calls=4.0,
        ),
    )
    incomplete = _grade(
        "open",
        0.0,
        metadata=_metadata(
            completed=0.0,
            efficiency=0.0,
            invalid_call_rate=0.75,
            invalid_calls=3.0,
            total_calls=4.0,
        ),
    )
    agent = aggregate("agentic", (completed, incomplete), exec_enabled=False)
    assert agent["completion_rate"] == 0.5
    assert agent["efficiency"] == 0.5
    assert agent["invalid_call_rate"] == 0.5

    constraint = _grade(
        "format",
        0.5,
        metadata=_metadata(is_needle=0.0, format_rate=0.5),
    )
    needle = _grade(
        "needle",
        1.0,
        metadata=_metadata(is_needle=1.0, needle_accuracy=1.0),
    )
    following = aggregate("ifollow", (constraint, needle), exec_enabled=False)
    assert following["format_rate"] == 0.5
    assert following["needle_accuracy"] == 1.0
    with pytest.raises(ValueError, match="unsupported"):
        aggregate("perplexity", (), exec_enabled=False)


def test_tooluse_aggregation_means_over_turns_not_tasks() -> None:
    one_turn = _grade(
        "one",
        1.0,
        metadata=_metadata(
            10.0,
            valid_calls=1.0,
            correct_tools=1.0,
            correct_args=1.0,
            tool_turns=1.0,
        ),
    )
    two_turn = _grade(
        "two",
        0.0,
        metadata=_metadata(
            valid_calls=0.0,
            correct_tools=0.0,
            correct_args=0.0,
            tool_turns=2.0,
        ),
    )
    metrics = aggregate("tooluse", (one_turn, two_turn), exec_enabled=False)
    assert metrics["valid_call_rate"] == pytest.approx(1 / 3)
    assert metrics["correct_tool_rate"] == pytest.approx(1 / 3)
    assert metrics["args_accuracy"] == pytest.approx(1 / 3)
