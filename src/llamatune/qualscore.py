"""Pure extraction, grading, replay, and aggregation for quality suites."""

from __future__ import annotations

import ast
import copy
import json
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from llamatune.types import TaskGrade

if TYPE_CHECKING:
    from llamatune.sandbox import ExecVerdict

_FENCE_RE = re.compile(r"```[ \t]*([^\r\n`]*)[ \t]*\r?\n(.*?)```", re.DOTALL)


def extract_code(text: str, language: str) -> str | None:
    """Return the first fenced block whose language exactly matches."""
    requested = language.strip().casefold()
    for match in _FENCE_RE.finditer(text):
        if match.group(1).strip().casefold() == requested:
            return match.group(2)
    return None


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _balanced_objects(text: str) -> list[str]:
    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if start is None:
            if character == "{":
                start = index
                depth = 1
                in_string = False
                escaped = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                objects.append(text[start : index + 1])
                start = None
    return objects


def extract_call(text: str) -> dict[str, Any] | None:
    """Extract the normative first fenced JSON or first valid bare call."""
    for match in _FENCE_RE.finditer(text):
        if match.group(1).strip().casefold() == "json":
            value = _json_object(match.group(2))
            if value is None or not ({"tool", "final"} & value.keys()):
                return None
            return value
    for candidate in _balanced_objects(text):
        value = _json_object(candidate)
        if value is not None and {"tool", "final"} & value.keys():
            return value
    return None


def _result(grader: str, passed: bool, detail: str, **metrics: Any) -> dict[str, Any]:
    return {"grader": grader, "passed": passed, "detail": detail, **metrics}


def _task_metadata(task: dict[str, Any], passed: bool, **metrics: float) -> dict[str, Any]:
    weight = float(task.get("weight", 1.0))
    return _result("task_metrics", passed, "aggregate inputs", weight=weight, **metrics)


def _definition(tree: ast.AST, name: str, kind: str) -> ast.AST | None:
    expected = (ast.FunctionDef, ast.AsyncFunctionDef) if kind == "function" else (ast.ClassDef,)
    return next(
        (node for node in ast.walk(tree) if isinstance(node, expected) and node.name == name),
        None,
    )


def _params(node: ast.AST) -> list[str] | None:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    args = node.args
    names = [argument.arg for argument in (*args.posonlyargs, *args.args)]
    if args.vararg is not None:
        names.append(f"*{args.vararg.arg}")
    names.extend(argument.arg for argument in args.kwonlyargs)
    if args.kwarg is not None:
        names.append(f"**{args.kwarg.arg}")
    return names


def _grade_coding(
    task: dict[str, Any],
    response: str,
    exec_runner: Callable[[str], ExecVerdict] | None,
) -> tuple[float, list[dict[str, Any]], str | None]:
    graders = task.get("graders", [])
    language = str(task.get("language", "python"))
    extract_language = next(
        (
            str(grader.get("language", language))
            for grader in graders
            if grader.get("type") == "code_extract"
        ),
        language,
    )
    code = extract_code(response, extract_language)
    tree: ast.AST | None = None
    parse_error: str | None = None
    if code is not None:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            parse_error = f"syntax error at line {exc.lineno}: {exc.msg}"

    results: list[dict[str, Any]] = []
    gate_passed = True
    partial_passed = 0.0
    partial_total = 0.0
    exec_skipped = False
    for grader in graders:
        grader_type = str(grader.get("type", ""))
        passed = False
        detail = ""
        if grader_type == "code_extract":
            grader_language = str(grader.get("language", language))
            extracted = extract_code(response, grader_language)
            passed = extracted is not None
            detail = f"{grader_language} fenced block {'found' if passed else 'missing'}"
            if passed and grader_language.casefold() == language.casefold():
                code = extracted
        elif grader_type == "python_parses":
            passed = tree is not None
            detail = "AST parse succeeded" if passed else (parse_error or "no extracted code")
        elif grader_type == "defines":
            name = str(grader.get("name", ""))
            kind = str(grader.get("kind", "function"))
            passed = tree is not None and _definition(tree, name, kind) is not None
            detail = f"{kind} {name!r} {'defined' if passed else 'not defined'}"
        elif grader_type == "signature":
            name = str(grader.get("name", ""))
            expected = [str(value) for value in grader.get("params", [])]
            node = _definition(tree, name, "function") if tree is not None else None
            actual = _params(node) if node is not None else None
            passed = actual == expected
            detail = f"parameters expected={expected!r}, actual={actual!r}"
        elif grader_type in {"contains_regex", "absent_regex"}:
            pattern = str(grader.get("pattern", ""))
            if code is None:
                detail = "no extracted code"
            else:
                try:
                    found = re.search(pattern, code) is not None
                    passed = found if grader_type == "contains_regex" else not found
                    detail = f"pattern {pattern!r} {'present' if found else 'absent'}"
                except re.error as exc:
                    detail = f"invalid regex: {exc}"
        elif grader_type == "exec_python":
            if exec_runner is None:
                exec_skipped = True
                results.append(_result(grader_type, False, "execution disabled", skipped=True))
                continue
            if code is None:
                detail = "no extracted code"
            else:
                tests = "\n".join(str(value) for value in grader.get("tests", []))
                verdict = exec_runner(f"{code.rstrip()}\n\n{tests}\n")
                passed = verdict.passed
                detail = (
                    f"exit={verdict.exit_code}, timed_out={verdict.timed_out}, "
                    f"stderr={verdict.stderr_tail!r}"
                )
        else:
            detail = f"unknown grader {grader_type!r}"

        results.append(_result(grader_type, passed, detail))
        if grader_type == "code_extract" and not passed:
            gate_passed = False
        elif bool(grader.get("partial", False)):
            weight = float(grader.get("weight", 1.0))
            partial_total += weight
            if passed:
                partial_passed += weight
        elif not passed:
            gate_passed = False

    score = 0.0 if not gate_passed else partial_passed / partial_total if partial_total else 1.0
    results.append(_task_metadata(task, score >= 0.999, exec_skipped=float(exec_skipped)))
    return score, results, "exec_skipped" if exec_skipped else None


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return value


def _subset(expected: Any, actual: Any) -> bool:
    expected = _normalize(expected)
    actual = _normalize(actual)
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _subset(value, actual[key]) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(expected) == len(actual)
            and all(_subset(left, right) for left, right in zip(expected, actual, strict=True))
        )
    return bool(expected == actual)


def _final_text(response: str) -> tuple[str | None, bool]:
    call = extract_call(response)
    if call is None or "final" not in call:
        return None, False
    value = call["final"]
    return (value if isinstance(value, str) else json.dumps(value, sort_keys=True)), True


def _grade_tooluse(
    task: dict[str, Any], responses: tuple[str, ...]
) -> tuple[float, list[dict[str, Any]]]:
    expectations = [turn for turn in task.get("turns", []) if "expect" in turn]
    final_expectations = [turn for turn in task.get("turns", []) if "expect_final" in turn]
    tools = {str(tool.get("name")) for tool in task.get("tools", [])}
    results: list[dict[str, Any]] = []
    valid_values: list[float] = []
    tool_values: list[float] = []
    args_values: list[float] = []
    turn_scores: list[float] = []
    for index, turn in enumerate(expectations):
        response = responses[index] if index < len(responses) else ""
        call = extract_call(response)
        valid = call is not None and "tool" in call and isinstance(call.get("args"), dict)
        expected = turn["expect"]
        called_tool = call.get("tool") if call is not None else None
        tool_correct = valid and called_tool in tools and called_tool == expected.get("tool")
        actual_args = call.get("args") if valid and call is not None else None
        if "args_equal" in expected:
            args_correct = tool_correct and _normalize(actual_args) == _normalize(
                expected["args_equal"]
            )
        else:
            args_correct = tool_correct and _subset(expected.get("args_subset", {}), actual_args)
        valid_values.append(float(valid))
        tool_values.append(float(tool_correct))
        args_values.append(float(args_correct))
        turn_scores.append((float(valid) + float(tool_correct) + float(args_correct)) / 3.0)
        results.extend(
            (
                _result("valid", valid, f"turn {index + 1}: call extraction"),
                _result(
                    "tool_correct",
                    tool_correct,
                    f"turn {index + 1}: expected={expected.get('tool')!r}, actual={called_tool!r}",
                ),
                _result(
                    "args_correct",
                    args_correct,
                    f"turn {index + 1}: argument comparison",
                ),
            )
        )

    final_index = len(expectations)
    final_response = responses[final_index] if final_index < len(responses) else ""
    final_text, final_valid = _final_text(final_response)
    final_rule = final_expectations[0]["expect_final"] if final_expectations else {}
    final_passed = final_valid and final_text is not None
    if final_text is not None and final_passed and "contains" in final_rule:
        folded = final_text.casefold()
        final_passed = all(str(value).casefold() in folded for value in final_rule["contains"])
    if final_text is not None and final_passed and "regex" in final_rule:
        try:
            final_passed = re.search(str(final_rule["regex"]), final_text) is not None
        except re.error:
            final_passed = False
    results.append(_result("final_answer", final_passed, "final-answer expectation"))
    turn_mean = sum(turn_scores) / len(turn_scores) if turn_scores else 1.0
    score = turn_mean * float(final_passed)
    metrics = {
        "valid_call_rate": sum(valid_values) / len(valid_values) if valid_values else 0.0,
        "correct_tool_rate": sum(tool_values) / len(tool_values) if tool_values else 0.0,
        "args_accuracy": sum(args_values) / len(args_values) if args_values else 0.0,
        "valid_calls": sum(valid_values),
        "correct_tools": sum(tool_values),
        "correct_args": sum(args_values),
        "tool_turns": float(len(turn_scores)),
    }
    results.append(_task_metadata(task, score >= 0.999, **metrics))
    return score, results


def _path_get(root: Any, path: str) -> Any:
    value = root
    for component in path.split(".") if path else ():
        if not isinstance(value, dict) or component not in value:
            raise KeyError(path)
        value = value[component]
    return value


def _path_parent(root: dict[str, Any], path: str) -> tuple[dict[str, Any], str]:
    components = path.split(".")
    if not components or any(not component for component in components):
        raise ValueError(f"invalid state path: {path!r}")
    parent = root
    for component in components[:-1]:
        value = parent.setdefault(component, {})
        if not isinstance(value, dict):
            raise ValueError(f"state path is not an object: {path!r}")
        parent = value
    return parent, components[-1]


def _argument_value(value: Any, args: dict[str, Any]) -> Any:
    if value == "$arg":
        return next(iter(args.values())) if len(args) == 1 else copy.deepcopy(args)
    if isinstance(value, str) and value.startswith("$arg."):
        return copy.deepcopy(_path_get(args, value[5:]))
    return copy.deepcopy(value)


def _apply_effect(state: dict[str, Any], effect: dict[str, Any], args: dict[str, Any]) -> None:
    operation, raw = next(iter(effect.items()))
    path, source = raw
    parent, leaf = _path_parent(state, str(path))
    value = _argument_value(source, args)
    if operation == "set":
        parent[leaf] = value
    elif operation == "append":
        target = parent.setdefault(leaf, [])
        if not isinstance(target, list):
            raise ValueError(f"append target is not a list: {path!r}")
        target.append(value)
    elif operation == "incr":
        current = parent.get(leaf, 0)
        if (
            not isinstance(current, (int, float))
            or isinstance(current, bool)
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            raise ValueError(f"increment requires numeric values: {path!r}")
        parent[leaf] = current + value
    else:
        raise ValueError(f"unknown effect: {operation!r}")


_TEMPLATE_RE = re.compile(r"\{(state|args)\.([^{}]+)\}")


def _render_result(template: str, state: dict[str, Any], args: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        root = state if match.group(1) == "state" else args
        try:
            value = _path_get(root, match.group(2))
        except KeyError:
            return match.group(0)
        return value if isinstance(value, str) else json.dumps(value, sort_keys=True)

    return _TEMPLATE_RE.sub(replace, template)


def replay_agentic(task: dict[str, Any], responses: tuple[str, ...]) -> dict[str, Any]:
    """Purely replay an agentic episode through the supplied response prefix."""
    environment = task.get("environment", {})
    state = copy.deepcopy(environment.get("initial_state", {}))
    tools = {str(tool.get("name")): tool for tool in environment.get("tools", [])}
    goal = environment.get("goal", {}).get("state_subset", {})
    max_steps = int(environment.get("max_steps", 0))
    completed = _subset(goal, state)
    done = completed
    tool_reply: str | None = None
    steps_used = 0
    invalid_calls = 0
    total_calls = 0

    for response in responses:
        if done:
            break
        call = extract_call(response)
        if call is not None and "final" in call:
            done = True
            tool_reply = None
            break
        total_calls += 1
        steps_used += 1
        if (
            call is None
            or not isinstance(call.get("tool"), str)
            or not isinstance(call.get("args"), dict)
        ):
            invalid_calls += 1
            tool_reply = "error: invalid tool call"
        elif call["tool"] not in tools:
            invalid_calls += 1
            tool_reply = f"error: unknown tool {call['tool']!r}"
        else:
            tool = tools[call["tool"]]
            args = call["args"]
            try:
                for effect in tool.get("effects", []):
                    _apply_effect(state, effect, args)
                tool_reply = _render_result(str(tool.get("result", "")), state, args)
            except (KeyError, TypeError, ValueError) as exc:
                invalid_calls += 1
                tool_reply = f"error: invalid effect: {exc}"
        completed = _subset(goal, state)
        done = completed or steps_used >= max_steps

    return {
        "state": state,
        "tool_reply": tool_reply,
        "done": done,
        "completed": completed,
        "steps_used": steps_used,
        "invalid_calls": invalid_calls,
        "total_calls": total_calls,
    }


def _grade_agentic(
    task: dict[str, Any], responses: tuple[str, ...]
) -> tuple[float, list[dict[str, Any]]]:
    replay = replay_agentic(task, responses)
    completed = bool(replay["completed"])
    steps = int(replay["steps_used"])
    optimal = int(task.get("environment", {}).get("optimal_steps", 1))
    efficiency = min(1.0, optimal / steps) if completed and steps else float(completed)
    total_calls = int(replay["total_calls"])
    invalid_rate = int(replay["invalid_calls"]) / total_calls if total_calls else 0.0
    score = float(completed) * (0.7 + 0.3 * efficiency)
    results = [
        _result(
            "completed", completed, "goal state subset satisfied" if completed else "goal unmet"
        ),
        _result("efficiency", completed, f"optimal={optimal}, used={steps}", value=efficiency),
        _result(
            "invalid_rate",
            invalid_rate == 0.0,
            f"invalid={replay['invalid_calls']}, calls={total_calls}",
            value=invalid_rate,
        ),
        _task_metadata(
            task,
            score >= 0.999,
            completed=float(completed),
            efficiency=efficiency,
            invalid_call_rate=invalid_rate,
            invalid_calls=float(replay["invalid_calls"]),
            total_calls=float(total_calls),
        ),
    ]
    return score, results


def _constraint(constraint: dict[str, Any], response: str) -> tuple[bool, str]:
    kind = str(constraint.get("type", ""))
    if kind == "json_object":
        value = _json_object(response.strip())
        subset = constraint.get("schema_subset")
        passed = value is not None and (subset is None or _subset(subset, value))
        return passed, "JSON object and schema subset" if subset is not None else "JSON object"
    if kind == "max_words":
        count = len(re.findall(r"\S+", response))
        limit = int(constraint.get("value", 0))
        return count <= limit, f"words={count}, max={limit}"
    if kind == "max_chars":
        limit = int(constraint.get("value", 0))
        return len(response) <= limit, f"chars={len(response)}, max={limit}"
    if kind in {"must_include", "must_not_include"}:
        folded = response.casefold()
        values = [str(value) for value in constraint.get("values", [])]
        present = [value.casefold() in folded for value in values]
        passed = all(present) if kind == "must_include" else not any(present)
        return passed, f"case-insensitive literals={values!r}"
    if kind == "regex_full":
        pattern = str(constraint.get("pattern", ""))
        try:
            return re.fullmatch(pattern, response, re.DOTALL) is not None, f"full regex={pattern!r}"
        except re.error as exc:
            return False, f"invalid regex: {exc}"
    if kind == "language":
        expected = str(constraint.get("value", "")).casefold()
        characters = [character for character in response if not character.isspace()]
        ratio = (
            sum(ord(character) < 128 for character in characters) / len(characters)
            if characters
            else 1.0
        )
        if expected in {"ascii", "english", "en"}:
            return ratio >= 0.9, f"ASCII ratio={ratio:.3f}, threshold=0.900"
        return False, f"unsupported language heuristic {expected!r}"
    if kind == "starts_with":
        prefix = str(constraint.get("value", ""))
        return response.startswith(prefix), f"starts with {prefix!r}"
    if kind == "ends_with":
        suffix = str(constraint.get("value", ""))
        return response.endswith(suffix), f"ends with {suffix!r}"
    return False, f"unknown constraint {kind!r}"


def _grade_ifollow(task: dict[str, Any], response: str) -> tuple[float, list[dict[str, Any]]]:
    if task.get("kind") == "needle":
        needle = str(task.get("needle", ""))
        passed = needle.casefold() in response.casefold()
        results = [
            _result("needle", passed, f"needle payload {needle!r}"),
            _task_metadata(task, passed, is_needle=1.0, needle_accuracy=float(passed)),
        ]
        return float(passed), results
    constraint_results: list[dict[str, Any]] = []
    passes: list[float] = []
    for constraint in task.get("constraints", []):
        passed, detail = _constraint(constraint, response)
        passes.append(float(passed))
        constraint_results.append(_result(str(constraint.get("type", "")), passed, detail))
    score = sum(passes) / len(passes) if passes else 0.0
    constraint_results.append(
        _task_metadata(task, score >= 0.999, is_needle=0.0, format_rate=score)
    )
    return score, constraint_results


def grade_task(
    task: dict[str, Any],
    responses: tuple[str, ...],
    exec_runner: Callable[[str], ExecVerdict] | None,
) -> TaskGrade:
    """Grade one task repetition or one multi-turn episode."""
    task_id = str(task.get("id", "unknown"))
    if not responses:
        return TaskGrade(
            task_id=task_id,
            score=0.0,
            status="error",
            reason="no responses",
            unstable=False,
            grader_results=(),
        )
    reason: str | None = None
    if "graders" in task:
        score, results, reason = _grade_coding(task, responses[0], exec_runner)
    elif "turns" in task and "tools" in task:
        score, results = _grade_tooluse(task, responses)
    elif "environment" in task:
        score, results = _grade_agentic(task, responses)
    elif "constraints" in task or task.get("kind") == "needle":
        score, results = _grade_ifollow(task, responses[0])
    else:
        return TaskGrade(
            task_id=task_id,
            score=0.0,
            status="error",
            reason="unknown task shape",
            unstable=False,
            grader_results=(),
        )
    return TaskGrade(
        task_id=task_id,
        score=max(0.0, min(1.0, score)),
        status="graded",
        reason=reason,
        unstable=False,
        grader_results=tuple(results),
    )


def combine_repetitions(grades: tuple[TaskGrade, ...]) -> TaskGrade:
    """Combine separately graded repetitions using conservative worst-of-reps."""
    if not grades:
        raise ValueError("at least one repetition grade is required")
    task_id = grades[0].task_id
    if any(grade.task_id != task_id for grade in grades):
        raise ValueError("repetition task ids must match")
    worst = min(grades, key=lambda grade: grade.score)
    if any(grade.status == "error" for grade in grades):
        status = "error"
        reason = next(grade.reason for grade in grades if grade.status == "error")
    elif any(grade.status == "skipped" for grade in grades):
        status = "skipped"
        reason = next(grade.reason for grade in grades if grade.status == "skipped")
    else:
        status = worst.status
        reason = worst.reason
    annotated = tuple(
        {**result, "rep": index}
        for index, grade in enumerate(grades, start=1)
        for result in grade.grader_results
    )
    unstable = (
        any(grade.unstable for grade in grades)
        or len({grade.score for grade in grades}) > 1
        or len({grade.status for grade in grades}) > 1
    )
    return TaskGrade(
        task_id=task_id,
        score=worst.score,
        status=status,
        reason=reason,
        unstable=unstable,
        grader_results=annotated,
    )


def _metadata(grade: TaskGrade) -> dict[str, Any]:
    return next(
        (
            result
            for result in reversed(grade.grader_results)
            if result.get("grader") == "task_metrics"
        ),
        {"weight": 1.0},
    )


def _mean_metric(grades: list[TaskGrade], key: str) -> float:
    return (
        sum(float(_metadata(grade).get(key, 0.0)) for grade in grades) / len(grades)
        if grades
        else 0.0
    )


def aggregate(
    kind: str,
    grades: tuple[TaskGrade, ...],
    *,
    exec_enabled: bool,
) -> dict[str, float]:
    """Aggregate one final grade per task into short quality.json metrics."""
    if kind not in {"coding", "tooluse", "agentic", "ifollow"}:
        raise ValueError(f"unsupported aggregate kind: {kind}")
    included = [grade for grade in grades if grade.status != "skipped"]
    graded = [grade for grade in included if grade.status == "graded"]
    weights = [float(_metadata(grade).get("weight", 1.0)) for grade in included]
    denominator = sum(weights)
    score = (
        sum(grade.score * weight for grade, weight in zip(included, weights, strict=True))
        / denominator
        if denominator
        else 0.0
    )
    metrics = {
        "score": score,
        "pass_rate": (
            sum(grade.score >= 0.999 for grade in graded) / len(graded) if graded else 0.0
        ),
    }
    if kind == "coding":
        metrics["exec_enabled"] = float(exec_enabled)
    elif kind == "tooluse":
        denominator = sum(float(_metadata(grade).get("tool_turns", 0.0)) for grade in included)
        for rate, numerator in (
            ("valid_call_rate", "valid_calls"),
            ("correct_tool_rate", "correct_tools"),
            ("args_accuracy", "correct_args"),
        ):
            total = sum(float(_metadata(grade).get(numerator, 0.0)) for grade in included)
            metrics[rate] = total / denominator if denominator else 0.0
    elif kind == "agentic":
        metrics["completion_rate"] = _mean_metric(included, "completed")
        completed = [grade for grade in included if _metadata(grade).get("completed") == 1.0]
        metrics["efficiency"] = _mean_metric(completed, "efficiency")
        total_calls = sum(float(_metadata(grade).get("total_calls", 0.0)) for grade in included)
        invalid_calls = sum(float(_metadata(grade).get("invalid_calls", 0.0)) for grade in included)
        metrics["invalid_call_rate"] = invalid_calls / total_calls if total_calls else 0.0
    else:
        constraints = [grade for grade in included if _metadata(grade).get("is_needle") == 0.0]
        needles = [grade for grade in included if _metadata(grade).get("is_needle") == 1.0]
        metrics["format_rate"] = _mean_metric(constraints, "format_rate")
        metrics["needle_accuracy"] = _mean_metric(needles, "needle_accuracy")
    return metrics
