"""Quality-suite schema validation, bundled loading, and haystack generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

_BUNDLED = ("coding", "tooluse", "agentic", "ifollow", "perplexity")
_GRADERS = {
    "code_extract",
    "python_parses",
    "defines",
    "signature",
    "contains_regex",
    "absent_regex",
    "exec_python",
}
_CONSTRAINTS = {
    "json_object",
    "max_words",
    "max_chars",
    "must_include",
    "must_not_include",
    "regex_full",
    "language",
    "starts_with",
    "ends_with",
}
_MAX_MESSAGE_BYTES = 8 * 1024


@dataclass(frozen=True, slots=True, kw_only=True)
class SuiteSpec:
    """One validated, identity-bound quality suite."""

    suite_id: str
    name: str
    kind: str
    tasks: tuple[dict[str, Any], ...]


def _error(field: str, detail: str) -> ValueError:
    return ValueError(f"{field}: {detail}")


def _string(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise _error(field, "must be a non-empty string")
    return value


def _bounded_message(value: Any, field: str) -> str:
    text = _string(value, field)
    if len(text.encode("utf-8")) > _MAX_MESSAGE_BYTES:
        raise _error(field, f"exceeds {_MAX_MESSAGE_BYTES} bytes")
    return text


def _positive(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0:
        raise _error(field, "must be positive")
    return float(value)


def _validate_common(task: dict[str, Any], index: int) -> None:
    prefix = f"tasks[{index}]"
    _string(task.get("id"), f"{prefix}.id")
    if "weight" in task:
        _positive(task["weight"], f"{prefix}.weight")
    if "max_tokens" in task:
        value = task["max_tokens"]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise _error(f"{prefix}.max_tokens", "must be a positive integer")
    if "ctx_min" in task:
        value = task["ctx_min"]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise _error(f"{prefix}.ctx_min", "must be a positive integer")
    if "system" in task:
        _bounded_message(task["system"], f"{prefix}.system")


def _validate_coding(task: dict[str, Any], index: int) -> None:
    prefix = f"tasks[{index}]"
    _bounded_message(task.get("prompt"), f"{prefix}.prompt")
    _string(task.get("language"), f"{prefix}.language")
    graders = task.get("graders")
    if not isinstance(graders, list) or not graders:
        raise _error(f"{prefix}.graders", "must be a non-empty array")
    for grader_index, grader in enumerate(graders):
        field = f"{prefix}.graders[{grader_index}]"
        if not isinstance(grader, dict):
            raise _error(field, "must be an object")
        grader_type = _string(grader.get("type"), f"{field}.type")
        if grader_type not in _GRADERS:
            raise _error(f"{field}.type", f"unknown grader {grader_type!r}")
        if "partial" in grader and not isinstance(grader["partial"], bool):
            raise _error(f"{field}.partial", "must be a boolean")
        if "weight" in grader:
            _positive(grader["weight"], f"{field}.weight")
        if grader_type == "code_extract":
            _string(grader.get("language"), f"{field}.language")
        if grader_type in {"defines", "signature"}:
            _string(grader.get("name"), f"{field}.name")
        if grader_type == "defines" and grader.get("kind") not in {"function", "class"}:
            raise _error(f"{field}.kind", "must be function or class")
        if grader_type == "signature":
            params = grader.get("params")
            if not isinstance(params, list) or not all(isinstance(item, str) for item in params):
                raise _error(f"{field}.params", "must be a string array")
        if grader_type in {"contains_regex", "absent_regex"}:
            _string(grader.get("pattern"), f"{field}.pattern")
        if grader_type == "exec_python":
            if grader.get("requires_exec") is not True:
                raise _error(f"{field}.requires_exec", "must be true")
            tests = grader.get("tests")
            if (
                not isinstance(tests, list)
                or not tests
                or not all(isinstance(t, str) for t in tests)
            ):
                raise _error(f"{field}.tests", "must be a non-empty string array")


def _validate_tools(tools: Any, field: str, *, descriptions: bool = False) -> set[str]:
    if not isinstance(tools, list) or not tools:
        raise _error(field, "must be a non-empty array")
    names: set[str] = set()
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise _error(f"{field}[{index}]", "must be an object")
        name = _string(tool.get("name"), f"{field}[{index}].name")
        if name in names:
            raise _error(f"{field}[{index}].name", "must be unique")
        names.add(name)
        if descriptions:
            _string(tool.get("description"), f"{field}[{index}].description")
        if not isinstance(tool.get("params", {}), dict):
            raise _error(f"{field}[{index}].params", "must be an object")
    return names


def _validate_tooluse(task: dict[str, Any], index: int) -> None:
    prefix = f"tasks[{index}]"
    names = _validate_tools(task.get("tools"), f"{prefix}.tools", descriptions=True)
    turns = task.get("turns")
    if not isinstance(turns, list) or len(turns) < 2:
        raise _error(f"{prefix}.turns", "must contain user and expectation steps")
    expect_user = True
    for turn_index, turn in enumerate(turns):
        field = f"{prefix}.turns[{turn_index}]"
        if not isinstance(turn, dict):
            raise _error(field, "must be an object")
        if "user" in turn:
            if not expect_user:
                raise _error(field, "user steps must alternate with expectations")
            _bounded_message(turn["user"], f"{field}.user")
            expect_user = False
        elif "expect" in turn:
            if expect_user:
                raise _error(field, "expectation must follow a user step")
            expected = turn["expect"]
            if not isinstance(expected, dict):
                raise _error(f"{field}.expect", "must be an object")
            if _string(expected.get("tool"), f"{field}.expect.tool") not in names:
                raise _error(f"{field}.expect.tool", "is not in the tool catalog")
            if not any(key in expected for key in ("args_equal", "args_subset")):
                raise _error(f"{field}.expect", "requires args_equal or args_subset")
            for key in ("args_equal", "args_subset"):
                if key in expected and not isinstance(expected[key], dict):
                    raise _error(f"{field}.expect.{key}", "must be an object")
            if "tool_result" not in turn:
                raise _error(f"{field}.tool_result", "is required")
            expect_user = True
        elif "expect_final" in turn:
            if expect_user or turn_index != len(turns) - 1:
                raise _error(field, "final expectation must follow the last user step")
            final = turn["expect_final"]
            if not isinstance(final, dict):
                raise _error(f"{field}.expect_final", "must be an object")
            if "contains" in final and (
                not isinstance(final["contains"], list)
                or not all(isinstance(value, str) for value in final["contains"])
            ):
                raise _error(f"{field}.expect_final.contains", "must be a string array")
            if "regex" in final:
                _string(final["regex"], f"{field}.expect_final.regex")
            if not any(key in final for key in ("contains", "regex")):
                raise _error(f"{field}.expect_final", "requires contains or regex")
            expect_user = True
        else:
            raise _error(field, "must be user, expect, or expect_final step")
    if "expect_final" not in turns[-1]:
        raise _error(f"{prefix}.turns", "must end with expect_final")


def _validate_effect(effect: Any, field: str) -> None:
    if not isinstance(effect, dict) or len(effect) != 1:
        raise _error(field, "must contain exactly one declarative effect")
    name, value = next(iter(effect.items()))
    if name not in {"set", "append", "incr"}:
        raise _error(field, f"unknown effect {name!r}")
    if not isinstance(value, list) or len(value) != 2 or not isinstance(value[0], str):
        raise _error(field, "effect must be [path, value]")


def _validate_agentic(task: dict[str, Any], index: int) -> None:
    prefix = f"tasks[{index}]"
    _bounded_message(task.get("prompt"), f"{prefix}.prompt")
    environment = task.get("environment")
    if not isinstance(environment, dict):
        raise _error(f"{prefix}.environment", "must be an object")
    if not isinstance(environment.get("initial_state"), dict):
        raise _error(f"{prefix}.environment.initial_state", "must be an object")
    _validate_tools(environment.get("tools"), f"{prefix}.environment.tools")
    for tool_index, tool in enumerate(environment["tools"]):
        effects = tool.get("effects")
        if not isinstance(effects, list):
            raise _error(f"{prefix}.environment.tools[{tool_index}].effects", "must be an array")
        for effect_index, effect in enumerate(effects):
            _validate_effect(
                effect,
                f"{prefix}.environment.tools[{tool_index}].effects[{effect_index}]",
            )
        _string(tool.get("result"), f"{prefix}.environment.tools[{tool_index}].result")
    goal = environment.get("goal")
    if not isinstance(goal, dict) or not isinstance(goal.get("state_subset"), dict):
        raise _error(f"{prefix}.environment.goal.state_subset", "must be an object")
    for name in ("max_steps", "optimal_steps"):
        value = environment.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise _error(f"{prefix}.environment.{name}", "must be a positive integer")


def _validate_ifollow(task: dict[str, Any], index: int) -> None:
    prefix = f"tasks[{index}]"
    if task.get("kind") == "needle":
        _string(task.get("needle"), f"{prefix}.needle")
        _string(task.get("question"), f"{prefix}.question")
        ratio = task.get("haystack_ratio")
        if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or not 0 < ratio <= 1:
            raise _error(f"{prefix}.haystack_ratio", "must be in (0, 1]")
        position = task.get("position")
        if (
            not isinstance(position, (int, float))
            or isinstance(position, bool)
            or not 0 <= position <= 1
        ):
            raise _error(f"{prefix}.position", "must be in [0, 1]")
        return
    _bounded_message(task.get("prompt"), f"{prefix}.prompt")
    constraints = task.get("constraints")
    if not isinstance(constraints, list) or not constraints:
        raise _error(f"{prefix}.constraints", "must be a non-empty array")
    for constraint_index, constraint in enumerate(constraints):
        field = f"{prefix}.constraints[{constraint_index}]"
        if not isinstance(constraint, dict):
            raise _error(field, "must be an object")
        constraint_type = _string(constraint.get("type"), f"{field}.type")
        if constraint_type not in _CONSTRAINTS:
            raise _error(f"{field}.type", f"unknown constraint {constraint_type!r}")
        if constraint_type == "json_object":
            subset = constraint.get("schema_subset")
            if subset is not None and not isinstance(subset, dict):
                raise _error(f"{field}.schema_subset", "must be an object")
        elif constraint_type in {"max_words", "max_chars"}:
            value = constraint.get("value")
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise _error(f"{field}.value", "must be a positive integer")
        elif constraint_type in {"must_include", "must_not_include"}:
            values = constraint.get("values")
            if (
                not isinstance(values, list)
                or not values
                or not all(isinstance(value, str) and value for value in values)
            ):
                raise _error(f"{field}.values", "must be a non-empty string array")
        elif constraint_type == "regex_full":
            _string(constraint.get("pattern"), f"{field}.pattern")
        else:
            _string(constraint.get("value"), f"{field}.value")


def _validated(document: dict[str, Any]) -> SuiteSpec:
    if document.get("schema_version") != 1:
        raise _error("schema_version", "must be 1")
    name = _string(document.get("name"), "name")
    version = document.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
        raise _error("version", "must be a positive integer")
    kind = _string(document.get("kind"), "kind")
    validators = {
        "coding": _validate_coding,
        "tooluse": _validate_tooluse,
        "agentic": _validate_agentic,
        "ifollow": _validate_ifollow,
    }
    if kind not in validators:
        raise _error("kind", f"unknown suite kind {kind!r}")
    tasks = document.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise _error("tasks", "must be a non-empty array")
    ids: set[str] = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise _error(f"tasks[{index}]", "must be an object")
        _validate_common(task, index)
        task_id = str(task["id"])
        if task_id in ids:
            raise _error(f"tasks[{index}].id", "must be unique")
        ids.add(task_id)
        validators[kind](task, index)
    canonical = json.dumps(tasks, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    return SuiteSpec(
        suite_id=f"{name}@{version}+{digest}", name=name, kind=kind, tasks=tuple(tasks)
    )


def _read_document(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise _error(str(path), "suite document must be an object")
    return value


def load_suite(name_or_path: str) -> SuiteSpec:
    """Load a bundled suite name or validate a user JSON suite path."""
    if name_or_path == "perplexity":
        canonical = json.dumps([], separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
        return SuiteSpec(
            suite_id=f"perplexity@1+{digest}", name="perplexity", kind="perplexity", tasks=()
        )
    if name_or_path in _BUNDLED:
        resource = resources.files("llamatune").joinpath("evaltasks", f"{name_or_path}.json")
        try:
            value = json.loads(resource.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"bundled suite {name_or_path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"bundled suite {name_or_path}: document must be an object")
        return _validated(value)
    path = Path(name_or_path)
    if not path.is_file():
        raise ValueError(f"unknown suite or unreadable path: {name_or_path}")
    return _validated(_read_document(path))


def bundled_suites() -> tuple[str, ...]:
    """Return bundled suite names in stable CLI order."""
    return _BUNDLED


def build_haystack(task_id: str, chars: int) -> str:
    """Return deterministic neutral filler of exactly ``chars`` characters."""
    if chars < 0:
        raise ValueError("chars must be non-negative")
    if chars == 0:
        return ""
    filler = (
        resources.files("llamatune").joinpath("evaltasks", "filler.txt").read_text(encoding="utf-8")
    )
    normalized = " ".join(filler.split())
    if not normalized:
        raise ValueError("bundled filler corpus is empty")
    offset = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) % len(normalized)
    rotated = normalized[offset:] + " " + normalized[:offset]
    repeated = (rotated + " ") * (chars // (len(rotated) + 1) + 2)
    return repeated[:chars]
