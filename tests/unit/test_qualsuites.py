"""Unit tests for quality-suite loading, validation, and bundled content."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from importlib import resources
from pathlib import Path
from typing import Any

import pytest

from llamatune.qualsuites import build_haystack, bundled_suites, load_suite


def _user_suite(path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "custom",
        "version": 1,
        "kind": "ifollow",
        "tasks": [
            {
                "id": "one",
                "prompt": "Reply with ok.",
                "constraints": [{"type": "must_include", "values": ["ok"]}],
            }
        ],
    }


def test_bundled_suite_order_and_minimum_content() -> None:
    assert bundled_suites() == ("coding", "tooluse", "agentic", "ifollow", "perplexity")
    coding = load_suite("coding")
    tooluse = load_suite("tooluse")
    agentic = load_suite("agentic")
    ifollow = load_suite("ifollow")
    perplexity = load_suite("perplexity")

    assert (coding.name, coding.kind, len(coding.tasks)) == ("coding", "coding", 20)
    assert (
        sum(
            any(grader.get("type") == "exec_python" for grader in task["graders"])
            for task in coding.tasks
        )
        >= 8
    )
    assert (tooluse.name, tooluse.kind, len(tooluse.tasks)) == ("tooluse", "tooluse", 15)
    assert sum(sum("expect" in turn for turn in task["turns"]) >= 2 for task in tooluse.tasks) >= 5
    assert (agentic.name, agentic.kind, len(agentic.tasks)) == ("agentic", "agentic", 8)
    assert (ifollow.name, ifollow.kind, len(ifollow.tasks)) == ("ifollow", "ifollow", 15)
    assert sum(task.get("kind") == "needle" for task in ifollow.tasks) >= 5
    assert perplexity.kind == "perplexity" and perplexity.tasks == ()
    assert all("+" in suite.suite_id for suite in (coding, tooluse, agentic, ifollow))


def test_bundled_filler_is_packaged_original_scale_data() -> None:
    text = resources.files("llamatune").joinpath("evaltasks", "filler.txt").read_text()
    assert len(text.encode()) >= 64 * 1024
    assert "A careful archivist" in text


def test_suite_id_is_stable_and_changes_with_one_character(tmp_path: Path) -> None:
    path = tmp_path / "suite.json"
    document = _user_suite(path)
    path.write_text(json.dumps(document))
    first = load_suite(str(path))
    second = load_suite(str(path))
    assert first == second

    tasks = document["tasks"]
    assert isinstance(tasks, list) and isinstance(tasks[0], dict)
    tasks[0]["prompt"] = "Reply with OK."
    path.write_text(json.dumps(document))
    changed = load_suite(str(path))
    assert changed.suite_id != first.suite_id


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        (lambda document: document.update(schema_version=2), "schema_version"),
        (lambda document: document.update(version=0), "version"),
        (lambda document: document.update(kind="unknown"), "kind"),
        (lambda document: document.update(tasks=[]), "tasks"),
        (lambda document: document["tasks"][0].update(id=""), "tasks[0].id"),
        (
            lambda document: document["tasks"][0].update(constraints=[{"type": "not-real"}]),
            "constraints[0].type",
        ),
    ],
)
def test_user_suite_validation_names_bad_field(
    tmp_path: Path, mutation: Callable[[dict[str, Any]], None], field: str
) -> None:
    path = tmp_path / "suite.json"
    document = _user_suite(path)
    mutation(document)
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=field.replace("[", r"\[").replace("]", r"\]")):
        load_suite(str(path))


def test_duplicate_ids_and_oversized_messages_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "suite.json"
    document = _user_suite(path)
    tasks = document["tasks"]
    assert isinstance(tasks, list) and isinstance(tasks[0], dict)
    tasks.append(dict(tasks[0]))
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="must be unique"):
        load_suite(str(path))

    tasks.pop()
    tasks[0]["prompt"] = "x" * (8 * 1024 + 1)
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="exceeds"):
        load_suite(str(path))


def test_coding_grader_fields_are_validated(tmp_path: Path) -> None:
    path = tmp_path / "coding.json"
    document = {
        "schema_version": 1,
        "name": "custom-coding",
        "version": 1,
        "kind": "coding",
        "tasks": [
            {
                "id": "one",
                "prompt": "Return Python code.",
                "language": "python",
                "graders": [{"type": "code_extract"}],
            }
        ],
    }
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=r"graders\[0\]\.language"):
        load_suite(str(path))


def test_tooluse_turns_must_alternate_and_end_with_final(tmp_path: Path) -> None:
    path = tmp_path / "tooluse.json"
    document = {
        "schema_version": 1,
        "name": "custom-tooluse",
        "version": 1,
        "kind": "tooluse",
        "tasks": [
            {
                "id": "one",
                "tools": [{"name": "lookup", "description": "Lookup.", "params": {}}],
                "turns": [{"user": "Look up one."}, {"user": "Again."}],
            }
        ],
    }
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="must alternate"):
        load_suite(str(path))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("weight", 0, "weight"),
        ("max_tokens", 0, "max_tokens"),
        ("ctx_min", False, "ctx_min"),
        ("system", "x" * (8 * 1024 + 1), "system"),
    ],
)
def test_common_task_bounds_name_the_field(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    path = tmp_path / "suite.json"
    document = _user_suite(path)
    document["tasks"][0][field] = value
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=message):
        load_suite(str(path))


def _coding_document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "custom-coding",
        "version": 1,
        "kind": "coding",
        "tasks": [
            {
                "id": "one",
                "prompt": "Return Python code.",
                "language": "python",
                "graders": [
                    {"type": "code_extract", "language": "python"},
                    {"type": "defines", "name": "answer", "kind": "function"},
                    {"type": "signature", "name": "answer", "params": []},
                    {"type": "contains_regex", "pattern": "answer"},
                    {
                        "type": "exec_python",
                        "requires_exec": True,
                        "tests": ["assert answer() == 1"],
                    },
                ],
            }
        ],
    }


@pytest.mark.parametrize(
    ("grader_index", "mutation", "message"),
    [
        (0, lambda grader: grader.update(type="unknown"), "unknown grader"),
        (0, lambda grader: grader.update(partial="yes"), "partial"),
        (0, lambda grader: grader.update(weight=0), "weight"),
        (1, lambda grader: grader.update(kind="module"), "kind"),
        (2, lambda grader: grader.update(params=[1]), "params"),
        (3, lambda grader: grader.pop("pattern"), "pattern"),
        (4, lambda grader: grader.update(requires_exec=False), "requires_exec"),
        (4, lambda grader: grader.update(tests=[]), "tests"),
    ],
)
def test_coding_rejects_malformed_grader_contracts(
    tmp_path: Path,
    grader_index: int,
    mutation: Callable[[dict[str, Any]], Any],
    message: str,
) -> None:
    path = tmp_path / "coding.json"
    document = _coding_document()
    mutation(document["tasks"][0]["graders"][grader_index])
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=message):
        load_suite(str(path))


def _tooluse_document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "custom-tooluse",
        "version": 1,
        "kind": "tooluse",
        "tasks": [
            {
                "id": "one",
                "tools": [{"name": "lookup", "description": "Lookup.", "params": {}}],
                "turns": [
                    {"user": "Look up one."},
                    {
                        "expect": {"tool": "lookup", "args_equal": {"id": 1}},
                        "tool_result": {"value": "one"},
                    },
                    {"user": "Summarize one."},
                    {"expect_final": {"contains": ["one"]}},
                ],
            }
        ],
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda doc: doc["tasks"][0]["tools"][0].pop("description"), "description"),
        (
            lambda doc: doc["tasks"][0]["turns"][1]["expect"].update(tool="missing"),
            "tool catalog",
        ),
        (
            lambda doc: doc["tasks"][0]["turns"][1]["expect"].pop("args_equal"),
            "requires args_equal",
        ),
        (lambda doc: doc["tasks"][0]["turns"][1].pop("tool_result"), "tool_result"),
        (
            lambda doc: doc["tasks"][0]["turns"][-1].update(expect_final={}),
            "requires contains",
        ),
        (lambda doc: doc["tasks"][0]["turns"].pop(), "must end"),
    ],
)
def test_tooluse_rejects_malformed_protocol(
    tmp_path: Path, mutation: Callable[[dict[str, Any]], Any], message: str
) -> None:
    path = tmp_path / "tooluse.json"
    document = deepcopy(_tooluse_document())
    mutation(document)
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=message):
        load_suite(str(path))


def _document_for_task(name: str, kind: str, task: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": name,
        "version": 1,
        "kind": kind,
        "tasks": [task],
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda env: env.update(initial_state=[]), "initial_state"),
        (lambda env: env.update(tools=[]), "tools"),
        (lambda env: env["tools"][0].update(effects={}), "effects"),
        (lambda env: env["tools"][0]["effects"].append({}), "declarative effect"),
        (
            lambda env: env["tools"][0]["effects"].append({"unknown": ["state", 1]}),
            "unknown effect",
        ),
        (lambda env: env["tools"][0]["effects"].append({"set": [1]}), "effect must"),
        (lambda env: env.update(goal={}), "state_subset"),
        (lambda env: env.update(max_steps=0), "max_steps"),
    ],
)
def test_agentic_rejects_unsafe_or_incomplete_environment(
    tmp_path: Path, mutation: Callable[[dict[str, Any]], Any], message: str
) -> None:
    path = tmp_path / "agentic.json"
    task = deepcopy(load_suite("agentic").tasks[0])
    mutation(task["environment"])
    path.write_text(json.dumps(_document_for_task("custom-agentic", "agentic", task)))
    with pytest.raises(ValueError, match=message):
        load_suite(str(path))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda task: task.update(haystack_ratio=0), "haystack_ratio"),
        (lambda task: task.update(position=2), "position"),
    ],
)
def test_needle_bounds_are_validated(
    tmp_path: Path, mutation: Callable[[dict[str, Any]], Any], message: str
) -> None:
    path = tmp_path / "ifollow.json"
    task = deepcopy(next(task for task in load_suite("ifollow").tasks if task.get("kind")))
    mutation(task)
    path.write_text(json.dumps(_document_for_task("custom-ifollow", "ifollow", task)))
    with pytest.raises(ValueError, match=message):
        load_suite(str(path))


@pytest.mark.parametrize(
    ("constraint", "message"),
    [
        ({"type": "json_object", "schema_subset": []}, "schema_subset"),
        ({"type": "max_words", "value": 0}, "value"),
        ({"type": "must_include", "values": []}, "values"),
        ({"type": "regex_full"}, "pattern"),
        ({"type": "starts_with"}, "value"),
    ],
)
def test_ifollow_constraint_fields_are_validated(
    tmp_path: Path, constraint: dict[str, Any], message: str
) -> None:
    path = tmp_path / "ifollow.json"
    task = {"id": "one", "prompt": "Reply.", "constraints": [constraint]}
    path.write_text(json.dumps(_document_for_task("custom-ifollow", "ifollow", task)))
    with pytest.raises(ValueError, match=message):
        load_suite(str(path))


def test_unknown_suite_and_invalid_documents_are_plain_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown suite"):
        load_suite("not-a-suite")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]")
    with pytest.raises(ValueError, match="must be an object"):
        load_suite(str(invalid))
    invalid.write_text("{broken")
    with pytest.raises(ValueError, match=r"invalid\.json"):
        load_suite(str(invalid))


def test_haystack_is_deterministic_seeded_and_exact_length() -> None:
    first = build_haystack("needle-a", 12_345)
    assert first == build_haystack("needle-a", 12_345)
    assert len(first) == 12_345
    assert first != build_haystack("needle-b", 12_345)
    assert build_haystack("needle-a", 0) == ""
    with pytest.raises(ValueError, match="non-negative"):
        build_haystack("needle-a", -1)
