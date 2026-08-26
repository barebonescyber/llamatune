"""Confined quality-run writer and strictly serial evaluation orchestrator."""

from __future__ import annotations

import contextlib
import dataclasses
import fnmatch
import importlib
import json
import math
import os
import re
import secrets
import signal
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self, cast

from llamatune import __version__, executor
from llamatune.config import hardware_signature
from llamatune.hardware import assess_hardware
from llamatune.llama import LlamaDiscoveryError, discover_llama
from llamatune.model import ModelInspectionError, inspect_model
from llamatune.qualserver import (
    ServerHandle,
    ServerProtocolError,
    ServerStartError,
    ServerUnavailableError,
    start,
)
from llamatune.qualsuites import SuiteSpec, build_haystack, load_suite
from llamatune.types import (
    HardwareReport,
    LlamaCppReport,
    ModelReport,
    QualityOptions,
    QualityOutcome,
    SuiteResult,
    TaskGrade,
    TrialConfig,
)

_SCHEMA_VERSION = 1
_CREATE_RETRIES = 10
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, frozenset):
        return sorted(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _confine(root: Path, *parts: str) -> Path:
    resolved = root.resolve()
    target = resolved.joinpath(*parts).resolve()
    if target != resolved and resolved not in target.parents:
        raise ValueError(f"quality path escapes run directory: {'/'.join(parts)}")
    return target


def _component(value: str) -> str:
    safe = _SAFE_COMPONENT.sub("_", value).strip("._")
    if not safe:
        safe = "item"
    if safe == value:
        return safe
    import hashlib

    return f"{safe}-{hashlib.sha256(value.encode()).hexdigest()[:8]}"


def _options_dict(options: QualityOptions) -> dict[str, Any]:
    return cast(dict[str, Any], _jsonable(dataclasses.asdict(options)))


def _options_from_dict(value: dict[str, Any]) -> QualityOptions:
    return QualityOptions(
        model_path=Path(value["model_path"]),
        llama_bin=Path(value["llama_bin"]) if value.get("llama_bin") is not None else None,
        sessions_dir=Path(value["sessions_dir"]),
        config_mode=str(value["config_mode"]),
        config_session=(
            Path(value["config_session"]) if value.get("config_session") is not None else None
        ),
        strict_config=bool(value["strict_config"]),
        compare_lossless=bool(value["compare_lossless"]),
        suites=tuple(str(item) for item in value["suites"]),
        task_filters=tuple(str(item) for item in value["task_filters"]),
        exec_enabled=bool(value["exec_enabled"]),
        ctx_size=int(value["ctx_size"]),
        quality_corpus=(
            Path(value["quality_corpus"]) if value.get("quality_corpus") is not None else None
        ),
        reps=int(value["reps"]),
        max_tokens=int(value["max_tokens"]),
        request_timeout_s=float(value["request_timeout_s"]),
        server_start_timeout_s=float(value["server_start_timeout_s"]),
        seed=int(value["seed"]),
        dry_run=bool(value["dry_run"]),
    )


class QualityRun:
    """Sole writer for one path-confined quality evidence directory."""

    def __init__(
        self,
        run_dir: Path,
        *,
        entries: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.dir = Path(run_dir).resolve()
        self._entries = entries or []
        self._metadata = metadata or {}
        server_root = self.dir / "server"
        self._server_launches = (
            max(
                (
                    int(path.name)
                    for path in server_root.iterdir()
                    if path.is_dir() and path.name.isdigit()
                ),
                default=0,
            )
            if server_root.is_dir()
            else 0
        )

    @property
    def entries(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._entries)

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    @classmethod
    def create(
        cls,
        sessions_dir: Path,
        *,
        options: QualityOptions,
        hardware: HardwareReport,
        llama: LlamaCppReport,
        model: ModelReport,
        argv: list[str],
    ) -> Self:
        root = sessions_dir.resolve() / "quality"
        root.mkdir(parents=True, exist_ok=True)
        run_dir: Path | None = None
        for _ in range(_CREATE_RETRIES):
            stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            candidate = root / f"{_component(model.path.stem)}-{stamp}-{secrets.token_hex(3)}"
            try:
                candidate.mkdir(exist_ok=False)
            except FileExistsError:
                continue
            run_dir = candidate
            break
        if run_dir is None:
            raise OSError(f"could not allocate a quality run under {root}")
        created = _utc_now()
        metadata = {
            "schema_version": _SCHEMA_VERSION,
            "tool_version": __version__,
            "argv": list(argv),
            "options": _options_dict(options),
            "created": created,
            "model_fingerprint": model.fingerprint,
            "build_discriminator": llama.bench_sha256 or llama.help_sha256,
        }
        run = cls(run_dir, metadata=metadata)
        run.write_json("run.json", metadata)
        run.write_json("hardware.json", _jsonable(dataclasses.asdict(hardware)))
        run.write_json("model.json", _jsonable(dataclasses.asdict(model)))
        run.write_json("llamacpp.json", _jsonable(dataclasses.asdict(llama)))
        run.append(
            {
                "type": "quality_start",
                "tool_version": __version__,
                "argv": list(argv),
                "model_fingerprint": model.fingerprint,
            }
        )
        return run

    @classmethod
    def load(cls, run_dir: Path) -> Self:
        root = Path(run_dir).resolve()
        metadata = _read_json(root / "run.json")
        if metadata.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("unsupported quality run schema version")
        entries: list[dict[str, Any]] = []
        journal = root / "journal.jsonl"
        if journal.is_file():
            lines = journal.read_text(encoding="utf-8", errors="replace").splitlines()
            for index, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    if index == len(lines) - 1:
                        break
                    raise ValueError(f"journal.jsonl line {index + 1} is corrupt") from None
                if not isinstance(value, dict):
                    raise ValueError(f"journal.jsonl line {index + 1} is not an object")
                entries.append(value)
        return cls(root, entries=entries, metadata=metadata)

    def _update_metadata(self, values: dict[str, Any]) -> None:
        self._metadata.update(_jsonable(values))
        self.write_json("run.json", self._metadata)

    def append(self, entry: dict[str, Any]) -> None:
        record = _jsonable(dict(entry))
        record.setdefault("ts", _utc_now())
        path = _confine(self.dir, "journal.jsonl")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._entries.append(record)

    def task_dir(self, suite: str, task_id: str, rep: int, side: str) -> Path:
        path = _confine(
            self.dir,
            "tasks",
            _component(suite),
            _component(task_id),
            f"rep-{rep}",
            f"side-{_component(side)}",
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _allocate_server_launch(self) -> int:
        self._server_launches += 1
        self.server_dir(self._server_launches)
        return self._server_launches

    def server_dir(self, number: int) -> Path:
        path = _confine(self.dir, "server", str(number))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def perplexity_dir(self, side: str) -> Path:
        path = _confine(self.dir, "perplexity", f"side-{_component(side)}")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, name: str, payload: dict[str, Any]) -> None:
        path = _confine(self.dir, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def write_text(self, name: str, text: str) -> None:
        path = _confine(self.dir, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


@dataclass(frozen=True, slots=True)
class _Resolved:
    options: QualityOptions
    hardware: HardwareReport
    llama: LlamaCppReport
    model: ModelReport
    suites: tuple[SuiteSpec, ...]
    config: TrialConfig | None
    config_source: str
    config_provenance: Path | None
    lossless_config: TrialConfig | None
    warnings: tuple[str, ...]


class _ForcedInterrupt(BaseException):
    pass


@dataclass(slots=True)
class _SignalState:
    count: int = 0
    stop_requested: bool = False
    forced: bool = False
    handle: ServerHandle | None = None

    def receive(self, signum: int, frame: Any) -> None:
        del signum, frame
        self.count += 1
        self.stop_requested = True
        if self.count < 2:
            return
        self.forced = True
        sandbox_module = sys.modules.get("llamatune.sandbox")
        terminate = getattr(sandbox_module, "terminate_active", None)
        if callable(terminate):
            terminate()
        raise _ForcedInterrupt


def _build_discriminator(llama: LlamaCppReport) -> str:
    return llama.bench_sha256 or llama.help_sha256


def _load_session_configs(
    path: Path, model: ModelReport
) -> tuple[TrialConfig, TrialConfig | None, int | None]:
    source_model = _read_json(path / "model.json")
    if source_model.get("fingerprint") != model.fingerprint:
        raise ValueError("config session model fingerprint does not match the target model")
    analysis = _read_json(path / "analysis.json")
    recommended = path / "recommended.json"
    if recommended.is_file():
        document = _read_json(recommended)
        if document.get("confirmed") is not True:
            raise ValueError(f"session recommendation is not confirmed: {path}")
        value = document.get("config")
    else:
        value = analysis.get("winner")
        value = (
            value.get("config")
            if isinstance(value, dict) and value.get("confirmed") is True
            else None
        )
    if not isinstance(value, dict):
        raise ValueError(f"no confirmed recommendation in session {path}")
    alternate = analysis.get("lossless_winner")
    alternate_config = alternate.get("config") if isinstance(alternate, dict) else None
    evaluated = TrialConfig.from_dict(value)
    validated_contexts: list[int] = []
    feasibility = analysis.get("feasibility")
    if isinstance(feasibility, dict):
        ctx_validated = feasibility.get("ctx_validated")
        if isinstance(ctx_validated, int) and not isinstance(ctx_validated, bool):
            validated_contexts.append(ctx_validated)
    envelope = analysis.get("context_envelope")
    if isinstance(envelope, list):
        for row in envelope:
            if not isinstance(row, dict) or row.get("status") != "ok":
                continue
            if row.get("fallback_config") is not None:
                continue
            row_config = row.get("config")
            if not isinstance(row_config, dict):
                continue
            try:
                matches = TrialConfig.from_dict(row_config) == evaluated
            except (KeyError, TypeError, ValueError):
                matches = False
            row_ctx = row.get("ctx")
            if matches and isinstance(row_ctx, int) and not isinstance(row_ctx, bool):
                validated_contexts.append(row_ctx)
    return (
        evaluated,
        TrialConfig.from_dict(alternate_config) if isinstance(alternate_config, dict) else None,
        max(validated_contexts) if validated_contexts else None,
    )


def _resolve_config(
    options: QualityOptions,
    hardware: HardwareReport,
    llama: LlamaCppReport,
    model: ModelReport,
) -> tuple[TrialConfig | None, str, Path | None, tuple[str, ...], TrialConfig | None]:
    warnings: list[str] = []
    if options.config_mode == "defaults":
        return None, "defaults", None, (), None
    if options.config_mode == "session":
        if options.config_session is None:
            raise ValueError("session config mode requires a session path")
        evaluated, lossless, validated_ctx = _load_session_configs(options.config_session, model)
        if validated_ctx is not None and validated_ctx < options.ctx_size:
            warnings.append(
                f"requested ctx_size {options.ctx_size} exceeds the session's "
                f"validated context {validated_ctx}"
            )
        return (
            evaluated,
            "session",
            options.config_session.resolve(),
            tuple(warnings),
            lossless,
        )
    if options.config_mode != "best":
        raise ValueError(f"unknown config mode: {options.config_mode}")
    from llamatune.registry import lookup

    result = lookup(
        options.sessions_dir / "registry.jsonl",
        model,
        llama,
        hardware_signature(hardware),
        ctx_size=options.ctx_size,
    )
    record = result.get("record")
    if result.get("status") == "hit" and isinstance(record, dict):
        config = record.get("config")
        if isinstance(config, dict):
            provenance = record.get("session_dir")
            return (
                TrialConfig.from_dict(config),
                "registry",
                Path(provenance).resolve() if isinstance(provenance, str) else None,
                (),
                None,
            )
    reasons = result.get("stale_reasons")
    detail = ", ".join(str(item) for item in reasons) if isinstance(reasons, list) else "no match"
    if options.strict_config:
        raise RuntimeError(f"no usable registry config ({detail or 'no match'})")
    warnings.append(f"no current registry config; evaluating defaults ({detail or 'no match'})")
    return None, "defaults", None, tuple(warnings), None


def _selected_suites(options: QualityOptions) -> tuple[SuiteSpec, ...]:
    suites = tuple(load_suite(name) for name in options.suites)
    return tuple(sorted(suites, key=lambda suite: suite.kind == "perplexity"))


def _lossless(config: TrialConfig | None) -> TrialConfig | None:
    if config is None:
        return None
    return replace(config, cache_type_k="f16", cache_type_v="f16")


def _resolve(options: QualityOptions) -> _Resolved:
    hardware = assess_hardware()
    llama = discover_llama(options.llama_bin)
    model = inspect_model(options.model_path)
    suites = _selected_suites(options)
    if any(suite.kind != "perplexity" for suite in suites) and llama.server_path is None:
        raise RuntimeError("llama-server was not found in the selected llama.cpp build")
    if any(suite.kind == "perplexity" for suite in suites):
        if options.quality_corpus is None or not options.quality_corpus.is_file():
            raise ValueError("perplexity requires a readable quality corpus")
        if llama.perplexity_path is None:
            raise RuntimeError("llama-perplexity was not found in the selected llama.cpp build")
    config, source, provenance, warnings, preferred_lossless = _resolve_config(
        options, hardware, llama, model
    )
    lossless: TrialConfig | None = None
    if options.compare_lossless:
        if config is not None and (config.cache_type_k != "f16" or config.cache_type_v != "f16"):
            lossless = preferred_lossless or _lossless(config)
        else:
            warnings = (*warnings, "lossless comparison is a no-op for the evaluated config")
    return _Resolved(
        options=options,
        hardware=hardware,
        llama=llama,
        model=model,
        suites=suites,
        config=config,
        config_source=source,
        config_provenance=provenance,
        lossless_config=lossless,
        warnings=warnings,
    )


def _server_argv(
    resolved: _Resolved,
    config: TrialConfig | None,
    *,
    port: str = "0",
) -> tuple[str, ...]:
    server = resolved.llama.server_path
    if server is None:
        raise RuntimeError("llama-server is unavailable")
    argv = [
        str(server),
        "-m",
        str(resolved.model.path),
        "-c",
        str(resolved.options.ctx_size),
    ]
    if config is not None:
        argv.extend(("-ngl", str(config.gpu_layers)))
        if "ncmoe" in resolved.llama.capabilities and config.ot_spec is None:
            argv.extend(("--n-cpu-moe", str(config.moe_cpu_layers)))
        if "fa" in resolved.llama.capabilities:
            argv.extend(("-fa", "on" if config.flash_attn else "off"))
        argv.extend(("-b", str(config.batch), "-ub", str(config.ubatch), "-t", str(config.threads)))
        if "ctk" in resolved.llama.capabilities:
            argv.extend(("-ctk", config.cache_type_k))
        if "ctv" in resolved.llama.capabilities:
            argv.extend(("-ctv", config.cache_type_v))
        if config.ot_spec is not None and "ot" in resolved.llama.capabilities:
            argv.extend(("-ot", config.ot_spec))
        if config.threads_batch is not None and "tb" in resolved.llama.capabilities:
            argv.extend(("-tb", str(config.threads_batch)))
        if config.tensor_split is not None and "ts" in resolved.llama.capabilities:
            argv.extend(("-ts", ",".join(f"{value:g}" for value in config.tensor_split)))
        if config.split_mode is not None and "sm" in resolved.llama.capabilities:
            argv.extend(("-sm", config.split_mode))
        if not config.mmap and "mmp" in resolved.llama.capabilities:
            argv.append("--no-mmap")
        if config.no_kv_offload and "nkvo" in resolved.llama.capabilities:
            argv.append("--no-kv-offload")
    argv.extend(
        (
            "--host",
            "127.0.0.1",
            "--port",
            port,
            "--seed",
            str(resolved.options.seed),
        )
    )
    return tuple(argv)


def _task_selected(task_id: str, filters: tuple[str, ...]) -> bool:
    return not filters or any(fnmatch.fnmatchcase(task_id, pattern) for pattern in filters)


def _task_key(suite_id: str, task_id: str, rep: int, side: str) -> tuple[str, str, int, str]:
    return suite_id, task_id, rep, side


def _grade_dict(grade: TaskGrade) -> dict[str, Any]:
    return {
        "id": grade.task_id,
        "score": grade.score,
        "status": grade.status,
        "reason": grade.reason,
        "unstable": grade.unstable,
        "graders": list(grade.grader_results),
    }


def _grade_from_dict(value: dict[str, Any]) -> TaskGrade:
    graders = value.get("graders", ())
    return TaskGrade(
        task_id=str(value["id"]),
        score=float(value["score"]),
        status=str(value["status"]),
        reason=str(value["reason"]) if value.get("reason") is not None else None,
        unstable=bool(value.get("unstable", False)),
        grader_results=tuple(item for item in graders if isinstance(item, dict)),
    )


def _journaled_grades(run: QualityRun) -> dict[tuple[str, str, int, str], TaskGrade]:
    found: dict[tuple[str, str, int, str], TaskGrade] = {}
    for entry in run.entries:
        if entry.get("type") != "task_result":
            continue
        try:
            key = _task_key(
                str(entry["suite_id"]),
                str(entry["task_id"]),
                int(entry["rep"]),
                str(entry["side"]),
            )
            grade = entry["grade"]
            if isinstance(grade, dict):
                found[key] = _grade_from_dict(grade)
        except (KeyError, TypeError, ValueError):
            continue
    return found


def _messages_for_task(task: dict[str, Any], ctx_size: int) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system = task.get("system")
    if isinstance(system, str):
        messages.append({"role": "system", "content": system})
    if task.get("kind") == "needle":
        needle = str(task["needle"])
        chars = max(0, int(ctx_size * 4 * float(task["haystack_ratio"])))
        haystack = build_haystack(str(task["id"]), chars)
        point = int(len(haystack) * float(task["position"]))
        content = f"{haystack[:point]}\n{needle}\n{haystack[point:]}\n\n{task['question']}"
    else:
        content = str(task.get("prompt", ""))
    messages.append({"role": "user", "content": content})
    return messages


def _tool_protocol(tools: Any) -> dict[str, Any]:
    catalog = json.dumps(tools, sort_keys=True, separators=(",", ":"))
    return {
        "role": "system",
        "content": (
            f"Available tools: {catalog}\n"
            "Reply with exactly one fenced ```json block containing: "
            '{"tool":"<name>","args":{...}} to call a tool, or '
            '{"final":"<answer>"} when finished.'
        ),
    }


def _normalized_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return float(actual) == float(expected)
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.strip() == expected.strip()
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _normalized_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _normalized_equal(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _normalized_subset(actual: Any, expected: Any) -> bool:
    return isinstance(actual, dict) and all(
        key in actual and _normalized_equal(actual[key], value) for key, value in expected.items()
    )


def _request_payload(
    messages: Sequence[dict[str, Any]], max_tokens: int, seed: int
) -> dict[str, Any]:
    return {
        "messages": list(messages),
        "temperature": 0,
        "top_k": 1,
        "top_p": 1,
        "seed": seed,
        "max_tokens": max_tokens,
    }


def _chat(
    run: QualityRun,
    handle: ServerHandle,
    task_dir: Path,
    turn: int,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    seed: int,
    timeout_s: float,
) -> str:
    relative = task_dir.relative_to(run.dir)
    run.write_json(
        str(relative / f"request-{turn}.json"),
        _request_payload(messages, max_tokens, seed),
    )
    try:
        response = handle.chat(
            messages,
            max_tokens=max_tokens,
            seed=seed,
            timeout_s=timeout_s,
        )
    except Exception as exc:
        run.write_json(str(relative / f"response-{turn}.json"), {"error": str(exc)})
        raise
    run.write_json(str(relative / f"response-{turn}.json"), {"content": response})
    return response


def _drive_task(
    run: QualityRun,
    handle: ServerHandle,
    suite: SuiteSpec,
    task: dict[str, Any],
    rep: int,
    side: str,
    options: QualityOptions,
) -> tuple[str, ...]:
    directory = run.task_dir(suite.name, str(task["id"]), rep, side)
    limit = min(int(task.get("max_tokens", options.max_tokens)), options.max_tokens)
    if suite.kind == "tooluse":
        messages: list[dict[str, Any]] = [_tool_protocol(task["tools"])]
        if isinstance(task.get("system"), str):
            messages.append({"role": "system", "content": task["system"]})
        responses: list[str] = []
        turn_number = 0
        for step in task["turns"]:
            if "user" in step:
                messages.append({"role": "user", "content": str(step["user"])})
                response = _chat(
                    run,
                    handle,
                    directory,
                    turn_number,
                    messages,
                    max_tokens=limit,
                    seed=options.seed,
                    timeout_s=options.request_timeout_s,
                )
                responses.append(response)
                messages.append({"role": "assistant", "content": response})
                turn_number += 1
            elif "tool_result" in step:
                from llamatune.qualscore import extract_call

                call = extract_call(responses[-1]) if responses else None
                expected = step.get("expect", {})
                expected_tool = expected.get("tool") if isinstance(expected, dict) else None
                expected_equal = expected.get("args_equal") if isinstance(expected, dict) else None
                expected_subset = (
                    expected.get("args_subset") if isinstance(expected, dict) else None
                )
                matches = isinstance(call, dict) and call.get("tool") == expected_tool
                args = call.get("args") if isinstance(call, dict) else None
                if isinstance(expected_equal, dict):
                    matches = matches and _normalized_equal(args, expected_equal)
                elif isinstance(expected_subset, dict):
                    matches = matches and _normalized_subset(args, expected_subset)
                else:
                    matches = False
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(
                            step["tool_result"] if matches else {"error": "invalid tool call"},
                            sort_keys=True,
                        ),
                    }
                )
        return tuple(responses)
    messages = _messages_for_task(task, options.ctx_size)
    if suite.kind == "agentic":
        messages.insert(0, _tool_protocol(task["environment"]["tools"]))
    responses = []
    for turn_number in range(int(task.get("environment", {}).get("max_steps", 1))):
        response = _chat(
            run,
            handle,
            directory,
            turn_number,
            messages,
            max_tokens=limit,
            seed=options.seed,
            timeout_s=options.request_timeout_s,
        )
        responses.append(response)
        if suite.kind != "agentic":
            break
        from llamatune.qualscore import replay_agentic

        replay = replay_agentic(task, tuple(responses))
        if replay.get("done"):
            break
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": str(replay.get("tool_reply", ""))})
    return tuple(responses)


def _error_grade(task_id: str, reason: str) -> TaskGrade:
    return TaskGrade(
        task_id=task_id,
        score=0.0,
        status="error",
        reason=reason,
        unstable=False,
        grader_results=(),
    )


def _skipped_grade(task_id: str, reason: str) -> TaskGrade:
    return TaskGrade(
        task_id=task_id,
        score=0.0,
        status="skipped",
        reason=reason,
        unstable=False,
        grader_results=(),
    )


def _record_grade(
    run: QualityRun,
    suite: SuiteSpec,
    task_id: str,
    rep: int,
    side: str,
    grade: TaskGrade,
) -> None:
    payload = _grade_dict(grade)
    directory = run.task_dir(suite.name, task_id, rep, side)
    run.write_json(str(directory.relative_to(run.dir) / "grade.json"), payload)
    run.append(
        {
            "type": "task_result",
            "suite_id": suite.suite_id,
            "task_id": task_id,
            "rep": rep,
            "side": side,
            "grade": payload,
        }
    )


def _launch(run: QualityRun, argv: tuple[str, ...], timeout_s: float) -> ServerHandle:
    errors: list[str] = []
    for _ in range(2):
        try:
            return start(run, argv, start_timeout_s=timeout_s)
        except ServerStartError as exc:
            errors.append(str(exc))
    raise ServerStartError("; ".join(errors))


def _evaluate_http_side(
    run: QualityRun,
    resolved: _Resolved,
    config: TrialConfig | None,
    side: str,
    prior: dict[tuple[str, str, int, str], TaskGrade],
    signals: _SignalState,
) -> tuple[tuple[SuiteResult, ...], bool, bool]:
    from llamatune.qualscore import aggregate, combine_repetitions, grade_task

    suites = tuple(suite for suite in resolved.suites if suite.kind != "perplexity")
    if not suites:
        return (), False, False
    pending = any(
        _task_selected(str(task["id"]), resolved.options.task_filters)
        and any(
            _task_key(suite.suite_id, str(task["id"]), rep, side) not in prior
            for rep in range(1, resolved.options.reps + 1)
        )
        for suite in suites
        for task in suite.tasks
    )
    handle: ServerHandle | None = None
    results: list[SuiteResult] = []
    harness_error = False
    abort = False
    try:
        if pending:
            handle = _launch(
                run,
                _server_argv(resolved, config),
                resolved.options.server_start_timeout_s,
            )
            signals.handle = handle
        for suite in suites:
            grades: list[TaskGrade] = []
            relaunch_used = False
            tasks = [
                task
                for task in suite.tasks
                if _task_selected(str(task["id"]), resolved.options.task_filters)
            ]
            for task_index, task in enumerate(tasks):
                task_id = str(task["id"])
                rep_grades: list[TaskGrade] = []
                deferred_exec: list[tuple[int, tuple[str, str, int, str], tuple[str, ...]]] = []
                needs_exec = resolved.options.exec_enabled and any(
                    isinstance(grader, dict) and grader.get("type") == "exec_python"
                    for grader in task.get("graders", ())
                )
                for rep in range(1, resolved.options.reps + 1):
                    key = _task_key(suite.suite_id, task_id, rep, side)
                    if key in prior:
                        rep_grades.append(prior[key])
                        continue
                    run.append(
                        {
                            "type": "task_start",
                            "suite_id": suite.suite_id,
                            "task_id": task_id,
                            "rep": rep,
                            "side": side,
                        }
                    )
                    ctx_min = task.get("ctx_min")
                    if isinstance(ctx_min, int) and resolved.options.ctx_size < ctx_min:
                        grade = _skipped_grade(
                            task_id,
                            f"ctx_size {resolved.options.ctx_size} is below ctx_min {ctx_min}",
                        )
                        _record_grade(run, suite, task_id, rep, side, grade)
                        prior[key] = grade
                        rep_grades.append(grade)
                        continue
                    current_grade: TaskGrade | None = None
                    while True:
                        try:
                            if handle is None:
                                handle = _launch(
                                    run,
                                    _server_argv(resolved, config),
                                    resolved.options.server_start_timeout_s,
                                )
                                signals.handle = handle
                            responses = _drive_task(
                                run,
                                handle,
                                suite,
                                task,
                                rep,
                                side,
                                resolved.options,
                            )
                            if needs_exec:
                                deferred_exec.append((rep, key, responses))
                            else:
                                current_grade = grade_task(task, responses, None)
                        except ServerUnavailableError as exc:
                            run.append(
                                {
                                    "type": "server_exit",
                                    "suite_id": suite.suite_id,
                                    "reason": str(exc),
                                    "stderr_tail": handle.stderr_tail if handle else "",
                                }
                            )
                            if handle is not None:
                                handle.stop()
                                handle = None
                                signals.handle = None
                            if relaunch_used:
                                current_grade = _error_grade(task_id, "server_unavailable")
                                harness_error = True
                                abort = True
                                break
                            relaunch_used = True
                            try:
                                handle = start(
                                    run,
                                    _server_argv(resolved, config),
                                    start_timeout_s=resolved.options.server_start_timeout_s,
                                )
                                signals.handle = handle
                            except ServerStartError:
                                current_grade = _error_grade(task_id, "server_unavailable")
                                harness_error = True
                                abort = True
                                break
                            continue
                        except (ServerProtocolError, OSError, ValueError) as exc:
                            current_grade = _error_grade(task_id, str(exc))
                            harness_error = True
                        break
                    if current_grade is not None:
                        _record_grade(run, suite, task_id, rep, side, current_grade)
                        prior[key] = current_grade
                        rep_grades.append(current_grade)
                    if abort:
                        break
                    if signals.stop_requested:
                        break
                if abort:
                    deferred_exec.clear()
                    for remaining_rep in range(1, resolved.options.reps + 1):
                        remaining_key = _task_key(suite.suite_id, task_id, remaining_rep, side)
                        if remaining_key in prior:
                            continue
                        grade = _error_grade(task_id, "server_unavailable")
                        _record_grade(run, suite, task_id, remaining_rep, side, grade)
                        prior[remaining_key] = grade
                    rep_grades = [
                        prior[_task_key(suite.suite_id, task_id, rep, side)]
                        for rep in range(1, resolved.options.reps + 1)
                    ]
                if deferred_exec and not abort:
                    if handle is not None:
                        handle.stop()
                        handle = None
                        signals.handle = None
                    from llamatune.sandbox import run_python

                    def exec_runner(code: str) -> Any:
                        return run_python(
                            code,
                            timeout_s=resolved.options.request_timeout_s,
                        )

                    for rep, key, responses in deferred_exec:
                        try:
                            grade = grade_task(task, responses, exec_runner)
                        except (OSError, RuntimeError, ValueError) as exc:
                            grade = _error_grade(task_id, str(exc))
                            harness_error = True
                        _record_grade(run, suite, task_id, rep, side, grade)
                        prior[key] = grade
                        rep_grades.append(grade)
                if rep_grades:
                    grades.append(combine_repetitions(tuple(rep_grades)))
                if abort or signals.stop_requested:
                    if abort:
                        for remaining in tasks[task_index + 1 :]:
                            remaining_id = str(remaining["id"])
                            remaining_grades: list[TaskGrade] = []
                            for remaining_rep in range(1, resolved.options.reps + 1):
                                remaining_key = _task_key(
                                    suite.suite_id,
                                    remaining_id,
                                    remaining_rep,
                                    side,
                                )
                                remaining_grade = prior.get(remaining_key)
                                if remaining_grade is None:
                                    remaining_grade = _error_grade(
                                        remaining_id, "server_unavailable"
                                    )
                                    _record_grade(
                                        run,
                                        suite,
                                        remaining_id,
                                        remaining_rep,
                                        side,
                                        remaining_grade,
                                    )
                                    prior[remaining_key] = remaining_grade
                                remaining_grades.append(remaining_grade)
                            grades.append(combine_repetitions(tuple(remaining_grades)))
                    break
            metrics = aggregate(
                suite.kind,
                tuple(grades),
                exec_enabled=resolved.options.exec_enabled,
            )
            result = SuiteResult(
                suite_id=suite.suite_id,
                name=suite.name,
                kind=suite.kind,
                metrics=metrics,
                tasks=tuple(grades),
            )
            results.append(result)
            run.append(
                {
                    "type": "suite_summary",
                    "side": side,
                    "suite_id": suite.suite_id,
                    "metrics": metrics,
                }
            )
            if abort or signals.stop_requested:
                break
    finally:
        if handle is not None:
            handle.stop()
        signals.handle = None
    return tuple(results), harness_error, abort


def _perplexity_result(
    run: QualityRun,
    resolved: _Resolved,
    config: TrialConfig | None,
    side: str,
) -> SuiteResult | None:
    suite = next((item for item in resolved.suites if item.kind == "perplexity"), None)
    if suite is None:
        return None
    if resolved.llama.perplexity_path is None or resolved.options.quality_corpus is None:
        raise RuntimeError("perplexity suite was not fully resolved")
    directory = run.perplexity_dir(side)
    from llamatune.bench import build_perplexity_argv, parse_perplexity_output

    argv = build_perplexity_argv(
        perplexity_path=resolved.llama.perplexity_path,
        model_path=resolved.model.path,
        config=config,
        corpus=resolved.options.quality_corpus,
        ctx=resolved.options.ctx_size,
        capabilities=resolved.llama.capabilities,
    )
    run.write_json(
        str(directory.relative_to(run.dir) / "command.json"),
        {"argv": argv, "env_names": sorted(executor.build_child_env())},
    )
    try:
        result = executor.run(
            argv,
            timeout_s=resolved.options.request_timeout_s,
            stdout_path=directory / "stdout",
            stderr_path=directory / "stderr",
        )
        text = (directory / "stdout").read_bytes() + (directory / "stderr").read_bytes()
        ppl = parse_perplexity_output(text)
        if (
            result.exit_code != 0
            or result.timed_out
            or ppl is None
            or not math.isfinite(ppl)
            or ppl <= 0
        ):
            raise ValueError("llama-perplexity did not produce a usable perplexity value")
        grade = TaskGrade(
            task_id="perplexity",
            score=0.0,
            status="graded",
            reason=None,
            unstable=False,
            grader_results=(),
        )
        metrics = {"ppl": ppl}
    except (OSError, ValueError) as exc:
        grade = _error_grade("perplexity", str(exc))
        metrics = {}
    return SuiteResult(
        suite_id=suite.suite_id,
        name=suite.name,
        kind=suite.kind,
        metrics=metrics,
        tasks=(grade,),
    )


def _suite_dict(result: SuiteResult) -> dict[str, Any]:
    return {
        "suite_id": result.suite_id,
        "name": result.name,
        "kind": result.kind,
        "metrics": result.metrics,
        "tasks": [_grade_dict(grade) for grade in result.tasks],
    }


def _comparison(
    evaluated: tuple[SuiteResult, ...],
    lossless: tuple[SuiteResult, ...],
    lossless_config: TrialConfig,
) -> dict[str, Any]:
    by_id = {suite.suite_id: suite for suite in lossless}
    suites: list[dict[str, Any]] = []
    degradation = False
    for suite in evaluated:
        other = by_id.get(suite.suite_id)
        if other is None:
            continue
        if (
            suite.kind == "perplexity"
            and "ppl" in suite.metrics
            and isinstance(other.metrics.get("ppl"), (int, float))
            and other.metrics["ppl"] > 0
        ):
            suite.metrics["delta_pct"] = (suite.metrics["ppl"] / other.metrics["ppl"] - 1.0) * 100.0
        delta_score = suite.metrics.get("score", 0.0) - other.metrics.get("score", 0.0)
        if suite.kind in {"coding", "tooluse"} and delta_score < -0.05:
            degradation = True
        suites.append(
            {
                "suite_id": suite.suite_id,
                "evaluated": suite.metrics,
                "lossless": other.metrics,
                "delta_score": delta_score,
            }
        )
    return {
        "lossless_config": lossless_config.to_dict(),
        "suites": suites,
        "degradation_warning": degradation,
    }


def _summary(
    run: QualityRun,
    resolved: _Resolved,
    evaluated: tuple[SuiteResult, ...],
    comparison: dict[str, Any] | None,
    warnings: Sequence[str],
    exec_isolation: str | None = None,
) -> dict[str, Any]:
    scores = [suite.metrics["score"] for suite in evaluated if "score" in suite.metrics]
    summary = {
        "schema_version": _SCHEMA_VERSION,
        "run_dir": str(run.dir),
        "created": str(run.metadata["created"]),
        "model": {
            "fingerprint": resolved.model.fingerprint,
            "name": resolved.model.name,
            "path": str(resolved.model.path),
            "size_bytes": resolved.model.size_bytes,
        },
        "hardware_signature": _jsonable(hardware_signature(resolved.hardware)),
        "build": {
            "bench_sha256": resolved.llama.bench_sha256,
            "help_sha256": resolved.llama.help_sha256,
            "build_commit": resolved.llama.build_commit,
        },
        "ctx": resolved.options.ctx_size,
        "seed": resolved.options.seed,
        "reps": resolved.options.reps,
        "config": resolved.config.to_dict() if resolved.config is not None else None,
        "config_source": resolved.config_source,
        "config_provenance": (
            str(resolved.config_provenance) if resolved.config_provenance is not None else None
        ),
        "exec_enabled": resolved.options.exec_enabled,
        "suites": [_suite_dict(suite) for suite in evaluated],
        "overall": sum(scores) / len(scores) if scores else 0.0,
        "comparison": comparison,
        "warnings": list(warnings),
    }
    if exec_isolation is not None:
        summary["exec_isolation"] = exec_isolation
    return summary


def _refresh_matrix(root: Path) -> None:
    try:
        module = importlib.import_module("llamatune.resultsmatrix")
    except ImportError:
        return
    try:
        module.refresh(root)
    except Exception as exc:  # Phase 4 refresh can never alter the quality result
        print(f"warning: results matrix refresh failed: {exc}", file=sys.stderr)


def _phase4(
    run: QualityRun,
    resolved: _Resolved,
    evaluated: tuple[SuiteResult, ...],
    comparison: dict[str, Any] | None,
    warnings: list[str],
    exit_code: int,
    exec_isolation: str | None = None,
) -> QualityOutcome:
    if comparison is not None and comparison.get("degradation_warning"):
        warnings.append("lossy cache measurably degrades quality on this machine")
    summary = _summary(run, resolved, evaluated, comparison, warnings, exec_isolation)
    run.write_json("quality.json", summary)
    from llamatune.qualityreport import render

    run.write_text("quality-report.md", render(summary))
    run.append({"type": "quality_end", "exit_code": exit_code})
    if exit_code in {0, 1, 4}:
        _refresh_matrix(resolved.options.sessions_dir)
    return QualityOutcome(run_dir=run.dir, summary=summary, exit_code=exit_code)


def _dry_run(resolved: _Resolved) -> QualityOutcome:
    task_count = sum(
        1
        for suite in resolved.suites
        for task in suite.tasks
        if _task_selected(str(task["id"]), resolved.options.task_filters)
    )
    request_count = sum(
        (
            sum(1 for turn in task.get("turns", ()) if "user" in turn)
            if suite.kind == "tooluse"
            else int(task.get("environment", {}).get("max_steps", 1))
            if suite.kind == "agentic"
            else 1
        )
        * resolved.options.reps
        for suite in resolved.suites
        for task in suite.tasks
        if _task_selected(str(task["id"]), resolved.options.task_filters)
    )
    request_count *= 2 if resolved.lossless_config is not None else 1
    plan = {
        "dry_run": True,
        "config": resolved.config.to_dict() if resolved.config else None,
        "config_source": resolved.config_source,
        "config_provenance": (
            str(resolved.config_provenance) if resolved.config_provenance else None
        ),
        "exec_enabled": resolved.options.exec_enabled,
        "task_count": task_count,
        "estimated_requests": request_count,
        "suites": [
            {
                "suite_id": suite.suite_id,
                "name": suite.name,
                "tasks": sum(
                    _task_selected(str(task["id"]), resolved.options.task_filters)
                    for task in suite.tasks
                ),
            }
            for suite in resolved.suites
        ],
        "server_argv": (
            list(_server_argv(resolved, resolved.config, port="<ephemeral>"))
            if any(suite.kind != "perplexity" for suite in resolved.suites)
            else None
        ),
        "warnings": list(resolved.warnings),
    }
    if resolved.options.exec_enabled:
        from llamatune.sandbox import describe_isolation

        plan["exec_isolation"] = describe_isolation().summary
    return QualityOutcome(run_dir=Path(), summary=plan, exit_code=0)


def _execute(
    resolved: _Resolved,
    *,
    run: QualityRun | None,
    now_fn: Callable[[], Any] | None,
) -> QualityOutcome:
    if resolved.options.dry_run:
        return _dry_run(resolved)
    if run is None:
        run = QualityRun.create(
            resolved.options.sessions_dir,
            options=resolved.options,
            hardware=resolved.hardware,
            llama=resolved.llama,
            model=resolved.model,
            argv=list(sys.argv),
        )
        run._update_metadata(
            {
                "suite_ids": [suite.suite_id for suite in resolved.suites],
                "config": resolved.config.to_dict() if resolved.config else None,
                "lossless_config": (
                    resolved.lossless_config.to_dict() if resolved.lossless_config else None
                ),
                "config_source": resolved.config_source,
                "config_provenance": (
                    str(resolved.config_provenance) if resolved.config_provenance else None
                ),
            }
        )
        run.write_json(
            "config.json",
            {
                "config": resolved.config.to_dict() if resolved.config else None,
                "source": resolved.config_source,
                "provenance": (
                    str(resolved.config_provenance) if resolved.config_provenance else None
                ),
            },
        )
        run.write_json(
            "plan.json",
            {
                "suites": [
                    {"suite_id": suite.suite_id, "tasks": len(suite.tasks)}
                    for suite in resolved.suites
                ],
                "sides": ["evaluated"]
                + (["lossless"] if resolved.lossless_config is not None else []),
                "server_argv": (
                    list(_server_argv(resolved, resolved.config, port="<ephemeral>"))
                    if any(suite.kind != "perplexity" for suite in resolved.suites)
                    else None
                ),
            },
        )
        run.append(
            {
                "type": "config_resolved",
                "source": resolved.config_source,
                "provenance": (
                    str(resolved.config_provenance) if resolved.config_provenance else None
                ),
            }
        )
        run.append({"type": "plan", "suite_ids": [s.suite_id for s in resolved.suites]})
    prior = _journaled_grades(run)
    signals = _SignalState()
    old_handlers: dict[signal.Signals, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(ValueError):
            old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, signals.receive)
    warnings = list(resolved.warnings)
    exec_isolation: str | None = None
    if resolved.options.exec_enabled:
        from llamatune.sandbox import describe_isolation

        report = describe_isolation()
        exec_isolation = report.summary
        warnings.extend(report.warnings)
        if report.warnings:
            print(
                f"WARNING: quality --exec degraded isolation: {'; '.join(report.warnings)}",
                file=sys.stderr,
            )
    evaluated: tuple[SuiteResult, ...] = ()
    lossless: tuple[SuiteResult, ...] = ()
    comparison = None
    harness_error = False
    interrupted = False
    from llamatune import qualserver as qualserver_module

    previous_clock = qualserver_module._swap_monotonic(now_fn) if now_fn is not None else None
    try:
        evaluated, failed, abort = _evaluate_http_side(
            run,
            resolved,
            resolved.config,
            "evaluated",
            prior,
            signals,
        )
        harness_error |= failed
        if not abort and not signals.stop_requested:
            perplexity = _perplexity_result(run, resolved, resolved.config, "evaluated")
            if perplexity is not None:
                evaluated = (*evaluated, perplexity)
                harness_error |= any(task.status == "error" for task in perplexity.tasks)
        if not abort and resolved.lossless_config is not None and not signals.stop_requested:
            lossless, failed, abort = _evaluate_http_side(
                run,
                resolved,
                resolved.lossless_config,
                "lossless",
                prior,
                signals,
            )
            harness_error |= failed
            if not abort:
                perplexity = _perplexity_result(
                    run,
                    resolved,
                    resolved.lossless_config,
                    "lossless",
                )
                if perplexity is not None:
                    lossless = (*lossless, perplexity)
                    harness_error |= any(task.status == "error" for task in perplexity.tasks)
                comparison = _comparison(evaluated, lossless, resolved.lossless_config)
                run.append({"type": "comparison", "comparison": comparison})
        interrupted = signals.stop_requested
    except _ForcedInterrupt:
        interrupted = True
    except ServerStartError as exc:
        warnings.append(str(exc))
        return _phase4(run, resolved, evaluated, comparison, warnings, 3, exec_isolation)
    finally:
        if previous_clock is not None:
            qualserver_module._swap_monotonic(previous_clock)
        if signals.handle is not None:
            signals.handle.stop()
            signals.handle = None
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    if interrupted:
        run.append({"type": "interrupted", "forced": signals.forced})
        return _phase4(run, resolved, evaluated, comparison, warnings, 4, exec_isolation)
    return _phase4(
        run,
        resolved,
        evaluated,
        comparison,
        warnings,
        1 if harness_error else 0,
        exec_isolation,
    )


def run_quality(
    options: QualityOptions,
    *,
    now_fn: Callable[[], Any] | None = None,
) -> QualityOutcome:
    """Resolve and execute one deterministic quality evaluation."""
    try:
        resolved = _resolve(options)
    except ValueError as exc:
        return QualityOutcome(run_dir=Path(), summary={"error": str(exc)}, exit_code=2)
    except (LlamaDiscoveryError, ModelInspectionError, OSError, RuntimeError) as exc:
        return QualityOutcome(run_dir=Path(), summary={"error": str(exc)}, exit_code=3)
    return _execute(resolved, run=None, now_fn=now_fn)


def resume_quality(
    run_dir: Path,
    *,
    llama_bin: Path | None = None,
    now_fn: Callable[[], Any] | None = None,
) -> QualityOutcome:
    """Resume one identity-bound run, skipping exact journaled task tuples."""
    try:
        run = QualityRun.load(run_dir)
        options_value = run.metadata.get("options")
        if not isinstance(options_value, dict):
            raise ValueError("run.json has no options object")
        options = _options_from_dict(options_value)
        options = replace(
            options,
            llama_bin=llama_bin if llama_bin is not None else options.llama_bin,
            dry_run=False,
        )
        identity_options = replace(
            options,
            config_mode="defaults",
            config_session=None,
            strict_config=False,
            compare_lossless=False,
        )
        resolved = _resolve(identity_options)
        if run.metadata.get("model_fingerprint") != resolved.model.fingerprint:
            raise RuntimeError("quality resume model fingerprint changed")
        if run.metadata.get("build_discriminator") != _build_discriminator(resolved.llama):
            raise RuntimeError("quality resume llama.cpp build changed")
        recorded_config = run.metadata.get("config")
        config = (
            TrialConfig.from_dict(recorded_config) if isinstance(recorded_config, dict) else None
        )
        recorded_lossless = run.metadata.get("lossless_config")
        provenance = run.metadata.get("config_provenance")
        resolved = replace(
            resolved,
            options=options,
            config=config,
            config_source=str(run.metadata.get("config_source", "defaults")),
            config_provenance=Path(provenance) if isinstance(provenance, str) else None,
            lossless_config=(
                TrialConfig.from_dict(recorded_lossless)
                if options.compare_lossless and isinstance(recorded_lossless, dict)
                else _lossless(config)
                if options.compare_lossless
                and config is not None
                and (config.cache_type_k != "f16" or config.cache_type_v != "f16")
                else None
            ),
        )
    except ValueError as exc:
        return QualityOutcome(run_dir=Path(run_dir), summary={"error": str(exc)}, exit_code=2)
    except (LlamaDiscoveryError, ModelInspectionError, OSError, RuntimeError) as exc:
        return QualityOutcome(run_dir=Path(run_dir), summary={"error": str(exc)}, exit_code=3)
    return _execute(resolved, run=run, now_fn=now_fn)
