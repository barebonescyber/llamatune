"""Typer CLI application (DESIGN §3).

Thin by design: no logic beyond argument handling and output formatting.
Heavy imports (hardware/llama/model/session/search/report) are lazy,
inside each command body, so `llamatune --help` stays fast.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import math
import os
import sys
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

if TYPE_CHECKING:
    from llamatune.types import HardwareReport, LlamaCppReport, TuneOutcome

app = typer.Typer(
    name="llamatune",
    help="Find the fastest llama.cpp runtime settings for one GGUF model.",
    add_completion=False,
    no_args_is_help=True,
)
matrix_app = typer.Typer(help="Build, query, show, and export the Results Matrix.")
app.add_typer(matrix_app, name="matrix")

_MATRIX_KINDS = frozenset(
    {
        "baseline",
        "recommendation",
        "lossless_recommendation",
        "context_envelope",
        "depth_profile",
        "operating_point",
        "ab_verification",
        "calibration",
        "quality_suite",
    }
)


class Target(StrEnum):
    balanced = "balanced"
    prompt = "prompt"
    generation = "generation"


class ProgressMode(StrEnum):
    auto = "auto"
    plain = "plain"
    rich = "rich"
    json = "json"
    none = "none"


def _make_reporter(mode: ProgressMode, *, tui: bool, quiet: bool) -> Any:
    """Resolve display aliases and lazily construct the progress reporter."""
    if tui and quiet:
        typer.echo("error: --tui and --quiet are mutually exclusive", err=True)
        raise typer.Exit(code=2)
    resolved = ProgressMode.rich if tui else ProgressMode.none if quiet else mode
    from llamatune import ui

    return ui.make_reporter(resolved.value)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, frozenset):
        return sorted(obj)
    msg = f"object of type {type(obj).__name__} is not JSON serializable"
    raise TypeError(msg)


def _echo_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _matrix_roots(sessions_dirs: list[Path] | None) -> tuple[Path, ...]:
    roots = tuple(sessions_dirs or (Path("./llamatune-sessions"),))
    for root in roots:
        if not root.is_dir() or not os.access(root, os.R_OK):
            typer.echo(f"error: sessions root is missing or unreadable: {root}", err=True)
            raise typer.Exit(code=3)
    return roots


def _refresh_results_matrix(root: Path, exit_code: int) -> None:
    if exit_code not in {0, 1, 4}:
        return
    try:
        from llamatune.resultsmatrix import refresh

        refresh(root)
    except Exception as exc:  # the owning command's result always wins
        typer.echo(f"warning: results matrix refresh failed: {exc}", err=True)


def _matrix_identity(llama_bin: Path) -> tuple[str, str]:
    from llamatune.config import hardware_signature
    from llamatune.hardware import assess_hardware
    from llamatune.llama import discover_llama

    llama = discover_llama(llama_bin)
    signature = hardware_signature(assess_hardware())
    canonical = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    hardware_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    discriminator = llama.bench_sha256 or llama.help_sha256
    return hardware_hash, discriminator


def _matrix_document(matrix: Any) -> dict[str, Any]:
    return {
        "schema_version": matrix.schema_version,
        "generated": matrix.generated,
        "roots": [str(root) for root in matrix.roots],
        "row_count": len(matrix.rows),
        "model_count": len({row.model_fingerprint for row in matrix.rows}),
        "rows": [row.to_dict() for row in matrix.rows],
        "warnings": list(matrix.warnings),
    }


def _matrix_config_summary(config: Any) -> str:
    if not isinstance(config, dict):
        return "defaults"
    return "/".join(
        (
            f"ngl={config.get('gpu_layers', '-')}",
            f"ncmoe={config.get('moe_cpu_layers', '-')}",
            f"fa={int(bool(config.get('flash_attn', False)))}",
            f"ub={config.get('ubatch', '-')}",
            f"b={config.get('batch', '-')}",
            f"t={config.get('threads', '-')}",
        )
    )


def _matrix_rank_value(payload: dict[str, Any], row: dict[str, Any]) -> Any:
    metrics = row.get("metrics", {})
    sort = payload.get("sort")
    if isinstance(sort, str):
        return metrics.get(sort)
    use_case = payload.get("use_case")
    keys = {
        "max-pp": "perf.pp",
        "max-tg": "perf.tg",
        "max-context": "ctx.validated",
        "coding": "quality.coding.score",
        "tool-use": "quality.tooluse.score",
        "agentic": "quality.agentic.score",
        "instruction": "quality.ifollow.score",
        "quality-overall": "quality.overall",
    }
    if use_case == "balanced":
        pp = metrics.get("perf.pp")
        tg = metrics.get("perf.tg")
        if isinstance(pp, (int, float)) and isinstance(tg, (int, float)):
            return (pp * tg) ** 0.5
        return None
    key = keys.get(use_case) if isinstance(use_case, str) else None
    return metrics.get(key) if key is not None else None


def _matrix_display(value: Any) -> Any:
    return "-" if value is None else value


def _echo_matrix_query(payload: dict[str, Any]) -> None:
    groups = payload.get("groups", [])
    count = sum(len(group.get("rows", [])) for group in groups)
    if count == 0:
        typer.echo("0 rows")
    for group in groups:
        typer.echo(f"hardware: {group.get('hardware_hash', '-')}")
        typer.echo("rank  model  config  ctx  depth  metric  pp  tg  flags  compat  evidence")
        for row in group.get("rows", []):
            metrics = row.get("metrics", {})
            ranking = _matrix_rank_value(payload, row)
            flags = ("C" if row.get("confirmed") else "-") + ("R" if row.get("replicated") else "-")
            model = row.get("model_name") or Path(str(row.get("model_path", "-"))).stem
            if row.get("quant"):
                model = f"{model} ({row['quant']})"
            typer.echo(
                f"{row.get('rank', '-')}  {model}  "
                f"{_matrix_config_summary(row.get('config'))}  "
                f"{_matrix_display(row.get('ctx'))}  {_matrix_display(row.get('depth'))}  "
                f"{_matrix_display(ranking)}  "
                f"{_matrix_display(metrics.get('perf.pp'))}  "
                f"{_matrix_display(metrics.get('perf.tg'))}  "
                f"{flags}  {row.get('compat', 'unknown')}  {row.get('evidence_dir', '-')}"
            )
        if group.get("rows"):
            top = group["rows"][0]
            if top.get("kind") == "recommendation":
                typer.echo(f"reproduce: {Path(top['evidence_dir']) / 'recommended.sh'}")
    excluded = payload.get("excluded", {})
    if excluded:
        typer.echo("excluded: " + ", ".join(f"{key}={value}" for key, value in excluded.items()))
    for warning in payload.get("warnings", []):
        typer.echo(f"warning: {warning}", err=True)


def _matrix_show_payload(matrix: Any) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    for row in matrix.rows:
        item = models.setdefault(
            row.model_fingerprint,
            {
                "model_fingerprint": row.model_fingerprint,
                "model_name": row.model_name,
                "kinds": set(),
                "best_pp": None,
                "best_tg": None,
                "best_balanced": None,
                "max_validated_ctx": None,
                "quality_suites": set(),
                "newest_ts": row.ts,
                "stale_identity": False,
            },
        )
        item["kinds"].add(row.kind)
        item["newest_ts"] = max(item["newest_ts"], row.ts)
        item["stale_identity"] = item["stale_identity"] or not row.current
        if row.suite_id is not None:
            item["quality_suites"].add(row.suite_id)
        if not row.current:
            continue
        pp = row.metrics.get("perf.pp")
        tg = row.metrics.get("perf.tg")
        if pp is not None:
            item["best_pp"] = pp if item["best_pp"] is None else max(item["best_pp"], pp)
        if tg is not None:
            item["best_tg"] = tg if item["best_tg"] is None else max(item["best_tg"], tg)
        if pp is not None and tg is not None and pp > 0 and tg > 0:
            balanced = (pp * tg) ** 0.5
            item["best_balanced"] = (
                balanced if item["best_balanced"] is None else max(item["best_balanced"], balanced)
            )
        validated = row.metrics.get("ctx.validated")
        if validated is not None:
            value = int(validated)
            item["max_validated_ctx"] = (
                value
                if item["max_validated_ctx"] is None
                else max(item["max_validated_ctx"], value)
            )
    normalized = []
    for item in models.values():
        item["kinds"] = sorted(item["kinds"])
        item["quality_suites"] = sorted(item["quality_suites"])
        normalized.append(item)
    normalized.sort(key=lambda item: (item["model_name"] or "", item["model_fingerprint"]))
    return {
        "rows": len(matrix.rows),
        "models": normalized,
        "warnings": list(matrix.warnings),
    }


@matrix_app.command("build")
def matrix_build(
    sessions_dirs: Annotated[list[Path] | None, typer.Option("--sessions-dir")] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Harvest evidence and materialize deterministic matrix artifacts."""
    from llamatune.resultsmatrix import build

    roots = _matrix_roots(sessions_dirs)
    destination = output or roots[0] / "matrix"
    try:
        summary = build(roots, destination)
    except (OSError, ValueError) as exc:
        typer.echo(f"error: could not write results matrix: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    if json_output:
        _echo_json(summary)
    else:
        typer.echo(
            f"matrix: {summary['rows']} rows, {summary['models']} models, "
            f"{len(summary['roots'])} roots, {len(summary['warnings'])} warnings"
        )
        for warning in summary["warnings"]:
            typer.echo(f"warning: {warning}", err=True)


@matrix_app.command("query")
def matrix_query(
    sessions_dirs: Annotated[list[Path] | None, typer.Option("--sessions-dir")] = None,
    use_case: Annotated[str | None, typer.Option("--use-case")] = None,
    sort: Annotated[str | None, typer.Option("--sort")] = None,
    ascending: Annotated[bool, typer.Option("--ascending")] = False,
    model: Annotated[str | None, typer.Option("--model")] = None,
    quant: Annotated[str | None, typer.Option("--quant")] = None,
    kinds: Annotated[list[str] | None, typer.Option("--kind")] = None,
    suite: Annotated[str | None, typer.Option("--suite")] = None,
    ctx: Annotated[int | None, typer.Option("--ctx")] = None,
    depth: Annotated[int | None, typer.Option("--depth")] = None,
    min_pp: Annotated[float | None, typer.Option("--min-pp")] = None,
    min_tg: Annotated[float | None, typer.Option("--min-tg")] = None,
    confirmed_only: Annotated[bool, typer.Option("--confirmed-only/--include-unconfirmed")] = True,
    current_only: Annotated[bool, typer.Option("--current-only/--include-superseded")] = True,
    compat: Annotated[str, typer.Option("--compat")] = "any",
    llama_bin: Annotated[Path | None, typer.Option("--llama-bin")] = None,
    limit: Annotated[int, typer.Option("--limit")] = 10,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Filter and rank freshly harvested matrix rows."""
    if use_case is not None and sort is not None:
        typer.echo("error: --use-case and --sort are mutually exclusive", err=True)
        raise typer.Exit(code=2)
    if use_case is None and sort is None:
        typer.echo("error: one of --use-case or --sort is required", err=True)
        raise typer.Exit(code=2)
    unknown_kinds = sorted(set(kinds or ()) - _MATRIX_KINDS)
    if unknown_kinds:
        typer.echo(f"error: unknown kind: {unknown_kinds[0]}", err=True)
        raise typer.Exit(code=2)
    if compat not in {"any", "current"}:
        typer.echo("error: --compat must be 'any' or 'current'", err=True)
        raise typer.Exit(code=2)
    if compat == "current" and llama_bin is None:
        typer.echo("error: --compat current requires --llama-bin", err=True)
        raise typer.Exit(code=2)
    if limit < 0:
        typer.echo("error: --limit must be >= 0", err=True)
        raise typer.Exit(code=2)
    roots = _matrix_roots(sessions_dirs)

    from llamatune import matrixquery, resultsmatrix
    from llamatune.llama import LlamaDiscoveryError
    from llamatune.types import MatrixQuerySpec

    if use_case is not None and use_case not in matrixquery.USE_CASES:
        typer.echo(f"error: unknown use case: {use_case}", err=True)
        raise typer.Exit(code=2)
    identity = None
    if llama_bin is not None:
        try:
            identity = _matrix_identity(llama_bin)
        except (LlamaDiscoveryError, OSError, ValueError) as exc:
            typer.echo(f"error: could not probe current identity: {exc}", err=True)
            raise typer.Exit(code=3) from exc
    spec = MatrixQuerySpec(
        use_case=use_case,
        sort=sort,
        ascending=ascending,
        model=model,
        quant=quant,
        kinds=tuple(kinds or ()),
        suite=suite,
        ctx_min=ctx,
        depth=depth,
        min_pp=min_pp,
        min_tg=min_tg,
        confirmed_only=confirmed_only,
        current_only=current_only,
        compat=compat,
        limit=limit,
    )
    payload = matrixquery.apply(resultsmatrix.harvest(roots), spec, identity)
    if json_output:
        _echo_json(payload)
    else:
        _echo_matrix_query(payload)


@matrix_app.command("show")
def matrix_show(
    sessions_dirs: Annotated[list[Path] | None, typer.Option("--sessions-dir")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show matrix coverage summarized per model."""
    from llamatune.resultsmatrix import harvest

    payload = _matrix_show_payload(harvest(_matrix_roots(sessions_dirs)))
    if json_output:
        _echo_json(payload)
        return
    if not payload["models"]:
        typer.echo("0 rows")
    else:
        typer.echo("model  kinds  best_pp  best_tg  balanced  max_ctx  quality  newest  stale")
        for item in payload["models"]:
            typer.echo(
                f"{item['model_name'] or item['model_fingerprint']}  "
                f"{','.join(item['kinds']) or '-'}  "
                f"{_matrix_display(item['best_pp'])}  "
                f"{_matrix_display(item['best_tg'])}  "
                f"{_matrix_display(item['best_balanced'])}  "
                f"{_matrix_display(item['max_validated_ctx'])}  "
                f"{','.join(item['quality_suites']) or '-'}  {item['newest_ts']}  "
                f"{'yes' if item['stale_identity'] else 'no'}"
            )
    for warning in payload["warnings"]:
        typer.echo(f"warning: {warning}", err=True)


@matrix_app.command("export")
def matrix_export(
    export_format: Annotated[str, typer.Option("--format")],
    sessions_dirs: Annotated[list[Path] | None, typer.Option("--sessions-dir")] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Export freshly harvested rows as JSON, CSV, or Markdown."""
    if export_format not in {"json", "csv", "md"}:
        typer.echo("error: --format must be json, csv, or md", err=True)
        raise typer.Exit(code=2)
    roots = _matrix_roots(sessions_dirs)

    from llamatune import matrixreport
    from llamatune.resultsmatrix import harvest

    matrix = harvest(roots)
    if export_format == "json":
        text = (
            json.dumps(_matrix_document(matrix), indent=2, sort_keys=True, default=_json_default)
            + "\n"
        )
    elif export_format == "csv":
        text = matrixreport.render_csv(matrix.rows)
    else:
        text = matrixreport.render_markdown(matrix)
    if output is None:
        typer.echo(text, nl=False)
    else:
        try:
            output.write_text(text, encoding="utf-8", newline="\n")
        except OSError as exc:
            typer.echo(f"error: could not write matrix export: {exc}", err=True)
            raise typer.Exit(code=3) from exc
    for warning in matrix.warnings:
        typer.echo(f"warning: {warning}", err=True)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def _parse_auto_nonnegative(value: str | None, option: str) -> int | None:
    if value is None or value.lower() == "auto":
        return None
    try:
        parsed = int(value)
    except ValueError:
        raise ValueError(f"{option} must be a non-negative integer or 'auto'") from None
    if parsed < 0:
        raise ValueError(f"{option} must be >= 0 or 'auto'")
    return parsed


def _parse_depth_profile(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError:
        raise ValueError("--depth-profile must be a comma-separated list of integers") from None
    if not parsed or any(item < 0 for item in parsed):
        raise ValueError("--depth-profile values must be >= 0")
    if tuple(sorted(set(parsed))) != parsed:
        raise ValueError("--depth-profile values must be distinct and ascending")
    return parsed


def _parse_depth_grid(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError:
        raise ValueError("--depth-grid must be a comma-separated list of integers") from None
    if not parsed or any(item < 0 for item in parsed):
        raise ValueError("--depth-grid values must be >= 0")
    if tuple(sorted(set(parsed))) != parsed:
        raise ValueError("--depth-grid values must be distinct and ascending")
    return parsed


def _parse_ctx_ladder(value: str | None) -> tuple[int | None, tuple[int, ...]]:
    if value is None:
        return None, ()
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError:
        raise ValueError("--ctx-size must be a comma-separated list of integers") from None
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError("--ctx-size values must be > 0")
    if tuple(sorted(set(parsed))) != parsed:
        raise ValueError("--ctx-size values must be distinct and ascending")
    return parsed[0], parsed[1:]


def _print_hardware_human(hardware: HardwareReport) -> None:
    typer.echo(f"OS: {hardware.os_name} ({hardware.arch})")
    typer.echo(f"CPU: {hardware.cpu_model}")
    cores = f"physical={hardware.physical_cores} logical={hardware.logical_cores}"
    if hardware.perf_cores is not None:
        cores += f" perf={hardware.perf_cores}"
    typer.echo(f"Cores: {cores}")
    typer.echo(f"RAM: {hardware.ram_mb} MiB")
    if hardware.gpus:
        for gpu in hardware.gpus:
            typer.echo(f"GPU: {gpu.name} ({gpu.vendor}) vram={gpu.vram_mb} MiB method={gpu.method}")
    else:
        typer.echo("GPU: none detected")
    for warning in hardware.warnings:
        typer.echo(f"WARNING: {warning}")


def _print_llama_human(llama: LlamaCppReport) -> None:
    typer.echo(f"llama-bench: {llama.bench_path}")
    typer.echo(f"llama-cli: {llama.cli_path or '(not found)'}")
    typer.echo(f"llama-server: {llama.server_path or '(not found)'}")
    typer.echo(f"capabilities: {', '.join(sorted(llama.capabilities)) or '(none)'}")
    typer.echo(f"help_sha256: {llama.help_sha256}")


def _emit_tune_outcome(outcome: TuneOutcome, *, json_output: bool) -> None:
    if json_output:
        if outcome.exit_code >= 2 and not outcome.analysis:
            _echo_json(_failure_payload(outcome))
        else:
            _echo_json(outcome.analysis)
        return
    typer.echo(f"session: {outcome.session_dir}")
    if outcome.exit_code == 2:
        stage = outcome.failure_stage or "session"
        reason = outcome.failure_reason or "session evidence is unavailable or unreadable"
        typer.echo(f"tuning could not continue: {stage}: {reason}")
        typer.echo(f"evidence: unavailable or unreadable at {outcome.session_dir}")
        typer.echo("resumable: no")
        typer.echo(f"exit_code: {outcome.exit_code}")
        return
    if outcome.exit_code == 3:
        stage = outcome.failure_stage or "startup"
        reason = outcome.failure_reason or "tuning could not establish a usable baseline"
        typer.echo(f"tuning did not start: {stage}: {reason}")
        typer.echo(f"evidence: {outcome.session_dir}")
        typer.echo("resumable: no")
        typer.echo(f"exit_code: {outcome.exit_code}")
        return
    if outcome.exit_code == 4 and not outcome.analysis:
        stage = outcome.failure_stage or "interruption"
        reason = outcome.failure_reason or "tuning stopped before completion"
        typer.echo(f"tuning stopped with a resumable session: {stage}: {reason}")
        typer.echo(f"evidence: {outcome.session_dir}")
        typer.echo("resumable: yes")
        typer.echo(f"exit_code: {outcome.exit_code}")
        return
    winner = outcome.analysis.get("winner")
    if winner is not None:
        typer.echo(f"winner: {winner.get('trial_id')}")
    else:
        typer.echo("no confirmed improvement over baseline")
    typer.echo(f"exit_code: {outcome.exit_code}")


def _failure_payload(outcome: TuneOutcome) -> dict[str, Any]:
    """Return the machine-readable contract for a failed tuning operation."""
    return {
        "status": "failed",
        "exit_code": outcome.exit_code,
        "session_dir": str(outcome.session_dir),
        "failure_stage": outcome.failure_stage,
        "failure_reason": outcome.failure_reason,
        "resumable": outcome.exit_code == 4,
    }


def _emit_startup_failure(reason: str, *, exit_code: int, stage: str, json_output: bool) -> None:
    """Render a pre-session failure without claiming that evidence exists."""
    if json_output:
        _echo_json(
            {
                "status": "failed",
                "exit_code": exit_code,
                "session_dir": None,
                "failure_stage": stage,
                "failure_reason": reason,
                "resumable": False,
            }
        )
    else:
        typer.echo(f"error: {reason}", err=True)


def _quality_exec_supported() -> bool:
    if os.name != "posix":
        return False
    try:
        importlib.import_module("resource")
    except ImportError:
        return False
    return True


def _emit_quality_outcome(outcome: Any, *, json_output: bool) -> None:
    if json_output:
        _echo_json(outcome.summary)
        return
    if outcome.summary.get("error") is not None:
        typer.echo(f"error: {outcome.summary['error']}", err=True)
        return
    if outcome.summary.get("dry_run") is True:
        typer.echo("quality dry run")
        typer.echo(f"config source: {outcome.summary.get('config_source', 'unknown')}")
        typer.echo(f"config provenance: {outcome.summary.get('config_provenance') or 'none'}")
        for suite in outcome.summary.get("suites", ()):
            typer.echo(
                f"suite: {suite.get('suite_id', suite.get('name', 'unknown'))} "
                f"({suite.get('tasks', 0)} tasks)"
            )
        server_argv = outcome.summary.get("server_argv")
        typer.echo(f"server argv: {json.dumps(server_argv) if server_argv else 'none'}")
        typer.echo(f"exec tier: {'enabled' if outcome.summary.get('exec_enabled') else 'disabled'}")
        if outcome.summary.get("exec_enabled"):
            typer.echo(f"exec isolation: {outcome.summary.get('exec_isolation', 'unknown')}")
        return
    typer.echo(f"quality run: {outcome.run_dir}")
    overall = outcome.summary.get("overall")
    if overall is not None:
        typer.echo(f"overall: {overall}")
    for warning in outcome.summary.get("warnings", ()):
        typer.echo(f"warning: {warning}", err=True)
    typer.echo(f"exit_code: {outcome.exit_code}")


@app.command()
def scan(
    llama_bin: Annotated[
        Path | None,
        typer.Option("--llama-bin", help="Directory containing llama.cpp binaries"),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Machine-readable output on stdout")
    ] = False,
) -> None:
    """Assess hardware and the llama.cpp installation; no model needed."""
    from llamatune.hardware import assess_hardware
    from llamatune.llama import LlamaDiscoveryError, discover_llama

    hardware = assess_hardware()

    llama_report = None
    llama_error: str | None = None
    try:
        llama_report = discover_llama(llama_bin)
    except LlamaDiscoveryError as exc:
        llama_error = str(exc)

    if json_output:
        payload = {
            "hardware": dataclasses.asdict(hardware),
            "llama": dataclasses.asdict(llama_report) if llama_report is not None else None,
            "llama_error": llama_error,
        }
        _echo_json(payload)
        return

    _print_hardware_human(hardware)
    if llama_report is not None:
        _print_llama_human(llama_report)
    else:
        typer.echo(f"llama-bench: NOT FOUND ({llama_error})")


@app.command()
def quality(
    ctx: typer.Context,
    model_path: Annotated[
        Path | None,
        typer.Argument(help="Path to a GGUF model file", metavar="MODEL"),
    ] = None,
    llama_bin: Annotated[Path | None, typer.Option("--llama-bin")] = None,
    sessions_dir: Annotated[Path, typer.Option("--sessions-dir")] = Path("./llamatune-sessions"),
    config: Annotated[str | None, typer.Option("--config")] = None,
    config_session: Annotated[Path | None, typer.Option("--config-session")] = None,
    strict_config: Annotated[bool, typer.Option("--strict-config")] = False,
    compare_lossless: Annotated[bool, typer.Option("--compare-lossless")] = False,
    suites: Annotated[list[str] | None, typer.Option("--suite")] = None,
    task_filters: Annotated[list[str] | None, typer.Option("--tasks")] = None,
    list_suites: Annotated[bool, typer.Option("--list-suites")] = False,
    exec_enabled: Annotated[bool, typer.Option("--exec")] = False,
    exec_allow_network: Annotated[
        bool, typer.Option("--exec-allow-network", help="Allow --exec without network isolation")
    ] = False,
    ctx_size: Annotated[int, typer.Option("--ctx-size")] = 8192,
    quality_corpus: Annotated[Path | None, typer.Option("--quality-corpus")] = None,
    reps: Annotated[int, typer.Option("--reps")] = 1,
    max_tokens: Annotated[int, typer.Option("--max-tokens")] = 1024,
    request_timeout_s: Annotated[float, typer.Option("--request-timeout")] = 300.0,
    server_start_timeout_s: Annotated[float, typer.Option("--server-start-timeout")] = 600.0,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    resume: Annotated[Path | None, typer.Option("--resume")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Experimental: evaluate deterministic quality suites for one model/configuration."""
    from llamatune.qualsuites import bundled_suites, load_suite

    if list_suites:
        try:
            loaded = tuple(load_suite(name) for name in bundled_suites())
        except ValueError as exc:
            typer.echo(f"error: bundled quality suite is invalid: {exc}", err=True)
            raise typer.Exit(code=3) from exc
        for suite in loaded:
            typer.echo(f"{suite.suite_id}  {suite.kind}  {len(suite.tasks)} tasks")
        return

    if resume is not None:
        new_run_parameters = (
            "model_path",
            "sessions_dir",
            "config",
            "config_session",
            "strict_config",
            "compare_lossless",
            "suites",
            "task_filters",
            "exec_enabled",
            "exec_allow_network",
            "ctx_size",
            "quality_corpus",
            "reps",
            "max_tokens",
            "request_timeout_s",
            "server_start_timeout_s",
            "seed",
            "dry_run",
        )
        if any(
            (source := ctx.get_parameter_source(name)) is not None and source.name == "COMMANDLINE"
            for name in new_run_parameters
        ):
            typer.echo("error: --resume is mutually exclusive with new-run options", err=True)
            raise typer.Exit(code=2)
        from llamatune.quality import resume_quality

        try:
            outcome = resume_quality(resume, llama_bin=llama_bin)
        except ValueError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        except OSError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=3) from exc
        _emit_quality_outcome(outcome, json_output=json_output)
        raise typer.Exit(code=outcome.exit_code)

    if model_path is None:
        typer.echo("error: MODEL is required unless --resume or --list-suites is used", err=True)
        raise typer.Exit(code=2)
    if config not in {None, "best", "defaults"}:
        typer.echo("error: --config must be best or defaults", err=True)
        raise typer.Exit(code=2)
    if config_session is not None and config is not None:
        typer.echo("error: --config-session and --config are mutually exclusive", err=True)
        raise typer.Exit(code=2)
    if not 1 <= reps <= 5:
        typer.echo("error: --reps must be between 1 and 5", err=True)
        raise typer.Exit(code=2)
    if ctx_size <= 0 or max_tokens <= 0:
        typer.echo("error: --ctx-size and --max-tokens must be positive", err=True)
        raise typer.Exit(code=2)
    if (
        not math.isfinite(request_timeout_s)
        or not math.isfinite(server_start_timeout_s)
        or request_timeout_s <= 0
        or server_start_timeout_s <= 0
    ):
        typer.echo("error: quality timeouts must be finite and positive", err=True)
        raise typer.Exit(code=2)
    if exec_enabled and not _quality_exec_supported():
        typer.echo("error: --exec requires POSIX resource limits", err=True)
        raise typer.Exit(code=2)
    if exec_enabled:
        from llamatune.sandbox import IsolationStatus, detect_network_isolation

        status = detect_network_isolation()
        if status is not IsolationStatus.AVAILABLE and not exec_allow_network:
            typer.echo(
                f"error: --exec requires confirmed network namespace isolation "
                f"(probe: {status.value})",
                err=True,
            )
            typer.echo(
                "error: pass --exec-allow-network to run model-generated code with network access",
                err=True,
            )
            raise typer.Exit(code=3)
        from llamatune import sandbox as sandbox_module

        sandbox_module.set_allow_network_fallback(exec_allow_network)
        if exec_allow_network:
            typer.echo(
                "WARNING: --exec-allow-network: model-generated code will run WITH network access",
                err=True,
            )

    selected = tuple(suites or ("coding", "tooluse", "agentic", "ifollow"))
    try:
        loaded = tuple(load_suite(name) for name in selected)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    includes_perplexity = any(suite.kind == "perplexity" for suite in loaded)
    if includes_perplexity != (quality_corpus is not None):
        typer.echo(
            "error: --quality-corpus is required iff the perplexity suite is selected",
            err=True,
        )
        raise typer.Exit(code=2)
    if quality_corpus is not None and not quality_corpus.is_file():
        typer.echo(f"error: quality corpus is unreadable: {quality_corpus}", err=True)
        raise typer.Exit(code=2)

    from llamatune.quality import run_quality
    from llamatune.types import QualityOptions

    options = QualityOptions(
        model_path=model_path,
        llama_bin=llama_bin,
        sessions_dir=sessions_dir,
        config_mode="session" if config_session is not None else (config or "best"),
        config_session=config_session,
        strict_config=strict_config,
        compare_lossless=compare_lossless,
        suites=selected,
        task_filters=tuple(task_filters or ()),
        exec_enabled=exec_enabled,
        ctx_size=ctx_size,
        quality_corpus=quality_corpus,
        reps=reps,
        max_tokens=max_tokens,
        request_timeout_s=request_timeout_s,
        server_start_timeout_s=server_start_timeout_s,
        seed=seed,
        dry_run=dry_run,
    )
    try:
        outcome = run_quality(options)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except OSError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    _emit_quality_outcome(outcome, json_output=json_output)
    raise typer.Exit(code=outcome.exit_code)


@app.command()
def tune(
    model_path: Annotated[Path, typer.Argument(help="Path to a GGUF model file", metavar="MODEL")],
    llama_bin: Annotated[
        Path | None,
        typer.Option("--llama-bin", help="Directory containing llama.cpp binaries"),
    ] = None,
    sessions_dir: Annotated[
        Path, typer.Option("--sessions-dir", help="Directory to create the session under")
    ] = Path("./llamatune-sessions"),
    target: Annotated[
        Target, typer.Option("--target", help="Optimization target")
    ] = Target.balanced,
    budget_trials: Annotated[
        int,
        typer.Option(
            "--budget-trials",
            help="Counted benchmark-measurement budget (pruned candidates consume none)",
        ),
    ] = 60,
    budget_minutes: Annotated[
        float | None, typer.Option("--budget-minutes", help="Wall-clock budget in minutes")
    ] = None,
    baseline_runs: Annotated[
        int, typer.Option("--baseline-runs", help="Independent baseline invocations (min 3)")
    ] = 3,
    reps_search: Annotated[
        int, typer.Option("--reps-search", help="llama-bench -r value for search trials")
    ] = 3,
    reps_confirm: Annotated[
        int, typer.Option("--reps-confirm", help="llama-bench -r value for baseline/confirmation")
    ] = 5,
    pp: Annotated[int, typer.Option("--pp", help="Prompt-processing workload size")] = 512,
    tg: Annotated[int, typer.Option("--tg", help="Token-generation workload size")] = 128,
    depth: Annotated[
        int | None, typer.Option("--depth", help="KV depth for the full tuning workload")
    ] = None,
    depth_profile: Annotated[
        str | None,
        typer.Option("--depth-profile", help="Ascending CSV depths for winner profiling"),
    ] = None,
    ctx_size: Annotated[
        str | None, typer.Option("--ctx-size", help="Ascending context sizes to validate (CSV)")
    ] = None,
    vram_reserve_mb: Annotated[
        int | None, typer.Option("--vram-reserve-mb", help="VRAM safety reserve in MiB")
    ] = None,
    initial_gpu_layers: Annotated[
        int | None, typer.Option("--initial-gpu-layers", help="First GPU-layer probe")
    ] = None,
    max_gpu_layers: Annotated[
        str | None, typer.Option("--max-gpu-layers", help="GPU-layer cap or 'auto'")
    ] = None,
    initial_cpu_moe: Annotated[
        str | None, typer.Option("--initial-cpu-moe", help="Initial CPU MoE layers or 'auto'")
    ] = None,
    allow_lossy: Annotated[
        bool,
        typer.Option("--allow-lossy", help="Experimental: include KV-quantization dimensions"),
    ] = False,
    cooldown: Annotated[
        float, typer.Option("--cooldown", help="Seconds to sleep between invocations")
    ] = 0.0,
    thermal_threshold_c: Annotated[
        float, typer.Option("--thermal-threshold-c", help="GPU temperature settling threshold")
    ] = 75.0,
    thermal_wait_cap_s: Annotated[
        float, typer.Option("--thermal-wait-cap-s", help="Maximum adaptive thermal wait")
    ] = 60.0,
    multi_gpu: Annotated[
        bool,
        typer.Option(
            "--multi-gpu",
            help="Experimental: pool known VRAM capacity across detected GPUs",
        ),
    ] = False,
    baseline_only: Annotated[
        bool, typer.Option("--baseline-only", help="Stop after the baseline phase")
    ] = False,
    full_hash: Annotated[
        bool, typer.Option("--full-hash", help="Compute the full SHA-256 of the model file")
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Machine-readable output on stdout")
    ] = False,
    progress: Annotated[
        ProgressMode, typer.Option("--progress", help="Progress renderer")
    ] = ProgressMode.auto,
    tui: Annotated[bool, typer.Option("--tui", help="Force the Rich live dashboard")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", help="Disable progress output")] = False,
    allow_core_dumps: Annotated[
        bool, typer.Option("--allow-core-dumps", help="Keep inherited child core-dump limits")
    ] = False,
    quiet_wait_s: Annotated[
        float, typer.Option("--quiet-wait-s", help="Maximum seconds to wait for low load")
    ] = 0.0,
    quiet_load: Annotated[
        float | None, typer.Option("--quiet-load", help="One-minute load threshold")
    ] = None,
    observe_vram: Annotated[
        bool,
        typer.Option("--observe-vram/--no-observe-vram", help="Sample GPU state around runs"),
    ] = True,
    validate_with_cli: Annotated[
        bool, typer.Option("--validate-with-cli", help="Validate the winner with llama-cli")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the search plan without executing")
    ] = False,
    quality_corpus: Annotated[
        Path | None,
        typer.Option(
            "--quality-corpus",
            help="Experimental: corpus for lossy-KV quality gating",
        ),
    ] = None,
    batched_trials: Annotated[
        bool,
        typer.Option("--batched-trials/--no-batched-trials", help="Batch compatible sweep trials"),
    ] = True,
    ot_search: Annotated[
        bool, typer.Option("--ot-search", help="Search tensor override placements")
    ] = False,
) -> None:
    """Find the fastest llama.cpp runtime settings for MODEL."""
    if tui and quiet:
        typer.echo("error: --tui and --quiet are mutually exclusive", err=True)
        raise typer.Exit(code=2)
    if baseline_runs < 3:
        typer.echo("error: --baseline-runs must be >= 3", err=True)
        raise typer.Exit(code=2)
    if budget_trials < baseline_runs:
        typer.echo("error: --budget-trials must be >= --baseline-runs", err=True)
        raise typer.Exit(code=2)
    try:
        max_gpu_layers_value = _parse_auto_nonnegative(max_gpu_layers, "--max-gpu-layers")
        initial_cpu_moe_value = _parse_auto_nonnegative(initial_cpu_moe, "--initial-cpu-moe")
        depth_profile_value = _parse_depth_profile(depth_profile)
        ctx_size_value, ctx_ladder_value = _parse_ctx_ladder(ctx_size)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from None
    validations = (
        (vram_reserve_mb is not None and vram_reserve_mb < 0, "--vram-reserve-mb must be >= 0"),
        (
            initial_gpu_layers is not None and initial_gpu_layers < 0,
            "--initial-gpu-layers must be >= 0",
        ),
        (quiet_wait_s < 0, "--quiet-wait-s must be >= 0"),
        (quiet_load is not None and quiet_load <= 0, "--quiet-load must be > 0"),
        (depth is not None and depth < 0, "--depth must be >= 0"),
        (thermal_threshold_c <= 0, "--thermal-threshold-c must be > 0"),
        (thermal_wait_cap_s < 0, "--thermal-wait-cap-s must be >= 0"),
        (
            initial_gpu_layers is not None
            and max_gpu_layers_value is not None
            and initial_gpu_layers > max_gpu_layers_value,
            "--initial-gpu-layers must not exceed --max-gpu-layers",
        ),
    )
    for invalid, message in validations:
        if invalid:
            typer.echo(f"error: {message}", err=True)
            raise typer.Exit(code=2)

    from llamatune.hardware import assess_hardware
    from llamatune.llama import LlamaDiscoveryError, discover_llama
    from llamatune.model import ModelInspectionError, inspect_model
    from llamatune.search import run_tuning
    from llamatune.session import Session, SessionPathError
    from llamatune.types import TuneOptions

    try:
        llama_report = discover_llama(llama_bin)
    except LlamaDiscoveryError as exc:
        _emit_startup_failure(
            str(exc), exit_code=3, stage="llama_discovery", json_output=json_output
        )
        raise typer.Exit(code=3) from exc

    if (
        depth is not None or depth_profile_value is not None
    ) and "d" not in llama_report.capabilities:
        reason = f"llama-bench '{llama_report.bench_path}' does not support -d/--n-depth"
        _emit_startup_failure(
            reason,
            exit_code=2,
            stage="capability",
            json_output=json_output,
        )
        raise typer.Exit(code=2)

    try:
        model_report = inspect_model(model_path, full_hash=full_hash)
    except ModelInspectionError as exc:
        _emit_startup_failure(
            str(exc), exit_code=3, stage="model_inspection", json_output=json_output
        )
        raise typer.Exit(code=3) from exc

    hardware_report = assess_hardware()
    effective_progress = (
        ProgressMode.none if json_output and progress == ProgressMode.auto else progress
    )
    reporter = _make_reporter(effective_progress, tui=tui, quiet=quiet)

    options = TuneOptions(
        target=target.value,
        budget_trials=budget_trials,
        budget_minutes=budget_minutes,
        reps_search=reps_search,
        reps_confirm=reps_confirm,
        baseline_runs=baseline_runs,
        pp=pp,
        tg=tg,
        allow_lossy=allow_lossy,
        cooldown_s=cooldown,
        baseline_only=baseline_only,
        llama_bin=llama_bin,
        sessions_dir=sessions_dir,
        full_hash=full_hash,
        ctx_size=ctx_size_value,
        ctx_ladder=ctx_ladder_value,
        vram_reserve_mb=vram_reserve_mb,
        initial_gpu_layers=initial_gpu_layers,
        max_gpu_layers=max_gpu_layers_value,
        initial_cpu_moe=initial_cpu_moe_value,
        allow_core_dumps=allow_core_dumps,
        quiet_wait_s=quiet_wait_s,
        quiet_load=(quiet_load if quiet_load is not None else hardware_report.physical_cores / 2),
        observe_vram=observe_vram,
        validate_with_cli=validate_with_cli,
        quality_corpus=quality_corpus,
        batched_trials=batched_trials,
        ot_search=ot_search,
        depth=depth,
        depth_profile=depth_profile_value,
        thermal_threshold_c=thermal_threshold_c,
        thermal_wait_cap_s=thermal_wait_cap_s,
        multi_gpu=multi_gpu,
    )

    if dry_run:
        from llamatune.report import render_search_plan
        from llamatune.types import TrialConfig, VramCalibration

        calibration = None
        calibration_path = sessions_dir / "calibration.json"
        try:
            if calibration_path.is_file():
                calibration = VramCalibration(**_load_json(calibration_path))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            typer.echo(f"warning: ignoring calibration file: {exc}", err=True)
        incumbent = TrialConfig(
            gpu_layers=min(
                model_report.ngl_all,
                initial_gpu_layers or 0,
                max_gpu_layers_value if max_gpu_layers_value is not None else model_report.ngl_all,
            ),
            moe_cpu_layers=(
                min(model_report.n_layer, initial_cpu_moe_value)
                if initial_cpu_moe_value is not None
                else model_report.n_layer
                if model_report.moe
                else 0
            ),
            flash_attn=False,
            ubatch=512,
            batch=2048,
            threads=max(1, hardware_report.physical_cores),
            mmap=True,
            no_kv_offload=False,
            cache_type_k="f16",
            cache_type_v="f16",
        )
        config_module = importlib.import_module("llamatune.config")
        build_search_plan = getattr(config_module, "build_search_plan", None)
        if build_search_plan is None:
            typer.echo("error: dry-run planning is not available", err=True)
            raise typer.Exit(code=2)
        plan = build_search_plan(
            hardware=hardware_report,
            model=model_report,
            llama=llama_report,
            options=options,
            incumbent=incumbent,
            calibration=calibration,
        )
        typer.echo(
            render_search_plan(
                plan, dataclasses.asdict(hardware_report), dataclasses.asdict(model_report)
            ),
            nl=False,
        )
        return

    try:
        session = Session.create(
            sessions_dir,
            model=model_report,
            hardware=hardware_report,
            llama=llama_report,
            options=options,
            argv=sys.argv,
        )
    except (OSError, SessionPathError) as exc:
        _emit_startup_failure(
            f"could not create session directory under {sessions_dir}: {exc}",
            exit_code=2,
            stage="session_creation",
            json_output=json_output,
        )
        raise typer.Exit(code=2) from exc

    calibration = None
    calibration_path = sessions_dir / "calibration.json"
    try:
        if calibration_path.is_file():
            from llamatune.types import VramCalibration

            calibration = VramCalibration(**_load_json(calibration_path))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        typer.echo(f"warning: ignoring calibration file: {exc}", err=True)

    outcome = run_tuning(
        session,
        hardware_report,
        model_report,
        llama_report,
        options,
        reporter=reporter,
        calibration=calibration,
    )
    _emit_tune_outcome(outcome, json_output=json_output)
    _refresh_results_matrix(sessions_dir, outcome.exit_code)
    raise typer.Exit(code=outcome.exit_code)


@app.command()
def resume(
    session_dir: Annotated[
        Path, typer.Argument(help="Session directory to resume", metavar="SESSION_DIR")
    ],
    json_output: Annotated[
        bool, typer.Option("--json", help="Machine-readable output on stdout")
    ] = False,
    progress: Annotated[
        ProgressMode, typer.Option("--progress", help="Progress renderer")
    ] = ProgressMode.auto,
    tui: Annotated[bool, typer.Option("--tui", help="Force the Rich live dashboard")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", help="Disable progress output")] = False,
) -> None:
    """Re-validate identity, skip journaled trials, and continue tuning."""
    from llamatune.search import resume_tuning

    effective_progress = (
        ProgressMode.none if json_output and progress == ProgressMode.auto else progress
    )
    reporter = _make_reporter(effective_progress, tui=tui, quiet=quiet)
    outcome = resume_tuning(session_dir, reporter=reporter)
    _emit_tune_outcome(outcome, json_output=json_output)
    _refresh_results_matrix(session_dir.resolve().parent, outcome.exit_code)
    raise typer.Exit(code=outcome.exit_code)


@app.command("report")
def report_cmd(
    session_dir: Annotated[
        Path, typer.Argument(help="Session directory to render a report for", metavar="SESSION_DIR")
    ],
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit analysis.json to stdout instead of a summary")
    ] = False,
) -> None:
    """Regenerate report.md from a session's recorded evidence."""
    from llamatune import report as report_module
    from llamatune.session import Session, SessionCorruptionError, SessionPathError

    try:
        session = Session.load(session_dir)
        analysis = session.read_json("analysis.json")
        session_meta = session.read_json("session.json")
        hardware = session.read_json("hardware.json")
        model = session.read_json("model.json")
        llamacpp = session.read_json("llamacpp.json")
        text = report_module.render(analysis, session_meta, hardware, model, llamacpp)
        session.write_text("report.md", text)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        json.JSONDecodeError,
        SessionCorruptionError,
        SessionPathError,
    ) as exc:
        typer.echo(f"error: could not read session evidence in {session_dir}: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    if json_output:
        _echo_json(analysis)
    else:
        typer.echo(f"report.md written to {session.dir / 'report.md'}")


@app.command("export")
def export_cmd(
    session_dir: Annotated[Path, typer.Argument(help="Session directory")],
    export_format: Annotated[
        str, typer.Option("--format", help="llama-server, llama-cli, systemd, llama-swap, or json")
    ] = "llama-server",
) -> None:
    """Export a confirmed recommendation in a ready-to-use format."""
    from llamatune.report import render_export

    try:
        recommended = _load_json(session_dir / "recommended.json")
        session_meta = _load_json(session_dir / "session.json")
        text = render_export(recommended, session_meta, export_format)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        typer.echo(f"error: could not export recommendation: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(text, nl=False)


@app.command()
def nightshift(
    models_dir: Annotated[
        Path, typer.Argument(help="Directory recursively scanned for GGUF models")
    ],
    llama_bin: Annotated[Path | None, typer.Option("--llama-bin")] = None,
    sessions_dir: Annotated[Path, typer.Option("--sessions-dir")] = Path("./llamatune-sessions"),
    until: Annotated[str | None, typer.Option("--until", help="Local deadline HH:MM")] = None,
    max_hours: Annotated[float | None, typer.Option("--max-hours")] = None,
    profile: Annotated[str, typer.Option("--profile", help="deep or standard")] = "deep",
    drift_threshold: Annotated[float, typer.Option("--drift-threshold")] = 0.05,
    calibration_runs: Annotated[int, typer.Option("--calibration-runs")] = 3,
    duplicates: Annotated[str, typer.Option("--duplicates", help="one or both")] = "one",
    include: Annotated[list[str] | None, typer.Option("--include")] = None,
    exclude: Annotated[list[str] | None, typer.Option("--exclude")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    target: Annotated[Target, typer.Option("--target")] = Target.balanced,
    allow_lossy: Annotated[bool, typer.Option("--allow-lossy")] = False,
    ctx_size: Annotated[
        str | None,
        typer.Option("--ctx-size", help="Ascending context sizes to validate (CSV)"),
    ] = None,
    depth: Annotated[int | None, typer.Option("--depth")] = None,
    vram_reserve_mb: Annotated[int | None, typer.Option("--vram-reserve-mb")] = None,
    cooldown_s: Annotated[float | None, typer.Option("--cooldown")] = None,
    full_hash: Annotated[bool, typer.Option("--full-hash")] = False,
    budget_trials: Annotated[int | None, typer.Option("--budget-trials")] = None,
    reps_search: Annotated[int | None, typer.Option("--reps-search")] = None,
    reps_confirm: Annotated[int | None, typer.Option("--reps-confirm")] = None,
    baseline_runs: Annotated[int | None, typer.Option("--baseline-runs")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Experimental: tune and verify local GGUF models unattended."""
    import re

    from llamatune.nightshift import run_nightshift
    from llamatune.types import NightshiftOptions

    error: str | None = None
    if until is not None:
        match = re.fullmatch(r"(\d{2}):(\d{2})", until)
        if match is None or int(match.group(1)) > 23 or int(match.group(2)) > 59:
            error = "--until must be HH:MM using a 24-hour clock"
    if max_hours is not None and max_hours <= 0:
        error = "--max-hours must be > 0"
    if profile not in {"deep", "standard"}:
        error = "--profile must be 'deep' or 'standard'"
    if duplicates not in {"one", "both"}:
        error = "--duplicates must be 'one' or 'both'"
    if drift_threshold < 0:
        error = "--drift-threshold must be >= 0"
    if calibration_runs < 2:
        error = "--calibration-runs must be >= 2"
    try:
        ctx_size_value, ctx_ladder_value = _parse_ctx_ladder(ctx_size)
    except ValueError as exc:
        error = str(exc)
        ctx_size_value, ctx_ladder_value = None, ()
    positive = {
        "--budget-trials": budget_trials,
        "--reps-search": reps_search,
        "--reps-confirm": reps_confirm,
        "--baseline-runs": baseline_runs,
    }
    for option, value in positive.items():
        if value is not None and value <= 0:
            error = f"{option} must be > 0"
            break
    if baseline_runs is not None and baseline_runs < 3:
        error = "--baseline-runs must be >= 3"
    if vram_reserve_mb is not None and vram_reserve_mb < 0:
        error = "--vram-reserve-mb must be >= 0"
    if cooldown_s is not None and cooldown_s < 0:
        error = "--cooldown must be >= 0"
    if depth is not None and depth < 0:
        error = "--depth must be >= 0"
    if error is not None:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=2)

    options = NightshiftOptions(
        models_dir=models_dir,
        llama_bin=llama_bin,
        sessions_dir=sessions_dir,
        until=until,
        max_hours=max_hours,
        profile=profile,
        drift_threshold=drift_threshold,
        calibration_runs=calibration_runs,
        duplicates=duplicates,
        include=tuple(include or ()),
        exclude=tuple(exclude or ()),
        dry_run=dry_run,
        target=target.value,
        allow_lossy=allow_lossy,
        ctx_size=ctx_size_value,
        ctx_ladder=ctx_ladder_value,
        vram_reserve_mb=vram_reserve_mb,
        cooldown_s=cooldown_s,
        full_hash=full_hash,
        budget_trials=budget_trials,
        reps_search=reps_search,
        reps_confirm=reps_confirm,
        baseline_runs=baseline_runs,
        depth=depth,
    )
    outcome = run_nightshift(options)
    if json_output:
        _echo_json(outcome.summary)
    else:
        if dry_run:
            typer.echo("Experimental Night Shift plan:")
            for item in outcome.summary.get("items", []):
                if not isinstance(item, dict):
                    continue
                estimate = item.get("estimated_minutes")
                estimate_text = f"{estimate} min" if estimate is not None else "unestimated"
                typer.echo(
                    f"- {item.get('kind', 'unknown')}: "
                    f"{item.get('model_path') or item.get('session_dir') or '-'} "
                    f"({estimate_text}) — {item.get('reason') or '-'}"
                )
        typer.echo(f"nightshift: {outcome.run_dir}")
        typer.echo(f"report: {outcome.run_dir / 'nightshift-report.md'}")
        typer.echo(f"exit_code: {outcome.exit_code}")
    if not dry_run:
        _refresh_results_matrix(sessions_dir, outcome.exit_code)
    raise typer.Exit(code=outcome.exit_code)


@app.command()
def marathon(
    model_path: Annotated[Path, typer.Argument(help="GGUF model file")],
    llama_bin: Annotated[Path | None, typer.Option("--llama-bin")] = None,
    sessions_dir: Annotated[Path, typer.Option("--sessions-dir")] = Path("./llamatune-sessions"),
    until: Annotated[str | None, typer.Option("--until", help="Local deadline HH:MM")] = None,
    max_hours: Annotated[float | None, typer.Option("--max-hours")] = None,
    rounds_max: Annotated[int, typer.Option("--rounds-max")] = 6,
    converge_rounds: Annotated[int, typer.Option("--converge-rounds")] = 2,
    ab_blocks: Annotated[int, typer.Option("--ab-blocks")] = 5,
    depth_grid: Annotated[str, typer.Option("--depth-grid")] = "0,8192,32768",
    ctx_size: Annotated[
        str | None, typer.Option("--ctx-size", help="Ascending context sizes to validate (CSV)")
    ] = None,
    matrix_refine: Annotated[bool, typer.Option("--matrix-refine/--no-matrix-refine")] = True,
    drift_threshold: Annotated[float, typer.Option("--drift-threshold")] = 0.05,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    target: Annotated[Target, typer.Option("--target")] = Target.balanced,
    allow_lossy: Annotated[bool, typer.Option("--allow-lossy")] = False,
    vram_reserve_mb: Annotated[int | None, typer.Option("--vram-reserve-mb")] = None,
    cooldown_s: Annotated[float | None, typer.Option("--cooldown")] = None,
    full_hash: Annotated[bool, typer.Option("--full-hash")] = False,
    pp: Annotated[int, typer.Option("--pp")] = 512,
    tg: Annotated[int, typer.Option("--tg")] = 128,
    quality_corpus: Annotated[Path | None, typer.Option("--quality-corpus")] = None,
    ot_search: Annotated[bool, typer.Option("--ot-search")] = False,
    budget_trials: Annotated[int | None, typer.Option("--budget-trials")] = None,
    reps_search: Annotated[int | None, typer.Option("--reps-search")] = None,
    reps_confirm: Annotated[int | None, typer.Option("--reps-confirm")] = None,
    baseline_runs: Annotated[int | None, typer.Option("--baseline-runs")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Experimental: spend a bounded or convergent window tuning one model."""
    import re

    error: str | None = None
    if until is not None:
        match = re.fullmatch(r"(\d{2}):(\d{2})", until)
        if match is None or int(match.group(1)) > 23 or int(match.group(2)) > 59:
            error = "--until must be HH:MM using a 24-hour clock"
    if max_hours is not None and max_hours <= 0:
        error = "--max-hours must be > 0"
    if rounds_max < 1:
        error = "--rounds-max must be >= 1"
    if converge_rounds < 1:
        error = "--converge-rounds must be >= 1"
    if ab_blocks < 3:
        error = "--ab-blocks must be >= 3"
    if drift_threshold < 0:
        error = "--drift-threshold must be >= 0"
    if vram_reserve_mb is not None and vram_reserve_mb < 0:
        error = "--vram-reserve-mb must be >= 0"
    if cooldown_s is not None and cooldown_s < 0:
        error = "--cooldown must be >= 0"
    if pp <= 0:
        error = "--pp must be > 0"
    if tg <= 0:
        error = "--tg must be > 0"
    positive = {
        "--budget-trials": budget_trials,
        "--reps-search": reps_search,
        "--reps-confirm": reps_confirm,
        "--baseline-runs": baseline_runs,
    }
    for option, value in positive.items():
        if value is not None and value <= 0:
            error = f"{option} must be > 0"
            break
    if baseline_runs is not None and baseline_runs < 3:
        error = "--baseline-runs must be >= 3"
    try:
        depth_grid_value = _parse_depth_grid(depth_grid)
        ctx_size_value, ctx_ladder_value = _parse_ctx_ladder(ctx_size)
    except ValueError as exc:
        error = str(exc)
        depth_grid_value = ()
        ctx_size_value, ctx_ladder_value = None, ()
    if error is not None:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=2)

    from llamatune.llama import LlamaDiscoveryError, discover_llama
    from llamatune.marathon import run_marathon
    from llamatune.types import MarathonOptions

    try:
        llama_report = discover_llama(llama_bin)
    except LlamaDiscoveryError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    if any(depth > 0 for depth in depth_grid_value) and "d" not in llama_report.capabilities:
        typer.echo(
            f"error: llama-bench '{llama_report.bench_path}' does not support -d/--n-depth",
            err=True,
        )
        raise typer.Exit(code=2)

    options = MarathonOptions(
        model_path=model_path,
        llama_bin=llama_bin,
        sessions_dir=sessions_dir,
        until=until,
        max_hours=max_hours,
        rounds_max=rounds_max,
        converge_rounds=converge_rounds,
        ab_blocks=ab_blocks,
        depth_grid=depth_grid_value,
        ctx_size=ctx_size_value,
        ctx_ladder=ctx_ladder_value,
        matrix_refine=matrix_refine,
        drift_threshold=drift_threshold,
        dry_run=dry_run,
        target=target.value,
        allow_lossy=allow_lossy,
        vram_reserve_mb=vram_reserve_mb,
        cooldown_s=cooldown_s,
        full_hash=full_hash,
        pp=pp,
        tg=tg,
        quality_corpus=quality_corpus,
        ot_search=ot_search,
        budget_trials=budget_trials,
        reps_search=reps_search,
        reps_confirm=reps_confirm,
        baseline_runs=baseline_runs,
    )
    outcome = run_marathon(options)
    if json_output:
        _echo_json(outcome.summary)
    elif dry_run:
        typer.echo("Experimental Marathon plan:")
        plan = outcome.summary.get("plan", outcome.summary)
        typer.echo(f"reconnaissance: {plan.get('recon_runs', 10)} runs")
        typer.echo(f"round 1 budget: {plan.get('round_1_budget', budget_trials or 240)} trials")
        typer.echo(f"coverage tiers: {plan.get('tiers', 'A, B, C, D')}")
        default_cells = len(depth_grid_value) * max(1, 1 + len(ctx_ladder_value))
        typer.echo(f"matrix cells: {plan.get('matrix_cells', default_cells)}")
        typer.echo(f"A/B blocks: {ab_blocks}")
    else:
        typer.echo(f"marathon: {outcome.run_dir}")
        typer.echo(f"report: {outcome.run_dir / 'marathon-report.md'}")
        typer.echo(f"exit_code: {outcome.exit_code}")
    if not dry_run:
        _refresh_results_matrix(sessions_dir, outcome.exit_code)
    raise typer.Exit(code=outcome.exit_code)


@app.command("sessions")
def sessions_cmd(
    sessions_dir: Annotated[Path, typer.Argument(help="Directory containing sessions")],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List complete, in-progress, and corrupt tuning sessions."""
    from llamatune.session import list_sessions

    rows = list_sessions(sessions_dir)
    if json_output:
        _echo_json(rows)
        return
    for row in rows:
        typer.echo(
            f"{row['session_dir']} model={row.get('model') or '-'} "
            f"status={row['status']} exit={row.get('exit_code')} "
            f"winner={row.get('winner_trial_id') or '-'} confirmed={row.get('confirmed', False)}"
        )


@app.command("best")
def best_cmd(
    model_path: Annotated[Path, typer.Argument(help="Path to a GGUF model")],
    llama_bin: Annotated[Path | None, typer.Option("--llama-bin")] = None,
    sessions_dir: Annotated[Path, typer.Option("--sessions-dir")] = Path("./llamatune-sessions"),
    ctx_size: Annotated[int | None, typer.Option("--ctx-size")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Look up the best matching confirmed recommendation."""
    from llamatune.config import hardware_signature
    from llamatune.hardware import assess_hardware
    from llamatune.llama import LlamaDiscoveryError, discover_llama
    from llamatune.model import ModelInspectionError, inspect_model
    from llamatune.registry import lookup
    from llamatune.report import render_registry_lookup

    try:
        model = inspect_model(model_path)
        llama = discover_llama(llama_bin)
    except (ModelInspectionError, LlamaDiscoveryError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    hardware = assess_hardware()
    result = lookup(
        sessions_dir / "registry.jsonl",
        model,
        llama,
        hardware_signature(hardware),
        ctx_size=ctx_size,
    )
    if json_output:
        _echo_json(result)
    else:
        typer.echo(render_registry_lookup(result), nl=False)
    raise typer.Exit(code=0 if result.get("status") == "hit" else 1)


@app.command("revalidate")
def revalidate_cmd(
    session_dir: Annotated[Path, typer.Argument(help="Session directory")],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Re-run confirmation for a recorded winner."""
    from llamatune.report import render_revalidation

    search_module = importlib.import_module("llamatune.search")
    revalidate_session = getattr(search_module, "revalidate_session", None)
    if revalidate_session is None:
        typer.echo("error: revalidation is not available", err=True)
        raise typer.Exit(code=2)
    outcome = revalidate_session(session_dir)
    if outcome.exit_code >= 2:
        if json_output:
            _echo_json(_failure_payload(outcome))
        else:
            stage = outcome.failure_stage or "revalidation"
            reason = outcome.failure_reason or "session evidence could not be revalidated"
            typer.echo(f"revalidation failed: {stage}: {reason}")
            typer.echo(f"evidence: {outcome.session_dir}")
            typer.echo(f"resumable: {'yes' if outcome.exit_code == 4 else 'no'}")
            typer.echo(f"exit_code: {outcome.exit_code}")
    elif json_output:
        _echo_json(outcome.analysis)
    else:
        typer.echo(render_revalidation(outcome.analysis), nl=False)
    _refresh_results_matrix(session_dir.resolve().parent, outcome.exit_code)
    raise typer.Exit(code=outcome.exit_code)


@app.command("calibrate")
def calibrate_cmd(
    sessions_dir: Annotated[
        Path, typer.Option("--sessions-dir", help="Directory containing sessions")
    ] = Path("./llamatune-sessions"),
) -> None:
    """Experimental: fit VRAM estimator correction factors from observed sessions."""
    try:
        calibrate_module = importlib.import_module("llamatune.calibrate")
    except ImportError as exc:
        typer.echo("error: calibration is not available", err=True)
        raise typer.Exit(code=2) from exc
    result = calibrate_module.calibrate(sessions_dir)
    typer.echo(
        "experimental calibration written: "
        f"samples={result['samples']} weights={result['weights_scale']:.4f} "
        f"kv={result['kv_scale']:.4f} compute={result['compute_scale']:.4f}"
    )
