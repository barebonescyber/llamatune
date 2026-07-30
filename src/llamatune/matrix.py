"""Context by KV-depth operating-point measurements for Marathon."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from llamatune import bench, executor
from llamatune.stats import weights_for_target
from llamatune.types import (
    LlamaCppReport,
    MarathonOptions,
    MatrixCell,
    ModelReport,
    TrialConfig,
)

_PROBE_TIMEOUT_S = 3600.0
_TRIAL_TIMEOUT_S = 3600.0


class MarathonRun(Protocol):
    dir: Path

    def matrix_dir(self, ctx: int, depth: int, refine: int | None = None) -> Path: ...

    def append(self, entry: dict[str, Any]) -> None: ...

    def write_json(self, name: str, payload: dict[str, Any]) -> None: ...


def _reps(options: MarathonOptions) -> int:
    return options.reps_search if options.reps_search is not None else 8


def _cooldown(options: MarathonOptions) -> float:
    return options.cooldown_s if options.cooldown_s is not None else 10.0


def _payload(argv: tuple[str, ...], result: executor.ExecResult) -> dict[str, Any]:
    return {
        "argv": list(argv),
        "exit_code": result.exit_code,
        "wall_s": result.wall_s,
        "timed_out": result.timed_out,
        "started": result.started,
        "ended": result.ended,
        "env_names": list(result.env_names),
        "stdout": {
            "sha256": result.stdout.sha256,
            "size_bytes": result.stdout.size_bytes,
            "truncated": result.stdout.truncated,
        },
        "stderr": {
            "sha256": result.stderr.sha256,
            "size_bytes": result.stderr.size_bytes,
            "truncated": result.stderr.truncated,
        },
    }


def _execute(
    run: MarathonRun, directory: Path, argv: tuple[str, ...], timeout: float
) -> tuple[float, float] | None:
    result = executor.run(
        argv,
        timeout_s=timeout,
        stdout_path=directory / "stdout.json",
        stderr_path=directory / "stderr.log",
    )
    run.write_json(str((directory / "command.json").relative_to(run.dir)), _payload(argv, result))
    if result.exit_code != 0 or result.timed_out or result.stdout.truncated:
        return None
    try:
        sample = bench.parse_bench_output(result.stdout.path.read_bytes())
    except (OSError, bench.BenchParseError):
        return None
    return sample.pp_avg, sample.tg_avg


def _fallbacks(champion: TrialConfig, model: ModelReport) -> tuple[TrialConfig, ...]:
    result: list[TrialConfig] = []
    seen = {champion.trial_id}
    moe_values = sorted({champion.moe_cpu_layers, model.n_layer, model.n_layer // 2})
    for ngl in range(champion.gpu_layers, -1, -1):
        for ncmoe in reversed(moe_values):
            candidate = dataclasses.replace(champion, gpu_layers=ngl, moe_cpu_layers=ncmoe)
            if candidate.trial_id not in seen:
                seen.add(candidate.trial_id)
                result.append(candidate)
    return tuple(result)


def _refinements(config: TrialConfig) -> tuple[TrialConfig, ...]:
    result: list[TrialConfig] = []
    seen = {config.trial_id}
    for ubatch in (max(1, config.ubatch // 2), min(config.batch, config.ubatch * 2)):
        candidate = dataclasses.replace(config, ubatch=ubatch)
        if candidate.trial_id not in seen:
            seen.add(candidate.trial_id)
            result.append(candidate)
    return tuple(result[:2])


def _score(value: tuple[float, float], weights: tuple[float, float]) -> float:
    return float(value[0] ** weights[0] * value[1] ** weights[1])


def run_matrix(
    run: MarathonRun,
    champion: TrialConfig,
    model: ModelReport,
    llama: LlamaCppReport,
    options: MarathonOptions,
    *,
    remaining_minutes_fn: Callable[[], float],
) -> tuple[MatrixCell, ...]:
    """Measure the resolved context ladder crossed with every requested depth."""
    contexts = ((options.ctx_size,) if options.ctx_size is not None else ()) + options.ctx_ladder
    contexts = tuple(dict.fromkeys(contexts))
    rows: list[MatrixCell] = []
    champion_failed_at: int | None = None
    weights = weights_for_target(options.target)
    for ctx in contexts:
        for depth in options.depth_grid:
            if remaining_minutes_fn() <= 0:
                cell = MatrixCell(
                    ctx=ctx,
                    depth=depth,
                    status="deferred",
                    config=None,
                    pp=None,
                    tg=None,
                    refined=False,
                    evidence=None,
                )
                rows.append(cell)
                run.append(
                    {
                        "type": "deferred",
                        "item": "matrix",
                        "ctx": ctx,
                        "depth": depth,
                        "reason": "deadline",
                    }
                )
                continue
            reuse = getattr(run, "champion_evidence", None)
            if (
                depth == 0
                and ctx == options.ctx_size
                and isinstance(reuse, dict)
                and reuse.get("ctx") == ctx
                and reuse.get("config") == champion.to_dict()
                and isinstance(reuse.get("pp"), (int, float))
                and isinstance(reuse.get("tg"), (int, float))
            ):
                cell = MatrixCell(
                    ctx=ctx,
                    depth=depth,
                    status="ok",
                    config=champion,
                    pp=float(reuse["pp"]),
                    tg=float(reuse["tg"]),
                    refined=False,
                    evidence=Path(str(reuse.get("evidence"))),
                )
                rows.append(cell)
                run.append({"type": "matrix_cell", "reused": True, **_cell_dict(cell)})
                continue
            directory = run.matrix_dir(ctx, depth)
            config: TrialConfig | None = None
            if champion_failed_at is None:
                probe_dir = run.matrix_dir(ctx, depth, -1)
                probe_argv = bench.build_context_probe_argv(
                    bench_path=llama.bench_path,
                    model_path=model.path,
                    ctx=ctx,
                    config=champion,
                    capabilities=llama.capabilities,
                )
                if _execute(run, probe_dir, probe_argv, _PROBE_TIMEOUT_S) is not None:
                    config = champion
                else:
                    champion_failed_at = ctx
            for fallback_index, candidate in enumerate(
                _fallbacks(champion, model) if config is None else (), 2
            ):
                probe_dir = run.matrix_dir(ctx, depth, -fallback_index)
                probe_argv = bench.build_context_probe_argv(
                    bench_path=llama.bench_path,
                    model_path=model.path,
                    ctx=ctx,
                    config=candidate,
                    capabilities=llama.capabilities,
                )
                if _execute(run, probe_dir, probe_argv, _PROBE_TIMEOUT_S) is not None:
                    config = candidate
                    break
            if config is None:
                status = (
                    "pruned"
                    if champion_failed_at is not None and ctx > champion_failed_at
                    else "failed"
                )
                cell = MatrixCell(
                    ctx=ctx,
                    depth=depth,
                    status=status,
                    config=None,
                    pp=None,
                    tg=None,
                    refined=False,
                    evidence=directory,
                )
                rows.append(cell)
                run.append({"type": "matrix_cell", **_cell_dict(cell)})
                continue
            argv = bench.build_bench_argv(
                bench_path=llama.bench_path,
                model_path=model.path,
                pp=options.pp,
                tg=options.tg,
                reps=_reps(options),
                config=config,
                capabilities=llama.capabilities,
                depth=depth,
            )
            measured = _execute(run, directory, argv, _TRIAL_TIMEOUT_S)
            best_config = config
            best = measured
            refined = False
            if measured is not None and options.matrix_refine:
                for index, candidate in enumerate(_refinements(config), 1):
                    if remaining_minutes_fn() <= 0:
                        break
                    refine_dir = run.matrix_dir(ctx, depth, index)
                    refine_argv = bench.build_bench_argv(
                        bench_path=llama.bench_path,
                        model_path=model.path,
                        pp=options.pp,
                        tg=options.tg,
                        reps=_reps(options),
                        config=candidate,
                        capabilities=llama.capabilities,
                        depth=depth,
                    )
                    candidate_value = _execute(run, refine_dir, refine_argv, _TRIAL_TIMEOUT_S)
                    if (
                        candidate_value is not None
                        and best is not None
                        and _score(candidate_value, weights) > _score(best, weights)
                    ):
                        best, best_config, refined = candidate_value, candidate, True
                    if _cooldown(options) > 0:
                        time.sleep(_cooldown(options))
            cell = MatrixCell(
                ctx=ctx,
                depth=depth,
                status="ok" if best is not None else "failed",
                config=best_config,
                pp=best[0] if best else None,
                tg=best[1] if best else None,
                refined=refined,
                evidence=directory,
            )
            rows.append(cell)
            run.append({"type": "matrix_cell", **_cell_dict(cell)})
    return tuple(rows)


def _cell_dict(cell: MatrixCell) -> dict[str, Any]:
    return {
        "ctx": cell.ctx,
        "depth": cell.depth,
        "status": cell.status,
        "config": cell.config.to_dict() if cell.config else None,
        "pp": cell.pp,
        "tg": cell.tg,
        "refined": cell.refined,
        "evidence": str(cell.evidence) if cell.evidence else None,
    }
