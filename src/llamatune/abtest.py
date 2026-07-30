"""Interleaved A/B measurements for Marathon."""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any, Protocol

from llamatune import bench, executor
from llamatune.stats import weights_for_target
from llamatune.types import ABResult, LlamaCppReport, MarathonOptions, ModelReport, TrialConfig

_TIMEOUT_S = 3600.0


class MarathonRun(Protocol):
    dir: Path

    def ab_dir(self, label: str, block: int, slot: str) -> Path: ...

    def append(self, entry: dict[str, Any]) -> None: ...

    def write_json(self, name: str, payload: dict[str, Any]) -> None: ...


def _reps(options: MarathonOptions) -> int:
    return options.reps_confirm if options.reps_confirm is not None else 12


def _cooldown(options: MarathonOptions) -> float:
    return options.cooldown_s if options.cooldown_s is not None else 10.0


def _argv(
    config: TrialConfig | None,
    model: ModelReport,
    llama: LlamaCppReport,
    options: MarathonOptions,
) -> tuple[str, ...]:
    if config is None:
        return bench.build_baseline_argv(
            bench_path=llama.bench_path,
            model_path=model.path,
            pp=options.pp,
            tg=options.tg,
            reps=_reps(options),
            capabilities=llama.capabilities,
        )
    return bench.build_bench_argv(
        bench_path=llama.bench_path,
        model_path=model.path,
        pp=options.pp,
        tg=options.tg,
        reps=_reps(options),
        config=config,
        capabilities=llama.capabilities,
    )


def _payload(argv: tuple[str, ...], result: executor.ExecResult) -> dict[str, Any]:
    return {
        "argv": list(argv),
        "timeout_s": _TIMEOUT_S,
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


def _measure(
    run: MarathonRun,
    config: TrialConfig | None,
    *,
    block: int,
    slot: str,
    label: str,
    model: ModelReport,
    llama: LlamaCppReport,
    options: MarathonOptions,
) -> tuple[float, float]:
    directory = run.ab_dir(label, block, slot)
    argv = _argv(config, model, llama, options)
    result = executor.run(
        argv,
        timeout_s=_TIMEOUT_S,
        stdout_path=directory / "stdout.json",
        stderr_path=directory / "stderr.log",
    )
    relative = (directory / "command.json").relative_to(run.dir)
    run.write_json(str(relative), _payload(argv, result))
    if result.exit_code != 0 or result.timed_out or result.stdout.truncated:
        raise RuntimeError(f"A/B invocation failed: {label} block {block} slot {slot}")
    try:
        sample = bench.parse_bench_output(result.stdout.path.read_bytes())
    except (OSError, bench.BenchParseError) as exc:
        raise RuntimeError(
            f"A/B output could not be parsed: {label} block {block} slot {slot}"
        ) from exc
    return sample.pp_avg, sample.tg_avg


def _score(pp: float, tg: float, weights: tuple[float, float]) -> float:
    return float((pp ** weights[0]) * (tg ** weights[1]))


def _block_verdict(
    a: tuple[float, float],
    b: tuple[float, float],
    *,
    threshold: float,
    weights: tuple[float, float],
) -> tuple[str, float]:
    a_score = _score(*a, weights)
    b_score = _score(*b, weights)
    if a_score > b_score * (1.0 + threshold):
        return "a", a_score / b_score - 1.0
    if b_score > a_score * (1.0 + threshold):
        return "b", -(b_score / a_score - 1.0)
    return "tie", a_score / b_score - 1.0


def _pooled_verdict(
    *,
    a_wins: int,
    b_wins: int,
    blocks: int,
    a_pp: float,
    a_tg: float,
    b_pp: float,
    b_tg: float,
    threshold: float,
    target: str,
) -> str:
    majority = blocks // 2 + 1
    dominant = 0 if target == "prompt" else 1
    a_metrics = (a_pp, a_tg)
    b_metrics = (b_pp, b_tg)
    if a_wins >= majority and b_metrics[dominant] <= a_metrics[dominant] * (1.0 + threshold):
        return "a"
    if b_wins >= majority and a_metrics[dominant] <= b_metrics[dominant] * (1.0 + threshold):
        return "b"
    return "tie"


def run_ab(
    run: MarathonRun,
    a_config: TrialConfig | None,
    b_config: TrialConfig | None,
    *,
    blocks: int,
    model: ModelReport,
    llama: LlamaCppReport,
    options: MarathonOptions,
    label: str,
) -> ABResult:
    """Run ABBA blocks and return the majority/pooled verdict."""
    if blocks < 1:
        raise ValueError("blocks must be positive")
    threshold = max(0.01, float(getattr(run, "decision_threshold", 0.01)))
    weights = weights_for_target(options.target)
    all_a: list[tuple[float, float]] = []
    all_b: list[tuple[float, float]] = []
    wins = {"a": 0, "b": 0, "tie": 0}
    margins: list[float] = []
    order = (("a1", a_config), ("b1", b_config), ("b2", b_config), ("a2", a_config))
    for block in range(1, blocks + 1):
        values: dict[str, tuple[float, float]] = {}
        for index, (slot, config) in enumerate(order):
            values[slot] = _measure(
                run,
                config,
                block=block,
                slot=slot,
                label=label,
                model=model,
                llama=llama,
                options=options,
            )
            if (block != blocks or index != len(order) - 1) and _cooldown(options) > 0:
                time.sleep(_cooldown(options))
        a_pair = (
            statistics.fmean((values["a1"][0], values["a2"][0])),
            statistics.fmean((values["a1"][1], values["a2"][1])),
        )
        b_pair = (
            statistics.fmean((values["b1"][0], values["b2"][0])),
            statistics.fmean((values["b1"][1], values["b2"][1])),
        )
        verdict, margin = _block_verdict(a_pair, b_pair, threshold=threshold, weights=weights)
        wins[verdict] += 1
        margins.append(margin)
        all_a.extend((values["a1"], values["a2"]))
        all_b.extend((values["b1"], values["b2"]))
        run.append(
            {
                "type": "ab_block",
                "label": label,
                "block": block,
                "a_pp": a_pair[0],
                "a_tg": a_pair[1],
                "b_pp": b_pair[0],
                "b_tg": b_pair[1],
                "margin": margin,
                "verdict": verdict,
            }
        )
    a_pp = statistics.fmean(value[0] for value in all_a)
    a_tg = statistics.fmean(value[1] for value in all_a)
    b_pp = statistics.fmean(value[0] for value in all_b)
    b_tg = statistics.fmean(value[1] for value in all_b)
    verdict = _pooled_verdict(
        a_wins=wins["a"],
        b_wins=wins["b"],
        blocks=blocks,
        a_pp=a_pp,
        a_tg=a_tg,
        b_pp=b_pp,
        b_tg=b_tg,
        threshold=threshold,
        target=options.target,
    )
    return ABResult(
        label=label,
        blocks=blocks,
        a_wins=wins["a"],
        b_wins=wins["b"],
        ties=wins["tie"],
        a_pp=a_pp,
        a_tg=a_tg,
        b_pp=b_pp,
        b_tg=b_tg,
        margin=statistics.fmean(margins),
        verdict=verdict,
    )
