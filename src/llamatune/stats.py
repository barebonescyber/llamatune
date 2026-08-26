"""Statistics: metric aggregation, noise floor, scoring, Pareto front.

Pure functions over numbers and :class:`llamatune.types.MetricStats`; this
module imports nothing above the standard library plus that one dataclass
(DESIGN §§6, 11 and the §13 layering rules).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

from llamatune.types import MetricStats

#: Geometric-scoring weights (w_pp, w_tg) per ``--target`` (DESIGN §11).
TARGET_WEIGHTS: dict[str, tuple[float, float]] = {
    "balanced": (0.5, 0.5),
    "prompt": (0.8, 0.2),
    "generation": (0.2, 0.8),
}

#: The noise floor is never reported below this value (DESIGN §6).
NOISE_FLOOR_MIN = 0.01

#: A single trial whose internal cv exceeds this on either metric is unstable.
STABILITY_THRESHOLD = 0.10


def weights_for_target(target: str) -> tuple[float, float]:
    """Return the (w_pp, w_tg) scoring weights for a target name."""
    try:
        return TARGET_WEIGHTS[target]
    except KeyError:
        msg = f"unknown target: {target!r}"
        raise ValueError(msg) from None


def metric_stats(values: Sequence[float]) -> MetricStats:
    """Aggregate a sequence of measurements into mean/stdev/cv/n.

    Uses the *sample* standard deviation (Bessel-corrected) over independent
    runs; with fewer than two values the stdev and cv are zero.
    """
    n = len(values)
    if n == 0:
        return MetricStats(mean=0.0, stdev=0.0, cv=0.0, n=0)
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if n >= 2 else 0.0
    cv = stdev / mean if mean > 0 else 0.0
    return MetricStats(mean=mean, stdev=stdev, cv=cv, n=n)


def sample_stats(avg_ts: float, stddev_ts: float, reps: int) -> MetricStats:
    """Wrap one llama-bench sample (avg/stddev over ``reps``) as MetricStats."""
    cv = stddev_ts / avg_ts if avg_ts > 0 else 0.0
    return MetricStats(mean=avg_ts, stdev=stddev_ts, cv=cv, n=reps)


def noise_floor_cv(pp: MetricStats, tg: MetricStats) -> float:
    """Session noise floor: max cv over {pp, tg}, floored at NOISE_FLOOR_MIN."""
    return max(pp.cv, tg.cv, NOISE_FLOOR_MIN)


def is_unstable(pp: MetricStats, tg: MetricStats) -> bool:
    """True when either metric's internal cv exceeds the stability threshold."""
    return pp.cv > STABILITY_THRESHOLD or tg.cv > STABILITY_THRESHOLD


def score(*, pp: float, tg: float, pp0: float, tg0: float, weights: tuple[float, float]) -> float:
    """Geometric score ``(pp/pp0)**w_pp * (tg/tg0)**w_tg`` (DESIGN §11)."""
    if pp0 <= 0 or tg0 <= 0:
        return 0.0
    w_pp, w_tg = weights
    return float((pp / pp0) ** w_pp * (tg / tg0) ** w_tg)


def pareto_front(points: Sequence[tuple[float, float]]) -> list[int]:
    """Indices of the Pareto-optimal points over (pp, tg); higher is better.

    A point is dominated when another has pp and tg both >= it, with at least
    one strictly greater; exact duplicates (equal pp AND equal tg) never
    dominate each other and are all retained. Implemented as a sort by pp
    descending with a running tg maximum per equal-pp group (PERF-013),
    reproducing the quadratic definition exactly and returning indices in
    input order.
    """
    n = len(points)
    if n <= 1:
        return list(range(n))
    order = sorted(range(n), key=lambda i: (-points[i][0], -points[i][1]))
    result: list[int] = []
    start = 0
    best_tg_above = -math.inf
    while start < n:
        group_pp = points[order[start]][0]
        end = start
        group_best_tg = -math.inf
        while end < n and points[order[end]][0] == group_pp:
            tg = points[order[end]][1]
            if tg > group_best_tg:
                group_best_tg = tg
            end += 1
        for index in range(start, end):
            i = order[index]
            tg = points[i][1]
            # Survives iff no strictly-better-pp point reaches this tg
            # (best_tg_above < tg) and no same-pp point strictly exceeds it
            # (group_best_tg <= tg).
            if best_tg_above < tg and group_best_tg <= tg:
                result.append(i)
        if group_best_tg > best_tg_above:
            best_tg_above = group_best_tg
        start = end
    result.sort()
    return result


def metric_to_dict(metric: MetricStats) -> dict[str, Any]:
    """Serialize MetricStats for analysis.json / journal entries."""
    return {"mean": metric.mean, "stdev": metric.stdev, "cv": metric.cv, "n": metric.n}


def metric_from_dict(data: dict[str, Any]) -> MetricStats:
    """Inverse of :func:`metric_to_dict`."""
    return MetricStats(
        mean=float(data["mean"]),
        stdev=float(data["stdev"]),
        cv=float(data["cv"]),
        n=int(data["n"]),
    )
