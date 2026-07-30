"""Unit tests for llamatune.stats (aggregation, noise floor, scoring, Pareto)."""

from __future__ import annotations

import math

import pytest

from llamatune import stats
from llamatune.types import MetricStats


class TestMetricStats:
    def test_empty(self) -> None:
        result = stats.metric_stats([])
        assert result == MetricStats(mean=0.0, stdev=0.0, cv=0.0, n=0)

    def test_single_value_has_zero_spread(self) -> None:
        result = stats.metric_stats([42.0])
        assert result.mean == 42.0
        assert result.stdev == 0.0
        assert result.cv == 0.0
        assert result.n == 1

    def test_multiple_values_sample_stdev(self) -> None:
        result = stats.metric_stats([10.0, 12.0, 14.0])
        assert result.mean == pytest.approx(12.0)
        assert result.stdev == pytest.approx(2.0)  # sample stdev
        assert result.cv == pytest.approx(2.0 / 12.0)
        assert result.n == 3

    def test_zero_mean_yields_zero_cv(self) -> None:
        result = stats.metric_stats([0.0, 0.0])
        assert result.cv == 0.0


class TestSampleStats:
    def test_wraps_bench_sample(self) -> None:
        result = stats.sample_stats(100.0, 5.0, 3)
        assert result.mean == 100.0
        assert result.stdev == 5.0
        assert result.cv == pytest.approx(0.05)
        assert result.n == 3

    def test_zero_avg_yields_zero_cv(self) -> None:
        assert stats.sample_stats(0.0, 1.0, 3).cv == 0.0


class TestNoiseFloor:
    def test_floored_at_minimum(self) -> None:
        pp = MetricStats(mean=100.0, stdev=0.1, cv=0.001, n=3)
        tg = MetricStats(mean=10.0, stdev=0.01, cv=0.001, n=3)
        assert stats.noise_floor_cv(pp, tg) == stats.NOISE_FLOOR_MIN

    def test_takes_max_cv_when_above_floor(self) -> None:
        pp = MetricStats(mean=100.0, stdev=2.0, cv=0.02, n=3)
        tg = MetricStats(mean=10.0, stdev=0.5, cv=0.05, n=3)
        assert stats.noise_floor_cv(pp, tg) == 0.05


class TestIsUnstable:
    def test_stable(self) -> None:
        metric = MetricStats(mean=100.0, stdev=1.0, cv=0.01, n=3)
        assert not stats.is_unstable(metric, metric)

    def test_unstable_on_either_metric(self) -> None:
        good = MetricStats(mean=100.0, stdev=1.0, cv=0.01, n=3)
        bad = MetricStats(mean=100.0, stdev=15.0, cv=0.15, n=3)
        assert stats.is_unstable(bad, good)
        assert stats.is_unstable(good, bad)


class TestScore:
    def test_balanced_is_geometric_mean(self) -> None:
        score = stats.score(pp=200.0, tg=20.0, pp0=100.0, tg0=10.0, weights=(0.5, 0.5))
        assert score == pytest.approx(2.0)

    def test_prompt_weighting(self) -> None:
        score = stats.score(pp=200.0, tg=10.0, pp0=100.0, tg0=10.0, weights=(0.8, 0.2))
        assert score == pytest.approx(2.0**0.8)

    def test_generation_weighting(self) -> None:
        score = stats.score(pp=100.0, tg=20.0, pp0=100.0, tg0=10.0, weights=(0.2, 0.8))
        assert score == pytest.approx(2.0**0.8)

    def test_zero_baseline_scores_zero(self) -> None:
        assert stats.score(pp=100.0, tg=10.0, pp0=0.0, tg0=10.0, weights=(0.5, 0.5)) == 0.0
        assert stats.score(pp=100.0, tg=10.0, pp0=100.0, tg0=0.0, weights=(0.5, 0.5)) == 0.0

    def test_equal_to_baseline_is_one(self) -> None:
        score = stats.score(pp=100.0, tg=10.0, pp0=100.0, tg0=10.0, weights=(0.5, 0.5))
        assert math.isclose(score, 1.0)


class TestWeightsForTarget:
    @pytest.mark.parametrize(
        ("target", "expected"),
        [("balanced", (0.5, 0.5)), ("prompt", (0.8, 0.2)), ("generation", (0.2, 0.8))],
    )
    def test_known_targets(self, target: str, expected: tuple[float, float]) -> None:
        assert stats.weights_for_target(target) == expected

    def test_unknown_target_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown target"):
            stats.weights_for_target("speed")


class TestParetoFront:
    def test_empty(self) -> None:
        assert stats.pareto_front([]) == []

    def test_single_point(self) -> None:
        assert stats.pareto_front([(1.0, 1.0)]) == [0]

    def test_dominated_point_excluded(self) -> None:
        points = [(100.0, 10.0), (90.0, 9.0), (80.0, 20.0)]
        assert stats.pareto_front(points) == [0, 2]

    def test_strictly_dominating_point_wins(self) -> None:
        points = [(1.0, 1.0), (2.0, 2.0)]
        assert stats.pareto_front(points) == [1]

    def test_identical_points_all_retained(self) -> None:
        points = [(5.0, 5.0), (5.0, 5.0)]
        assert stats.pareto_front(points) == [0, 1]

    def test_partial_tie_dominates(self) -> None:
        # (5, 6) dominates (5, 5): equal pp, strictly better tg.
        points = [(5.0, 5.0), (5.0, 6.0)]
        assert stats.pareto_front(points) == [1]


class TestMetricDictRoundTrip:
    def test_round_trip(self) -> None:
        metric = MetricStats(mean=123.4, stdev=5.6, cv=0.045, n=5)
        assert stats.metric_from_dict(stats.metric_to_dict(metric)) == metric
