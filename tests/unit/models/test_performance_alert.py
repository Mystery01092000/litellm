from datetime import date
from typing import Final

import pytest

from litellm.models.performance_alert import DailyMetrics, PerformanceWindow, detect_performance_alerts

DAY: Final = date(2026, 1, 8)
NORMAL: Final = DailyMetrics(spend=2, requests=1000, successful=990, failed=10, response_ms=100000, timed_requests=1000)


def window(today: DailyMetrics, history: tuple[DailyMetrics, ...] = (NORMAL,) * 7) -> PerformanceWindow:
    return PerformanceWindow(day=DAY, resource_type="team", resource_id="example-team", days=(*history, today))


def test_all_families_share_one_window_and_keep_original_metric_units() -> None:
    today: Final = DailyMetrics(
        spend=20, requests=10000, successful=9000, failed=1000, response_ms=10000000, timed_requests=10000
    )
    alerts: Final = detect_performance_alerts(window(today))
    assert tuple((a.type, a.evidence.value, a.evidence.baseline, a.evidence.unit) for a in alerts) == (
        ("spend", 20, 2, "usd"),
        ("traffic", 10000, 1000, "requests"),
        ("latency", 1000, 100, "milliseconds"),
        ("reliability", 0.1, 0.01, "ratio"),
    )
    assert all(a.window_start.date() == DAY and a.resource_id == "example-team" for a in alerts)


def test_zero_traffic_still_has_priority_and_produces_a_drop() -> None:
    candidate: Final = window(DailyMetrics())
    alerts: Final = detect_performance_alerts(candidate)
    assert candidate.priority == 1000
    assert tuple((a.type, a.evidence.value, a.evidence.sample_size) for a in alerts) == (("traffic", 0, 0),)


def test_newly_busy_resources_get_current_day_priority() -> None:
    assert window(DailyMetrics(requests=50000)).priority == 50000


def test_unchanged_usage_produces_no_alerts() -> None:
    assert detect_performance_alerts(window(NORMAL)) == ()


def test_new_resource_does_not_treat_missing_history_as_a_zero_baseline() -> None:
    assert detect_performance_alerts(window(DailyMetrics(spend=100, requests=50000), (DailyMetrics(),) * 7)) == ()


def test_rates_and_duration_use_weighted_totals() -> None:
    quiet: Final = DailyMetrics(requests=1, failed=1, response_ms=1000, timed_requests=1)
    busy: Final = DailyMetrics(requests=1000, successful=1000, response_ms=100000, timed_requests=1000)
    today: Final = DailyMetrics(requests=1000, successful=900, failed=100, response_ms=600000, timed_requests=1000)
    alerts: Final = detect_performance_alerts(window(today, (quiet,) * 6 + (busy,)))
    values: Final = {a.type: a.evidence.baseline for a in alerts}
    assert values["latency"] == pytest.approx(106000 / 1006)
    assert values["reliability"] == pytest.approx(6 / 1006)


@pytest.mark.parametrize(
    "today",
    (
        DailyMetrics(requests=10, successful=9, failed=1, response_ms=100000, timed_requests=10),
        DailyMetrics(requests=1000, successful=990, failed=10),
    ),
)
def test_low_volume_or_missing_timing_does_not_produce_latency_or_failure_alerts(today: DailyMetrics) -> None:
    assert not {a.type for a in detect_performance_alerts(window(today))} & {"latency", "reliability"}
