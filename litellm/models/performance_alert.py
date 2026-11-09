from datetime import date, datetime, time, timedelta, timezone
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from litellm.models.alert import MAX_ALERT_ID, AlertDetection, AlertResourceType, AlertType, MetricEvidence


class DailyMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    spend: float = Field(default=0, ge=0, allow_inf_nan=False)
    requests: int = Field(default=0, ge=0, le=MAX_ALERT_ID)
    successful: int = Field(default=0, ge=0, le=MAX_ALERT_ID)
    failed: int = Field(default=0, ge=0, le=MAX_ALERT_ID)
    response_ms: int = Field(default=0, ge=0, le=MAX_ALERT_ID)
    timed_requests: int = Field(default=0, ge=0, le=MAX_ALERT_ID)


class PerformanceWindow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    day: date
    resource_type: AlertResourceType
    resource_id: str = Field(min_length=1, max_length=1024)
    days: tuple[DailyMetrics, ...] = Field(min_length=8, max_length=8)
    invalid: bool = False

    @property
    def priority(self) -> float:
        return max(self.days[-1].requests, sum(point.requests for point in self.days[:-1]) / 7)


def detect_performance_alerts(window: PerformanceWindow) -> tuple[AlertDetection, ...]:
    history: Final = window.days[:-1]
    today: Final = window.days[-1]
    observed_days: Final = sum(point.requests > 0 for point in history)
    if window.invalid or observed_days < 3:
        return ()
    baseline_spend: Final = sum(point.spend / 7 for point in history)
    baseline_requests: Final = sum(point.requests for point in history) / 7
    historical_timed: Final = sum(point.timed_requests for point in history)
    historical_requests: Final = sum(point.successful + point.failed for point in history)
    current_requests: Final = today.successful + today.failed
    baseline_latency: Final = sum(point.response_ms for point in history) / historical_timed if historical_timed else 0
    current_latency: Final = today.response_ms / today.timed_requests if today.timed_requests else 0
    baseline_error_rate: Final = (
        sum(point.failed for point in history) / historical_requests if historical_requests else 0
    )
    current_error_rate: Final = today.failed / current_requests if current_requests else 0

    comparisons: Final[
        tuple[tuple[bool, AlertType, float, float, Literal["usd", "requests", "milliseconds", "ratio"], int], ...]
    ] = (
        (today.spend >= max(10, 3 * baseline_spend), "spend", today.spend, baseline_spend, "usd", today.requests),
        (
            today.requests >= max(1000, 10 * baseline_requests)
            or (observed_days == 7 and baseline_requests >= 500 and today.requests < baseline_requests / 5),
            "traffic",
            today.requests,
            baseline_requests,
            "requests",
            today.requests,
        ),
        (
            today.timed_requests >= 100
            and historical_timed >= 100
            and current_latency >= max(500, 3 * baseline_latency),
            "latency",
            current_latency,
            baseline_latency,
            "milliseconds",
            today.timed_requests,
        ),
        (
            500 <= current_requests <= MAX_ALERT_ID
            and historical_requests >= 500
            and current_error_rate >= max(0.02, 3 * baseline_error_rate),
            "reliability",
            current_error_rate,
            baseline_error_rate,
            "ratio",
            current_requests,
        ),
    )
    start: Final = datetime.combine(window.day, time.min, timezone.utc)
    return tuple(
        AlertDetection(
            type=kind,
            resource_type=window.resource_type,
            resource_id=window.resource_id,
            window_start=start,
            window_end=start + timedelta(days=1),
            evidence=MetricEvidence(value=value, baseline=baseline, unit=unit, sample_size=samples),
        )
        for triggered, kind, value, baseline, unit, samples in comparisons
        if triggered
    )


class PerformanceProgress(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    day: date
    phase: Literal["cleanup", "collect", "evaluate", "done"] = "cleanup"
    source: Literal["user", "team", "gateway"] = "user"
    day_index: int = Field(default=0, ge=0, le=7)
    cursor: str = Field(default="", max_length=4096)
    route: str = Field(default="", max_length=1024)
    scanned: int = Field(default=0, ge=0)
    evaluated: int = Field(default=0, ge=0)
    discovered: int = Field(default=0, ge=0)
    invalid: int = Field(default=0, ge=0)
