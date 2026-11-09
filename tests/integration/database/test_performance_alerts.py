import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import psycopg
import pytest
import pytest_asyncio
from psycopg import sql

from litellm.models.performance_alert import PerformanceProgress
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.repositories.alert_repository import AlertRepository
from litellm.repositories.performance_alert_repository import PerformanceAlertRepository
from tests.integration._support.database import scratch_database

_MIGRATIONS: Final = Path(__file__).resolve().parents[3] / "litellm-proxy-extras/litellm_proxy_extras/migrations"
_DAY: Final = date(2026, 1, 8)
_KEY: Final = "a" * 64


@dataclass(frozen=True, slots=True)
class Detector:
    connection: psycopg.Connection[tuple[object, ...]]
    repository: PerformanceAlertRepository
    alerts: AlertRepository

    async def collect(self, owner: str) -> PerformanceProgress:
        for _ in range(100):
            progress: Final = await self.repository.collect(owner)
            assert progress is not None
            if progress.phase == "evaluate":
                return progress
        pytest.fail("Collection did not finish")

    def seed(self, *, current_requests: int = 10000, resource: str = "example", key: str = _KEY) -> None:
        for day in range(8):
            for table, entity in (("LiteLLM_DailyUserSpend", "user_id"), ("LiteLLM_DailyTeamSpend", "team_id")):
                self.connection.execute(
                    sql.SQL(
                        "INSERT INTO {} (id, date, {}, api_key, model, model_group, spend, api_requests, "
                        "successful_requests, failed_requests, total_response_time_ms, timed_requests) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                    ).format(sql.Identifier(table), sql.Identifier(entity)),
                    (
                        f"{resource}-{day:02}",
                        (_DAY - timedelta(days=7 - day)).isoformat(),
                        resource,
                        key,
                        "example-provider-model",
                        "example-model",
                        2 if day < 7 else 20,
                        1000 if day < 7 else current_requests,
                        990 if day < 7 else current_requests * 9 // 10,
                        10 if day < 7 else current_requests // 10,
                        100000 if day < 7 else current_requests * 1000,
                        1000 if day < 7 else current_requests,
                    ),
                )


@pytest_asyncio.fixture(loop_scope="function")
async def detector(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Detector]:
    with scratch_database("performance_alerts") as url, psycopg.connect(url, autocommit=True) as connection:
        connection.execute(
            'CREATE TABLE "LiteLLM_Config" (param_name text PRIMARY KEY, param_value jsonb, reload_revision bigint)'
        )
        connection.execute('CREATE TABLE "LiteLLM_VerificationToken" (token text PRIMARY KEY)')
        connection.execute('INSERT INTO "LiteLLM_VerificationToken" VALUES (%s)', (_KEY,))
        connection.execute(
            'CREATE TABLE "LiteLLM_SpendLogDelivery" (state text, created_at timestamp, next_attempt_at timestamp)'
        )
        connection.execute(
            'CREATE TABLE "LiteLLM_DailyGatewayRequests" (date text, category text, route text, '
            "successful_requests bigint, failed_requests bigint, updated_at timestamp DEFAULT NOW(), "
            "PRIMARY KEY (date, category, route))"
        )
        connection.execute(
            'INSERT INTO "LiteLLM_DailyGatewayRequests" VALUES '
            "(to_char(NOW() AT TIME ZONE 'UTC', 'YYYY-MM-DD'), 'llm', '/v1/chat/completions', 1, 0, NOW())"
        )
        for day in range(8):
            connection.execute(
                'INSERT INTO "LiteLLM_DailyGatewayRequests" (date, category, route, successful_requests, '
                "failed_requests) VALUES (%s, 'llm', '/v1/chat/completions', %s, %s)",
                ((_DAY - timedelta(days=7 - day)).isoformat(), 990 if day < 7 else 9000, 10 if day < 7 else 1000),
            )
        for table, entity in (("LiteLLM_DailyUserSpend", "user_id"), ("LiteLLM_DailyTeamSpend", "team_id")):
            connection.execute(
                sql.SQL(
                    "CREATE TABLE {} (id text PRIMARY KEY, date text, {} text, api_key text, "
                    "model text, model_group text, spend double precision, api_requests bigint, "
                    "successful_requests bigint, failed_requests bigint, total_response_time_ms bigint, "
                    "timed_requests bigint)"
                ).format(sql.Identifier(table), sql.Identifier(entity))
            )
        for migration in (
            "20261109080000_performance_alert_work",
            "20261109080100_performance_alert_user_scan",
            "20261109080200_performance_alert_team_scan",
        ):
            connection.execute((_MIGRATIONS / migration / "migration.sql").read_text())
        alert_migration: Final = next(_MIGRATIONS.glob("*_add_alerts/migration.sql"))
        connection.execute(alert_migration.read_text())
        monkeypatch.setenv("DATABASE_URL", url)
        client: Final = PrismaClient(url, ProxyLogging(UserApiKeyCache()))
        await client.connect()
        try:
            yield Detector(connection, PerformanceAlertRepository(client), AlertRepository(client))
        finally:
            await client.disconnect()


@pytest.mark.asyncio
async def test_all_scopes_and_families_are_persisted_without_double_counting(detector: Detector) -> None:
    detector.seed()
    assert await detector.repository.claim(_DAY, "worker") is not None
    collected: Final = await detector.collect("worker")
    assert collected.discovered == 5
    assert (await detector.repository.evaluate("worker")).evaluated == 5
    alerts: Final = (await detector.alerts.list("open", 100, None)).data
    assert len(alerts) == 20
    assert {alert.resource_type for alert in alerts} == {"gateway", "model", "team", "user", "key"}
    assert {alert.resource_id for alert in alerts} == {"gateway", "example-model", "example", _KEY}
    for alert in alerts:
        assert (alert.evidence.value, alert.evidence.baseline) == {
            "spend": (20, 2),
            "traffic": (10000, 1000),
            "latency": (1000, 100),
            "reliability": (0.1, 0.01),
        }[alert.type]
    saved: Final = alerts[0]
    await detector.alerts.set_status(int(saved.id), "resolved", "reviewer")
    detector.connection.execute('UPDATE "LiteLLM_PerformanceAlertWork" SET evaluated=false')
    await detector.repository.evaluate("worker")
    assert (await detector.alerts.get(int(saved.id))).status == "resolved"
    assert len((await detector.alerts.list("open", 100, None)).data) == 19


@pytest.mark.asyncio
async def test_missing_current_day_keeps_traffic_drop_and_jwt_user_without_inventing_a_key(detector: Detector) -> None:
    detector.seed(key="example-jwt-user")
    for table in ("LiteLLM_DailyUserSpend", "LiteLLM_DailyTeamSpend"):
        detector.connection.execute(
            sql.SQL("DELETE FROM {} WHERE date=%s").format(sql.Identifier(table)), (_DAY.isoformat(),)
        )
    await detector.repository.claim(_DAY, "worker")
    await detector.collect("worker")
    await detector.repository.evaluate("worker")
    alerts: Final = (await detector.alerts.list("open", 100, None)).data
    assert {a.resource_type for a in alerts if a.type == "traffic" and a.evidence.value == 0} == {
        "model",
        "team",
        "user",
    }
    assert not any(a.resource_type == "key" for a in alerts)


@pytest.mark.asyncio
async def test_one_owner_resumes_pages_and_new_day_replaces_old_work(detector: Detector) -> None:
    detector.seed()
    claims: Final = await asyncio.gather(*(detector.repository.claim(_DAY, str(i)) for i in range(2)))
    assert sum(progress is not None for progress in claims) == 1
    owner: Final = str(next(i for i, progress in enumerate(claims) if progress is not None))
    await detector.repository.collect(owner)
    page: Final = await detector.repository.collect(owner)
    await detector.repository.release(owner)
    assert await detector.repository.claim(_DAY, "replacement") == page
    assert await detector.repository.collect(owner) is None
    collected: Final = await detector.collect("replacement")
    assert collected.scanned == 24
    await detector.repository.evaluate("replacement")
    await detector.repository.evaluate("replacement")
    await detector.repository.release("replacement")
    assert await detector.repository.claim(_DAY, "again") is None
    assert (await detector.repository.claim(_DAY + timedelta(days=1), "new-day")).phase == "cleanup"
    await detector.repository.collect("new-day")
    assert detector.connection.execute('SELECT count(*) FROM "LiteLLM_PerformanceAlertWork"').fetchone() == (0,)


@pytest.mark.asyncio
async def test_backlog_invalidates_partial_totals_even_when_retry_is_scheduled_in_the_future(
    detector: Detector,
) -> None:
    detector.seed()
    await detector.repository.claim(_DAY, "worker")
    await detector.repository.collect("worker")
    await detector.repository.collect("worker")
    detector.connection.execute(
        'INSERT INTO "LiteLLM_SpendLogDelivery" VALUES '
        "('pending', NOW() - interval '10 minutes', NOW() + interval '1 hour')"
    )
    assert await detector.repository.collect("worker") is None
    detector.connection.execute('DELETE FROM "LiteLLM_SpendLogDelivery"')
    await detector.repository.release("worker")
    assert (await detector.repository.claim(_DAY, "replacement")).phase == "cleanup"
    collected: Final = await detector.collect("replacement")
    assert collected.scanned == 24
    await detector.repository.evaluate("replacement")
    assert len((await detector.alerts.list("open", 100, None)).data) == 20


@pytest.mark.asyncio
async def test_stale_gateway_data_defers_detection(detector: Detector) -> None:
    detector.connection.execute("UPDATE \"LiteLLM_DailyGatewayRequests\" SET updated_at=NOW() - interval '1 hour'")
    assert await detector.repository.claim(_DAY, "worker") is None
    assert (await detector.alerts.list("open", 100, None)).data == ()


@pytest.mark.asyncio
async def test_large_text_is_not_copied_and_invalid_metrics_do_not_generate_alerts(detector: Detector) -> None:
    detector.seed(resource="é" * 1024, key="x" * 102400)
    detector.connection.execute(
        "UPDATE \"LiteLLM_DailyUserSpend\" SET model_group=%s, spend='Infinity'", ("x" * 102400,)
    )
    await detector.repository.claim(_DAY, "worker")
    await detector.collect("worker")
    result: Final = await detector.repository.evaluate("worker")
    assert result.invalid == 2
    alerts: Final = (await detector.alerts.list("open", 100, None)).data
    assert {a.resource_type for a in alerts} == {"team"}
    assert detector.connection.execute(
        'SELECT max(octet_length(data::text)) < 8192 FROM "LiteLLM_PerformanceAlertWork"'
    ).fetchone() == (True,)


@pytest.mark.asyncio
async def test_global_priority_checks_busy_resources_first_then_resumes_the_tail(detector: Detector) -> None:
    for index in range(40):
        detector.seed(resource=f"resource-{index:02}", current_requests=10000 + index * 1000)
    await detector.repository.claim(_DAY, "worker")
    await detector.collect("worker")
    first: Final = await detector.repository.evaluate("worker")
    assert first is not None and first.evaluated == 25
    alerts: Final = (await detector.alerts.list("open", 100, None)).data
    assert {a.resource_type for a in alerts if a.resource_id == "resource-39"} == {"user", "team"}
    assert not any(a.resource_id == "resource-00" for a in alerts)
    await detector.repository.release("worker")
    assert await detector.repository.claim(_DAY, "replacement") == first
    for _ in range(4):
        await detector.repository.evaluate("replacement")
    assert detector.connection.execute(
        'SELECT count(DISTINCT resource_id) FROM "LiteLLM_Alert" WHERE resource_type=%s', ("user",)
    ).fetchone() == (40,)


@pytest.mark.asyncio
async def test_alert_write_timeout_rolls_back_progress_and_can_resume(detector: Detector) -> None:
    from prisma.errors import RawQueryError

    detector.seed()
    await detector.repository.claim(_DAY, "worker")
    collected: Final = await detector.collect("worker")
    with psycopg.connect(detector.connection.info.dsn) as locker:
        locker.execute('LOCK TABLE "LiteLLM_Alert" IN ACCESS EXCLUSIVE MODE')
        with pytest.raises(RawQueryError, match="lock timeout"):
            await detector.repository.evaluate("worker")
    await detector.repository.release("worker")
    assert await detector.repository.claim(_DAY, "replacement") == collected
    await detector.repository.evaluate("replacement")
    assert len((await detector.alerts.list("open", 100, None)).data) == 20
