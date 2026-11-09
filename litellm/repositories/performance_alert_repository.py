import hashlib
import json
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import date, timedelta
from functools import reduce
from itertools import chain, groupby
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast  # noqa: TID251  # Prisma generates transaction types

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.models.performance_alert import (
    DailyMetrics,
    PerformanceProgress,
    PerformanceWindow,
    detect_performance_alerts,
)
from litellm.repositories.alert_repository import record_alerts

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


class _Transaction(Protocol):
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]: ...
    async def execute_raw(self, query: str, *args: object) -> int: ...


class _Database(Protocol):
    def tx(self, *, timeout: timedelta) -> AbstractAsyncContextManager[_Transaction]: ...


class _ProgressRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    progress: PerformanceProgress


class _SourceRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    route: str = ""
    entity_id: str = ""
    model: str = ""
    api_key: str = ""
    metrics: object


class _WorkRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    resource_hash: str
    data: PerformanceWindow


_PROGRESS: Final = TypeAdapter(tuple[_ProgressRow, ...])
_SOURCE: Final = TypeAdapter(tuple[_SourceRow, ...])
_WORK: Final = TypeAdapter(tuple[_WorkRow, ...])
_NAME: Final = "performance_alerts_progress"
_PAGE_ROWS: Final = 100
_PAGE_BYTES: Final = 262144
_ZERO: Final = DailyMetrics()
_SET_LIMITS: Final = """SELECT set_config('statement_timeout', '750ms', true),
    set_config('lock_timeout', '50ms', true), set_config('work_mem', '1MB', true),
    set_config('max_parallel_workers_per_gather', '0', true)"""
_CLAIM: Final = """INSERT INTO "LiteLLM_Config" (param_name, param_value, reload_revision)
VALUES ($1, jsonb_build_object('owner', $2::text, 'lease_until', NOW() + interval '90 seconds',
    'progress', $3::jsonb), 0)
ON CONFLICT (param_name) DO UPDATE SET param_value = jsonb_build_object(
    'owner', $2::text, 'lease_until', NOW() + interval '90 seconds',
    'progress', CASE WHEN "LiteLLM_Config".param_value->'progress'->>'day' = $4
        THEN "LiteLLM_Config".param_value->'progress' ELSE $3::jsonb END)
WHERE COALESCE(("LiteLLM_Config".param_value->>'lease_until')::timestamptz, '-infinity') < NOW()
    AND ("LiteLLM_Config".param_value->'progress'->>'day' IS DISTINCT FROM $4
         OR "LiteLLM_Config".param_value->'progress'->>'phase' <> 'done')
RETURNING param_value->'progress' AS progress"""
_LOCK: Final = """SELECT param_value->'progress' AS progress FROM "LiteLLM_Config"
WHERE param_name = $1 AND param_value->>'owner' = $2
    AND (param_value->>'lease_until')::timestamptz > NOW() FOR UPDATE NOWAIT"""
_SAVE: Final = """UPDATE "LiteLLM_Config" SET param_value = jsonb_set(param_value, '{progress}', $3::jsonb)
WHERE param_name = $1 AND param_value->>'owner' = $2"""
_UPSERT: Final = """INSERT INTO "LiteLLM_PerformanceAlertWork"
    (day, resource_hash, priority, data, evaluated)
SELECT $1::date, resource_hash, priority, data, false
FROM jsonb_to_recordset($2::jsonb) AS rows(resource_hash text, priority double precision, data jsonb)
ON CONFLICT (day, resource_hash) DO UPDATE SET data = EXCLUDED.data, priority = EXCLUDED.priority"""
_PENDING: Final = """SELECT resource_hash, data FROM "LiteLLM_PerformanceAlertWork"
WHERE day = $1::date AND evaluated = false ORDER BY priority DESC, resource_hash LIMIT 25"""
_ACCOUNTING_DELAYED: Final = """SELECT 1 FROM "LiteLLM_SpendLogDelivery" WHERE state = 'parked'
UNION ALL SELECT 1 FROM "LiteLLM_SpendLogDelivery"
WHERE state = 'pending' AND created_at < NOW() - interval '5 minutes'
UNION ALL SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM "LiteLLM_DailyGatewayRequests"
    WHERE date = to_char(NOW() AT TIME ZONE 'UTC', 'YYYY-MM-DD')
      AND updated_at >= NOW() - interval '15 minutes') LIMIT 1"""


def _source_sql(source: Literal["user", "team", "gateway"]) -> str:
    if source == "gateway":
        return """SELECT category AS id, route,
            jsonb_build_object('requests', successful_requests::numeric + failed_requests,
                'successful', successful_requests, 'failed', failed_requests) AS metrics
        FROM "LiteLLM_DailyGatewayRequests" WHERE date = $1 AND (category, route) > ($2, $3)
        ORDER BY category, route LIMIT 100"""
    table: Final = "LiteLLM_DailyUserSpend" if source == "user" else "LiteLLM_DailyTeamSpend"
    entity: Final = "user_id" if source == "user" else "team_id"
    return f"""WITH page AS MATERIALIZED (
        SELECT id,
            CASE WHEN octet_length({entity}) > 4096 THEN ''
                WHEN length({entity}) <= 1024 THEN COALESCE({entity}, '') ELSE '' END AS entity_id,
            CASE WHEN octet_length(COALESCE(NULLIF(model_group, ''), model)) > 4096 THEN ''
                WHEN length(COALESCE(NULLIF(model_group, ''), model)) <= 1024 THEN COALESCE(NULLIF(model_group, ''), model, '') ELSE '' END AS model,
            CASE WHEN octet_length(api_key) = 64 AND api_key ~ '^[0-9a-f]{{64}}$' AND EXISTS (
                SELECT 1 FROM "LiteLLM_VerificationToken" AS keys WHERE keys.token = api_key)
                THEN api_key ELSE '' END AS api_key,
            jsonb_build_object('spend', spend, 'requests', api_requests,
                'successful', successful_requests, 'failed', failed_requests,
                'response_ms', total_response_time_ms, 'timed_requests', timed_requests) AS metrics
        FROM "{table}" WHERE date = $1 AND id > $2 ORDER BY id LIMIT $3::integer
    ), sized AS (
        SELECT *, SUM(octet_length(id) + octet_length(entity_id) + octet_length(model)
            + octet_length(api_key) + octet_length(metrics::text) + 256) OVER (ORDER BY id) AS bytes FROM page
    ) SELECT id, entity_id, model, api_key, metrics FROM sized WHERE bytes <= $4::integer ORDER BY id"""


def _hash(window: PerformanceWindow) -> str:
    return hashlib.sha256(json.dumps((window.resource_type, window.resource_id)).encode()).hexdigest()


def _project(row: _SourceRow, progress: PerformanceProgress) -> tuple[PerformanceWindow, ...]:
    try:
        metrics: Final = DailyMetrics.model_validate(row.metrics)
    except ValidationError:
        return tuple(
            candidate.model_copy(update={"invalid": True})
            for candidate in _project(row.model_copy(update={"metrics": {}}), progress)
        )
    identities: Final = (
        (("gateway", "gateway", metrics),)
        if progress.source == "gateway"
        else (("team", row.entity_id, metrics),)
        if progress.source == "team"
        else (
            ("gateway", "gateway", metrics.model_copy(update={"requests": 0, "successful": 0, "failed": 0})),
            ("model", row.model, metrics),
            ("user", row.entity_id, metrics),
            ("key", row.api_key, metrics),
        )
    )
    return tuple(
        PerformanceWindow.model_validate(
            {
                "day": progress.day,
                "resource_type": kind,
                "resource_id": resource,
                "days": tuple(metric if index == progress.day_index else _ZERO for index in range(8)),
            }
        )
        for kind, resource, metric in identities
        if resource
    )


def _sum_day(left: DailyMetrics, right: DailyMetrics) -> DailyMetrics:
    if left == _ZERO:
        return right
    if right == _ZERO:
        return left
    return DailyMetrics(
        spend=left.spend + right.spend,
        requests=left.requests + right.requests,
        successful=left.successful + right.successful,
        failed=left.failed + right.failed,
        response_ms=left.response_ms + right.response_ms,
        timed_requests=left.timed_requests + right.timed_requests,
    )


def _merge(left: PerformanceWindow, right: PerformanceWindow) -> PerformanceWindow:
    try:
        days: Final = tuple(_sum_day(a, b) for a, b in zip(left.days, right.days, strict=True))
        return left.model_copy(update={"days": days, "invalid": left.invalid or right.invalid})
    except ValidationError:
        return left.model_copy(update={"invalid": True})


def _next_source(progress: PerformanceProgress) -> PerformanceProgress:
    if progress.day_index < 7:
        return progress.model_copy(update={"day_index": progress.day_index + 1, "cursor": "", "route": ""})
    if progress.source == "gateway":
        return progress.model_copy(update={"phase": "evaluate"})
    return progress.model_copy(
        update={"source": "team" if progress.source == "user" else "gateway", "day_index": 0, "cursor": "", "route": ""}
    )


class PerformanceAlertRepository:
    def __init__(self, prisma_client: "PrismaClient") -> None:
        self._database: Final = cast(  # cast-ok: generated Prisma transactions expose the bounded SQL surface
            _Database, prisma_client.writer_db
        )

    @asynccontextmanager
    async def _transaction(self) -> AsyncGenerator[_Transaction]:
        async with self._database.tx(timeout=timedelta(seconds=2)) as transaction:
            await transaction.query_raw(_SET_LIMITS)
            yield transaction

    async def claim(self, day: date, owner: str) -> PerformanceProgress | None:
        async with self._transaction() as transaction:
            rows: Final = _PROGRESS.validate_python(
                await transaction.query_raw(
                    _CLAIM, _NAME, owner, PerformanceProgress(day=day).model_dump_json(), day.isoformat()
                )
            )
            if not rows:
                return None
            return await self._ready(transaction, owner, rows[0].progress)

    async def _ready(
        self, transaction: _Transaction, owner: str, progress: PerformanceProgress
    ) -> PerformanceProgress | None:
        if not await transaction.query_raw(_ACCOUNTING_DELAYED):
            return progress
        await transaction.execute_raw(_SAVE, _NAME, owner, PerformanceProgress(day=progress.day).model_dump_json())
        return None

    async def release(self, owner: str) -> None:
        async with self._transaction() as transaction:
            await transaction.execute_raw(
                """UPDATE "LiteLLM_Config"
                SET param_value = jsonb_set(param_value, '{lease_until}', to_jsonb(NOW()))
                WHERE param_name = $1 AND param_value->>'owner' = $2""",
                _NAME,
                owner,
            )

    async def collect(self, owner: str) -> PerformanceProgress | None:
        async with self._transaction() as transaction:
            locked: Final = _PROGRESS.validate_python(await transaction.query_raw(_LOCK, _NAME, owner))
            if not locked:
                return None
            progress: Final = await self._ready(transaction, owner, locked[0].progress)
            if progress is None:
                return None
            if progress.phase == "cleanup":
                deleted: Final = await transaction.execute_raw(
                    """DELETE FROM "LiteLLM_PerformanceAlertWork"
                    WHERE (day, resource_hash) IN (SELECT day, resource_hash FROM "LiteLLM_PerformanceAlertWork"
                        WHERE day <= $1::date ORDER BY day, resource_hash LIMIT 1000)""",
                    progress.day.isoformat(),
                )
                cleaned: Final = progress if deleted == 1000 else progress.model_copy(update={"phase": "collect"})
                await transaction.execute_raw(_SAVE, _NAME, owner, cleaned.model_dump_json())
                return cleaned
            source_day: Final = (progress.day - timedelta(days=7 - progress.day_index)).isoformat()
            params: Final = (
                (source_day, progress.cursor, progress.route)
                if progress.source == "gateway"
                else (source_day, progress.cursor, _PAGE_ROWS, _PAGE_BYTES)
            )
            rows: Final = _SOURCE.validate_python(await transaction.query_raw(_source_sql(progress.source), *params))
            if not rows:
                advanced: Final = _next_source(progress)
                await transaction.execute_raw(_SAVE, _NAME, owner, advanced.model_dump_json())
                return advanced
            projected: Final = tuple(chain.from_iterable(_project(row, progress) for row in rows))
            folded: Final = tuple(
                reduce(_merge, group) for _, group in groupby(sorted(projected, key=_hash), key=_hash)
            )
            previous: Final = (
                _WORK.validate_python(
                    await transaction.query_raw(
                        'SELECT resource_hash, data FROM "LiteLLM_PerformanceAlertWork" '
                        "WHERE day = $1::date AND resource_hash = ANY($2::text[])",
                        progress.day.isoformat(),
                        tuple(_hash(window) for window in folded),
                    )
                )
                if folded
                else ()
            )
            by_hash: Final = MappingProxyType({row.resource_hash: row.data for row in previous})
            merged: Final = tuple(
                _merge(by_hash[_hash(window)], window) if _hash(window) in by_hash else window for window in folded
            )
            if merged:
                await transaction.execute_raw(
                    _UPSERT,
                    progress.day.isoformat(),
                    json.dumps(
                        tuple(
                            {
                                "resource_hash": _hash(window),
                                "priority": window.priority,
                                "data": window.model_dump(mode="json"),
                            }
                            for window in merged
                        )
                    ),
                )
            saved: Final = progress.model_copy(
                update={
                    "cursor": rows[-1].id,
                    "route": rows[-1].route,
                    "scanned": progress.scanned + len(rows),
                    "discovered": progress.discovered + len(folded) - len(previous),
                }
            )
            await transaction.execute_raw(_SAVE, _NAME, owner, saved.model_dump_json())
            return saved

    async def evaluate(self, owner: str) -> PerformanceProgress | None:
        async with self._transaction() as transaction:
            locked: Final = _PROGRESS.validate_python(await transaction.query_raw(_LOCK, _NAME, owner))
            if not locked:
                return None
            progress: Final = await self._ready(transaction, owner, locked[0].progress)
            if progress is None:
                return None
            rows: Final = _WORK.validate_python(await transaction.query_raw(_PENDING, progress.day.isoformat()))
            detections: Final = tuple(chain.from_iterable(detect_performance_alerts(row.data) for row in rows))
            await record_alerts(transaction, detections)
            await transaction.execute_raw(
                'UPDATE "LiteLLM_PerformanceAlertWork" SET evaluated = true '
                "WHERE day = $1::date AND resource_hash = ANY($2::text[])",
                progress.day.isoformat(),
                tuple(row.resource_hash for row in rows),
            )
            saved: Final = progress.model_copy(
                update={
                    "evaluated": progress.evaluated + len(rows),
                    "invalid": progress.invalid + sum(row.data.invalid for row in rows),
                    "phase": "evaluate" if rows else "done",
                }
            )
            await transaction.execute_raw(_SAVE, _NAME, owner, saved.model_dump_json())
            return saved
