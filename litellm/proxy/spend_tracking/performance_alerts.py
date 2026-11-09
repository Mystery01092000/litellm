import asyncio
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_proxy_logger
from litellm.repositories.performance_alert_repository import PerformanceAlertRepository

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


async def run_performance_alerts(prisma_client: "PrismaClient") -> None:
    from prisma.errors import PrismaError
    from pydantic import ValidationError

    now: Final = datetime.now(timezone.utc)
    if now.hour < 1:
        return
    repository: Final = PerformanceAlertRepository(prisma_client)
    owner: Final = uuid.uuid4().hex
    deadline: Final = time.monotonic() + 60
    try:
        claimed: Final = await repository.claim(now.date() - timedelta(days=1), owner)
        if claimed is None:
            return
        progress = claimed  # rebind-ok: advance the persisted checkpoint without retaining earlier pages
        while progress.phase != "done" and time.monotonic() < deadline:
            next_progress = (
                await repository.evaluate(owner) if progress.phase == "evaluate" else await repository.collect(owner)
            )
            if next_progress is None:
                return
            progress = next_progress
            await asyncio.sleep(0.1)
        verbose_proxy_logger.info(
            "Performance alerts recorded Usage day=%s phase=%s rows=%s resources=%s evaluated=%s invalid=%s",
            progress.day,
            progress.phase,
            progress.scanned,
            progress.discovered,
            progress.evaluated,
            progress.invalid,
        )
    except (PrismaError, ValidationError, TimeoutError) as exc:
        verbose_proxy_logger.warning("Performance alerts deferred (%s)", type(exc).__name__)
    finally:
        try:
            await repository.release(owner)
        except (PrismaError, TimeoutError) as exc:
            verbose_proxy_logger.warning("Performance alerts lease release deferred (%s)", type(exc).__name__)
