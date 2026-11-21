"""Adaptive concurrency and rate-limit retries for Claude through the Databricks gateway.

The gateway path has no endpoint pool (that speaks OpenAI to vLLM servers), so the same additive-increase,
halve-on-pressure control (:class:`~matchminer_ai.llm.adaptive_concurrency.AdaptiveConcurrency`, the remote
section's ``adaptive_concurrency`` settings) runs here: the limit grows while requests succeed and more are waiting,
and halves on a 429/503/529 or a dropped connection (measured 2026-11-21: the gateway sheds load both ways). Such
requests wait (``Retry-After`` when the gateway sends one) and retry
without counting as failed attempts. Each change of the limit is a line of the gateway request log.
"""
from __future__ import annotations

import asyncio
import random
from typing import Any

from .adaptive_concurrency import AdaptiveConcurrency
from .anthropic_messages import GatewayError, log_event

RATE_LIMITED = frozenset({429, 503, 529})
#: Rate-limit waits one request may take before it fails.
MAX_RATE_LIMIT_WAITS = 30


class GatewayLimiter:
    """At most ``limit`` requests in flight, adapted from outcomes (see the module docstring)."""

    def __init__(self, settings: dict | None, maximum: int):
        values = dict(settings or {})
        self.control = AdaptiveConcurrency(
            maximum=maximum,
            initial=min(maximum, int(values.get("initial", 32))),
            minimum=int(values.get("minimum", 1)),
            success_threshold=int(values.get("success_threshold", 32)),
            increase_step=int(values.get("increase_step", 32)),
            adjustment_interval=float(values.get("adjustment_interval", 10.0)),
            backoff_factor=float(values.get("backoff_factor", 0.5)),
        )
        self.active = self.waiting = 0
        self.condition = asyncio.Condition()
        log_event("concurrency", limit=self.control.limit, reason="initial", maximum=maximum)

    @property
    def limit(self) -> int:
        return self.control.limit

    async def acquire(self) -> None:
        async with self.condition:
            self.waiting += 1
            try:
                await self.condition.wait_for(lambda: self.active < self.control.limit)
            finally:
                self.waiting -= 1
            self.active += 1

    async def release(self, *, successful: bool = False, pressure: bool = False) -> None:
        async with self.condition:
            saturated = self.waiting > 0
            self.active -= 1
            before = self.control.limit
            self.control.observe(successful=successful, pressure=pressure, saturated=saturated)
            if self.control.limit != before:
                log_event("concurrency", limit=self.control.limit, reason=self.control.reason,
                          active=self.active, waiting=self.waiting)
            self.condition.notify_all()


def _wait(error: GatewayError, waits: int) -> float:
    if error.retry_after:
        return float(error.retry_after) * random.uniform(1.0, 1.25)
    return min(60.0, 2.0 ** min(waits, 6)) * random.uniform(0.5, 1.5)


async def limited_request(limiter: GatewayLimiter, create, request: dict[str, Any], *, timeout: float,
                          max_retries: int):
    """``await create(**request)`` under ``limiter``: rate limits wait and retry; other failures retry with backoff
    up to ``max_retries`` attempts. Returns the response, or raises the last failure."""
    from matchminer_ai.cancellation import check_cancelled

    failures = waits = 0
    while True:
        check_cancelled()
        await limiter.acquire()
        try:
            response = await asyncio.wait_for(create(**request), timeout=timeout)
        except GatewayError as exc:
            limited = exc.http_status in RATE_LIMITED or getattr(exc, "connection", False)
            await limiter.release(pressure=limited)
            if limited and waits < MAX_RATE_LIMIT_WAITS:
                waits += 1
                await asyncio.sleep(_wait(exc, waits))
                continue
            error: BaseException = exc
        except asyncio.CancelledError:
            await limiter.release()
            raise
        except Exception as exc:  # noqa: BLE001 -- timeouts and transport failures retry like any other
            await limiter.release()
            error = exc
        else:
            await limiter.release(successful=True)
            return response
        failures += 1
        if failures >= max_retries:
            raise error
        await asyncio.sleep(min((2 ** failures) * 5, 120))


__all__ = ["MAX_RATE_LIMIT_WAITS", "RATE_LIMITED", "GatewayLimiter", "limited_request"]
