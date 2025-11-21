"""Async execution helpers for dataset builder."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


def run_sync(coro: Awaitable[T]) -> T:
    """Run an async coroutine from sync context, handling nested loops."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    if loop.is_running():
        try:
            import nest_asyncio  # type: ignore

            nest_asyncio.apply(loop)
        except Exception:  # pragma: no cover - defensive
            pass
    return loop.run_until_complete(coro)


async def gather_limited(
    coros: list[Callable[[], Awaitable[T]]],
    *,
    limit: int = 8,
) -> list[T]:
    """Execute coroutines with concurrency limit."""

    semaphore = asyncio.Semaphore(max(1, limit))
    results: list[T] = []

    async def _run(fn: Callable[[], Awaitable[T]]) -> None:
        async with semaphore:
            result = await fn()
            results.append(result)

    tasks = [asyncio.create_task(_run(fn)) for fn in coros]
    if tasks:
        await asyncio.gather(*tasks)
    return results
