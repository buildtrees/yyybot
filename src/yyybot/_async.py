"""Small async helpers kept private to the runtime."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import threading
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


async def run_sync(handler: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run blocking work without tying the runtime to an async HTTP library."""

    loop = asyncio.get_running_loop()
    future: asyncio.Future[T] = loop.create_future()
    context = contextvars.copy_context()
    call = functools.partial(context.run, handler, *args, **kwargs)

    def resolve(value: T | None = None, error: BaseException | None = None) -> None:
        if future.done():
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(value)  # type: ignore[arg-type]

    def work() -> None:
        try:
            value = call()
        except BaseException as exc:
            loop.call_soon_threadsafe(resolve, None, exc)
        else:
            loop.call_soon_threadsafe(resolve, value, None)

    threading.Thread(target=work, name="yyybot-worker", daemon=True).start()
    return await future
