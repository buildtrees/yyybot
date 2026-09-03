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

    context = contextvars.copy_context()
    call = functools.partial(context.run, handler, *args, **kwargs)
    result: list[T] = []
    errors: list[BaseException] = []
    done = threading.Event()

    def work() -> None:
        try:
            result.append(call())
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    threading.Thread(target=work, name="yyybot-worker", daemon=True).start()
    while not done.is_set():
        await asyncio.sleep(0.01)
    if errors:
        raise errors[0]
    return result[0]
