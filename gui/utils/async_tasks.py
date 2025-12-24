"""Async helpers.

In the original GUI this handled background work.
For tests, `run_async` simply executes the callable immediately.
"""

from __future__ import annotations

from typing import Any, Callable


def run_async(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return fn(*args, **kwargs)
