import time
from typing import Any, Callable

def time_execution(func: Callable, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Times any execution in seconds."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    duration = time.perf_counter() - start
    return result, duration