import time
from typing import Callable, Any, Tuple

def measure_time(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Measures the execution time of a function.

    Args:
        func: The function to execute.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        A tuple containing the function's result and the elapsed time in seconds.
    """
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    elapsed = end - start
    return result, elapsed