"""Base utilities for test control and plotting functions."""

import numpy as np
from functools import wraps
from ..constants import get_test_mode


def test_decorator(func):
    """Decorator to control execution of plotting functions in test mode.

    This decorator enables conditional execution of plotting functions based on the global
    test mode setting. When test mode is disabled, the decorated function will be skipped
    and return None. This helps prevent unwanted plot generation during automated testing
    or batch processing.

    Parameters
    ----------
    func : callable
        The plotting function to be decorated

    Returns
    -------
    callable
        A wrapped function that checks test mode before executing the original function

    Notes
    -----
    The test mode can be controlled using the `set_test_mode()` function from the
    constants module. By default, test mode is disabled.

    Examples
    --------
    >>> @test_decorator
    ... def plot_data(data):
    ...     # Plot generation code
    ...     pass
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if get_test_mode():
            return func(*args, **kwargs)
        return None

    return wrapper
