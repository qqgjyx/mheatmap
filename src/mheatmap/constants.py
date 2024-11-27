"""Global constants"""

# Global test mode flag - use a list to make it mutable
_TEST_MODE_CONTAINER = [False]


def set_test_mode(enabled: bool = True) -> None:
    """Set the global test mode flag.

    Parameters
    ----------
    enabled : bool, optional
        Whether to enable test mode, by default True

    Returns
    -------
    None
        This function doesn't return anything
    """
    _TEST_MODE_CONTAINER[0] = enabled


def get_test_mode() -> bool:
    """Get the current test mode setting.

    Returns
    -------
    bool
        Current test mode value
    """
    return _TEST_MODE_CONTAINER[0]
