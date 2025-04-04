import pytest
import numpy as np


@pytest.fixture
def sample_matrix_3x3():
    """Fixture that provides a sample 3x3 matrix."""
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


@pytest.fixture
def sample_matrix_3x3_symmetric():
    """Fixture that provides a sample 3x3 symmetric matrix."""
    return np.array([[1, 0.5, 0.2], [0.5, 1, 0.7], [0.2, 0.7, 1]])


@pytest.fixture
def sample_matrix_3x3_with_zeros():
    """Fixture that provides a sample 3x3 matrix with zeros."""
    return np.array([[1, 0, 3], [0, 5, 0], [7, 0, 9]])


@pytest.fixture
def sample_matrix_10x10():
    """Fixture that provides a sample 10x10 random matrix."""
    matrix = np.random.rand(10, 10)
    # Make it symmetric
    return (matrix + matrix.T) / 2 