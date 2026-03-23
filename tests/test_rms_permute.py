import numpy as np

from mheatmap._rms_permute import rms_permute


class TestRMSPermute:
    """Test cases for the RMS permutation function."""

    def test_basic_functionality(self):
        """Test that rms_permute works with a simple 3x3 matrix."""
        # Create a simple 3x3 matrix
        matrix = np.array([[1, 0.5, 0.2], [0.5, 1, 0.7], [0.2, 0.7, 1]])
        labels = np.array(["A", "B", "C"])

        # Call the function
        (
            permuted_matrix,
            permuted_labels,
            rms_label_map,
            rms_map_matrix,
            rms_map_type,
        ) = rms_permute(matrix, labels)

        # Check that we got the expected return values
        assert isinstance(permuted_matrix, np.ndarray)
        assert isinstance(permuted_labels, np.ndarray)
        assert isinstance(rms_label_map, dict)
        assert isinstance(rms_map_matrix, np.ndarray)
        assert isinstance(rms_map_type, np.ndarray)

        # Check that the shape is preserved
        assert permuted_matrix.shape == matrix.shape
        assert len(permuted_labels) == len(labels)

        # Check that the labels are a permutation of the original labels
        assert set(permuted_labels) == set(labels)

        # Check that the matrix is still symmetric
        assert np.allclose(permuted_matrix, permuted_matrix.T)

    def test_with_symmetric_matrix(self):
        """Test rms_permute with a symmetric matrix."""
        # Create a symmetric 4x4 matrix
        matrix = np.array(
            [
                [1, 0.3, 0.5, 0.2],
                [0.3, 1, 0.4, 0.6],
                [0.5, 0.4, 1, 0.7],
                [0.2, 0.6, 0.7, 1],
            ]
        )
        labels = np.array(["A", "B", "C", "D"])

        # Call the function
        (
            permuted_matrix,
            permuted_labels,
            rms_label_map,
            rms_map_matrix,
            rms_map_type,
        ) = rms_permute(matrix, labels)

        # Check that we got the expected return values
        assert isinstance(permuted_matrix, np.ndarray)
        assert isinstance(permuted_labels, np.ndarray)
        assert isinstance(rms_label_map, dict)
        assert isinstance(rms_map_matrix, np.ndarray)
        assert isinstance(rms_map_type, np.ndarray)

        # Check that the shape is preserved
        assert permuted_matrix.shape == matrix.shape
        assert len(permuted_labels) == len(labels)

        # Check that the labels are a permutation of the original labels
        assert set(permuted_labels) == set(labels)

        # Check that the matrix is still symmetric
        assert np.allclose(permuted_matrix, permuted_matrix.T)

    def test_with_large_matrix(self):
        """Test rms_permute with a larger matrix."""
        # Create a 10x10 matrix
        matrix = np.random.rand(10, 10)
        # Make it symmetric
        matrix = (matrix + matrix.T) / 2
        labels = np.array([f"Class_{i}" for i in range(10)])

        # Call the function
        (
            permuted_matrix,
            permuted_labels,
            rms_label_map,
            rms_map_matrix,
            rms_map_type,
        ) = rms_permute(matrix, labels)

        # Check that we got the expected return values
        assert isinstance(permuted_matrix, np.ndarray)
        assert isinstance(permuted_labels, np.ndarray)
        assert isinstance(rms_label_map, dict)
        assert isinstance(rms_map_matrix, np.ndarray)
        assert isinstance(rms_map_type, np.ndarray)

        # Check that the shape is preserved
        assert permuted_matrix.shape == matrix.shape
        assert len(permuted_labels) == len(labels)

        # Check that the labels are a permutation of the original labels
        assert set(permuted_labels) == set(labels)

        # Check that the matrix is still symmetric
        assert np.allclose(permuted_matrix, permuted_matrix.T)

    def test_with_parameters(self):
        """Test rms_permute with different parameters."""
        matrix = np.array([[1, 0.5, 0.2], [0.5, 1, 0.7], [0.2, 0.7, 1]])
        labels = np.array(["A", "B", "C"])

        result = rms_permute(matrix, labels)

        assert isinstance(result.permuted_matrix, np.ndarray)
        assert result.permuted_matrix.shape == matrix.shape
        assert set(result.permuted_labels) == set(labels)

    def test_single_class(self):
        """Test rms_permute with a 1x1 matrix."""
        matrix = np.array([[5]])
        labels = np.array(["A"])

        result = rms_permute(matrix, labels)
        assert result.permuted_matrix.shape == (1, 1)
        assert result.permuted_matrix[0, 0] == 5
        assert len(result.rms_label_map) == 0

    def test_perfect_diagonal(self):
        """Test rms_permute with a perfect diagonal (no merges/splits)."""
        matrix = np.diag([10, 20, 30])
        labels = np.array(["A", "B", "C"])

        result = rms_permute(matrix, labels)
        assert np.array_equal(result.permuted_matrix, matrix)
        assert len(result.rms_label_map) == 0
