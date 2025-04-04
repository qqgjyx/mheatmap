import numpy as np
import pytest
from mheatmap._rms_permute import rms_permute


class TestRMSPermute:
    """Test cases for the RMS permutation function."""

    def test_basic_functionality(self):
        """Test that rms_permute works with a simple 3x3 matrix."""
        # Create a simple 3x3 matrix
        matrix = np.array([[1, 0.5, 0.2], [0.5, 1, 0.7], [0.2, 0.7, 1]])
        labels = np.array(['A', 'B', 'C'])
        
        # Call the function
        permuted_matrix, permuted_labels, rms_label_map, rms_map_matrix, rms_map_type = rms_permute(matrix, labels)
        
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
        matrix = np.array([
            [1, 0.3, 0.5, 0.2],
            [0.3, 1, 0.4, 0.6],
            [0.5, 0.4, 1, 0.7],
            [0.2, 0.6, 0.7, 1]
        ])
        labels = np.array(['A', 'B', 'C', 'D'])
        
        # Call the function
        permuted_matrix, permuted_labels, rms_label_map, rms_map_matrix, rms_map_type = rms_permute(matrix, labels)
        
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
        labels = np.array([f'Class_{i}' for i in range(10)])
        
        # Call the function
        permuted_matrix, permuted_labels, rms_label_map, rms_map_matrix, rms_map_type = rms_permute(matrix, labels)
        
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
        # Create a simple 3x3 matrix
        matrix = np.array([[1, 0.5, 0.2], [0.5, 1, 0.7], [0.2, 0.7, 1]])
        labels = np.array(['A', 'B', 'C'])
        
        # Call the function
        permuted_matrix, permuted_labels, rms_label_map, rms_map_matrix, rms_map_type = rms_permute(matrix, labels)
        
        # Check that we got the expected number of return values
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