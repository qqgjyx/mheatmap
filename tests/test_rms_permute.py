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
        reordered_matrix, reordered_labels = rms_permute(matrix, labels)
        
        # Check that the results are numpy arrays
        assert isinstance(reordered_matrix, np.ndarray)
        assert isinstance(reordered_labels, np.ndarray)
        
        # Check that the shape is preserved
        assert reordered_matrix.shape == matrix.shape
        assert len(reordered_labels) == len(labels)
        
        # Check that the labels are a permutation of the original labels
        assert set(reordered_labels) == set(labels)
    
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
        reordered_matrix, reordered_labels = rms_permute(matrix, labels)
        
        # Check that the results are numpy arrays
        assert isinstance(reordered_matrix, np.ndarray)
        assert isinstance(reordered_labels, np.ndarray)
        
        # Check that the shape is preserved
        assert reordered_matrix.shape == matrix.shape
        assert len(reordered_labels) == len(labels)
        
        # Check that the labels are a permutation of the original labels
        assert set(reordered_labels) == set(labels)
        
        # Check that the matrix is still symmetric
        assert np.allclose(reordered_matrix, reordered_matrix.T)
    
    def test_with_large_matrix(self):
        """Test rms_permute with a larger matrix."""
        # Create a 10x10 matrix
        matrix = np.random.rand(10, 10)
        # Make it symmetric
        matrix = (matrix + matrix.T) / 2
        labels = np.array([f'Class_{i}' for i in range(10)])
        
        # Call the function
        reordered_matrix, reordered_labels = rms_permute(matrix, labels)
        
        # Check that the results are numpy arrays
        assert isinstance(reordered_matrix, np.ndarray)
        assert isinstance(reordered_labels, np.ndarray)
        
        # Check that the shape is preserved
        assert reordered_matrix.shape == matrix.shape
        assert len(reordered_labels) == len(labels)
        
        # Check that the labels are a permutation of the original labels
        assert set(reordered_labels) == set(labels)
        
        # Check that the matrix is still symmetric
        assert np.allclose(reordered_matrix, reordered_matrix.T)
    
    def test_with_parameters(self):
        """Test rms_permute with different parameters."""
        # Create a simple 3x3 matrix
        matrix = np.array([[1, 0.5, 0.2], [0.5, 1, 0.7], [0.2, 0.7, 1]])
        
        # Call the function with different parameters
        result1 = rms_permute(matrix, max_iter=10)
        result2 = rms_permute(matrix, tol=1e-4)
        
        # Check that both results are numpy arrays
        assert isinstance(result1, np.ndarray)
        assert isinstance(result2, np.ndarray)
        
        # Check that both shapes are preserved
        assert result1.shape == matrix.shape
        assert result2.shape == matrix.shape 