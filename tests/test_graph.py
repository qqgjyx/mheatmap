import numpy as np
import pytest
from mheatmap.graph import (
    copermute_from_bipermute,
    spectral_permute,
    two_walk_laplacian
)


class TestGraphFunctions:
    """Test cases for the graph-based permutation functions."""

    def test_copermute_from_bipermute(self):
        """Test the copermute_from_bipermute function."""
        # Create a simple 3x3 matrix
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Create test parameters
        B_sizes = [3, 3]  # 3x3 matrix
        B_subrows = np.array([0, 2, 1])  # Use all rows
        B_subcols = np.array([2, 0, 1])  # Use all columns
        p_Asub = np.array([0, 5, 2, 3, 1, 4])  # Example permutation
        
        # Call the function
        p_Brows, p_Bcols = copermute_from_bipermute(B_sizes, B_subrows, B_subcols, p_Asub)
        
        # Check that the results are numpy arrays
        assert isinstance(p_Brows, np.ndarray)
        assert isinstance(p_Bcols, np.ndarray)
        
        # Check that the permutation vectors have correct lengths
        assert len(p_Brows) == B_sizes[0]
        assert len(p_Bcols) == B_sizes[1]
    
    def test_spectral_permute(self):
        """Test the spectral_permute function."""
        # Create a simple 3x3 matrix
        matrix = np.array([[1, 0.5, 0.2], [0.5, 1, 0.7], [0.2, 0.7, 1]])
        labels = np.array(['A', 'B', 'C'])
        
        # Call the function
        reordered_matrix, reordered_labels = spectral_permute(matrix, labels)
        
        # Check that the results are numpy arrays
        assert isinstance(reordered_matrix, np.ndarray)
        assert isinstance(reordered_labels, np.ndarray)
        
        # Check that the shape is preserved
        assert reordered_matrix.shape == matrix.shape
        assert len(reordered_labels) == len(labels)
        
        # Check that the labels are a permutation of the original labels
        assert set(reordered_labels) == set(labels)
    
    def test_two_walk_laplacian(self):
        """Test the two_walk_laplacian function."""
        # Create a simple 3x3 matrix
        matrix = np.array([[1, 0.5, 0.2], [0.5, 1, 0.7], [0.2, 0.7, 1]])
        
        # Call the function
        L_tw = two_walk_laplacian(matrix)
        
        # Check that the result is a numpy array
        assert isinstance(L_tw, np.ndarray)
        
        # Check that the matrix is square
        assert L_tw.shape[0] == L_tw.shape[1]
        
        # Check that the matrix is symmetric
        assert np.allclose(L_tw, L_tw.T)
        
        # Check that the diagonal elements are non-negative
        assert np.all(np.diag(L_tw) >= 0)
        
        # Check that the off-diagonal elements are non-positive
        off_diag = L_tw - np.diag(np.diag(L_tw))
        assert np.all(off_diag <= 0) 