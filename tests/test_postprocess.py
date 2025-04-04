import numpy as np
import pytest
from mheatmap._amc_postprocess import amc_postprocess, mask_zeros_from_gt


class TestPostProcessFunctions:
    """Test cases for the post-processing utility functions."""

    def test_mask_zeros_from_gt(self):
        """Test the mask_zeros_from_gt function."""
        # Create a simple 3x3 matrix of labels
        labels = np.array([1, 2, 3, 4, 5, 6])
        
        # Create a ground truth matrix with zeros
        gt = np.array([1, 0, 3, 0, 5, 0])
        
        # Call the function
        result = mask_zeros_from_gt(labels, gt)
        
        # Check that we got a numpy array back
        assert isinstance(result, np.ndarray)
        
        # Check that zeros in gt are masked in the result
        expected = np.array([1, 3, 5])
        np.testing.assert_array_equal(result, expected)
    
    def test_amc_postprocess(self):
        """Test the amc_postprocess function."""
        # Create test data
        pred = np.array([1, 2, 3, 1, 2, 3])
        gt = np.array([2, 3, 1, 2, 3, 1])
        
        # Call the function
        aligned_pred = amc_postprocess(pred, gt)
        
        # Check that we got a numpy array back
        assert isinstance(aligned_pred, np.ndarray)
        
        # Check that the shape is preserved
        assert aligned_pred.shape == pred.shape
        
        # Check that the values are properly aligned
        # The exact values will depend on the alignment algorithm,
        # but they should be a permutation of the original values
        assert set(aligned_pred) == set(pred) 