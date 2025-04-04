import numpy as np
import matplotlib.pyplot as plt
import pytest
from mheatmap.matrix import mosaic_heatmap


class TestMosaicHeatmap:
    """Test cases for the mosaic_heatmap function."""

    def test_basic_functionality(self):
        """Test that mosaic_heatmap works with a simple 2x2 matrix."""
        # Create a simple 2x2 matrix
        matrix = np.array([[1, 2], [3, 4]])
        
        # Call the function
        ax = mosaic_heatmap(matrix)
        
        # Check that we got an Axes object back
        assert isinstance(ax, plt.Axes)
        
        # Clean up
        plt.close(ax.figure)
    
    def test_with_labels(self):
        """Test mosaic_heatmap with row and column labels."""
        # Create a 3x3 matrix
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Call the function with labels
        ax = mosaic_heatmap(matrix, xticklabels=['X', 'Y', 'Z'], yticklabels=['A', 'B', 'C'])
        
        # Check that we got an Axes object back
        assert isinstance(ax, plt.Axes)
        
        # Check that the labels are set
        assert [t.get_text() for t in ax.get_xticklabels()] == ['X', 'Y', 'Z']
        assert [t.get_text() for t in ax.get_yticklabels()] == ['A', 'B', 'C']
        
        # Clean up
        plt.close(ax.figure)
    
    def test_with_custom_colormap(self):
        """Test mosaic_heatmap with a custom colormap."""
        # Create a 2x3 matrix
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        
        # Call the function with a custom colormap
        ax = mosaic_heatmap(matrix, cmap='viridis')
        
        # Check that we got an Axes object back
        assert isinstance(ax, plt.Axes)
        
        # Clean up
        plt.close(ax.figure)
    
    def test_with_zero_values(self):
        """Test mosaic_heatmap with a matrix containing zeros."""
        # Create a matrix with zeros
        matrix = np.array([[1, 0, 3], [0, 5, 0], [7, 0, 9]])
        
        # Call the function
        ax = mosaic_heatmap(matrix)
        
        # Check that we got an Axes object back
        assert isinstance(ax, plt.Axes)
        
        # Clean up
        plt.close(ax.figure)
    
    def test_with_large_matrix(self):
        """Test mosaic_heatmap with a larger matrix."""
        # Create a 10x10 matrix
        matrix = np.random.rand(10, 10)
        
        # Call the function
        ax = mosaic_heatmap(matrix)
        
        # Check that we got an Axes object back
        assert isinstance(ax, plt.Axes)
        
        # Clean up
        plt.close(ax.figure) 