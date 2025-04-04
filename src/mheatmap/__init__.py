"""
mheatmap - Advanced heatmap visualization and matrix analysis tools

This package provides tools for creating enhanced heatmap visualizations and performing
matrix analysis, with a focus on confusion matrices and classification results.

Key Features:
- Mosaic/proportional heatmap visualization
- Confusion matrix post-processing and analysis
- Graph-based and RMS-based matrix permutation algorithms

Author: Juntang Wang @ Duke University
"""

# Core visualization
from .matrix import mosaic_heatmap

# Post-processing utilities
from ._amc_postprocess import amc_postprocess, mask_zeros_from_gt

# Matrix permutation
from ._rms_permute import rms_permute

# Graph-based algorithms
from .graph import copermute_from_bipermute, spectral_permute, two_walk_laplacian

# Store original matplotlib configuration
import matplotlib as mpl
_original_rcParams = mpl.rcParams.copy()

# Package version
__version__ = "1.2.5"

# Public API
__all__ = [
    "mosaic_heatmap",
    "mask_zeros_from_gt",
    "amc_postprocess",
    "rms_permute",
    "copermute_from_bipermute",
    "spectral_permute",
    "two_walk_laplacian",
]
