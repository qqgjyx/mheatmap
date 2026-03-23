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

from ._amc_postprocess import amc_postprocess, mask_zeros_from_gt
from ._rms_permute import RMSResult, rms_permute
from .graph import copermute_from_bipermute, spectral_permute, two_walk_laplacian
from .matrix import mosaic_heatmap

# Package version
__version__ = "1.2.5"

# Public API
__all__ = [
    "RMSResult",
    "amc_postprocess",
    "copermute_from_bipermute",
    "mask_zeros_from_gt",
    "mosaic_heatmap",
    "rms_permute",
    "spectral_permute",
    "two_walk_laplacian",
]
