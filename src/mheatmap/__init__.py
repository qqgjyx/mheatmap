"""
mheatmap - Advanced heatmap visualization and matrix analysis tools
"""

from .matrix import mosaic_heatmap
from ._amc_postprocess import (
    amc_postprocess,
    mask_zeros_from_gt,
)
from ._rms_permute import rms_permute
from ._spectral_permute import spectral_permute

# Capture the original matplotlib rcParams
import matplotlib as mpl

_original_rcParams = mpl.rcParams.copy()

__version__ = "1.0.1"

__all__ = [
    "amc_postprocess",
    "mosaic_heatmap",
    "rms_permute",
    "spectral_permute",
    "mask_zeros_from_gt",
]
