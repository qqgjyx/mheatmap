"""
mheatmap - Advanced heatmap visualization and matrix analysis tools
"""

from .matrix import mosaic_heatmap
from .amc_postprocess import (
    AMCPostprocess, 
    amc_postprocess,
    mask_zeros_from_gt,
)
from .rms_permute import (
    RMSPermute, 
    rms_permute
)
from .spectral_permute import spectral_permute

# Capture the original matplotlib rcParams
import matplotlib as mpl
_original_rcParams = mpl.rcParams.copy()

__version__ = "0.1.0"

__all__ = [
    "mosaic_heatmap",
    
    "AMCPostprocess",
    "amc_postprocess",
    "mask_zeros_from_gt",
    
    "RMSPermute",
    "rms_permute",
    
    "spectral_permute",
] 