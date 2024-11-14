"""
mheatmap - Advanced heatmap visualization and matrix analysis tools
"""

from ._mosaic_heatmap import mosaic_heatmap
from ._amc_postprocess import AMCPostprocess, amc_postprocess
from ._rms_permute import RMSPermute, rms_permute
from ._spectral_permute import spectral_permute
from ._helper import make_gs, plot_heatmap_with_gs

__version__ = "0.1.0"

__all__ = [
    "mosaic_heatmap",
    "AMCPostprocess",
    "amc_postprocess",
    "RMSPermute",
    "rms_permute",
    "spectral_permute",
    "make_gs",
    "plot_heatmap_with_gs"
] 