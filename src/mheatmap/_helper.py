"""
Helper functions for postprocessing
"""

# Copyright (c) 2024, Juntang Wang
# All rights reserved.
# 
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from ._mosaic_heatmap import mosaic_heatmap

########################################################################################################################

########################################################################################################################



# ----------------------------------------------------------------------------------------------------------------------
#                                   GridSpec
# ----------------------------------------------------------------------------------------------------------------------

def make_gs(
    rows, 
    cols, 
    panel_width, 
    img_shape,
    rows_of_images=2
):
    """Create a figure and GridSpec layout optimized for displaying images and panels.

    Parameters
    ----------
    rows : int
        Number of rows in the GridSpec layout.
    cols : int 
        Number of columns in the GridSpec layout.
    panel_width : float
        Width of each panel in the layout.
    img_shape : tuple of (int, int)
        Shape of the images to be displayed (height, width).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    gs : matplotlib.gridspec.GridSpec
        The GridSpec layout object.

    Notes
    -----
    The layout is designed with the first two rows having height proportional to the image aspect ratio,
    and remaining rows having square proportions. This accommodates both image displays and other
    visualization panels.
    """
    # Calculate panel heights based on image aspect ratio
    img_panel_height = panel_width * (img_shape[0] / img_shape[1])
    sqr_panel_height = panel_width

    # Calculate overall figure dimensions
    fig_width = cols * panel_width
    fig_height = (rows_of_images * img_panel_height) + (sqr_panel_height * (rows - rows_of_images))

    # Create figure and GridSpec with appropriate height ratios
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(rows, cols, 
                 height_ratios=[img_panel_height] * rows_of_images + [sqr_panel_height] * (rows - rows_of_images))
    
    return fig, gs


def plot_heatmap_with_gs(
    data: np.ndarray,
    gs: GridSpec,
    cmap: str = 'YlGnBu',
    norm: BoundaryNorm = None,
    xticklabels: list = None,
    yticklabels: list = None, 
    annot: bool = False,
    fmt: str = 'd',
    size: int = 18,
    mode: str = 'conf_mat',
    **kwargs
) -> plt.Axes:
    """Plot a heatmap using a GridSpec layout with configurable visualization modes.
    
    Parameters
    ----------
    data : np.ndarray
        The data to plot in the heatmap
    gs : matplotlib.gridspec.GridSpec
        GridSpec instance defining the subplot layout
    cmap : str, optional
        Colormap name, by default 'YlGnBu'
    norm : matplotlib.colors.BoundaryNorm, optional
        Normalization for the colormap, by default None
    xticklabels : list, optional
        Labels for x-axis ticks, by default None
    yticklabels : list, optional
        Labels for y-axis ticks, by default None
    annot : bool, optional
        Whether to annotate cells with numerical value, by default False
    fmt : str, optional
        String formatting code for annotations, by default 'd'
    size : int, optional
        Font size for annotations and labels, by default 18
    mode : str, optional
        Visualization mode - one of ['conf_mat', 'rms_map', 'mosaic'], by default 'conf_mat'
    **kwargs
        Additional keyword arguments passed to the plotting function
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plotted heatmap
    """
    ax = plt.subplot(gs, aspect='equal')
    
    if mode == 'rms_map':
        xticklabels = ['GT1', 'GT2', 'PRED1', 'PRED2']
        annot = True
        kwargs['annot_kws'] = {'size': size}
        kwargs['cbar'] = False
        plotter = sns.heatmap
    else:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        kwargs['cbar_ax'] = cax
        kwargs['square'] = True
        plotter = sns.heatmap if mode == 'conf_mat' else mosaic_heatmap

    plotter(
        data,
        cmap=cmap,
        norm=norm,
        annot=annot,
        fmt=fmt,
        ax=ax,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        **kwargs
    )

    if mode == 'rms_map':
        ax.tick_params(axis='both', which='major', labelsize=size)
        ax.axvline(x=2, color='black', linewidth=2)  # Separator between GT and PRED
    elif mode in ['conf_mat', 'mosaic']:
        ax.xaxis.tick_top()
        
    return ax
        