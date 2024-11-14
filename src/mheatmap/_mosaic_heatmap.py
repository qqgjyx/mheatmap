"""
Mosaic heatmap (Name TBD)
"""

# Copyright (c) 2024 Juntang Wang
# License: MIT

# Project: pyC4H
# Code prototype: Xiaobai Sun, 2024

# Drafted imitating seaborn.heatmap()

from matplotlib import pyplot as plt
import numpy as np
from seaborn import despine
from seaborn.matrix import _HeatMapper
from seaborn.utils import (
    _draw_figure,
    axis_ticklabels_overlap
)

########################################################################################################################
#                                                                                                                      #
#                                           Mosaic Confusion Matrix                                                #
#                                                                                                                      #
########################################################################################################################

def get_mh_xyc(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the x, y, edges and Xc, Yc of the mosaic data."""

    # Calculate total samples and proportions for rows and columns
    total_sum = data.sum()
    row_sums = data.sum(axis=1)
    col_sums = data.sum(axis=0)
    
    # Normalize to get row and column weights
    row_weights = row_sums / total_sum
    col_weights = col_sums / total_sum
    
    # Compute cumulative positions for cell edges
    x_edges = np.concatenate(([0], np.cumsum(col_weights)))
    y_edges = np.concatenate(([0], np.cumsum(row_weights)))
    
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    
#     Xc, Yc = np.meshgrid(x_centers, y_centers)
    return x_edges, y_edges, x_centers, y_centers


# Override the _HeatMapper class
class _MosaicHeatMapper(_HeatMapper):
    """Draw a mosaic heatmap plot of a matrix with nice labels and colormaps."""
    def __init__(self, data, vmin, vmax, cmap, center, robust, annot, fmt,
                 annot_kws, cbar, cbar_kws,
                 xticklabels=True, yticklabels=True, mask=None):
        # Initialize the _HeatMapper class
        super().__init__(data, vmin, vmax, cmap, center, robust, annot, fmt,
                         annot_kws, cbar, cbar_kws,
                         xticklabels, yticklabels, mask)
    
    def _annotate_mosaic_heatmap(self, ax):
            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[1]):
                        ax.text(
                        self.x_centers[j],
                        self.y_centers[i],
                        f'{self.data[i, j]}',
                        ha='center',
                        va='center',
                        color='black'
                        )
    
    def plot(self, ax, cax, kws):
        # Remove all the Axes spines
        despine(ax=ax, left=True, bottom=True)
        
        # setting vmin/vmax in addition to norm is deprecated
        # so avoid setting if norm is set
        if kws.get("norm") is None:
            kws.setdefault("vmin", self.vmin)
            kws.setdefault("vmax", self.vmax)
        
        # Get the x and y edges of the mosaic data
        x_edges, y_edges, x_centers, y_centers = get_mh_xyc(self.plot_data)
        
        mesh = ax.pcolormesh(x_edges, y_edges, self.plot_data, shading='flat', cmap=self.cmap, **kws)
        
        # Set the axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        self.x_centers = x_centers
        self.y_centers = y_centers

        xticks = x_centers
        yticks = y_centers
        xticklabels = self.xticklabels
        yticklabels = self.yticklabels

        # Invert the y axis to show the plot in matrix form
        ax.invert_yaxis()

        # Possibly add a colorbar
        if self.cbar:
            cb = ax.figure.colorbar(mesh, cax, ax, **self.cbar_kws)
            cb.outline.set_linewidth(0)
            # If rasterized is passed to pcolormesh, also rasterize the
            # colorbar to avoid white lines on the PDF rendering
            if kws.get('rasterized', False):
                cb.solids.set_rasterized(True)
        
        ax.set(xticks=xticks, yticks=yticks)
        xtl = ax.set_xticklabels(xticklabels)
        ytl = ax.set_yticklabels(yticklabels, rotation="vertical")
        # plt.setp(ytl, va="center")  # GH2484

        # # Possibly rotate them if they overlap
        # _draw_figure(ax.figure)

        if axis_ticklabels_overlap(xtl):
            plt.setp(xtl, rotation="vertical")
        if axis_ticklabels_overlap(ytl):
            plt.setp(ytl, rotation="horizontal")

        # Add the axis labels
        ax.set(xlabel=self.xlabel, ylabel=self.ylabel)

        # Annotate the cells with the formatted values
        if self.annot:
            self._annotate_mosaic_heatmap(ax)
        
        
def mosaic_heatmap(
    data, *,
    vmin=None, vmax=None, cmap=None, center=None, robust=False,
    annot=None, fmt=".2g", annot_kws=None,
    linewidths=0, linecolor="white",
    cbar=True, cbar_kws=None, cbar_ax=None,
    square=False, xticklabels="auto", yticklabels="auto",
    mask=None, ax=None,
    **kwargs
):
    """Plot mosaic data as a color-encoded matrix.
    
    This is an Axes-level function and will draw the heatmap into the
    currently-active Axes if none is provided to the ``ax`` argument.  Part of
    this Axes space will be taken and used to plot a colormap, unless ``cbar``
    is False or a separate Axes is provided to ``cbar_ax``.
    
    `masaci` : column width / row height is proportional to the ration of
        column / row sum of the data.
    
    Parameters
    ----------
    data : mosaic dataset
        2D dataset that can be coerced into an ndarray. If a Pandas DataFrame
        is provided, the index/column information will be used to label the
        columns and rows.
        vmin, vmax : floats, optional
        Values to anchor the colormap, otherwise they are inferred from the
        data and other keyword arguments.
    cmap : matplotlib colormap name or object, or list of colors, optional
        The mapping from data values to color space. If not provided, the
        default will depend on whether ``center`` is set.
    center : float, optional
        The value at which to center the colormap when plotting divergent data.
        Using this parameter will change the default ``cmap`` if none is
        specified.
    robust : bool, optional
        If True and ``vmin`` or ``vmax`` are absent, the colormap range is
        computed with robust quantiles instead of the extreme values.
    annot : bool or rectangular dataset, optional
        If True, write the data value in each cell. If an array-like with the
        same shape as ``data``, then use this to annotate the heatmap instead
        of the data. Note that DataFrames will match on position, not index.
    fmt : str, optional
        String formatting code to use when adding annotations.
    annot_kws : dict of key, value mappings, optional
        Keyword arguments for :meth:`matplotlib.axes.Axes.text` when ``annot``
        is True.
    linewidths : float, optional
        Width of the lines that will divide each cell.
    linecolor : color, optional
        Color of the lines that will divide each cell.
    cbar : bool, optional
        Whether to draw a colorbar.
    cbar_kws : dict of key, value mappings, optional
        Keyword arguments for :meth:`matplotlib.figure.Figure.colorbar`.
    cbar_ax : matplotlib Axes, optional
        Axes in which to draw the colorbar, otherwise take space from the
        main Axes.
    square : bool, optional
        If True, set the Axes aspect to "equal" so each cell will be
        square-shaped.
    xticklabels, yticklabels : "auto", bool, list-like, or int, optional
        If True, plot the column names of the dataframe. If False, don't plot
        the column names. If list-like, plot these alternate labels as the
        xticklabels. If an integer, use the column names but plot only every
        n label. If "auto", try to densely plot non-overlapping labels.
    mask : bool array or DataFrame, optional
        If passed, data will not be shown in cells where ``mask`` is True.
        Cells with missing values are automatically masked.
    ax : matplotlib Axes, optional
        Axes in which to draw the plot, otherwise use the currently-active
        Axes.
    kwargs : other keyword arguments
        All other keyword arguments are passed to
        :meth:`matplotlib.axes.Axes.pcolormesh`.
        
    Returns
    -------
    ax : matplotlib Axes
        Axes object with the heatmap.
    """
    # Initialize the _MosaicHeatMapper class
    plotter = _MosaicHeatMapper(data, vmin, vmax, cmap, center, robust, annot, fmt,
                                annot_kws, cbar, cbar_kws,
                                xticklabels, yticklabels, mask)
    
    # Add the linewidths and linecolor kwargs
#     kwargs["linewidths"] = linewidths
#     kwargs["linecolor"] = linecolor
    
    # Draw the plot and return the Axes
    if ax is None:
        ax = plt.gca()
    if square:
        ax.set_aspect("equal")
    plotter.plot(ax, cbar_ax, kwargs)
    return ax
