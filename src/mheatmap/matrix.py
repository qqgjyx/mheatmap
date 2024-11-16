"""Functions to visualize matrices of data"""

# Copyright (c) 2024 Juntang Wang
# All rights reserved.
# Licensed under the MIT License.

import numpy as np
import matplotlib.pyplot as plt
from seaborn import despine
from seaborn.matrix import _HeatMapper
from seaborn.utils import axis_ticklabels_overlap

###############################################################################
#                                                                             #
#                         Mosaic Confusion Matrix                             #
#                                                                             #
###############################################################################


def _get_mh_xyc(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    # Xc, Yc = np.meshgrid(x_centers, y_centers)
    return x_edges, y_edges, x_centers, y_centers


# Override the _HeatMapper class
class _MosaicHeatMapper(_HeatMapper):
    """Draw a mosaic heatmap plot of a matrix with nice labels and colormaps."""

    def __init__(
        self,
        data,
        vmin,
        vmax,
        cmap,
        center,
        robust,
        annot,
        fmt,
        annot_kws,
        cbar,
        cbar_kws,
        xticklabels=True,
        yticklabels=True,
        mask=None,
    ):
        # Initialize the _HeatMapper class
        super().__init__(
            data,
            vmin,
            vmax,
            cmap,
            center,
            robust,
            annot,
            fmt,
            annot_kws,
            cbar,
            cbar_kws,
            xticklabels,
            yticklabels,
            mask,
        )

    def _annotate_mosaic_heatmap(self, ax):
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                ax.text(
                    self.x_centers[j],
                    self.y_centers[i],
                    f"{self.data[i, j]}",
                    ha="center",
                    va="center",
                    color="black",
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
        x_edges, y_edges, x_centers, y_centers = _get_mh_xyc(self.plot_data)

        mesh = ax.pcolormesh(
            x_edges, y_edges, self.plot_data, shading="flat", cmap=self.cmap, **kws
        )

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
            if kws.get("rasterized", False):
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
    data,
    *,
    vmin=None,
    vmax=None,
    cmap=None,
    center=None,
    robust=False,
    annot=None,
    fmt=".2g",
    annot_kws=None,
    linewidths=0,
    linecolor="white",
    cbar=True,
    cbar_kws=None,
    cbar_ax=None,
    square=False,
    xticklabels="auto",
    yticklabels="auto",
    mask=None,
    ax=None,
    **kwargs,
):
    """`mosaic_heatmap(data, ...)`

    Plot mosaic data as a color-encoded matrix.

    Creates a mosaic heatmap where the column widths and row heights are proportional
    to the marginal sums of the data matrix. This provides a visualization that
    encodes both the cell values through color and the marginal distributions
    through cell sizes.

    Parameters
    ----------
    data : array-like
        2D dataset that can be coerced into an ndarray. If a pandas DataFrame
        is provided, the index/column information will be used to label the
        columns and rows.
    vmin, vmax : float, optional
        Values to anchor the colormap. If not provided, they are inferred from the
        data and other keyword arguments.
    cmap : str or matplotlib.colors.Colormap, optional
        The mapping from data values to color space. If not provided, the
        default depends on whether ``center`` is set.
    center : float, optional
        The value at which to center the colormap for divergent data.
        Changes the default ``cmap`` if none is specified.
    robust : bool, optional
        If True and ``vmin`` or ``vmax`` are absent, compute colormap range using
        robust quantiles instead of extreme values.
    annot : bool or array-like, optional
        If True, write the data value in each cell. If array-like with same shape
        as ``data``, use this for annotation instead of the data. DataFrames match
        on position, not index.
    fmt : str, optional
        String formatting code for annotation values. Default: '.2g'
    annot_kws : dict, optional
        Keyword arguments for matplotlib.axes.Axes.text when ``annot`` is True.
    linewidths : float, optional
        Width of cell divider lines. Default: 0
    linecolor : color, optional
        Color of cell divider lines. Default: 'white'
    cbar : bool, optional
        Whether to draw a colorbar. Default: True
    cbar_kws : dict, optional
        Keyword arguments for matplotlib.figure.Figure.colorbar.
    cbar_ax : matplotlib.axes.Axes, optional
        Axes in which to draw the colorbar. If None, takes space from main Axes.
    square : bool, optional
        If True, set aspect ratio to "equal" for square cells. Default: False
    xticklabels, yticklabels : 'auto', bool, array-like, or int, optional
        - True: plot column/row names
        - False: don't plot labels
        - array-like: plot custom labels
        - int: plot every nth label
        - 'auto': plot non-overlapping labels
    mask : bool array or DataFrame, optional
        If True in a cell, data is not shown. Missing values are auto-masked.
    ax : matplotlib.axes.Axes, optional
        Axes in which to draw the plot. Uses current axes if None.
    **kwargs : dict
        Additional keyword arguments passed to matplotlib.axes.Axes.pcolormesh.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object with the heatmap.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from mheatmap import mosaic_heatmap
    >>>
    >>> # Generate sample confusion matrix data
    >>> data = np.array([[10, 2, 0], [1, 8, 3], [0, 1, 12]])
    >>>
    >>> # Create mosaic heatmap with annotations
    >>> fig, ax = plt.subplots(figsize=(8, 6))
    >>> mosaic_heatmap(data, annot=True, cmap='YlOrRd', fmt='d',
    ...               xticklabels=['A', 'B', 'C'],
    ...               yticklabels=['A', 'B', 'C'])
    >>> plt.title('Mosaic Confusion Matrix')
    >>> plt.show()

    Notes
    -----
    The mosaic heatmap is particularly useful for confusion matrices and contingency
    tables where the marginal distributions provide additional context beyond the
    cell values themselves.
    """
    # Initialize the _MosaicHeatMapper class
    plotter = _MosaicHeatMapper(
        data,
        vmin,
        vmax,
        cmap,
        center,
        robust,
        annot,
        fmt,
        annot_kws,
        cbar,
        cbar_kws,
        xticklabels,
        yticklabels,
        mask,
    )

    # Add the linewidths and linecolor kwargs
    # kwargs["linewidths"] = linewidths
    # kwargs["linecolor"] = linecolor

    # Draw the plot and return the Axes
    if ax is None:
        ax = plt.gca()
    if square:
        ax.set_aspect("equal")
    plotter.plot(ax, cbar_ax, kwargs)
    return ax
