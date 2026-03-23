"""Functions to visualize matrices of data"""

# Copyright (c) 2024 Juntang Wang
# All rights reserved.
# Licensed under the MIT License.

import matplotlib.pyplot as plt
import numpy as np
from seaborn import despine
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

    return x_edges, y_edges, x_centers, y_centers


class _MosaicHeatMapper:
    """Prepare and plot mosaic heatmap data.

    Handles data normalization, tick label processing, colormap setup,
    and rendering — without depending on seaborn's private _HeatMapper.
    """

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
        # Convert data to ndarray (handle DataFrame)
        if hasattr(data, "values"):
            plot_data = data.values
        else:
            plot_data = np.asarray(data)

        # Handle mask
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            plot_data = np.ma.masked_where(mask, plot_data)

        self.data = plot_data
        self.plot_data = plot_data.astype(float)

        # Handle tick labels from DataFrame
        if hasattr(data, "index") and hasattr(data, "columns"):
            if xticklabels is True or xticklabels == "auto":
                xticklabels = list(data.columns)
            if yticklabels is True or yticklabels == "auto":
                yticklabels = list(data.index)
            self.xlabel = data.columns.name or ""
            self.ylabel = data.index.name or ""
        else:
            if xticklabels is True or xticklabels == "auto":
                xticklabels = list(range(plot_data.shape[1]))
            if yticklabels is True or yticklabels == "auto":
                yticklabels = list(range(plot_data.shape[0]))
            self.xlabel = ""
            self.ylabel = ""

        if xticklabels is False:
            xticklabels = []
        if yticklabels is False:
            yticklabels = []

        self.xticklabels = xticklabels
        self.yticklabels = yticklabels

        # Colormap
        if cmap is None:
            cmap = "rocket" if center is None else "icefire"
        self.cmap = cmap

        # Value range
        if robust and vmin is None:
            vmin = np.nanpercentile(plot_data, 2)
        if robust and vmax is None:
            vmax = np.nanpercentile(plot_data, 98)
        self.vmin = vmin if vmin is not None else np.nanmin(plot_data)
        self.vmax = vmax if vmax is not None else np.nanmax(plot_data)

        if center is not None:
            max_diff = max(abs(self.vmax - center), abs(self.vmin - center))
            self.vmin = center - max_diff
            self.vmax = center + max_diff

        # Annotation
        self.annot = annot
        self.fmt = fmt
        self.annot_kws = annot_kws or {}

        # Colorbar
        self.cbar = cbar
        self.cbar_kws = cbar_kws or {}

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

    Creates a mosaic heatmap where the column widths and row
    heights are proportional to the marginal sums of the data
    matrix.

    Parameters
    ----------
    data : array-like
        2D dataset that can be coerced into an ndarray. If a
        pandas DataFrame is provided, the index/column information
        will be used to label the columns and rows.
    vmin, vmax : float, optional
        Values to anchor the colormap.
    cmap : str or matplotlib.colors.Colormap, optional
        The mapping from data values to color space.
    center : float, optional
        The value at which to center the colormap for divergent data.
    robust : bool, optional
        If True, compute colormap range using robust quantiles.
    annot : bool or array-like, optional
        If True, write the data value in each cell.
    fmt : str, optional
        String formatting code for annotation values. Default: '.2g'
    annot_kws : dict, optional
        Keyword arguments for annotation text.
    cbar : bool, optional
        Whether to draw a colorbar. Default: True
    cbar_kws : dict, optional
        Keyword arguments for colorbar.
    cbar_ax : matplotlib.axes.Axes, optional
        Axes in which to draw the colorbar.
    square : bool, optional
        If True, set aspect ratio to "equal". Default: False
    xticklabels, yticklabels : 'auto', bool, array-like, optional
        Tick label specification.
    mask : bool array or DataFrame, optional
        If True in a cell, data is not shown.
    ax : matplotlib.axes.Axes, optional
        Axes in which to draw the plot. Uses current axes if None.
    **kwargs : dict
        Additional keyword arguments passed to pcolormesh.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object with the heatmap.

    Examples
    --------
    >>> import numpy as np
    >>> from mheatmap import mosaic_heatmap
    >>> data = np.array([[10, 2, 0], [1, 8, 3], [0, 1, 12]])
    >>> ax = mosaic_heatmap(data, annot=True, cmap='YlOrRd')
    """
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

    if ax is None:
        ax = plt.gca()
    if square:
        ax.set_aspect("equal")
    plotter.plot(ax, cbar_ax, kwargs)
    return ax
