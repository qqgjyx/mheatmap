from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from mheatmap import mosaic_heatmap


def diagonal_band_mass(matrix: np.ndarray, frac: float) -> float:
    total = float(matrix.sum())
    if total <= 0:
        return 0.0

    n_rows, n_cols = matrix.shape
    band = max(1, int(frac * min(n_rows, n_cols)))
    score = 0.0

    for row_index in range(n_rows):
        center = row_index * n_cols / n_rows
        start = max(0, int(np.floor(center - band)))
        stop = min(n_cols, int(np.ceil(center + band + 1)))
        score += float(matrix[row_index, start:stop].sum())

    return score / total


def recover_column_labels(
    original_matrix: np.ndarray,
    original_row_labels: np.ndarray,
    original_col_labels: np.ndarray,
    reordered_matrix: np.ndarray,
    reordered_row_labels: np.ndarray,
) -> np.ndarray:
    row_lookup = {label: index for index, label in enumerate(original_row_labels)}
    row_order = np.array(
        [row_lookup[label] for label in reordered_row_labels],
        dtype=int,
    )
    row_reordered_original = original_matrix[row_order, :]

    column_lookup: dict[bytes, list[int]] = defaultdict(list)
    for col_index in range(row_reordered_original.shape[1]):
        key = np.ascontiguousarray(row_reordered_original[:, col_index]).tobytes()
        column_lookup[key].append(col_index)

    recovered_indices = []
    for col_index in range(reordered_matrix.shape[1]):
        key = np.ascontiguousarray(reordered_matrix[:, col_index]).tobytes()
        matches = column_lookup[key]
        if not matches:
            raise ValueError("Could not recover reordered column labels from matrix.")
        recovered_indices.append(matches.pop(0))

    return original_col_labels[np.array(recovered_indices, dtype=int)]


def maybe_ticklabels(labels: np.ndarray, max_labels: int = 40):
    return labels if len(labels) <= max_labels else False


def write_matrix_csv_fast(
    path: Path,
    matrix: np.ndarray,
    row_labels: np.ndarray,
    col_labels: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_label", *list(col_labels)])
        rounded = np.rint(matrix)
        use_int = np.allclose(matrix, rounded)
        for label, row, rounded_row in zip(row_labels, matrix, rounded, strict=True):
            values = rounded_row.astype(int) if use_int else row
            writer.writerow([label, *values.tolist()])


def write_metadata_csv(
    path: Path,
    header: tuple[str, ...],
    rows: list[tuple[object, ...]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def plot_triptych(
    original_matrix: np.ndarray,
    tw_matrix: np.ndarray,
    original_row_labels: np.ndarray,
    original_col_labels: np.ndarray,
    tw_row_labels: np.ndarray,
    tw_col_labels: np.ndarray,
    output_path: Path,
    figure_title: str,
    subtitle: str,
    colorbar_label: str,
    cmap: str = "YlGnBu",
    band_frac: float = 0.12,
    mask_below: float = 0.0,
) -> None:
    scale_values = np.concatenate(
        [
            original_matrix[original_matrix > mask_below],
            tw_matrix[tw_matrix > mask_below],
        ]
    )
    if scale_values.size == 0:
        raise ValueError("Cannot plot an all-zero matrix.")

    norm = LogNorm(vmin=float(scale_values.min()), vmax=float(scale_values.max()))
    original_masked = np.ma.masked_where(
        original_matrix <= mask_below,
        original_matrix,
    )
    tw_masked = np.ma.masked_where(tw_matrix <= mask_below, tw_matrix)

    fig, axes = plt.subplots(1, 3, figsize=(22, 8), constrained_layout=True)

    panels = [
        ("Original", original_masked, original_row_labels, original_col_labels),
        (
            "TW Spectral Reordered",
            tw_masked,
            tw_row_labels,
            tw_col_labels,
        ),
    ]

    for ax, (title, matrix, row_labels, col_labels) in zip(
        axes[:2], panels, strict=True
    ):
        image = ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )
        ax.set_title(title, fontsize=18, color="#4A4A4A")
        xlabels = maybe_ticklabels(col_labels)
        ylabels = maybe_ticklabels(row_labels)
        if xlabels is False:
            ax.set_xticks([])
        else:
            ax.set_xticks(np.arange(len(col_labels)))
            ax.set_xticklabels(xlabels, rotation=90, fontsize=7)
            ax.xaxis.set_ticks_position("top")
        if ylabels is False:
            ax.set_yticks([])
        else:
            ax.set_yticks(np.arange(len(row_labels)))
            ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows")
        ax.tick_params(colors="#4A4A4A")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    axes[2].set_title("TW Spectral Reordered + Mosaic", fontsize=18, color="#4A4A4A")
    mosaic_heatmap(
        tw_matrix,
        ax=axes[2],
        xticklabels=maybe_ticklabels(tw_col_labels),
        yticklabels=maybe_ticklabels(tw_row_labels),
        cmap=cmap,
        mask=tw_matrix <= mask_below,
        norm=norm,
        cbar=True,
        cbar_kws={"label": colorbar_label},
        rasterized=True,
    )
    axes[2].tick_params(colors="#4A4A4A")
    axes[2].xaxis.set_ticks_position("top")
    axes[2].set_xlabel("Columns")
    axes[2].set_ylabel("Rows")

    original_score = diagonal_band_mass(original_matrix, band_frac)
    tw_score = diagonal_band_mass(tw_matrix, band_frac)
    score_text = (
        f"Diagonal-band mass: original={original_score:.3f}, "
        f"tw={tw_score:.3f}"
    )
    fig.suptitle(
        f"{figure_title}\n{subtitle}\n{score_text}",
        y=1.02,
        fontsize=14,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
