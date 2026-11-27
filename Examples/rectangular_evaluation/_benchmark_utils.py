from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from _rectangular_tw_common import diagonal_band_mass
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.linalg import eigh
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from mheatmap.graph import (
    copermute_from_bipermute,
    spectral_permute,
    two_walk_laplacian,
)


@dataclass(frozen=True)
class ReorderResult:
    matrix: np.ndarray
    row_order: np.ndarray
    col_order: np.ndarray


def load_matrix_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    col_labels = np.array(rows[0][1:])
    row_labels = np.array([row[0] for row in rows[1:]])
    matrix = np.array([[float(value) for value in row[1:]] for row in rows[1:]])
    return matrix, row_labels, col_labels


def apply_orders(
    matrix: np.ndarray,
    row_order: np.ndarray,
    col_order: np.ndarray,
) -> np.ndarray:
    return matrix[row_order][:, col_order]


def recover_column_order(
    original_matrix: np.ndarray,
    row_order: np.ndarray,
    reordered_matrix: np.ndarray,
) -> np.ndarray:
    row_reordered_original = original_matrix[row_order, :]

    column_lookup: dict[bytes, list[int]] = {}
    for col_index in range(row_reordered_original.shape[1]):
        key = np.ascontiguousarray(row_reordered_original[:, col_index]).tobytes()
        column_lookup.setdefault(key, []).append(col_index)

    recovered = []
    for col_index in range(reordered_matrix.shape[1]):
        key = np.ascontiguousarray(reordered_matrix[:, col_index]).tobytes()
        matches = column_lookup.get(key)
        if not matches:
            raise ValueError("Could not recover reordered column order from matrix.")
        recovered.append(matches.pop(0))
    return np.array(recovered, dtype=int)


def normalized_two_sum(matrix: np.ndarray) -> float:
    total = float(np.sum(matrix))
    if total <= 0:
        return 0.0

    n_rows, n_cols = matrix.shape
    row_positions = np.linspace(0.0, 1.0, n_rows)[:, None]
    if n_cols == 1:
        col_positions = np.zeros((1, 1))
    else:
        col_positions = np.linspace(0.0, 1.0, n_cols)[None, :]
    squared_distance = (row_positions - col_positions) ** 2
    return float(np.sum(matrix * squared_distance) / total)


def mwb_auc(matrix: np.ndarray, widths: np.ndarray) -> float:
    scores = np.array([diagonal_band_mass(matrix, width) for width in widths])
    return float(np.trapezoid(scores, widths) / (widths[-1] - widths[0]))


def orient_orders_for_diagonal(
    matrix: np.ndarray,
    row_order: np.ndarray,
    col_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = [
        (row_order, col_order),
        (row_order[::-1], col_order),
        (row_order, col_order[::-1]),
        (row_order[::-1], col_order[::-1]),
    ]

    best_score = None
    best_pair = candidates[0]
    for candidate_rows, candidate_cols in candidates:
        candidate_matrix = apply_orders(matrix, candidate_rows, candidate_cols)
        score = normalized_two_sum(candidate_matrix)
        if best_score is None or score < best_score:
            best_score = score
            best_pair = (candidate_rows, candidate_cols)
    return best_pair


def nonzero_axis_indices(
    matrix: np.ndarray,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.sum(matrix, axis=axis)
    nonzero = np.where(sums > 0)[0]
    zero = np.where(sums <= 0)[0]
    return nonzero, zero


def _center_of_mass(values: np.ndarray, axis_length: int) -> np.ndarray:
    positions = np.arange(axis_length, dtype=float)
    totals = values.sum(axis=1)
    weighted = values @ positions
    return np.divide(
        weighted,
        totals,
        out=np.full(values.shape[0], axis_length, dtype=float),
        where=totals > 0,
    )


def marginal_sort_reorder(matrix: np.ndarray) -> ReorderResult:
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    row_centers = _center_of_mass(matrix, matrix.shape[1])
    col_centers = _center_of_mass(matrix.T, matrix.shape[0])

    row_order = np.lexsort((row_centers, -row_sums))
    col_order = np.lexsort((col_centers, -col_sums))
    row_order, col_order = orient_orders_for_diagonal(matrix, row_order, col_order)
    return ReorderResult(
        apply_orders(matrix, row_order, col_order),
        row_order,
        col_order,
    )


def _olo_order(vectors: np.ndarray) -> np.ndarray:
    count = vectors.shape[0]
    if count <= 1:
        return np.arange(count, dtype=int)
    if count == 2:
        totals = np.sum(vectors, axis=1)
        return np.argsort(-totals, kind="mergesort")

    distances = pdist(vectors, metric="cosine")
    if not np.isfinite(distances).all() or np.allclose(distances, 0):
        totals = np.sum(vectors, axis=1)
        centers = _center_of_mass(vectors, vectors.shape[1])
        return np.lexsort((centers, -totals))

    tree = linkage(distances, method="average")
    ordered_tree = optimal_leaf_ordering(tree, distances)
    return leaves_list(ordered_tree).astype(int)


def hierarchical_olo_reorder(matrix: np.ndarray) -> ReorderResult:
    row_nonzero, row_zero = nonzero_axis_indices(matrix, axis=1)
    col_nonzero, col_zero = nonzero_axis_indices(matrix, axis=0)

    if len(row_nonzero) == 0 or len(col_nonzero) == 0:
        row_order = np.arange(matrix.shape[0], dtype=int)
        col_order = np.arange(matrix.shape[1], dtype=int)
        return ReorderResult(matrix.copy(), row_order, col_order)

    row_local = _olo_order(matrix[row_nonzero][:, col_nonzero])
    col_local = _olo_order(matrix[row_nonzero][:, col_nonzero].T)
    row_order = np.concatenate([row_nonzero[row_local], row_zero])
    col_order = np.concatenate([col_nonzero[col_local], col_zero])
    row_order, col_order = orient_orders_for_diagonal(matrix, row_order, col_order)
    return ReorderResult(
        apply_orders(matrix, row_order, col_order),
        row_order,
        col_order,
    )


def one_walk_reorder(matrix: np.ndarray) -> ReorderResult:
    rows, cols = matrix.shape
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    nonzero_rows = np.where(row_sums > 0)[0]
    nonzero_cols = np.where(col_sums > 0)[0]

    if len(nonzero_rows) == 0 or len(nonzero_cols) == 0:
        row_order = np.arange(rows, dtype=int)
        col_order = np.arange(cols, dtype=int)
        return ReorderResult(matrix.copy(), row_order, col_order)

    B_sub = matrix[np.ix_(nonzero_rows, nonzero_cols)].astype(float)
    if np.max(B_sub) > 0:
        B_sub = B_sub / np.max(B_sub)

    zeros_rr = np.zeros((B_sub.shape[0], B_sub.shape[0]), dtype=float)
    zeros_cc = np.zeros((B_sub.shape[1], B_sub.shape[1]), dtype=float)
    adjacency = np.block([[zeros_rr, B_sub], [B_sub.T, zeros_cc]])
    degree = np.diag(np.sum(adjacency, axis=1))
    laplacian = degree - adjacency
    eigenvalues, eigenvectors = eigh(laplacian)
    fiedler_index = np.where(np.abs(eigenvalues) > 1e-10)[0][0]
    bipartite_order = np.argsort(eigenvectors[:, fiedler_index])
    row_order, col_order = copermute_from_bipermute(
        [rows, cols],
        nonzero_rows,
        nonzero_cols,
        bipartite_order,
    )
    row_order, col_order = orient_orders_for_diagonal(matrix, row_order, col_order)
    return ReorderResult(
        apply_orders(matrix, row_order, col_order),
        row_order,
        col_order,
    )


def tw_reorder(matrix: np.ndarray) -> ReorderResult:
    row_labels = np.arange(matrix.shape[0], dtype=int)
    tw_matrix, tw_row_labels = spectral_permute(matrix, row_labels, mode="tw")
    row_order = tw_row_labels.astype(int)
    col_order = recover_column_order(matrix, row_order, tw_matrix)
    row_order, col_order = orient_orders_for_diagonal(matrix, row_order, col_order)
    return ReorderResult(
        apply_orders(matrix, row_order, col_order),
        row_order,
        col_order,
    )


def tw_alpha_reorder(matrix: np.ndarray, alpha: float) -> ReorderResult:
    rows, cols = matrix.shape
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    nonzero_rows = np.where(row_sums > 0)[0]
    nonzero_cols = np.where(col_sums > 0)[0]

    if len(nonzero_rows) == 0 or len(nonzero_cols) == 0:
        row_order = np.arange(rows, dtype=int)
        col_order = np.arange(cols, dtype=int)
        return ReorderResult(matrix.copy(), row_order, col_order)

    B_sub = matrix[np.ix_(nonzero_rows, nonzero_cols)].astype(float)
    if np.max(B_sub) > 0:
        B_sub = B_sub / np.max(B_sub)

    laplacian = two_walk_laplacian(B_sub, alpha=alpha)
    eigenvalues, eigenvectors = eigh(laplacian)
    fiedler_index = np.where(np.abs(eigenvalues) > 1e-10)[0][0]
    bipartite_order = np.argsort(eigenvectors[:, fiedler_index])
    row_order, col_order = copermute_from_bipermute(
        [rows, cols],
        nonzero_rows,
        nonzero_cols,
        bipartite_order,
    )
    row_order, col_order = orient_orders_for_diagonal(matrix, row_order, col_order)
    return ReorderResult(
        apply_orders(matrix, row_order, col_order),
        row_order,
        col_order,
    )


def compute_curves(
    method_matrices: dict[str, np.ndarray],
    widths: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        method: np.array([diagonal_band_mass(matrix, width) for width in widths])
        for method, matrix in method_matrices.items()
    }


def _adjacent_cosine_gaps(vectors: np.ndarray) -> np.ndarray:
    if vectors.shape[0] <= 1:
        return np.array([], dtype=float)

    left = vectors[:-1].astype(float, copy=False)
    right = vectors[1:].astype(float, copy=False)
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denom = left_norm * right_norm
    dot = np.sum(left * right, axis=1)
    cosine = np.divide(
        dot,
        denom,
        out=np.zeros_like(dot, dtype=float),
        where=denom > 0,
    )
    both_zero = (left_norm == 0) & (right_norm == 0)
    cosine[both_zero] = 1.0
    cosine = np.clip(cosine, -1.0, 1.0)
    return 1.0 - cosine


def contiguous_gap_partition_labels(
    vectors: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    n_items = vectors.shape[0]
    if n_items == 0:
        return np.array([], dtype=int)
    if n_clusters <= 1 or n_items == 1:
        return np.zeros(n_items, dtype=int)

    n_clusters = min(n_clusters, n_items)
    gaps = _adjacent_cosine_gaps(vectors)
    if gaps.size < n_clusters - 1 or np.allclose(gaps, gaps[0] if gaps.size else 0.0):
        boundaries = np.linspace(0, n_items, n_clusters + 1, dtype=int)[1:-1]
    else:
        boundaries = np.sort(np.argsort(gaps)[-(n_clusters - 1) :] + 1)

    labels = np.empty(n_items, dtype=int)
    start = 0
    for cluster_id, stop in enumerate(np.append(boundaries, n_items)):
        labels[start:stop] = cluster_id
        start = stop
    return labels


def contiguous_cluster_scores(
    reordered_matrix: np.ndarray,
    true_row_labels: np.ndarray,
    true_col_labels: np.ndarray,
) -> dict[str, float]:
    row_k = int(np.unique(true_row_labels).size)
    col_k = int(np.unique(true_col_labels).size)
    pred_row_labels = contiguous_gap_partition_labels(reordered_matrix, row_k)
    pred_col_labels = contiguous_gap_partition_labels(reordered_matrix.T, col_k)

    ari_row = float(adjusted_rand_score(true_row_labels, pred_row_labels))
    ari_col = float(adjusted_rand_score(true_col_labels, pred_col_labels))
    nmi_row = float(normalized_mutual_info_score(true_row_labels, pred_row_labels))
    nmi_col = float(normalized_mutual_info_score(true_col_labels, pred_col_labels))
    return {
        "ari_row": ari_row,
        "ari_col": ari_col,
        "ari_mean": 0.5 * (ari_row + ari_col),
        "nmi_row": nmi_row,
        "nmi_col": nmi_col,
        "nmi_mean": 0.5 * (nmi_row + nmi_col),
    }
