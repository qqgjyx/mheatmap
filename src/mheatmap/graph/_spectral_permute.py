"""
Spectral Reordering of Confusion Matrices

This module implements spectral reordering algorithms for confusion matrices,
using graph Laplacian eigenvectors to reveal block structures and patterns.
"""

# Copyright (c) 2024 Juntang Wang
# License: MIT
#
# This work was developed as part of:
# CS521 Matrix Analysis and Scientific Computing
# Duke University, Fall 2024
# Instructor: Prof. Xiaobai Sun
#
# Based on research from:
# "Clustering Methods for Hyperspectral Image Analysis"
# Authors: Juntang Wang, Dimitris Floros, Nikos Pitsianis, Xiaobai Sun

import numpy as np
from matplotlib.pylab import eigh

from ._copermute_from_bipermute import copermute_from_bipermute
from ._two_walk_laplacian import two_walk_laplacian
from ..utils import (
    plot_bipartite_confusion_matrix,
    plot_eigen,
)


###############################################################################
#                                                                             #
#                         Spectral Permutation                                #
#                                                                             #
###############################################################################
def spectral_permute(
    B: np.ndarray, labels: np.ndarray, mode: str = "tw"
) -> tuple[np.ndarray, np.ndarray]:
    """`spectral_permute(B, labels, mode='tw')`

    Perform spectral reordering of a confusion matrix using graph Laplacian eigenvectors.

    This function implements spectral reordering to reveal block structures in confusion matrices
    by analyzing the eigenvectors of the graph Laplacian. The reordering is based on the Fiedler
    vector (eigenvector corresponding to the second smallest eigenvalue), which provides an optimal
    ordering that groups similar classes together.

    Parameters
    ----------
    B : np.ndarray
        Input confusion matrix to be reordered, shape (n_classes, n_classes)
    labels : np.ndarray
        Class labels corresponding to matrix rows/columns, shape (n_classes,)
    mode : {'tw', 'fiedler'}, default='tw'
        Spectral reordering method:
        - 'tw': Use two-walk Laplacian for bipartite graph analysis
        - 'fiedler': Use standard Fiedler vector approach

    Returns
    -------
    reordered_cm : np.ndarray
        Reordered confusion matrix with revealed block structure
    reordered_labels : np.ndarray
        Class labels reordered to match the permuted matrix rows/columns

    See Also
    --------
    mheatmap.rms_permute : Alternative reordering using merge/split patterns
    mheatmap.amc_postprocess : Post-processing tools for confusion matrices
    mheatmap.graph.two_walk_laplacian : Two-walk Laplacian computation

    Notes
    -----
    The algorithm proceeds in the following steps:
    1. For mode='tw':
        - Constructs two-walk Laplacian capturing bipartite graph structure
        - Handles isolated vertices automatically
    2. For mode='fiedler':
        - Computes standard graph Laplacian L = D - A
    3. Finds Fiedler vector (second smallest eigenvector)
    4. Sorts vertices based on Fiedler vector components
    5. Applies resulting permutation to matrix and labels

    References
    ----------
    .. [1] Fiedler, M. (1973). Algebraic connectivity of graphs.
           Czechoslovak Mathematical Journal, 23(2), 298-305.
    .. [2] Sun, X. (2024). Matrix, Graph and Network Analysis.
           CS521 Course Notes, Duke University.

    Examples
    --------
    >>> import numpy as np
    >>> conf_mat = np.array([[5, 2, 0], [2, 3, 1], [0, 1, 4]])
    >>> labels = np.array(['A', 'B', 'C'])
    >>> reordered_mat, reordered_labs = spectral_permute(conf_mat, labels)
    """
    rows, cols = B.shape

    if mode == "fiedler":
        # Compute standard graph Laplacian L = D - A
        D = np.diag(np.sum(B, axis=1))
        L = D - B

        # Get eigendecomposition and find Fiedler vector
        eigenvalues, eigenvectors = eigh(L)
        nonzero_idx = np.where(np.abs(eigenvalues) > 1e-10)[0][0]
        fiedler_vector = eigenvectors[:, nonzero_idx]

        # Visualize eigenspectrum
        plot_eigen(eigenvalues, eigenvectors)

        # Sort vertices based on Fiedler vector
        sorted_rows_indices = np.argsort(fiedler_vector)
        sorted_cols_indices = sorted_rows_indices

    elif mode == "tw":
        # Handle isolated vertices
        B, labels = _put_zero_rows_cols_tail(B, labels)
        B_sub, Bsub_rows, Bsub_cols = _get_B_sub(B)

        # Compute two-walk Laplacian and its eigendecomposition
        L = two_walk_laplacian(B_sub)
        eigenvalues, eigenvectors = eigh(L)

        # Get Fiedler vector for spectral ordering
        nonzero_idx = np.where(np.abs(eigenvalues) > 1e-10)[0][0]
        fiedler_vector = eigenvectors[:, nonzero_idx]

        # Visualize eigenspectrum
        plot_eigen(eigenvalues, eigenvectors)

        # Get permutation from Fiedler vector
        p_Asub = np.argsort(fiedler_vector)

        # Map bipartite permutation to matrix dimensions
        sorted_rows_indices, sorted_cols_indices = copermute_from_bipermute(
            [rows, cols], Bsub_rows, Bsub_cols, p_Asub
        )

    # Apply permutation to get reordered matrix and labels
    reordered_B = B[sorted_rows_indices, :][:, sorted_cols_indices]
    reordered_labels = labels[sorted_rows_indices]

    # Visualize result
    plot_bipartite_confusion_matrix(reordered_B, reordered_labels)

    return reordered_B, reordered_labels


###############################################################################
#                                                                             #
#                         Helper Functions                                    #
#                                                                             #
###############################################################################
def _get_B_sub(B: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract non-zero submatrix from confusion matrix.

    Removes zero rows and columns from the input matrix and returns the resulting
    submatrix along with mapping indices.

    Parameters
    ----------
    B : np.ndarray
        Input confusion matrix. Should be a 2D numpy array.

    Returns
    -------
    B_sub : np.ndarray
        Normalized submatrix with zero rows/columns removed
    nonzero_rows : np.ndarray
        Indices mapping submatrix rows to original matrix rows
    nonzero_cols : np.ndarray
        Indices mapping submatrix columns to original matrix columns
    """
    # Identify zero rows/columns using row/column sums
    zero_rows = np.where(np.sum(B, axis=1) == 0)[0]
    zero_cols = np.where(np.sum(B, axis=0) == 0)[0]

    # Get indices of non-zero rows/columns
    nonzero_rows = np.setdiff1d(np.arange(B.shape[0]), zero_rows)
    nonzero_cols = np.setdiff1d(np.arange(B.shape[1]), zero_cols)

    # Extract and normalize submatrix
    B_sub = B[np.ix_(nonzero_rows, nonzero_cols)]
    B_sub = B_sub / np.max(B_sub)  # Normalize to [0,1] range

    return B_sub, nonzero_rows, nonzero_cols


def _put_zero_rows_cols_tail(
    B: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Move zero rows and columns to the end of the matrix.

    Identifies rows and columns with negligible values (sum < 0.1% of total) and moves them
    to the end of the matrix. This helps isolate the core structure of the confusion matrix
    from sparse/empty regions.

    Parameters
    ----------
    B : np.ndarray
        Input confusion matrix to be reordered
    labels : np.ndarray
        Labels corresponding to matrix rows that will be reordered to match

    Returns
    -------
    B : np.ndarray
        Reordered matrix with negligible rows/columns at the end
    labels : np.ndarray
        Labels reordered to match the new row ordering
    """
    # Identify negligible rows/columns (sum < 0.1% of total)
    total = np.sum(B)
    threshold = 1e-3 * total

    zero_rows = np.where(np.sum(B, axis=1) < threshold)[0]
    zero_cols = np.where(np.sum(B, axis=0) < threshold)[0]

    # Get indices of significant rows/columns
    nonzero_rows = np.setdiff1d(np.arange(B.shape[0]), zero_rows)
    nonzero_cols = np.setdiff1d(np.arange(B.shape[1]), zero_cols)

    # Reorder matrix by concatenating significant and negligible sections
    B = np.concatenate([B[nonzero_rows], B[zero_rows]], axis=0)
    B = np.concatenate([B[:, nonzero_cols], B[:, zero_cols]], axis=1)

    # Reorder labels to maintain correspondence with matrix rows
    labels = np.concatenate([labels[nonzero_rows], labels[zero_rows]])

    return B, labels
