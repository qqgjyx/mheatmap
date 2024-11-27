"""
Two-walk Laplacian

This module implements the two-walk Laplacian matrix computation for bipartite graphs,
which captures both direct connections and two-step walks in the graph structure.
"""

# Copyright (c) 2024 Juntang Wang, Xiaobai Sun, Dimitris Floros
# License: MIT

import numpy as np


def two_walk_laplacian(
    B_sub: np.ndarray, alpha: float = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`two_walk_laplacia(B_sub, alpha=1)`

    Compute the two-walk Laplacian matrix of a bipartite graph.

    For a bipartite graph with biadjacency matrix B, constructs the two-walk Laplacian
    by first forming the two-walk adjacency matrix A_tw and then computing L_tw = D_tw - A_tw,
    where D_tw is the diagonal degree matrix.

    Parameters
    ----------
    B_sub : np.ndarray
        Biadjacency matrix of shape (m, n) representing connections between two vertex sets
    alpha : float, default=1.0
        Scaling factor for the adjacency matrix term in the Laplacian computation

    Returns
    -------
    L_tw : np.ndarray
        Two-walk Laplacian matrix of shape (m+n, m+n)
    Bsub_rows : np.ndarray
        Indices of non-zero rows in the input matrix
    Bsub_cols : np.ndarray
        Indices of non-zero columns in the input matrix

    Notes
    -----
    The two-walk adjacency matrix A_tw has the block structure:
        [BB^T    αB  ]
        [αB^T   B^TB ]

    where α is the scaling factor controlling the influence of direct connections.

    The implementation automatically handles isolated vertices by removing rows/columns
    with all zeros before computation. The returned indices enable mapping back to
    the original matrix dimensions.

    References
    ----------
    .. [1] Sun, X. (2024). Graph Algorithms for Matrix Analysis. CS521 Course Notes,
           Duke University.

    Examples
    --------
    >>> B = np.array([[1, 0], [1, 1]])
    >>> L_tw, rows, cols = two_walk_laplacian(B)
    >>> print(L_tw.shape)
    (4, 4)
    """
    # Form the two-walk adjacency matrix with block structure
    A_tw = np.block(
        [[B_sub @ B_sub.T, alpha * B_sub], [alpha * B_sub.T, B_sub.T @ B_sub]]
    )

    # Compute degree matrix and Laplacian
    D_tw = np.diag(np.sum(A_tw, axis=1))
    L_tw = D_tw - A_tw

    return L_tw
