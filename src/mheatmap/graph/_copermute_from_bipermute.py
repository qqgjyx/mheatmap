"""
Copermute from bi-permutation
"""

# Copyright (c) 2024 Dimitris Floros, Xiaobai Sun, Juntang Wang

import numpy as np


def copermute_from_bipermute(
    B_sizes: list[int], B_subrows: np.ndarray, B_subcols: np.ndarray, p_Asub: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`copermute_from_bipermute(B_sizes, B_subrows, B_subcols, p_Asub)`

    Copermute from bi-permutation.

    Renders row permutation and column permutation of matrix B according to a co-permutation
    of a submatrix Bsub via a bi-permutation in its symmetric embedding:
        Asub = [[0, Bsub], [Bsub.T, 0]]

    Parameters
    ----------
    B_sizes : array_like
        A 1x2 array containing the dimensions of matrix B: [nrows, ncols]
    B_subrows : array_like
        Row indices defining the submatrix Bsub, nrBsub x 1 integer array where nrBsub <= nrB
    B_subcols : array_like
        Column indices defining the submatrix Bsub, ncBsub x 1 integer array where ncBsub <= ncB
    p_Asub : array_like
        Permutation vector for the symmetric embedding of Bsub, (nr+nc)x1 integer array

    Returns
    -------
    tuple
        - p_Brows : Row permutation vector for matrix B, Bsizes[0]x1
        - p_Bcols : Column permutation vector for matrix B, Bsizes[1]x1

    Examples
    --------
    >>> import numpy as np
    >>> m, n = 5, 4  # matrix dimensions
    >>> B_sizes = [m, n]
    >>> # Use entire matrix as submatrix
    >>> p_Brows, p_Bcols = copermute_from_bipermute(
    ...     B_sizes,
    ...     np.arange(1,m+1),
    ...     np.arange(1,n+1),
    ...     np.random.permutation(m+n)+1
    ... )

    Notes
    -----
    Revision of recover_nonsymmetric_perm.m
    All variables renamed to be self-evident + additional documentation
    Nov. 22, 2024

    Authors
    -------
    - Dimitris Floros <dimitrios.floros@duke.edu>
    - Xiaobai Sun
    - Juntang Wang
    """
    nr_B = B_sizes[0]
    nc_B = B_sizes[1]

    nr_Bsub = len(B_subrows)
    nc_Bsub = len(B_subcols)

    # Set the markers for bipartite-embedding of Bsub: 1 for rows; 2 for columns
    bi_marker = np.concatenate(
        [np.ones(nr_Bsub, dtype=int), 2 * np.ones(nc_Bsub, dtype=int)]
    )
    bi_marker = bi_marker[p_Asub]

    # Separate row and column indices in the bi-permutation
    p_r = p_Asub[bi_marker == 1]
    p_c = p_Asub[bi_marker == 2] - nr_Bsub

    # Permute the given indices at input
    pr_Bsub = B_subrows[p_r]
    pc_Bsub = B_subcols[p_c]

    # Render co-permutation in B: place Bsub first, the remaining to the end
    p_Brows = np.concatenate([pr_Bsub, np.setdiff1d(np.arange(0, nr_B), pr_Bsub)])

    p_Bcols = np.concatenate([pc_Bsub, np.setdiff1d(np.arange(0, nc_B), pc_Bsub)])

    return p_Brows, p_Bcols
