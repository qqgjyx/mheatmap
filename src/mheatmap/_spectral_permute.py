"""
Spectral reordering of confusion matrix (prototype)
"""

# Copyright (c) 2024, Juntang Wang
# All rights reserved.
#
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

from matplotlib.pylab import eigh
import numpy as np


###############################################################################
#                                                                             #
#                         Spectral Permutation                                #
#                                                                             #
###############################################################################
def spectral_permute(conf_mat, labels):
    """`spectral_permute(conf_mat, labels)`

    Perform spectral reordering of a confusion matrix using the Fiedler vector.

    This function applies spectral reordering to rearrange the rows and columns of a
    confusion matrix based on the eigenvector corresponding to the second smallest
    eigenvalue (Fiedler vector) of the graph Laplacian. This ordering tends to group
    similar classes together, making block structures more apparent.

    Parameters
    ----------
    conf_mat : numpy.ndarray
        Square confusion matrix to be reordered, shape (n_classes, n_classes)
    labels : numpy.ndarray
        Array of class labels corresponding to matrix rows/columns, shape (n_classes,)

    Returns
    -------
    reordered_cm : numpy.ndarray
        Confusion matrix after spectral reordering
    reordered_labels : numpy.ndarray
        Class labels reordered to match the permuted matrix

    See Also
    --------
    mheatmap.rms_permute : Alternative permutation using merge/split relationships
    mheatmap.amc_postprocess : Related module for confusion matrix post-processing

    Notes
    -----
    The spectral reordering process:
    1. Computes the degree matrix D and Laplacian matrix L = D - A
    2. Finds the eigenvector corresponding to second smallest eigenvalue (Fiedler vector)
    3. Sorts matrix rows/columns based on Fiedler vector values

    Examples
    --------
    >>> import numpy as np
    >>> conf_mat = np.array([[5, 2, 0], [2, 3, 1], [0, 1, 4]])
    >>> labels = np.array([1, 2, 3])
    >>> reordered_mat, reordered_labs = spectral_permute(conf_mat, labels)
    """
    # Step 1: Compute the degree matrix D and Laplacian matrix L
    D = np.diag(np.sum(conf_mat, axis=1))
    L = D - conf_mat

    # Step 2: Compute the Fiedler vector (second-smallest eigenvector)
    eigenvalues, eigenvectors = eigh(L)
    fiedler_vector = eigenvectors[:, 1]

    # Step 3: Sort indices based on the Fiedler vector values
    sorted_indices = np.argsort(fiedler_vector)

    # Step 4: Reorder the confusion matrix
    reordered_cm = conf_mat[sorted_indices, :][:, sorted_indices]
    reordered_labels = labels[sorted_indices]

    return reordered_cm, reordered_labels
