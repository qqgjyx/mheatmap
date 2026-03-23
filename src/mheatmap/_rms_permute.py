"""
This script is used to permute the confusion matrix based on merge split idea.

    Greedly
    1. find elements to reverse merge/split, if precision/ecall is less than threshold.
    2. find the maximum entry in the corresponding row i or column i.
    *. NOTE: j_row means a merge of gt (e.g. gt1, gt2 -> pred1),
    *. j_col means a split of gt (e.g. gt1 -> pred1, pred2).
    3. permute (merge) by bring j to i + 1. `merge_map` record the merge relationship.
    4. update the conf_mat.
"""

# Copyright (c) 2024 Juntang Wang
# All rights reserved.
# Licensed under the MIT License.

# Code prototype: Xiaobai Sun, 2024
# Code inspired by: Dimitris, 2024
# Code modified by: Juntang Wang, 2024

from typing import NamedTuple

import numpy as np

# Default precision/recall threshold for identifying weak classifications.
# Below this threshold, a class is considered a candidate for merge/split grouping.
_DEFAULT_THRESHOLD = 0.37

###############################################################################
#                                                                             #
#                         Helper Functions                                    #
#                                                                             #
###############################################################################


def _make_rms_map(conf_mat: np.ndarray, threshold: float = _DEFAULT_THRESHOLD) -> dict:
    """
    Generates a reverse merge/split mapping based on confusion matrix analysis.

    Analyzes a confusion matrix to identify potential merge or split relationships
    between classes based on comparing precision/recall against off-diagonal maximums.
    A merge indicates combining ground truth classes, while a split indicates
    subdividing a ground truth class.

    Parameters
    ----------
    conf_mat : np.ndarray
        The confusion matrix to analyze
    threshold : float, optional
        Threshold ratio relative to matrix sum for identifying negligible
        diagonal elements, by default 1e-4

    Returns
    -------
    dict
        Mapping of class indices to their merge/split relationships, where:
        - Keys are source class indices
        - Values are tuples of (target_idx, relationship_type)
        - relationship_type is either 'rmerge' or 'rsplit'
    """
    # matrix_sum = conf_mat.sum()
    rms_map = {}

    precision = conf_mat.diagonal() / conf_mat.sum(
        axis=0
    )  # precision = TP / (TP + FP), diagonal divided by column sum
    recall = conf_mat.diagonal() / conf_mat.sum(
        axis=1
    )  # recall = TP / (TP + FN), diagonal divided by row sum

    for i in range(len(conf_mat)):
        if precision[i] < threshold or recall[i] < threshold:
            row_max = conf_mat[i, :].max()
            col_max = conf_mat[:, i].max()

            if row_max > col_max:  # Merge case - row maximum dominates
                j = np.argmax(conf_mat[i, :])
                rms_map[i] = (j, "rmerge")
            else:  # Split case - column maximum dominates
                j = np.argmax(conf_mat[:, i])
                rms_map[i] = (j, "rsplit")

    return rms_map


def _make_rms_p(rms_map: dict, n: int) -> np.ndarray:
    """
    Constructs a permutation matrix based on merge/split relationships.

    Creates an n x n permutation matrix that reorders indices according to the
    merge/split relationships defined in rms_map. The permutation follows an
    insertion sort-like process where index j is inserted after index i for each
    mapping in rms_map.

    For example, given rms_map = {2: (0, 'rmerge')}, the permutation would be:
    [0,1,2,3] -> [0,2,1,3]

    The resulting permutation matrix P can be used to permute a confusion matrix C:
    C^P = P @ C @ P^T

    Parameters
    ----------
    rms_map : dict
        Dictionary defining merge/split relationships where:
        - Keys (j) are indices to be moved
        - Values are tuples (i, perm_type) where:
            - i is the target index to insert after
            - perm_type is either 'rmerge' or 'rsplit' (unused)
    n : int
        Size of the square permutation matrix

    Returns
    -------
    np.ndarray
        An n x n permutation matrix as integer numpy array

    Examples
    --------
    >>> rms_map = {2: (0, 'rmerge')}
    >>> P = make_p(rms_map, 4)
    >>> print(P)
    [[1 0 0 0]
     [0 0 1 0]
     [0 1 0 0]
     [0 0 0 1]]
    """
    # Create initial ordered indices
    indices = list(range(n))

    # Perform insertions according to rms_map
    for j, (i, _) in rms_map.items():
        j_idx = indices.index(j)
        i_idx = indices.index(i)
        # Remove j and insert after i
        moved_value = indices.pop(j_idx)
        indices.insert(i_idx, moved_value)

    # Construct permutation matrix by reordering identity matrix
    P = np.eye(n, dtype=np.int_)
    P = P[indices]

    return P


def _permute_matrix(matrix: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Permute a matrix using a permutation matrix P.

    Performs the similarity transformation: P @ matrix @ P.T

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix to be permuted
    P : np.ndarray
        Permutation matrix of same size as input matrix

    Returns
    -------
    np.ndarray
        The permuted matrix P @ matrix @ P.T
    """
    return P @ matrix @ P.T


def _permute_labels(labels: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Permute labels using a permutation matrix P.

    Parameters
    ----------
    labels : np.ndarray
        Array of labels to be permuted
    P : np.ndarray
        Permutation matrix to apply to labels

    Returns
    -------
    np.ndarray
        The permuted labels array
    """
    # Convert permutation matrix to indices
    perm_indices = np.where(P)[1]
    return labels[perm_indices]


def _make_rms_label_map(labels: np.ndarray, rms_map: dict) -> dict:
    """
    Convert an RMS map using numeric indices to one using class labels.

    Parameters
    ----------
    labels : np.ndarray
        Array of class labels corresponding to matrix indices
    rms_map : dict
        Dictionary mapping source indices to (target index, merge type) tuples

    Returns
    -------
    dict
        Dictionary mapping source labels to (target label, merge type) tuples

    Examples
    --------
    >>> labels = np.array(['A', 'B', 'C'])
    >>> rms_map = {2: (0, 'rmerge')}
    >>> make_rms_label_map(labels, rms_map)
    {'C': ('A', 'rmerge')}
    """
    rms_label_map = {}
    for j, (i, merge_type) in rms_map.items():
        rms_label_map[labels[j]] = (labels[i], merge_type)

    return rms_label_map


###############################################################################
#                                                                             #
#                         RMS Permutation                                     #
#                                                                             #
###############################################################################


def _compute_rms_map_matrix(
    rms_label_map: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Build matrix representation of merge/split relationships."""
    sorted_rels = sorted(
        rms_label_map.items(),
        key=lambda x: (x[1][1], x[1][0]),
    )

    n = len(sorted_rels)
    rms_matrix = np.zeros((n, 4), dtype=object)
    rel_types = np.array(["" for _ in range(n)], dtype=object)

    for i, (pred_label, (gt_label, rel_type)) in enumerate(sorted_rels):
        rel_types[i] = rel_type
        if rel_type == "rmerge":
            rms_matrix[i] = [pred_label, gt_label, gt_label, gt_label]
        else:
            rms_matrix[i] = [gt_label, gt_label, gt_label, pred_label]

    return rms_matrix, rel_types


class RMSResult(NamedTuple):
    """Result of RMS permutation analysis."""

    permuted_matrix: np.ndarray
    permuted_labels: np.ndarray
    rms_label_map: dict
    rms_map_matrix: np.ndarray
    rms_map_type: np.ndarray


def rms_permute(confusion_matrix: np.ndarray, labels: np.ndarray) -> RMSResult:
    """`rms_permute(confusion_matrix, labels)`

    Perform reverse merge/split (RMS) permutation analysis
    on a confusion matrix.

    Analyzes and permutes a confusion matrix to identify merge
    and split relationships between predicted and ground truth
    labels.

    Parameters
    ----------
    confusion_matrix : numpy.ndarray
        The confusion matrix to analyze, shape (n_classes, n_classes)
    labels : numpy.ndarray
        The labels corresponding to the confusion matrix
        rows/columns, shape (n_classes,)

    Returns
    -------
    RMSResult
        Named tuple with fields:
        - permuted_matrix: reordered confusion matrix
        - permuted_labels: reordered labels
        - rms_label_map: dict of merge/split relationships
        - rms_map_matrix: Nx4 matrix of relationships
        - rms_map_type: array of 'rmerge' or 'rsplit'

    Examples
    --------
    >>> import numpy as np
    >>> conf_mat = np.array([[2, 1, 0], [0, 3, 1], [1, 0, 2]])
    >>> labels = np.array([1, 2, 3])
    >>> result = rms_permute(conf_mat, labels)
    >>> result.permuted_matrix  # or unpack as tuple
    """
    rms_map = _make_rms_map(confusion_matrix)
    n = len(confusion_matrix)
    P = _make_rms_p(rms_map, n)
    rms_label_map = _make_rms_label_map(labels, rms_map)

    permuted_matrix = _permute_matrix(confusion_matrix, P)
    permuted_labels = _permute_labels(labels, P)
    rms_map_matrix, rms_map_type = _compute_rms_map_matrix(rms_label_map)

    return RMSResult(
        permuted_matrix,
        permuted_labels,
        rms_label_map,
        rms_map_matrix,
        rms_map_type,
    )
