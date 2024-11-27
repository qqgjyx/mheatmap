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

import numpy as np


###############################################################################
#                                                                             #
#                         Helper Functions                                    #
#                                                                             #
###############################################################################


def _make_rms_map(conf_mat: np.ndarray, threshold: float = 0.37) -> dict:
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
        The permuted labels array P @ labels
    """
    return P @ labels


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
class _RMSPermute:
    def __init__(
        self, conf_mat: np.ndarray, labels: np.ndarray, threshold: float = 0.37
    ):
        """
        Initialize RMSPermute to analyze and permute confusion matrices.

        A wrapper class that provides functionality for analyzing confusion matrices
        to identify merge/split relationships between classes and perform permutations
        based on those relationships.

        Parameters
        ----------
        conf_mat : np.ndarray
            The confusion matrix to analyze
        labels : np.ndarray
            Array of class labels corresponding to matrix indices
        threshold : float, optional
            Threshold ratio for identifying negligible diagonal elements,
            by default 1e-4
        """
        self.conf_mat = conf_mat
        self.labels = labels
        self.threshold = threshold
        self.n = len(conf_mat)

        # Compute mappings and permutation matrix
        self.rms_map = _make_rms_map(conf_mat, threshold)
        self.P = _make_rms_p(self.rms_map, self.n)
        self.rms_label_map = _make_rms_label_map(labels, self.rms_map)

    def get_permuted_matrix(self) -> np.ndarray:
        """
        Get the permuted confusion matrix.

        Returns
        -------
        np.ndarray
            The permuted confusion matrix
        """
        return _permute_matrix(self.conf_mat, self.P)

    def get_permuted_labels(self) -> np.ndarray:
        """
        Get the permuted class labels.

        Returns
        -------
        np.ndarray
            The permuted labels array
        """
        return _permute_labels(self.labels, self.P)

    def get_rms_mapping(self) -> dict:
        """
        Get the merge/split relationships between classes.

        Returns
        -------
        dict
            Dictionary mapping source labels to (target label, merge type) tuples
        """
        return self.rms_label_map

    def get_rms_map_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a matrix representation of the merge/split relationships
        between classes.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
            1. An Nx4 matrix where N is the number of source labels:
               - Column 0: GT1
               - Column 1: GT2
               - Column 2: PRED1
               - Column 3: PRED2
               The matrix is sorted primarily by relationship type, then:
               - For rmerge: sorted by predicted labels
               - For rsplit: sorted by ground truth labels
            2. An array of relationship types ('rmerge' or 'rsplit')

        Notes
        -----
        Relationship types indicate:
        - 'rmerge': Multiple GT labels map to same PRED label
        (e.g., GT1,GT2 -> PRED1,PRED1)
        - 'rsplit': Single GT label maps to multiple PRED labels
        (e.g., GT1,GT1 -> PRED1,PRED2)
        """
        # Sort relationships by type first,
        # then by pred label for rmerge or gt label for rsplit
        sorted_relationships = sorted(
            self.rms_label_map.items(),
            key=lambda x: (
                x[1][1],  # Primary sort by type
                x[1][0],  # Secondary sort by gt
            ),
        )

        # Initialize output arrays
        n_relationships = len(sorted_relationships)
        rms_matrix = np.zeros((n_relationships, 4), dtype=object)
        relationship_types = np.array(
            ["" for _ in range(n_relationships)], dtype=object
        )

        # Fill arrays
        for i, (pred_label, (gt_label, rel_type)) in enumerate(sorted_relationships):
            relationship_types[i] = rel_type

            if rel_type == "rmerge":
                # GT1,GT2 -> PRED1,PRED1
                rms_matrix[i] = [pred_label, gt_label, gt_label, gt_label]
            else:  # rsplit
                # GT1,GT1 -> PRED1,PRED2
                rms_matrix[i] = [gt_label, gt_label, gt_label, pred_label]

        return np.array(rms_matrix, dtype=np.int_), relationship_types


def rms_permute(confusion_matrix: np.ndarray, labels: np.ndarray) -> tuple:
    """`rms_permute(confusion_matrix, labels)`

    Perform reverse merge/split (RMS) permutation analysis on a confusion matrix.

    This function analyzes and permutes a confusion matrix to identify merge and split
    relationships between predicted and ground truth labels. A merge relationship occurs
    when multiple ground truth labels map to the same predicted label, while a split
    relationship occurs when a single ground truth label maps to multiple predicted labels.

    Parameters
    ----------
    confusion_matrix : numpy.ndarray
        The confusion matrix to analyze, shape (n_classes, n_classes)
    labels : numpy.ndarray
        The labels corresponding to the confusion matrix rows/columns, shape (n_classes,)

    Returns
    -------
    permuted_matrix : numpy.ndarray
        The confusion matrix after permuting rows and columns to group related labels
    permuted_labels : numpy.ndarray
        The reordered labels corresponding to the permuted matrix
    rms_label_map : dict
        Dictionary mapping predicted labels to tuples of (ground truth label, relationship type)
    rms_map_matrix : numpy.ndarray
        Matrix representation of merge/split relationships between labels
    rms_map_type : numpy.ndarray
        Array of relationship types ('rmerge' or 'rsplit') for each relationship

    See Also
    --------
    mheatmap.amc_postprocess : Related module for confusion matrix post-processing
    mheatmap.spectral_permute : Alternative permutation method using spectral ordering

    Notes
    -----
    The function identifies two types of label relationships:
    - Merge (rmerge): Multiple ground truth labels map to same predicted label
    - Split (rsplit): Single ground truth label maps to multiple predicted labels

    Examples
    --------
    >>> import numpy as np
    >>> conf_mat = np.array([[2, 1, 0], [0, 3, 1], [1, 0, 2]])
    >>> labels = np.array([1, 2, 3])
    >>> perm_mat, perm_labs, rmap, rmat, rtypes = rms_permute(conf_mat, labels)
    """
    rms = _RMSPermute(confusion_matrix, labels)
    permuted_matrix = rms.get_permuted_matrix()
    permuted_labels = rms.get_permuted_labels()
    rms_label_map = rms.rms_label_map
    rms_map_matrix, rms_map_type = rms.get_rms_map_matrix()

    return permuted_matrix, permuted_labels, rms_label_map, rms_map_matrix, rms_map_type
