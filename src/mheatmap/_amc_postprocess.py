"""AMC (Align, Mask, Confusion) Matrix Post-processing.

This module provides functionality for post-processing confusion matrices using the AMC
(Align, Mask, Confusion) approach. It includes tools for aligning predicted and ground
truth labels, masking unlabeled points, and computing optimized confusion matrices.

See Also
--------
mheatmap.rms_permute : Related module for matrix permutation analysis

mheatmap.spectral_permute : Related module for spectral reordering

Notes
-----
.. versionadded:: 0.1
"""

# Copyright (c) 2024 Juntang Wang
# All rights reserved.
# Licensed under the MIT License.

import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


###############################################################################
#                                                                             #
#                         Helper Functions                                    #
#                                                                             #
###############################################################################
def mask_zeros_from_gt(labels, gt, mode="labels"):
    """`mask_zeros_from_gt(labels, gt, mode="labels")`

    Mask unlabeled points (zeros) from ground truth labels.

    This function handles both 1D and 2D label arrays by masking points that correspond
    to zeros in the ground truth labels. For image mode, it preserves the 2D spatial
    structure of the input.

    Parameters
    ----------
    labels : numpy.ndarray
        Labels to be masked. Can be either:
        - 1D array of shape (n_samples,)
        - 2D array of shape (height, width) for image mode
    gt : numpy.ndarray
        Ground truth labels of shape (n_samples,) containing zeros for unlabeled points
    mode : {'labels', 'image'}, default='labels'
        Operating mode:
        - 'labels': Returns 1D masked array
        - 'image': Returns 2D masked array preserving spatial structure

    Returns
    -------
    numpy.ma.MaskedArray
        Masked array where unlabeled points (zeros in ground truth) are masked.
        Shape matches input labels.

    Raises
    ------
    ValueError
        If labels dimensions don't match the mode or ground truth shape

    Examples
    --------
    >>> import numpy as np
    >>> # Example with 1D labels
    >>> labels = np.array([1, 2, 3, 4, 5])
    >>> gt = np.array([1, 0, 3, 0, 5])
    >>> masked = mask_zeros_from_gt(labels, gt)
    >>> print(masked)
    [1 3 5]

    >>> # Example with 2D image labels
    >>> img_labels = np.array([[1, 2], [3, 4]])
    >>> img_gt = np.array([[1, 0], [3, 0]])
    >>> masked_img = mask_zeros_from_gt(img_labels, img_gt, mode='image')
    >>> print(masked_img)
    [[1 --]
     [3 --]]
    """
    if mode == "image":
        if len(labels.shape) != 2:
            raise ValueError("Labels must be 2D array for image mode")

        gt_image = gt.reshape(labels.shape) if labels.shape != gt.shape else gt
        mask = gt_image != 0
        return np.ma.masked_where(~mask, labels)

    elif mode == "labels":
        if labels.shape != gt.shape:
            labels = labels.reshape(-1)
            if labels.shape != gt.shape:
                raise ValueError(
                    "Labels shape must match ground truth after flattening"
                )

        mask = gt != 0
        return labels[mask]

    else:
        raise ValueError("Mode must be either 'labels' or 'image'")


def _align_labels(y_true, y_pred, mask_zeros=True):
    """Align predicted cluster labels with ground truth labels using sklearn's metrics.

    Use Jonker-Volgenant algorithm to find the optimal assignment.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels (0 indicates unlabeled points)
    y_pred : array-like
        Predicted cluster labels

    Returns
    -------
    aligned_pred : array-like
        Aligned predicted labels
    conf_mat : array-like
        Confusion matrix of filtered labels
    """
    if mask_zeros:
        # Filter out zeros from both arrays
        y_true_filtered = mask_zeros_from_gt(y_true, y_true)
        y_pred_filtered = mask_zeros_from_gt(y_pred, y_true)
    else:
        y_true_filtered = y_true
        y_pred_filtered = y_pred

    # get the labels as union of the unique values
    # of y_true_filtered and y_pred_filtered
    labels = np.union1d(np.unique(y_true_filtered), np.unique(y_pred_filtered))

    # Get confusion matrix using jonker-volgenant algorithm
    conf_mat = confusion_matrix(y_true_filtered, y_pred_filtered, labels=labels)

    # Find optimal assignment using Jonker-Volgenant algorithm
    row_ind, col_ind = linear_sum_assignment(-conf_mat)

    mapping = {}

    for row_i, col_i in zip(row_ind, col_ind):
        mapping[labels[col_i]] = labels[row_i]

    aligned_pred = np.array([mapping.get(p, p) for p in y_pred])
    return aligned_pred


###############################################################################
#                                                                             #
#                         AMC Post-processing                                 #
#                                                                             #
###############################################################################
class _AMCPostprocess:
    """Post-processing class for Automatic Model Calibration (AMC) Align Mask Confusion.

    A comprehensive post-processing pipeline for model predictions that handles:
    - Label alignment between predictions and ground truth
    - Zero-value masking for unlabeled data points
    - Confusion matrix generation and spectral reordering

    The class implements a systematic workflow to prepare model outputs for evaluation
    and analysis, ensuring proper handling of unlabeled data and optimal label matching.

    Parameters
    ----------
    pred_ : np.ndarray
        Raw model prediction labels, shape (n_samples,)
    gt : np.ndarray
        Ground truth labels, shape (n_samples,)

    Attributes
    ----------
    pred_ : np.ndarray
        Model predictions shifted by +1 to ensure positive label values
    gt : np.ndarray
        Original ground truth labels preserved for reference

    aligned_pred_ : np.ndarray
        Predictions optimally aligned to ground truth via linear assignment

    masked_pred : np.ndarray
        Aligned predictions with unlabeled points (zeros) masked
    masked_gt : np.ndarray
        Ground truth with unlabeled points masked

    conf_mat_labels : np.ndarray
        Union of unique labels from masked ground truth and predictions
    conf_mat : np.ndarray
        Confusion matrix computed from masked and aligned predictions
    Notes
    -----
    The class assumes zero values represent unlabeled data points. The prediction
    labels are automatically shifted by +1 during initialization to avoid conflicts
    with the zero-value convention for unlabeled points.
    """

    def __init__(self, pred_: np.ndarray, gt: np.ndarray) -> None:
        # Initialize with shifted predictions to reserve 0 for unlabeled points
        self.pred_ = pred_ + 1
        self.gt = gt.copy()  # Store original ground truth

        # Perform optimal label alignment
        self.aligned_pred_ = _align_labels(self.gt, self.pred_)

        # Apply zero-value masking
        self.masked_pred = mask_zeros_from_gt(self.aligned_pred_, self.gt)
        self.masked_gt = mask_zeros_from_gt(self.gt, self.gt)

        # Compute confusion matrix with unified label set
        self.conf_mat_labels = np.union1d(
            np.unique(self.masked_gt), np.unique(self.masked_pred)
        )
        self.conf_mat = confusion_matrix(self.masked_gt, self.masked_pred)

    def get_gt(self):
        return self.gt

    def get_aligned_pred(self):
        return self.aligned_pred_

    def get_C(self):
        return self.conf_mat

    def get_C_labels(self):
        return self.conf_mat_labels


def amc_postprocess(
    pred_: np.ndarray, gt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`amc_postprocess(pred_, gt)`

    Post-process predictions by aligning labels and computing confusion matrix.
    This function performs Align-Mask-Confusion (AMC) post-processing on predicted labels
    to optimally align them with ground truth labels and compute the confusion matrix.

    Parameters
    ----------
    pred_ : np.ndarray
        Array of predicted labels
    gt : np.ndarray
        Array of ground truth labels

    Returns
    -------
    aligned_pred : np.ndarray
        Predictions with labels optimally aligned to ground truth

    conf_mat : np.ndarray
        Confusion matrix computed from aligned predictions

    conf_mat_labels : np.ndarray
        Union of unique labels from ground truth and predictions

    Notes
    -----
    The function assumes zero values represent unlabeled data points. Prediction
    labels are automatically shifted by +1 during processing to avoid conflicts
    with the zero-value convention for unlabeled points.

    Examples
    --------
    >>> import numpy as np
    >>> # Example with 3 classes and some unlabeled points (zeros)
    >>> gt = np.array([1, 2, 0, 3, 1, 2])
    >>> pred = np.array([2, 3, 0, 1, 2, 3])
    >>> aligned_pred, conf_mat, labels = amc_postprocess(pred, gt)
    >>> print("Aligned predictions:", aligned_pred)
    Aligned predictions: [1 2 0 3 1 2]
    >>> print("Confusion matrix:")
    >>> print(conf_mat)
    [[2 0 0]
     [0 2 0]
     [0 0 1]]
    >>> print("Labels:", labels)
    Labels: [1 2 3]
    """
    amc = _AMCPostprocess(pred_, gt)
    pr = amc.get_aligned_pred()
    C = amc.get_C()
    C_labels = amc.get_C_labels()
    return pr, C, C_labels
