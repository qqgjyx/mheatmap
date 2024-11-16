"""
Advanced features demonstration for mheatmap package

This script demonstrates advanced features of mheatmap including:
- AMC (Align, Mask, Confusion) post-processing for confusion matrices
- Spectral reordering for matrix visualization optimization  
- RMS (Reverse Merge/Split) permutation analysis

The example uses real hyperspectral image data (Salinas dataset) to show how these
features can help analyze and visualize complex classification results.

Key features demonstrated:
- Loading and preprocessing classification results
- AMC post-processing to align predicted and true labels
- Spectral reordering to reveal matrix structure
- RMS permutation to analyze merge/split patterns
- Visualization comparisons using mosaic heatmaps
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(
    str(Path(__file__).parent.parent)
)  # Add parent directory to Python path

import numpy as np
from src.mheatmap import AMCPostprocess, RMSPermute, spectral_permute, mosaic_heatmap
import matplotlib.pyplot as plt
import scipy.io

# Load sample data from Salinas hyperspectral dataset
current_dir = Path(__file__).parent

# Load ground truth labels
y_true = scipy.io.loadmat(current_dir / "data" / "Salinas_gt.mat")["salinas_gt"].reshape(-1)
print(f"y_true shape: {y_true.shape}")

# Load predicted labels from spectral clustering
y_pred = np.array(
    pd.read_csv(
        current_dir / "data" / "Salinas_spectralclustering.csv",
        header=None,
        low_memory=False,
    )
    .values[1:]
    .flatten()
)
print(f"y_pred shape: {len(y_pred)}")

# Demonstrate AMC post-processing
amc = AMCPostprocess(y_pred, y_true)
conf_mat = amc.get_C()  # Get confusion matrix
labels = amc.get_C_labels()  # Get aligned labels

# Demonstrate spectral reordering
reordered_mat, reordered_labels = spectral_permute(conf_mat, labels)

# Visualize original vs spectrally reordered matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

mosaic_heatmap(conf_mat, ax=ax1, xticklabels=labels, yticklabels=labels, cmap="YlGnBu")
ax1.set_title("Original", fontsize=18)
ax1.xaxis.set_ticks_position('top')

mosaic_heatmap(
    reordered_mat,
    ax=ax2,
    xticklabels=reordered_labels,
    yticklabels=reordered_labels,
    cmap="YlGnBu",
)
ax2.set_title("Spectral Reordered", fontsize=18)
ax2.xaxis.set_ticks_position('top')

plt.tight_layout()
plt.savefig(current_dir / "images" / "spectral_reordering.png")
plt.close()

# Demonstrate RMS permutation analysis
rms = RMSPermute(conf_mat, labels)
rms_C = rms.get_permuted_matrix()
rms_labels = rms.get_permuted_labels()

# Visualize original vs RMS permuted matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

mosaic_heatmap(conf_mat, ax=ax1, xticklabels=labels, yticklabels=labels, cmap="YlGnBu")
ax1.set_title("Original", fontsize=18)

mosaic_heatmap(
    rms_C, ax=ax2, xticklabels=rms_labels, yticklabels=rms_labels, cmap="YlGnBu"
)
ax2.set_title("RMS Permuted", fontsize=18)

plt.tight_layout()
plt.savefig(current_dir / "images" / "rms_permutation.png")
plt.close()
