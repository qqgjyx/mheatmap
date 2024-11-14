"""
Demonstration of advanced features in mheatmap
"""
import sys
from pathlib import Path

import pandas as pd
sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to Python path

import numpy as np
from src.mheatmap import (
    AMCPostprocess,
    RMSPermute,
    spectral_permute,
    mosaic_heatmap
)
import matplotlib.pyplot as plt
import scipy.io
# Create sample data

# get the current directory
current_dir = Path(__file__).parent

y_true = scipy.io.loadmat(current_dir / 'Salinas_gt.mat')['salinas_gt'].reshape(-1)
print(f"y_true shape: {y_true.shape}")
y_pred = np.array(pd.read_csv(current_dir / 'Salinas_spectralclustering.csv', header=None, low_memory=False).values[1:].flatten())
print(f"y_pred shape: {len(y_pred)}")

# AMC post-processing example
amc = AMCPostprocess(y_pred, y_true)
conf_mat = amc.get_C()
labels = amc.get_C_labels()

# Spectral reordering example
reordered_mat, reordered_labels = spectral_permute(conf_mat, labels)

# Plot original vs reordered
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

mosaic_heatmap(conf_mat,
               ax=ax1,
               xticklabels=labels,
               yticklabels=labels,
               cmap='YlGnBu')

mosaic_heatmap(reordered_mat,
               ax=ax2,
               xticklabels=reordered_labels,
               yticklabels=reordered_labels,
               cmap='YlGnBu')

plt.tight_layout()
plt.savefig(current_dir / 'spectral_reordering.png')
plt.close()


rms = RMSPermute(conf_mat, labels)

rms_C = rms.get_permuted_matrix()
rms_labels = rms.get_permuted_labels()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

mosaic_heatmap(conf_mat,
               ax=ax1,
               xticklabels=labels,
               yticklabels=labels,
               cmap='YlGnBu')

mosaic_heatmap(rms_C,
               ax=ax2,
               xticklabels=rms_labels,
               yticklabels=rms_labels,
               cmap='YlGnBu')

plt.tight_layout()
plt.savefig(current_dir / 'rms_permutation.png')
plt.close() 