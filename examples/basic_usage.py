"""
Basic usage examples for mheatmap package

This script demonstrates the basic usage of mheatmap by comparing a traditional
heatmap visualization with mheatmap's mosaic heatmap visualization.

The example uses a sample 4x4 confusion matrix to show how mosaic heatmaps can
better represent the proportional relationships between matrix values compared
to standard heatmaps.

Key features demonstrated:
- Basic heatmap plotting with seaborn
- Mosaic heatmap plotting with mheatmap
- Side-by-side comparison of visualization approaches
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(
    str(Path(__file__).parent.parent)
)  # Add parent directory to Python path

import numpy as np
import seaborn as sns
from src.mheatmap import mosaic_heatmap

# Create a sample 4x4 confusion matrix with some interesting patterns
conf_mat = np.array([
    [85, 10, 5,   0],  # High accuracy for class 0 
    [0,  50, 30,  0],  # Some confusion between classes 1 and 2
    [0,   0,  0,  0],  # Empty row representing unused class
    [0,   0,  0, 100]  # Perfect classification for class 3
])

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot traditional heatmap using seaborn
sns.heatmap(conf_mat, annot=False, fmt="d", cmap="YlGnBu", ax=ax1)
ax1.set_title("Normal Heatmap", fontsize=18)
ax1.xaxis.set_ticks_position('top')

# Plot mosaic heatmap using mheatmap
mosaic_heatmap(conf_mat, annot=False, fmt="d", cmap="YlGnBu", ax=ax2)
ax2.set_title("Mosaic Heatmap", fontsize=18)
ax2.xaxis.set_ticks_position('top')

# Save the comparison figure
current_dir = Path(__file__).parent
plt.savefig(current_dir / "images" / "basic_mosaic_heatmap.png")
plt.close()
