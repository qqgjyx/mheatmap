"""
Basic usage examples for mheatmap package
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to Python path

import numpy as np
from src.mheatmap import mosaic_heatmap

# Create sample confusion matrix
conf_mat = np.array([
    [85, 10,  5, 0],
    [0, 50, 30, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 100]
])

# Basic mosaic heatmap
mosaic_heatmap(conf_mat, 
               annot=False,
               fmt='d',
               cmap='YlGnBu')

# Save the plot
import matplotlib.pyplot as plt

current_dir = Path(__file__).parent
plt.savefig(current_dir / 'basic_mosaic_heatmap.png')
plt.close() 