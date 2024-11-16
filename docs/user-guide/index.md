---
hide:
  - navigation
---

# User Guide

## Introduction

mheatmap is a Python package for advanced heatmap
visualization and matrix analysis.

It provides tools for:

- Creating proportional/mosaic heatmaps
- AMC post-processing of confusion matrices
- Spectral permutation analysis
- RMS (Reverse Merge/Split) permutation analysis

## Basic Usage

### Mosaic Heatmap

The mosaic heatmap is a novel visualization
where cell sizes are proportional to their row and column sums:

```python linenums="1" hl_lines="10 11 12 13"
import numpy as np
from mheatmap import mosaic_heatmap

conf_mat = np.array([
    [85, 10,  5],
    [15, 70, 15],
    [ 5, 20, 75]
])

mosaic_heatmap(conf_mat, 
               annot=True,    # (1)!
               fmt='d',       # (2)!
               cmap='YlOrRd') # (3)!
```

1. Show values in cells
2. Format as integers
3. Use Yellow-Orange-Red colormap

### AMC Post-processing

AMC (Align-Mask-Confusion) is a technique for post-processing confusion matrices
or to say pre-processing before mosaic heatmap visualization:

$$
C_{masked} = C_{raw} \odot M, \quad
\text{where } \odot \text{ is the element-wise multiplication}
$$

$$
\Pi = \underset{\pi}{\arg\min} \sum_{i,j} C_{masked}[i,\Pi(j)], \quad
\text{where } \Pi \text{ is a permutation matrix}
$$

$$
\tilde{y} = \Pi \cdot \hat{y}
$$

$$
\tilde{C} = \Pi \cdot C_{masked}
$$

$$
\tilde{l} = \Pi \cdot l
$$

```python linenums="1" hl_lines="3"
from mheatmap import amc_postprocess

aligned_pred, processed_mat, processed_labels = amc_postprocess(conf_mat, labels)
```

### Spectral Reordering

Reorder matrices based on spectral analysis:

Note: see `examples` for more details.

```python linenums="1" hl_lines="3"
from mheatmap import spectral_permute

reordered_mat, reordered_labels = spectral_permute(conf_mat, labels)
```

### RMS Permutation Analysis

RMS (Reverse Merge/Split) analysis helps identify
merge and split relationships in matrices:

Note: see `examples` for more details.

```python linenums="1" hl_lines="3"
from mheatmap import rms_permute

reordered_mat, reordered_labels, rms_mapping = rms_permute(conf_mat, labels)
```

## Tips and Tricks

### Memory Efficiency

For large matrices:

```python linenums="1" hl_lines="3"
from scipy import sparse
sparse_mat = sparse.csr_matrix(large_matrix) # (1)!
mosaic_heatmap(sparse_mat, ...)
```

1. Use sparse matrices for efficiency

### Performance Tips

1. Use `numpy` arrays instead of lists
2. Pre-compute reusable values
3. Use sparse matrices for large, sparse data
4. Consider downsampling for visualization of very large matrices

## References

1. [Confusion Matrix Visualization Paper](https://arxiv.org/abs/xxx)
2. [AMC Post-processing Algorithm](https://arxiv.org/abs/zzz)
3. [Spectral Reordering](https://arxiv.org/abs/www)
4. [RMS Permutation Analysis](https://arxiv.org/abs/www)
