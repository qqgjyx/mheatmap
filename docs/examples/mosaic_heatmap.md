# Mosaic Heatmap
---

## Importing Packages


```python
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
from src.mheatmap import mosaic_heatmap
```

## Example


```python
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
ax1.set_title("Normal Heatmap", fontsize=18, color='#4A4A4A')  # Medium gray
ax1.xaxis.set_ticks_position('top')
ax1.tick_params(colors='#4A4A4A')

# Plot mosaic heatmap using mheatmap
mosaic_heatmap(conf_mat, annot=False, fmt="d", cmap="YlGnBu", ax=ax2)
ax2.set_title("Mosaic Heatmap", fontsize=18, color='#4A4A4A')  # Medium gray
ax2.xaxis.set_ticks_position('top')
ax2.tick_params(colors='#4A4A4A')

plt.show()
```


    
![png](images/basic_mosaic_heatmap.png)
    



```python

```
