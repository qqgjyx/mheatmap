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

########################################################################################################################

def spectral_permute(conf_mat, labels):
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