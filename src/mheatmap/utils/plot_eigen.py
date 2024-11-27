"""Plot the eigenvalues and the eigenvectors"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from ._base import test_decorator


@test_decorator
def plot_eigen(eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> None:
    """Plot eigenvalues and eigenvectors of a matrix.

    Creates two visualization plots:
    1. Scatter plot of eigenvalues on log scale, with negative values in red and positive in blue
    2. Heatmap visualization of the eigenvector matrix

    Parameters
    ----------
    eigenvalues : np.ndarray
        1D array of eigenvalues
    eigenvectors : np.ndarray
        2D array where columns are eigenvectors

    Returns
    -------
    None
        Displays the plots using matplotlib
    """
    # Plot eigenvalues
    plt.figure(figsize=(8, 4))
    colors = ["red" if ev <= 0 else "blue" for ev in eigenvalues]
    plt.gca().set_yscale("log")
    plt.scatter(np.arange(len(eigenvalues)), np.abs(eigenvalues), c=colors, alpha=0.6)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue (log scale)")
    plt.title("Eigenvalue Spectrum")
    plt.show()

    # Plot eigenvectors as heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        eigenvectors,
        cmap="YlGnBu",
        xticklabels="auto",
        yticklabels="auto",
        cbar_kws={"label": "Component Value"},
    )
    plt.xlabel("Eigenvector Index")
    plt.ylabel("Component Index")
    plt.title("Eigenvector Components")
    plt.show()
