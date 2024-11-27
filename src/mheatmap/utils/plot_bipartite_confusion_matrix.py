"""Create a visually enhanced bipartite graph visualization from confusion matrix."""

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import seaborn as sns

from ._base import test_decorator


@test_decorator
def plot_bipartite_confusion_matrix(reordered_cm, reordered_labels, epsilon=1):
    """`plot_bipartite_confusion_matrix(reordered_cm, reordered_labels, epsilon=1)`

    Plot an enhanced bipartite graph visualization of a confusion matrix.

    Creates a visually appealing bipartite graph visualization where ground truth classes
    are represented as nodes on the left side and predicted classes as nodes on the right
    side. Edge weights represent confusion matrix values, with thicker edges indicating
    stronger connections. Node sizes are scaled based on class frequencies.

    Parameters
    ----------
    reordered_cm : numpy.ndarray
        The reordered confusion matrix after spectral ordering
    reordered_labels : numpy.ndarray
        The reordered class labels corresponding to the matrix rows/columns
    epsilon : float, default=1
        Minimum percentage threshold for displaying edges

    Returns
    -------
    None
        The plot is displayed using matplotlib

    Notes
    -----
    - Ground truth nodes (left) and predicted nodes (right) use distinct color schemes
    - Edge thicknesses and transparencies scale with confusion matrix values
    - Node sizes reflect the total frequency of each class
    - Custom color palette and styling for enhanced readability
    - Interactive plot with adjustable figure size
    - Edge labels show percentage of total predictions

    .. versionadded:: 1.1.0
    """
    # Configure plot style
    plt.style.use("seaborn-v0_8")
    plt.rcParams["figure.facecolor"] = "#f0f0f0"

    total_weight = np.sum(reordered_cm)
    n_classes = len(reordered_labels)

    # Initialize bipartite graph
    graph = nx.Graph()

    # Create node identifiers
    gt_nodes = [f"GT_{i}" for i in range(n_classes)]
    pred_nodes = [f"Pred_{i}" for i in range(n_classes)]

    # Add nodes with bipartite attributes
    graph.add_nodes_from(gt_nodes, bipartite=0)
    graph.add_nodes_from(pred_nodes, bipartite=1)

    # Add weighted edges and collect edge properties
    edge_weights = []
    edge_percentages = []
    edges = []
    for i in range(n_classes):
        for j in range(n_classes):
            weight = reordered_cm[i, j]
            percentage = (weight / total_weight) * 100
            if percentage > epsilon:
                graph.add_edge(f"GT_{i}", f"Pred_{j}", weight=weight)
                edge_weights.append(weight)
                edge_percentages.append(percentage)
                edges.append((f"GT_{i}", f"Pred_{j}"))

    # Calculate node sizes proportional to class frequencies
    gt_sizes = np.sum(reordered_cm, axis=1)
    pred_sizes = np.sum(reordered_cm, axis=0)
    node_sizes = {
        **{f"GT_{i}": 2000 * (s / np.max(gt_sizes)) for i, s in enumerate(gt_sizes)},
        **{
            f"Pred_{i}": 2000 * (s / np.max(pred_sizes))
            for i, s in enumerate(pred_sizes)
        },
    }

    # Configure plot layout
    plt.figure(figsize=(15, 10))
    plt.title(
        "Confusion Matrix Bipartite Graph", fontsize=16, pad=20, fontweight="bold"
    )
    plt.axis("off")

    # Create bipartite layout
    pos = {}
    y_coords = np.linspace(1, -1, n_classes)
    for i, y in enumerate(y_coords):
        pos[gt_nodes[i]] = [-1.2, y]  # Left side
        pos[pred_nodes[i]] = [1.2, y]  # Right side

    # Draw ground truth nodes
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=gt_nodes,
        node_color=sns.color_palette("Blues", n_colors=1),
        node_size=[node_sizes[node] for node in gt_nodes],
        edgecolors="white",
        linewidths=2,
    )

    # Draw prediction nodes
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=pred_nodes,
        node_color=sns.color_palette("Oranges", n_colors=1),
        node_size=[node_sizes[node] for node in pred_nodes],
        edgecolors="white",
        linewidths=2,
    )

    # Configure edge styling
    max_weight = max(edge_weights)
    edge_colors = [plt.cm.viridis(w / max_weight) for w in edge_weights]
    edge_alphas = [0.4 + 0.6 * (w / max_weight) for w in edge_weights]
    edge_widths = [1 + 10 * (w / max_weight) for w in edge_weights]

    # Draw edges
    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=edge_alphas,
        edge_cmap=plt.cm.viridis,
    )

    # Add edge percentage labels
    edge_labels = {edges[i]: f"{edge_percentages[i]:.1f}%" for i in range(len(edges))}
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=edge_labels, font_size=12, font_weight="bold"
    )

    # Add node labels
    labels = {}
    for i in range(n_classes):
        labels[gt_nodes[i]] = str(reordered_labels[i])
        labels[pred_nodes[i]] = str(reordered_labels[i])
    nx.draw_networkx_labels(graph, pos, labels, font_size=14)

    plt.tight_layout()
    plt.show()
