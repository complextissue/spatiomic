from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray

if TYPE_CHECKING:
    from spatiomic.dimension import som


def som_clusters(
    som: "som",
    clusters: NDArray,
    title: str = "Self-Organizing Map Clusters",
    colormap: Optional[ListedColormap] = None,
) -> plt.Figure:
    """Plot a self-organizing map in 2D with corresponding clusters.

    Args:
        som (som): The self-organizing map.
        clusters (NDArray): The clusters.
        title (str, optional): The title for the plot. Defaults to "Self-Organizing Map Clusters".
        colormap (Optional[ListedColormap], optional): Custom colormap. Defaults to None.

    Returns:
        plt.Figure: The figure.
    """
    from . import colormap as create_colormap

    sns.set_theme(style="white", font="Arial")
    sns.set_context("paper")

    cluster_count = len(np.unique(clusters))

    cmap = create_colormap(color_count=cluster_count) if colormap is None else colormap

    # visualize the som map
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.set_title(title)

    ax1.imshow(
        np.array(clusters).reshape(som.node_count),
        cmap=cmap,
    )

    ax1.set_xticks([])
    ax1.set_yticks([])

    # draw the colorbar as an image
    divider = make_axes_locatable(ax1)

    ax2 = divider.append_axes("right", size="5%", pad=0.05)
    ax2.imshow(np.expand_dims(np.arange(0, cluster_count), axis=1), cmap=cmap)

    ax2.set_yticks(np.arange(0, cluster_count))
    ax2.set_yticklabels(np.arange(0, cluster_count))
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    ax2.set_xticks([])

    sns.despine(left=True, bottom=True)

    plt.tight_layout()

    return fig
