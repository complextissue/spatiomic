from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray

from ._colormap import colormap as create_colormap


def cluster_scatter(
    data: NDArray,
    clusters: Union[NDArray, List],
    title_x: str = "Dimension 1",
    title_y: str = "Dimension 2",
    figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
    colormap: Optional[ListedColormap] = None,
) -> plt.Figure:
    """Scatter the points and color them according to the clusters.

    Args:
        data (NDArray): The data to be scattered.
        clusters (Union[NDArray, List]): The clusters for the data.
        title_x (str, optional): The title for the x-axis. Defaults to "Dimension 1".
        title_y (str, optional): The title for the y-axis. Defaults to "Dimension 2".
        colormap (Optional[ListedColormap], optional): The colormap to use. Defaults to None.

    Returns:
        plt.Figure: The figure.
    """
    assert data.shape[1] == 2, "The data must have exactly 2 dimensions."

    cluster_count = (
        np.max(np.unique(np.array(clusters))) + 1
        if all(isinstance(i, int) for i in clusters)
        else len(np.unique(clusters))
    )
    cmap = create_colormap(color_count=cluster_count) if colormap is None else colormap

    sns.set_theme(style="white", font="Arial")
    sns.set_context("paper")

    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot()

    ax1.set_xticks([])
    ax1.set_yticks([])

    ax1.set_xlabel(title_x)
    ax1.set_ylabel(title_y)

    sns.scatterplot(
        x=data[:, 0],
        y=data[:, 1],
        hue=clusters,
        palette=cmap.colors,
        legend=False,
    )

    # draw the colorbar as an image
    divider = make_axes_locatable(ax1)

    ax2 = divider.append_axes("right", size="5%", pad=0.05)
    ax2.imshow(np.expand_dims(np.arange(0, cluster_count), axis=1), cmap=cmap)

    ax2.set_yticks(np.arange(0, cluster_count))
    ax2.set_yticklabels(
        np.arange(0, cluster_count) if all(isinstance(i, int) for i in clusters) else np.unique(clusters)
    )
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    ax2.set_xticks([])

    sns.despine(ax=ax1)
    sns.despine(ax=ax2, left=True, bottom=True)

    plt.tight_layout()

    return fig
