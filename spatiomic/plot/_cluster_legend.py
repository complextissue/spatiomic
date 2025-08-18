from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap

from ._colormap import colormap as create_colormap


def cluster_legend(
    cluster_count: int,
    cluster_labels: Optional[List[str]] = None,
    colormap: Optional[ListedColormap] = None,
    figsize: Optional[Union[Tuple[Union[int, float], Union[int, float]]]] = None,
    ax: Optional[plt.Axes] = None,
) -> Union[plt.Figure, plt.Axes]:
    """Plot a legend for the clusters.

    Args:
        cluster_count (int): The number of clusters.
        cluster_labels (Optional[List[str]], optional): The labels for the clusters. Defaults to None.
        colormap (ListedColormap, optional): The colormap to use. Defaults to None.
        figsize (Optional[Union[int, float, Tuple[Union[int, float], Union[int, float]]], optional): The figure size.
            Defaults to None.
        ax: Existing axes to plot on. If None, creates new figure. Defaults to None.

    Returns:
        plt.Figure if ax is None, otherwise the provided plt.Axes.
    """
    colormap = create_colormap(color_count=cluster_count, seed=0) if colormap is None else colormap

    # Setup plotting
    if ax is None:
        sns.set_theme(style="white", font="Arial")
        sns.set_context("paper")
        fig = plt.figure(figsize=figsize if figsize is not None else (cluster_count, 1), dpi=300)
        ax = fig.add_subplot()
        return_fig = True
    else:
        return_fig = False

    ax.imshow(np.expand_dims(np.arange(0, cluster_count), axis=0), cmap=colormap)

    ax.set_xticks(np.arange(0, cluster_count))

    if cluster_labels is not None:
        assert len(cluster_labels) == cluster_count, "The number of cluster labels must match the number of clusters."
    else:
        cluster_labels = [f"{i}" for i in range(cluster_count)]

    ax.set_xticklabels(cluster_labels, fontsize=12)
    ax.xaxis.tick_bottom()
    ax.xaxis.set_label_position("bottom")

    ax.set_yticks([])

    sns.despine(left=True, bottom=True)

    if return_fig:
        plt.tight_layout(pad=0.5)
        return fig
    else:
        return ax
