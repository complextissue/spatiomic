from math import ceil, sqrt
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import NDArray


def cluster_location(
    image: NDArray,
    title: str = "Isolated Clusters",
    cluster_title: str = "Cluster",
    cluster_labels: Optional[List[Union[str, int]]] = None,
    figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
) -> plt.Figure:
    """Plot the location of each cluster in a labelled image individually.

    Args:
        image (NDArray): The clustered image.
        title (str, optional): Title for the figure. Defaults to "Isolated Clusters".
        cluster_title (str, optional): Text to be preprended to the cluster number. Defaults to "Cluster".
        cluster_labels (Optional[List[Union[str, int]]], optional): Custom names for the clusters. Defaults to None.
        figsize (Optional[Tuple[Union[int, float], Union[int, float]]], optional): The size of the figure.
            Defaults to None.

    Returns:
        plt.Figure: The figure.
    """
    image_shape = image.shape
    image = image.ravel()

    clusters = np.unique(image)
    column_count = ceil(sqrt(len(clusters)))
    row_count = ceil(sqrt(len(clusters)))

    assert cluster_labels is None or len(cluster_labels) >= np.max(clusters), (
        "When labels are provided, the label count has to be equal or greater than the maximum cluster number."
    )

    sns.set_theme(style="white", font="Arial")
    sns.set_context("paper")

    fig = plt.figure(figsize=figsize if figsize is not None else (column_count * 2, row_count * 2))

    plt.title(title, y=1.02)
    plt.xticks([])
    plt.yticks([])

    # check if we really need to ceil both counts
    if (len(clusters) / column_count) <= (row_count - 1):
        row_count = row_count - 1

    for idx, cluster in enumerate(clusters):
        fig.add_subplot(row_count, column_count, idx + 1)

        plt.xticks([])
        plt.yticks([])

        if cluster_labels is None:
            plt.title(str(cluster_title) + " " + str(cluster))
        else:
            plt.title(str(cluster_title) + " " + str(cluster_labels[cluster]))

        plt.imshow(
            (image == cluster).reshape(image_shape),
            cmap=sns.color_palette("Blues", as_cmap=True),
        )

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    return fig
