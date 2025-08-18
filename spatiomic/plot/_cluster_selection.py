from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray

from ._colormap import colormap as create_colormap


def cluster_selection(
    image: NDArray,
    clusters: Union[NDArray, List[int]],
    colormap: Optional[ListedColormap] = None,
    ax: Optional[plt.Axes] = None,
) -> Union[plt.Figure, plt.Axes]:
    """Plot the image with the clusters shown in colors and a colorbar.

    Args:
        image (NDArray): The clustered image.
        clusters (Union[NDArray, List[int]]): Which clusters to show. All others will be collapsed into a single black
            background cluster.
        colormap (Optional[ListedColormap], optional): The colormap to use. Defaults to None.
        ax: Existing axes to plot on. If None, creates new figure. Defaults to None.

    Returns:
        plt.Figure if ax is None, otherwise the provided plt.Axes.
    """
    img_shape = image.shape
    image = image.ravel()

    clusters = np.unique(clusters)
    cluster_count = np.max(np.unique(np.array(image))) + 1
    cluster_count_included = len(clusters)

    cmap_raw = create_colormap(color_count=cluster_count) if colormap is None else colormap

    colors = ["#000"]

    # create a blank image and set the background to -1
    image_combined = np.zeros((image.shape)) - 1
    colormap_raw = cmap_raw.colors

    assert len(colormap_raw) >= cluster_count, (  # type: ignore
        "When a colormap is provided, the color count has to be equal or greater than the cluster count."
    )

    for idx, cluster in enumerate(clusters):
        # add cluster color to colors for colormap
        colors.append(colormap_raw[cluster])  # type: ignore

        image_combined[image == cluster] = idx

    cmap = ListedColormap(colors)

    # Setup plotting
    if ax is None:
        sns.set_theme(style="white", font="Arial")
        sns.set_context("paper")
        fig = plt.figure()
        ax1 = fig.add_subplot()
        return_fig = True
        show_colorbar = True
    else:
        ax1 = ax
        return_fig = False
        show_colorbar = False

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(image_combined.reshape(img_shape), cmap=cmap)

    # draw the colorbar as an image only when creating new figure
    if show_colorbar:
        divider = make_axes_locatable(ax1)

        ax2 = divider.append_axes("right", size="5%", pad=0.05)
        ax2.imshow(np.expand_dims(np.arange(0, cluster_count_included + 1), axis=1), cmap=cmap)

        ax2.set_yticks(np.arange(0, cluster_count_included + 1))
        ax2.set_yticklabels(["BG", *clusters.tolist()])
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")

        ax2.set_xticks([])

        sns.despine(left=True, bottom=True, ax=ax2)

    sns.despine(left=True, bottom=True, ax=ax1)

    if return_fig:
        plt.tight_layout()
        return fig
    else:
        return ax1


def save_cluster_selection(
    image: NDArray,
    save_path: str,
    clusters: Union[NDArray, List[int]],
    colormap: Optional[ListedColormap] = None,
) -> None:
    """Save the image with the clusters shown in colors.

    Args:
        image (NDArray): The image of the clusters.
        save_path (str): The path to save the image to.
        clusters (Union[NDArray, List[int]]): The clusters to show.
        colormap (Optional[ListedColormap], optional): The colormap to use. Defaults to None.
    """
    img_shape = image.shape
    image = image.ravel()

    clusters = np.array(clusters)
    cluster_count = np.max(np.unique(np.array(clusters))) + 1
    cmap_raw = create_colormap(color_count=cluster_count) if colormap is None else colormap

    colors = ["#000"]

    # create a blank image and set the background to -1
    image_combined = np.zeros((image.shape)) - 1
    colormap_raw = cmap_raw.colors

    assert len(colormap_raw) >= cluster_count, (  # type: ignore
        "When a colormap is provided, the color count has to be equal or greater than the cluster count."
    )

    for idx, cluster in enumerate(clusters):
        # add cluster color to colors for colormap
        colors.append(colormap_raw[cluster])  # type: ignore

        image_combined[image == cluster] = idx

    cmap = ListedColormap(colors)

    plt.imsave(save_path, image_combined.reshape(img_shape), cmap=cmap)
