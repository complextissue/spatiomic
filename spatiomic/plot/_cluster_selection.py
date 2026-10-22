from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray

from ._colormap import colormap as create_colormap
from ._colormap import get_cluster_colors


def get_selection_image(
    image: NDArray,
    clusters: Union[NDArray, List[int]],
    colormap: Optional[ListedColormap] = None,
) -> tuple[NDArray, ListedColormap, NDArray]:
    """Map the selected clusters to 1 to n and all other pixels, including unassigned ones (-1), to 0.

    Args:
        image (NDArray): The clustered image.
        clusters (Union[NDArray, List[int]]): The clusters to show.
        colormap (Optional[ListedColormap], optional): The colormap with one color per cluster. Defaults to None.

    Raises:
        ValueError: If the colormap has fewer colors than the highest cluster label plus one.

    Returns:
        tuple[NDArray, ListedColormap, NDArray]: The selection image, the colormap with black for the background
            followed by the colors of the selected clusters, and the sorted selected clusters.
    """
    clusters = np.unique(clusters)
    cluster_count = int(max(np.max(image), np.max(clusters))) + 1
    colormap = create_colormap(color_count=cluster_count) if colormap is None else colormap

    if len(colormap.colors) < cluster_count:  # type: ignore
        raise ValueError(
            "When a colormap is provided, the color count has to be equal or greater than the cluster count."
        )

    image_selection = np.zeros(image.shape, dtype=np.int64)
    for index, cluster in enumerate(clusters):
        image_selection[image == cluster] = index + 1

    selection_colormap = ListedColormap(["#000", *[colormap.colors[cluster] for cluster in clusters]])  # type: ignore
    return image_selection, selection_colormap, clusters


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
    image_selection, cmap, clusters = get_selection_image(image, clusters, colormap)
    cluster_count_included = len(clusters)

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
    ax1.imshow(get_cluster_colors(image_selection, cmap), interpolation="nearest")

    # draw the colorbar as an image only when creating new figure
    if show_colorbar:
        divider = make_axes_locatable(ax1)

        ax2 = divider.append_axes("right", size="5%", pad=0.05)
        ax2.imshow(get_cluster_colors(np.expand_dims(np.arange(0, cluster_count_included + 1), axis=1), cmap))

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
    image_selection, cmap, _ = get_selection_image(image, clusters, colormap)
    plt.imsave(save_path, get_cluster_colors(image_selection, cmap))
