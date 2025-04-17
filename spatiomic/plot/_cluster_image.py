from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray

from ._colormap import colormap as create_colormap


def cluster_image(
    image: NDArray,
    colormap: Optional[ListedColormap] = None,
) -> plt.Figure:
    """Plot the image with the clusters shown in colors and a colorbar.

    Args:
        image (NDArray): The image of the clusters.
        colormap (ListedColormap, optional): The colormap to use. Defaults to None.

    Returns:
        plt.Figure: The figure.
    """
    cluster_count = np.max(image) + 1

    colormap = create_colormap(color_count=cluster_count, seed=0) if colormap is None else colormap

    sns.set_theme(style="white", font="Arial")
    sns.set_context("paper")

    fig = plt.figure()

    ax1 = fig.add_subplot()
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(image, cmap=colormap)

    # draw the colorbar as an image
    divider = make_axes_locatable(ax1)

    ax2 = divider.append_axes("right", size="5%", pad=0.05)
    ax2.imshow(np.expand_dims(np.arange(0, cluster_count), axis=1), cmap=colormap)

    ax2.set_yticks(np.arange(0, cluster_count))
    ax2.set_yticklabels(np.arange(0, cluster_count))
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    ax2.set_xticks([])

    sns.despine(left=True, bottom=True)
    plt.tight_layout(pad=0.5)

    return fig


def save_cluster_image(
    image: NDArray,
    save_path: str,
    colormap: Optional[ListedColormap] = None,
) -> None:
    """Save the cluster image to a file.

    Args:
        image (NDArray): The image of the clusters to save.
        save_path (str): The path to save the image to.
        colormap (ListedColormap, optional): The colormap to use. Defaults to None.
    """
    plt.imsave(
        save_path,
        image,
        cmap=(create_colormap(color_count=np.max(image) + 1) if colormap is None else colormap),
    )
