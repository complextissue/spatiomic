from typing import Optional, Tuple, Union

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
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
    show_colorbar: bool = True,
    title: Optional[str] = None,
) -> Union[plt.Figure, plt.Axes]:
    """Display cluster/mask image with automatic colormap and optional colorbar.

    Args:
        image: Cluster/mask image where each region has a unique positive integer ID.
            Background should be 0.
        colormap: Custom colormap for displaying clusters. If None, uses default colormap
            with black background. Defaults to None.
        ax: Existing axes to plot on. If None, creates new figure. Defaults to None.
        figsize: Figure size as (width, height). Only used if ax is None. Defaults to None.
        show_colorbar: Whether to show a colorbar with cluster/mask IDs. Defaults to True.
        title: Title for the plot. Defaults to None.

    Returns:
        plt.Figure if ax is None, otherwise the provided plt.Axes.
    """
    # Calculate cluster count and create colormap if needed
    cluster_count = np.max(image) + 1
    if colormap is None:
        colormap = create_colormap(color_count=cluster_count, color_override={0: "#000000"})

    # Setup plotting
    if ax is None:
        sns.set_theme(style="white", font="Arial")
        sns.set_context("paper")
        fig, ax = plt.subplots(figsize=figsize)
        return_fig = True
    else:
        return_fig = False

    # Display image
    ax.imshow(image, cmap=colormap)
    ax.set_xticks([])
    ax.set_yticks([])

    if title is not None:
        ax.set_title(title)

    # Add colorbar if requested and we're creating a new figure
    if show_colorbar and return_fig:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.imshow(np.expand_dims(np.arange(0, cluster_count), axis=1), cmap=colormap)
        cax.set_yticks(np.arange(0, cluster_count))
        cax.set_yticklabels(np.arange(0, cluster_count))
        cax.yaxis.tick_right()
        cax.yaxis.set_label_position("right")
        cax.set_xticks([])
        sns.despine(ax=cax, left=True, bottom=True)

    # Clean up main axes
    sns.despine(ax=ax, left=True, bottom=True)

    if return_fig:
        plt.tight_layout(pad=0.5)
        return fig
    else:
        return ax


def save_cluster_image(
    image: NDArray,
    save_path: str,
    colormap: Optional[ListedColormap] = None,
    dpi: int = 300,
) -> None:
    """Save cluster/mask image to file without borders.

    Args:
        image: Cluster/mask image where each region has a unique positive integer ID.
        save_path: Path where to save the image.
        colormap: Custom colormap for clusters. If None, uses default colormap with
            black background. Defaults to None.
        dpi: Resolution for saved image. Defaults to 300.
    """
    # Create colormap if needed
    cluster_count = np.max(image) + 1
    if colormap is None:
        colormap = create_colormap(color_count=cluster_count, color_override={0: "#000000"})

    # Save using matplotlib's imsave
    plt.imsave(save_path, image, cmap=colormap, dpi=dpi)
