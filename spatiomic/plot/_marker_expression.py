from math import ceil, sqrt
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from numpy.typing import NDArray


def marker_expression(
    image: NDArray,
    title: str = "Marker Expression",
    channel_names: Optional[List[Union[str, int]]] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
    colormap: Optional[ListedColormap] = None,
) -> plt.Figure:
    """Plot the expression of each marker in an image.

    Args:
        image (NDArray): The XYC image with the markers.
        title (str, optional): The title of the plot. Defaults to "Marker Expression".
        channel_names (Optional[List[Union[str, int]]], optional): The names of the channels. Defaults to None.
        min_value (Optional[float], optional): The minimum value for the colormap. Defaults to None.
        max_value (Optional[float], optional): The maximum value for the colormap. Defaults to None.
        figsize (Optional[Tuple[Union[int, float], Union[int, float]]], optional): The size of the figure.
            Defaults to None.
        colormap (Optional[ListedColormap], optional): The colormap to use. Defaults to None.

    Returns:
        plt.Figure: The figure.
    """
    image_shape = image.shape
    dimension_count = image_shape[-1]
    image = image.reshape(-1, image_shape[-1])

    column_count = ceil(sqrt(dimension_count))
    row_count = ceil(sqrt(dimension_count))

    sns.set_theme(style="white", font="Arial")
    sns.set_context("paper")

    figsize = figsize if figsize is not None else (column_count * 2, row_count * 2)
    fig = plt.figure(figsize=figsize)

    plt.title(title, y=1.02)
    plt.xticks([])
    plt.yticks([])

    # check if we really need to ceil both counts
    if (dimension_count / column_count) <= (row_count - 1):
        row_count = row_count - 1

    for dimension_idx in range(0, dimension_count):
        fig.add_subplot(row_count, column_count, dimension_idx + 1)

        plt.xticks([])
        plt.yticks([])

        if channel_names is None:
            plt.title(str(dimension_idx))
        else:
            plt.title(str(channel_names[dimension_idx]))

        plt.imshow(
            image[:, dimension_idx].reshape(image_shape[:-1]),
            cmap=sns.color_palette("Blues", as_cmap=True) if colormap is None else colormap,
            vmin=0 if min_value is None else min_value,
            vmax=np.max(image) if max_value is None else max_value,
        )

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    return fig
