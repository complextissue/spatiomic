from math import ceil, sqrt
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import seaborn as sns
from numpy.typing import NDArray


def som_marker_expression(
    data: NDArray,
    channel_names: Optional[List[Union[str, int]]] = None,
    figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
) -> plt.Figure:
    """Visualize marker expressions of all markers.

    Args:
        data (NDArray): The multi-channel image.
        channel_names (Optional[List[Union[str, int]]], optional): Custom names for the channels. Defaults to None.
        figsize (Optional[Tuple[Union[int, float], Union[int, float]]], optional): The size of the figure.
            Defaults to None.

    Returns:
        plt.Figure: The figure.
    """
    assert len(data.shape) == 3, "Data has to be 3-dimensional, two spatial dimensions and one channel dimension."

    sns.set_theme(style="white", font="Arial")
    sns.set_context("paper")

    dimension_count = data.shape[-1]

    row_column_count = ceil(sqrt(dimension_count))

    figsize = figsize if figsize is not None else (row_column_count * 2, row_column_count * 2)
    fig = plt.figure(figsize=figsize)

    plt.title("SOM Node Marker Intensity", y=1.05)
    plt.yticks([])
    plt.xticks([])

    for n in range(1, row_column_count + 1):
        for j in range(1, row_column_count + 1):
            idx = (n - 1) * row_column_count + (j - 1)
            if idx < dimension_count:
                fig.add_subplot(row_column_count, row_column_count, idx + 1)

                plt.imshow(data[:, :, idx], cmap="vlag")
                if channel_names is not None:
                    plt.title(str(channel_names[idx]))

                plt.xticks([])
                plt.yticks([])

    sns.despine(left=True, bottom=True)
    fig.tight_layout()

    return fig
