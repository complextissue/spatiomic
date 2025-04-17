from math import ceil, sqrt
from typing import List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
from numpy.typing import NDArray


def cluster_contributor_histogram(
    data: NDArray,
    channel_names: List[str],
    max_data: Optional[float] = None,
    bin_count: int = 10,
    scaling_factor_y: float = 100.0,
) -> plt.Figure:
    """Create a histogram of the mean of the cluster contributors.

    Args:
        data (NDArray): The data of a single cluster.
        channel_names (List[str]): The channel names.
        scaling_factor_y (float, optional): The scaling factor for the y-axis. Defaults to 100.0.

    Returns:
        plt.Figure: The figure.
    """
    assert len(channel_names) == data.shape[1], "The channel names must match the data shape."

    sns.set_theme(style="white", font="Arial")
    sns.set_context("paper")

    axes_count = len(channel_names)
    axes_count_sqrt = ceil(sqrt(axes_count))
    fig, axes = plt.subplots(
        nrows=axes_count_sqrt,
        ncols=axes_count_sqrt,
        figsize=(axes_count_sqrt * 2, axes_count_sqrt * 2),
        dpi=100,
    )

    max_data = data.max() if max_data is None else max_data
    ylim = data.shape[0] / (scaling_factor_y * bin_count)

    for i, ax in enumerate(axes.flat):  # type: ignore
        if i >= axes_count:
            ax.set_visible(False)
            continue

        ax.hist(
            data[:, i],
            bins=bin_count,
            range=(0, max_data),
            color="blue",
        )

        # have the x-ticks and y-ticks closer to the plot
        ax.tick_params(axis="x", direction="out", pad=-2)
        ax.tick_params(axis="y", direction="out", pad=-2)

        ax.set_xlim(0, max_data)
        ax.set_ylim(0, ylim)
        ax.set_title(channel_names[i], fontsize=10, pad=2)

    fig.tight_layout(pad=0.5)
    return fig
