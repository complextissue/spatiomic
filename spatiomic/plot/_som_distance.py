from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

if TYPE_CHECKING:
    from spatiomic.dimension import som


def som_distance(
    som: "som",
) -> plt.Figure:
    """Create and a figure of the distance between SOM nodes.

    Args:
        som (som): The self-organizing map.

    Returns:
        plt.Figure: The figure.
    """
    sns.set_theme(style="white", font="Arial")
    sns.set_context("paper")

    fig, ax = plt.subplots()

    ax.set_title("Node Distance")
    ax.set_ylabel("SOM Dimension 2")
    ax.set_xlabel("SOM Dimension 1")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.pcolor(som.som.distance_map(), cmap="vlag")

    divider = make_axes_locatable(ax)

    ax2 = divider.append_axes("right", size="3%", pad=0.08)
    ax2.imshow(np.flip(np.expand_dims(np.arange(0, 1, step=0.25), axis=1), axis=0), cmap="vlag")
    ax2.set_yticks(
        [
            0,
            3,
        ]
    )
    ax2.set_yticklabels(
        [
            "Distant",
            "Close",
        ]
    )
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_xticks([])
    ax2.set_title("Relative Distance", loc="left", size=10)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    return fig
