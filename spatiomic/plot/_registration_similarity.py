from typing import Optional

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


def registration_similarity(
    df_similarity: pd.DataFrame,
    before_label: str = "Before Registration",
    after_label: str = "After Registration",
    title: Optional[str] = "Structural Similarity",
    point_size: int = 70,
    xtick_min: float = 0.0,
    xtick_max: float = 1.0,
) -> plt.Figure:
    """Plot the change in the MSSIM before and after registration.

    Args:
        df (pd.DataFrame): [description]
        before_label (str, optional): [description]. Defaults to "Before Registration".
        after_label (str, optional): [description]. Defaults to "After Registration".
        title (Optional[str], optional): [description]. Defaults to "Structural Similarity".
        point_size (int, optional): [description]. Defaults to 70.
        xtick_min (float, optional): [description]. Defaults to 0.0.
        xtick_max (float, optional): [description]. Defaults to 1.0.

    Returns:
        plt.Figure: [description]
    """
    sns.set_theme(style="white", font="Arial")
    sns.set_context("paper")

    # make index available as "index" column
    df_similarity = df_similarity.reset_index()

    fig, ax1 = plt.subplots(1, 1)

    ax1.scatter(
        y=df_similarity["index"],
        x=df_similarity[before_label],
        s=point_size,
        color="#bb0000",
        alpha=1.0,
    )
    ax1.scatter(
        y=df_similarity["index"],
        x=df_similarity[after_label],
        s=point_size,
        color="#00aa00",
        alpha=1.0,
    )

    # draw the dashed lines between the before and after points
    for idx, ssim_before, ssim_after in zip(
        df_similarity["index"], df_similarity[before_label], df_similarity[after_label], strict=True
    ):
        ax1 = plt.gca()
        line = mlines.Line2D(
            [ssim_before, ssim_after],
            [idx, idx],
            color="grey",
            alpha=0.5,
            linestyle="--",
        )
        ax1.add_line(line)

    ax1.set_xticks(np.arange(xtick_min, xtick_max, 0.1))
    ax1.tick_params(axis="y", pad=20)
    ax1.tick_params(axis="x", pad=30)

    if title is not None:
        ax1.set_title(title, pad=30)

    # append a legend
    divider = make_axes_locatable(ax1)
    ax2 = divider.append_axes("right", size="5%", pad=0.3)
    ax2.imshow(
        np.expand_dims(np.arange(0, 2), axis=1),
        cmap=ListedColormap(["#00aa00", "#bb0000"]),
    )

    ax2.set_xticks([])

    ax2.set_yticks([0, 1])
    ax2.set_yticklabels([after_label, before_label])

    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    ax2.tick_params(axis="y")
    ax2.tick_params(axis="x")

    sns.despine(bottom=True, left=True)
    plt.tight_layout()

    return fig
