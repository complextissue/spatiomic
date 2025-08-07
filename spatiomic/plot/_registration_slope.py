from typing import Optional, Tuple

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def draw_connecting_line(
    point_1: Tuple[float, float],
    point_2: Tuple[float, float],
) -> mlines.Line2D:
    """Draw a line betwween two points.

    Args:
        point_1 (Tuple[float, float]): The coordinates of the first point.
        point_2 (Tuple[float, float]): The coordinates of the second point.

    Returns:
        mlines.Line2D: The line between the points.
    """
    ax = plt.gca()
    line = mlines.Line2D(
        [point_1[0], point_2[0]],
        [point_1[1], point_2[1]],
        color="red" if point_1[1] - point_2[1] > 0 else "green",
        marker="o",
        markersize=1,
        linewidth=1,
        alpha=0.2,
        linestyle="-",
    )
    ax.add_line(line)
    return line


def registration_slope(
    df_similarity: pd.DataFrame,
    before_label: str = "Before Registration",
    after_label: str = "After Registration",
    ssim_label: str = "SSIM",
    title: Optional[str] = "Change in Structural Similarity",
) -> plt.Figure:
    """Plot the distribution violinplot and histogram for SSIM scores before and after registration.

    Args:
        df_similarity (pd.DataFrame): [description]
        before_label (str, optional): [description]. Defaults to "Before Registration".
        after_label (str, optional): [description]. Defaults to "After Registration".
        ssim_label (str, optional): [description]. Defaults to "SSIM".
        title (Optional[str], optional): [description]. Defaults to "Change in Structural Similarity".

    Returns:
        plt.Figure: [description]
    """
    sns.set_theme(style="white", font="Arial")
    sns.set_context("paper")

    fig, ax = plt.subplots(1, 1, figsize=(6, 8))

    ax.scatter(y=df_similarity[before_label], x=np.repeat(1, df_similarity.shape[0]), s=10, color="black", alpha=0.7)
    ax.scatter(y=df_similarity[after_label], x=np.repeat(2, df_similarity.shape[0]), s=10, color="black", alpha=0.7)

    for point_1, point_2 in zip(df_similarity[before_label], df_similarity[after_label], strict=True):
        draw_connecting_line((1, point_1), (2, point_2))

    # Decoration
    if title is not None:
        ax.set_title(title, pad=30)
    ax.set(xlim=(0.85, 2.15), ylim=(-0.01, 0.5))
    ax.set_ylabel(ylabel=ssim_label, labelpad=30)
    ax.set_yticks(np.arange(0.0, 1.1, step=0.1))
    ax.set_yticklabels(str(round(n, 1)) for n in np.arange(0.0, 1.1, step=0.1))
    ax.set_xticks([1, 2])
    ax.set_xticklabels([before_label, after_label])

    ax.tick_params(axis="y", pad=15)
    ax.tick_params(axis="x", pad=15)

    sns.despine(bottom=True, left=True)
    plt.tight_layout(pad=1)

    return fig
