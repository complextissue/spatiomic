from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text


def cluster_contributors(
    data: pd.DataFrame,
    cluster_idx: int,
    cluster_column: str = "group",
    channel_column: str = "marker",
    log2_fold_change_column: str = "log2_fold_change",
    log10_p_value_column: Optional[str] = "log10_p_value",
    p_value_column: Optional[str] = None,
    increased_color: str = "#AA3333",
    neutral_color: str = "#AAAAAA",
    increased_p_value_threshold: float = 0.05,
    increased_log2_fold_change_threshold: float = 1.0,
    show_neutral: bool = True,
    fontsize: int = 10,
    mean_max: float = 1.0,
    size_increased: int = 20,
    size_neutral: int = 20,
    annotate_increased: bool = True,
    annotate_neutral: bool = False,
    figsize: Tuple[Union[int, float], Union[int, float]] = (8, 5),
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Union[plt.Figure, plt.Axes]:
    """Create half of a volcano plot to show the increased contributors to a cluster.

    .. warning:: When only p-values are available, the -log10(p-value) will be calculated with a small pseudo count
    of 1e-38 to avoid log(0) errors.

    Args:
        data (pd.DataFrame): The data, output from `so.tool.get_stats`.
        cluster_idx (int): The index of the cluster to plot.
        cluster_column (str): The column name for the cluster. Defaults to "group".
        channel_column (str): The column name for the channels. Defaults to "marker".
        log2_fold_change_column (str, optional): The column name for the log2 fold change.
            Defaults to "log2_fold_change".
        log10_p_value_column (Optional[str], optional): The column name for the log10 p value.
            Defaults to "log10_p_value".
        p_value_column (Optional[str], optional): The column name for the p value. Defaults to None.
        increased_color (str, optional): The color for the increase. Defaults to "red".
        neutral_color (str, optional): The color for the non-significant. Defaults to "grey".
        significant_mean_threshold (Optional[float], optional): The threshold for the mean. Defaults to None.
        significant_p_value_threshold (float, optional): The threshold for the p value. Defaults to 0.05.
        significant_log2_fold_change_threshold (float, optional): The threshold for the log2 fold change.
            Defaults to 1.0.
        show_neutral (bool, optional): Whether to show the neutral points. Defaults to True.
        fontsize (int, optional): The font size for the annotations. Defaults to 10.
        mean_max (float, optional): The maximum value for the mean. Defaults to 1.0.
        annotate_increased (bool, optional): Whether to shwo the names of the significant values. Defaults to True.
        annotate_neutral (bool, optional): Whether to annotate the neutral values. Defaults to False.
        figsize (Tuple[Union[int, float], Union[int, float]], optional): The size of the figure. Defaults to (8, 5).
        title (Optional[str], optional): The title of the plot. Defaults to None.
        ax: Existing axes to plot on. If None, creates new figure. Defaults to None.

    Returns:
        plt.Figure if ax is None, otherwise the provided plt.Axes.
    """
    # copy the data and check the columns
    df_contributors = data.copy()

    assert channel_column in df_contributors.columns, "channel_column not found in the DataFrame."
    assert log2_fold_change_column in df_contributors.columns, "log2_fold_change_column not found in the DataFrame."
    assert (
        (log10_p_value_column in df_contributors.columns, "log10_p_value_column not found in the DataFrame.")
        if log10_p_value_column is not None
        else (p_value_column in df_contributors.columns, "p_value_column not found in the DataFrame.")
    )

    df_contributors = df_contributors[df_contributors[cluster_column] == cluster_idx]

    if log10_p_value_column is not None and p_value_column is not None:
        raise ValueError(
            "Only one of `log10_p_value_column` and `p_value_column` can be used. "
            "Please set `log10_p_value_column` to None when specifying `p_value_column`."
        )

    if (
        log10_p_value_column is not None
        and log10_p_value_column in df_contributors.columns
        and log10_p_value_column != "log10_p_value"
    ):
        # remove any existing log10_p_value column
        if "log10_p_value" in df_contributors.columns:
            df_contributors = df_contributors.drop(columns=["log10_p_value"])
        df_contributors = df_contributors.rename(columns={log10_p_value_column: "log10_p_value"})

    if p_value_column is not None and p_value_column in df_contributors.columns:
        if "log10_p_value" in df_contributors.columns:
            df_contributors = df_contributors.drop(columns=["log10_p_value"])

        # add a small pseudo count to avoid log(0) errors
        df_contributors["log10_p_value"] = -np.log10(df_contributors[p_value_column].to_numpy() + 1e-38)  # type: ignore

        # replace inf values with the maximum value
        df_contributors["log10_p_value"] = df_contributors["log10_p_value"].replace([np.inf, -np.inf], -np.log10(1e-38))

        # fill NaN values with the maximum value
        df_contributors["log10_p_value"] = df_contributors["log10_p_value"].fillna(-np.log10(1e-38))

    if log2_fold_change_column != "log2_fold_change":
        if "log2_fold_change" in df_contributors.columns:
            df_contributors = df_contributors.drop(columns=["log2_fold_change"])
        df_contributors = df_contributors.rename(columns={log2_fold_change_column: "log2_fold_change"})

    # ensure that the log10_p_value and log2_fold_change columns are floats
    df_contributors["log10_p_value"] = df_contributors["log10_p_value"].astype(float)
    df_contributors["log2_fold_change"] = df_contributors["log2_fold_change"].astype(float)

    # create the plot
    if ax is None:
        sns.set_theme(style="white", font="Arial")
        sns.set_context("paper")
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        return_fig = True
    else:
        return_fig = False

    ax.axvline(x=increased_log2_fold_change_threshold, color="#aaaaaa", linestyle="--")

    # very small grid
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    ax.set_xlabel("log$_2$ FC")
    ax.set_ylabel("Mean intensity")

    if show_neutral:
        # scatter the neutral points
        neutral_points = df_contributors[
            (df_contributors["log2_fold_change"] <= increased_log2_fold_change_threshold)
            | (df_contributors["log10_p_value"] <= -np.log10(increased_p_value_threshold))
        ]

        ax.scatter(
            neutral_points["log2_fold_change"],
            neutral_points["mean_group"],
            color=neutral_color,
            s=size_neutral,
            alpha=0.25,
        )

    # scatter the significantly increased points
    increased_points = df_contributors[
        (df_contributors["log2_fold_change"] > increased_log2_fold_change_threshold)
        & (df_contributors["log10_p_value"] >= -np.log10(increased_p_value_threshold))
    ]

    ax.scatter(
        increased_points["log2_fold_change"],
        increased_points["mean_group"],
        color=increased_color,
        s=size_increased,
        alpha=0.9,
    )

    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylim(0, ax.get_ylim()[1])

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    texts = []

    if annotate_neutral:
        for _, row in neutral_points.iterrows():
            if row["log2_fold_change"] <= xlim[0] or row["log2_fold_change"] >= xlim[1]:
                continue

            texts.append(
                ax.text(
                    row["log2_fold_change"],
                    row["mean_group"],
                    str(row[channel_column]),
                    fontsize=fontsize,
                    color="black",
                )
            )

    if annotate_increased:
        for _, row in increased_points.iterrows():
            if row["log2_fold_change"] <= xlim[0] or row["log2_fold_change"] >= xlim[1]:
                continue

            texts.append(
                ax.text(
                    row["log2_fold_change"],
                    row["mean_group"],
                    str(row[channel_column]),
                    fontsize=fontsize,
                    color="black",
                )
            )

    # add lines to avoid text overlap
    if len(texts) > 0:
        adjust_text(ax.texts, arrowprops={"arrowstyle": "-", "color": "black"})

    # show every integer on the x-axis
    ax.set_xticks(np.arange(int(xlim[0]), int(xlim[1]) + 1, 1))
    ax.set_yticks(np.arange(0, mean_max, (mean_max / 10)))

    if title is not None:
        ax.set_title(title)

    if return_fig:
        fig.tight_layout()
        return fig
    else:
        return ax
