from typing import Optional, Tuple, Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text


def volcano(
    data: pd.DataFrame,
    channel_column: str,
    log2_fold_change_column: str = "log2_fold_change",
    log10_p_value_column: Optional[str] = "log10_p_value",
    p_value_column: Optional[str] = None,
    increased_color: str = "#AA3333",
    decreased_color: str = "#3333AA",
    neutral_color: str = "#AAAAAA",
    significant_p_value_threshold: float = 0.05,
    significant_log2_fold_change_threshold: float = 1.0,
    annotate_significant: bool = True,
    annotate_neutral: bool = False,
    size_significant: int = 20,
    size_neutral: int = 20,
    fontsize: int = 8,
    figsize: Tuple[Union[int, float], Union[int, float]] = (10, 5),
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Union[plt.Figure, plt.Axes]:
    """Create a volcano plot of the data.

    Args:
        data (pd.DataFrame): The data.
        channel_column (str): The column name for the channel.
        log2_fold_change_column (str, optional): The column name for the log2 fold change.
            Defaults to "log2_fold_change".
        log10_p_value_column (Optional[str], optional): The column name for the log10 p value.
            Defaults to "log10_p_value".
        p_value_column (Optional[str], optional): The column name for the p value. Defaults to None.
        increased_color (str, optional): The color for the increase. Defaults to "#AA3333".
        decreased_color (str, optional): The color for the decrease. Defaults to "#3333AA".
        neutral_color (str, optional): The color for the non-significant. Defaults to "#AAAAAA".
        significant_p_value_threshold (float, optional): The threshold for the p value. Defaults to 0.05.
        significant_log2_fold_change_threshold (float, optional): The threshold for the log2 fold change.
            Defaults to 1.0.
        annotate_significant (bool, optional): Whether to annotate the significant values. Defaults to True.
        annotate_neutral (bool, optional): Whether to annotate the neutral values. Defaults to False.
        size_significant (int, optional): The size of the significant points. Defaults to 20.
        size_neutral (int, optional): The size of the neutral points. Defaults to 20.
        fontsize (int, optional): The font size for the annotations. Defaults to 8.
        figsize (Tuple[Union[int, float], Union[int, float]], optional): The size of the figure. Defaults to (10, 5).
        title (Optional[str], optional): The title of the plot. Defaults to None.
        ax: Existing axes to plot on. If None, creates new figure. Defaults to None.

    Returns:
        plt.Figure if ax is None, otherwise the provided plt.Axes.
    """
    # copy the data and check the columns
    df_volcano = data.copy()

    assert channel_column in df_volcano.columns, "channel_column not found in the DataFrame."
    assert log2_fold_change_column in df_volcano.columns, "log2_fold_change_column not found in the DataFrame."
    assert (
        (log10_p_value_column in df_volcano.columns, "log10_p_value_column not found in the DataFrame.")
        if log10_p_value_column is not None
        else (p_value_column in df_volcano.columns, "p_value_column not found in the DataFrame.")
    )

    if log10_p_value_column is not None and p_value_column is not None:
        raise ValueError(
            "Only one of `log10_p_value_column` and `p_value_column` can be used. "
            "Please set `log10_p_value_column` to None when specifying `p_value_column`."
        )

    if (
        log10_p_value_column is not None
        and log10_p_value_column in df_volcano.columns
        and log10_p_value_column != "log10_p_value"
    ):
        if "log10_p_value" in df_volcano.columns:
            df_volcano = df_volcano.drop(columns=["log10_p_value"])
        df_volcano = df_volcano.rename(columns={log10_p_value_column: "log10_p_value"})

    if p_value_column is not None and p_value_column in df_volcano.columns:
        df_volcano["log10_p_value"] = -np.log10(df_volcano[p_value_column] + 1e-38)

    if log2_fold_change_column != "log2_fold_change":
        if "log2_fold_change" in df_volcano.columns:
            df_volcano = df_volcano.drop(columns=["log2_fold_change"])
        df_volcano = df_volcano.rename(columns={log2_fold_change_column: "log2_fold_change"})

    # parse the data as floats
    df_volcano["log10_p_value"] = df_volcano["log10_p_value"].astype(float)
    df_volcano["log2_fold_change"] = df_volcano["log2_fold_change"].astype(float)

    # check for infinite values in the log10_p_value column
    if np.isinf(df_volcano["log10_p_value"]).any():
        raise ValueError("Infinite values found in the log10_p_value column.")
    elif np.isnan(df_volcano["log10_p_value"]).any():
        warn("NaN values found in the log10_p_value column.", UserWarning, stacklevel=2)
        # replace NaN values with 0
        df_volcano["log10_p_value"] = df_volcano["log10_p_value"].fillna(0)

    # check for infinite values in the log2_fold_change column
    if np.isinf(df_volcano["log2_fold_change"]).any():
        raise ValueError("Infinite values found in the log2_fold_change column.")
    elif np.isnan(df_volcano["log2_fold_change"]).any():
        warn("NaN values found in the log2_fold_change column.", UserWarning, stacklevel=2)
        # replace NaN values with 0
        df_volcano["log2_fold_change"] = df_volcano["log2_fold_change"].fillna(0)

    # create the plot
    if ax is None:
        sns.set_theme(style="white", font="Arial")
        sns.set_context("paper")
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        return_fig = True
    else:
        return_fig = False

    ax.grid(True, which="both", linestyle=":", linewidth=0.5, zorder=0)

    ax.axvline(x=significant_log2_fold_change_threshold, color="grey", linestyle="--", linewidth=1, zorder=1)
    ax.axvline(x=-significant_log2_fold_change_threshold, color="grey", linestyle="--", linewidth=1, zorder=1)
    ax.axhline(y=-np.log10(significant_p_value_threshold), color="grey", linestyle="--", linewidth=1, zorder=1)

    ax.set_xlabel("log$_2$ FC")
    ax.set_ylabel("-log$_{10}$(p-value)")

    # scatter the neutral points
    neutral_points = df_volcano[
        (
            (df_volcano["log2_fold_change"] <= significant_log2_fold_change_threshold)
            & (df_volcano["log2_fold_change"] >= -significant_log2_fold_change_threshold)
        )
        | (df_volcano["log10_p_value"] <= -np.log10(significant_p_value_threshold))
    ]

    ax.scatter(
        neutral_points["log2_fold_change"],
        neutral_points["log10_p_value"],
        color=neutral_color,
        s=size_neutral,
        zorder=2,
    )

    # scatter the significantly decreased points
    decreased_points = df_volcano[
        (df_volcano["log2_fold_change"].astype(float) < -significant_log2_fold_change_threshold)
        & (df_volcano["log10_p_value"].astype(float) >= -np.log10(significant_p_value_threshold))
    ]

    ax.scatter(
        decreased_points["log2_fold_change"],
        decreased_points["log10_p_value"],
        color=decreased_color,
        s=size_significant,
        zorder=3,
    )

    # scatter the significantly increased points
    increased_points = df_volcano[
        (df_volcano["log2_fold_change"] > significant_log2_fold_change_threshold)
        & (df_volcano["log10_p_value"] >= -np.log10(significant_p_value_threshold))
    ]

    ax.scatter(
        increased_points["log2_fold_change"],
        increased_points["log10_p_value"],
        color=increased_color,
        s=size_significant,
        zorder=3,
    )

    ax.set_ylim(0, ax.get_ylim()[1])

    texts = []

    if annotate_neutral:
        for _, row in neutral_points.iterrows():
            texts.append(
                ax.text(
                    row["log2_fold_change"],
                    row["log10_p_value"],
                    str(row[channel_column]),
                    fontsize=fontsize,
                    color="black",
                    zorder=3,
                )
            )

    if annotate_significant:
        for df_significant in [increased_points, decreased_points]:
            for _, row in df_significant.iterrows():
                texts.append(
                    ax.text(
                        row["log2_fold_change"],
                        row["log10_p_value"],
                        str(row[channel_column]),
                        fontsize=fontsize,
                        color="black",
                        zorder=4,
                    )
                )

    # add lines to avoid text overlap
    if len(texts) > 0:
        adjust_text(ax.texts, arrowprops={"arrowstyle": "-", "color": "black"})

    # show every integer on the x-axis
    ax.set_xticks(np.arange(int(ax.get_xlim()[0]), int(ax.get_xlim()[1]) + 1, 1))
    ax.set_yticks(np.arange(0, int(ax.get_ylim()[1]) + 1, 2))

    if title is not None:
        ax.set_title(title)

    if return_fig:
        return fig
    else:
        return ax
