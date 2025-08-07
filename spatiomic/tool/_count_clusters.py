from typing import List
from warnings import warn

import numpy as np
import pandas as pd


def count_clusters(
    file_paths: List[str],
    cluster_count: int,
    sort: bool = False,
    normalize: bool = False,
) -> pd.DataFrame:
    """Count the number of clusters in each image.

    Args:
        file_paths (List[str]): The paths to the files to count the clusters in.
        cluster_count (int): The number of clusters to count.
        sort (bool, optional): Whether to sort the file paths. Defaults to False.
        normalize (bool, optional): Whether to return the row-normalized cluster count. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the cluster counts.
    """
    if sort:
        warn(
            "Sorting the file paths, make sure manually added columns later on match this new order or set `sort` to "
            "False.",
            UserWarning,
            stacklevel=2,
        )
        file_paths = sorted(file_paths)

    cluster_counts = np.zeros((len(file_paths), cluster_count), dtype=int)

    df_cluster_counts = pd.DataFrame(cluster_counts, columns=[*range(0, cluster_count)])
    df_cluster_counts["file_path"] = file_paths
    df_cluster_counts = df_cluster_counts.set_index("file_path")

    for file_idx, file_path in enumerate(file_paths):
        data = np.load(file_path).reshape(-1)

        df_cluster_counts.iloc[file_idx, :] = [  # type: ignore
            *[np.sum(data == i) for i in range(0, cluster_count)],
        ]

    if normalize:
        df_cluster_counts = df_cluster_counts.div(df_cluster_counts.sum(axis=1), axis=0)

    return df_cluster_counts
