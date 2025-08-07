from typing import List, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from spatiomic._internal._data_method import data_method


@data_method
def mean_cluster_intensity(
    data: NDArray,
    clusters: NDArray,
    channel_names: Optional[Union[List[str], List[int]]] = None,
) -> pd.DataFrame:
    """Calculate the mean intensity for the channels of data for each label category.

    Args:
        data (NDArray): The image data, channel-last.
        clusters (NDArray): The clusters for the data points.
        channel_names (Optional[Union[List[str], List[int]]], optional): The column names for the channels
            in the mean label intensity DataFrame. Defaults to None.

    Returns:
        pd.DataFrame: A dataframe of the channel-weise mean intensity per label with the clusters as the rows
            and the channels as columns.
    """
    clusters = np.array(clusters)

    assert np.multiply.reduce(clusters.shape) == np.multiply.reduce(data.shape[:-1]), (
        "The number of clusters has to match the number of data points."
    )

    # flatten the data in every but the channel dimension
    data_shape = data.shape
    dimension_count = data_shape[-1]
    data = data.reshape((-1, dimension_count))

    clusters_unique = np.sort(np.unique(clusters.ravel()))

    clusters_mean: Union[List[NDArray], NDArray] = []

    if isinstance(clusters_mean, list):  # should always be true but mypy complains otherwise
        for label_unique in clusters_unique:
            # get the indices of where the data has the specific label
            data_idx = np.argwhere(clusters == label_unique).ravel()
            label_mean = np.zeros((dimension_count)) if len(data_idx) == 0 else data[data_idx, :].mean(axis=0)

            clusters_mean.append(label_mean)

    clusters_mean = np.array(clusters_mean)

    clusters_mean_df = pd.DataFrame(
        clusters_mean,
        index=list(clusters_unique),
        columns=(list(np.arange(0, dimension_count)) if channel_names is None else channel_names),
    )

    return clusters_mean_df
