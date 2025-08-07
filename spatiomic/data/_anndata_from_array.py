from typing import List, Optional

import numpy as np
import pandas as pd
from anndata import AnnData
from libpysal.weights import W, lat2SW
from numpy.typing import NDArray

from spatiomic._internal._check_channel_dimension import check_channel_dimension


def anndata_from_array(
    data: NDArray,
    channel_names: List[str],
    clusters: Optional[NDArray] = None,
    cluster_key: str = "clusters",
    spatial_weights: Optional[W] = None,
) -> AnnData:
    """Create an AnnData object from an image data array.

    Args:
        data (NDArray): The image data array.
        channel_names (List[str]): The channel names.
        clusters (Optional[NDArray], optional): The clusters. Defaults to None.
        cluster_key (str, optional): The cluster key. Defaults to "clusters".
        spatial_weights (Optional[W], optional): The spatial weights. Defaults to None.

    Returns:
        AnnData: The AnnData object
    """
    check_channel_dimension(data.shape, dimension_count_min=2, dimension_count_max=3)

    data_shape = data.shape
    rows, cols, channels = data_shape
    data_flat = data.reshape(-1, channels)

    spatial_weights = (
        lat2SW(data_shape[0], data_shape[1], criterion="queen").tocsr()
        if spatial_weights is None
        else spatial_weights.to_sparse(fmt="csr")
    )

    # Create the AnnData object
    adata = AnnData(X=data_flat)

    # Add the channel names
    if not len(channel_names) == channels:
        raise ValueError("The number of channel names must match the number of channels.")
    adata.var_names = channel_names

    # Add the coordinates
    grid_indices = np.array(np.meshgrid(range(rows), range(cols))).T.reshape(-1, 2)
    adata.obsm["spatial"] = grid_indices

    # Add the clusters
    if clusters is not None:
        clusters = clusters.ravel()
        if len(clusters) != (rows * cols):
            raise ValueError("The number of clusters must match the number of pixels.")

        adata.obs[cluster_key] = pd.Categorical(clusters)

    # Add the spatial weights
    adata.obsp["spatial_connectivities"] = spatial_weights

    return adata
