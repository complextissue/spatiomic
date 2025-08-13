from typing import Any, List, Literal, Optional

import numpy as np
import pandas as pd
from esda.lee import Spatial_Pearson
from esda.moran import Moran_BV
from libpysal.weights import W, lat2W
from numpy.typing import NDArray


class BivariateCorrelation:
    """Class to perform bivariate spatial correlation analysis on a 2D image of clusters and an XYC image.

    Usage:

    .. code-block:: python

        data = my_clustered_image # 2d, integers representing clusters
        channel_image = my_xyc_image # or one-hot encoded cluster image

        channel_names = ["CD3", "CD4", "CD8", "CD20", "CD68"] # optional, defaults to channel indices

        df_morans_i = so.spatial.bivariate_correlation.predict(
            data,
            channel_image,
            channel_names
            method="moran",
            permutation_count=0,
        )
    """

    def predict(
        self,
        data: NDArray,
        channel_image: NDArray,
        channel_names: Optional[List[str]] = None,
        spatial_weights: Optional[W] = None,
        method: Literal["moran", "lee"] = "moran",
        permutation_count: int = 0,
        *args: Any,  # noqa: ARG002
        **kwargs: dict,  # noqa: ARG002
    ) -> pd.DataFrame:
        """Perform spatial correlation analysis on a 2D image of clusters and an XYC image.

        This analysis allows you to determine which clusters spatially correlate with which channels in the neighboring
        pixels of the image.

        Args:
            data (NDArray): The clustered image.
            channel_image (NDArray): The XYC image to correlate with the clusters.
            channel_names (Optional[List[str]], optional): The names of the channels in the XYC image.
                Defaults to None.
            spatial_weights (Optional[W], optional): The spatial weights to use. If None, queen contiguity weights are
                generated based on the data shape using `lat2W`. Defaults to None.
            method (Literal["moran", "lee"], optional): The method to use for spatial correlation analysis.
            permutation_count (int, optional): The number of permutations to use for p-value calculation, 0 if no
                simulation is to be performed.
                Defaults to 0.

        Returns:
            pd.DataFrame: A DataFrame containing the Moran's I analysis results for each cluster.
        """
        data_shape = data.shape
        channel_image_shape = channel_image.shape

        assert data_shape == channel_image_shape[:-1], (
            "The data and channel image must have the same shape except for the channel dimension."
        )

        spatial_weights = (
            lat2W(data_shape[0], data_shape[1], rook=False) if spatial_weights is None else spatial_weights
        )

        unique_clusters = np.unique(data)
        unique_channels = np.arange(channel_image_shape[-1])

        base_columns = [
            "cluster",
            "channel",
            "morans_i",
        ]
        columns = base_columns.copy()
        if permutation_count > 0:
            columns.extend(["p_value", "z_score"])

        # Collect all results in a list to avoid repeated concatenation
        results_list = []

        if method == "moran":
            autocorrelation_function = Moran_BV
            get_correlation = lambda result: result.I
        elif method == "lee":
            spatial_pearson = Spatial_Pearson(
                connectivity=spatial_weights,
                permutations=permutation_count,
            )
            autocorrelation_function = lambda x, y, w, permutations: spatial_pearson.fit(  # noqa: ARG005
                x,
                y,
            )
            get_correlation = lambda _result: spatial_pearson.association_[0, 1]

        for cluster in unique_clusters:
            for channel in unique_channels:
                result = autocorrelation_function(
                    (data == cluster).astype(int),
                    channel_image[:, :, channel],
                    spatial_weights,
                    permutations=permutation_count,
                )

                row_data = [
                    cluster,
                    (channel if channel_names is None else channel_names[channel]),
                    get_correlation(result),
                ]

                if permutation_count > 0 and method == "moran":
                    row_data.extend([result.p_sim, result.z_sim])

                results_list.append(row_data)

        # Create DataFrame from all results at once
        df_clusters = pd.DataFrame(results_list, columns=columns)

        return df_clusters
