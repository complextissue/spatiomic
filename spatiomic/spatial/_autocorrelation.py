from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from esda.geary import Geary
from esda.moran import Moran
from libpysal.weights import W, lat2W
from numpy.typing import NDArray
from tqdm import tqdm


class Autocorrelation:
    """Class to perform spatial autocorrelation analysis on a 2D image of clusters or an XYC image."""

    def predict(
        self,
        data: NDArray,
        spatial_weights: Optional[W] = None,
        method: Literal["moran", "geary"] = "moran",
        permutation_count: int = 0,
        *args: Any,  # noqa: ARG002
        **kwargs: dict,  # noqa: ARG002
    ) -> pd.DataFrame:
        """Perform spatial autocorrelation analysis on a 2D image of clusters or an XYC image.

        Analyse the spatial autocorrelation of each cluster or each channel in the image using Moran's I or Geary's C.

        Args:
            data (NDArray): The 2D image to perform autocorrelation analysis on.
            spatial_weights (Optional[W], optional): The spatial weights to use. If None, queen contiguity weights are
                generated based on the data shape using `lat2W`. Defaults to None.
            method (Literal["moran", "geary"], optional): The method to use. Defaults to "moran".
            permutation_count (int, optional): The number of permutations to use for p-value calculation, 0 if no
                simulation is to be performed.
                Defaults to 0.

        Returns:
            pd.DataFrame: A DataFrame containing the Moran's I analysis results for each cluster.
        """
        data_shape = data.shape
        channel_image = len(data_shape) == 3
        assert len(data_shape) == 2 or len(data_shape) == 3, (
            "Data must be a 2D image where the pixel values are the clusters or an image with a channel dimensionality."
        )
        assert channel_image or data.dtype in [
            int,
            np.int8,
            np.int16,
            np.uint8,
            np.uint16,
        ], "Clusters in data must be integer type."
        assert channel_image or bool(np.any(data < 0)) is False, (
            "Data contains negative values. Clusters cannot be negative."
        )

        spatial_weights_matrix = (
            lat2W(data_shape[0], data_shape[1], rook=False) if spatial_weights is None else spatial_weights
        )
        unique_clusters = np.unique(data) if not channel_image else np.arange(data_shape[-1])

        base_columns = [
            ("cluster" if not channel_image else "channel"),
            ("morans_i" if method == "moran" else "gearys_c"),
            ("morans_i_expected" if method == "moran" else "gearys_c_expected"),
        ]
        columns = base_columns.copy()
        if permutation_count > 0:
            columns.extend(["p_value", "z_score"])

        df_clusters = pd.DataFrame(columns=columns)

        autocorrelation_function = Moran if method == "moran" else Geary
        correlation_key = "I" if method == "moran" else "C"

        for cluster in tqdm(unique_clusters, desc="Calculating spatial autocorrelation for each channel/cluster"):
            result = autocorrelation_function(
                ((data == cluster).astype(int) if not channel_image else data[:, :, cluster]),
                spatial_weights_matrix,
                permutations=permutation_count,
            )

            df_correlation = pd.DataFrame(
                [[cluster, getattr(result, correlation_key), getattr(result, f"E{correlation_key}")]],
                columns=base_columns,
            )

            if permutation_count > 0:
                df_correlation["p_value"] = result.p_sim
                df_correlation["z_score"] = result.z_sim

            df_clusters = (
                pd.concat([df_clusters, df_correlation], ignore_index=True) if not df_clusters.empty else df_correlation
            )

        return df_clusters
