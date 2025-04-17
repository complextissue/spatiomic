from typing import Any, Literal, Optional
from warnings import warn

import numpy as np
import pandas as pd
from esda.getisord import G_Local
from esda.moran import Moran_Local
from libpysal.weights import W, lat2W
from numpy.typing import NDArray


class LocalAutocorrelation:
    """Class to perform local spatial autocorrelation analysis on a 2D image of clusters or an XYC image."""

    @classmethod
    def predict(
        cls,
        data: NDArray,
        spatial_weights: Optional[W] = None,
        method: Literal["moran", "getisord"] = "moran",
        permutation_count: int = 0,
        *args: Any,  # noqa: ARG003
        **kwargs: dict,
    ) -> pd.DataFrame:
        """Perform local spatial autocorrelation analysis on a 2D image of clusters.

        Args:
            data (NDArray): The data to calculate the local autocorrelation on.
            spatial_weights (Optional[W], optional): The spatial weights to use. If None, queen contiguity weights are
                generated based on the data shape using `lat2W`. Defaults to None.
            method (Literal["moran", "getisord"], optional): The method to use for local spatial autocorrelation
                analysis. Defaults to "moran".
            permutation_count (int, optional): The number of permutations to use for the pseudo-p-value calculation.
                Only used if method is "moran". Defaults to 0.

        Returns:
            pd.DataFrame: The results of the local spatial autocorrelation analysis.
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

        if channel_image and data.dtype != np.float64:
            warn(
                "Data is not in float64 format. Converting to float64 required by `esda`.",
                UserWarning,
                stacklevel=2,
            )
            data = data.astype(np.float64)

        w = lat2W(data.shape[0], data.shape[1], rook=False) if spatial_weights is None else spatial_weights

        if method == "moran":
            if permutation_count == 0:
                warn(
                    "The number of permutations is set to 0. No pseudo-p-value will be calculated.",
                    UserWarning,
                    stacklevel=2,
                )

            moran = Moran_Local(data, w, transformation="B", permutations=permutation_count, **kwargs)
            results_df = pd.DataFrame(
                {
                    "local_statistic": moran.Is,
                    "quadrant": moran.q,
                    "p_value": moran.p_sim if permutation_count > 0 else None,
                    "z_score": moran.z_sim if permutation_count > 0 else None,
                    "expected_value": moran.EI_sim if permutation_count > 0 else None,
                }
            )
        elif method == "getisord":
            star = kwargs.pop("star", True)
            getisord = G_Local(data, w, transform="B", star=star, **kwargs)
            results_df = pd.DataFrame(
                {
                    "local_statistic": getisord.Gs,
                    "z_score": getisord.Zs,
                    "p_value": getisord.p_sim,
                    "expected_value": getisord.EG_sim,
                }
            )

        return results_df
