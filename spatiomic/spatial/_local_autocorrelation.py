from typing import Any, Literal
from warnings import warn

import numpy as np
import pandas as pd
from esda.getisord import G_Local
from esda.moran import Moran_Local
from libpysal.graph import Graph
from libpysal.weights import W
from numpy.typing import NDArray

from spatiomic._internal._check_cluster_image import check_cluster_image
from spatiomic.spatial._spatial_weights import queen_contiguity_weights


class LocalAutocorrelation:
    """Class to perform local spatial autocorrelation analysis on a 2D image of clusters or an XYC image."""

    @classmethod
    def predict(
        cls,
        data: NDArray,
        spatial_weights: Graph | W | None = None,
        method: Literal["moran", "getisord"] = "moran",
        permutation_count: int = 0,
        *args: Any,  # noqa: ARG003
        **kwargs: dict,
    ) -> pd.DataFrame:
        """Perform local spatial autocorrelation analysis on a 2D image of clusters.

        Args:
            data (NDArray): The data to calculate the local autocorrelation on.
            spatial_weights (Graph | W | None, optional): The spatial weights to use, e.g., from
                `so.spatial.spatial_weights`. If None, queen contiguity weights are used. Defaults to None.
            method (Literal["moran", "getisord"], optional): The method to use for local spatial autocorrelation
                analysis. Defaults to "moran".
            permutation_count (int, optional): The number of permutations to use for the pseudo-p-value calculation.
                Only used if method is "moran". Defaults to 0.

        Raises:
            TypeError: If a clustered image is not of integer type.
            ValueError: If a clustered image contains negative (including unassigned) labels.

        Returns:
            pd.DataFrame: The results of the local spatial autocorrelation analysis.
        """
        data_shape = data.shape
        channel_image = len(data_shape) == 3
        assert len(data_shape) == 2 or len(data_shape) == 3, (
            "Data must be a 2D image where the pixel values are the clusters or an image with a channel dimensionality."
        )
        if not channel_image:
            check_cluster_image(data, allow_unassigned=False)

        if channel_image and data.dtype != np.float64:
            warn(
                "Data is not in float64 format. Converting to float64 required by `esda`.",
                UserWarning,
                stacklevel=2,
            )
            data = data.astype(np.float64)

        w = queen_contiguity_weights(data.shape) if spatial_weights is None else spatial_weights

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
