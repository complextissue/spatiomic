from typing import Any, List, Literal, Optional, Union
from warnings import warn

import numpy as np
import pandas as pd
from esda.losh import LOSH
from libpysal.weights import W, lat2SW
from numpy.typing import NDArray

from spatiomic._internal._check_channel_dimension import check_channel_dimension


class LocalHeteroskedasticity:
    """Class to calculate local heteroskedasticity (LOSH) on a 2D image of clusters or an XYC image."""

    @classmethod
    def predict(
        cls,
        data: NDArray,
        channel_names: Optional[List[str]] = None,
        spatial_weights: Optional[W] = None,
        inference: Literal["chi-square", "permutation"] = "chi-square",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Perform local heteroskedasticity (LOSH) analysis on a 2D image of clusters or an XYC image.

        Args:
            data (NDArray): The input data. If the data is a 2D array, it is assumed to be a clustered image where each
                value represents a cluster label (integer). If the data is a 3D array (XYC), it is assumed to be a
                multi-channel image.
            channel_names (Optional[List[str]], optional): Names for each channel. Defaults to None.
            spatial_weights (Optional[W], optional): The spatial weights to use. If None, queen contiguity weights are
                generated based on the data shape using `lat2SW`. Defaults to None.
            inference (Literal["chi-square", "permutation"], optional):  Method for p-value inference in LOSH.
                Defaults to "chi-square".
            **kwargs: Additional keyword arguments to be passed to the `esda.losh.LOSH` class.

        Returns:
            pd.DataFrame: DataFrame containing LOSH values and p-values for each channel/cluster.
                Columns are 'losh' and 'p_value'.
                Index is based on `channel_names` if provided, otherwise a default RangeIndex.

        Raises:
            ValueError: If input data dimensions are not 2D or 3D.
            ValueError: If input data is a clustered image but not of integer type.
            ValueError: If input data is a clustered image and contains negative values.
            ValueError: If `channel_names` are provided for a clustered image (2D input).
            ValueError: If `channel_names` are provided for a 3D image (XYC) but do not match the number of channels.
        """
        data_shape = data.shape
        channel_image = len(data_shape) == 3
        check_channel_dimension(data_shape, dimension_count_min=2, dimension_count_max=3)

        if not channel_image:
            if not np.issubdtype(data.dtype, np.integer):
                raise ValueError("Data must be a clustered image of integer type.")
            if np.any(data < 0):
                raise ValueError("Clustered image data cannot contain negative values.")
            if channel_names is not None:
                warn(
                    "`channel_names` are ignored for clustered image input (2D data).",
                    UserWarning,
                    stacklevel=2,
                )
                channel_names = None
        else:
            if channel_names is not None and len(channel_names) != data_shape[-1]:
                raise ValueError(
                    "The number of channel names must match the number of channels in the input data (XYC)."
                )

        w = (
            lat2SW(data.shape[0], data.shape[1], criterion="queen").tocsr()
            if spatial_weights is None
            else spatial_weights
        )
        losh = LOSH(connectivity=w, inference=inference, **kwargs)

        losh_values: List[NDArray] = []
        p_values: List[NDArray] = []
        index: Union[List[str], List[int]]

        if not channel_image:
            # Process clustered image (2D integer array)
            cluster_count = data.max() + 1
            # One-hot encode the cluster labels
            data_for_losh = np.eye(cluster_count)[data].reshape((-1, cluster_count))
            index = list(np.arange(cluster_count))
        else:
            # Process channel image (3D array XYC)
            data_for_losh = data.reshape((-1, data_shape[-1]))
            index = channel_names if channel_names is not None else list(np.arange(data_shape[-1]))

        for i in range(data_for_losh.shape[-1]):
            losh.fit(data_for_losh[..., i])
            losh_values.append(losh.Hi)
            p_values.append(losh.pval)

        return pd.DataFrame(
            {
                "losh": losh_values,
                "p_value": p_values,
            },
            index=index,
        )
