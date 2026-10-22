"""Test the data_method decorator function."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

import spatiomic as so

EXAMPLE_DATA = np.random.default_rng(0).integers(size=(10, 3, 3), low=0, high=200)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "cluster_data",
    [
        EXAMPLE_DATA.astype(np.uint8),
        EXAMPLE_DATA.astype(np.uint16),
        EXAMPLE_DATA.astype(np.float16),
        EXAMPLE_DATA.astype(np.float32),
        EXAMPLE_DATA.astype(np.float64),
        pd.DataFrame(EXAMPLE_DATA[:, :, 0]),
        ad.AnnData(X=EXAMPLE_DATA.reshape(-1, EXAMPLE_DATA.shape[-1])),
    ],
    ids=["uint8", "uint16", "float16", "float32", "float64", "dataframe", "anndata"],
)
def test_data_method(cluster_data: NDArray | pd.DataFrame | ad.AnnData) -> None:
    """Test that the data_method decorator accepts integer, float, DataFrame and AnnData input."""
    cluster_count = 3

    clusterer = so.cluster.kmeans(
        cluster_count=cluster_count,
        iteration_count=10,
        use_gpu=False,
        seed=1308419541,
    )

    communities = clusterer.fit_predict(cluster_data)

    np.testing.assert_array_equal(
        np.sort(np.unique(communities)).astype(np.uint32),
        np.arange(0, cluster_count, step=1).astype(np.uint32),
    )
