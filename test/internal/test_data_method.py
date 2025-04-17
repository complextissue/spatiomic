"""Test the data_method decorator function."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import spatiomic as so


@pytest.mark.cpu
def test_data_method() -> None:
    """Test the data_method decorator function."""
    example_data = np.random.randint(size=(10, 3, 3), low=0, high=200)

    cluster_data_list = [
        example_data.astype(np.uint8),
        example_data.astype(np.uint16),
        example_data.astype(np.float16),
        example_data.astype(np.float32),
        example_data.astype(np.float64),
        pd.DataFrame(
            example_data[:, :, 0],
            index=np.arange(0, example_data.shape[0]),
            columns=np.arange(0, example_data.shape[1]),
        ),
        ad.AnnData(
            X=example_data.reshape(-1, example_data.shape[-1]),
        ),
    ]

    cluster_count = 3

    for cluster_data in cluster_data_list:
        # test it with the KMeans cluster function
        clusterer = so.cluster.kmeans(
            cluster_count=cluster_count,
            iteration_count=10,
            use_gpu=False,
            seed=1308419541,
        )

        communities = clusterer.fit_predict(
            cluster_data,
        )

        np.testing.assert_array_equal(
            np.sort(np.unique(communities)).astype(np.uint32),
            np.arange(0, cluster_count, step=1).astype(np.uint32),
        )
