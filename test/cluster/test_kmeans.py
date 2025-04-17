"""Test the KMeans clustering implementation."""

import numpy as np
import pytest
from numpy.typing import NDArray

import spatiomic as so


@pytest.mark.cpu
def test_kmeans_cpu(cluster_data: NDArray) -> None:
    """Test KMeans clustering.

    Args:
        cluster_data (NDArray): The test data to be clustered.
    """
    clusterer = so.cluster.kmeans(
        cluster_count=10,
        iteration_count=10,
        use_gpu=False,
        seed=1308419541,
    )

    communities = clusterer.fit_predict(
        cluster_data,
    )

    np.testing.assert_array_equal(
        np.sort(np.unique(communities)).astype(np.uint32),
        np.arange(0, 10, step=1).astype(np.uint32),
    )
