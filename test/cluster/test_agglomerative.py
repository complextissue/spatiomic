"""Test the Agglomerative clustering implementation."""

import numpy as np
import pytest
from numpy.typing import NDArray

import spatiomic as so


@pytest.mark.cpu
def test_agglomerative_cpu(cluster_data: NDArray) -> None:
    """Test Agglomerative clustering.

    Args:
        cluster_data (NDArray): The test data to be clustered.
    """
    clusterer = so.cluster.agglomerative(
        cluster_count=10,
        use_gpu=False,
    )

    communities = clusterer.fit_predict(
        cluster_data,
        seed=1308419541,
    )

    np.testing.assert_array_equal(
        np.sort(np.unique(communities)).astype(np.uint32),
        np.arange(0, 10, step=1).astype(np.uint32),
    )
