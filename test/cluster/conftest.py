"""Data fixtures for cluster module testing."""

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from sklearn import datasets


@pytest.fixture(scope="session")
def cluster_data() -> NDArray:
    """Create example blob data with 10 clusters."""
    blobs, _ = datasets.make_blobs(
        n_samples=400,
        centers=10,
        n_features=10,
        random_state=1308419541,
        return_centers=False,
    )

    return np.array(blobs).reshape((20, 20, -1))


@pytest.fixture(scope="session")
def biclustering_data() -> pd.DataFrame:
    """Create an example biclustering data frame."""
    biclusters, _rows, _columns = datasets.make_biclusters(
        shape=(20, 20),
        n_clusters=10,
        noise=0.5,
        shuffle=True,
        minval=0,
        maxval=10,
        random_state=1308419541,
    )

    df = pd.DataFrame(
        biclusters,
        index=np.arange(0, biclusters.shape[0], step=1),
        columns=np.arange(0, biclusters.shape[1], step=1),
    )

    return df
