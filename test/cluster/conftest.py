"""Data fixtures for cluster module testing."""

import numpy as np
import pandas as pd
import pytest
from igraph import Graph
from sklearn import datasets


@pytest.fixture(scope="session")
def neighborhood_graph() -> Graph:
    """Create an exmaple neighborhood graph."""
    points = np.expand_dims(np.arange(0, 10, step=1), axis=1)
    connections = np.copy(points)

    np.random.shuffle(connections)
    edges = np.append(points, connections, axis=1)

    return Graph(edges)


@pytest.fixture(scope="session")
def cluster_data() -> Graph:
    """Create an exmaple neighborhood graph."""
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
    """Create an exmaple neighborhood graph."""
    biclusters, rows, cols = datasets.make_biclusters(
        shape=(20, 20),
        n_clusters=10,
        noise=0.5,
        shuffle=True,
        minval=0,
        maxval=10,
        random_state=1308419541,
    )

    print(biclusters)

    df = pd.DataFrame(
        biclusters,
        index=np.arange(0, biclusters.shape[0], step=1),
        columns=np.arange(0, biclusters.shape[1], step=1),
    )

    return df
