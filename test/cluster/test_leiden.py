"""Test the Leiden graph clustering implementation."""

import networkx as nx
import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

import spatiomic as so


@pytest.mark.cpu
@pytest.mark.parametrize("iteration_count", [None, 10])
def test_leiden(iteration_count: int | None) -> None:
    """Test that Leiden clustering of a kNN graph recovers well-separated blobs and reports their modularity."""
    data, labels = make_blobs(n_samples=300, centers=5, n_features=10, cluster_std=1.0, random_state=0)
    knn_graph = so.neighbor.knn_graph.create(data, neighbor_count=10, use_gpu=False)

    leiden = so.cluster.leiden()
    communities, modularity = leiden.predict(
        knn_graph,
        resolution=1.0,
        iteration_count=iteration_count,
        seed=0,
        use_gpu=False,
    )

    assert leiden.graph is knn_graph
    assert adjusted_rand_score(labels, communities) > 0.95
    assert modularity == pytest.approx(
        nx.community.modularity(knn_graph, [set(np.flatnonzero(communities == c)) for c in np.unique(communities)])
    )

    communities_repeated, _ = so.cluster.leiden().predict(knn_graph, iteration_count=iteration_count, use_gpu=False)
    np.testing.assert_array_equal(communities, communities_repeated)

    _, modularity_single_level = so.cluster.leiden().predict(knn_graph, iteration_count=1, use_gpu=False)
    assert modularity_single_level <= modularity + 1e-12


@pytest.mark.cpu
def test_leiden_isolated_node() -> None:
    """Test that every node, including isolated ones, gets a community in the order of the nodes."""
    graph = so.neighbor.knn_graph.create_graph([(0, 1), (1, 2), (0, 2), (3, 4)], node_count=6)

    communities, _ = so.cluster.leiden().predict(graph, use_gpu=False)

    assert len(communities) == 6
    assert communities[0] == communities[1] == communities[2]
    assert communities[3] == communities[4] != communities[0]
    assert communities[5] not in communities[:5]
