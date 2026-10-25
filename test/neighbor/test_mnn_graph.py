"""Test the creation of a mutual k-nearest neighbor graph."""

from typing import Literal

import networkx as nx
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

import spatiomic as so

SCIPY_METRICS = {"euclidean": "euclidean", "manhattan": "cityblock", "correlation": "correlation", "cosine": "cosine"}


def get_nearest_neighbors(data: NDArray, neighbor_count: int, distance_metric: str) -> NDArray:
    """Get the `neighbor_count` nearest other data points of each data point, sorted by distance, by brute force."""
    distances = cdist(data, data, metric=SCIPY_METRICS[distance_metric])
    np.fill_diagonal(distances, np.inf)
    return np.argsort(distances, axis=1, kind="stable")[:, :neighbor_count]


def get_mutual_edges(data: NDArray, neighbor_count: int, distance_metric: str) -> set[tuple[int, int]]:
    """Get the pairs of data points that are among each other's `neighbor_count` nearest neighbors with sets."""
    neighborhoods = [set(neighbors) for neighbors in get_nearest_neighbors(data, neighbor_count, distance_metric)]
    return {
        (node_1, node_2)
        for node_1 in range(len(data))
        for node_2 in range(node_1 + 1, len(data))
        if node_2 in neighborhoods[node_1] and node_1 in neighborhoods[node_2]
    }


def get_edge_set(graph: nx.Graph) -> set[tuple[int, int]]:
    """Get the edges of a graph as sorted node pairs."""
    return {(min(edge), max(edge)) for edge in graph.edges}


@pytest.mark.cpu
@pytest.mark.parametrize("neighbor_count", [1, 5, 15])
@pytest.mark.parametrize("distance_metric", ["euclidean", "manhattan", "correlation", "cosine"])
def test_mnn_graph_matches_brute_force(
    neighbor_count: int,
    distance_metric: Literal["euclidean", "manhattan", "correlation", "cosine"],
) -> None:
    """Test that the edges are exactly the mutual nearest neighbor pairs and that there are no loops."""
    data = np.random.default_rng(0).normal(size=(80, 4))

    mnn_graph = so.neighbor.mnn_graph.create(
        data,
        neighbor_count=neighbor_count,
        distance_metric=distance_metric,
        connect_isolated_nodes=False,
        use_gpu=False,
    )

    assert list(mnn_graph.nodes) == list(range(len(data)))
    assert nx.number_of_selfloops(mnn_graph) == 0
    assert get_edge_set(mnn_graph) == get_mutual_edges(data, neighbor_count, distance_metric)


@pytest.mark.cpu
def test_mnn_graph_separates_distant_group() -> None:
    """Test that a small distant group is connected to the rest in the kNN graph, but not in the MNN graph."""
    rng = np.random.default_rng(0)
    data = np.concatenate([rng.normal(size=(500, 2)), rng.normal(loc=20.0, scale=0.1, size=(4, 2))])
    group = set(range(500, 504))

    knn_graph = so.neighbor.knn_graph.create(data, neighbor_count=15, use_gpu=False)
    mnn_graph = so.neighbor.mnn_graph.create(data, neighbor_count=15, use_gpu=False)

    def get_crossing_edges(graph: nx.Graph) -> list[tuple[int, int]]:
        return [edge for edge in graph.edges if len(group & set(edge)) == 1]

    assert len(get_crossing_edges(knn_graph)) > 0
    assert get_crossing_edges(mnn_graph) == []
    assert nx.is_isomorphic(mnn_graph.subgraph(group), nx.complete_graph(4))


@pytest.mark.cpu
def test_mnn_graph_connect_isolated_nodes() -> None:
    """Test that data points without mutual neighbors keep the edge to their nearest neighbor, and only these edges."""
    rng = np.random.default_rng(0)
    data = np.concatenate([rng.normal(size=(200, 3)), [[10.0, 10.0, 10.0]]])

    mutual_graph = so.neighbor.mnn_graph.create(data, neighbor_count=3, connect_isolated_nodes=False, use_gpu=False)
    connected_graph = so.neighbor.mnn_graph.create(data, neighbor_count=3, connect_isolated_nodes=True, use_gpu=False)

    isolated_nodes = list(nx.isolates(mutual_graph))
    assert len(data) - 1 in isolated_nodes
    assert list(nx.isolates(connected_graph)) == []

    nearest_neighbors = get_nearest_neighbors(data, 1, "euclidean")[:, 0]
    expected_added_edges = {
        (min(node, nearest_neighbors[node]), max(node, nearest_neighbors[node])) for node in isolated_nodes
    }
    assert get_edge_set(connected_graph) == get_edge_set(mutual_graph) | expected_added_edges


@pytest.mark.cpu
def test_mnn_graph_som() -> None:
    """Test that the graph of a fitted SOM connects the mutual nearest neighbors among its nodes."""
    data = np.random.default_rng(0).normal(size=(200, 3)).astype(np.float32)
    data_som = so.dimension.som(node_count=(6, 6), dimension_count=3, seed=0, use_gpu=False)
    data_som.fit(data, iteration_count=5)

    mnn_graph = so.neighbor.mnn_graph.create(data_som, neighbor_count=5, connect_isolated_nodes=False, use_gpu=False)

    assert list(mnn_graph.nodes) == list(range(36))
    assert get_edge_set(mnn_graph) == get_mutual_edges(data_som.get_nodes(), 5, "euclidean")
