"""Test the creation of a shared nearest neighbor graph."""

from typing import Literal

import networkx as nx
import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

import spatiomic as so


@pytest.fixture(scope="module")
def blob_data() -> tuple[NDArray, NDArray]:
    """Create well-separated blobs and their labels."""
    data, labels = make_blobs(n_samples=120, centers=4, n_features=5, cluster_std=1.0, random_state=0)
    return data, labels


def get_jaccard_edges(
    data: NDArray,
    neighbor_count: int,
    distance_metric: str,
    prune_threshold: float,
) -> dict[tuple[int, int], float]:
    """Compute the pruned Jaccard indices of all pairs of neighborhoods (including the point itself) with sets."""
    neighbor_idx = (
        NearestNeighbors(n_neighbors=neighbor_count + 1, metric=distance_metric, algorithm="brute")
        .fit(data)
        .kneighbors(data, return_distance=False)
    )
    neighborhoods = [{*neighbors, node} for node, neighbors in enumerate(neighbor_idx.tolist())]

    edges = {}
    for node_1 in range(len(data)):
        for node_2 in range(node_1 + 1, len(data)):
            shared_count = len(neighborhoods[node_1] & neighborhoods[node_2])
            jaccard = shared_count / len(neighborhoods[node_1] | neighborhoods[node_2])
            if jaccard > prune_threshold:
                edges[(node_1, node_2)] = jaccard
    return edges


@pytest.mark.cpu
@pytest.mark.parametrize("neighbor_count", [5, 15])
@pytest.mark.parametrize("distance_metric", ["euclidean", "manhattan", "correlation", "cosine"])
def test_snn_graph_matches_jaccard(
    blob_data: tuple[NDArray, NDArray],
    neighbor_count: int,
    distance_metric: Literal["euclidean", "manhattan", "correlation", "cosine"],
) -> None:
    """Test that edges and weights are the pruned Jaccard indices of the neighborhoods and that there are no loops."""
    data, _ = blob_data

    snn_graph = so.neighbor.snn_graph.create(
        data,
        neighbor_count=neighbor_count,
        distance_metric=distance_metric,
        connect_isolated_nodes=False,
        use_gpu=False,
    )

    assert list(snn_graph.nodes) == list(range(len(data)))
    assert nx.number_of_selfloops(snn_graph) == 0

    edges_expected = get_jaccard_edges(data, neighbor_count, distance_metric, prune_threshold=1 / 15)
    edges = {tuple(sorted(edge)): weight for *edge, weight in snn_graph.edges(data="weight")}
    assert edges.keys() == edges_expected.keys()
    np.testing.assert_allclose([edges[edge] for edge in edges_expected], list(edges_expected.values()), rtol=1e-6)


@pytest.mark.cpu
def test_snn_graph_connect_isolated_nodes(blob_data: tuple[NDArray, NDArray]) -> None:
    """Test that nodes left without edges by pruning keep their strongest edge, and only these edges are added."""
    data, _ = blob_data
    prune_threshold = 0.6

    pruned_graph = so.neighbor.snn_graph.create(
        data, neighbor_count=5, prune_threshold=prune_threshold, connect_isolated_nodes=False, use_gpu=False
    )
    connected_graph = so.neighbor.snn_graph.create(
        data, neighbor_count=5, prune_threshold=prune_threshold, connect_isolated_nodes=True, use_gpu=False
    )

    isolated_nodes = list(nx.isolates(pruned_graph))
    assert len(isolated_nodes) > 0
    assert list(nx.isolates(connected_graph)) == []

    all_edges = get_jaccard_edges(data, 5, "euclidean", prune_threshold=0.0)
    for node in isolated_nodes:
        strongest_weight = max(weight for edge, weight in all_edges.items() if node in edge)
        node_weights = [weight for *_, weight in connected_graph.edges(node, data="weight")]
        assert max(node_weights) == pytest.approx(strongest_weight)

    added_edges = set(map(frozenset, connected_graph.edges)) - set(map(frozenset, pruned_graph.edges))
    assert all(edge & set(isolated_nodes) for edge in added_edges)


@pytest.mark.cpu
def test_snn_graph_leiden(blob_data: tuple[NDArray, NDArray]) -> None:
    """Test that Leiden clustering of the weighted SNN graph recovers the blobs."""
    data, labels = blob_data

    snn_graph = so.neighbor.snn_graph.create(data, neighbor_count=10, use_gpu=False)
    communities, _ = so.cluster.leiden().predict(snn_graph, seed=0, use_gpu=False)

    assert adjusted_rand_score(labels, communities) > 0.95
