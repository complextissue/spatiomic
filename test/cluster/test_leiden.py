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

    _, modularity_single_iteration = so.cluster.leiden().predict(knn_graph, iteration_count=1, use_gpu=False)
    assert modularity_single_iteration <= modularity + 1e-12


def count_misplaced_nodes(graph: nx.Graph, communities: np.ndarray, resolution: float = 1.0) -> int:
    """Count the nodes whose move to a neighboring or an empty community would increase the modularity."""
    degrees = dict(graph.degree(weight="weight"))
    total_degree = sum(degrees.values())
    community_degrees = np.bincount(communities, weights=[degrees[node] for node in graph.nodes])

    misplaced_count = 0
    for node in graph.nodes:
        links: dict[int, float] = {}
        for neighbor, edge in graph[node].items():
            links[communities[neighbor]] = links.get(communities[neighbor], 0.0) + edge.get("weight", 1.0)

        def gain(community: int, own_degree: float, node: int = node, links: dict[int, float] = links) -> float:
            return links.get(community, 0.0) - (
                resolution * degrees[node] * (community_degrees[community] - own_degree) / total_degree
            )

        own_community = communities[node]
        stay_gain = gain(own_community, degrees[node])
        move_gains = [gain(community, 0.0) for community in links if community != own_community]
        misplaced_count += max([0.0, *move_gains]) > stay_gain + 1e-9
    return misplaced_count


@pytest.mark.cpu
def test_leiden_converges() -> None:
    """Test that iterating without a limit stops only when no node can increase the modularity by moving."""
    graph = nx.LFR_benchmark_graph(500, 2.5, 1.5, 0.5, average_degree=15, min_community=30, max_degree=50, seed=3)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    communities, modularity = so.cluster.leiden().predict(nx.Graph(graph), iteration_count=None, seed=0, use_gpu=False)
    _, modularity_single_iteration = so.cluster.leiden().predict(
        nx.Graph(graph), iteration_count=1, seed=0, use_gpu=False
    )

    assert count_misplaced_nodes(graph, communities) == 0
    assert modularity >= modularity_single_iteration


@pytest.mark.cpu
def test_leiden_isolated_node() -> None:
    """Test that every node, including isolated ones, gets a community in the order of the nodes."""
    graph = so.neighbor.knn_graph.create_graph([(0, 1), (1, 2), (0, 2), (3, 4)], node_count=6)

    communities, _ = so.cluster.leiden().predict(graph, use_gpu=False)

    assert len(communities) == 6
    assert communities[0] == communities[1] == communities[2]
    assert communities[3] == communities[4] != communities[0]
    assert communities[5] not in communities[:5]

    communities, modularity = so.cluster.leiden().predict(nx.empty_graph(3), use_gpu=False)
    np.testing.assert_array_equal(communities, [0, 1, 2])
    assert modularity == 0.0
