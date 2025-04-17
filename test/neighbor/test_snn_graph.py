"""Test the creation of a shared nearest neighbor graph."""

import pytest
from igraph import Graph

import spatiomic as so


@pytest.fixture
def knn_graph() -> Graph:
    """Create a simple k-nearest neighbor graph for testing."""
    g = Graph()
    g.add_vertices(5)
    g.add_edges([(0, 1), (0, 2), (1, 2), (3, 4)])
    return g


def test_snn_graph_creation(
    knn_graph: Graph,
) -> None:
    """Test the creation of a shared nearest neighbor graph.

    Args:
        knn_graph (Graph): A k-nearest neighbor graph.
    """
    shared_count_min_values = [1, 2]
    fix_lonely_nodes_values = [True, False]

    for shared_count_min, fix_lonely_nodes in zip(shared_count_min_values, fix_lonely_nodes_values, strict=True):
        snn_graph = so.neighbor.snn_graph.create(
            knn_graph,
            shared_count_min,
            fix_lonely_nodes,
            fix_lonely_nodes_method="random",
        )

        assert isinstance(snn_graph, Graph)

        # test whether lonely nodes are connected to their nearest neighbors if fix_lonely_nodes is True
        if fix_lonely_nodes:
            for node in snn_graph.vs.indices:
                neighbors = snn_graph.neighbors(node)

                # all nodes should have at least one neighbor
                assert len(neighbors) > 0
        else:
            # test whether shared_count_min is satisfied for connected nodes
            for edge in snn_graph.get_edgelist():
                node_1, node_2 = edge
                neighbors_1 = set(knn_graph.neighbors(node_1))
                neighbors_2 = set(knn_graph.neighbors(node_2))
                shared_neighbors = neighbors_1 & neighbors_2
                assert len(shared_neighbors) >= shared_count_min
