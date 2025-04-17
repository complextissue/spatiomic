"""Test the Leiden graph clustering implementation."""

import pytest
from igraph import Graph

import spatiomic as so


@pytest.mark.cpu
def test_leiden(neighborhood_graph: Graph) -> None:
    """Test Leiden clustering.

    Args:
        neighborhood_graph (Graph): The graph to be clustered.
    """
    iteration_count = 10
    resolution = 1.5

    leiden = so.cluster.leiden()
    vertex_cluster = leiden.predict(
        neighborhood_graph,
        resolution=resolution,
        iteration_count=iteration_count,
        use_gpu=False,
    )

    assert len(vertex_cluster) == 2, "Two elements should be returned."
    communities, modularity = vertex_cluster

    assert len(communities) == neighborhood_graph.ecount(), (
        "The number of communities should be equal to the number of edges."
    )
    assert type(modularity) is float, "The modularity should be a float."
    assert isinstance(leiden.graph, Graph), "The graph should be an igraph.Graph."
