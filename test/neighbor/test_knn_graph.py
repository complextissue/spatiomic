"""Test the KNN graph construction function."""

from typing import List, Literal

import numpy as np
import pytest
from igraph import Graph

import spatiomic as so


@pytest.mark.cpu
def test_knn_graph() -> None:
    """Test KNN graph construction function."""
    example_data = np.random.randint(size=(20, 20, 3), low=0, high=200)
    example_data_batch = np.random.randint(size=400, low=0, high=1)
    distance_metrics: List[Literal["euclidean", "manhattan", "correlation", "cosine"]] = [
        "euclidean",
        "manhattan",
        "correlation",
        "cosine",
    ]
    methods: List[Literal["simple", "batch_balanced"]] = ["simple", "batch_balanced"]
    accuracies: List[Literal["fast", "accurate"]] = ["fast", "accurate"]

    for method in methods:
        for accuracy in accuracies:
            for distance_metric in distance_metrics:
                for neighbor_count in [2, 5, 10]:
                    knn_graph = so.neighbor.knn_graph.create(
                        data=example_data,
                        batch=example_data_batch,
                        neighbor_count=neighbor_count,
                        method=method,
                        distance_metric=distance_metric,
                        accuracy=accuracy,
                    )

                    assert isinstance(knn_graph, Graph)

                    # Check neighbor count for each node (may vary since the graph is undirected)
                    for node in knn_graph.vs:
                        assert len(knn_graph.neighbors(node)) >= neighbor_count


if __name__ == "__main__":
    test_knn_graph()
