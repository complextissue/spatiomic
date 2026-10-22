"""Test the KNN graph construction function."""

from typing import Literal

import networkx as nx
import numpy as np
import pytest

import spatiomic as so


@pytest.mark.cpu
@pytest.mark.parametrize("neighbor_count", [2, 5, 10])
@pytest.mark.parametrize("method", ["simple", "batch_balanced"])
@pytest.mark.parametrize("accuracy", ["fast", "accurate"])
@pytest.mark.parametrize("distance_metric", ["euclidean", "manhattan", "correlation", "cosine"])
def test_knn_graph(
    neighbor_count: int,
    method: Literal["simple", "batch_balanced"],
    accuracy: Literal["fast", "accurate"],
    distance_metric: Literal["euclidean", "manhattan", "correlation", "cosine"],
) -> None:
    """Test that every node has at least `neighbor_count` neighbors, in each batch for the batch-balanced method."""
    rng = np.random.default_rng(0)
    example_data = rng.integers(size=(20, 20, 3), low=0, high=200)
    example_data_batch = rng.integers(size=400, low=0, high=2)

    knn_graph = so.neighbor.knn_graph.create(
        data=example_data,
        batch=example_data_batch,
        neighbor_count=neighbor_count,
        method=method,
        distance_metric=distance_metric,
        accuracy=accuracy,
        use_gpu=False,
    )

    assert isinstance(knn_graph, nx.Graph)
    assert list(knn_graph.nodes) == list(range(example_data_batch.shape[0]))

    batch_ids = np.unique(example_data_batch) if method == "batch_balanced" else [None]
    for node in knn_graph.nodes:
        neighbor_batches = example_data_batch[list(knn_graph.neighbors(node))]
        for batch_id in batch_ids:
            batch_neighbor_count = len(neighbor_batches) if batch_id is None else np.sum(neighbor_batches == batch_id)
            assert batch_neighbor_count >= neighbor_count
