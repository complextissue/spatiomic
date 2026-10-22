"""Test Som class."""

import os
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

import spatiomic as so


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("distance_metric", "scipy_metric"),
    [("euclidean", "euclidean"), ("manhattan", "cityblock"), ("cosine", "cosine"), ("correlation", "correlation")],
)
def test_som_cpu(
    example_data: NDArray,
    distance_metric: Literal["euclidean", "manhattan", "cosine", "correlation"],
    scipy_metric: str,
    tmp_path: str,
) -> None:
    """Test that the SOM finds the nearest node, its distance and the node labels and survives saving and loading."""
    node_count = (3, 4)
    data_som = so.dimension.som(
        node_count=node_count,
        dimension_count=example_data.shape[-1],
        distance_metric=distance_metric,
        use_gpu=False,
        seed=0,
    )

    subsample_data = so.data.subsample().fit_transform(example_data, method="count", count=1000, seed=0)
    data_som.fit(subsample_data, iteration_count=5)

    nodes = data_som.get_nodes()
    assert np.array_equal(data_som.get_nodes(flatten=False)[1, 2], nodes[1 * node_count[1] + 2])

    data_flat = example_data.reshape(-1, example_data.shape[-1])
    distances_expected = cdist(data_flat, nodes, metric=scipy_metric)
    distances, node_ids = data_som.predict(example_data, return_distance=True)

    np.testing.assert_allclose(distances_expected[np.arange(len(data_flat)), node_ids], distances, atol=1e-5)
    np.testing.assert_allclose(distances_expected.min(axis=1), distances, atol=1e-5)
    assert data_som.get_quantization_error(example_data) == pytest.approx(np.mean(distances), abs=1e-5)

    clusters = np.random.default_rng(0).integers(low=0, high=5, size=np.prod(node_count))
    data_labelled = data_som.label(example_data, clusters=clusters)
    assert np.array_equal(data_labelled, clusters[node_ids].reshape(example_data.shape[:-1]))

    save_path = os.path.join(tmp_path, "som.p")
    data_som.save(save_path=save_path)
    loaded_som = so.dimension.som(use_gpu=False)

    with pytest.raises(ValueError, match="No SOM"):
        loaded_som.save(save_path=save_path)

    loaded_som.load(save_path)

    assert loaded_som.get_config() == data_som.get_config()
    assert np.array_equal(loaded_som.get_nodes(), nodes)
    assert np.array_equal(loaded_som.predict(example_data), node_ids)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("distance_metric", "data_point_without_node"),
    [("cosine", [0.0, 0.0]), ("correlation", [2.0, 2.0])],
)
def test_som_data_point_without_node(
    distance_metric: Literal["cosine", "correlation"],
    data_point_without_node: list[float],
) -> None:
    """Test that zero-norm (cosine) and constant (correlation) data points have no nearest node."""
    data_som = so.dimension.som(node_count=(1, 2), dimension_count=2, distance_metric=distance_metric, use_gpu=False)
    data_som.set_nodes(np.array([[1.0, 0.0], [0.0, 1.0]]))

    data = np.array([data_point_without_node, [0.0, 2.0]])
    distances, node_ids = data_som.predict(data, return_distance=True)

    assert np.array_equal(node_ids, [-1, 1])
    assert np.isnan(distances[0]) and distances[1] == pytest.approx(0.0, abs=1e-6)
    assert np.array_equal(data_som.label(data, clusters=[3, 4]), [-1, 4])
    assert data_som.get_quantization_error(data) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.cpu
def test_som_distance_map() -> None:
    """Test that the distance map sums the distances to the grid neighbors and is normalized to a maximum of 1."""
    data_som = so.dimension.som(node_count=(1, 3), dimension_count=1, use_gpu=False)
    data_som.set_nodes(np.array([[0.0], [1.0], [3.0]]))

    np.testing.assert_allclose(data_som.get_distance_map(), [[1 / 3, 1.0, 2 / 3]])


@pytest.mark.cpu
def test_som_unsupported_metric(example_data: NDArray) -> None:
    """Test that distance metrics that fastersom does not support raise an error."""
    data_som = so.dimension.som(dimension_count=example_data.shape[-1], distance_metric="chebyshev", use_gpu=False)  # type: ignore

    with pytest.raises(ValueError, match="metric"):
        data_som.fit(example_data, iteration_count=1)
