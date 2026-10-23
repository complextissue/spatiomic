"""Test Som class."""

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

import spatiomic as so

NODE_COUNT = (3, 4)


def fit_som(
    data: NDArray,
    distance_metric: Literal["euclidean", "manhattan", "cosine", "correlation"] = "euclidean",
    initialization: Literal["random", "sample", "pca"] = "random",
    iteration_count: int = 5,
) -> so.dimension.som:
    """Fit a small SOM on the CPU."""
    data_som = so.dimension.som(
        node_count=NODE_COUNT,
        dimension_count=data.shape[-1],
        distance_metric=distance_metric,
        initialization=initialization,
        use_gpu=False,
        seed=0,
    )
    data_som.fit(data, iteration_count=iteration_count)
    return data_som


@pytest.mark.cpu
@pytest.mark.parametrize("initialization", ["random", "sample", "pca"])
@pytest.mark.parametrize(
    ("distance_metric", "scipy_metric"),
    [("euclidean", "euclidean"), ("manhattan", "cityblock"), ("cosine", "cosine"), ("correlation", "correlation")],
)
def test_som_cpu(
    example_data: NDArray,
    distance_metric: Literal["euclidean", "manhattan", "cosine", "correlation"],
    scipy_metric: str,
    initialization: Literal["random", "sample", "pca"],
    tmp_path: Path,
) -> None:
    """Test that the SOM finds the nearest node, its distance and the node labels and survives saving and loading."""
    subsample_data = so.data.subsample().fit_transform(example_data, method="count", count=1000, seed=0)
    data_som = fit_som(subsample_data, distance_metric=distance_metric, initialization=initialization)

    nodes = data_som.get_nodes()
    assert np.array_equal(data_som.get_nodes(flatten=False)[1, 2], nodes[1 * NODE_COUNT[1] + 2])

    data_flat = example_data.reshape(-1, example_data.shape[-1])
    distances_expected = cdist(data_flat, nodes, metric=scipy_metric)
    distances, node_ids = data_som.predict(example_data, return_distance=True)

    np.testing.assert_allclose(distances_expected[np.arange(len(data_flat)), node_ids], distances, atol=1e-5)
    np.testing.assert_allclose(distances_expected.min(axis=1), distances, atol=1e-5)
    assert data_som.get_quantization_error(example_data) == pytest.approx(np.mean(distances), abs=1e-5)

    clusters = np.random.default_rng(0).integers(low=0, high=5, size=np.prod(NODE_COUNT))
    data_labelled, label_distances = data_som.label(example_data, clusters=clusters, return_distance=True)
    assert np.array_equal(data_labelled, clusters[node_ids].reshape(example_data.shape[:-1]))
    assert np.array_equal(label_distances, distances)
    assert np.array_equal(data_som.label(example_data, clusters=clusters, flatten=True), clusters[node_ids])

    save_path = str(tmp_path / "som.p")
    data_som.save(save_path=save_path)
    loaded_som = so.dimension.som(use_gpu=False)

    with pytest.raises(ValueError, match="No SOM"):
        loaded_som.save(save_path=save_path)

    loaded_som.load(save_path)

    assert loaded_som.get_config() == data_som.get_config()
    assert np.array_equal(loaded_som.get_nodes(), nodes)
    assert np.array_equal(loaded_som.predict(example_data), node_ids)


@pytest.mark.cpu
@pytest.mark.parametrize("verbose", [False, True])
def test_som_verbose(example_data: NDArray, verbose: bool, capsys: pytest.CaptureFixture[str]) -> None:
    """Test that fitting shows a progress bar over the epochs only when verbose."""
    data_som = so.dimension.som(
        node_count=NODE_COUNT, dimension_count=example_data.shape[-1], use_gpu=False, seed=0, verbose=verbose
    )
    data_som.fit(example_data, iteration_count=3)

    assert ("3/3" in capsys.readouterr().err) == verbose


def test_som_deterministic(example_data: NDArray) -> None:
    """Test that deterministic fits give identical nodes, also on a GPU."""
    nodes = []
    for _ in range(3):
        data_som = so.dimension.som(
            node_count=NODE_COUNT, dimension_count=example_data.shape[-1], seed=0, deterministic=True
        )
        data_som.fit(example_data, iteration_count=5)
        nodes.append(data_som.get_nodes())

    for other_nodes in nodes[1:]:
        np.testing.assert_array_equal(other_nodes, nodes[0])


@pytest.mark.cpu
@pytest.mark.parametrize("initialization", ["random", "sample", "pca"])
def test_som_training_reduces_quantization_error(
    example_data: NDArray,
    initialization: Literal["random", "sample", "pca"],
) -> None:
    """Test that batch training improves the fit over the initial nodes."""
    untrained_som = fit_som(example_data, initialization=initialization, iteration_count=0)
    trained_som = fit_som(example_data, initialization=initialization, iteration_count=5)

    assert trained_som.get_quantization_error(example_data) < untrained_som.get_quantization_error(example_data)


@pytest.mark.cpu
def test_som_pca_initialization(example_data: NDArray) -> None:
    """Test that PCA initialisation centres the nodes on the data mean within the plane of the first two PCs."""
    data_flat = example_data.reshape(-1, example_data.shape[-1])
    nodes = fit_som(example_data, initialization="pca", iteration_count=0).get_nodes()

    np.testing.assert_allclose(nodes.mean(axis=0), data_flat.mean(axis=0), atol=1e-5)

    _, _, components = np.linalg.svd(data_flat - data_flat.mean(axis=0), full_matrices=False)
    node_offsets = nodes - data_flat.mean(axis=0)
    np.testing.assert_allclose(node_offsets @ components[:2].T @ components[:2], node_offsets, atol=1e-5)


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
def test_som_invalid_input(example_data: NDArray) -> None:
    """Test that unfitted SOMs, unsupported metrics and mismatched inputs raise informative errors."""
    unfitted_som = so.dimension.som(node_count=NODE_COUNT, dimension_count=example_data.shape[-1], use_gpu=False)
    with pytest.raises(ValueError, match="No SOM"):
        unfitted_som.get_nodes()

    with pytest.raises(ValueError, match="channels"):
        unfitted_som.fit(example_data[..., :2])

    with pytest.raises(ValueError, match="channels"):
        unfitted_som.set_nodes(np.zeros((np.prod(NODE_COUNT), 2)))

    unsupported_som = so.dimension.som(
        dimension_count=example_data.shape[-1],
        distance_metric="chebyshev",  # type: ignore
        use_gpu=False,
    )
    with pytest.raises(ValueError, match="metric"):
        unsupported_som.fit(example_data, iteration_count=1)

    data_som = fit_som(example_data, iteration_count=1)
    with pytest.raises(ValueError, match="same number of elements"):
        data_som.label(example_data, clusters=[0, 1])

    with pytest.raises(ValueError, match=r"\.npy"):
        data_som.label(example_data, clusters=np.zeros(np.prod(NODE_COUNT)), save_path="labels.txt")


@pytest.mark.gpu
def test_som_cupy_input(example_data: NDArray) -> None:
    """Test that CuPy data gives the same nearest nodes and labels as NumPy data."""
    cupy = pytest.importorskip("cupy")
    data_som = so.dimension.som(node_count=NODE_COUNT, dimension_count=example_data.shape[-1], seed=0)
    data_som.fit(cupy.asarray(example_data), iteration_count=5)

    distances, node_ids = data_som.predict(cupy.asarray(example_data), return_distance=True)
    distances_numpy, node_ids_numpy = data_som.predict(example_data, return_distance=True)

    assert np.array_equal(node_ids, node_ids_numpy)
    np.testing.assert_allclose(distances, distances_numpy)

    clusters = np.arange(np.prod(NODE_COUNT))
    assert np.array_equal(
        data_som.label(cupy.asarray(example_data), clusters=clusters), data_som.label(example_data, clusters=clusters)
    )
