"""Test Som class."""

import os
from typing import List, Literal
from uuid import uuid4

import numpy as np
import pytest
from numpy.typing import NDArray

import spatiomic as so


@pytest.mark.cpu
def test_som_cpu(example_data: NDArray) -> None:
    """Test the Som class."""
    node_count = (3, 3)
    dimension_count = 4
    distance_metrics: List[Literal["euclidean", "manhattan", "correlation", "cosine"]] = [
        "euclidean",
        "manhattan",
        # "correlation",  # Requires the custom implementation available at https://github.com/complextissue/xpysom
        "cosine",
    ]

    for distance_metric in distance_metrics:
        data_som = so.dimension.som(
            node_count=node_count,
            dimension_count=dimension_count,
            distance_metric=distance_metric,  # type: ignore
            use_gpu=False,
            seed=0,
        )

        # test configuration retrieval and setting
        config = data_som.get_config()
        data_som.set_config(**config)

        # ensure that the configuration didn't change through setting and retrieval
        assert data_som.get_config() == config

        subsample_data = so.data.subsample().fit_transform(example_data, method="count", count=100)

        # test fitting with and without pca initialisation
        for pca_init in [True, False]:
            data_som.fit(
                subsample_data,
                iteration_count=1,
                pca_init=pca_init,
            )

        # get the nodes
        for flatten in [True, False]:
            nodes = data_som.get_nodes(flatten=flatten)

            if flatten:
                assert nodes.shape[0] == np.multiply.reduce(np.array(node_count))
            else:
                assert nodes.shape[:2] == node_count

            assert nodes.shape[-1] == dimension_count

        # create random labels for the som nodes
        labels = np.random.randint(
            low=0,
            high=10,
            size=np.multiply.reduce(np.array(node_count)),
        )

        # label data according to the node labels
        data_labelled = data_som.label(
            example_data,
            clusters=labels,
        )

        assert np.max(data_labelled) <= np.max(labels)
        assert np.min(data_labelled) >= np.min(labels)
        assert data_labelled.shape == example_data.shape[:-1]

        # TODO: add test cases for flattening and returning distances

    # test quantization error calculation
    quantization_error = data_som.get_quantization_error(example_data)
    assert isinstance(quantization_error, float)

    if distance_metric in ["correlation", "cosine"]:
        assert quantization_error >= 0.0 and quantization_error <= 1.0, (
            f"Quantization error out of bounds for {distance_metric} distance metric: {quantization_error}"
        )

    # test saving and loading
    temp_file_name = f"{uuid4()}.p"
    temp_file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), temp_file_name)
    data_som.save(save_path=temp_file_name)

    assert os.path.isfile(temp_file_name)

    new_som = so.dimension.som()
    new_som.load(save_path=temp_file_name)

    assert data_som.get_config() == new_som.get_config()
    assert np.all(data_som.get_nodes() == new_som.get_nodes())

    # remove the temp file
    os.remove(temp_file_name)
