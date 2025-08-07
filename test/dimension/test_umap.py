"""Test the Umap class."""

import numpy as np
import pytest
from numpy.typing import NDArray

import spatiomic as so


@pytest.mark.cpu
def test_umap(example_data: NDArray) -> None:
    """Test the Umap class."""
    # for faster execution, limit the size
    example_data = example_data[:2, :5, :5, :4]

    for dimension_count in [2, 3]:
        for distance_metric in ["euclidean", "manhattan"]:
            umap_estimator = so.dimension.umap(
                dimension_count=dimension_count,
                distance_metric=distance_metric,  # type: ignore
                neighbor_count=10,
                use_gpu=False,
            )

            data_umap_flat = umap_estimator.fit_transform(
                example_data,
                flatten=True,
            )

            data_umap_shaped = umap_estimator.fit_transform(
                example_data,
                flatten=False,
            )

            assert data_umap_flat.shape[-1] == dimension_count
            assert data_umap_flat.shape[:-1] == np.multiply.reduce(example_data.shape[:-1])

            assert data_umap_shaped.shape[-1] == dimension_count
            assert data_umap_shaped.shape[:-1] == example_data.shape[:-1]
