"""Test the Umap class."""

from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

import spatiomic as so


@pytest.mark.cpu
@pytest.mark.parametrize("dimension_count", [2, 3])
@pytest.mark.parametrize("distance_metric", ["euclidean", "manhattan"])
def test_umap(
    example_data: NDArray,
    dimension_count: int,
    distance_metric: Literal["euclidean", "manhattan"],
) -> None:
    """Test the Umap class."""
    # for faster execution, limit the size
    example_data = example_data[:2, :5, :5, :4]

    umap_estimator = so.dimension.umap(
        dimension_count=dimension_count,
        distance_metric=distance_metric,
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
