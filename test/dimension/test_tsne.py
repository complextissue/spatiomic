"""Test the Tsne class."""

from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

import spatiomic as so


@pytest.mark.cpu
@pytest.mark.parametrize("dimension_count", [2, 3])
@pytest.mark.parametrize("distance_metric", ["euclidean", "manhattan"])
def test_tsne(
    example_data: NDArray,
    dimension_count: int,
    distance_metric: Literal["euclidean", "manhattan"],
) -> None:
    """Test the Tsne class."""
    # for faster execution, limit the size
    example_data = example_data[:2, :20, :20, :5]

    tsne_estimator = so.dimension.tsne(
        dimension_count=dimension_count,
        distance_metric=distance_metric,
        iteration_count=300,
        iteration_count_without_progress=100,
        use_gpu=False,
    )

    data_tsne_flat = tsne_estimator.fit_transform(
        example_data,
        flatten=True,
    )

    data_tsne_shaped = tsne_estimator.fit_transform(
        example_data,
        flatten=False,
    )

    assert data_tsne_flat.shape[-1] == dimension_count
    assert data_tsne_flat.shape[:-1] == np.multiply.reduce(example_data.shape[:-1])

    assert data_tsne_shaped.shape[-1] == dimension_count
    assert data_tsne_shaped.shape[:-1] == example_data.shape[:-1]
