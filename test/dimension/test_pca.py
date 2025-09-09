"""Test the Pca class."""

from typing import List, Literal

import pytest
from numpy.typing import NDArray

import spatiomic as so


@pytest.mark.cpu
def test_pca(example_data: NDArray) -> None:
    """Test the Pca class."""
    # reduce to three dimensions
    dimension_count = 3
    flavors: List[Literal["full", "incremental", "auto", "nmf"]] = ["full", "incremental", "auto", "nmf"]

    for batch_size in [None, 40]:
        for flavor in flavors:
            pca_estimator = so.dimension.pca(
                dimension_count=dimension_count,
                batch_size=batch_size,
                use_gpu=False,
                flavor=flavor,
            )

            data_pca = pca_estimator.fit_transform(example_data)
            assert data_pca.shape[-1] == dimension_count

            if flavor == "nmf":
                continue

            assert pca_estimator.get_explained_variance_ratio().shape[0] == dimension_count
