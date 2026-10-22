"""Test the Pca class."""

from typing import Literal

import pytest
from numpy.typing import NDArray

import spatiomic as so


@pytest.mark.cpu
@pytest.mark.parametrize("batch_size", [None, 40])
@pytest.mark.parametrize("flavor", ["full", "incremental", "auto", "nmf"])
def test_pca(
    example_data: NDArray,
    batch_size: int | None,
    flavor: Literal["full", "incremental", "auto", "nmf"],
) -> None:
    """Test the Pca class."""
    # reduce to three dimensions
    dimension_count = 3

    pca_estimator = so.dimension.pca(
        dimension_count=dimension_count,
        batch_size=batch_size,
        use_gpu=False,
        flavor=flavor,
    )

    data_pca = pca_estimator.fit_transform(example_data)
    assert data_pca.shape[-1] == dimension_count

    if flavor == "nmf":
        return

    assert pca_estimator.get_explained_variance_ratio().shape[0] == dimension_count
