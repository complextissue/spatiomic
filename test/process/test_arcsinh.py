"""Test the Arcsinh class."""

import numpy as np
import pytest
from numpy.typing import NDArray

import spatiomic as so


@pytest.mark.cpu
def test_arcsinh_cpu(example_data_unclipped_positive: NDArray) -> None:
    """Test the Arcsinh class."""
    processer = so.process.arcsinh(use_gpu=False)

    test_data_arcsinh_transformed = processer.fit_transform(
        example_data_unclipped_positive,
    )

    np.testing.assert_array_almost_equal(
        test_data_arcsinh_transformed,
        np.arcsinh(example_data_unclipped_positive.reshape(-1, example_data_unclipped_positive.shape[-1])).reshape(
            example_data_unclipped_positive.shape
        ),
        decimal=4,
    )

    np.testing.assert_array_almost_equal(
        example_data_unclipped_positive,
        processer.inverse_transform(test_data_arcsinh_transformed),
        decimal=4,
    )
