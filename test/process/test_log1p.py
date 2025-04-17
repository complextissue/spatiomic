"""Test the Log1p class."""

import numpy as np
import pytest
from numpy.typing import NDArray

import spatiomic as so


@pytest.mark.cpu
def test_log1p_cpu(example_data_unclipped_positive: NDArray) -> None:
    """Test the Log1p class."""
    processer = so.process.log1p(use_gpu=False)

    test_data_log1p_transformed = processer.fit_transform(
        example_data_unclipped_positive,
    )

    np.testing.assert_array_almost_equal(
        test_data_log1p_transformed,
        np.log1p(example_data_unclipped_positive.reshape(-1, example_data_unclipped_positive.shape[-1])).reshape(
            example_data_unclipped_positive.shape
        ),
        decimal=4,
    )

    np.testing.assert_array_almost_equal(
        example_data_unclipped_positive,
        processer.inverse_transform(test_data_log1p_transformed),
        decimal=4,
    )
