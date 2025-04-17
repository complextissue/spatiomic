"""Test the ZScore class."""

import numpy as np
import pytest
from numpy.typing import NDArray

import spatiomic as so


@pytest.mark.cpu
def test_zscore_cpu(example_data: NDArray) -> None:
    """Test the ZScore class."""
    for fraction in [0.1, 1.0]:
        zscorer = so.process.zscore(use_gpu=False)

        example_data_zscored = zscorer.fit_transform(
            example_data,
            fraction=fraction,
        )

        assert example_data_zscored.shape == example_data.shape
        assert isinstance(zscorer.mean, np.ndarray)
        assert isinstance(zscorer.std, np.ndarray)

        if fraction == 1.0:
            np.testing.assert_array_almost_equal(
                zscorer.mean,
                np.mean(example_data.reshape(-1, example_data.shape[-1]), axis=0),
                decimal=1,
            )

            np.testing.assert_array_almost_equal(
                zscorer.std,
                np.std(example_data.reshape(-1, example_data.shape[-1]), axis=0),
                decimal=1,
            )
