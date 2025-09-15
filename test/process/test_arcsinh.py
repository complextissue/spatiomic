"""Test the Arcsinh class."""

import numpy as np
import pytest
from numpy.typing import NDArray

import spatiomic as so


@pytest.mark.cpu
def test_arcsinh_cpu(example_data_unclipped_positive: NDArray) -> None:
    """Test the Arcsinh class."""
    processer = so.process.arcsinh(use_gpu=False)

    for cofactors in [1, 2, np.random.rand(example_data_unclipped_positive.shape[-1]) * 10 + 1]:
        test_data_arcsinh_transformed = processer.fit_transform(
            example_data_unclipped_positive,
            cofactors=cofactors,
        )

        np.testing.assert_allclose(
            test_data_arcsinh_transformed,
            np.arcsinh(
                (example_data_unclipped_positive.reshape(-1, example_data_unclipped_positive.shape[-1])) / cofactors
            ).reshape(example_data_unclipped_positive.shape),
            rtol=1e-6,
            atol=1e-3,
        )

        np.testing.assert_allclose(
            example_data_unclipped_positive,
            processer.inverse_transform(test_data_arcsinh_transformed, cofactors=cofactors),
            rtol=1e-6,
            atol=1e-3,
        )
