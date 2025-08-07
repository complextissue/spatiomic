"""Test the Normalize class."""

import numpy as np
import pytest
from numpy.typing import NDArray

import spatiomic as so


@pytest.mark.cpu
def test_normalize_cpu(example_data_unclipped: NDArray) -> None:
    """Test the Normalize class."""
    example_data = example_data_unclipped

    for min_value, max_value in [(-1, 1), (0, 1)]:
        for clip in [True, False]:
            for fraction in [0.1, 1.0]:
                normalizer = so.process.normalize(
                    min_value=min_value,
                    max_value=max_value,
                    use_gpu=False,
                )

                example_data_normalized = normalizer.fit_transform(
                    example_data,
                    fraction=fraction,
                    clip=clip,
                )

                assert normalizer.fitted
                assert example_data_normalized.shape == example_data.shape

                if fraction == 1.0 or clip is True:
                    np.testing.assert_almost_equal(
                        np.max(example_data_normalized),
                        max_value,
                        decimal=2,
                    )

                    np.testing.assert_almost_equal(
                        np.min(example_data_normalized),
                        min_value,
                        decimal=2,
                    )
