"""Test the Clip class."""

from typing import List, Literal

import numpy as np
import pytest
from numpy.typing import NDArray

import spatiomic as so


@pytest.mark.cpu
def test_clip_cpu(
    example_data_unclipped: NDArray,
    example_data_unclipped_flat: NDArray,
) -> None:
    """Test the Clip class."""
    example_data_unclipped = np.copy(example_data_unclipped)

    methods: List[Literal["percentile", "minmax"]] = ["minmax", "percentile"]

    min_values = np.random.rand(20) * 0.1
    max_values = np.random.rand(20) * 0.1 + 0.8

    combinations = [
        (0.1, 0.8),
        (None, 0.8),
        (0.1, None),
        (min_values, max_values),
    ]

    for method in methods:
        for fitting_reference in [example_data_unclipped, example_data_unclipped_flat]:
            for fraction in [0.1, 1.0]:
                for fill_value in [None, -1.0]:
                    for min_value, max_value in combinations:
                        clipper = so.process.clip(
                            method=method,
                            percentile_min=2,
                            percentile_max=98,
                            use_gpu=False,
                            fill_value=fill_value,
                            min_value=min_value,  # type: ignore
                            max_value=max_value,  # type: ignore
                        )

                        clipper.fit(
                            fitting_reference,
                            fraction=fraction,
                        )

                        test_data_clipped = clipper.transform(
                            example_data_unclipped,
                        )

                        if fraction == 1.0 and method == "percentile" and fill_value is None:
                            np.testing.assert_array_almost_equal(
                                test_data_clipped,
                                np.clip(
                                    example_data_unclipped,
                                    a_min=np.expand_dims(
                                        np.percentile(
                                            fitting_reference.reshape(-1, fitting_reference.shape[-1]),
                                            2,
                                            axis=0,
                                        ),
                                        axis=0,
                                    ),
                                    a_max=np.expand_dims(
                                        np.percentile(
                                            fitting_reference.reshape(-1, fitting_reference.shape[-1]),
                                            98,
                                            axis=0,
                                        ),
                                        axis=0,
                                    ),
                                ),
                                decimal=3,
                            )
                        elif method == "minmax" and fill_value is None:
                            np.testing.assert_array_almost_equal(
                                test_data_clipped,
                                np.clip(
                                    example_data_unclipped,
                                    a_min=np.expand_dims(min_value, axis=0) if min_value is not None else None,
                                    a_max=np.expand_dims(max_value, axis=0) if max_value is not None else None,
                                ),
                                decimal=3,
                            )
                        elif fill_value is not None and method == "percentile":
                            # everywhere where we have the max or min value, we should have the fill value
                            np.testing.assert_array_almost_equal(
                                np.where(
                                    test_data_clipped
                                    <= np.min(fitting_reference.reshape(-1, fitting_reference.shape[-1]), axis=0),
                                    fill_value,
                                    test_data_clipped,
                                ),
                                test_data_clipped,
                                decimal=3,
                            )
                            np.testing.assert_array_almost_equal(
                                np.where(
                                    test_data_clipped
                                    >= np.max(fitting_reference.reshape(-1, fitting_reference.shape[-1]), axis=0),
                                    fill_value,
                                    test_data_clipped,
                                ),
                                test_data_clipped,
                                decimal=3,
                            )
