"""Test the Clip class."""

import numpy as np
import pytest
from numpy.typing import NDArray

import spatiomic as so


def get_expected_clip(
    data: NDArray,
    min_value: float | NDArray | None,
    max_value: float | NDArray | None,
    fill_value: float | None,
) -> NDArray:
    """Clip the data or replace values outside of the range with the fill value."""
    if fill_value is None:
        return np.clip(data, a_min=min_value, a_max=max_value)

    outside = np.zeros(data.shape, dtype=bool)
    if min_value is not None:
        outside |= data < min_value
    if max_value is not None:
        outside |= data > max_value
    return np.where(outside, fill_value, data)


@pytest.mark.cpu
@pytest.mark.parametrize("fitting_reference_name", ["example_data_unclipped", "example_data_unclipped_flat"])
@pytest.mark.parametrize("fraction", [0.1, 1.0])
@pytest.mark.parametrize("fill_value", [None, -1.0, 0.0])
def test_clip_percentile_cpu(
    request: pytest.FixtureRequest,
    example_data_unclipped: NDArray,
    fitting_reference_name: str,
    fraction: float,
    fill_value: float | None,
) -> None:
    """Test that percentile clipping learns the channel-wise percentiles of the reference and applies them."""
    fitting_reference = request.getfixturevalue(fitting_reference_name)
    reference_flat = fitting_reference.reshape(-1, fitting_reference.shape[-1])

    clipper = so.process.clip(
        method="percentile",
        percentile_min=2,
        percentile_max=98,
        fill_value=fill_value,
        use_gpu=False,
    )
    clipper.fit(fitting_reference, fraction=fraction)

    min_value = np.percentile(reference_flat, 2, axis=0)
    max_value = np.percentile(reference_flat, 98, axis=0)
    # a subsample only estimates the percentiles, the data spans a range of 500
    tolerance = 1e-3 if fraction == 1.0 else 50
    np.testing.assert_allclose(clipper.percentile_min_absolute.ravel(), min_value, atol=tolerance)  # type: ignore
    np.testing.assert_allclose(clipper.percentile_max_absolute.ravel(), max_value, atol=tolerance)  # type: ignore

    np.testing.assert_allclose(
        clipper.transform(example_data_unclipped),
        get_expected_clip(
            example_data_unclipped,
            min_value=clipper.percentile_min_absolute,
            max_value=clipper.percentile_max_absolute,
            fill_value=fill_value,
        ),
        atol=1e-3,
    )


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("min_value", "max_value"),
    [(0.1, 0.8), (None, 0.8), (0.1, None), (np.linspace(0.0, 0.1, 20), np.linspace(0.8, 0.9, 20))],
    ids=["both", "max_only", "min_only", "per_channel"],
)
@pytest.mark.parametrize("fill_value", [None, -1.0, 0.0])
def test_clip_minmax_cpu(
    example_data_unclipped: NDArray,
    min_value: float | NDArray | None,
    max_value: float | NDArray | None,
    fill_value: float | None,
) -> None:
    """Test that min-max clipping clips to or fills outside of the given channel-wise range."""
    clipper = so.process.clip(
        method="minmax",
        min_value=min_value,
        max_value=max_value,
        fill_value=fill_value,
        use_gpu=False,
    )

    np.testing.assert_allclose(
        clipper.fit_transform(example_data_unclipped),
        get_expected_clip(example_data_unclipped, min_value=min_value, max_value=max_value, fill_value=fill_value),
        atol=1e-6,
    )
