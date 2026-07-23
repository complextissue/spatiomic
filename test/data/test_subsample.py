"""Test the Subsample class."""

import numpy as np
import pytest
from anndata import AnnData
from numpy.typing import NDArray

from spatiomic.data import subsample


def test_subsample(
    image_pixels: NDArray,
) -> None:
    """Test the Subsample class.

    Args:
        image_pixels (NDArray): The example image.
    """
    image_pixels = image_pixels.reshape((-1, image_pixels.shape[-1]))
    subsample_pixels = subsample.fit_transform(image_pixels, fraction=0.1)

    assert subsample_pixels.shape[0] == int(0.1 * image_pixels.shape[0])

    # check with AnnData
    adata = AnnData(image_pixels)
    _ = subsample.fit_transform(adata, fraction=0.1, output_unstructured_name="X_subsample")

    assert adata.uns["X_subsample"].shape[0] == int(0.1 * image_pixels.shape[0])


def test_subsample_without_replacement() -> None:
    """Subsampling must not duplicate pixels (sampling without replacement)."""
    # Each pixel has a unique value so any duplication or omission is detectable.
    pixels = np.arange(200, dtype=np.float32).reshape((-1, 1))

    result = subsample.fit_transform(pixels, method="count", count=200, seed=0)

    assert result.shape[0] == 200
    assert np.array_equal(np.sort(result.ravel()), np.arange(200))


def test_subsample_count_exceeds_pixels_is_clamped() -> None:
    """Requesting more pixels than available clamps to all pixels (with a warning), without replacement."""
    pixels = np.arange(50, dtype=np.float32).reshape((-1, 1))

    with pytest.warns(UserWarning):
        result = subsample.fit_transform(pixels, method="count", count=1000, seed=0)

    assert result.shape[0] == 50
    assert np.array_equal(np.sort(result.ravel()), np.arange(50))
