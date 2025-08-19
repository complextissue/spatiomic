"""Test the extend_mask functionality."""

import numpy as np
import pytest
from numpy.typing import NDArray

import spatiomic as so


@pytest.mark.cpu
def test_extend_mask_basic(basic_masks: NDArray) -> None:
    """Test basic extend_mask functionality."""
    extended = so.segment.extend_mask(basic_masks, dilation_pixels=1)

    # Check that original non-zero pixels are preserved
    assert np.all(extended[basic_masks != 0] == basic_masks[basic_masks != 0])

    # Check that some background pixels were assigned
    assert np.sum(extended != 0) > np.sum(basic_masks != 0)


@pytest.mark.cpu
def test_extend_mask_no_background() -> None:
    """Test extend_mask with no background pixels."""
    masks = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)

    extended = so.segment.extend_mask(masks, dilation_pixels=1)

    # Should be unchanged since no background
    np.testing.assert_array_equal(masks, extended)


@pytest.mark.cpu
def test_extend_mask_custom_background() -> None:
    """Test extend_mask with custom background label."""
    masks = np.array(
        [
            [9, 9, 9, 9, 9],
            [9, 1, 9, 2, 9],
            [9, 9, 9, 9, 9],
        ],
        dtype=np.int32,
    )

    extended = so.segment.extend_mask(masks, dilation_pixels=1, background_label=9)

    # Check that original non-background pixels are preserved
    assert np.all(extended[masks != 9] == masks[masks != 9])
