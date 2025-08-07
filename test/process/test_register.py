"""Test the registration class."""

import numpy as np
from numpy.typing import NDArray

from spatiomic.process import register


def test_register(
    cycle_pixels_registered: NDArray,
) -> None:
    """Test the registration class."""
    cycle_pixels_shifted = register.apply_shift(
        cycle_pixels_registered[:, :, 0],
        shift=(50, -20),
    )

    shift_tuple = register.get_shift(
        cycle_pixels_shifted,
        cycle_pixels_registered[:, :, 0],
        method="phase_correlation",
    )

    shift = np.array(shift_tuple)
    assert shift.shape == (2,)
    assert shift.dtype in [np.float32, np.float64]

    # check if shift is correct, should be (-50, 20)
    assert shift[0] <= -49 and shift[0] >= -51
    assert shift[1] >= 19 and shift[1] <= 21

    cycle_pixels_corrected = register.apply_shift(
        cycle_pixels_shifted,
        shift,
    )

    mssim_before = register.get_ssim(
        cycle_pixels_registered[:, :, 0],
        cycle_pixels_shifted,
    )

    mssim = register.get_ssim(
        cycle_pixels_registered[:, :, 0],
        cycle_pixels_corrected,
    )

    assert mssim_before < mssim
