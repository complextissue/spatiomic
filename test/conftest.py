"""Create example data for pytest."""

import os

import numpy as np
import pytest
from numpy.typing import NDArray
from skimage.io import imread

from spatiomic.process import register

np.random.seed(1308419541)

FILE_DIR = os.path.dirname(os.path.realpath(__file__))


def generate_cycle_pixels(
    shift: bool = True,
) -> NDArray:
    """Return pixels for a cycle with 3 channels."""
    example_img = imread(os.path.join(FILE_DIR, "data/test_files/kidney-tissue-fluorescence.tiff"))[0, :, :, :]
    offset: NDArray[np.float32] = np.random.default_rng().integers(size=2, low=5, high=20).astype(np.float32)

    if shift:
        example_img = register.apply_shift(example_img, shift=offset)

    return np.array(example_img[16:496, 16:496, :])


@pytest.fixture(scope="session")
def cycle_pixels() -> NDArray:
    """Return pixels for a cycle with 3 channels."""
    return generate_cycle_pixels()  # type: ignore


@pytest.fixture(scope="session")
def cycle_pixels_registered() -> NDArray:
    """Return pixels for a cycle with 3 channels."""
    return generate_cycle_pixels(shift=False)


@pytest.fixture(scope="session")
def image_pixels() -> NDArray:
    """Return pixels for a cycle with 3 channels."""
    return np.dstack([generate_cycle_pixels() for _ in range(0, 3)])


@pytest.fixture(scope="session")
def image_pixels_registered() -> NDArray:
    """Return pixels for a cycle with 3 channels."""
    return np.dstack([generate_cycle_pixels(shift=False) for _ in range(0, 3)])


@pytest.fixture(scope="session")
def example_data_unclipped() -> NDArray:
    """Create an example data array with 20 channels with values outside the range (0, 255)."""
    test_data = np.random.rand(5, 200, 200, 20) * 500 - 200
    return test_data.astype(np.float32)


@pytest.fixture(scope="session")
def example_data_unclipped_flat() -> NDArray:
    """Create an example data array with 20 channels with values outside the range (0, 255)."""
    test_data = np.random.rand(2000, 20) * 500 - 200
    return test_data.astype(np.float32)


@pytest.fixture(scope="session")
def example_data_unclipped_positive() -> NDArray:
    """Create an example data array with 20 channels with values outside the range (0, 255).

    This data only contains positive values.
    """
    test_data = np.random.rand(5, 200, 200, 20) * 500
    return test_data.astype(np.float32)


@pytest.fixture(scope="session")
def example_data() -> NDArray:
    """Create an example data array with 4 channels."""
    test_data = np.random.rand(5, 50, 50, 4)
    return test_data


@pytest.fixture(scope="session")
def example_data_small() -> NDArray:
    """Create an example data array with 20 channels."""
    test_data = np.random.rand(5, 50, 50, 20)
    return test_data
