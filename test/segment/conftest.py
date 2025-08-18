"""Data fixtures for segment module testing."""

import numpy as np
import pytest
from skimage.draw import disk


@pytest.fixture(scope="module")
def sample_2d_image() -> np.ndarray:
    """Create a simple 2D test image with circular objects."""
    image = np.zeros((100, 100), dtype=np.float32)
    # Add a few circular objects
    rr, cc = disk((25, 25), 10)
    image[rr, cc] = 1.0
    rr, cc = disk((75, 75), 8)
    image[rr, cc] = 1.0
    return image


@pytest.fixture(scope="module")
def sample_3d_image() -> np.ndarray:
    """Create a simple 3D RGB test image."""
    image = np.zeros((100, 100, 3), dtype=np.float32)
    # Add circular objects in green channel
    rr, cc = disk((25, 25), 10)
    image[rr, cc, 1] = 1.0
    rr, cc = disk((75, 75), 8)
    image[rr, cc, 1] = 1.0
    return image


@pytest.fixture(scope="module")
def sample_clustered_image() -> np.ndarray:
    """Create a simple clustered image for testing."""
    # 4x4 image with 3 clusters (0, 1, 2)
    return np.array([[0, 0, 1, 1], [0, 1, 1, 2], [2, 2, 1, 1], [2, 2, 0, 0]], dtype=np.int32)


@pytest.fixture(scope="module")
def sample_mask_image() -> np.ndarray:
    """Create a simple mask image for testing."""
    # 4x4 mask with 3 regions (0=background, 1, 2, 3)
    return np.array([[0, 0, 1, 1], [0, 1, 1, 2], [2, 2, 3, 3], [2, 2, 3, 3]], dtype=np.int32)


@pytest.fixture(scope="module")
def basic_masks() -> np.ndarray:
    """Create simple test masks for extend_mask testing."""
    return np.array(
        [[0, 0, 0, 0, 0], [0, 1, 0, 2, 0], [0, 0, 0, 0, 0], [0, 3, 0, 4, 0], [0, 0, 0, 0, 0]], dtype=np.int32
    )
