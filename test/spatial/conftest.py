"""Data fixtures for spatial module testing."""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def ordered_data() -> np.ndarray:
    """Generate some example data."""
    return np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])


@pytest.fixture(scope="module")
def unordered_data() -> np.ndarray:
    """Generate some example data."""
    return np.random.randint(0, 4, size=64).reshape(4, 4, 4).astype(np.float32)


@pytest.fixture(scope="module")
def clustered_data() -> np.ndarray:
    """Generate some example data."""
    return np.random.randint(0, 10, size=4096).reshape(64, 64).astype(np.float32)


@pytest.fixture(scope="module")
def cluster_assignments() -> np.ndarray:
    """Generate some example data."""
    return np.random.randint(0, 5, size=4096).reshape(64, 64).astype(np.float32)
