"""Test configuration for data tests."""

from pathlib import Path

import pytest


@pytest.fixture
def example_tiff() -> Path:
    """Path to the example TIFF file."""
    return Path(__file__).parent.parent.parent / "docs" / "source" / "tutorials" / "data" / "example.tiff"


@pytest.fixture
def kidney_tiff() -> Path:
    """Path to the kidney TIFF file."""
    return Path(__file__).parent / "test_files" / "kidney-tissue-fluorescence.tiff"


@pytest.fixture
def example_czi() -> Path:
    """Path to the example CZI file."""
    return Path(__file__).parent / "test_files" / "example.czi"


@pytest.fixture
def example_uint16_czi() -> Path:
    """Path to a 16-bit CZI file with 3 channels of 48 x 64 pixels and S, T, C, Z, Y, X dimensions.

    Written with pylibCZIrw from `np.random.default_rng(0).integers(0, 65535, size=(3, 48, 64), dtype=np.uint16)`.
    """
    return Path(__file__).parent / "test_files" / "example_uint16.czi"


@pytest.fixture
def example_lif() -> Path:
    """Path to the example LIF file."""
    return Path(__file__).parent / "test_files" / "lif_example.lif"
