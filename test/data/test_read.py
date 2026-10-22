"""Test the read class."""

from pathlib import Path

import numpy as np
import pytest

import spatiomic as so


def test_read_tiff(example_tiff: Path) -> None:
    """Test the read_tiff function."""
    assert example_tiff.exists()

    # Test default behavior (no dtype conversion)
    image = so.data.read.read_tiff(example_tiff)
    assert isinstance(image, np.ndarray)
    original_dtype = image.dtype

    # Test explicit dtype conversion
    image_float32 = so.data.read.read_tiff(example_tiff, dtype=np.float32)
    assert isinstance(image_float32, np.ndarray)
    assert image_float32.dtype == np.float32
    assert image_float32.shape == image.shape

    # Test that original dtype is preserved when dtype=None
    image_none = so.data.read.read_tiff(example_tiff, dtype=None)
    assert isinstance(image_none, np.ndarray)
    assert image_none.dtype == original_dtype

    # Test channel return functionality
    image_with_channels = so.data.read.read_tiff(example_tiff, return_channels=True)
    if isinstance(image_with_channels, tuple):
        image_data, channels = image_with_channels
        assert isinstance(image_data, np.ndarray)
        assert isinstance(channels, list)
    else:
        assert isinstance(image_with_channels, np.ndarray)


@pytest.mark.parametrize("image_idx", [0, 1])
def test_read_lif(example_lif: Path, image_idx: int) -> None:
    """Test the read_lif function."""
    lif_file_path = example_lif

    # Test default behavior (no dtype conversion)
    channels = so.data.read.read_lif(
        lif_file_path,
        image_idx=image_idx,
    )

    assert len(channels) == 3
    assert channels.shape == (3, 2048, 2048)
    assert isinstance(channels, np.ndarray)
    original_dtype = channels.dtype

    # Test explicit dtype conversion
    channels_float32 = so.data.read.read_lif(
        lif_file_path,
        image_idx=image_idx,
        dtype=np.float32,
    )
    assert channels_float32.dtype == np.float32

    # Test that original dtype is preserved when dtype=None
    channels_none = so.data.read.read_lif(
        lif_file_path,
        image_idx=image_idx,
        dtype=None,
    )
    assert channels_none.dtype == original_dtype


def test_read_czi(example_czi: Path) -> None:
    """Test the read_czi function."""
    assert example_czi.exists()

    # Test default behavior (no dtype conversion)
    channels = so.data.read.read_czi(example_czi)
    assert len(channels) == 4
    assert isinstance(channels, np.ndarray)
    original_dtype = channels.dtype

    # Test explicit dtype conversion
    channels_float32 = so.data.read.read_czi(example_czi, dtype=np.float32)
    assert channels_float32.dtype == np.float32

    # Test that original dtype is preserved when dtype=None
    channels_none = so.data.read.read_czi(example_czi, dtype=None)
    assert channels_none.dtype == original_dtype


def test_read_czi_uint16(example_uint16_czi: Path) -> None:
    """Test that 16-bit CZI files are read without truncation, independent of their dimension layout."""
    channels_expected = np.random.default_rng(0).integers(0, 65535, size=(3, 48, 64), dtype=np.uint16)

    channels = so.data.read.read_czi(example_uint16_czi)

    assert channels.dtype == np.uint16
    np.testing.assert_array_equal(channels, channels_expected)
