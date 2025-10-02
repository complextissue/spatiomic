"""Test the read class."""

from pathlib import Path
from zipfile import ZipFile

import numpy as np

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


def test_read_lif(example_lif_zip: Path) -> None:
    """Test the read_lif function."""
    assert example_lif_zip.exists()

    # Extract the zip file
    temp_dir = example_lif_zip.parent / "temp_lif"
    temp_dir.mkdir(exist_ok=True)

    with ZipFile(example_lif_zip) as lif_zip_file:
        lif_zip_file.extractall(temp_dir)
    lif_file_path = temp_dir / "lif_example.lif"
    assert lif_file_path.exists()

    for image_idx in [0, 1]:
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
