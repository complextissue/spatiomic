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
def example_lif() -> Path:
    """Path to the example LIF file."""
    return Path(__file__).parent / "test_files" / "lif_example.lif"


@pytest.fixture
def example_lif_zip() -> Path:
    """Path to the example LIF zip file."""
    return Path(__file__).parent / "test_files" / "lif_example.lif.zip"
