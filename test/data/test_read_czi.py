"""Test the read_czi function."""

import os
from pathlib import Path

import numpy as np
import pytest

import spatiomic as so

CZI_FILE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "test_files",
)


@pytest.mark.datafiles(os.path.join(CZI_FILE_DIR, "example.czi"))
def test_read_czi(
    datafiles: Path,
) -> None:
    """Test the read_czi function.

    Args:
        datafiles (Path): [description]
    """
    czi_file_path = str(datafiles)
    assert len(os.listdir(czi_file_path)) == 1
    assert os.path.isfile(os.path.join(czi_file_path, "example.czi"))

    channels = so.data.read.read_czi(
        str(os.path.join(czi_file_path, "example.czi")),
    )

    assert len(channels) == 4
    assert isinstance(channels, np.ndarray)
