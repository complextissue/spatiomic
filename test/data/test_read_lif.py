"""Test the read_lif function."""

import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

import spatiomic as so

LIF_FILE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "test_files",
)


@pytest.mark.datafiles(os.path.join(LIF_FILE_DIR, "lif_example.lif.zip"))
def test_read_lif(
    datafiles: Path,
) -> None:
    """Test the read_lif function.

    Args:
        datafiles (Path): Path to the test file folder for an example .lif image.
    """
    lif_zip_file_path = os.path.join(str(datafiles), "lif_example.lif.zip")
    assert len(os.listdir(str(datafiles))) == 1
    assert os.path.isfile(lif_zip_file_path)

    lif_zip_file = ZipFile(lif_zip_file_path)
    lif_zip_file.extractall(str(datafiles))

    lif_file_path = os.path.join(str(datafiles), "lif_example.lif")

    for image_idx in [0, 1]:
        channels = so.data.read.read_lif(
            lif_file_path,
            image_idx=image_idx,
        )

        assert len(channels) == 3
        assert channels.shape == (3, 2048, 2048)
        assert isinstance(channels, np.ndarray)
