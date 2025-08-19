"""Test Cellpose segmentation functionality."""

import numpy as np
import pytest
from numpy.typing import NDArray

import spatiomic as so


@pytest.mark.cpu
def test_cellpose_initialization() -> None:
    """Test Cellpose class initialization."""
    cellpose = so.segment.cellpose()
    # GPU usage depends on actual GPU availability, just check it's boolean
    assert isinstance(cellpose.use_gpu, bool)
    assert cellpose.pretrained_model == "cpsam"

    cellpose_custom = so.segment.cellpose(use_gpu=False, pretrained_model="custom_model")
    assert cellpose_custom.use_gpu is False
    assert cellpose_custom.pretrained_model == "custom_model"


@pytest.mark.cpu
def test_cellpose_predict_2d(sample_2d_image: NDArray) -> None:
    """Test Cellpose prediction on 2D image."""
    cellpose = so.segment.cellpose(use_gpu=False)
    masks, flows, styles = cellpose.predict(sample_2d_image)

    assert isinstance(masks, np.ndarray)
    assert masks.shape == sample_2d_image.shape
    assert masks.dtype in [np.int32, np.int64, np.uint16, np.uint32]


@pytest.mark.cpu
def test_cellpose_predict_3d(sample_3d_image: NDArray) -> None:
    """Test Cellpose prediction on 3D image."""
    cellpose = so.segment.cellpose(use_gpu=False)
    masks, flows, styles = cellpose.predict(sample_3d_image)

    assert isinstance(masks, np.ndarray)
    assert masks.shape == sample_3d_image.shape[:2]
    assert masks.dtype in [np.int32, np.int64, np.uint16, np.uint32]


@pytest.mark.cpu
def test_cellpose_predict_with_kwargs(sample_2d_image: NDArray) -> None:
    """Test Cellpose prediction with additional kwargs."""
    cellpose = so.segment.cellpose(use_gpu=False)
    masks, flows, styles = cellpose.predict(
        sample_2d_image,
        flow_threshold=0.3,
        cellprob_threshold=0.1,
    )

    assert isinstance(masks, np.ndarray)
    assert masks.shape == sample_2d_image.shape


@pytest.mark.cpu
def test_cellpose_invalid_dimensions() -> None:
    """Test Cellpose with invalid input dimensions."""
    cellpose = so.segment.cellpose(use_gpu=False)

    with pytest.raises(ValueError, match="Input image must be 2D or 3D"):
        cellpose.predict(np.array([1, 2, 3]))

    with pytest.raises(ValueError, match="Input image must be 2D or 3D"):
        cellpose.predict(np.zeros((10, 10, 3, 3)))
