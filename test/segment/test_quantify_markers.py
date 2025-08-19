"""Test marker quantification functionality."""

import numpy as np
import pandas as pd
import pytest

import spatiomic as so


@pytest.mark.cpu
def test_quantify_markers_basic() -> None:
    """Test basic marker quantification functionality."""
    # Create simple test data
    image = np.random.rand(4, 4, 3).astype(np.float32)
    masks = np.array([[0, 0, 1, 1], [0, 1, 1, 2], [2, 2, 3, 3], [2, 2, 3, 3]], dtype=np.int32)

    result = so.segment.quantify_markers(
        image=image,
        masks=masks,
        quantification_function=np.mean,
        exclude_background=True,
        use_gpu=False,
    )

    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == 3  # 3 channels
    assert result.index.name == "mask_region"
    assert list(result.columns) == [0, 1, 2]

    # Check that background (region 0) is excluded
    assert 0 not in result.index

    # Check that all expected regions are present
    expected_regions = [1, 2, 3]
    assert sorted(result.index.tolist()) == expected_regions

    # Check data types
    assert all(result.dtypes == np.float64)


@pytest.mark.cpu
def test_quantify_markers_channel_names() -> None:
    """Test marker quantification with custom channel names."""
    image = np.random.rand(4, 4, 3).astype(np.float32)
    masks = np.array([[0, 0, 1, 1], [0, 1, 1, 2], [2, 2, 3, 3], [2, 2, 3, 3]], dtype=np.int32)

    channel_names: list[str | int] = ["DAPI", "CD3", "CD20"]

    result = so.segment.quantify_markers(
        image=image,
        masks=masks,
        quantification_function=np.mean,
        channel_names=channel_names,
        exclude_background=True,
        use_gpu=False,
    )

    # Check column names
    assert list(result.columns) == channel_names


@pytest.mark.cpu
def test_quantify_markers_different_functions() -> None:
    """Test marker quantification with different quantification functions."""
    # Create test data with known values
    image = np.ones((2, 2, 2), dtype=np.float32)
    image[:, :, 0] = 10.0  # Channel 0: all 10s
    image[:, :, 1] = 20.0  # Channel 1: all 20s

    masks = np.ones((2, 2), dtype=np.int32)  # Single region

    # Test mean
    result_mean = so.segment.quantify_markers(
        image=image,
        masks=masks,
        quantification_function=np.mean,
        exclude_background=False,
        use_gpu=False,
    )

    # Check mean values
    assert np.isclose(result_mean.loc[1, 0], 10.0)  # type: ignore[arg-type]
    assert np.isclose(result_mean.loc[1, 1], 20.0)  # type: ignore[arg-type]

    # Test median
    result_median = so.segment.quantify_markers(
        image=image,
        masks=masks,
        quantification_function=np.median,
        exclude_background=False,
        use_gpu=False,
    )

    # Check median values (should be same as mean for constant values)
    assert np.isclose(result_median.loc[1, 0], 10.0)  # type: ignore[arg-type]
    assert np.isclose(result_median.loc[1, 1], 20.0)  # type: ignore[arg-type]

    # Test custom function (max)
    result_max = so.segment.quantify_markers(
        image=image,
        masks=masks,
        quantification_function=np.max,
        exclude_background=False,
        use_gpu=False,
    )

    # Check max values
    assert np.isclose(result_max.loc[1, 0], 10.0)  # type: ignore[arg-type]
    assert np.isclose(result_max.loc[1, 1], 20.0)  # type: ignore[arg-type]


@pytest.mark.cpu
def test_quantify_markers_percentile() -> None:
    """Test marker quantification with percentile function."""
    # Create test data with gradient values
    image = np.zeros((4, 4, 1), dtype=np.float32)
    image[:, :, 0] = np.arange(16).reshape(4, 4)  # Values 0-15

    masks = np.ones((4, 4), dtype=np.int32)  # Single region

    # Test 50th percentile (median)
    result_p50 = so.segment.quantify_markers(
        image=image,
        masks=masks,
        quantification_function=lambda x: np.percentile(x, 50),
        exclude_background=False,
        use_gpu=False,
    )

    # 50th percentile of 0-15 should be 7.5
    assert np.isclose(result_p50.loc[1, 0], 7.5)  # type: ignore[arg-type]

    # Test 90th percentile
    result_p90 = so.segment.quantify_markers(
        image=image,
        masks=masks,
        quantification_function=lambda x: np.percentile(x, 90),
        exclude_background=False,
        use_gpu=False,
    )

    # 90th percentile of 0-15 should be 13.5
    assert np.isclose(result_p90.loc[1, 0], 13.5)  # type: ignore[arg-type]


@pytest.mark.cpu
def test_quantify_markers_empty_regions() -> None:
    """Test marker quantification with empty regions."""
    image = np.random.rand(4, 4, 2).astype(np.float32)

    # Create mask with a gap (region 2 missing)
    masks = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 3, 3], [0, 0, 3, 3]], dtype=np.int32)

    result = so.segment.quantify_markers(
        image=image,
        masks=masks,
        quantification_function=np.mean,
        exclude_background=True,
        use_gpu=False,
    )

    # Should only have regions 1 and 3
    assert sorted(result.index.tolist()) == [1, 3]
    assert len(result) == 2


@pytest.mark.cpu
def test_quantify_markers_include_background() -> None:
    """Test marker quantification including background."""
    image = np.random.rand(4, 4, 2).astype(np.float32)
    masks = np.array([[0, 0, 1, 1], [0, 1, 1, 2], [2, 2, 2, 2], [2, 2, 2, 2]], dtype=np.int32)

    result = so.segment.quantify_markers(
        image=image,
        masks=masks,
        quantification_function=np.mean,
        exclude_background=False,
        use_gpu=False,
    )

    # Should include background (region 0)
    assert 0 in result.index
    assert sorted(result.index.tolist()) == [0, 1, 2]


@pytest.mark.cpu
def test_quantify_markers_input_validation() -> None:
    """Test input validation for marker quantification."""
    image = np.random.rand(4, 4, 3).astype(np.float32)
    masks = np.ones((4, 4), dtype=np.int32)

    # Test wrong dimensions for image
    with pytest.raises(ValueError, match="image must be 3D"):
        so.segment.quantify_markers(
            image=np.random.rand(4, 4),  # 2D instead of 3D
            masks=masks,
            quantification_function=np.mean,
        )

    # Test wrong dimensions for masks
    with pytest.raises(ValueError, match="masks must be 2D"):
        so.segment.quantify_markers(
            image=image,
            masks=np.ones((4, 4, 1), dtype=np.int32),  # 3D instead of 2D
            quantification_function=np.mean,
        )

    # Test mismatched spatial dimensions
    with pytest.raises(ValueError, match="Spatial dimensions must match"):
        so.segment.quantify_markers(
            image=image,
            masks=np.ones((5, 5), dtype=np.int32),  # Different size
            quantification_function=np.mean,
        )

    # Test wrong number of channel names
    with pytest.raises(ValueError, match="Number of channel names"):
        so.segment.quantify_markers(
            image=image,
            masks=masks,
            quantification_function=np.mean,
            channel_names=["A", "B"],  # 2 names for 3 channels
        )


@pytest.mark.cpu
def test_quantify_markers_edge_cases() -> None:
    """Test edge cases for marker quantification."""
    # Test with single pixel regions
    image = np.random.rand(2, 2, 2).astype(np.float32)
    masks = np.array([[1, 2], [3, 4]], dtype=np.int32)

    result = so.segment.quantify_markers(
        image=image,
        masks=masks,
        quantification_function=np.mean,
        exclude_background=True,
        use_gpu=False,
    )

    # Should work with single-pixel regions
    assert len(result) == 4
    assert sorted(result.index.tolist()) == [1, 2, 3, 4]

    # Test with no regions (all background)
    masks_bg = np.zeros((2, 2), dtype=np.int32)

    result_bg = so.segment.quantify_markers(
        image=image,
        masks=masks_bg,
        quantification_function=np.mean,
        exclude_background=True,
        use_gpu=False,
    )

    # Should return empty DataFrame
    assert len(result_bg) == 0
    assert result_bg.shape[1] == 2  # Still 2 channels


@pytest.mark.cpu
def test_quantify_markers_nan_handling() -> None:
    """Test handling of functions that might produce NaN."""
    # Create data where std of single value should be 0
    image = np.ones((2, 2, 1), dtype=np.float32) * 5.0
    masks = np.array([[1, 0], [0, 2]], dtype=np.int32)

    # Test with std function on single values
    result = so.segment.quantify_markers(
        image=image,
        masks=masks,
        quantification_function=np.std,
        exclude_background=True,
        use_gpu=False,
    )

    # Standard deviation of single value should be 0
    assert np.isclose(result.loc[1, 0], 0.0)  # type: ignore
    assert np.isclose(result.loc[2, 0], 0.0)  # type: ignore


@pytest.mark.cpu
def test_quantify_markers_vs_count_clusters_pattern() -> None:
    """Test that quantify_markers follows similar patterns to count_clusters."""
    # Create test data similar to count_clusters test
    image = np.random.rand(4, 4, 3).astype(np.float32)
    masks = np.array([[0, 0, 1, 1], [0, 1, 1, 2], [2, 2, 3, 3], [2, 2, 3, 3]], dtype=np.int32)

    result = so.segment.quantify_markers(
        image=image,
        masks=masks,
        quantification_function=np.mean,
        exclude_background=True,
        use_gpu=False,
    )

    # Check structure consistency with count_clusters
    assert isinstance(result, pd.DataFrame)
    assert result.index.name == "mask_region"
    assert 0 not in result.index  # Background excluded
    assert all(isinstance(col, int) for col in result.columns)  # Numeric column names by default


@pytest.mark.gpu
def test_quantify_markers_gpu_cpu_consistency() -> None:
    """Test that GPU and CPU implementations give consistent results."""
    image = np.random.rand(4, 4, 2).astype(np.float32)
    masks = np.array([[0, 0, 1, 1], [0, 1, 1, 2], [2, 2, 3, 3], [2, 2, 3, 3]], dtype=np.int32)

    # CPU result
    result_cpu = so.segment.quantify_markers(
        image=image,
        masks=masks,
        quantification_function=np.mean,
        exclude_background=True,
        use_gpu=False,
    )

    # GPU result (will fall back to CPU if CuPy not available)
    result_gpu = so.segment.quantify_markers(
        image=image,
        masks=masks,
        quantification_function=np.mean,
        exclude_background=True,
        use_gpu=True,
    )

    # Results should be nearly identical
    pd.testing.assert_frame_equal(result_cpu, result_gpu, rtol=1e-6)
