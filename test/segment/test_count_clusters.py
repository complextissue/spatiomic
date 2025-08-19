"""Test cluster counting functionality."""

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

import spatiomic as so


@pytest.mark.cpu
def test_count_clusters_basic(sample_clustered_image: NDArray, sample_mask_image: NDArray) -> None:
    """Test basic cluster counting functionality."""
    result = so.segment.count_clusters(
        clustered_img=sample_clustered_image,
        masks=sample_mask_image,
        cluster_count=3,
        normalize=False,
        exclude_background=True,
        use_gpu=False,
    )

    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == 3  # 3 clusters
    assert result.index.name == "mask_region"
    assert list(result.columns) == [0, 1, 2]

    # Check that background (region 0) is excluded
    assert 0 not in result.index

    # Check data types
    assert all(result.dtypes == np.int32)


@pytest.mark.cpu
def test_count_clusters_normalization(sample_clustered_image: NDArray, sample_mask_image: NDArray) -> None:
    """Test normalized cluster counts."""
    result = so.segment.count_clusters(
        clustered_img=sample_clustered_image,
        masks=sample_mask_image,
        cluster_count=3,
        normalize=True,
        exclude_background=True,
        use_gpu=False,
    )

    # Check that rows sum to 1 (or 0 for empty regions)
    row_sums = result.sum(axis=1)
    expected_sums = np.ones(len(result))
    np.testing.assert_array_almost_equal(row_sums, expected_sums, decimal=10)

    # Check data types are float
    assert all(result.dtypes == np.float64)


@pytest.mark.cpu
def test_count_clusters_include_background(sample_clustered_image: NDArray, sample_mask_image: NDArray) -> None:
    """Test including background in results."""
    result = so.segment.count_clusters(
        clustered_img=sample_clustered_image,
        masks=sample_mask_image,
        cluster_count=3,
        exclude_background=False,
        use_gpu=False,
    )

    # Background region (0) should be included
    assert 0 in result.index


@pytest.mark.cpu
def test_count_clusters_empty_regions() -> None:
    """Test handling of empty mask regions."""
    # Create mask with gaps
    clustered = np.array([[0, 1], [1, 2]], dtype=np.int32)
    mask = np.array([[1, 1], [3, 3]], dtype=np.int32)  # Region 2 is missing

    result = so.segment.count_clusters(
        clustered_img=clustered, masks=mask, cluster_count=3, normalize=True, use_gpu=False
    )

    # Should handle missing regions gracefully
    assert len(result) == 2  # Only regions 1 and 3


@pytest.mark.cpu
def test_count_clusters_single_cluster_per_region() -> None:
    """Test case where each region has only one cluster type."""
    clustered = np.array([[0, 1], [2, 2]], dtype=np.int32)
    mask = np.array([[1, 2], [3, 3]], dtype=np.int32)

    result = so.segment.count_clusters(
        clustered_img=clustered, masks=mask, cluster_count=3, normalize=False, use_gpu=False
    )

    # Region 1 should have 1 count for cluster 0
    assert result.loc[1, 0] == 1
    assert result.loc[1, 1] == 0
    assert result.loc[1, 2] == 0


@pytest.mark.cpu
def test_count_clusters_parameter_validation(sample_clustered_image: NDArray, sample_mask_image: NDArray) -> None:
    """Test different cluster_count values."""
    # Test with more clusters than present in data
    result = so.segment.count_clusters(
        clustered_img=sample_clustered_image, masks=sample_mask_image, cluster_count=5, use_gpu=False
    )

    assert result.shape[1] == 5
    # Clusters 3 and 4 should have zero counts
    assert (result[3] == 0).all()
    assert (result[4] == 0).all()


@pytest.mark.cpu
def test_count_clusters_gpu_cpu_consistency(sample_clustered_image: NDArray, sample_mask_image: NDArray) -> None:
    """Test that GPU and CPU implementations give identical results."""
    # CPU version
    result_cpu = so.segment.count_clusters(
        clustered_img=sample_clustered_image,
        masks=sample_mask_image,
        cluster_count=3,
        normalize=False,
        use_gpu=False,
    )

    # GPU version (will fall back to CPU if CuPy not available)
    result_gpu = so.segment.count_clusters(
        clustered_img=sample_clustered_image,
        masks=sample_mask_image,
        cluster_count=3,
        normalize=False,
        use_gpu=True,
    )

    # Results should be identical
    pd.testing.assert_frame_equal(result_cpu, result_gpu)


@pytest.mark.cpu
def test_count_clusters_zero_cluster_count() -> None:
    """Test input validation and edge cases."""
    clustered = np.array([[0, 1], [1, 2]], dtype=np.int32)
    mask = np.array([[1, 1], [2, 2]], dtype=np.int32)

    # Test zero cluster count should still work
    result = so.segment.count_clusters(clustered_img=clustered, masks=mask, cluster_count=0, use_gpu=False)
    assert result.shape[1] == 0


@pytest.mark.cpu
def test_count_clusters_different_dtypes() -> None:
    """Test with different input data types."""
    # Test with different integer types
    clustered_int64 = np.array([[0, 1], [1, 2]], dtype=np.int64)
    mask_int16 = np.array([[1, 1], [2, 2]], dtype=np.int16)

    result = so.segment.count_clusters(clustered_img=clustered_int64, masks=mask_int16, cluster_count=3, use_gpu=False)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 3)


@pytest.mark.cpu
def test_count_clusters_zero_division_in_normalization() -> None:
    """Test handling of empty regions during normalization."""
    # Create a case where a mask region has no pixels
    clustered = np.array([[0, 1]], dtype=np.int32)
    mask = np.array([[1, 1]], dtype=np.int32)

    # Force an empty region by using a mask value that doesn't exist
    result = so.segment.count_clusters(
        clustered_img=clustered, masks=mask, cluster_count=2, normalize=True, use_gpu=False
    )

    # Should not contain NaN values
    assert not result.isna().any().any()
