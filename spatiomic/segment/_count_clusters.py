"""Cluster counting implementation for spatiomic."""

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from spatiomic._internal._import_package import import_package


def count_clusters(
    clustered_img: npt.NDArray[np.integer[Any]],
    masks: npt.NDArray[np.integer[Any]],
    cluster_count: int,
    normalize: bool = False,
    exclude_background: bool = True,
    use_gpu: bool = False,
) -> pd.DataFrame:
    """Count the number of clusters within each mask region.

    Args:
        clustered_img: The clustered image array where each pixel value represents a cluster ID.
        masks: The segmented cell mask image where each pixel value represents a mask region ID.
        cluster_count: The number of clusters to count.
        normalize: Whether to return row-normalized cluster counts (relative) instead of absolute counts.
        exclude_background: Whether to exclude background pixels (mask value 0) from analysis.
        use_gpu: Whether to use GPU acceleration with CuPy. Defaults to False.

    Returns:
        A DataFrame containing cluster counts for each mask region. Rows represent mask regions,
        columns represent cluster IDs (0 to cluster_count-1), and values are either absolute counts
        or normalized proportions based on the normalize parameter.

    Note:
        - GPU acceleration is available when CuPy is installed
        - For large images with many mask regions, GPU acceleration can provide significant speedup
        - Memory usage scales with the number of unique mask regions and cluster count
    """
    # Import appropriate array library
    if TYPE_CHECKING or not use_gpu:  # noqa: SIM108
        xp = np
    else:
        xp = import_package("cupy", alternative=np)

    # Convert arrays to appropriate backend
    clustered_img_xp = xp.asarray(clustered_img)
    masks_xp = xp.asarray(masks)

    # Get unique mask regions, optionally excluding background (0)
    unique_regions = xp.unique(masks_xp)
    if exclude_background and 0 in unique_regions:
        unique_regions = unique_regions[unique_regions != 0]

    # Initialize cluster count matrix
    if len(unique_regions) == 0 or cluster_count == 0:
        cluster_counts = xp.zeros((len(unique_regions), cluster_count), dtype=xp.int32)
    else:
        # Efficient vectorized approach: use advanced indexing
        cluster_counts = xp.zeros((len(unique_regions), cluster_count), dtype=xp.int32)

        # Vectorized counting for all pixels at once
        for i, region in enumerate(unique_regions):
            region_mask = masks_xp == region
            if xp.any(region_mask):
                region_clusters = clustered_img_xp[region_mask]
                counts = xp.bincount(region_clusters, minlength=cluster_count)[:cluster_count]
                cluster_counts[i, :] = counts

    # Convert back to numpy for pandas compatibility
    if use_gpu and xp != np:
        cluster_counts = cluster_counts.get()  # type: ignore
        unique_regions = unique_regions.get()  # type: ignore

    # Create DataFrame with proper indexing
    df_cluster_counts = pd.DataFrame(
        cluster_counts,
        columns=range(cluster_count),
        index=unique_regions,
    )
    df_cluster_counts.index.name = "mask_region"

    # Normalize if requested
    if normalize:
        df_cluster_counts = df_cluster_counts.div(df_cluster_counts.sum(axis=1), axis=0).fillna(
            0
        )  # Handle division by zero for empty regions

    return df_cluster_counts
