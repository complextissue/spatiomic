"""Community assignment implementation for spatiomic."""

from typing import Any, List, Union

import numpy as np
import numpy.typing as npt


def assign_communities(
    masks: npt.NDArray[np.integer[Any]],
    communities: Union[npt.NDArray[np.integer[Any]], List[int]],
) -> npt.NDArray[np.integer[Any]]:
    """Assign community labels to mask regions to create a community-labeled image.

    This function creates an image where each pixel value represents the community assignment
    of the corresponding mask region. Mask region `i` (where i > 0) gets assigned the
    community label from `communities[i-1]`.

    Args:
        masks: Segmented mask image where each pixel value represents a mask region ID.
            Background pixels should have value 0, and mask regions should have positive integer IDs.
        communities: Array or list of community labels for each mask region. The length should
            match the number of unique mask regions (excluding background).

    Returns:
        An image array where each pixel contains the community ID + 1 of its corresponding
        mask region. Background pixels (mask value 0) remain 0.

    Raises:
        ValueError: If the length of communities doesn't match the number of mask regions.

    Example:
        >>> masks = np.array([[0, 1, 1], [0, 2, 2]])
        >>> communities = [0, 1]  # region 1 -> community 0, region 2 -> community 1
        >>> result = assign_communities(masks, communities)
        >>> # result = [[0, 1, 1], [0, 2, 2]]  # community labels + 1
    """
    communities_array = np.asarray(communities)

    # Get unique mask regions, excluding background (0)
    unique_regions = np.unique(masks)
    unique_regions = unique_regions[unique_regions != 0]

    if len(communities_array) != len(unique_regions):
        raise ValueError(
            f"Number of communities ({len(communities_array)}) must match "
            f"number of mask regions ({len(unique_regions)})"
        )

    # Initialize output with same shape as masks
    community_image = np.zeros_like(masks)

    # Assign community labels (adding 1 to avoid confusion with background)
    for i, region_id in enumerate(unique_regions):
        community_image[masks == region_id] = communities_array[i] + 1

    return community_image
