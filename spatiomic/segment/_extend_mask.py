"""Mask extension implementation for spatiomic."""

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from spatiomic._internal._import_package import import_package


def extend_mask(
    masks: npt.NDArray[np.integer[Any]],
    dilation_pixels: int = 1,
    background_label: int = 0,
    use_gpu: bool = False,
) -> npt.NDArray[np.integer[Any]]:
    """Extend segmentation masks by dilating them up to halfway to the nearest neighboring mask.

    This function dilates each mask region by a specified number of pixels, but stops dilation
    at the halfway point to the nearest neighboring mask to ensure fair distribution of space
    between adjacent regions.

    Args:
        masks: Input segmentation masks where each unique integer represents a different segment.
            Background pixels should have a consistent label (default 0).
        dilation_pixels: Maximum number of pixels to dilate each mask. The actual dilation
            may be less if masks meet halfway. Must be positive.
        background_label: The label value representing background pixels. Defaults to 0.
        use_gpu: Whether to use GPU acceleration with CuPy. Defaults to False.

    Returns:
        Extended masks with the same shape and dtype as input, where each mask has been
        dilated up to the halfway point to neighboring masks.

    Raises:
        ValueError: If dilation_pixels is not positive or if input is not 2D.

    Example:
        Basic usage for extending cell masks:

        ```python
        import spatiomic as so
        import numpy as np

        # Create example segmentation masks
        masks = np.array([[0, 0, 0, 0, 0], [0, 1, 0, 2, 0], [0, 0, 0, 0, 0], [0, 3, 0, 4, 0], [0, 0, 0, 0, 0]])

        # Extend masks by 1 pixel
        extended_masks = so.segment.extend_mask(masks, dilation_pixels=1)

        # Each mask will expand towards neighboring masks but stop halfway
        ```

    Note:
        - The algorithm uses distance transforms for efficient computation
        - Memory usage scales with image size and number of unique labels
        - GPU acceleration is available when CuPy is installed
    """
    if TYPE_CHECKING or not use_gpu:
        xp = np
        ndimage_pkg = ndimage
    else:
        xp = import_package("cupy", alternative=np)
        ndimage_pkg = import_package("cupyx.scipy.ndimage", alternative=ndimage)

    if dilation_pixels <= 0:
        msg = f"dilation_pixels must be positive, got {dilation_pixels}"
        raise ValueError(msg)

    if masks.ndim != 2:
        msg = f"Input masks must be 2D, got {masks.ndim}D"
        raise ValueError(msg)

    # Convert to GPU array if using CuPy
    masks_xp = xp.asarray(masks)

    # Get unique labels excluding background
    unique_labels = xp.unique(masks_xp)
    unique_labels = unique_labels[unique_labels != background_label]

    if len(unique_labels) == 0:
        return masks  # No masks to extend

    # Create arrays for efficient vectorized computation
    extended_masks = masks_xp.copy()
    background_mask = masks_xp == background_label

    if not xp.any(background_mask):
        # No background pixels to extend into
        if xp.__name__ == "cupy" and hasattr(extended_masks, "get"):
            return extended_masks.get()  # type: ignore[no-any-return]
        return np.asarray(extended_masks)  # type: ignore[no-any-return]

    # For efficient computation, we'll use distance transforms
    # Create distance maps for each label
    label_distances = xp.full((len(unique_labels), *masks_xp.shape), xp.inf, dtype=xp.float32)

    for idx, label in enumerate(unique_labels):
        # Create binary mask for this label
        label_mask = masks_xp == label

        if xp.__name__ == "cupy":
            # Convert to numpy for scipy operations
            label_mask_np = label_mask.get() if hasattr(label_mask, "get") else np.asarray(label_mask)
            dist_transform = ndimage.distance_transform_edt(~label_mask_np)
            label_distances[idx] = xp.asarray(dist_transform)
        else:
            dist_transform = ndimage_pkg.distance_transform_edt(~label_mask)
            label_distances[idx] = dist_transform

    # For each background pixel, find closest two labels and their distances
    background_indices = xp.where(background_mask)

    if len(background_indices[0]) > 0:
        # Vectorized computation for all background pixels
        distances_at_bg = label_distances[:, background_indices[0], background_indices[1]]

        # Find two closest labels for each background pixel
        sorted_indices = xp.argsort(distances_at_bg, axis=0)
        closest_distances = xp.take_along_axis(distances_at_bg, sorted_indices, axis=0)
        closest_labels = unique_labels[sorted_indices]

        # Calculate conditions for assignment
        closest_dist = closest_distances[0]
        within_dilation = closest_dist <= dilation_pixels

        # For pixels with multiple nearby labels, check halfway condition
        has_second_neighbor = len(unique_labels) > 1
        if has_second_neighbor:
            second_closest_dist = closest_distances[1] if len(unique_labels) > 1 else xp.inf
            halfway_dist = (closest_dist + second_closest_dist) / 2.0
            within_halfway = closest_dist <= halfway_dist
            assignment_condition = within_dilation & within_halfway
        else:
            assignment_condition = within_dilation

        # Assign pixels to closest labels where conditions are met
        valid_assignments = xp.where(assignment_condition)[0]
        if len(valid_assignments) > 0:
            assigned_rows = background_indices[0][valid_assignments]
            assigned_cols = background_indices[1][valid_assignments]
            assigned_labels = closest_labels[0][valid_assignments]
            extended_masks[assigned_rows, assigned_cols] = assigned_labels

    # Convert back to numpy if we used CuPy
    if xp.__name__ == "cupy" and hasattr(extended_masks, "get"):
        return extended_masks.get()  # type: ignore[no-any-return]

    return np.asarray(extended_masks)  # type: ignore[no-any-return]
