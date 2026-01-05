"""Mask extension implementation for spatiomic."""

from typing import TYPE_CHECKING, Any, cast

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

    Uses Voronoi tessellation via distance transform for O(1) complexity regardless of label count.
    Each background pixel is assigned to its nearest mask, constrained by dilation_pixels.

    Args:
        masks: 2D input segmentation masks where each unique integer represents a different segment.
        dilation_pixels: Maximum number of pixels to dilate each mask. Must be positive.
        background_label: The label value representing background pixels. Defaults to 0.
        use_gpu: Whether to use GPU acceleration with CuPy/cuCIM. Defaults to False.

    Returns:
        Extended masks with the same shape and dtype as input.

    Raises:
        ValueError: If dilation_pixels is not positive or if input is not 2D.
    """
    if dilation_pixels <= 0:
        msg = f"dilation_pixels must be positive, got {dilation_pixels}"
        raise ValueError(msg)

    if masks.ndim != 2:
        msg = f"Input masks must be 2D, got {masks.ndim}D"
        raise ValueError(msg)

    original_dtype = masks.dtype

    if TYPE_CHECKING or not use_gpu:
        xp = np
        cucim_morphology = None
    else:
        xp = import_package("cupy", alternative=np)
        cucim_morphology = import_package("cucim.core.operations.morphology", alternative=None)
        if cucim_morphology is None:
            use_gpu = False
            xp = np

    masks_xp = xp.asarray(masks)
    background_mask = masks_xp == background_label

    if not xp.any(background_mask):
        if use_gpu:
            return cast(npt.NDArray[np.integer[Any]], masks_xp.get().astype(original_dtype))  # type: ignore[attr-defined]
        return masks.copy()

    foreground_mask = ~background_mask

    if not xp.any(foreground_mask):
        if use_gpu:
            return cast(npt.NDArray[np.integer[Any]], masks_xp.get().astype(original_dtype))  # type: ignore[attr-defined]
        return masks.copy()

    if use_gpu and cucim_morphology is not None:
        distances, indices = cucim_morphology.distance_transform_edt(background_mask, return_indices=True)
        extended = masks_xp[indices[0], indices[1]]
        extended = xp.where(distances <= dilation_pixels, extended, background_label)  # type: ignore[union-attr]
        extended = xp.where(foreground_mask, masks_xp, extended)  # type: ignore[union-attr]
        del distances, indices, background_mask, foreground_mask, masks_xp
        xp.get_default_memory_pool().free_all_blocks()  # type: ignore[union-attr]
        return cast(npt.NDArray[np.integer[Any]], extended.get().astype(original_dtype))

    distances, indices = ndimage.distance_transform_edt(background_mask, return_indices=True)
    extended = masks[indices[0], indices[1]]
    extended = np.where(distances <= dilation_pixels, extended, background_label)
    extended = np.where(foreground_mask, masks, extended)

    return cast(npt.NDArray[np.integer[Any]], extended.astype(original_dtype))
