"""Marker quantification implementation for spatiomic."""

from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from spatiomic._internal._import_package import import_package


def quantify_markers(
    image: npt.NDArray[np.floating[Any]],
    masks: npt.NDArray[np.integer[Any]],
    quantification_function: Callable[[npt.NDArray[np.floating[Any]]], float] = np.mean,
    channel_names: Optional[List[Union[str, int]]] = None,
    exclude_background: bool = True,
    use_gpu: bool = False,
) -> pd.DataFrame:
    """Quantify marker values within each mask region using a specified quantification function.

    This function applies a quantification function (e.g., mean, median, percentile) to each channel
    of a multichannel image within each mask region, similar to how count_clusters counts cluster
    occurrences but for continuous marker intensities.

    Args:
        image: The multichannel image array where the last dimension represents channels.
            Should have shape (H, W, C) where C is the number of channels.
        masks: The segmented mask image where each pixel value represents a mask region ID.
            Should have shape (H, W) matching the spatial dimensions of image.
        quantification_function: Function to apply for quantification within each mask region.
            Should accept a 1D array and return a single float value. Common examples:
            - np.mean (default): Average intensity
            - np.median: Median intensity
            - lambda x: np.percentile(x, 90): 90th percentile
            - np.std: Standard deviation
        channel_names: Optional names for the channels. If None, channels will be numbered 0, 1, 2, etc.
        exclude_background: Whether to exclude background pixels (mask value 0) from analysis.
        use_gpu: Whether to use GPU acceleration with CuPy. Defaults to False.

    Returns:
        A DataFrame containing quantified marker values for each mask region. Rows represent mask regions,
        columns represent channels, and values are the result of applying quantification_function to each
        channel within each mask region.

    Raises:
        ValueError: If input arrays have incompatible shapes or invalid dimensions.

    Example:
        Basic usage for quantifying mean marker intensity:

        ```python
        import spatiomic as so
        import numpy as np

        # Create example multichannel image (H, W, C)
        image = np.random.rand(100, 100, 5)

        # Create example mask image (H, W)
        masks = np.zeros((100, 100), dtype=int)
        masks[25:75, 25:75] = 1
        masks[10:40, 10:40] = 2

        # Quantify mean intensity for each marker in each mask region
        marker_quantification = so.segment.quantify_markers(image, masks, quantification_function=np.mean)

        # Quantify 90th percentile intensity
        percentile_quantification = so.segment.quantify_markers(
            image, masks, quantification_function=lambda x: np.percentile(x, 90)
        )
        ```
    """
    # Import appropriate array library
    if TYPE_CHECKING or not use_gpu:  # noqa: SIM108
        xp = np
    else:
        xp = import_package("cupy", alternative=np)

    # Validate input dimensions
    if image.ndim != 3:
        msg = f"image must be 3D (H, W, C), got {image.ndim}D"
        raise ValueError(msg)

    if masks.ndim != 2:
        msg = f"masks must be 2D (H, W), got {masks.ndim}D"
        raise ValueError(msg)

    if image.shape[:2] != masks.shape:
        msg = f"Spatial dimensions must match: image {image.shape[:2]} vs masks {masks.shape}"
        raise ValueError(msg)

    # Convert arrays to appropriate backend
    image_xp = xp.asarray(image)
    masks_xp = xp.asarray(masks)  # Get image dimensions
    _height, _width, channel_count = image_xp.shape

    # Get unique mask regions, optionally excluding background (0)
    unique_regions = xp.unique(masks_xp)
    if exclude_background and 0 in unique_regions:
        unique_regions = unique_regions[unique_regions != 0]

    # Initialize quantification matrix
    if len(unique_regions) == 0 or channel_count == 0:
        quantified_values = xp.zeros((len(unique_regions), channel_count), dtype=xp.float64)
    else:
        quantified_values = xp.zeros((len(unique_regions), channel_count), dtype=xp.float64)

        # Vectorized quantification for each region and channel
        for i, region in enumerate(unique_regions):
            region_mask = masks_xp == region
            if xp.any(region_mask):
                # Extract pixel values for this region across all channels
                region_pixels = image_xp[region_mask]  # Shape: (n_pixels_in_region, channel_count)

                # Apply quantification function to each channel
                for channel_idx in range(channel_count):
                    channel_values = region_pixels[:, channel_idx]
                    if len(channel_values) > 0:
                        # Convert to numpy for function compatibility if needed
                        if xp.__name__ == "cupy":
                            channel_values_np = channel_values.get()
                            quantified_values[i, channel_idx] = quantification_function(channel_values_np)
                        else:
                            quantified_values[i, channel_idx] = quantification_function(channel_values)
                    else:
                        # Handle empty regions - apply function to empty array
                        try:
                            empty_array = np.array([]) if xp.__name__ == "cupy" else xp.array([])
                            quantified_values[i, channel_idx] = quantification_function(empty_array)
                        except (ValueError, RuntimeError):
                            # If function can't handle empty arrays, use NaN
                            quantified_values[i, channel_idx] = xp.nan

    # Convert back to numpy for pandas compatibility
    if use_gpu and xp != np:
        quantified_values = quantified_values.get()  # type: ignore
        unique_regions = unique_regions.get()  # type: ignore

    # Set up channel names
    column_names: List[Union[str, int]]
    if channel_names is None:
        column_names = list(range(channel_count))
    else:
        if len(channel_names) != channel_count:
            msg = f"Number of channel names ({len(channel_names)}) must match number of channels ({channel_count})"
            raise ValueError(msg)
        column_names = channel_names

    # Create DataFrame with proper indexing
    df_quantified = pd.DataFrame(
        quantified_values,
        columns=column_names,
        index=unique_regions,
    )
    df_quantified.index.name = "mask_region"

    return df_quantified
