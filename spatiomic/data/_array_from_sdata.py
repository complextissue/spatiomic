from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy.typing import NDArray

from ._subset import subset

if TYPE_CHECKING:
    from spatialdata import SpatialData
else:
    SpatialData = Any


def array_from_sdata(
    data: SpatialData,
    image_name: Optional[str] = None,
    channel_names_subset: Optional[list[str]] = None,
) -> tuple[NDArray, list[str]]:
    """Convert SpatialData image to numpy array.

    Args:
        data (SpatialData): The SpatialData object containing the image.
        image_name (str, optional): The name of the image to convert. If None, the first image will be used.
        channel_names_subset (list[str], optional): The channel names to get from the image. Use this to reorder
            the channels or to select a subset of channels. If None, all channels will be used in the original order.

    Returns:
        tuple[NDArray, list[str]]: A tuple containing the image as a numpy array and the channel names.
    """
    if not data.images:
        raise ValueError("The SpatialData object contains no images.")
    if len(data.images) > 1 and image_name is None:
        raise ValueError("The SpatialData object contains multiple images. Please specify which one to use.")

    if image_name and image_name not in data.images:
        raise ValueError(f"The image '{image_name}' is not found in the SpatialData object.")

    image = data.images[image_name] if image_name else next(iter(data.images.values()))
    channel_names_data = image.coords["c"].values.tolist()

    if image.dims == ("c", "y", "x"):
        image = image.transpose("x", "y", "c")

    pixels = np.asarray(image.to_numpy())

    if channel_names_subset is not None and channel_names_subset != channel_names_data:
        if len(channel_names_subset) > len(channel_names_data):
            raise ValueError("The number of channel names cannot be greater than the number of channels in the image.")

        pixels = subset(
            pixels,
            channel_names_data=channel_names_data,
            channel_names_subset=channel_names_subset,
        )

        channel_names_data = channel_names_subset

    return pixels, channel_names_data
