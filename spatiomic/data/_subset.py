from typing import List

from numpy.typing import NDArray

from spatiomic._internal._check_channel_dimension import check_channel_dimension


def subset(
    data: NDArray,
    channel_names_data: List[str],
    channel_names_subset: List[str],
) -> NDArray:
    """Subset the data with the given channel names to the subset channel names in the given order.

    Args:
        data (NDArray): The image data, channel-last.
        channel_names_data (List[str]): The channel names of the data.
        channel_names_subset (List[str]): The channel names to include in the subset.

    Returns:
        NDArray: The subsetted data, with channel ordered according to the subset channel names.
    """
    check_channel_dimension(data.shape)

    # Check that each of the subset channel names is in the data channel names
    assert all(name in channel_names_data for name in channel_names_subset), (
        "Each subset channel name must be in the data channel names."
    )

    # Get the indices of the subset channels
    indices = [channel_names_data.index(name) for name in channel_names_subset]

    # Subset the data
    return data[..., indices]
