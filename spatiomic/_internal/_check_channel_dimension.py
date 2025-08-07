from typing import Optional
from warnings import warn


def check_channel_dimension(
    data_shape: tuple,
    dimension_count_min: Optional[int] = 2,
    dimension_count_max: Optional[int] = None,
) -> None:
    """Check if the data has a channel dimension and warn if not or raise an error if the data is 2D and square.

    Args:
        data_shape (tuple): The shape of the data.
        dimension_count_min (Optional[int], optional): The minimum number of dimensions the data must have.
            Defaults to 2.
        dimension_count_max (Optional[int], optional): The maximum number of dimensions the data must have.
            Defaults to None.

    Raises:
        ValueError: If the data is 2D and square.

    Warns:
        UserWarning: If the data is 2D and has no channel dimension but is not square.
    """
    if dimension_count_min is not None:
        assert len(data_shape) >= dimension_count_min, f"Data must have at least {dimension_count_min} dimensions."
    if dimension_count_max is not None:
        assert len(data_shape) <= dimension_count_max, f"Data must have at most {dimension_count_max} dimensions."

    if len(data_shape) == 2:
        # check if the image is close to a square
        if abs(data_shape[0] - data_shape[1]) < max(data_shape) * 0.2:
            raise ValueError(
                "Data has only two dimensions and is square, assuming image without channel dimension."
                "Please add a channel dimension to the data for correct processing."
            )
        else:
            warn(
                "Data has only two dimensions. Assuming flat data with one channel dimension."
                " This will yield unwanted results when called on 2D images without channel dimension.",
                UserWarning,
                stacklevel=2,
            )
