import numpy as np
from numpy.typing import NDArray


def check_cluster_image(
    data: NDArray,
    allow_unassigned: bool = True,
) -> None:
    """Check that the data is an image of integer cluster labels.

    Cluster labels are non-negative integers. The label -1 marks unassigned pixels, e.g., pixels without a nearest SOM
    node from `so.dimension.som().label`.

    Args:
        data (NDArray): The clustered image.
        allow_unassigned (bool, optional): Whether pixels may have the label -1. Defaults to True.

    Raises:
        TypeError: If the data is not of integer type.
        ValueError: If the data contains labels below -1, or -1 when `allow_unassigned` is False.
    """
    if not np.issubdtype(data.dtype, np.integer):
        raise TypeError(f"Clustered images must be of integer type, got {data.dtype}.")

    label_min = -1 if allow_unassigned else 0
    if data.size > 0 and np.min(data) < label_min:
        raise ValueError(
            f"Clustered images must not contain labels below {label_min}"
            + (" (-1 marks unassigned pixels)." if allow_unassigned else ".")
        )
