from typing import Any, Literal, Tuple

import numpy as np
from libpysal.weights import W
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from tqdm import tqdm


def neighborhood_offset(
    neighborhood_type: Literal["queen", "rook", "circle"] = "queen",
    order: int = 1,
    exclude_inner_order: int = 0,
) -> NDArray:
    """Create a neighborhood offset array for a given neighborhood type and steps.

    For queen neighborhoods, the order parameter defines the maximum Chebyshev distance from the focal cell to the
    neighbor. For rook neighborhoods, the order parameter defines the maximum Manhattan distance from the focal cell
    to the neighbor. For circle neighborhoods, the order parameter defines the radius from the focal cell to the
    neighbor.

    E.g., for a queen neighborhood with order=1 and exclude_inner_order=0, the neighborhood offset array would be:
    array([[-1, -1],
           [-1,  0],
           [-1,  1],
           [ 0, -1],
           [ 0,  1],
           [ 1, -1],
           [ 1,  0],
           [ 1,  1]])

    Args:
        neighborhood_type (Literal["queen", "rook", "circle"], optional): The type of neighborhood to create.
            Defaults to "queen".
        order (int, optional): The order of the neighborhood. Defaults to 1.
        exclude_inner_order (int, optional): The inner order to exclude from the neighborhood. Defaults to 0.

    Returns:
        NDArray: The neighborhood offset array.
    """
    assert order > 0, "`order` must be greater than 0"
    assert exclude_inner_order >= 0, "`exclude_inner_order` must be greater than or equal to 0"
    assert exclude_inner_order < order, "`exclude_inner_order` must be smaller than order"

    offsets = []
    for i in range(-order, order + 1):
        for j in range(-order, order + 1):
            if neighborhood_type == "rook":
                manhattan_distance = abs(i) + abs(j)
                if 0 < manhattan_distance <= order and manhattan_distance > exclude_inner_order:
                    offsets.append((i, j))
            elif neighborhood_type == "circle":
                euclidean_distance = np.sqrt(i**2 + j**2)
                if 0 < euclidean_distance <= order and euclidean_distance > exclude_inner_order:
                    offsets.append((i, j))
            else:  # neighborhood_type == "queen" or default
                chebyshev_distance = max(abs(i), abs(j))
                if 0 < chebyshev_distance <= order and chebyshev_distance > exclude_inner_order:
                    offsets.append((i, j))

    return np.array(offsets)


def spatial_weights(
    data_shape: Tuple[int, int],
    neighborhood_offset: NDArray,
) -> Any:
    """Create a spatial weights matrix from a given data shape and neighborhood offset.

    .. code-block:: python

        from spatiomic.spatial import neighborhood_offset, spatial_weights

        data_shape = (3, 3)
        neighborhood_offset = neighborhood_offset(
            neighborhood_type="queen",
            steps=5,
            exclude_inner_steps=2,
        )
        spatial_weights_matrix = spatial_weights(
            data_shape,
            neighborhood_offset,
        )

    Args:
        data_shape (tuple[int, int]): The shape of the data.
        neighborhood_offset (NDArray): The neighborhood offset array.

    Returns:
        NDArray: The spatial weights matrix.
    """
    rows, cols = data_shape
    cell_count = rows * cols

    # Create the grid and get flattened indices
    grid_indices = np.arange(cell_count).reshape(data_shape)

    focals_list = []
    neighbors_list = []
    weight_list = []

    for offset_x, offset_y in tqdm(neighborhood_offset, desc="Creating spatial weights for each offset"):
        if offset_x >= 0:
            focal_range_x = slice(0, rows - offset_x) if offset_x > 0 else slice(0, rows)
            neighbor_range_x = slice(offset_x, rows)
        else:
            focal_range_x = slice(abs(offset_x), rows)
            neighbor_range_x = slice(0, rows + offset_x)

        if offset_y >= 0:
            focal_range_y = slice(0, cols - offset_y) if offset_y > 0 else slice(0, cols)
            neighbor_range_y = slice(offset_y, cols)
        else:
            focal_range_y = slice(abs(offset_y), cols)
            neighbor_range_y = slice(0, cols + offset_y)

        # Get source and target indices
        focal_indices = grid_indices[focal_range_x, focal_range_y].flatten()
        neighbor_indices = grid_indices[neighbor_range_x, neighbor_range_y].flatten()

        focals_list.append(focal_indices)
        neighbors_list.append(neighbor_indices)
        weight_list.append(np.full(len(focal_indices), 1.0))

    if not focals_list:
        sparse_matrix = coo_matrix((np.array([]), (np.array([]), np.array([]))), shape=(cell_count, cell_count))
    else:
        sparse_matrix = coo_matrix(
            (
                np.concatenate(weight_list),
                (np.concatenate(focals_list), np.concatenate(neighbors_list)),
            ),
            shape=(cell_count, cell_count),
        )

    return W.from_sparse(sparse_matrix)
