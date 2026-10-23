"""Test the spatial weights construction."""

from typing import Literal

import numpy as np
import pytest

import spatiomic as so


@pytest.mark.parametrize(
    ("neighborhood_type", "order", "distance"),
    [
        ("queen", 1, lambda dx, dy: np.maximum(abs(dx), abs(dy))),
        ("queen", 2, lambda dx, dy: np.maximum(abs(dx), abs(dy))),
        ("rook", 2, lambda dx, dy: abs(dx) + abs(dy)),
    ],
)
def test_spatial_weights(
    neighborhood_type: Literal["queen", "rook"],
    order: int,
    distance: np.ufunc,
) -> None:
    """Test that every pixel is connected exactly to the pixels within the neighborhood order."""
    data_shape = (7, 9)
    offset = so.spatial.neighborhood_offset(neighborhood_type=neighborhood_type, order=order)

    graph = so.spatial.spatial_weights(data_shape=data_shape, neighborhood_offset=offset)

    rows, cols = np.unravel_index(np.arange(data_shape[0] * data_shape[1]), data_shape)
    pixel_distances = distance(rows[:, None] - rows[None], cols[:, None] - cols[None])
    expected_adjacency = (pixel_distances > 0) & (pixel_distances <= order)

    np.testing.assert_array_equal(graph.sparse.toarray() > 0, expected_adjacency)
