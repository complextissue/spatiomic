"""Tests the pixel vicinity function."""

import numpy as np
import pandas as pd

from spatiomic.spatial import vicinity_composition


def test_vicinity_composition(
    clustered_data: np.ndarray,
) -> None:
    """Test the pixel vicinity function."""
    result = vicinity_composition(clustered_data.astype(np.uint16))

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (len(np.unique(clustered_data)), len(np.unique(clustered_data)))
    assert all(result.index == result.columns), "The DataFrame index and columns are not the same."

    assert all(result.index == np.unique(clustered_data)), (
        "The DataFrame index is not the same as the unique values in the data."
    )

    assert all(result.columns == np.unique(clustered_data)), (
        "The DataFrame columns are not the same as the unique values in the data."
    )

    assert all(result.index == np.unique(clustered_data)), (
        "The DataFrame index is not the same as the unique values in the data."
    )

    assert all(result.columns == np.unique(clustered_data)), (
        "The DataFrame columns are not the same as the unique values in the data."
    )

    assert all(result.values.diagonal() == 0), "The DataFrame diagonal values are not all 0."

    assert np.array_equal(result.values, result.values.T), "The DataFrame values are not symmetric."
