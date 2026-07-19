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

    assert all(result.to_numpy().diagonal() == 0), "The DataFrame diagonal values are not all 0."

    assert np.array_equal(result.to_numpy(), result.to_numpy().T), "The DataFrame values are not symmetric."


def test_vicinity_composition_permutation(
    clustered_data: np.ndarray,
) -> None:
    """The permutation path returns valid p-values with the correct shape."""
    n = len(np.unique(clustered_data))

    result, p_values = vicinity_composition(
        clustered_data.astype(np.uint16),
        permutations=20,
        seed=0,
        n_jobs=1,
    )

    assert isinstance(result, pd.DataFrame)
    assert isinstance(p_values, pd.DataFrame)
    assert p_values.shape == (n, n)

    # p-values must lie in (0, 1]
    p_values_array = p_values.to_numpy()
    finite_p_values = p_values_array[~np.isnan(p_values_array)]
    assert np.all((finite_p_values > 0) & (finite_p_values <= 1))

    # ignore_identities defaults to True, so the diagonal is NaN
    assert np.all(np.isnan(np.diag(p_values_array)))
