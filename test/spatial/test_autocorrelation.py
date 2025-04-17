"""Tests the spatial autocorrelation class."""

import numpy as np
import pandas as pd

from spatiomic.spatial import autocorrelation


def test_autocorrelation(
    ordered_data: np.ndarray,
    unordered_data: np.ndarray,
) -> None:
    """Test the spatial auto-correlation class."""
    for example_data in [ordered_data, unordered_data]:
        for permutation_count in [0, 99]:
            for method in ["moran", "geary"]:
                result = autocorrelation().predict(
                    example_data,
                    method=method,  # type: ignore
                    permutation_count=permutation_count,
                )

                assert isinstance(result, pd.DataFrame)
                assert result.shape[0] == len(np.unique(example_data))

                expected_columns = [("cluster" if len(example_data.shape) == 2 else "channel")]

                if method == "moran":
                    expected_columns.extend(["morans_i", "morans_i_expected"])
                else:
                    expected_columns.extend(["gearys_c", "gearys_c_expected"])

                if permutation_count > 0:
                    expected_columns.extend(["p_value", "z_score"])

                assert all(column in result.columns for column in expected_columns), (
                    "The DataFrame columns are not as expected. "
                    f"Expected: {expected_columns}. "
                    f"Actual: {result.columns.tolist()}."
                )

                # check that the values are as expected
                assert permutation_count == 0 or all(0 <= p <= 1 for p in result["p_value"]), (
                    "p-values must be between 0 and 1."
                )
                assert "morans_i" not in result.columns or all(-1 <= i <= 1 for i in result["morans_i"]), (
                    "Moran's I values must be between -1 and 1."
                )
                assert "gearys_c" not in result.columns or all(0 <= c <= 2 for c in result["gearys_c"]), (
                    "Geary's C values must be between 0 and 2."
                )
