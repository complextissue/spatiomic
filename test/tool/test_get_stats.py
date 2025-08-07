"""Test the get_stats function."""

import numpy as np
import pandas as pd

from spatiomic.tool import get_stats


def test_get_stats() -> None:
    """Test the get_stats function."""
    # Generate some test data
    np.random.seed(42)
    n_samples = 100
    n_markers = 10
    data = np.random.rand(n_samples, n_markers)
    group_numbers = np.random.choice([1, 2, 3], size=n_samples).tolist()
    group_letters = np.random.choice(["A", "B", "C"], size=n_samples)
    channel_names = [f"marker_{i}" for i in range(n_markers)]

    for group in [group_numbers, group_letters]:
        for log1p in [False, True]:
            # Test using t-test
            result_t = get_stats(
                data,
                group,
                channel_names=channel_names,
                comparison="all",
                test="t",
                is_log1p=log1p,
                dependent=False,
                correction="holm-sidak",
            )
            assert isinstance(result_t, pd.DataFrame)

            # Test using Wilcoxon rank sum test
            result_wilcoxon = get_stats(
                data,
                group,
                comparison=group[1],
                test="wilcoxon",
                is_log1p=log1p,
                dependent=False,
                correction="fdr",
            )
            assert isinstance(result_wilcoxon, pd.DataFrame)

            # Test using permutation test
            result_permutation = get_stats(
                data,
                group,
                comparison=group[1],
                test="t",
                is_log1p=log1p,
                dependent=False,
                correction="fdr",
                permutation_count=100,
            )

            # Ensure the DataFrame columns are as expected
            expected_columns = sorted(
                [
                    "group",
                    "comparison",
                    "marker",
                    "mean_group",
                    "mean_comparison",
                    "p_value",
                    "p_value_corrected",
                    "log10_p_value",
                    "log10_p_value_corrected",
                    "ranksum",
                    "log2_fold_change",
                ]
            )
            assert result_t.columns.sort_values().tolist() == expected_columns
            assert result_wilcoxon.columns.sort_values().tolist() == expected_columns
            assert result_permutation.columns.sort_values().tolist() == expected_columns

            # check that group and comparison columns are never the same in the same row
            assert not result_wilcoxon.apply(lambda x: x["group"] == x["comparison"], axis=1).any()
            assert not result_t.apply(lambda x: x["group"] == x["comparison"], axis=1).any()
            assert not result_permutation.apply(lambda x: x["group"] == x["comparison"], axis=1).any()

            # Check that p-value columns are not NaN
            assert not result_t["p_value"].isna().any()
            assert not result_wilcoxon["p_value"].isna().any()
            assert not result_permutation["p_value"].isna().any()

            # Check that p-value corrected columns are not NaN
            assert not result_t["p_value_corrected"].isna().any()
            assert not result_wilcoxon["p_value_corrected"].isna().any()
            assert not result_permutation["p_value_corrected"].isna().any()

            # Check that ranksum columns are not NaN
            assert result_t["ranksum"].isna().all(), "t-test should not have ranksum"
            assert not result_wilcoxon["ranksum"].isna().any(), "Wilcoxon should have ranksum"
