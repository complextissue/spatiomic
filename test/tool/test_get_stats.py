"""Test the get_stats function."""

from typing import Literal

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from spatiomic.tool import get_stats


@pytest.mark.parametrize("group_key", ["numeric", "letter"])
@pytest.mark.parametrize("log1p", [False, True])
def test_get_stats(group_key: Literal["numeric", "letter"], log1p: bool) -> None:
    """Test the get_stats function."""
    np.random.seed(42)
    n_samples = 100
    n_markers = 10
    data = np.random.rand(n_samples, n_markers)
    group = (
        np.random.choice([1, 2, 3], size=n_samples).tolist()
        if group_key == "numeric"
        else np.random.choice(["A", "B", "C"], size=n_samples)
    )
    channel_names = [f"marker_{i}" for i in range(n_markers)]

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


@pytest.mark.parametrize("equal_variance", [False, True])
def test_get_stats_t_permutation_all(equal_variance: bool) -> None:
    """Test that one-vs-all permutation t-tests match scipy's permutation test, also for channel-last 3D data."""
    rng = np.random.default_rng(0)
    data = rng.gamma(2.0, size=(10, 30, 4))
    data[:5, :, 0] += 3.0  # a clear effect for the first group in the first marker
    group = np.repeat(["A", "B", "C"], [50, 150, 100])
    permutation_count = 5000

    result = get_stats(
        data,
        group,
        comparison="all",
        test="t",
        equal_variance=equal_variance,
        permutation_count=permutation_count,
        correction=None,
    )

    def t_statistic(data_group: np.ndarray, data_comparison: np.ndarray, axis: int) -> np.ndarray:
        return stats.ttest_ind(data_group, data_comparison, equal_var=equal_variance, axis=axis).statistic

    data_flat = data.reshape(-1, data.shape[-1])
    assert len(result) == 3 * data.shape[-1]

    for row in result.itertuples():
        expected = stats.permutation_test(
            [data_flat[group == row.group, row.marker], data_flat[group != row.group, row.marker]],
            t_statistic,
            n_resamples=permutation_count,
            vectorized=True,
            rng=0,
        ).pvalue

        assert row.mean_group == pytest.approx(data_flat[group == row.group, row.marker].mean())
        # both are Monte Carlo estimates from different resamples
        assert row.p_value == pytest.approx(expected, abs=0.03)

    assert result.query("group == 'A' and marker == 0")["p_value"].item() < 1e-3


@pytest.mark.parametrize("equal_variance", [False, True])
def test_get_stats_t_permutation_equal_variance(equal_variance: bool) -> None:
    """Test that pairwise permutation t-tests use Student's or Welch's t-statistic as requested."""
    rng = np.random.default_rng(0)
    group = np.repeat(["A", "B"], [20, 80])
    data = rng.normal(size=(100, 2)) * np.where(group == "A", 4.0, 1.0)[:, np.newaxis]
    data[group == "A"] += np.array([0.0, 1.0])
    permutation_count = 200

    result = get_stats(
        data,
        group,
        comparison="B",
        test="t",
        equal_variance=equal_variance,
        permutation_count=permutation_count,
        correction=None,
    )

    for row in result.itertuples():
        expected = {
            equal_var: stats.permutation_test(
                [data[group == "A", row.marker], data[group == "B", row.marker]],
                lambda x, y, axis, equal_var=equal_var: stats.ttest_ind(x, y, equal_var=equal_var, axis=axis).statistic,
                n_resamples=permutation_count,
                vectorized=True,
                rng=0,
            ).pvalue
            for equal_var in (False, True)
        }

        assert expected[False] != expected[True]
        assert row.p_value == expected[equal_variance]
