from collections.abc import Callable
from inspect import signature
from itertools import combinations
from logging import warning
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, TypeAlias, Union, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from scipy.stats._resampling import PermutationTestResult
from statsmodels.stats.multitest import multipletests

if TYPE_CHECKING:
    from scipy.stats._mannwhitneyu import MannwhitneyuResult
    from scipy.stats._morestats import WilcoxonResult
    from scipy.stats._stats_py import RanksumsResult, TtestResult

SCIPY_TEST_TYPE: TypeAlias = Callable[
    [NDArray, NDArray],
    Union[
        "TtestResult",
        "RanksumsResult",
        "MannwhitneyuResult",
        "WilcoxonResult",
        float,
    ],
]


def _get_test_function(
    test: Literal["t", "wilcoxon", "mwu", "mannwhitneyu"],
    dependent: bool,
) -> SCIPY_TEST_TYPE:
    """Get the statistical test function based on the test and dependent arguments.

    Args:
        test (Literal["t", "wilcoxon", "mwu", "mannwhitneyu"]): The statistical test to be used.
        dependent (bool): Whether the data is dependent.

    Returns:
        callable: The statistical test.
    """
    if test == "mannwhitneyu":
        test = "mwu"

    if test == "t" and not dependent:
        test_function = cast(Callable, stats.ttest_ind)
    elif test == "t" and dependent:
        test_function = cast(Callable, stats.ttest_rel)
    elif test == "wilcoxon" and not dependent:
        test_function = cast(Callable, stats.ranksums)
    elif test == "mwu" and not dependent:
        # this provides similar results to ranksums but with continuity correction
        test_function = cast(Callable, stats.mannwhitneyu)
    elif test == "wilcoxon" and dependent:
        test_function = cast(Callable, stats.wilcoxon)
    else:
        raise NotImplementedError(
            f"The test {test} for {'dependent' if dependent else 'independent'} data is not implemented."
        )

    return test_function


def _permutation_test(
    data: NDArray,
    data_comparison: NDArray,
    test_function: SCIPY_TEST_TYPE,
    dependent: bool,
    permutation_count: int,
    seed: int,
    **kwargs: dict,  # noqa: ARG001
) -> PermutationTestResult:
    """Perform a permutation test to calculate the p-value for the statistical test.

    Args:
        data (NDArray): The data to test.
        data_comparison (NDArray): The data to compare to.
        test_function (Union["ttest_ind", "ttest_rel", "ranksums", "mannwhitneyu", "wilcoxon"]): The statistical test
            function to use after permuting the data.
        dependent (bool): Whether the data is dependent.
        permutation_count (int): The number of permutations to perform.
        seed (int): The random seed to use for the permutations.
        **kwargs: Additional keyword arguments to be passed to the statistical test function.
    """
    test_function_parameters = signature(test_function).parameters
    vectorized = "axis" in test_function_parameters

    def apply_test(
        data: NDArray,
        data_comparison: NDArray,
        axis: Optional[int] = None,
        **kwargs: Dict[str, Any],
    ) -> float:
        """Apply the statistical test to the data."""
        if vectorized:
            kwargs["axis"] = axis  # type: ignore

        if "dependent" in test_function_parameters:
            kwargs["dependent"] = dependent  # type: ignore

        # Ensure we only pass arguments that the function expects
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in test_function_parameters}
        result = test_function(data, data_comparison, **filtered_kwargs)
        return result.statistic if hasattr(result, "statistic") else result  # type: ignore

    return stats.permutation_test(
        [data, data_comparison],
        statistic=apply_test,
        n_resamples=permutation_count,
        permutation_type=("independent" if not dependent else "samples"),
        vectorized=vectorized,
        axis=0,
        rng=seed,
    )


def _calculate_statistics(
    data: NDArray,
    data_comparison: NDArray,
    test_func: Callable,
    dependent: bool,
    permutation_count: Optional[int] = None,
    seed: int = 0,
    **test_kwargs: dict,
) -> dict:
    """Calculate statistics for a comparison between two datasets.

    Args:
        data (NDArray): The data to test.
        data_comparison (NDArray): The data to compare to.
        test_func (Callable): The statistical test function.
        dependent (bool): Whether the data is dependent.
        permutation_count (Optional[int]): Number of permutations to perform.
        seed (int): Random seed for reproducibility.
        **test_kwargs: Additional keyword arguments for the test function.

    Returns:
        dict: Dictionary containing the test results.
    """
    result = {}

    # Calculate means
    data_group_mean = np.mean(data)
    data_comparison_mean = np.mean(data_comparison)
    result["mean_group"] = data_group_mean
    result["mean_comparison"] = data_comparison_mean

    # Perform statistical test
    if permutation_count is not None and permutation_count > 0:
        test_result = _permutation_test(
            data,
            data_comparison,
            test_func,
            dependent,
            permutation_count or 100,
            seed,
            **test_kwargs,
        )
        p_value = test_result.pvalue
        ranksum = test_result.statistic if test_func in [stats.wilcoxon, stats.ranksums, stats.mannwhitneyu] else None
    else:
        test_result = test_func(data, data_comparison, **test_kwargs)
        p_value = test_result.pvalue
        ranksum = test_result.statistic if test_func in [stats.wilcoxon, stats.ranksums, stats.mannwhitneyu] else None

    result["p_value"] = p_value
    result["log10_p_value"] = -np.log10(p_value + (1e-38 if np.any(p_value < 1e-38) else 0))
    result["ranksum"] = ranksum

    # Calculate log fold change if possible
    if np.any(data < 0) or np.any(data_comparison < 0):
        warning("Data contains negative values. Cannot calculate log fold change.")
        result["log2_fold_change"] = None
    else:
        pseudocount = 1e-38 if (np.any(data_group_mean < 1e-38) or np.any(data_comparison_mean < 1e-38)) else 0
        result["log2_fold_change"] = np.log2(data_group_mean + pseudocount) - np.log2(
            data_comparison_mean + pseudocount
        )

    return result


def get_stats(
    data: NDArray,
    group: Union[NDArray, List[Union[str, int]]],
    channel_names: Optional[List[str]] = None,
    comparison: Union[Literal["all", "each"], str, int, List[Union[str, int]]] = "all",
    is_log1p: bool = False,
    test: Union[Literal["t", "wilcoxon", "mwu", "mannwhitneyu", "permutation"], Callable] = "t",
    dependent: bool = False,
    equal_variance: Optional[bool] = None,
    correction: Union[Literal["holm-sidak", "bonferroni", "fdr"], None] = "holm-sidak",
    correction_family_alpha: float = 0.05,
    permutation_count: Optional[int] = None,
    permutation_seed: int = 0,
    test_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """Calculate the statistics for marker or cluster expression/abundance between groups.

    .. warning:: A pseudo count of 1e-38 is added to the data to avoid log(0) errors when calculating log fold change
        and log10 p-values.

    Usage:

    .. code-block:: python

        data = [
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
        ]
        group = [
            "healthy",
            "healthy",
            "healthy",
            "healthy",
            "disease",
            "disease",
            "disease",
            "disease",
            "treated",
            "treated",
            "treated",
            "treated",
        ]
        # if only one comparison group is provided, all other groups are compared to it
        comparison = "healthy"

        df = get_stats(
            data,
            group,
            comparison,
            test="t",
            dependent=False,
            correction="bonferroni",
        )

    Args:
        data (NDArray): Image data to calculate the statistics for.
        group (Union[NDArray, List[Union[str, int]]]): The group labels for each pixel.
        channel_names (Optional[List[str]], optional): The names of the channels. Defaults to None.
        comparison (Union[Literal["all", "each"], str, int, List[Union[str, int]]]): The group labels to compare to.
            Defaults to "all".
        is_log1p (bool, optional): Whether the data is log1p transformed. Defaults to False.
        test (Literal["t", "wilcoxon", "mwu", "mannwhitneyu"], optional): The statistical test to be used. When
            permutation_count is not None, this refers to the statistic to use for the permutation test. You can provide
            your own statistic that accepts the data to compare as the first two positional arguments and also accepts
            **kwargs. If the function has an `axis` and or a `dependent` parameter, the respective arguments will also
            be passed to the custom function.
            Defaults to "t".
        dependent (bool, optional): Whether the data is dependent. Defaults to False.
        equal_variance (bool, optional): Whether to assume equal variance for the t-test. If False or None, the Welch's
            t-test is used. Defaults to None.
        correction (Union[Literal["holmsidak", "bonferroni", "fdr"], None], optional): The correction to apply to the
            p-values to control the family-wise error rate.
            Defaults to "holmsidak".
        correction_family_alpha (float, optional): The family-wise alpha value to use for the correction. Currently
            has no effect as only the corrected p-values are returned but may be used in the future.
            Defaults to 0.05.
        permutation_count (int, optional): The number of permutations to perform. If None or 0, no permutation test is
            performed. Defaults to None.
        permutation_seed (int, optional): The random seed to use for the permutations. Defaults to 0.
        test_kwargs (dict, optional): Additional keyword arguments to be passed to the statistical test function.
            Defaults to None

    Returns:
        pd.DataFrame: A pandas DataFrame containing the statistics for each marker and comparison.
    """
    assert data.ndim >= 2, "The data must have at least two dimensions."
    data_shape = data.shape
    data = data.reshape((-1, data_shape[-1]))

    assert "all" not in list(group), "The group labels cannot contain the string 'all'."
    assert "each" not in list(group), "The group labels cannot contain the string 'each'."

    if isinstance(group, np.ndarray):
        group = group.flatten()

    assert len(group) == np.multiply.reduce(data.shape[:-1]), "The group labels must have the same length as the data."

    unique_groups = list(set(group))
    unique_group_idx: List[int] = list(np.arange(len(unique_groups)))
    group_idx = np.array([unique_groups.index(g) for g in group])

    if is_log1p:
        from spatiomic.process import log1p

        data = log1p(use_gpu=False).inverse_transform(data)

    group_data = [data[group_idx == idx, :] for idx in unique_group_idx]

    test_kwargs = {} if not isinstance(test_kwargs, dict) else test_kwargs
    if test == "t" and not dependent and equal_variance is not None:
        test_kwargs["equal_var"] = equal_variance
    elif equal_variance is not None:
        raise ValueError("The equal_variance argument can only be used with independent t-tests.")

    test_func = test if callable(test) else _get_test_function(test, dependent)  # type: ignore[arg-type]
    comparisons: List[Tuple[int, int]] = []

    # get all possible comparisons
    if comparison == "all":
        # Compare each group with "all"
        comparisons = [(idx, -1) for idx in unique_group_idx]
    elif comparison == "each":
        # Compare each pair of different groups
        comparisons = list(combinations(unique_group_idx, 2))
    elif isinstance(comparison, (str, int)):
        # Compare each group with the specified comparison group
        comparison_index = unique_groups.index(comparison)
        comparisons = [(unique_group_idx[idx], comparison_index) for idx in unique_group_idx if idx != comparison_index]
    elif isinstance(comparison, list):
        # Compare each group with multiple specified comparison groups
        comparison_indices = [unique_groups.index(comp) for comp in comparison]
        other_indices = [idx for idx in unique_group_idx if idx not in comparison_indices]

        comparisons = [(idx, comp_idx) for idx in other_indices for comp_idx in comparison_indices]

    # calculate the statistics for each comparison
    comparison_results: List[dict] = []

    for data_idx, comparison_idx in comparisons:
        for marker_idx in range(0, data_shape[1]):
            data_group = group_data[data_idx][:, marker_idx]

            if comparison_idx == -1:
                # compare the group with all other groups
                comparison_name = "all"
                data_comparison = np.concatenate(
                    [group_data[idx][:, marker_idx] for idx in unique_group_idx if idx != data_idx]
                )
            else:
                # compare the group with the specified comparison group
                comparison_name = unique_groups[comparison_idx]
                data_comparison = group_data[comparison_idx][:, marker_idx]

            stats_result = _calculate_statistics(
                data_group,
                data_comparison,
                test_func,
                dependent,
                permutation_count,
                permutation_seed,
                **test_kwargs,
            )

            result = {
                "group": unique_groups[data_idx],
                "comparison": comparison_name,
                "marker": channel_names[marker_idx] if channel_names is not None else marker_idx,
                "p_value_corrected": None,
                "log10_p_value_corrected": None,
                **stats_result,
            }

            comparison_results.append(result)

    df_comparisons = pd.DataFrame(comparison_results)

    if correction is not None:
        df_comparisons["p_value"] = df_comparisons["p_value"].fillna(1.0)

        switch_correction = {
            "bonferroni": "bonferroni",
            "fdr": "fdr_bh",
            "holm-sidak": "holm-sidak",
        }
        correction_method = switch_correction[correction]

        _reject, pvals_corrected, _alpha_sidak, _alpha_bonferroni = multipletests(
            df_comparisons["p_value"].values,
            alpha=correction_family_alpha,
            method=correction_method,
        )
        df_comparisons["p_value_corrected"] = pvals_corrected

        df_comparisons["p_value_corrected"] = df_comparisons["p_value_corrected"].fillna(1.0)
        df_comparisons["log10_p_value_corrected"] = -np.log10(
            df_comparisons["p_value_corrected"]
            + (1e-38 if np.any(np.array(df_comparisons["p_value_corrected"].values) < 1e-38) else 0)
        )

    return df_comparisons
