"""Tests the spatial correlation class."""

import numpy as np
import pandas as pd

from spatiomic.spatial import bivariate_correlation


def test_bivariate_correlation(
    cluster_assignments: np.ndarray,
) -> None:
    """Test the spatial correlation class that compares markers and cluster assignments."""
    channel_image = np.random.rand(64, 64, 4)

    channel_names = [
        "CD3",
        "CD4",
        "CD8",
        "CD20",
    ]

    bivariate_correlation_quantifier = bivariate_correlation()

    for permutation_count in [0, 99]:
        df_morans_i = bivariate_correlation_quantifier.predict(
            cluster_assignments,
            channel_image=channel_image,
            channel_names=channel_names,
            method="moran",
            permutation_count=permutation_count,
        )

        # check the resulting DataFrame
        assert isinstance(df_morans_i, pd.DataFrame)
        assert len(df_morans_i) == len(channel_names) * len(np.unique(cluster_assignments))

        assert "cluster" in df_morans_i.columns
        assert "channel" in df_morans_i.columns
        assert "morans_i" in df_morans_i.columns

        if permutation_count > 0:
            assert "p_value" in df_morans_i.columns
            assert "z_score" in df_morans_i.columns
