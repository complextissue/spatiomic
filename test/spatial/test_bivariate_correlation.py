"""Tests the spatial correlation class."""

from typing import Literal

import numpy as np
import pandas as pd
import pytest

from spatiomic.spatial import bivariate_correlation


@pytest.mark.parametrize("method", ["moran", "lee"])
@pytest.mark.parametrize("permutation_count", [0, 99])
def test_bivariate_correlation(
    cluster_assignments: np.ndarray,
    method: Literal["moran", "lee"],
    permutation_count: int,
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

    df_morans_i = bivariate_correlation_quantifier.predict(
        cluster_assignments,
        channel_image=channel_image,
        channel_names=channel_names,
        method=method,
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


@pytest.mark.parametrize("method", ["moran", "lee"])
def test_bivariate_correlation_detects_colocalization(method: Literal["moran", "lee"]) -> None:
    """Test that a channel marking a spatially contiguous cluster correlates with it, and a random channel does not."""
    rng = np.random.default_rng(0)
    clusters = np.kron(rng.integers(0, 4, size=(8, 8)), np.ones((8, 8), dtype=int))
    channel_image = np.stack(
        [(clusters == 0) + rng.normal(scale=0.1, size=clusters.shape), rng.random(clusters.shape)],
        axis=-1,
    )

    df_correlation = (
        bivariate_correlation()
        .predict(clusters, channel_image=channel_image, channel_names=["marker", "random"], method=method)
        .set_index(["cluster", "channel"])["morans_i"]
    )

    assert df_correlation[(0, "marker")] > 0.5
    assert abs(df_correlation[(0, "random")]) < 0.05
