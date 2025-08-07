"""Test the count_clusters function."""

import os

import numpy as np

from spatiomic.tool import count_clusters


def test_count_clusters() -> None:
    """Test the count_clusters function."""
    test_image = np.random.randint(0, 10, size=(100, 100))

    np.save("test_image.npy", test_image)

    for normalize in [True, False]:
        df_cluster_counts = count_clusters(
            ["test_image.npy"],
            cluster_count=10,
            normalize=normalize,
        )

        assert df_cluster_counts.shape == (1, 10)
        assert df_cluster_counts.index[0] == "test_image.npy"

        if normalize:
            assert np.allclose(df_cluster_counts.sum(axis=1), 1)

    os.remove("test_image.npy")
