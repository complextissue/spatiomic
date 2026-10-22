"""Test the count_clusters function."""

from pathlib import Path

import numpy as np
import pytest

from spatiomic.tool import count_clusters


@pytest.mark.parametrize("normalize", [True, False])
def test_count_clusters(normalize: bool, tmp_path: Path) -> None:
    """Test the count_clusters function."""
    test_image = np.random.randint(0, 10, size=(100, 100))

    image_path = tmp_path / "test_image.npy"
    np.save(image_path, test_image)

    df_cluster_counts = count_clusters(
        [str(image_path)],
        cluster_count=10,
        normalize=normalize,
    )

    assert df_cluster_counts.shape == (1, 10)
    assert df_cluster_counts.index[0] == str(image_path)

    if normalize:
        assert np.allclose(df_cluster_counts.sum(axis=1), 1)
