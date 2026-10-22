"""Test that cluster plots show each cluster in the color with the same index in the colormap."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import ListedColormap
from numpy.typing import NDArray

import spatiomic as so


@pytest.fixture
def cluster_labels() -> NDArray:
    """Create a clustered image with unassigned pixels (-1) that lacks clusters 1, 4 and 5."""
    return np.random.default_rng(0).choice([-1, 0, 2, 3], size=(20, 30)).astype(np.int32)


@pytest.fixture
def cluster_colormap() -> ListedColormap:
    """Create a colormap with more colors than clusters in the image."""
    return so.plot.colormap(color_count=6, flavor="random_hls", seed=0)


def get_expected_colors(labels: NDArray, colormap: ListedColormap) -> NDArray:
    """Get the color of each label from the colormap, transparent for unassigned labels."""
    colors = np.zeros((*labels.shape, 4))
    assigned = labels >= 0
    colors[assigned, :3] = np.asarray(colormap.colors)[labels[assigned]]
    colors[assigned, 3] = 1.0
    return colors


@pytest.mark.cpu
def test_cluster_image_colors(cluster_labels: NDArray, cluster_colormap: ListedColormap) -> None:
    """Test that the image and colorbar colors do not depend on which clusters are present."""
    fig = so.plot.cluster_image(cluster_labels, colormap=cluster_colormap)

    np.testing.assert_allclose(fig.axes[0].images[0].get_array(), get_expected_colors(cluster_labels, cluster_colormap))
    np.testing.assert_allclose(
        fig.axes[1].images[0].get_array()[:, 0, :3], np.asarray(cluster_colormap.colors)[: cluster_labels.max() + 1]
    )
    plt.close(fig)


@pytest.mark.cpu
def test_save_cluster_image_colors(cluster_labels: NDArray, cluster_colormap: ListedColormap, tmp_path: Path) -> None:
    """Test that saved cluster images use the same colors as the colormap."""
    save_path = str(tmp_path / "clusters.png")
    so.plot.save_cluster_image(cluster_labels, save_path, colormap=cluster_colormap)

    np.testing.assert_allclose(
        plt.imread(save_path), get_expected_colors(cluster_labels, cluster_colormap), atol=1 / 255
    )


@pytest.mark.cpu
def test_cluster_legend_colors(cluster_colormap: ListedColormap) -> None:
    """Test that the legend shows the first colors of the colormap, even if it has more colors than clusters."""
    fig = so.plot.cluster_legend(cluster_count=4, colormap=cluster_colormap)

    np.testing.assert_allclose(fig.axes[0].images[0].get_array()[0, :, :3], np.asarray(cluster_colormap.colors)[:4])
    plt.close(fig)


@pytest.mark.cpu
def test_cluster_selection_colors(cluster_labels: NDArray, cluster_colormap: ListedColormap) -> None:
    """Test that selected clusters keep their colors and all other pixels, including unassigned ones, are black."""
    fig = so.plot.cluster_selection(cluster_labels, clusters=[2, 3], colormap=cluster_colormap)

    expected_colors = get_expected_colors(cluster_labels, cluster_colormap)
    unselected = ~np.isin(cluster_labels, [2, 3])
    expected_colors[unselected] = [0.0, 0.0, 0.0, 1.0]

    np.testing.assert_allclose(fig.axes[0].images[0].get_array(), expected_colors)
    plt.close(fig)


@pytest.mark.cpu
def test_cluster_location_skips_unassigned(cluster_labels: NDArray) -> None:
    """Test that unassigned pixels are not shown as a cluster."""
    fig = so.plot.cluster_location(cluster_labels)

    assert [ax.get_title() for ax in fig.axes[1:]] == ["Cluster 0", "Cluster 2", "Cluster 3"]
    plt.close(fig)


@pytest.mark.cpu
@pytest.mark.parametrize("colormap", ["flag", matplotlib.colormaps["prism"]], ids=["name", "continuous"])
def test_cluster_image_continuous_colormap(colormap: str | matplotlib.colors.Colormap, tmp_path: Path) -> None:
    """Test that colormap names and continuous colormaps are spread from label 0 to the highest label."""
    masks = np.random.default_rng(0).integers(0, 300, size=(20, 30))
    masks[0, :5] = -1
    colormap_resolved = matplotlib.colormaps[colormap] if isinstance(colormap, str) else colormap

    expected_colors = colormap_resolved(masks / masks.max())
    expected_colors[masks < 0] = 0.0

    ax = so.plot.cluster_image(masks, colormap=colormap, ax=plt.figure().add_subplot())
    np.testing.assert_allclose(ax.images[0].get_array(), expected_colors)
    plt.close("all")

    save_path = str(tmp_path / "masks.png")
    so.plot.save_cluster_image(masks, save_path, colormap=colormap)
    np.testing.assert_allclose(plt.imread(save_path), expected_colors, atol=1 / 255)
