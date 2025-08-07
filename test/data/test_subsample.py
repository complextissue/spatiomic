"""Test the Subsample class."""

from anndata import AnnData
from numpy.typing import NDArray

from spatiomic.data import subsample


def test_subsample(
    image_pixels: NDArray,
) -> None:
    """Test the Subsample class.

    Args:
        image_pixels (NDArray): The example image.
    """
    image_pixels = image_pixels.reshape((-1, image_pixels.shape[-1]))
    subsample_pixels = subsample.fit_transform(image_pixels, fraction=0.1)

    assert subsample_pixels.shape[0] == int(0.1 * image_pixels.shape[0])

    # check with AnnData
    adata = AnnData(image_pixels)
    _ = subsample.fit_transform(adata, fraction=0.1, output_unstructured_name="X_subsample")

    assert adata.uns["X_subsample"].shape[0] == int(0.1 * image_pixels.shape[0])
