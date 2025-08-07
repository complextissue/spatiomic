from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from spatiomic._internal._anndata_method import anndata_method
from spatiomic._internal._data_method import data_method
from spatiomic._internal._seed_method import seed_method


class Subsample:
    """A class to subsample image data."""

    @staticmethod
    @anndata_method(input_attribute="X", output_uns="X_subsample")
    @data_method
    @seed_method
    def fit_transform(
        data: NDArray,
        method: Literal["fraction", "count"] = "fraction",
        fraction: float = 0.1,
        count: int = 100000,
        *args: Any,  # noqa: ARG004
        **kwargs: dict,  # noqa: ARG004
    ) -> NDArray:
        """Create a random subsample of some image data pixels, preserving channel dimensionality.

        Args:
            data (NDArray): Image data to take the subsample from.
            method (Literal["fraction", "count"], optional): The method to use. Defaults to "fraction".
            fraction (float, optional): The fraction of pixels to take when using the "fraction" method. When this
                results in a non-integer number of pixels, the number of pixels is rounded down.
                Defaults to 0.1.
            count (int, optional): The number of pixels to take. Defaults to 10000 when using the "count" method.
                Defaults to 100,000.
            seed (Optional[int], optional): The seed to use for random number generation. Defaults to None.

        Returns:
            NDArray: A flattened, random subsample of the pixels in the provided image data.
        """
        img_shape = data.shape
        pixel_count = np.multiply.reduce(img_shape[:-1])
        dimension_count = img_shape[-1]

        if method == "fraction":
            assert 0 < fraction <= 1, "Fraction must be between 0 and 1."
            count = int(np.floor(fraction * pixel_count))

        subsample_pixels = data.reshape((-1, dimension_count))[
            np.random.choice(
                pixel_count,
                size=count,
            ),
            :,
        ]

        return subsample_pixels
