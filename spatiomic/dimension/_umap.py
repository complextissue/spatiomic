from typing import Literal, Optional
from warnings import warn

import numpy as np
from numpy.typing import NDArray

from spatiomic._internal._anndata_method import anndata_method
from spatiomic._internal._data_method import data_method
from spatiomic.dimension._base import DimensionReducer


class Umap(DimensionReducer):
    """Expose UMAP dimensionality reduction."""

    def __init__(
        self,
        dimension_count: int = 2,
        distance_min: float = 0.2,
        distance_metric: Literal["euclidean", "manhattan", "correlation", "cosine"] = "euclidean",
        spread: float = 1.0,
        neighbor_count: int = 100,
        seed: Optional[int] = None,
        use_gpu: bool = True,
        **kwargs: dict,
    ) -> None:
        """Initialise a UMAP estimator with the provided configuration.

        Keyword arguments are passed to the UMAP estimator, so that it is possible to use `precomputed_knn` for example.

        Args:
            dimension_count (int, optional): The desired (reduced) dimensionality. Defaults to 2.
            distance_min (float, optional): A key paramter of the UMAP function. Defaults to 0.2.
            distance_metric (Literal["euclidean", "manhattan", "correlation", "cosine"], optional): The distance metric
                to be used for nearest neighbor calculation. Defaults to "euclidean".
            spread (float, optional): A key paramter of the UMAP function. Defaults to 1.0.
            neighbor_count (int, optional): A key paramter of the UMAP function. Defaults to 100.
            seed (Optional[int], optional): Random seed. Defaults to None.
            use_gpu (bool, optional): Whether to use the cuml implementation on the GPU. Defaults to True.
        """
        warn(
            "UMAP embeddings do not represent global distance and unreliably represent local distance. Use with care.",
            UserWarning,
            stacklevel=2,
        )
        self.dimension_count = dimension_count
        self.neighbor_count = neighbor_count
        self.distance_metric = distance_metric
        self.distance_min = distance_min
        self.spread = spread
        self.seed = seed

        self.use_gpu = use_gpu
        self.set_estimator(**kwargs)

    def set_estimator(
        self,
        **kwargs: dict,
    ) -> None:
        """Set the UMAP estimator."""
        # Try importing the RAPIDS UMAP package but default to umap-learn in case of exception
        cuml = False
        umap_cuml = None

        # Try importing the RAPIDS UMAP package but default to umap-learn in case of exception
        if self.use_gpu:
            try:
                from cuml import UMAP as umap_cuml  # type: ignore

                cuml = True
            except Exception:
                cuml = False

        if not cuml or umap_cuml is None:
            from umap import UMAP as umap_cpu

            self.estimator = umap_cpu(
                n_neighbors=self.neighbor_count,
                n_components=self.dimension_count,
                metric=self.distance_metric,
                min_dist=self.distance_min,
                spread=self.spread,
                random_state=self.seed,
                **kwargs,
            )
        else:
            self.estimator = umap_cuml(
                n_neighbors=self.neighbor_count,
                n_components=self.dimension_count,
                min_dist=self.distance_min,
                metric=self.distance_metric,
                spread=self.spread,
                random_state=self.seed,
                **kwargs,
            )

    @anndata_method(input_attribute="X")
    @data_method
    def fit(
        self,
        data: NDArray,
    ) -> None:
        """Fit the UMAP estimator on the data.

        Args:
            data (NDArray): The data (channel-last) to fit the UMAP by.
        """
        self.estimator.fit(data)

    @anndata_method(input_attribute="X", output_obsm="X_umap")
    @data_method
    def transform(
        self,
        data: NDArray,
        flatten: bool = True,
    ) -> NDArray:
        """Transform the UMAP dimensions for the data with a previously fit estimator.

        Args:
            data (NDArray): The data (channel-last) to be reduced in dimensionality.
            flatten (bool, optional): Whether to flatten the data in every but the channel dimension.
                Defaults to True.

        Returns:
            NDArray: The UMAP representation of the data.
        """
        data_shape = data.shape
        data = data.reshape(-1, data_shape[-1])

        data = np.array(self.estimator.transform(data))

        if not flatten:
            data = data.reshape((*data_shape[:-1], self.dimension_count))

        return data

    @anndata_method(input_attribute="X", output_obsm="X_umap")
    @data_method
    def fit_transform(
        self,
        data: NDArray,
        flatten: bool = True,
    ) -> NDArray:
        """Fit a UMAP estimator and transform the UMAP dimensions for the data.

        Args:
            data (NDArray): The data (channel-last) to be reduced in dimensionality.
            flatten (bool, optional): Whether to flatten the data in every but the channel dimension.
                Defaults to True.

        Returns:
            NDArray: The UMAP representation of the data.
        """
        # always fit on the flat data
        self.fit(data.reshape(-1, data.shape[-1]))
        return np.array(self.transform(data, flatten=flatten))
