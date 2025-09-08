from typing import Literal, Optional
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA as PCASk
from sklearn.decomposition import IncrementalPCA as IncrementalPCASk

from spatiomic._internal._anndata_method import anndata_method
from spatiomic._internal._data_method import data_method
from spatiomic.dimension._base import DimensionReducer


class Pca(DimensionReducer):
    """Class to perform principal-component analysis."""

    def __init__(
        self,
        dimension_count: int = 2,
        batch_size: Optional[int] = None,
        flavor: Literal["auto", "full", "incremental", "nmf"] = "auto",
        use_gpu: bool = True,
        **kwargs: dict,
    ) -> None:
        """Initialise the PCA class, set the dimension count and the estimator.

        Args:
            dimension_count (int, optional): The number of principal components that the data is to be reduced to.
                Defaults to 2.
            batch_size (Optional[int], optional): The batch size for the IncrementalPCA algorithm for a smaller memory
                profile. Defaults to None.
            flavor (Literal["auto", "full", "incremental", "nmf"], optional): The flavor of PCA to be used.
                Defaults to "auto".
            use_gpu (bool, optional): Whether to use the cuml implementation on the GPU. Defaults to True.
        """
        warn(
            "Principal components explain varying degrees of variance and only contain linear combinations of the"
            " original features. Keep this in mind before using them.",
            UserWarning,
            stacklevel=2,
        )
        self.dimension_count = dimension_count
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.set_estimator(
            flavor=flavor,
            **kwargs,
        )

    def set_estimator(
        self,
        flavor: Literal["auto", "full", "incremental", "nmf"] = "auto",
        **kwargs: dict,
    ) -> None:
        """Set the IncrementalPCA estimator.

        Args:
            flavor (Literal["auto", "full", "incremental", "nmf"], optional): The flavor of PCA to be used. When set to
                "auto" or "full" the full PCA implementation is used. When set to "incremental", the IncrementalPCA
                implementation is used. The "auto" flavor may try to determine the best implementation based on free
                memory in the future.
                Defaults to "auto".

        Raises:
            ValueError: If the flavor is not supported.
        """
        if self.use_gpu:
            try:
                if flavor == "auto" or flavor == "full":
                    from cuml import pca  # type: ignore
                elif flavor == "incremental":
                    from cuml import IncrementalPCA as pca  # type: ignore
                else:
                    raise ValueError(f"Flavor {flavor} is not supported for GPU-based PCA.")
            except Exception:
                warn(
                    "Could not import cuml implementation. Switching to non-GPU implementation.",
                    UserWarning,
                    stacklevel=2,
                )
                self.use_gpu = False

        if not self.use_gpu:
            if flavor == "auto" or flavor == "full":
                pca = PCASk
            elif flavor == "incremental":
                pca = IncrementalPCASk
            elif flavor == "nmf":
                from sklearn.decomposition import NMF

                pca = NMF
            else:
                raise ValueError(f"Flavor {flavor} is not supported.")

        if self.batch_size is None or flavor != "incremental":
            self.estimator = pca(
                n_components=self.dimension_count,
                **kwargs,
            )
        else:
            self.estimator = pca(
                n_components=self.dimension_count,
                batch_size=self.batch_size,
                **kwargs,
            )

    @anndata_method(input_attribute="X")
    @data_method
    def fit(
        self,
        data: NDArray,
    ) -> None:
        """Fit the PCA estimator of the class to the data.

        Args:
            data (NDArray): The data (channel-last) to be reduced in dimensionality.
        """
        # transform flattened data
        data = self.estimator.fit(np.reshape(data, (-1, data.shape[-1])))

    @anndata_method(input_attribute="X", output_obsm="X_pca")
    @data_method
    def transform(
        self,
        data: NDArray,
        flatten: bool = True,
    ) -> NDArray:
        """Perform principal-component analysis on the data with the settings of the class.

        Args:
            data (NDArray):  The data (channel-last) to be reduced in dimensionality.
            flatten (bool, optional): Whether to flatten the data in every but the channel dimension.
                Defaults to True.

        Returns:
            NDArray: The principal components of the data.
        """
        # ensure that the estimator is fitted
        assert self.estimator.components_ is not None, (
            "The PC estimator has not been fitted yet. Please call `fit` first."
        )

        # flatten data in every dimension but the last
        data_shape = data.shape
        data = np.reshape(data, (-1, data.shape[-1]))

        # transform flattened data
        data = self.estimator.transform(data)

        if not flatten:
            # expand data again
            data = np.reshape(data, (*data_shape[:-1], self.dimension_count))

        return data

    @anndata_method(input_attribute="X", output_obsm="X_pca")
    @data_method
    def fit_transform(
        self,
        data: NDArray,
        flatten: bool = True,
    ) -> NDArray:
        """Perform principal-component analysis on the data with the settings of the class.

        Args:
            data (NDArray):  The data (channel-last) to be reduced in dimensionality.
            flatten (bool, optional): Whether to flatten the data in every but the channel dimension.
                Defaults to True.

        Returns:
            NDArray: The principal components of the data.
        """
        # flatten data in every dimension but the last
        data_shape = data.shape
        data = np.reshape(data, (-1, data.shape[-1]))

        # transform flattened data
        data = self.estimator.fit_transform(data)

        if not flatten:
            # expand data again
            data = np.reshape(data, (*data_shape[:-1], self.dimension_count))

        return data

    def get_explained_variance_ratio(self) -> NDArray:
        """Get the explained variance ratio of the principal components.

        Returns:
            NDArray: The explained variance ratio of the principal components.
        """
        return np.array(self.estimator.explained_variance_ratio_)

    def get_loadings(self) -> NDArray:
        """Get the loadings of the principal components.

        Returns:
            NDArray: The loadings of the principal components.
        """
        return np.array(self.estimator.components_)
