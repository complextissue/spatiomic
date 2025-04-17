from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans as SkKMeans

from spatiomic._internal._anndata_method import anndata_method
from spatiomic._internal._data_method import data_method
from spatiomic.cluster._base import ClusterInterface


class KMeans(ClusterInterface):
    """A class that exposes k-means clustering."""

    def __init__(
        self,
        cluster_count: int = 20,
        run_count: int = 1,
        iteration_count: int = 300,
        tolerance: float = 1e-3,
        init: Literal["random", "k-means++"] = "k-means++",
        seed: Optional[int] = None,
        use_gpu: bool = True,
    ) -> None:
        """Set the configuration for the k-means clustering class and initialise it.

        Args:
            cluster_count (int, optional): The number of k-means clusters to split the data into. Defaults to 20.
            run_count (int, optional): The number of k-means runs to perform. Defaults to 1.
            iteration_count (int, optional): Maximum number of k-means iterations per run. Defaults to 300.
            tolerance (float, optional): Maximum tolerance in center shift to declare convergence. Defaults to 1e-3.
            init (Literal["random", "k-means++"], optional): How to initialise the centers, either "random"
                or "k-means++". Defaults to "k-means++".
            seed (Optional[int], optional): Random seed, either to be temporarily set as the numpy random seed
                or directly for libKmCuda. Defaults to None.
            use_gpu (bool, optional): Whether to use the cuml or the sklearn kMeans class. Defaults to True.
        """
        self.cluster_count = cluster_count
        self.run_count = run_count
        self.iteration_count = iteration_count
        self.tolerance = tolerance
        self.init = init
        self.seed = seed

        self.use_gpu = use_gpu

        self.set_estimator()

    def set_estimator(
        self,
    ) -> None:
        """Set the KMeans estimator."""
        if not TYPE_CHECKING and self.use_gpu:
            try:
                from cuml import KMeans as k_means  # type: ignore
            except Exception:
                k_means = SkKMeans
        else:
            k_means = SkKMeans

        self.estimator = k_means(
            n_clusters=self.cluster_count,
            n_init=self.run_count,
            init=self.init,
            max_iter=self.iteration_count,
            tol=self.tolerance,
            random_state=self.seed,
        )

    @anndata_method(input_attribute="X", output_obs="kmeans")
    @data_method
    def fit_predict(
        self,
        data: NDArray,
    ) -> NDArray:
        """Perform k-means clustering on the data with the settings of the class.

        Args:
            data (NDArray): The data to be clustered, last dimension being features.

        Returns:
            NDArray: An array containing the cluster for each of the data points.
        """
        # flatten in every but the channel dimension
        data_shape = data.shape
        data = data.reshape(-1, data_shape[-1])
        clusters = self.estimator.fit_predict(data)

        return np.array(clusters)
