from typing import TYPE_CHECKING, Literal, Optional

from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering as SkAgglomerativeClustering

from spatiomic._internal._anndata_method import anndata_method
from spatiomic._internal._data_method import data_method
from spatiomic._internal._seed_method import seed_method
from spatiomic.cluster._base import ClusterInterface


class Agglomerative(ClusterInterface):
    """A class that exposes hierarchical agglomerative clustering."""

    def __init__(
        self,
        cluster_count: int = 10,
        distance_metric: Literal["euclidean", "manhattan", "cosine"] = "euclidean",
        use_gpu: bool = True,
    ) -> None:
        """Set the configuration for the agglomerative clustering class and initialise it.

        Args:
            cluster_count (int, optional): The number of clusters to group the data into. Defaults to 10.
            distance_metric (Literal["euclidean", "manhattan", "cosine"], optional): The distance metric to use.
                Defaults to "euclidean".
            use_gpu (bool, optional): Whether to use the cuml or the sklearn AgglomerativeClustering class.
                Defaults to True.
        """
        self.cluster_count = cluster_count
        self.distance_metric = distance_metric

        self.connectivity: Optional[str]

        self.use_gpu = use_gpu
        self.set_estimator()

    def set_estimator(
        self,
    ) -> None:
        """Set the AgglomerativeClustering estimator."""
        if not TYPE_CHECKING and self.use_gpu:
            try:
                from cuml import AgglomerativeClustering as agglomerative_clustering  # type: ignore

                cuml = True
            except Exception:
                agglomerative_clustering = SkAgglomerativeClustering
                cuml = False
        else:
            agglomerative_clustering = SkAgglomerativeClustering
            cuml = False

        # cuml agglomerative clustering only works with Euclidean if the connectivity is knn (the default)
        if (cuml and self.distance_metric != "euclidean" and self.distance_metric != "l1") or cuml:
            self.connectivity = "pairwise"
        else:
            self.connectivity = None

        self.estimator = agglomerative_clustering(
            n_clusters=self.cluster_count,
            metric=self.distance_metric,
            connectivity=self.connectivity,
        )

    @anndata_method(input_attribute="X", output_obs="agglomerative")
    @data_method
    @seed_method
    def fit_predict(
        self,
        data: NDArray,
        **kwargs: dict,  # noqa: ARG002
    ) -> NDArray:
        """Perform agglomerative hierarchical clustering on the data with the settings of the class.

        Args:
            data (Union[pd.DataFrame, NDArray]): The data to be clustered, last dimension being features.

        Returns:
            NDArray: An array containing the cluster for each of the data points.

        Raises:
            ValueError: If the estimator does not have a fit_predict method.
        """
        # flatten in every but the channel dimension
        data_shape = data.shape
        data = data.reshape(-1, data_shape[-1])

        if hasattr(self.estimator, "fit_predict"):
            clusters: NDArray = self.estimator.fit_predict(data)
        else:
            raise ValueError("Estimator does not have a fit_predict method.")

        return clusters
