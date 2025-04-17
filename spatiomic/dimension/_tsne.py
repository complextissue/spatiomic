from typing import Literal, Optional
from warnings import warn

from numpy.typing import NDArray
from sklearn.manifold import TSNE as TsneSk

from spatiomic._internal._anndata_method import anndata_method
from spatiomic._internal._data_method import data_method
from spatiomic.dimension._base import DimensionReducer


class Tsne(DimensionReducer):
    """Expose tSNE dimensionality reduction."""

    def __init__(
        self,
        dimension_count: int = 2,
        distance_metric: Literal["euclidean", "manhattan"] = "euclidean",
        iteration_count: int = 1000,
        iteration_count_without_progress: int = 300,
        learning_rate: float = 200.0,
        perplexity: float = 50.0,
        seed: Optional[int] = None,
        use_gpu: bool = True,
    ) -> None:
        """Initialise a tSNE estimator with the provided configuration.

        Args:
            dimension_count (int, optional): The dimensions to reduce the data to. Defaults to 2.
            distance_metric (Literal[, optional): tSNE distance metric. Defaults to "euclidean".
            iteration_count (int, optional): tSNE algorithm iteration count. Defaults to 1000.
            iteration_count_without_progress (int, optional): Iterations to continue without progress is made.
                Defaults to 300.
            learning_rate (float, optional): tSNE learning rate. Defaults to 200.0.
            perplexity (float, optional): Determines the spread of the tSNE data points. Defaults to 50.0.
            seed (Optional[int], optional): Random seed. Defaults to None.
            use_gpu (bool, optional): Whether to use the cuml implementation on the GPU. Defaults to True.
        """
        warn(
            "tSNE embeddings do not represent global distance and unreliably represent local distance. Use with care.",
            UserWarning,
            stacklevel=2,
        )
        self.dimension_count = dimension_count
        self.distance_metric = distance_metric
        self.iteration_count = iteration_count
        self.iteration_count_without_progress = iteration_count_without_progress
        self.learning_rate = learning_rate
        self.perplexity = perplexity
        self.seed = seed

        self.use_gpu = use_gpu
        self.set_estimator()

    def set_estimator(
        self,
    ) -> None:
        """Set the tSNE estimator."""
        if self.use_gpu:
            try:
                from cuml import TSNE  # type: ignore

                cuml = True
            except Exception:
                cuml = False
        else:
            cuml = False

        # check that we are not using RAPIDS with a distance metric that is not euclidean
        if cuml is True and self.distance_metric != "euclidean":
            warn(
                "cuML TSNE only supports euclidean distance, defaulting to sklearn.manifold.TSNE.",
                UserWarning,
                stacklevel=2,
            )
            cuml = False

        if not cuml:
            self.estimator = TsneSk(
                n_components=self.dimension_count,
                metric=self.distance_metric,
                max_iter=self.iteration_count,
                n_iter_without_progress=self.iteration_count_without_progress,
                learning_rate=self.learning_rate,
                perplexity=self.perplexity,
                random_state=self.seed,
            )
        else:
            self.estimator = TSNE(  # type: ignore
                n_components=self.dimension_count,
                metric=self.distance_metric,
                n_iter=self.iteration_count,
                n_iter_without_progress=self.iteration_count_without_progress,
                learning_rate=self.learning_rate,
                perplexity=self.perplexity,
                random_state=self.seed,
            )

    @anndata_method(input_attribute="X", output_layer="X_tsne")
    @data_method
    def fit_transform(
        self,
        data: NDArray,
        flatten: bool = True,
    ) -> NDArray:
        """Fit a tSNE estimator and transform the tSNE dimensions for the data.

        Args:
            data (NDArray): The data (channel-last) to be reduced in dimensionality.
            flatten (bool, optional): Whether to flatten the data in every but the channel dimension.
                Defaults to True.

        Returns:
            NDArray: The tSNE representation of the data.
        """
        data_shape = data.shape
        data = data.reshape(-1, data_shape[-1])
        data = self.estimator.fit_transform(data)

        if not flatten:
            data = data.reshape((*data_shape[:-1], self.dimension_count))

        return data
