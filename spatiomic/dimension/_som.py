import pickle
from math import ceil
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from xpysom import XPySom

from spatiomic._internal._anndata_method import anndata_method
from spatiomic._internal._data_method import data_method
from spatiomic._internal._get_neighbor_finder import get_neighbor_finder
from spatiomic._internal._import_package import import_package
from spatiomic.dimension._base import LoadableDimensionReducer


class Som(LoadableDimensionReducer):
    """Reduce sample dimensionality with self-organizing maps."""

    def __init__(
        self,
        node_count: Tuple[int, int] = (50, 50),
        dimension_count: int = 5,
        distance_metric: Literal["euclidean", "manhattan", "correlation", "cosine"] = "euclidean",
        neighborhood: str = "gaussian",
        neighborhood_accuracy: Literal["fast", "accurate"] = "fast",
        learning_rate_initial: float = 0.2,
        learning_rate_final: float = 3e-3,
        sigma_initial: Optional[int] = None,
        sigma_final: Optional[int] = None,
        parallel_count: int = 8096,
        n_jobs: int = -1,
        seed: Optional[int] = None,
        use_gpu: bool = True,
    ) -> None:
        """Initialise a self-organising map with the provided configuration.

        The advantage of self-organizing maps are, that they reduce the dimensionality of the data while preserving the
        feature dimensions. This potentially allows for a better interpretation of the data. The disadvantage is, that
        its output are `representative` nodes and not the actual data points. There are four things to keep in mind so
        that the SOM best represents the biology of your data:
        - The SOM node count should be large enough to capture the topography of the data. If you data is very uniform,
            you can use a smaller node count. However, when working with tissue, very different imaging markers and
            multiple disease states, a larger node count is recommended.
        - The SOM should be trained for long enough to capture the topography of the data.
        - The SOM should be trained with a final learning rate that is not too high, so that the SOM can accurately
            represent small differences in the data.
        - The SOM should be trained with a final neighborhood size that is not too large. SOM nodes are not individually
            updated during training, but rather in a neighborhood. If your neighborhood is too large, other, perhaps
            more abundant biological signals will pull nodes that represent less abundant signals towards them, leading
            to a worse representation of the latter.

        Args:
            node_count (Tuple[int, int], optional): Tuple determining the SOM node size. Defaults to (50, 50).
            dimension_count (int, optional): Dimensionality of the original data. Defaults to 5.
            distance_metric (Literal["euclidean", "manhattan", "correlation", "cosine"], optional): Distance metric to
                use. Defaults to "euclidean".
            neighborhood (str, optional): The type of neighborhood used to determine related notes.
                Defaults to "gaussian".
            neighborhood_accuracy (Literal["fast", "accurate"], optional): The accuracy to use for the neighborhood.
                Defaults to "fast".
            learning_rate_initial (float, optional): The initial learning rate. Defaults to 0.2.
            learning_rate_final (float, optional): The learning rate at the end of the SOM training. Defaults to 3e-3.
            sigma_initial (Optional[int], optional): The initial size of the neighborhoods, higher values.
                Defaults to None.
            sigma_final (Optional[int], optional): The final size of the neighborhood, lower values. Defaults to None.
            parallel_count (int, optional): Data points to process concurrently. Defaults to 8096.
            n_jobs (int, optional): Jobs to perform simoustaneously when using the sklearn NearestNeighbor class.
                The value -1 means unlimited jobs. Defaults to -1.
            seed (Optional[int], optional): The random seed. Defaults to None.
            use_gpu (bool, optional): Whether to use cupy. Defaults to True.
        """
        # set the class variables
        self.set_config(
            node_count=node_count,
            dimension_count=dimension_count,
            distance_metric=distance_metric,  # type: ignore
            neighborhood=neighborhood,
            neighborhood_accuracy=neighborhood_accuracy,
            learning_rate_initial=learning_rate_initial,
            learning_rate_final=learning_rate_final,
            sigma_initial=sigma_initial,
            sigma_final=sigma_final,
            parallel_count=parallel_count,
            n_jobs=n_jobs,
            seed=seed,
        )

        self.use_gpu = use_gpu
        self.set_estimators()

    def set_estimators(
        self,
    ) -> None:
        """Set the XPySOM and nearest neighbor finder estimators."""
        # Check whether we can use cupy to work on the GPU
        self.xp, self.cuml = (
            import_package("cupy", alternative=np, return_success=True) if self.use_gpu else (np, False)  # type: ignore
        )

        # Initialise XPySOM instance
        try:
            self.som = XPySom(
                self.node_count[0],
                self.node_count[1],
                self.dimension_count,
                activation_distance=self.distance_metric,
                neighborhood_function=self.neighborhood,
                learning_rate=self.learning_rate_initial,
                learning_rateN=self.learning_rate_final,
                sigma=self.sigma_initial,
                sigmaN=self.sigma_final,
                xp=self.xp,
                n_parallel=self.parallel_count,
                random_seed=self.seed,
            )
        except ValueError as excp:
            if self.distance_metric == "correlation":
                raise ValueError(
                    "Using XPySOM with Pearson correlation requires a custom implementation. "
                    "You can install it via `pip install git+https://github.com/complextissue/xpysom`."
                ) from excp
            else:
                raise excp

        # Create the nearest neighbor finder
        self.neighbor_estimator = get_neighbor_finder(
            neighbor_count=1,
            channel_count=self.dimension_count,
            distance_metric=self.distance_metric,
            accuracy=self.accuracy,
            job_count=self.n_jobs,
            use_gpu=self.use_gpu,
        )

    def set_config(
        self,
        node_count: Tuple[int, int] = (50, 50),
        dimension_count: int = 5,
        distance_metric: Literal["euclidean", "manhattan", "correlation", "cosine"] = "euclidean",
        neighborhood: str = "gaussian",
        neighborhood_accuracy: Literal["fast", "accurate"] = "fast",
        learning_rate_initial: float = 0.2,
        learning_rate_final: float = 3e-3,
        sigma_initial: Optional[int] = None,
        sigma_final: Optional[int] = None,
        parallel_count: int = 8096,
        n_jobs: int = -1,
        seed: Optional[int] = None,
    ) -> None:
        """Set the config of the SOM class.

        Args:
            node_count (Tuple[int, int], optional): Tuple determining the SOM node size. Defaults to (50, 50).
            dimension_count (int, optional): Dimensionality of the original data. Defaults to 5.
            distance_metric (Literal["euclidean", "manhattan", "correlation", "cosine"], optional): Distance metric to
                use. Defaults to "euclidean".
            neighborhood (str, optional): The type of neighborhood used to determine related notes.
                Defaults to "gaussian".
            neighborhood_accuracy (Literal["fast", "accurate"], optional): The accuracy to use for the neighborhood.
                Defaults to "fast".
            learning_rate_initial (float, optional): The initial learning rate. Defaults to 0.2.
            learning_rate_final (float, optional): The learning rate at the end of the SOM training. Defaults to 3e-3.
            sigma_initial (Optional[int], optional): The initial size of the neighborhoods, higher values.
                Defaults to None.
            sigma_final (Optional[int], optional): The final size of the neighborhood, lower values. Defaults to None.
            parallel_count (int, optional): Data points to process concurrently. Defaults to 8096.
            seed (Optional[int], optional): The random seed. Defaults to None.
        """
        if sigma_initial is None:
            sigma_initial = ceil(min(node_count[0], node_count[1]) / 2)

        if sigma_final is None:
            sigma_final = 0

        # set the configuration as class variable so they can be used and or updated when saving/loading a map
        self.node_count = node_count
        self.dimension_count = dimension_count
        self.distance_metric = distance_metric
        self.neighborhood = neighborhood
        self.accuracy = neighborhood_accuracy
        self.learning_rate_initial = learning_rate_initial
        self.learning_rate_final = learning_rate_final
        self.sigma_initial = sigma_initial
        self.sigma_final = sigma_final
        self.parallel_count = parallel_count
        self.n_jobs = n_jobs
        self.seed = seed

    def get_config(
        self,
    ) -> dict:
        """Get the config of the SOM class.

        Returns:
            dict: The class configuration as a dictionary.
        """
        return {
            "node_count": self.node_count,
            "dimension_count": self.dimension_count,
            "distance_metric": self.distance_metric,
            "neighborhood": self.neighborhood,
            "neighborhood_accuracy": self.accuracy,
            "learning_rate_initial": self.learning_rate_initial,
            "learning_rate_final": self.learning_rate_final,
            "sigma_initial": self.sigma_initial,
            "sigma_final": self.sigma_final,
            "parallel_count": self.parallel_count,
            "n_jobs": self.n_jobs,
            "seed": self.seed,
        }

    @anndata_method(input_attribute="X")
    @data_method
    def fit(
        self,
        data: NDArray,
        iteration_count: int = 50,
        pca_init: bool = False,
    ) -> None:
        """Fit the SOM on the data.

        Args:
            data (Union[pd.DataFrame, NDArray]): The data (channel-last) of which the SOM should capture the topography.
            iteration_count (int, optional): The iterations to train the SOM for. Defaults to 50.
            pca_init (bool, optional): Whether to initialise the SOM through PCA. Defaults to False.
        """
        if data.ndim >= 3:
            data = data.reshape((-1, data.shape[-1]))

        if pca_init:
            self.som.pca_weights_init(data)

        self.som.train(
            data,
            iteration_count,
        )

    @anndata_method(input_attribute="X", output_uns="som_neighbors")
    @data_method
    def predict(
        self,
        data: NDArray,
        return_distance: bool = False,
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """Get the id of the nearest SOM node for each data point and optionally the distance to the node.

        Args:
            data (NDArray): The data to be labelled based on the label of the nearest SOM node.
            return_distance (bool, optional): Whether to return the distance to the nearest node. Defaults to False.

        Returns:
            Union[NDArray, Tuple[NDArray, NDArray]]: The labels for each data point or the labels and distances.
        """
        assert data.shape[-1] == self.dimension_count, (
            "The data must have the same number of channels as the SOM nodes."
        )

        # label the data based on the nearest neighbors in the map
        nodes = self.get_nodes().reshape((-1, self.dimension_count))

        self.neighbor_estimator.fit(
            nodes,
        )

        result: Union[NDArray, Tuple[NDArray, NDArray]] = self.neighbor_estimator.kneighbors(
            data.reshape((-1, self.dimension_count)),
            return_distance=return_distance,
        )

        return result

    @anndata_method(input_attribute="X", output_obs="som_neighbors")
    @data_method
    def fit_predict(
        self,
        data: NDArray,
        iteration_count: int = 50,
        pca_init: bool = False,
        return_distance: bool = False,
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """Fit the SOM on the data and return the id of the nearest SOM node for each data point.

        Args:
            data (NDArray): The data to be labelled based on the label of the nearest SOM node.
            iteration_count (int, optional): The iterations to train the SOM for. Defaults to 50.
            pca_init (bool, optional): Whether to initialise the SOM through PCA. Defaults to False.
            return_distance (bool, optional): Whether to return the distance to the nearest node. Defaults to False.

        Returns:
            Union[NDArray, Tuple[NDArray, NDArray]]: The id of the nearest SOM node for each data point or the ids and
                distances.
        """
        self.fit(
            data,
            iteration_count=iteration_count,
            pca_init=pca_init,
        )

        return self.predict(data, return_distance=return_distance)  # type: ignore

    @anndata_method(input_attribute="X", output_obs="som_labels")
    @data_method
    def label(
        self,
        data: NDArray,
        clusters: Union[List[int], NDArray],
        save_path: Optional[str] = None,
        return_distance: bool = False,
        flatten: bool = False,
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """Get the label for each data point based on the label for its closest SOM node.

        This function internally uses the predict method to get the nearest node for each data point and then assigns
        the label of the nearest node to the data point. Labels have to be provided for each SOM node based on a
        clustering of the SOM nodes.

        Args:
            data (NDArray): The data to be labelled based on the label of the nearest SOM node.
            clusters (Union[List[int], NDArray]): A list of clusters (one for each SOM node).
            save_path (Optional[str], optional): The path where to save the SOM and its configuration to.
                Defaults to None.
            return_distance (bool, optional): Whether to return the distance to the nearest node. Defaults to False.
            flatten (bool, optional): Whether to flatten the input data in every but the channel dimension.
                Defaults to False.

        Returns:
            Union[NDArray, Tuple[NDArray, NDArray]]: The cluster for each data point or the clusters and distances.
        """
        # cast labels to numpy array
        clusters = np.array(clusters, dtype=np.uint16).ravel()

        assert len(clusters) == np.multiply.reduce(self.node_count), (
            "`clusters` must have the same number of elements as the self-organizing map node count."
        )

        data_shape = data.shape

        result = self.predict(data, return_distance=return_distance)

        distance = None
        neighbor_idx: NDArray
        if return_distance:
            distance, neighbor_idx = result
        else:
            neighbor_idx = np.array(result)

        data_labelled = np.array([clusters[neighbor] for neighbor in neighbor_idx.ravel()])
        data_labelled = data_labelled.reshape(data_shape[:-1]) if not flatten else data_labelled

        # save the labelled data if a path was provided
        if save_path:
            assert ".npy" in save_path, "The save path must end with `.npy`. Please add it to the path."
            np.save(save_path, data_labelled)

        if return_distance and isinstance(distance, np.ndarray):
            return data_labelled, distance
        else:
            return data_labelled

    def load(
        self,
        save_path: str,
    ) -> None:
        """Load a previously pickled SOM.

        Args:
            save_path (str): The path where to load the SOM and its configuration from.
        """
        # load and set the som and load the class config
        with open(save_path, "rb") as infile:
            config, self.som = pickle.load(infile)  # nosec

        # initialise an XPySOM object with the data and set the class variables
        self.set_config(**config)

    def save(
        self,
        save_path: str,
    ) -> None:
        """Pickle and save a previously fit SOM.

        Args:
            save_path (str): The path where to save the SOM and its configuration to.

        Raises:
            ValueError: If no self-organizing map has been created or loaded.
        """
        if not self.som:
            raise ValueError("No SOM map has been created or loaded yet.")

        with open(save_path, "wb") as outfile:
            pickle.dump([self.get_config(), self.som], outfile)

    def get_nodes(
        self,
        flatten: bool = True,
    ) -> NDArray:
        """Get the weights of a previously fitted SOM.

        Args:
            flatten (bool, optional): Whether to flatten the SOM dimensions (but not the channel dimensionality).
                Defaults to True.

        Raises:
            ValueError: If no self-organizing map has been created or loaded.

        Returns:
            NDArray: The weights of the SOM nodes.
        """
        if not self.som:
            raise ValueError("No SOM map has been created or loaded yet.")

        nodes: NDArray = self.som.get_weights()

        if flatten:
            return nodes.reshape((-1, nodes.shape[-1]))
        else:
            return nodes

    def set_nodes(
        self,
        nodes: NDArray,
    ) -> None:
        """Set the weights of a previously fit SOM.

        Args:
            nodes (NDArray): The weights of the SOM nodes.
        """
        self.som._weights = nodes.reshape(self.som.get_weights().shape)

    def get_quantization_error(
        self,
        data: NDArray,
        return_distances: bool = False,
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """Get the quantization error of the SOM on the provided data.

        Uses the neighbor finder to find the nearest neighbor of each data point in the SOM and calculates the
        quantization error based on the distance between the data point and its nearest neighbor.

        Args:
            data (NDArray): The data to get the quantization error for.

        Returns:
            Union[float, Tuple[float, np.ndarray]]: The mean quantization error and all distances if `return_distances``
                is set to True.
        """
        # label the data based on the nearest neighbors in the map
        self.neighbor_estimator.fit(
            self.get_nodes().reshape((-1, self.dimension_count)),
        )

        distances, _ = self.neighbor_estimator.kneighbors(
            data.reshape((-1, self.dimension_count)),
            return_distance=True,
        )

        return float(np.mean(distances)) if return_distances is False else (float(np.mean(distances)), distances)
