import pickle
from typing import Literal

import numpy as np
from fastersom import SOM
from numpy.typing import NDArray

from spatiomic._internal._anndata_method import anndata_method
from spatiomic._internal._data_method import data_method
from spatiomic.dimension._base import LoadableDimensionReducer


class Som(LoadableDimensionReducer):
    """Reduce sample dimensionality with self-organizing maps."""

    def __init__(
        self,
        node_count: tuple[int, int] = (50, 50),
        dimension_count: int = 5,
        distance_metric: Literal["euclidean", "manhattan", "cosine", "correlation"] = "euclidean",
        sigma_initial: float | None = None,
        sigma_final: float | None = None,
        initialization: Literal["random", "sample"] = "random",
        batch_size: int | None = None,
        seed: int | None = None,
        use_gpu: bool = True,
    ) -> None:
        """Initialise a self-organising map with the provided configuration.

        The map is a batch self-organizing map (Kohonen's batch map) on a rectangular grid with a gaussian neighborhood
        whose width decays exponentially, implemented by `fastersom` in PyTorch.

        The advantage of self-organizing maps are, that they reduce the dimensionality of the data while preserving the
        feature dimensions. This potentially allows for a better interpretation of the data. The disadvantage is, that
        its output are `representative` nodes and not the actual data points. There are three things to keep in mind so
        that the SOM best represents the biology of your data:
        - The SOM node count should be large enough to capture the topography of the data. If you data is very uniform,
            you can use a smaller node count. However, when working with tissue, very different imaging markers and
            multiple disease states, a larger node count is recommended.
        - The SOM should be trained for long enough to capture the topography of the data.
        - The SOM should be trained with a final neighborhood size that is not too large. SOM nodes are not individually
            updated during training, but rather in a neighborhood. If your neighborhood is too large, other, perhaps
            more abundant biological signals will pull nodes that represent less abundant signals towards them, leading
            to a worse representation of the latter.

        Args:
            node_count (tuple[int, int], optional): Tuple determining the SOM node size. Defaults to (50, 50).
            dimension_count (int, optional): Dimensionality of the original data. Defaults to 5.
            distance_metric (Literal["euclidean", "manhattan", "cosine", "correlation"], optional): Distance metric to
                use. Samples with zero norm (cosine) or constant channels (correlation) have no nearest node.
                Defaults to "euclidean".
            sigma_initial (float | None, optional): The initial width of the gaussian neighborhood in grid units.
                Defaults to None, which uses ceil(min(node_count) / 2) / 2.
            sigma_final (float | None, optional): The width the neighborhood decays towards at the end of training.
                Defaults to None, which uses 0.1 * sigma_initial.
            initialization (Literal["random", "sample"], optional): Whether to initialise the nodes as random unit
                vectors or as randomly chosen samples. Defaults to "random".
            batch_size (int | None, optional): Data points to process concurrently. Only affects memory usage and
                speed. Defaults to None, which uses 65536 on the GPU and 8192 on the CPU.
            seed (int | None, optional): The random seed. Defaults to None.
            use_gpu (bool, optional): Whether to use a GPU (CUDA or MPS) if one is available. Defaults to True.
        """
        self.set_config(
            node_count=node_count,
            dimension_count=dimension_count,
            distance_metric=distance_metric,
            sigma_initial=sigma_initial,
            sigma_final=sigma_final,
            initialization=initialization,
            batch_size=batch_size,
            seed=seed,
        )

        self.use_gpu = use_gpu
        self.som: SOM | None = None

    def set_config(
        self,
        node_count: tuple[int, int] = (50, 50),
        dimension_count: int = 5,
        distance_metric: Literal["euclidean", "manhattan", "cosine", "correlation"] = "euclidean",
        sigma_initial: float | None = None,
        sigma_final: float | None = None,
        initialization: Literal["random", "sample"] = "random",
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> None:
        """Set the config of the SOM class.

        Args:
            node_count (tuple[int, int], optional): Tuple determining the SOM node size. Defaults to (50, 50).
            dimension_count (int, optional): Dimensionality of the original data. Defaults to 5.
            distance_metric (Literal["euclidean", "manhattan", "cosine", "correlation"], optional): Distance metric to
                use. Defaults to "euclidean".
            sigma_initial (float | None, optional): The initial width of the gaussian neighborhood in grid units.
                Defaults to None.
            sigma_final (float | None, optional): The width the neighborhood decays towards at the end of training.
                Defaults to None.
            initialization (Literal["random", "sample"], optional): Whether to initialise the nodes as random unit
                vectors or as randomly chosen samples. Defaults to "random".
            batch_size (int | None, optional): Data points to process concurrently. Defaults to None.
            seed (int | None, optional): The random seed. Defaults to None.
        """
        self.node_count = node_count
        self.dimension_count = dimension_count
        self.distance_metric = distance_metric
        self.sigma_initial = sigma_initial
        self.sigma_final = sigma_final
        self.initialization = initialization
        self.batch_size = batch_size
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
            "sigma_initial": self.sigma_initial,
            "sigma_final": self.sigma_final,
            "initialization": self.initialization,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }

    def _get_som_parameters(
        self,
    ) -> dict:
        """Get the fastersom SOM parameters of the configuration, except the grid size, epochs and initialization.

        Returns:
            dict: The keyword arguments for the fastersom SOM.
        """
        return {
            "sigma_initial": self.sigma_initial,
            "sigma_final": self.sigma_final,
            "metric": self.distance_metric,
            "random_state": self.seed,
            "device": None if self.use_gpu else "cpu",
            "batch_size": self.batch_size,
        }

    def _get_fitted_som(
        self,
    ) -> SOM:
        """Get the fitted fastersom SOM.

        Raises:
            ValueError: If no self-organizing map has been created or loaded.

        Returns:
            SOM: The fitted SOM.
        """
        if self.som is None:
            raise ValueError("No SOM map has been created or loaded yet.")

        return self.som

    @anndata_method(input_attribute="X")
    @data_method
    def fit(
        self,
        data: NDArray,
        iteration_count: int = 50,
    ) -> None:
        """Fit the SOM on the data.

        Args:
            data (NDArray): The data (channel-last) of which the SOM should capture the topography.
            iteration_count (int, optional): The epochs to train the SOM for. Each epoch makes one update of all nodes
                from all data points. Defaults to 50.

        Raises:
            ValueError: If the data does not have `dimension_count` channels.
        """
        if data.shape[-1] != self.dimension_count:
            raise ValueError(f"data must have {self.dimension_count} channels, got {data.shape[-1]}")

        self.som = SOM(
            *self.node_count,
            n_epochs=iteration_count,
            init=self.initialization,
            **self._get_som_parameters(),
        ).fit(data)

    @anndata_method(input_attribute="X", output_uns="som_neighbors")
    @data_method
    def predict(
        self,
        data: NDArray,
        return_distance: bool = False,
    ) -> NDArray | tuple[NDArray, NDArray]:
        """Get the id of the nearest SOM node for each data point and optionally the distance to the node.

        Data points with zero norm (cosine) or constant channels (correlation) have no nearest node: their id is -1 and
        their distance is NaN.

        Args:
            data (NDArray): The channel-last data to find the nearest SOM node for.
            return_distance (bool, optional): Whether to return the distance to the nearest node. Defaults to False.

        Returns:
            NDArray | tuple[NDArray, NDArray]: The node id for each data point or the distances and node ids.
        """
        if not return_distance:
            return self._get_fitted_som().predict(data)

        node_ids, distances = self._get_fitted_som().predict(data, return_distance=True)

        return distances, node_ids

    @anndata_method(input_attribute="X", output_obs="som_neighbors")
    @data_method
    def fit_predict(
        self,
        data: NDArray,
        iteration_count: int = 50,
        return_distance: bool = False,
    ) -> NDArray | tuple[NDArray, NDArray]:
        """Fit the SOM on the data and return the id of the nearest SOM node for each data point.

        Args:
            data (NDArray): The data to fit the SOM on and to find the nearest SOM node for.
            iteration_count (int, optional): The epochs to train the SOM for. Defaults to 50.
            return_distance (bool, optional): Whether to return the distance to the nearest node. Defaults to False.

        Returns:
            NDArray | tuple[NDArray, NDArray]: The node id for each data point or the distances and node ids.
        """
        self.fit(data, iteration_count=iteration_count)

        return self.predict(data, return_distance=return_distance)  # type: ignore

    @anndata_method(input_attribute="X", output_obs="som_labels")
    @data_method
    def label(
        self,
        data: NDArray,
        clusters: list[int] | NDArray,
        save_path: str | None = None,
        return_distance: bool = False,
        flatten: bool = False,
    ) -> NDArray | tuple[NDArray, NDArray]:
        """Get the label for each data point based on the label for its closest SOM node.

        This function internally uses the predict method to get the nearest node for each data point and then assigns
        the label of the nearest node to the data point. Labels have to be provided for each SOM node based on a
        clustering of the SOM nodes. Data points without a nearest node (see `predict`) are labelled -1.

        Args:
            data (NDArray): The data to be labelled based on the label of the nearest SOM node.
            clusters (list[int] | NDArray): A list of clusters (one for each SOM node).
            save_path (str | None, optional): The `.npy` path where to save the labelled data to. Defaults to None.
            return_distance (bool, optional): Whether to return the distance to the nearest node. Defaults to False.
            flatten (bool, optional): Whether to flatten the input data in every but the channel dimension.
                Defaults to False.

        Raises:
            ValueError: If the number of clusters does not match the node count or the save path is not `.npy`.

        Returns:
            NDArray | tuple[NDArray, NDArray]: The cluster for each data point or the clusters and distances.
        """
        clusters = np.asarray(clusters, dtype=np.int32).ravel()

        if len(clusters) != np.prod(self.node_count):
            raise ValueError("`clusters` must have the same number of elements as the self-organizing map node count.")

        if save_path and not save_path.endswith(".npy"):
            raise ValueError("The save path must end with `.npy`. Please add it to the path.")

        if return_distance:
            distance, node_ids = self.predict(data, return_distance=True)
        else:
            node_ids = self.predict(data)

        data_labelled: NDArray = np.where(node_ids >= 0, clusters[node_ids], -1)
        data_labelled = data_labelled if flatten else data_labelled.reshape(data.shape[:-1])

        if save_path:
            np.save(save_path, data_labelled)

        return (data_labelled, distance) if return_distance else data_labelled

    def load(
        self,
        save_path: str,
    ) -> None:
        """Load a previously pickled SOM.

        Maps saved by spatiomic versions based on `xpysom` can be loaded if `xpysom` is installed.

        Args:
            save_path (str): The path where to load the SOM and its configuration from.

        Raises:
            ModuleNotFoundError: If the map was saved with `xpysom` and it is not installed.
        """
        with open(save_path, "rb") as infile:
            try:
                config, nodes = pickle.load(infile)  # nosec
            except ModuleNotFoundError as error:
                raise ModuleNotFoundError(
                    f"{error}. This map was saved by an xpysom-based version of spatiomic, "
                    "install `xpysom==1.0.7` to load it and save it again."
                ) from error

        if not isinstance(nodes, np.ndarray):
            nodes = nodes.get_weights()
            config = {key: value for key, value in config.items() if key in self.get_config()}
            config["sigma_initial"] = config["sigma_final"] = None

        self.set_config(**config)
        self.set_nodes(nodes)

    def save(
        self,
        save_path: str,
    ) -> None:
        """Pickle and save a previously fit SOM.

        Args:
            save_path (str): The path where to save the SOM and its configuration to.
        """
        state = [self.get_config(), self.get_nodes()]

        with open(save_path, "wb") as outfile:
            pickle.dump(state, outfile)

    def get_nodes(
        self,
        flatten: bool = True,
    ) -> NDArray:
        """Get the weights of a previously fitted SOM.

        Args:
            flatten (bool, optional): Whether to flatten the SOM dimensions (but not the channel dimensionality).
                Defaults to True.

        Returns:
            NDArray: The weights of the SOM nodes.
        """
        nodes: NDArray = self._get_fitted_som().weights_

        return nodes if flatten else nodes.reshape((*self.node_count, -1))

    def set_nodes(
        self,
        nodes: NDArray,
    ) -> None:
        """Set the weights of the SOM nodes, e.g., to restore a previously fit SOM.

        Args:
            nodes (NDArray): The weights of the SOM nodes, shape (node count, channel count) or (*node_count, channel
                count).

        Raises:
            ValueError: If the nodes do not match the dimension count or node count of the configuration.
        """
        if nodes.shape[-1] != self.dimension_count:
            raise ValueError(f"nodes must have {self.dimension_count} channels, got {nodes.shape[-1]}")

        self.som = SOM.from_weights(nodes, *self.node_count, **self._get_som_parameters())

    def get_distance_map(
        self,
    ) -> NDArray:
        """Get the summed euclidean distance of each SOM node to its (up to eight) neighbors on the grid.

        Returns:
            NDArray: The distances normalized to a maximum of 1, shape `node_count`.
        """
        return self._get_fitted_som().distance_map()

    def get_quantization_error(
        self,
        data: NDArray,
        return_distances: bool = False,
    ) -> float | tuple[float, NDArray]:
        """Get the quantization error of the SOM on the provided data.

        The quantization error is the mean distance between each data point and its nearest SOM node. Data
        points without a nearest node (see `predict`) are left out of the mean and their distance is NaN.

        Args:
            data (NDArray): The data to get the quantization error for.
            return_distances (bool, optional): Whether to also return the distance of each data point to its nearest
                node. Defaults to False.

        Returns:
            float | tuple[float, NDArray]: The mean quantization error and all distances if `return_distances` is set to
                True.
        """
        if not return_distances:
            return self._get_fitted_som().quantization_error(data)

        distances, _ = self.predict(data, return_distance=True)

        return float(np.nanmean(distances)), distances
