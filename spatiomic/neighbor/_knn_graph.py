from typing import Iterable, List, Literal, Optional, Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from spatiomic._internal._get_neighbor_finder import get_neighbor_finder
from spatiomic.dimension._som import Som


class KnnGraph:
    """A class that exposes a static method for k-nearest neighbor graph construction."""

    @classmethod
    def create(
        cls,
        data: Union[NDArray, Som],
        batch: Optional[Union[NDArray, List[int], List[str]]] = None,
        neighbor_count: int = 20,
        distance_metric: Literal["euclidean", "manhattan", "correlation", "cosine"] = "euclidean",
        method: Literal["simple", "batch_balanced"] = "simple",
        accuracy: Literal["fast", "accurate"] = "accurate",
        distance_max: Optional[float] = None,
        job_count: int = -1,
        use_gpu: bool = True,
    ) -> nx.Graph:
        """Construct a k-nearest neighbor graph of the data.

        The graph is undirected and has one integer node per data point, in the order of the flattened data.

        Args:
            data (Union[NDArray, Som]): A channel-last array of the data points to be used for graph construction.
            batch (Optional[Union[NDArray, List[int], List[str]]], optional): The batch labels for the data points.
                Defaults to None.
            neighbor_count (int, optional): The neighbor count for the neighborhood graph. Defaults to 20.
            distance_metric (Literal["euclidean", "manhattan", "correlation", "cosine"], optional): The distance metric
                to be used for nearest neighbor calculation. Defaults to "euclidean".
            method (Literal["simple", "batch_balanced"], optional): The method for nearest neighbor calculation.
                Defaults to "simple".
            accuracy (Literal["fast", "accurate"], optional): The accuracy of the nearest neighbor calculation.
                Defaults to "accurate".
            distance_max (Optional[float], optional): The maximum distance for nearest neighbor calculation. Currently
                only supported for the `simple` method. Defaults to None.
            job_count (int, optional): Parallelization core count when method is `simple`. Defaults to -1.
            use_gpu (bool, optional): Whether to use the GPU for nearest neighbor calculation if possible.
                Defaults to True.

        Raises:
            ValueError: Raised when the `distance_max` parameter is used with the `batch_balanced` method or the GPU.
            NotImplementedError: Raised when the specified neighborhood identification method has not been implemented.

        Returns:
            nx.Graph: The neighborhood graph.
        """
        if isinstance(data, Som):
            data = data.get_nodes()

        # assert that the data is channel-last
        assert len(data.shape) >= 2, "Data must have at least two dimensions and be channel-last."

        # flatten in every but the channel dimension
        data_shape = data.shape
        data = data.reshape(-1, data_shape[-1])

        # create the neighbor finder
        neighbor_finder = get_neighbor_finder(
            neighbor_count=neighbor_count,
            channel_count=data_shape[-1],
            distance_metric=distance_metric,
            accuracy=accuracy,
            job_count=job_count,
            use_gpu=use_gpu,
        )

        if method == "simple":
            if distance_max is not None and use_gpu:
                raise ValueError("The `distance_max` parameter is not supported when using the GPU.")

            neighbor_finder.fit(data)

            if distance_max is None:
                neighbor_idx = neighbor_finder.kneighbors(
                    data,
                    n_neighbors=neighbor_count + 1,  # since we include the point itself
                    return_distance=False,
                )
                edges = cls.get_edges(neighbor_idx)
            else:
                # radius_neighbors returns a ragged object array of the neighbors within the radius of each point
                neighbor_idx = neighbor_finder.radius_neighbors(data, radius=distance_max, return_distance=False)
                edges = _deduplicate_edges(
                    np.repeat(np.arange(len(neighbor_idx)), [len(neighbors) for neighbors in neighbor_idx]),
                    np.concatenate(neighbor_idx),
                    node_count=data.shape[0],
                )

            knn_graph = cls.create_graph(edges, node_count=data.shape[0])

        elif method == "batch_balanced" and batch is not None:
            if distance_max is not None:
                raise ValueError("The `distance_max` parameter is not supported for the `batch_balanced` method.")

            batch = np.asarray(batch)
            batch_indices = [np.flatnonzero(batch == batch_id) for batch_id in np.unique(batch)]

            # find the neighbors of the points of every batch within every batch, including their own
            sources: list[NDArray] = []
            targets: list[NDArray] = []

            for fit_indices in batch_indices:
                neighbor_finder.fit(data[fit_indices])

                for query_indices in batch_indices:
                    is_own_batch = query_indices is fit_indices
                    nearest_neighbors = neighbor_finder.kneighbors(
                        data[query_indices],
                        n_neighbors=neighbor_count + 1 if is_own_batch else neighbor_count,  # include the point itself
                        return_distance=False,
                    )

                    if is_own_batch:
                        edges = fit_indices[cls.get_edges(nearest_neighbors)]
                        sources.append(edges[:, 0])
                        targets.append(edges[:, 1])
                    else:
                        sources.append(np.repeat(query_indices, nearest_neighbors.shape[1]))
                        targets.append(fit_indices[nearest_neighbors].ravel())

            knn_graph = cls.create_graph(
                _deduplicate_edges(np.concatenate(sources), np.concatenate(targets), node_count=data.shape[0]),
                node_count=data.shape[0],
            )

        else:
            raise NotImplementedError(
                f"The method {method} is not implemented. Please use 'simple' or 'batch_balanced'."
            )

        return knn_graph

    @staticmethod
    def create_graph(
        edges: Iterable[Tuple[int, int]] | NDArray,
        node_count: int,
        weights: Optional[Iterable[float]] = None,
    ) -> nx.Graph:
        """Create an undirected graph with the nodes `0` to `node_count - 1` and the given edges.

        Args:
            edges (Iterable[Tuple[int, int]] | NDArray): The edges as pairs of node indices.
            node_count (int): The number of nodes, including nodes without edges.
            weights (Optional[Iterable[float]], optional): The weight of each edge, stored as the `weight` edge
                attribute. Defaults to None.

        Returns:
            nx.Graph: The graph.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(node_count))

        if weights is None and isinstance(edges, np.ndarray):
            graph.add_edges_from(edges.tolist())
        elif weights is None:
            graph.add_edges_from((int(source), int(target)) for source, target in edges)
        else:
            graph.add_weighted_edges_from(
                (int(source), int(target), float(weight))
                for (source, target), weight in zip(edges, weights, strict=True)
            )

        return graph

    @staticmethod
    def get_edges(neighbor_idx: NDArray) -> NDArray:
        """Create an edgelist from a neighbor index array.

        Args:
            neighbor_idx (NDArray): The neighbor index array.

        Raises:
            ValueError: Raised when a row of `neighbor_idx` contains its own index more than once.

        Returns:
            NDArray: The unique undirected edges, each sorted as (lower, higher) node index.
        """
        # stack the n indices m times to get a matrix of shape (n, m)
        n = neighbor_idx.shape[0]
        m = neighbor_idx.shape[1]
        indices = np.repeat(np.arange(n), m).reshape(n, m)

        # only keep values where the indices are not equal to the neighbor_idx
        mask = indices != neighbor_idx

        # set the last column to False for mask rows where the sum is equal to m in case the index is not part of the
        # neighborhood
        mask[np.sum(mask, axis=1) == m, -1] = False

        if np.any(np.sum(mask, axis=1) != m - 1):
            raise ValueError("The neighbor_idx array must be a valid neighbor index array.")

        return _deduplicate_edges(np.repeat(np.arange(n), m - 1), neighbor_idx[mask], node_count=n)


def _deduplicate_edges(sources: NDArray, targets: NDArray, node_count: int) -> NDArray:
    """Turn directed (source, target) pairs into unique undirected edges without self-loops.

    Args:
        sources (NDArray): The source node of each pair.
        targets (NDArray): The target node of each pair.
        node_count (int): The number of nodes, an exclusive upper bound of the node indices.

    Returns:
        NDArray: The unique edges as an array of shape (edge_count, 2), each sorted as (lower, higher) node index.
    """
    lower = np.minimum(sources, targets).astype(np.int64)
    higher = np.maximum(sources, targets).astype(np.int64)
    keep = lower != higher

    # encode each pair as a single integer, as np.unique on 1D data is much faster than on rows
    keys = np.unique(lower[keep] * node_count + higher[keep])

    return np.column_stack(np.divmod(keys, node_count))
