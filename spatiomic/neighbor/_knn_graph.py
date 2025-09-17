from typing import List, Literal, Optional, Union

import numpy as np
from igraph import Graph
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
    ) -> Graph:
        """Construct a k-nearest neighbor graph of the data.

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
            ValueError: Raised when the `distance_max` parameter is used with the `batch_balanced` method.
            NotImplementedError: Raised when the specified neighborhood identification method has not been implemented.

        Returns:
            Graph: The neighborhood graph.
        """
        if isinstance(data, Som):
            data = data.get_nodes()

        # assert that the data is channel-last
        assert len(data.shape) >= 2, "Data must have at least two dimensions and be channel-last."

        # flatten in every but the channel dimension
        data_shape = data.shape
        data = data.reshape(-1, data_shape[-1])

        edgelist: List = []

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
            # use the NearestNeighbors class to find the nearest neighbors and construct the graph
            neighbor_finder.fit(data)

            if distance_max is None:
                neighbor_idx = neighbor_finder.kneighbors(
                    data,
                    n_neighbors=neighbor_count + 1,  # since we include the point itself
                    return_distance=False,
                )
            else:
                if use_gpu:
                    raise ValueError("The `distance_max` parameter is not supported when using the GPU.")

                neighbor_idx, distances = neighbor_finder.radius_neighbors(
                    data,
                    radius=distance_max,
                    return_distance=True,
                )

                # filter the neighbor_idx array to only include the indices within the specified distance
                neighbor_idx = np.array([np.where(distances[i] <= distance_max)[0] for i in range(0, len(distances))])

            knn_graph = Graph(cls.get_edges(neighbor_idx))

        elif method == "batch_balanced" and batch is not None:
            if distance_max is not None:
                raise ValueError("The `distance_max` parameter is not supported for the `batch_balanced` method.")

            # separate the data into batches
            batch = np.array(batch) if isinstance(batch, list) else batch
            batch_ids = np.unique(batch)
            batch_data = [data[np.array(batch) == batch_id] for batch_id in batch_ids]

            # get the indices for each batch where data[batch == batch_id]
            batch_indices = [np.where(batch == batch_ids[i])[0] for i in range(0, len(batch_data))]

            # calculate the nearest neighbors for each batch with each other batch
            batch_edges = []

            for i in range(0, len(batch_data)):
                neighbor_finder.fit(batch_data[i])

                for j in range(0, len(batch_data)):
                    # will find k neighbors in all batches
                    nearest_neighbors = neighbor_finder.kneighbors(
                        batch_data[j],
                        n_neighbors=(
                            neighbor_count + 1 if i == j else neighbor_count  # since we include the point itself
                        ),
                        return_distance=False,
                    )

                    for k in range(0, nearest_neighbors.shape[0]):
                        for length_idx in range(0, nearest_neighbors.shape[1]):
                            neighbor = nearest_neighbors[k, length_idx]

                            # discard the connection to the same point
                            if i == j and neighbor == k:
                                continue

                            batch_edges.append([[j, k], [i, neighbor]])

            # combine the batch_edges into a singleedges
            edgelist.extend(
                [
                    tuple(
                        sorted(
                            [
                                batch_indices[edge[0][0]][edge[0][1]],
                                batch_indices[edge[1][0]][edge[1][1]],
                            ]
                        )
                    )
                    for edge in batch_edges
                ]
            )

            # deduplicate the edges and convert to a graph
            knn_graph = Graph(list(set(edgelist)))

        else:
            raise NotImplementedError(
                f"The method {method} is not implemented. Please use 'simple' or 'batch_balanced'."
            )

        return knn_graph

    @staticmethod
    def get_edges(neighbor_idx: NDArray) -> NDArray:
        """Create an edgelist from a neighbor index array.

        Args:
            neighbor_idx (NDArray): The neighbor index array.

        Returns:
            List: The edgelist.
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

        assert np.all(np.sum(mask, axis=1) == m - 1, axis=0), (
            "The neighbor_idx array must be a valid neighbor index array."
        )

        # flatten the mask and the neighbor_idx array, then use the mask to filter the neighbor_idx array
        mask = mask.flatten()
        neighbor_idx = neighbor_idx.flatten()
        neighbor_idx = neighbor_idx[mask]

        # convert neighborhood array into an array with [index, neighbor] pairs
        edgelist = np.column_stack([np.repeat(np.arange(n), m - 1), neighbor_idx])

        # only keep unique pairs
        edgelist = np.sort(edgelist, axis=1)

        # only keep unique pairs
        edgelist = np.unique(edgelist, axis=0)

        return edgelist  # type: ignore[no-any-return]
