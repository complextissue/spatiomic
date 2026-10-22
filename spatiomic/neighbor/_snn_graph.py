from typing import Literal, Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from spatiomic._internal._get_neighbor_finder import get_neighbor_finder
from spatiomic.dimension._som import Som
from spatiomic.neighbor._knn_graph import KnnGraph


class SnnGraph:
    """A class that exposes a static method for shared nearest neighbor graph construction."""

    @classmethod
    def create(
        cls,
        data: Union[NDArray, Som],
        neighbor_count: int = 20,
        prune_threshold: float = 1 / 15,
        connect_isolated_nodes: bool = True,
        distance_metric: Literal["euclidean", "manhattan", "correlation", "cosine"] = "euclidean",
        accuracy: Literal["fast", "accurate"] = "accurate",
        job_count: int = -1,
        use_gpu: bool = True,
    ) -> nx.Graph:
        """Construct a Jaccard-weighted shared nearest neighbor graph of the data.

        Each data point's neighborhood consists of itself and its `neighbor_count` nearest neighbors. Two data points
        are connected by an edge whose `weight` is the Jaccard index of their neighborhoods, i.e., the number of shared
        neighbors divided by the size of the union of both neighborhoods, as in Seurat and PhenoGraph. Edges with a
        weight of at most `prune_threshold` are removed. The graph has no self-loops.

        Args:
            data (Union[NDArray, Som]): A channel-last array of the data points or a fitted SOM whose nodes are used.
            neighbor_count (int, optional): The number of nearest neighbors of each data point. Defaults to 20.
            prune_threshold (float, optional): Edges with a Jaccard index less than or equal to this value are removed.
                Defaults to 1 / 15.
            connect_isolated_nodes (bool, optional): Whether to keep the edge with the highest Jaccard index for data
                points that would have no edges after pruning. Defaults to True.
            distance_metric (Literal["euclidean", "manhattan", "correlation", "cosine"], optional): The distance metric
                to be used for nearest neighbor calculation. Defaults to "euclidean".
            accuracy (Literal["fast", "accurate"], optional): The accuracy of the nearest neighbor calculation.
                Defaults to "accurate".
            job_count (int, optional): Parallelization core count for the nearest neighbor calculation on the CPU.
                Defaults to -1.
            use_gpu (bool, optional): Whether to use the GPU for nearest neighbor calculation if possible.
                Defaults to True.

        Returns:
            nx.Graph: The weighted, undirected shared nearest neighbor graph with one integer node per data point, in
                the order of the flattened data.
        """
        if isinstance(data, Som):
            data = data.get_nodes()

        data = data.reshape(-1, data.shape[-1])
        node_count = data.shape[0]

        neighbor_finder = get_neighbor_finder(
            neighbor_count=neighbor_count,
            channel_count=data.shape[-1],
            distance_metric=distance_metric,
            accuracy=accuracy,
            job_count=job_count,
            use_gpu=use_gpu,
        )
        neighbor_finder.fit(data)
        neighbor_idx = neighbor_finder.kneighbors(data, n_neighbors=neighbor_count + 1, return_distance=False)

        # each neighborhood includes the data point itself, even if duplicates displace it from its neighbor list
        rows = np.concatenate([np.repeat(np.arange(node_count), neighbor_idx.shape[1]), np.arange(node_count)])
        columns = np.concatenate([np.asarray(neighbor_idx).ravel(), np.arange(node_count)])
        neighborhoods = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, columns)), shape=(node_count,) * 2)
        neighborhoods.data[:] = 1.0

        shared = (neighborhoods @ neighborhoods.T).tocoo()
        neighborhood_sizes = np.asarray(neighborhoods.sum(axis=1)).ravel()

        off_diagonal = shared.row != shared.col
        sources, targets = shared.row[off_diagonal], shared.col[off_diagonal]
        shared_counts = shared.data[off_diagonal]
        weights = shared_counts / (neighborhood_sizes[sources] + neighborhood_sizes[targets] - shared_counts)

        keep = weights > prune_threshold

        if connect_isolated_nodes:
            isolated = np.bincount(sources[keep], minlength=node_count) == 0
            strongest_first = np.lexsort((-weights, sources))
            _, first_index = np.unique(sources[strongest_first], return_index=True)
            strongest = strongest_first[first_index]
            keep[strongest[isolated[sources[strongest]]]] = True

        return KnnGraph.create_graph(
            zip(sources[keep], targets[keep], strict=True),
            node_count=node_count,
            weights=weights[keep],
        )
