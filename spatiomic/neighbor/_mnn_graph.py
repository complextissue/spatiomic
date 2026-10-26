from typing import Literal

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from spatiomic._internal._get_neighbor_finder import get_neighbor_finder
from spatiomic.dimension._som import Som
from spatiomic.neighbor._knn_graph import KnnGraph, _deduplicate_edges, _exclude_self


class MnnGraph:
    """A class that exposes a static method for mutual k-nearest neighbor graph construction."""

    @classmethod
    def create(
        cls,
        data: NDArray | Som,
        neighbor_count: int = 20,
        connect_isolated_nodes: bool = True,
        distance_metric: Literal["euclidean", "manhattan", "correlation", "cosine"] = "euclidean",
        accuracy: Literal["fast", "accurate"] = "accurate",
        job_count: int = -1,
        use_gpu: bool = True,
    ) -> nx.Graph:
        """Construct a mutual k-nearest neighbor graph of the data.

        Two data points are connected by an edge only if each is among the `neighbor_count` nearest neighbors of the
        other, excluding the data point itself. In contrast to a k-nearest neighbor graph, small groups of data points
        far from the rest of the data are not connected to it, as the data points they would be connected to have
        closer neighbors. The graph is unweighted and has no self-loops.

        Args:
            data (NDArray | Som): A channel-last array of the data points or a fitted SOM whose nodes are used.
            neighbor_count (int, optional): The number of nearest neighbors of each data point. Defaults to 20.
            connect_isolated_nodes (bool, optional): Whether to keep the edge to the nearest other data point for data
                points that would have no mutual neighbors. Defaults to True.
            distance_metric (Literal["euclidean", "manhattan", "correlation", "cosine"], optional): The distance metric
                to be used for nearest neighbor calculation. Defaults to "euclidean".
            accuracy (Literal["fast", "accurate"], optional): The accuracy of the nearest neighbor calculation.
                Defaults to "accurate".
            job_count (int, optional): Parallelization core count for the nearest neighbor calculation on the CPU.
                Defaults to -1.
            use_gpu (bool, optional): Whether to use the GPU for nearest neighbor calculation if possible.
                Defaults to True.

        Returns:
            nx.Graph: The unweighted, undirected mutual k-nearest neighbor graph with one integer node per data point,
                in the order of the flattened data.
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
        neighbors = _exclude_self(
            neighbor_finder.kneighbors(data, n_neighbors=neighbor_count + 1, return_distance=False)
        ).astype(np.int64)

        sources = np.repeat(np.arange(node_count, dtype=np.int64), neighbors.shape[1])
        targets = neighbors.ravel()

        # a directed pair is mutual if its reverse pair is also present
        mutual = np.isin(targets * node_count + sources, sources * node_count + targets)
        sources, targets = sources[mutual], targets[mutual]

        if connect_isolated_nodes:
            isolated = np.flatnonzero(np.bincount(sources, minlength=node_count) == 0)
            sources = np.concatenate([sources, isolated])
            targets = np.concatenate([targets, neighbors[isolated, 0]])

        return KnnGraph.create_graph(_deduplicate_edges(sources, targets, node_count=node_count), node_count=node_count)
