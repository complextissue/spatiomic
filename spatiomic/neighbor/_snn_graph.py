from itertools import combinations
from random import shuffle
from typing import List, Literal, Optional, Set, Tuple, Union

from igraph import Graph
from numpy.typing import NDArray

from spatiomic._internal._get_neighbor_finder import get_neighbor_finder
from spatiomic.dimension._som import Som


class SnnGraph:
    """A class that exposes a static method for shared nearest neighbor graph construction."""

    @staticmethod
    def get_shared_neighbors(
        nodes: Union[List[int], Set[int]],
        neighbors: List[set],
        shared_count_threshold: int = 1,
    ) -> List[Tuple[int, int]]:
        """Get the shared neighbors between nodes.

        Args:
            nodes (List[int]): The nodes.
            neighbors (List[set]): The neighbors of each node.
            shared_count_threshold (int, optional): The minimum number of shared neighbors for two nodes to be linked.
                Defaults to 1.

        Returns:
            List[tuple]: A list of edges.
        """
        edges: List[Tuple[int, int]] = []

        for node_1, node_2 in list(combinations(nodes, r=2)):
            if len(neighbors[node_1] & neighbors[node_2]) > shared_count_threshold:
                edges.append((node_1, node_2))

        return edges

    @classmethod
    def create(
        cls,
        knn_graph: Graph,
        shared_count_threshold: int = 1,
        fix_lonely_nodes: bool = True,
        fix_lonely_nodes_method: Literal["random", "distance"] = "distance",
        distance_data: Optional[Union[NDArray, Som]] = None,
        distance_metric: Literal["euclidean", "manhattan", "correlation", "cosine"] = "euclidean",
        accuracy: Literal["fast", "accurate"] = "accurate",
        job_count: int = -1,
        use_gpu: bool = True,
    ) -> Graph:
        """Construct a shared nearest neighbor graph based on a k-nearest neighbor graph.

        The shared nearest neighbor graph is an undirected graph where two nodes are connected if they have at least
        `shared_count_threshold` neighbors in common.

        If `fix_lonely_nodes` is True, then the lonely nodes (nodes that don't have any neighbors in the k-nearest
        neighbor graph) are connected to their nearest neighbor. The method for connecting lonely nodes can be
        specified with `fix_lonely_nodes_method`. If `fix_lonely_nodes_method` is `random`, then the lonely nodes are
        connected to a random neighbor. If `fix_lonely_nodes_method` is `distance`, then the lonely nodes are connected
        to their nearest neighbor based on the distance between them.

        .. warning:: This method is not optimized for batch-balanced nearest neighbor calculation.

        Args:
            knn_graph (Graph): A k-nearest neighbor graph.
            shared_count_threshold (int, optional): The minimum number of shared neighbors for two nodes to be
                connected. Defaults to 1.
            fix_lonely_nodes (bool, optional): Whether to connect lonely nodes to the nearest neighbor.
                Defaults to True.
            fix_lonely_nodes_method (Literal["random", "distance"], optional): The method to use for connecting
                lonely nodes. Defaults to "distance".
            distance_data (Union[NDArray, Som]): The channel-last array of the data points used for knn graph
                construction. Used for distance calculation if fix_lonely_nodes_method is `distance`.
                Defaults to None.
            distance_metric (Literal["euclidean", "manhattan", "correlation", "cosine"], optional): The distance metric
                to be used for nearest neighbor calculation. Defaults to "euclidean".
            accuracy (Literal["fast", "accurate"], optional): The accuracy of the nearest neighbor calculation. Used
                only when method is `simple` or `batch_balanced`. Defaults to "accurate".
            job_count (int, optional): Parallelization core count when method is `simple`. Defaults to -1.
            use_gpu (bool, optional): Whether to use the GPU for nearest neighbor calculation if possible.
                Defaults to True.

        Returns:
            Graph: The shared nearest neighbor graph.

        Raises:
            ValueError: Raised when the distance data is not provided when using the distance method.
            NotImplementedError: Raised when the specified method for connecting lonely nodes is not supported.
        """
        neighbors: List[Set[int]] = [set(adjacency_list) for adjacency_list in knn_graph.get_adjlist(mode="all")]
        nodes: Union[List[int], Set[int]] = knn_graph.vs.indices
        edges = cls.get_shared_neighbors(nodes, neighbors, shared_count_threshold)

        if fix_lonely_nodes:
            # lonely nodes are the nodes that don't appear in the edge list
            lonely_nodes = set(nodes) - {node for edge in edges for node in edge}

            if fix_lonely_nodes_method == "random":
                # for each lonely node, get all its neighbors in the knn graph and connect it to one of them
                for node in lonely_nodes:
                    # get all neighbors and shuffle them
                    node_neighbors: List[int] = knn_graph.neighbors(node)
                    shuffle(node_neighbors)

                    if len(node_neighbors) > 0:
                        edges.append((node, node_neighbors[0]))

                # remove lonely nodes without any neighbors in the original knn graph
                nodes = {node for edge in edges for node in edge}

            elif fix_lonely_nodes_method == "distance":
                if distance_data is None:
                    raise ValueError("Distance data must be provided when using the distance method.")

                if isinstance(distance_data, Som):
                    distance_data = distance_data.get_nodes()

                data_shape = distance_data.shape
                distance_data = distance_data.reshape(-1, data_shape[-1])

                # for each lonely node, find its nearest neighbor and connect them
                neighbor_finder = get_neighbor_finder(
                    neighbor_count=1,
                    channel_count=data_shape[-1],
                    distance_metric=distance_metric,
                    accuracy=accuracy,
                    job_count=job_count,
                    use_gpu=use_gpu,
                )
                neighbor_finder.fit(distance_data)
                neighbor_idx = neighbor_finder.kneighbors(
                    distance_data,
                    n_neighbors=2,
                    return_distance=False,
                )

                for node in lonely_nodes:
                    neighbor = neighbor_idx[node, 1]
                    edges.append((node, neighbor))

            else:
                raise NotImplementedError(f"Method {fix_lonely_nodes_method} is not supported.")

        return Graph(set(edges), directed=False)
