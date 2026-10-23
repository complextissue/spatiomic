from types import ModuleType
from typing import Any
from warnings import warn

import graspologic_native as gn
import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from spatiomic._internal._anndata_method import anndata_method
from spatiomic._internal._import_package import import_package
from spatiomic._internal._seed_method import seed_method


def _leiden_cpu(
    edges: list[tuple[str, str, float]],
    resolution: float,
    iteration_count: int | None,
    seed: int,
) -> tuple[dict[str, int], float]:
    """Optimise the modularity of an edge list with `graspologic_native.leiden`.

    Args:
        edges (list[tuple[str, str, float]]): The edges as (source, target, weight), nodes encoded as strings.
        resolution (float): The modularity resolution.
        iteration_count (int | None): The number of Leiden iterations, each starting from the previous partition.
            None iterates until the modularity no longer improves.
        seed (int): The random seed.

    Returns:
        tuple[dict[str, int], float]: The community of each node in the edge list and the modularity.
    """

    def leiden(starting_communities: dict[str, int] | None, iterations: int) -> tuple[float, dict[str, int]]:
        result: tuple[float, dict[str, int]] = gn.leiden(
            edges,
            starting_communities=starting_communities,
            resolution=resolution,
            randomness=0.001,
            iterations=iterations,
            use_modularity=True,
            seed=seed,
            trials=1,
        )
        return result

    # a single call avoids converting the edge list for every iteration
    if iteration_count is not None:
        modularity, partition = leiden(None, iteration_count)
        return partition, modularity

    modularity, partition = leiden(None, 1)
    while True:
        new_modularity, new_partition = leiden(partition, 1)
        if new_modularity <= modularity:
            return partition, modularity
        modularity, partition = new_modularity, new_partition


def _leiden_gpu(
    graph_library: ModuleType,
    edges: list[tuple[int, int, float]],
    resolution: float,
    iteration_count: int | None,
    seed: int,
) -> tuple[dict[int, int], float]:
    """Optimise the modularity of an edge list with `cugraph.leiden`.

    Args:
        graph_library (ModuleType): The imported `cugraph` module.
        edges (list[tuple[int, int, float]]): The edges as (source, target, weight).
        resolution (float): The modularity resolution.
        iteration_count (int | None): The maximum number of levels, None for 100.
        seed (int): The random seed.

    Returns:
        tuple[dict[int, int], float]: The community of each node in the edge list and the modularity.
    """
    parts, modularity = graph_library.leiden(
        graph_library.from_pandas_edgelist(
            pd.DataFrame(edges, columns=["source", "destination", "weight"]), edge_attr="weight"
        ),
        max_iter=100 if iteration_count is None else iteration_count,
        resolution=resolution,
        random_state=seed,
    )
    return dict(zip(parts["vertex"].to_numpy(), parts["partition"].to_numpy(), strict=True)), float(modularity)


def _community_array(partition: dict[Any, int], node_count: int) -> NDArray[np.int64]:
    """Convert a node-to-community mapping to consecutive community labels in node order.

    Nodes missing from the mapping (nodes without edges) each get a community of their own.

    Args:
        partition (dict[Any, int]): The community of each node index (as int or str) that has edges.
        node_count (int): The number of nodes in the graph.

    Returns:
        NDArray[np.int64]: The community of each node.
    """
    communities = np.full(node_count, -1, dtype=np.int64)
    for node, community in partition.items():
        communities[int(node)] = community

    connected = communities >= 0
    communities[connected] = np.unique(communities[connected], return_inverse=True)[1]
    communities[~connected] = communities.max(initial=-1) + 1 + np.arange(np.count_nonzero(~connected))
    return communities


class Leiden:
    """Class that exposes the Leiden graph clustering algorithm."""

    def __init__(self) -> None:
        """Initialise the Leiden clustering class."""
        self.graph: nx.Graph | None = None

    @anndata_method(output_obs="leiden_clusters", pass_data=False)
    @seed_method
    def predict(
        self,
        graph: nx.Graph,
        resolution: float = 1.0,
        iteration_count: int | None = 5,
        seed: int = 0,
        use_gpu: bool = False,
        *args: Any,  # noqa: ARG002
        **kwargs: dict,  # noqa: ARG002
    ) -> tuple[NDArray[np.int64], float]:
        """Perform Leiden clustering on a graph by optimising its modularity.

        The CPU version uses `graspologic_native.leiden`, the GPU version `cugraph.leiden`. Edges are weighted by their
        `weight` attribute if present.

        .. warning:: The GPU version is not the full Leiden algorithm and is therefore not the default. `cugraph.leiden`
            refines communities greedily: its random refinement is disabled and its `theta` parameter is ignored
            (https://github.com/rapidsai/cugraph/issues/5724). It also runs a single Leiden pass, so nodes can remain
            in communities that lower the modularity. Its communities can therefore differ from, and have a lower
            modularity than, those of the CPU version.

        Args:
            graph (nx.Graph): An undirected networkx Graph with integer nodes `0` to `n - 1`, e.g., from
                `so.neighbor.knn_graph`, to be optimised for community detection.
            resolution (float, optional): Scales the minimum interconnectedness for a positive modularity.
                Higher values result in more but deeper interconnected communities. Defaults to 1.0.
            iteration_count (int | None, optional): The number of Leiden iterations, each starting from the
                previous partition (CPU), or the maximum number of levels (GPU). None iterates until the modularity
                no longer improves on the CPU and allows 100 levels on the GPU. Defaults to 5.
            seed (int, optional): Random seed to use for the Leiden algorithm. Defaults to 0.
            use_gpu (bool, optional): Whether to use the GPU (`cugraph`) instead of the CPU for the Leiden algorithm,
                see the warning above. Defaults to False.

        Returns:
            tuple[NDArray[np.int64], float]: The community of each node, in the order of the nodes, and the final
                modularity.
        """
        graph_library: ModuleType | None
        graph_library, use_gpu = (
            import_package(  # type: ignore
                "cugraph",
                raise_error=False,
                return_success=True,
            )
            if use_gpu
            else (None, False)
        )

        self.graph = graph
        node_index = {node: index for index, node in enumerate(graph.nodes)}
        edges = [
            (node_index[source], node_index[target], float(weight))
            for source, target, weight in graph.edges(data="weight", default=1.0)
        ]

        if not edges:
            return _community_array({}, len(node_index)), 0.0

        partition: dict[Any, int]

        if use_gpu:
            warn(
                "The GPU version of Leiden is not the full Leiden algorithm and may provide different communities "
                "with a lower modularity than the CPU version.",
                UserWarning,
                stacklevel=2,
            )
            partition, modularity = _leiden_gpu(graph_library, edges, resolution, iteration_count, seed)  # type: ignore
        else:
            partition, modularity = _leiden_cpu(
                [(str(source), str(target), weight) for source, target, weight in edges],
                resolution,
                iteration_count,
                seed,
            )

        return _community_array(partition, len(node_index)), float(modularity)
