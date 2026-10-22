from types import ModuleType
from typing import Any, Optional, Tuple, Union
from warnings import warn

import networkx as nx
import numpy as np
import pandas as pd

from spatiomic._internal._anndata_method import anndata_method
from spatiomic._internal._import_package import import_package
from spatiomic._internal._seed_method import seed_method


class Leiden:
    """Class that exposes the Leiden graph clustering algorithm."""

    def __init__(self) -> None:
        """Initialise the Leiden clustering class."""
        self.graph: Union[nx.Graph, None] = None

    @anndata_method(output_obs="leiden_clusters", pass_data=False)
    @seed_method
    def predict(
        self,
        graph: nx.Graph,
        resolution: float = 1.0,
        iteration_count: Optional[int] = None,
        seed: int = 0,
        use_gpu: bool = True,
        *args: Any,  # noqa: ARG002
        **kwargs: dict,  # noqa: ARG002
    ) -> Tuple[np.ndarray, float]:
        """Perform Leiden clustering on a graph by optimising its modularity.

        The CPU version uses `networkx.community.leiden_communities`, the GPU version `cugraph.leiden`. Edges are
        weighted by their `weight` attribute if present.

        .. warning:: The GPU version of Leiden may provide different results than the CPU version.

        Args:
            graph (nx.Graph): An undirected networkx Graph with integer nodes `0` to `n - 1`, e.g., from
                `so.neighbor.knn_graph`, to be optimised for community detection.
            resolution (float, optional): Scales the minimum interconnectedness for a positive modularity.
                Higher values result in more but deeper interconnected communities. Defaults to 1.0.
            iteration_count (Optional[int], optional): The maximum number of levels (CPU) or iterations (GPU) of the
                Leiden algorithm. Defaults to None, which runs until the modularity no longer improves on the CPU and
                for 100 iterations on the GPU.
            seed (int, optional): Random seed to use for the Leiden algorithm. Defaults to 0.
            use_gpu (bool, optional): Whether to use the GPU or CPU for the Leiden algorithm. Defaults to True.

        Returns:
            Tuple[np.ndarray, float]: The community of each node, in the order of the nodes, and the final modularity.
        """
        graph_library: Optional[ModuleType]
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
        communities = np.full(graph.number_of_nodes(), -1, dtype=np.int64)

        if use_gpu:
            warn(
                "The GPU version of Leiden may provide different results than the CPU version.",
                UserWarning,
                stacklevel=2,
            )

            edges = pd.DataFrame(
                [
                    (node_index[source], node_index[target], weight)
                    for source, target, weight in graph.edges(data="weight", default=1.0)
                ],
                columns=["source", "destination", "weight"],
            )
            parts, modularity = graph_library.leiden(  # type: ignore
                graph_library.from_pandas_edgelist(edges, edge_attr="weight"),  # type: ignore
                max_iter=100 if iteration_count is None else iteration_count,
                resolution=resolution,
                random_state=seed,
            )

            communities[parts["vertex"].to_numpy()] = parts["partition"].to_numpy()
            return communities, float(modularity)

        community_sets = nx.community.leiden_communities(
            graph,
            resolution=resolution,
            max_level=iteration_count,
            seed=seed,
            metric="modularity",
        )

        for community, nodes in enumerate(community_sets):
            communities[[node_index[node] for node in nodes]] = community

        modularity = nx.community.modularity(graph, community_sets, resolution=resolution)

        return communities, float(modularity)
