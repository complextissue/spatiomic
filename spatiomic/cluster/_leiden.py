import random
from types import ModuleType
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from igraph import Graph, set_random_number_generator

from spatiomic._internal._anndata_method import anndata_method
from spatiomic._internal._import_package import import_package
from spatiomic._internal._seed_method import seed_method


class _IgraphRandomNumberGenerator:
    """RNG class to set a random state for igraph.

    Adapted from scverse/scanpy.
    """

    def __init__(self, random_state: int = 0) -> None:
        self._rng = np.random.RandomState(np.random.PCG64(np.random.SeedSequence(random_state)))

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._rng, "normal" if attr == "gauss" else attr)


class Leiden:
    """Class that exposes the Leiden graph clustering algorithm."""

    def __init__(self) -> None:
        """Initialise the Leiden clustering class."""
        self.graph: Union[Graph, None] = None

    @anndata_method(output_obs="leiden_clusters", pass_data=False)
    @seed_method
    def predict(
        self,
        graph: Graph,
        resolution: float = 1.0,
        iteration_count: int = 1000,
        seed: int = 0,
        use_gpu: bool = True,
        *args: Any,  # noqa: ARG002
        **kwargs: dict,  # noqa: ARG002
    ) -> Tuple[np.ndarray, float]:
        """Create a neighborhood-graph based on the data and perform Leiden clustering on it.

        .. warning:: The GPU version of Leiden may provide different results than the CPU version.

        Args:
            graph (Graph): An igraph Graph to be optimised for community detection.
            resolution (float, optional): Scales the minimum interconnectedness for a positive modularity.
                Higher values result in more but deeper interconnected communities. Defaults to 1.0.
            iteration_count (int, optional): Iteration count to run the Leiden algorithm for. Defaults to 1000.
            seed (int, optional): Random seed to use for the Leiden algorithm. Defaults to 0.
            use_gpu (bool, optional): Whether to use the GPU or CPU for the Leiden algorithm. Defaults to True.

        Returns:
            Tuple[List[int], float, Graph]: A Tuple of a list of the assigned communities, the final modularity
                and the optimised igraph Graph.
        """
        communities: List
        modularity: float
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

        if use_gpu:
            from warnings import warn

            warn(
                "The GPU version of Leiden may provide different results than the CPU version.",
                UserWarning,
                stacklevel=2,
            )

            self.graph = graph

            # perform the Leiden clustering with cuGraph
            parts, modularity = graph_library.leiden(  # type: ignore
                graph_library.from_edgelist(  # type: ignore
                    pd.DataFrame(
                        {
                            "source": [edge.source for edge in graph.es],
                            "destination": [edge.target for edge in graph.es],
                        }
                    ),
                ),
                max_iter=iteration_count,
                resolution=resolution,
                random_state=seed,
            )

            parts = parts.sort_values("vertex")

            return (parts["partition"].to_numpy(), modularity)
        else:
            set_random_number_generator(_IgraphRandomNumberGenerator(seed))

            # return value is an instance of https://igraph.org/python/doc/api/igraph.clustering.VertexClustering.html
            vertex_cluster = graph.community_leiden(
                objective_function="modularity",
                resolution=resolution,
                n_iterations=iteration_count,
            )

            set_random_number_generator(random)

            # get the relevant properties of the graph to return
            communities = vertex_cluster.membership
            modularity = vertex_cluster.modularity
            graph = vertex_cluster.graph

            self.graph = graph

            return (np.array(communities), modularity)
