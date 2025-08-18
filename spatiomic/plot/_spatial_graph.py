from colorsys import hls_to_rgb
from typing import Optional, Union
from warnings import warn

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from networkx import DiGraph, Graph


def spatial_graph(
    graph: Union[DiGraph, Graph],
    reference_graph: Optional[Union[DiGraph, Graph]] = None,
    edge_threshold: float = 0.0,
    remove_isolated_nodes: bool = True,
    edge_stroke_weight_multiplier: float = 10.0,
    edge_color_multiplier: float = 0.6,
    title: Optional[str] = None,
    seed: int = 0,
    ax: Optional[plt.Axes] = None,
) -> Union[plt.Figure, plt.Axes]:
    """Plot a graph of spatial cluster-cluster relationships.

    Args:
        graph (Union[DiGraph, Graph]): The neighborhood graph.
        reference_graph (Optional[Union[DiGraph, Graph]], optional): The reference graph. Defaults to None.
        edge_threshold (float, optional): The threshold for the edge weights. Defaults to 0.0.
        remove_isolated_nodes (bool, optional): Whether to remove isolated nodes. Defaults to True.
        edge_stroke_weight_multiplier (float, optional): The multiplier for the edge weights for stroke width.
            Defaults to 10.0.
        edge_color_multiplier (float, optional): The multiplier for the edge weights for color. Defaults to 0.6.
        title (str, optional): The title of the plot. Defaults to None.
        seed (int, optional): Seed for the layout algorithm, only used if graphviz is not installed. Defaults to 0.
        ax: Existing axes to plot on. If None, creates new figure. Defaults to None.

    Returns:
        plt.Figure if ax is None, otherwise the provided plt.Axes.
    """
    # plot the graph
    if ax is None:
        sns.set_theme(style="white", font="Arial")
        sns.set_context("paper")
        fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=200)
        return_fig = True
    else:
        return_fig = False

    graph = graph.copy()

    try:
        # get the positions of the nodes via graphviz with the Fruchterman-Reingold algorithm
        pos = nx.nx_agraph.graphviz_layout(graph, prog="fdp", args="-Goverlap=false -Gmaxiter=100000 -Gpenalty=0.75")
    except ImportError:
        warn("Graphviz not installed, using spring layout from networkx.", ImportWarning, stacklevel=2)
        pos = nx.spring_layout(graph, k=0.5, iterations=1000, seed=seed)

    # remove edges that are not in the reference graph
    if isinstance(reference_graph, (DiGraph, Graph)):
        edges_to_remove = [(u, v) for (u, v) in graph.edges() if not reference_graph.has_edge(u, v)]
        graph.remove_edges_from(edges_to_remove)

    # remove edges with a weight below a threshold
    if edge_threshold > 0.0:
        edges_to_remove = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] < edge_threshold]
        graph.remove_edges_from(edges_to_remove)

    # remove isolated nodes
    if remove_isolated_nodes:
        graph.remove_nodes_from(list(nx.isolates(graph)))

        if isinstance(reference_graph, (DiGraph, Graph)):
            reference_graph.remove_nodes_from(list(nx.isolates(graph)))

    edge_color: Union[list, str] = "black"

    # calculate the color of the edges if a reference is set
    if isinstance(reference_graph, (DiGraph, Graph)):
        weight_difference = [
            d["weight"] - reference_graph.edges[(u, v)]["weight"] for (u, v, d) in graph.edges(data=True)
        ]
        edge_color = [
            (
                hls_to_rgb(0, 0.6 - d * edge_color_multiplier, 1)
                if d > 0
                else hls_to_rgb(0.6, 0.6 + d * edge_color_multiplier, 1)
            )
            for d in weight_difference
        ]

    # draw the nodes
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=400,
        node_color="lightblue",
        linewidths=1,
        edgecolors="black",
        alpha=1,
        ax=ax,
        node_shape="o",
    )

    # draw the edges
    nx.draw_networkx_edges(
        graph,
        pos,
        node_size=400,
        width=[(d["weight"] * edge_stroke_weight_multiplier) for (u, v, d) in graph.edges(data=True)],
        edge_color=edge_color,
        arrowsize=10,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
        ax=ax,
    )

    # draw the labels
    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=12,
        font_color="black",
        font_family="Arial",
        ax=ax,
    )

    # set the title
    if title is not None:
        ax.set_title(title, fontsize=16)

    ax.axis("off")

    if return_fig:
        plt.tight_layout()
        return fig
    else:
        return ax
