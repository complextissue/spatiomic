from typing import Any, List, Optional, Tuple, Union
from warnings import warn

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from igraph import Graph as Graph
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray

from ._colormap import colormap as create_colormap


def connectivity_graph(
    graph: Graph,
    title: str = "Connectivity Graph",
    clusters: Optional[NDArray] = None,
    cluster_legend: bool = False,
    node_size: int = 50,
    figsize: Tuple[Union[int, float], Union[int, float]] = (7.5, 7.5),
    colormap: Optional[ListedColormap] = None,
    seed: int = 0,
    ax: Optional[plt.Axes] = None,
) -> Union[plt.Figure, plt.Axes]:
    """Plot a connectivity graph.

    Args:
        graph (Graph): The graph to be plotted.
        title (str, optional): Title for the figure. Defaults to "Connectivity Graph".
        clusters (Optional[NDArray], optional): clusters for each vertex of the graph to be displayed.
            Defaults to None.
        cluster_legend (bool, optional): Whether to include a legend for the clusters. Defaults to False.
        node_size (int, optional): Size of the nodes in the figure. Defaults to 50.
        figsize (Tuple[Union[int, float], Union[int, float]], optional): Size of the figure. Defaults to (7.5, 7.5).
        colormap (Optional[ListedColormap], optional): The colormap to use. Defaults
            to None.
        seed (int, optional): Seed for the layout algorithm, only used if graphviz is not installed. Defaults to 0.

        ax: Existing axes to plot on. If None, creates new figure. Defaults to None.

    Returns:
        plt.Figure if ax is None, otherwise the provided plt.Axes.
    """
    # plot the graph
    if ax is None:
        sns.set_theme(style="white", font="Arial")
        sns.set_context("paper")
        fig = plt.figure(figsize=figsize, dpi=300)
        ax1 = fig.add_subplot()
        return_fig = True
        show_colorbar = True
    else:
        ax1 = ax
        return_fig = False
        show_colorbar = False

    graph_nx = graph.to_networkx()

    try:
        # get the positions of the nodes via graphviz with the Fruchterman-Reingold algorithm
        pos = nx.nx_agraph.graphviz_layout(
            graph_nx,
            prog="fdp",
            args="-Goverlap=false -Gmaxiter=100000 -Gpenalty=0.75",
        )
    except ImportError:
        warn("Graphviz not installed, using spring layout from networkx.", ImportWarning, stacklevel=2)
        pos = nx.spring_layout(graph_nx, k=0.5, iterations=100, seed=seed)

    cmap_raw: ListedColormap
    node_colors: Union[str, List[Any]]

    if clusters is not None:
        cluster_count = np.max(clusters).astype(int) + 1
        cmap_raw = create_colormap(color_count=cluster_count) if colormap is None else colormap
        colormap_raw = cmap_raw.colors

        node_colors = [colormap_raw[label] for label in clusters]  # type: ignore
    else:
        node_colors = "lightblue"

    # draw the nodes
    nx.draw_networkx_nodes(
        graph_nx,
        pos,
        node_size=node_size,
        node_color=node_colors,
        linewidths=1,
        edgecolors="black",
        alpha=1,
        ax=ax1,
        node_shape="o",
    )

    # draw the edges
    nx.draw_networkx_edges(
        graph_nx,
        pos,
        node_size=node_size,
        width=1,
        edge_color="grey",
        arrowsize=10,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
        ax=ax1,
    )

    ax1.set_title(title, y=1.05)
    ax1.set_yticks([])
    ax1.set_xticks([])

    if clusters is not None and cluster_legend and isinstance(cmap_raw, ListedColormap) and show_colorbar:
        # draw the colorbar as an image
        divider = make_axes_locatable(ax1)

        cluster_count = len(np.unique(clusters))

        ax2 = divider.append_axes("right", size="5%", pad=0.05)
        ax2.imshow(np.expand_dims(np.arange(0, cluster_count), axis=1), cmap=cmap_raw)

        ax2.set_yticks(np.arange(0, cluster_count))
        ax2.set_yticklabels(np.arange(0, cluster_count))
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")

        ax2.set_xticks([])

    sns.despine(left=True, bottom=True)

    if return_fig:
        fig.tight_layout()
        return fig
    else:
        return ax1
