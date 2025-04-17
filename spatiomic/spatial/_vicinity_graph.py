from typing import Union

import numpy as np
import pandas as pd
from networkx import DiGraph, Graph
from numpy.typing import NDArray


def vicinity_graph(
    data: Union[NDArray, pd.DataFrame],
    ignore_identities: bool = True,
    normalize: bool = True,
    weighted: bool = True,
    directed: bool = True,
) -> Graph:
    """Construct a vicinity graph from a dataframe.

    Args:
        data (Union[NDArray, pd.DataFrame]): The dataframe or its values where each row is a cluster and all columns are
            the counts of clusters in the vicinity.
        ignore_identities (bool, optional): Whether to ignore pixels from the same cluster in the vicinity.
            Defaults to True.
        normalize (bool, optional): Whether to normalize the counts in the vicinity. Defaults to True.
        weighted (bool, optional): Whether to weighted the edges by the counts in the vicinity. Defaults to True.
        directed (bool, optional): Whether to construct a directed graph. Defaults to True.

    Returns:
        Graph: The spatial vicinity graph.
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    assert len(data.shape) == 2, "The dataframe must be two-dimensional."
    assert data.shape[0] == data.shape[1], "The dataframe must be square."
    assert (data.index == data.columns).all(), "The dataframe must have the same row and column names."
    assert (data.to_numpy() >= 0).all(), "The dataframe must not contain negative values."

    # set the eye to zero if the identities should be ignored
    if ignore_identities:
        np.fill_diagonal(data.to_numpy(), 0)

    # row normalize the dataframe if the counts should be normalized
    if normalize:
        data = data.div(data.sum(axis=1), axis=0)

    # get the edges
    edges = np.argwhere(data.to_numpy() > 0)

    # get the weights
    weights = data.to_numpy()[edges[:, 0], edges[:, 1]]

    # sum the weights of the edges if the graph is undirected and filter unique edges
    if not directed:
        edges = np.sort(edges, axis=1)

        if weighted:
            weights = data.to_numpy()[edges[:, 0], edges[:, 1]] + data.to_numpy()[edges[:, 1], edges[:, 0]]

        edges = np.unique(edges, axis=0)

    # construct the graph
    graph = DiGraph() if directed else Graph()

    # add the edges
    if weighted:
        graph.add_weighted_edges_from([(edges[i, 0], edges[i, 1], weights[i]) for i in range(edges.shape[0])])
    else:
        graph.add_edges_from([(edges[i, 0], edges[i, 1]) for i in range(edges.shape[0])])

    return graph
