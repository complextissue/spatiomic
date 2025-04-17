"""Exposes neighborhood graph functions."""

from ._knn_graph import KnnGraph as knn_graph
from ._snn_graph import SnnGraph as snn_graph

__all__ = [
    "knn_graph",
    "snn_graph",
]
