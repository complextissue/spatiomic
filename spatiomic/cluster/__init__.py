"""Make all clustering classes available in the cluster submodule."""

from spatiomic.dimension._som import Som as som

from ._agglomerative import Agglomerative as agglomerative
from ._kmeans import KMeans as kmeans
from ._leiden import Leiden as leiden

__all__ = [
    "agglomerative",
    "biclustering",
    "kmeans",
    "leiden",
    "som",
]
