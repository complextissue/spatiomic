"""Spatiomic: Spatial omics analyses in Python."""

submodules = [
    "cluster",
    "data",
    "dimension",
    "neighbor",
    "plot",
    "process",
    "spatial",
    "tool",
]

__all__ = [*submodules]

from . import cluster, data, dimension, neighbor, plot, process, spatial, tool  # noqa: F401
