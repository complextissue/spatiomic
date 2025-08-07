"""Spatiomic: Spatial omics analyses in Python."""

from ._version import __version__

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

__all__ = submodules + ["__version__"]

from . import cluster, data, dimension, neighbor, plot, process, spatial, tool  # noqa: F401
