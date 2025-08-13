"""Spatiomic: Spatial omics analyses in Python."""

from importlib.metadata import version

__version__ = version("spatiomic")

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

__all__ = [*submodules, "__version__"]

from . import cluster, data, dimension, neighbor, plot, process, spatial, tool  # noqa: F401
