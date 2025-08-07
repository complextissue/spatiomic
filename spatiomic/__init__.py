"""Library for spatial omics analysis. Lazy-loads submodules for faster import times."""

import os
from importlib.metadata import version
from importlib.util import find_spec
from typing import List

import lazy_loader as lazy

__version__ = version("spatiomic")

# Explicitly load all submodules on RTD
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if on_rtd:
    from . import cluster, data, dimension, neighbor, plot, process, spatial, tool
else:
    # Use lazy loading for normal package usage.
    import lazy_loader as lazy

    __getattr__, __lazy_dir__, _ = lazy.attach_stub(__name__, __file__)

__all__ = [
    "cluster",
    "data",
    "dimension",
    "neighbor",
    "plot",
    "process",
    "spatial",
    "tool",
]


def __dir__() -> List[str]:
    """List the available submodules."""
    if on_rtd:
        # Return a static list of submodule names for the RTD build.
        return [*__all__, "__version__"]
    return [*__lazy_dir__(), "__version__"]


if find_spec("cuml") is not None:
    import cuml  # type: ignore

    # Always return the cuml results as np.ndarray, this ensures compatibility between GPU and CPU algorithms
    cuml.set_global_output_type("numpy")
