"""Library for spatial omics analysis. Lazy-loads submodules for faster import times."""

from importlib.metadata import version
from importlib.util import find_spec
from typing import List

import lazy_loader as lazy

__version__ = version("spatiomic")

__getattr__, __lazy_dir__, _ = lazy.attach_stub(__name__, __file__)


def __dir__() -> List[str]:
    """List the available submodules."""
    return [*__lazy_dir__(), "__version__"]


if find_spec("cuml") is not None:
    import cuml  # type: ignore

    # Always return the cuml results as np.ndarray, this ensures compatibility between GPU and CPU algorithms
    cuml.set_global_output_type("numpy")
