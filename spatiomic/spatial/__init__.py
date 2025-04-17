"""Make all spatial analysis classes available in the spatial submodule.

.. warning:: Methods in the spatial module are not yet GPU accelerated.
"""

import os

# Set the environment variable so that geopandas uses Shapeley instead of PyGEOS
os.environ["USE_PYGEOS"] = "0"

from ._autocorrelation import Autocorrelation as autocorrelation
from ._bivariate_correlation import (
    BivariateCorrelation as bivariate_correlation,
)
from ._local_autocorrelation import LocalAutocorrelation as local_autocorrelation
from ._local_heteroskedasticity import LocalHeteroskedasticity as local_heteroskedasticity
from ._spatial_weights import neighborhood_offset, spatial_weights
from ._vicinity_composition import vicinity_composition
from ._vicinity_graph import vicinity_graph

__all__ = [
    "autocorrelation",
    "bivariate_correlation",
    "local_autocorrelation",
    "local_heteroskedasticity",
    "neighborhood_offset",
    "spatial_weights",
    "vicinity_composition",
    "vicinity_graph",
]
