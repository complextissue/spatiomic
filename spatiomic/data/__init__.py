"""Expose i/o functions from the data submodule."""

from ._anndata_from_array import anndata_from_array
from ._read import Read as read
from ._subsample import Subsample as subsample
from ._subset import subset

__all__ = [
    "anndata_from_array",
    "read",
    "subsample",
    "subset",
]
