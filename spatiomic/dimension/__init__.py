"""Make all dimensionality reduction classes available in the dimension submodule."""

from ._pca import Pca as pca
from ._som import Som as som
from ._tsne import Tsne as tsne
from ._umap import Umap as umap

__all__ = [
    "pca",
    "som",
    "tsne",
    "umap",
]
