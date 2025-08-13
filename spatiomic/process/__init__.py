"""Pre-/post-processing classes for image data.

`spatiomic` provides classes that facilitate common pre-/post-processing tasks for image data. However, this selection
is not exhaustive. Many tools for transforming image data exist and users are encouraged to use the `spatiomic` classes
togther with other tools, such as `scikit-image`.
"""

from ._arcsinh import Arcsinh as arcsinh
from ._clip import Clip as clip
from ._log1p import Log1p as log1p
from ._normalize import Normalize as normalize
from ._register import Register as register
from ._zscore import ZScore as zscore

__all__ = [
    "arcsinh",
    "clip",
    "log1p",
    "normalize",
    "register",
    "zscore",
]
