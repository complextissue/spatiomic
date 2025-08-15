from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from spatiomic._internal._anndata_method import anndata_method
from spatiomic._internal._check_channel_dimension import check_channel_dimension
from spatiomic._internal._data_method import data_method
from spatiomic._internal._import_package import import_package
from spatiomic.process._base import Processer


class Arcsinh(Processer):
    """A class to arcsinh data channel-wise."""

    def __init__(
        self,
        use_gpu: bool = True,
    ) -> None:
        """Initialise the Arcsinh class and set the numpy/cupy backend.

        Usage:

        .. code-block:: python

            data = my_xyc_image

            arcsinh = so.process.arcsinh()
            data_arcsinh = arcsinh.fit_transform(data)

            data_recovered = arcsinh.inverse_transform(data_arcsinh)

        .. warning:: Evaluate whether the arcsinh transform is appropriate for your data. While often used for variance
            stabilization, it may distort the data in ways that are not desirable.

        Args:
            use_gpu (bool, optional): Whether to force numpy usage or use cupy or numpy depending on availability.
                Defaults to True.
        """
        import_result = (
            import_package("cupy", alternative=np, return_success=True) if use_gpu else (np, False)  # type: ignore
        )

        if TYPE_CHECKING:
            self.xp = np
            self.use_gpu = False
        else:
            self.xp: "np" = import_result[0]
            self.use_gpu = import_result[1]

    @anndata_method(input_attribute="X", output_layer="X_arcsinh")
    @data_method
    def transform(
        self,
        data: NDArray,
    ) -> NDArray:
        """Arcsinh transform the data channel-wise.

        Args:
            data (NDArray): Data to arcsinh transform, channels being the last dimension.

        Returns:
            NDArray: The arcsinh transformed data.
        """
        check_channel_dimension(data.shape)

        data_shape = data.shape
        data = self.xp.reshape(self.xp.array(data), (-1, data_shape[-1]))
        data = self.xp.arcsinh(data).get() if self.xp.__name__ == "cupy" else self.xp.arcsinh(data)

        return data.reshape(data_shape)

    @anndata_method(input_attribute="X", output_layer="X_inverse_arcsinh")
    @data_method
    def inverse_transform(
        self,
        data: NDArray,
    ) -> NDArray:
        """Inverse arcsinh transform the data channel-wise.

        Args:
            data (NDArray): Data to inverse arcsinh transform, channels being the last dimension.

        Returns:
            NDArray: The inverse arcsinh transformed data.
        """
        data_shape = data.shape
        data = self.xp.reshape(self.xp.array(data), (-1, data_shape[-1]))
        data = self.xp.sinh(data).get() if self.xp.__name__ == "cupy" else self.xp.sinh(data)

        return data.reshape(data_shape)

    @anndata_method(input_attribute="X", output_layer="X_arcsinh")
    @data_method
    def fit_transform(
        self,
        data: NDArray,
    ) -> NDArray:
        """Fit and transform the data channel-wise.

        Args:
            data (NDArray): Data to fit and transform, channels being the last dimension.

        Returns:
            NDArray: The arcsinh transformed data.
        """
        return self.transform(data)  # type: ignore
