from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from spatiomic._internal._anndata_method import anndata_method
from spatiomic._internal._check_channel_dimension import check_channel_dimension
from spatiomic._internal._data_method import data_method
from spatiomic._internal._import_package import import_package
from spatiomic.process._base import Processer


class Log1p(Processer):
    """A class to log1p data channel-wise."""

    def __init__(
        self,
        use_gpu: bool = True,
    ) -> None:
        """Initialise the Log1p class and set the numpy/cupy backend.

        Usage:

        .. code-block:: python

            data = my_xyc_image

            log1p = so.process.log1p()
            data_log1p = log1p.fit_transform(data)

            data_recovered = log1p.inverse_transform(data_log1p)

        .. warning:: Critically evaluate whether the log1p transform is appropriate for your data. While often used for
            variance stabilization, it may distort the data in ways that are not desirable and is often not necessary.

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

    @anndata_method(input_attribute="X", output_layer="X_log1p")
    @data_method
    def transform(
        self,
        data: NDArray,
    ) -> NDArray:
        """Log1p transform the data channel-wise.

        Args:
            data (NDArray): Data to log1p transform, channels being the last dimension.

        Returns:
            NDArray: The log1p transformed data.
        """
        # check that the data is not negative
        assert bool(self.xp.any(data < 0)) is False, (
            "Data contains negative values. Log1p transform will result in NaNs."
        )
        check_channel_dimension(data.shape)

        data_shape = data.shape
        data = self.xp.reshape(self.xp.array(data), (-1, data_shape[-1]))
        data = self.xp.log1p(data).get() if self.xp.__name__ == "cupy" else self.xp.log1p(data)

        return data.reshape(data_shape)

    @anndata_method(input_attribute="X", output_layer="X_inverse_log1p")
    @data_method
    def inverse_transform(
        self,
        data: NDArray,
    ) -> NDArray:
        """Inverse log1p transform the data channel-wise.

        Args:
            data (NDArray): Data to inverse log1p transform, channels being the last dimension.

        Returns:
            NDArray: The inverse log1p transformed data.
        """
        # check that the data is not negative
        assert bool(np.any(data < 0)) is False, (
            "Data contains negative values. Inverse log1p transform will result in NaNs."
        )

        data_shape = data.shape
        data = self.xp.reshape(self.xp.array(data), (-1, data_shape[-1]))
        data = data = self.xp.expm1(data).get() if self.xp.__name__ == "cupy" else self.xp.expm1(data)

        return data.reshape(data_shape)

    @anndata_method(input_attribute="X", output_layer="X_log1p")
    @data_method
    def fit_transform(
        self,
        data: NDArray,
    ) -> NDArray:
        """Fit and transform the data channel-wise.

        Args:
            data (NDArray): Data to fit and transform, channels being the last dimension.

        Returns:
            NDArray: The log1p transformed data.
        """
        return self.transform(data)  # type: ignore
