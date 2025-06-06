from pickle import dump, load
from typing import TYPE_CHECKING, Literal, Optional, Union
from warnings import warn

import numpy as np
from numpy.typing import NDArray

from spatiomic._internal._anndata_method import anndata_method
from spatiomic._internal._check_channel_dimension import check_channel_dimension
from spatiomic._internal._data_method import data_method
from spatiomic._internal._import_package import import_package
from spatiomic.data._subsample import Subsample
from spatiomic.process._base import Processer


class Clip(Processer):
    """Clip data channel-wise to specified percentiles or to a specified minimum and maximum range."""

    def __init__(
        self,
        method: Literal["percentile", "minmax"] = "percentile",
        percentile_min: float = 1.0,
        percentile_max: float = 99.9,
        min_value: Optional[Union[float, NDArray]] = None,
        max_value: Optional[Union[float, NDArray]] = None,
        fill_value: Optional[Union[float, NDArray]] = None,
        use_gpu: bool = True,
    ) -> None:
        """Initialise the Clip class and set the clipping parameters.

        There are two methods of operation, either using percentiles or a specified range. If percentiles are used, the
        channel-wise minimum and maximum values are calculated based on the provided percentile values. If a range is
        used, the provided minimum and maximum values are used to clip the data. If a fill value is provided, values
        outside of the range are replaced with the fill value.

        Args:
            method (Literal["percentile", "minmax"], optional): Whether to use percentiles or min and max values.
                Defaults to "percentile".
            percentile_min (float, optional): The percentile to calculate the channel-wise minimum clipping value.
                Defaults to 1.0.
            percentile_max (float, optional): The percentile to calculate the channel-wise maximum clipping value.
                Defaults to 99.9.
            min_value (Optional[Union[float, NDArray]], optional): The minimum clipping value. Defaults to None.
            max_value (Optional[Union[float, NDArray]], optional): The maximum clipping value. Defaults to None.
            fill_value (Optional[Union[float, NDArray]], optional): The value to replace values outside of the range
                with. Defaults to None.
            use_gpu (bool, optional): Whether to use cupy on the GPU or numpy on the CPU. Defaults to True.
        """
        assert method != "percentile" or (percentile_min <= 99.99 and percentile_min > 0.0)
        assert method != "percentile" or (percentile_max <= 100.0 and percentile_max > 0.01)

        self.method = method
        self.fitted = method != "percentile"

        self.percentile_min = percentile_min
        self.percentile_max = percentile_max
        self.percentile_min_absolute: Optional[NDArray] = None
        self.percentile_max_absolute: Optional[NDArray] = None

        import_result = (
            import_package("cupy", alternative=np, return_success=True) if use_gpu else (np, False)  # type: ignore
        )

        if TYPE_CHECKING:
            self.xp = np
            self.use_gpu = False
        else:
            self.xp: "np" = import_result[0]
            self.use_gpu = import_result[1]

        if isinstance(min_value, float):
            min_value = self.xp.array(min_value).reshape((1))
        elif min_value is not None:
            min_value = self.xp.array(min_value)

        if isinstance(max_value, float):
            max_value = self.xp.array(max_value).reshape((1))
        elif max_value is not None:
            max_value = self.xp.array(max_value)

        fill_value_array: Optional[NDArray] = None
        if isinstance(fill_value, float):
            fill_value_array = self.xp.array(fill_value).reshape((1))
        elif fill_value is not None:
            fill_value_array = self.xp.array(fill_value)

        self.min_value: Optional[NDArray] = min_value
        self.max_value: Optional[NDArray] = max_value
        self.fill_value: Optional[NDArray] = fill_value_array

    @anndata_method(input_attribute="X")
    @data_method
    def fit(
        self,
        data: NDArray,
        fraction: float = 1.0,
    ) -> None:
        """Calculate the channel-wise minimum and maximum clipping value based on provided percentiles.

        Args:
            data (NDArray): An array with channels being the last dimension.
            fraction (float, optional): Fraction of data to randomly use for min and max value calculation.
                Defaults to 1.0.
        """
        check_channel_dimension(data.shape)

        if self.method != "percentile":
            warn("Only percentile method needs to be fitted.", UserWarning, stacklevel=2)
            return

        # flatten array in every dimension but the channel dimension
        data = self.xp.reshape(self.xp.asarray(data), (-1, data.shape[-1]))
        data = self.xp.asarray(data)

        # get random subsample if fraction is not 1.0
        if fraction < 1.0:
            data = Subsample().fit_transform(
                data,
                fraction=fraction,
            )

        if self.method == "percentile":
            self.percentile_min_absolute = self.xp.expand_dims(
                self.xp.percentile(data, self.percentile_min, axis=0),
                axis=0,
            )
            self.percentile_max_absolute = self.xp.expand_dims(
                self.xp.percentile(data, self.percentile_max, axis=0),
                axis=0,
            )

            if self.xp.__name__ == "cupy":
                self.percentile_min_absolute = self.percentile_min_absolute.get()  # type: ignore
                self.percentile_max_absolute = self.percentile_max_absolute.get()  # type: ignore

        self.fitted = True

    @anndata_method(input_attribute="X", output_layer="X_clipped")
    @data_method
    def transform(
        self,
        data: NDArray,
    ) -> NDArray:
        """Calculate the clipped data based on previously fit min and max clipping values.

        Args:
            data (NDArray): The data, at least two-dimensional, last dimension being channels.

        Returns:
            NDArray: The channel-wise clipped data.
        """
        assert self.fitted is True, "Clip class has not been fitted yet."
        assert self.method == "percentile" or self.min_value is not None or self.max_value is not None, (
            "No min and max clipping limit set."
        )
        assert self.percentile_min_absolute is None or data.shape[-1] == self.percentile_min_absolute.shape[-1], (
            f"Data and min/max values must have same channel dimensionality. Data has {data.shape[-1]} channels,"
            f"min/max values have {self.percentile_min_absolute.shape[-1]} channels."
        )

        # flatten array in every dimension but the channel dimension
        data_shape = data.shape
        data = self.xp.array(data).reshape((-1, data_shape[-1]))

        if self.method == "percentile":
            percentile_min_absolute = (
                self.xp.array(self.percentile_min_absolute) if self.percentile_min_absolute is not None else None
            )
            percentile_max_absolute = (
                self.xp.array(self.percentile_max_absolute) if self.percentile_max_absolute is not None else None
            )

            if self.fill_value is not None:
                if self.percentile_min_absolute is not None:
                    data = self.xp.where(
                        data < self.percentile_min_absolute,
                        self.fill_value[..., np.newaxis],
                        data,
                    )
                if self.percentile_max_absolute is not None:
                    data = self.xp.where(
                        data > self.percentile_max_absolute,
                        self.fill_value[..., np.newaxis],
                        data,
                    )
            elif self.percentile_max_absolute is not None or percentile_min_absolute is not None:
                data = self.xp.clip(
                    data,
                    a_min=(percentile_min_absolute if percentile_min_absolute is not None else None),
                    a_max=(percentile_max_absolute if percentile_max_absolute is not None else None),
                )
        elif self.method == "minmax":
            if self.max_value is not None or self.min_value is not None:
                if self.fill_value:
                    if self.min_value is not None:
                        data = self.xp.where(
                            data < self.min_value,
                            self.fill_value[..., np.newaxis],
                            data,
                        )

                    if self.max_value is not None:
                        data = self.xp.where(
                            data > self.max_value,
                            self.fill_value[..., np.newaxis],
                            data,
                        )
                else:
                    data = self.xp.clip(
                        data,
                        a_min=(self.xp.expand_dims(self.min_value, axis=0) if self.min_value is not None else None),
                        a_max=(self.xp.expand_dims(self.max_value, axis=0) if self.max_value is not None else None),
                    )
            else:
                raise ValueError("No min and max clipping limit set during initialization.")

        if self.xp.__name__ == "cupy":
            return self.xp.reshape(data, data_shape).get()  # type: ignore
        else:
            return self.xp.reshape(data, data_shape)  # type: ignore

    @anndata_method(input_attribute="X", output_layer="X_clipped")
    @data_method
    def fit_transform(
        self,
        data: NDArray,
        fraction: float = 1.0,
    ) -> NDArray:
        """Fit the min and max value and transform provided data.

        Args:
            data (NDArray): An array with channels being the last dimension.
            fraction (float, optional): Fraction of data to randomly use for min and max value calculation.
                Defaults to 1.0.

        Returns:
            NDArray: The channel-wise clipped data.
        """
        if self.method == "percentile":
            self.fit(
                data,
                fraction=fraction,
            )
        return np.array(self.transform(data))

    def save(
        self,
        save_path: str,
    ) -> None:
        """Save the Clip object to a file.

        Args:
            save_path (str): The path to save the Clip object to.
        """
        config = {
            "init": {
                "method": self.method,
                "percentile_min": self.percentile_min,
                "percentile_max": self.percentile_max,
                "min_value": self.min_value,
                "max_value": self.max_value,
                "fill_value": self.fill_value,
                "use_gpu": self.use_gpu,
            },
            "attributes": {
                "fitted": self.fitted,
                "percentile_min_absolute": self.percentile_min_absolute,
                "percentile_max_absolute": self.percentile_max_absolute,
            },
        }

        with open(save_path, "wb") as file:
            dump(config, file)

    @classmethod
    def load(
        cls,
        save_path: str,
    ) -> "Clip":
        """Load a Clip object from a file.

        Args:
            save_path (str): The path to load the Clip object from.

        Returns:
            Clip: The loaded Clip object.
        """
        with open(save_path, "rb") as file:
            config = load(file)  # nosec

        clipper = cls(**config["init"])

        for key, value in config["attributes"].items():
            clipper.__setattr__(key, value)

        return clipper
