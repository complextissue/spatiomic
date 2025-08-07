from pickle import dump, load
from typing import TYPE_CHECKING, Union
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler as MinMaxScalerSk

from spatiomic._internal._anndata_method import anndata_method
from spatiomic._internal._check_channel_dimension import check_channel_dimension
from spatiomic._internal._data_method import data_method
from spatiomic.data._subsample import Subsample
from spatiomic.process._base import Processer


class Normalize(Processer):
    """A class to normalize data channel-wise to a range, commonly (0, 1) or (-1, 1)."""

    def __init__(
        self,
        min_value: Union[int, float] = 0,
        max_value: Union[int, float] = 1,
        use_gpu: bool = True,
    ) -> None:
        """Set the parameters for the Normalize class to normalize data channel-wise to a given range.

        Args:
            min_value (int, optional): The lower bound of the range the data should be normalised to. Defaults to 0.0.
            max_value (int, optional): The upper bound of the range the data should be normalised to. Defaults to 1.0.
            use_gpu (bool, optional): Whether to use cuml and cupy on the GPU or sklearn and numpy on the CPU.
                Defaults to True.
        """
        self.min_value = min_value
        self.max_value = max_value
        self.use_gpu = use_gpu

        if TYPE_CHECKING or not self.use_gpu:
            self.xp = np
            min_max_scaler = MinMaxScalerSk
        else:
            try:
                import cupy as cp  # type: ignore
                from cuml.preprocessing import MinMaxScaler  # type: ignore

                min_max_scaler = MinMaxScaler
                self.xp = cp
            except Exception:
                self.xp = np
                self.use_gpu = False
                min_max_scaler = MinMaxScalerSk

        self.fitted = False
        self.estimator = min_max_scaler(feature_range=(min_value, max_value))

    @anndata_method(input_attribute="X")
    @data_method
    def fit(
        self,
        data: NDArray,
        fraction: float = 1.0,
    ) -> None:
        """Fit the MinMaxScaler estimator of the class channel-wise to the data.

        Args:
            data (NDArray): The data of at least two dimensions, the last one being the channels.
            fraction (float, optional): The random subsampling fraction to use to fit the estimator.
                Defaults to 1.0.
        """
        check_channel_dimension(data.shape)

        # flatten array in every dimension but the channel dimension
        data = self.xp.array(data)
        data = self.xp.reshape(data, (-1, data.shape[-1]))

        # get random subsample if fraction is not 1.0
        if fraction < 1.0:
            data = Subsample().fit_transform(
                data,
                fraction=fraction,
            )

        self.estimator.fit(data)
        self.fitted = True

    @anndata_method(input_attribute="X", output_layer="X_normalized")
    @data_method
    def transform(
        self,
        data: NDArray,
        clip: bool = True,
    ) -> NDArray:
        """Normalize the data with the previously fit normalization estimator of the class.

        Args:
            data (NDArray): The data of at least two dimensions, the last one being the channels.
            clip (bool, optional): Whether to restrict the data strictly to the minimum and maximum of the range
                the estimator was fitted to. This only has consequences if the data to be
                transformed differs from the data used for fitting. Defaults to True.

        Returns:
            NDArray: The normalized data.
        """
        assert self.fitted, "The Normalize object has to be fitted to data first."

        data = self.xp.array(data)
        data_shape = data.shape

        # flatten for transform, reshape again thereafter
        data = self.xp.reshape(
            self.estimator.transform(
                self.xp.reshape(data, (-1, data_shape[-1])),
            ),
            data_shape,
        )

        if clip:
            # clip to the desired range in case transformed data not used during fit
            # results in smaller or larger values
            data = self.xp.clip(
                data,
                self.min_value,
                self.max_value,
            )

        if self.xp.__name__ == "cupy" and hasattr(data, "get") and callable(data.get):  # type: ignore
            data = data.get()  # type: ignore

        return np.array(data)

    @anndata_method(input_attribute="X", output_layer="X_normalized")
    @data_method
    def fit_transform(
        self,
        data: NDArray,
        fraction: float = 1.0,
        clip: bool = True,
    ) -> NDArray:
        """Fit the MinMaxScaler and transform the data to the range.

        Args:
            data (NDArray): The data of at least two dimensions, the last one being the channels.
            fraction (float, optional): The random subsampling fraction to use to fit the estimator.
                Defaults to 1.0.
            clip (bool, optional): Whether to restrict the data strictly to the minimum and maximum of the range
                the estimator was fitted to. This only has consequences if the data to be
                transformed differs from the data used for fitting. Defaults to True.

        Returns:
            NDArray: The normalized data.
        """
        self.fit(
            data,
            fraction=fraction,
        )
        return np.array(self.transform(data, clip=clip))

    def save(
        self,
        save_path: str,
    ) -> None:
        """Save the Clip object to a file.

        Args:
            save_path (str): The path to save the Clip object to.
        """
        if not self.fitted:
            warn("The Normalize object has not been fitted yet.", UserWarning, stacklevel=2)

        config = {
            "init": {
                "min_value": self.min_value,
                "max_value": self.max_value,
            },
            "attributes": {
                "fitted": self.fitted,
                "estimator": self.estimator,
            },
        }

        with open(save_path, "wb") as f:
            dump(config, f)

    @classmethod
    def load(
        cls,
        load_path: str,
    ) -> "Normalize":
        """Load a Normalize object from a file.

        Args:
            load_path (str): The path to load the Normalize object from.

        Returns:
            Normalize: The loaded Normalize object.
        """
        with open(load_path, "rb") as f:
            config = load(f)  # nosec

        normalizer = cls(**config["init"])

        for key, value in config["attributes"].items():
            normalizer.__setattr__(key, value)

        return normalizer
