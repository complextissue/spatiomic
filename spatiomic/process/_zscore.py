from pickle import dump, load
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from spatiomic._internal._anndata_method import anndata_method
from spatiomic._internal._check_channel_dimension import check_channel_dimension
from spatiomic._internal._data_method import data_method
from spatiomic._internal._import_package import import_package
from spatiomic.data._subsample import Subsample
from spatiomic.process._base import Processer


class ZScore(Processer):
    """A class to z-score data channel-wise."""

    def __init__(
        self,
        use_gpu: bool = True,
    ) -> None:
        """Initialise the ZScore class and set mean and standard deviation to None.

        Args:
            use_gpu (bool, optional): Whether to force numpy usage or use cupy or numpy depending on availability.
                Defaults to True.
        """
        self.std: Optional[NDArray] = None
        self.mean: Optional[NDArray] = None
        self.use_gpu: bool = use_gpu

        self.xp, self.use_gpu = (
            import_package("cupy", alternative=np, return_success=True) if use_gpu else (np, False)  # type: ignore
        )

    @anndata_method(input_attribute="X")
    @data_method
    def fit(
        self,
        data: NDArray,
        fraction: float = 1.0,
    ) -> None:
        """Calculate the channel-wise mean and standard deviation, optionally only on a random subset.

        Args:
            data (NDArray): Data for mean and standard deviation calculations, channels being the last dimension.
            fraction (float, optional): Fraction of data to randomly use for mean and standard deviation calculation.
                Defaults to 1.0.
        """
        assert fraction <= 1.0 and fraction > 0.0
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

        self.std = self.xp.clip(self.xp.nan_to_num(self.xp.std(data, axis=0)), 0, None)
        self.mean = self.xp.clip(self.xp.nan_to_num(self.xp.mean(data, axis=0)), 0, None)

        if self.xp.__name__ == "cupy" and self.std is not None and self.mean is not None:
            if hasattr(self.std, "get"):
                self.std = self.std.get()
            if hasattr(self.mean, "get"):
                self.mean = self.mean.get()

    @anndata_method(input_attribute="X", output_layer="X_zscored")
    @data_method
    def transform(
        self,
        data: NDArray,
    ) -> NDArray:
        """Calculate z-scores based on previously fit mean and standard deviation.

        Args:
            data (NDArray): Data for z-scoring, channels being the last dimension.

        Returns:
            NDArray: The channel-wise z-scored data.
        """
        assert self.std is not None and self.mean is not None, (
            "No standard deviation and mean set, please call the fit method first."
        )

        # flatten array in every dimension but the channel dimension
        data_shape = data.shape
        data = self.xp.array(data)
        data = self.xp.reshape(data, (-1, data.shape[-1]))

        data = (data - self.xp.array(self.mean)) / self.xp.array(self.std)

        if self.xp.__name__ == "cupy":
            return self.xp.reshape(data, data_shape).get()  # type: ignore
        else:
            return self.xp.reshape(data, data_shape)  # type: ignore

    @anndata_method(input_attribute="X", output_layer="X_zscored")
    @data_method
    def fit_transform(
        self,
        data: NDArray,
        fraction: float = 1.0,
    ) -> NDArray:
        """Fit the mean and standard deviation and calculate z-scores on provided data.

        Args:
            data (NDArray): Data for z-scoring, channels being the last dimension.
            fraction (float, optional): Fraction of data to randomly use for mean and standard deviation calculation.
                Defaults to 1.0.

        Returns:
            NDArray: The channel-wise z-scored data.
        """
        self.fit(data, fraction=fraction)
        return np.array(self.transform(data))

    def save(
        self,
        save_path: str,
    ) -> None:
        """Save the ZScore object to a file.

        Args:
            save_path (str): The path to save the ZScore object to.
        """
        assert self.std is not None and self.mean is not None, (
            "No standard deviation and mean set, please call the fit method first."
        )

        config = {
            "init": {
                "use_gpu": self.use_gpu,
            },
            "attributes": {
                "mean": self.mean,
                "std": self.std,
            },
        }

        # eliminate None values
        config = {key: value for key, value in config.items() if value is not None}

        with open(save_path, "wb") as file:
            dump(config, file)

    @classmethod
    def load(
        cls,
        save_path: str,
    ) -> "ZScore":
        """Load a ZScore object from a file.

        Args:
            save_path (str): The path to load the ZScore object from.

        Returns:
            ZScore: The loaded ZScore object.
        """
        with open(save_path, "rb") as file:
            config: dict = load(file)  # nosec

        zscorer = cls(**config["init"])

        for key, value in config["attributes"].items():
            zscorer.__setattr__(key, value)

        return zscorer
