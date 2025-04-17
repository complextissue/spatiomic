import inspect
import warnings
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

from spatiomic._internal._import_package import import_package


def data_method(method: Callable) -> Callable:
    """Automatically check and get the values from the data parameter for the decorated function.

    Args:
        method (Callable): The decorated method that takes the processed data.

    Returns:
        Callable: A wrapper function that invokes the passed method with the processed data.
    """

    @wraps(method)
    def wrapper(
        *args: Any,
        **kwargs: dict,
    ) -> Any:
        """Get float NDArray data from passed data and invoke the decorated method with the processed data.

        Args:
            self (object): The reference to the object containing the decorated method.
            data (Union[NDArray, pd.DataFrame, MultiplexedImage, MultiplexedImageCollection]): The data source.

        Returns:
            Any: The result of the invoked method.
        """
        # Check if 'self' is present as an argument
        method_signature = inspect.signature(method)
        parameters = method_signature.parameters

        has_self = "self" in parameters

        if has_self:
            self_index = list(parameters.keys()).index("self")
            self = args[self_index]
            args = args[:self_index] + args[self_index + 1 :]
        else:
            self = None

        # Check if 'data' is present as a keyword argument, otherwise check if it is the first argument
        data = kwargs.pop("data", None)
        if data is None and len(args) > 0:
            data = args[0]
            args = args[1:]

        if TYPE_CHECKING:
            xp = np
        else:
            xp: "np" = import_package("cupy", alternative=np)

        assert data is not None and (isinstance(data, (pd.DataFrame, np.ndarray, xp.ndarray))), (
            "DataFrame or numpy array required."
        )

        # convert dataframes to numpy arrays
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        # check that data is float
        if data.dtype not in [  # type: ignore
            np.float32,
            np.float64,
            xp.float32,
            xp.float64,
        ]:
            warnings.warn(
                "Data does not appear to be a numpy/cupy float, casting to numpy float32.",
                UserWarning,
                stacklevel=2,
            )
            data = np.float32(data if isinstance(data, np.ndarray) else data.get())  # type: ignore

        return method(self, data, *args, **kwargs) if self is not None else method(data, *args, **kwargs)

    wrapper.__doc__ = method.__doc__
    return wrapper
