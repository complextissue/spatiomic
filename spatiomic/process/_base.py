"""The base interfaces that spatiomic.process classes should implement."""

import abc
from typing import Any

from numpy.typing import NDArray


class Processer(metaclass=abc.ABCMeta):
    """An interface that pre- or postprocessing classes should implement."""

    @classmethod
    def __subclasshook__(
        cls,
        subclass: Any,
    ) -> bool:
        """Check that classes that implement the Processer have all required methods.

        Args:
            subclass (Any): The subclass.

        Returns:
            bool: Whether the subclass has all required attributes.
        """
        return (
            hasattr(subclass, "fit_transform")
            and callable(subclass.fit_transform)
            and hasattr(subclass, "transform")
            and callable(subclass.transform)
            and hasattr(subclass, "fit")
            and callable(subclass.fit)
        )

    def fit(
        self,
        data: NDArray,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Fit on the data.

        Args:
            data (NDArray): The data to be fitted.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def transform(
        self,
        data: NDArray,
        *args: Any,
        **kwargs: Any,
    ) -> NDArray:
        """Transform on the data.

        Args:
            data (NDArray): The data to be transformed.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Returns:
            NDArray: The transformed data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit_transform(
        self,
        data: NDArray,
        *args: Any,
        **kwargs: Any,
    ) -> NDArray:
        """Fit and transform on the data.

        Args:
            data (NDArray): The data to be transformed.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Returns:
            NDArray: The transformed data.
        """
        raise NotImplementedError
