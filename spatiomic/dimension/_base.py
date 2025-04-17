"""The base interfaces that spatiomic.dimension classes should implement."""

import abc
from typing import Any

from numpy.typing import NDArray


class DimensionReducer(metaclass=abc.ABCMeta):
    """An interface that dimensionality reduction classes should implement."""

    @classmethod
    def __subclasshook__(
        cls,
        subclass: Any,
    ) -> bool:
        """Check that classes that implement the DimensionReducer have all required methods.

        Args:
            subclass (Any): The subclass.

        Returns:
            bool: Whether the subclass has all required attributes.
        """
        return hasattr(subclass, "fit_transform") and callable(subclass.fit_transform)

    @abc.abstractmethod
    def fit_transform(
        self,
        data: NDArray,
        flatten: bool,
    ) -> NDArray:
        """Fit and transform on the data.

        Args:
            data (NDArray): The data to be transformed.
            flatten (bool): Whether to flatten the dimensionality reduced data in every but the channel dimension.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Returns:
            NDArray: The transformed data.
        """
        raise NotImplementedError


class LoadableDimensionReducer(metaclass=abc.ABCMeta):
    """An interface that dimensionality save-able reduction classes should implement."""

    @classmethod
    def __subclasshook__(
        cls,
        subclass: Any,
    ) -> bool:
        """Check that classes that implement the LoadableDimensionReducer have all required methods.

        Args:
            subclass (Any): The subclass.

        Returns:
            bool: Whether the subclass has all required attributes.
        """
        return (
            hasattr(subclass, "fit")
            and callable(subclass.fit)
            and hasattr(subclass, "save")
            and callable(subclass.save)
            and hasattr(subclass, "load")
            and callable(subclass.load)
        )

    @abc.abstractmethod
    def save(
        self,
        save_path: str,
    ) -> None:
        """Save the state of the class.

        Args:
            save_path (str): The path where to load the estimator or its configuration from.

        Raises:
            NotImplementedError: If the method has not been implemented by the subclass.
        """
        raise NotImplementedError("Not implemented.")

    @abc.abstractmethod
    def load(
        self,
        load_path: str,
    ) -> None:
        """Load a state of the class.

        Args:
            save_path (str): The path where to load the estimator or its configuration from.

        Raises:
            NotImplementedError: If the method has not been implemented by the subclass.
        """
        raise NotImplementedError("Not implemented.")
