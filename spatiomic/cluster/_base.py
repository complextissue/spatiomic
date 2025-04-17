"""The base interfaces that spatiomic.cluster classes should implement."""

import abc
from typing import Any

from numpy.typing import NDArray


class ClusterInterface(metaclass=abc.ABCMeta):
    """An interface that clustering classes should implement."""

    @classmethod
    def __subclasshook__(
        cls,
        subclass: Any,
    ) -> bool:
        """Check that classes that implement the ClusterInterface have all required methods.

        Args:
            subclass (Any): The subclass.

        Returns:
            bool: Whether the subclass has all required attributes.
        """
        return hasattr(subclass, "fit_predict") and callable(subclass.fit_predict)

    @abc.abstractmethod
    def fit_predict(
        self,
        data: NDArray,
        **kwargs: dict,
    ) -> NDArray:
        """Fit and predict on the data.

        Args:
            data (NDArray): The data to be clustered.

        Raises:
            NotImplementedError: If the subclass has not implemented this method.

        Returns:
            NDArray: The clustered data.
        """
        raise NotImplementedError
