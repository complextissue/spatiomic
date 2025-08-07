import inspect
import random
from functools import wraps
from typing import Any, Callable, Optional

import numpy as np


def seed_method(method: Callable) -> Callable:
    """Set the `numpy` and `random` random states to a provided seed and reverts it after method execution.

    Args:
        method (Callable): The method to be invoked after setting the random state.

    Returns:
        Callable: The wrapper function that invokes the method.
    """

    @wraps(method)
    def wrapper(
        *args: Any,
        **kwargs: dict,
    ) -> Any:
        """Set the `numpy` and `random` random states to a provided seed and reverts it after method execution.

        Args:
            ref (object): The reference to the object containing the decorated method.

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

        seed: Optional[int] = kwargs["seed"] if "seed" in kwargs and isinstance(kwargs["seed"], int) else None

        random_state_np = np.random.get_state()
        random_state_sys = random.getstate()

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        result = method(self, *args, **kwargs) if self is not None else method(*args, **kwargs)

        # reset the state
        if seed is not None:
            np.random.set_state(random_state_np)
            random.setstate(random_state_sys)

        return result

    wrapper.__doc__ = method.__doc__
    return wrapper
