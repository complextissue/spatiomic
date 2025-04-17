import importlib
from types import ModuleType
from typing import Callable, Optional, Tuple, Union


def import_package(
    package_name: str,
    package_attribute: Optional[str] = None,
    alternative: Optional[ModuleType] = None,
    message: Optional[str] = None,
    raise_error: bool = True,
    return_success: bool = False,
) -> Optional[Union[Union[ModuleType, Callable], Tuple[Optional[Union[ModuleType, Callable]], bool]]]:
    """Import a package, if it fails, return an alternative and state that an alternative is being used.

    If no alternative is provided, raise an error, informing the user of the missing package.

    Args:
        package_name (str): The name of the package to import. May be a submodule, e.g. "numpy.random".
        package_attribute (Optional[str], optional): The attribute of the package to return, e.g., a function or class.
            Defaults to None.
        alternative (ModuleType): The alternative to use if the package cannot be imported. Defaults to None.
        message (Optional[str], optional): The message to print if the package cannot be imported. Defaults to None.
        raise_error (bool, optional): Whether to raise an error if the package cannot be imported and no alternative is
            provided. Defaults to True.
        return_success (bool, optional): Whether to return a boolean indicating whether the import was successful.
            Defaults to False.

    Returns:
        Union[Union[ModuleType, Callable], Tuple[Union[ModuleType, Callable], bool]]: The imported package, or the
            alternative if the package could not be imported. If return_success is True, also returns a boolean
            indicating whether the import was successful.

    Raises:
        ImportError: If the package cannot be imported and no alternative is provided.
    """
    try:
        package = importlib.import_module(package_name)

        if package_attribute is not None:
            package = getattr(package, package_attribute)
    except ImportError as excp:
        if alternative is not None:
            if message is not None:
                print(message)

            return alternative if not return_success else (alternative, False)
        elif raise_error:
            raise ImportError(
                f"Could not import {package_name}. Please install it in order to use this functionality."
            ) from excp
        else:
            return None if not return_success else (None, False)
    else:
        return package if not return_success else (package, True)
