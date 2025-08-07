import inspect
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import ArrayLike, NDArray

from ._import_package import import_package


def _get_data_from_anndata(
    adata: AnnData,
    input_attribute_name: Optional[str] = None,
    input_layer_name: Optional[str] = None,
    input_obs_name: Optional[str] = None,
    input_obsm_name: Optional[str] = None,
    input_var_name: Optional[str] = None,
    input_varm_name: Optional[str] = None,
    input_uns_name: Optional[str] = None,
) -> NDArray:
    """Extract NDArray data from AnnData object based on input parameters."""
    if input_attribute_name is not None:
        return_value = getattr(adata, input_attribute_name)
    elif input_layer_name is not None:
        return_value = adata.layers[input_layer_name]
    elif input_obs_name is not None:
        obs_data = adata.obs[input_obs_name]
        return_value = obs_data.to_numpy() if isinstance(obs_data, (pd.DataFrame, pd.Series)) else obs_data
    elif input_obsm_name is not None:
        return_value = adata.obsm[input_obsm_name]
    elif input_var_name is not None:
        var_data = adata.var[input_var_name]
        return_value = var_data.to_numpy() if isinstance(var_data, (pd.DataFrame, pd.Series)) else var_data
    elif input_varm_name is not None:
        return_value = adata.varm[input_varm_name]
    elif input_uns_name is not None:
        return_value = adata.uns[input_uns_name]
    else:
        return_value = adata.X.toarray()

    if not isinstance(return_value, ArrayLike) and not isinstance(return_value, np.ndarray):
        raise TypeError("Returned data must be a numpy array.")
    elif isinstance(return_value, ArrayLike):
        return_value = np.array(return_value)

    return return_value


def _store_data_in_anndata(
    adata: AnnData,
    result: NDArray,
    output_layer_name: Optional[str] = None,
    output_obsm_name: Optional[str] = None,
    output_obs_name: Optional[str] = None,
    output_varm_name: Optional[str] = None,
    output_var_name: Optional[str] = None,
    output_uns_name: Optional[str] = None,
) -> None:
    """Store result NDArray data into AnnData object based on output parameters."""
    if output_layer_name is not None:
        adata.layers[output_layer_name] = result
    elif output_obsm_name is not None:
        adata.obsm[output_obsm_name] = result
    elif output_obs_name is not None:
        adata.obs[output_obs_name] = result
    elif output_varm_name is not None:
        adata.varm[output_varm_name] = result
    elif output_var_name is not None:
        adata.var[output_var_name] = result
    elif output_uns_name is not None:
        adata.uns[output_uns_name] = result


def anndata_method(
    pass_data: bool = True,
    input_attribute: Optional[str] = None,
    input_layer: Optional[str] = None,
    input_obs: Optional[str] = None,
    input_obsm: Optional[str] = None,
    input_var: Optional[str] = None,
    input_varm: Optional[str] = None,
    input_uns: Optional[str] = None,
    output_layer: Optional[str] = None,
    output_obsm: Optional[str] = None,
    output_obs: Optional[str] = None,
    output_varm: Optional[str] = None,
    output_var: Optional[str] = None,
    output_uns: Optional[str] = None,
    result_index: Optional[int] = None,
) -> Callable:
    """Decorator to handle AnnData input and output for methods.

    This decorator simplifies the process of writing methods that can work with both AnnData objects
    and raw NDArrays/DataFrames. It automatically extracts data from AnnData based on specified
    input parameters and stores the result back into AnnData based on output parameters.

    .. code-block:: python
        @anndata_method(pass_data=True, input_obs="obs_column", output_uns="my_method_result")
        def my_method(data: NDArray) -> NDArray:
            return data


        adata = AnnData(X=np.random.rand(10, 10))
        adata.obs["obs_column"] = np.random.rand(10)

        # The result will be stored in adata.uns["my_method_result"] but the method will still return the result
        result = my_method(adata)

    .. note::
        User provided input/output parameter names will override the default parameter names. If the user provides
        multiple input/output parameter names, the order of precedence is as follows:
        input_attribute > input_layer > input_obs > input_obsm > input_var > input_varm > input_uns
        output_layer > output_obs > output_obsm > output_var > output_varm > output_uns

    .. note::
        If the decorated method returns a tuple, the index of the element to store in AnnData can be specified
        with the `result_index` parameter.

    Args:
        pass_data (bool, optional): Whether to pass the extracted data to the decorated method. Defaults to True.
        input_attribute (Optional[str]): Attribute name in AnnData to retrieve data from (e.g., 'raw').
        input_layer (Optional[str]): Layer name in AnnData to retrieve data from.
        input_obs (Optional[str]): .obs column name in AnnData to retrieve data from.
        input_obsm (Optional[str]): .obsm name in AnnData to retrieve data from.
        input_var (Optional[str]): .var column name in AnnData to retrieve data from.
        input_varm (Optional[str]): .varm name in AnnData to retrieve data from.
        input_uns (Optional[str]): .uns key in AnnData to retrieve data from.
        output_layer (Optional[str]): Layer name in AnnData to store the result.
        output_obsm (Optional[str]): .obsm name in AnnData to store the result.
        output_obs (Optional[str]): .obs column name in AnnData to store the result.
        output_varm (Optional[str]): .varm name in AnnData to store the result.
        output_var (Optional[str]): .var column name in AnnData to store the result.
        output_uns (Optional[str]): .uns key in AnnData to store the result.
        result_index (Optional[int]): If the decorated method returns a tuple, this index specifies which element of the
            tuple to store in AnnData. Defaults to None (store the first element).

    Returns:
        Callable: The decorated method.

    Raises:
        ValueError: If 'data' argument is missing when pass_data=True and no AnnData object found in arguments.
        TypeError: If 'data' is not AnnData, DataFrame, or NDArray when pass_data=True.
    """

    def decorator(method: Callable) -> Callable:
        @wraps(method)
        def wrapper(
            *args: Tuple[Any, ...],
            **kwargs: Dict[str, Any],
        ) -> Any:
            method_signature = inspect.signature(method)
            parameters = method_signature.parameters
            has_self = "self" in parameters

            self_obj = None
            processed_args: list[Any] = list(args)  # Convert to list for easier manipulation

            if has_self:
                self_index = list(parameters.keys()).index("self")
                if len(processed_args) > self_index:
                    self_obj = processed_args.pop(self_index)

            if TYPE_CHECKING:
                xp = np
            else:
                xp: "np" = import_package("cupy", alternative=np)

            data_arg = kwargs.pop("data", None)
            if (
                data_arg is None
                and processed_args
                and isinstance(processed_args[0], (AnnData, pd.DataFrame, np.ndarray, xp.ndarray))
            ):
                data_arg = processed_args.pop(0)

            array_data = None
            adata_obj = None

            if isinstance(data_arg, AnnData):
                adata_obj = data_arg

                input_params: Dict[str, Optional[str]] = {
                    "input_attribute_name": kwargs.pop("input_attribute", input_attribute),  # type: ignore
                    "input_layer_name": kwargs.pop("input_layer", input_layer),  # type: ignore
                    "input_obs_name": kwargs.pop("input_obs", input_obs),  # type: ignore
                    "input_obsm_name": kwargs.pop("input_obsm", input_obsm),  # type: ignore
                    "input_var_name": kwargs.pop("input_var", input_var),  # type: ignore
                    "input_varm_name": kwargs.pop("input_varm", input_varm),  # type: ignore
                    "input_uns_name": kwargs.pop("input_uns", input_uns),  # type: ignore
                }
                array_data = _get_data_from_anndata(adata_obj, **input_params)

                output_params: Dict[str, Optional[str]] = {
                    "output_layer_name": kwargs.pop("output_layer", output_layer),  # type: ignore
                    "output_obs_name": kwargs.pop("output_obs", output_obs),  # type: ignore
                    "output_obsm_name": kwargs.pop("output_obsm", output_obsm),  # type: ignore
                    "output_var_name": kwargs.pop("output_var", output_var),  # type: ignore
                    "output_varm_name": kwargs.pop("output_varm", output_varm),  # type: ignore
                    "output_uns_name": kwargs.pop("output_uns", output_uns),  # type: ignore
                }
                result_index_parsed = kwargs.pop("result_index", result_index)

            elif data_arg is not None and isinstance(data_arg, (pd.DataFrame, np.ndarray, xp.ndarray)):
                array_data = data_arg
            elif pass_data:
                raise ValueError("Data argument missing.")
            elif data_arg is not None:  # this case should not raise TypeError if pass_data=False
                raise TypeError("Data must be an AnnData object, DataFrame or NDArray if pass_data=True.")

            method_args = [self_obj] if self_obj else []
            if pass_data and array_data is not None:  # Only pass data if pass_data is True and we have data to pass
                method_args.append(array_data)

            result = method(*method_args, *processed_args, **kwargs)

            if adata_obj is not None:
                ad_result_to_store = result
                if isinstance(result, tuple):
                    result_index_parsed = 0 if result_index_parsed is None else result_index_parsed
                    if isinstance(result_index_parsed, int) and 0 <= result_index_parsed < len(result):
                        ad_result_to_store = result[result_index_parsed]
                    else:
                        ad_result_to_store = result[0]  # default to first element if index invalid

                if xp.__name__ == "cupy" and isinstance(ad_result_to_store, xp.ndarray):
                    ad_result_to_store = ad_result_to_store.get()  # type: ignore

                if isinstance(ad_result_to_store, np.ndarray):  # only store numpy array-like results in AnnData
                    _store_data_in_anndata(
                        adata_obj,
                        ad_result_to_store,
                        **output_params,
                    )

            return result

        wrapper.__doc__ = method.__doc__

        additional_doc = """

        This method can be used with both AnnData objects and raw NDArrays/DataFrames. If an AnnData object is provided,
        the data will be extracted based on the following parameters:
            - input_attribute: Name of the attribute in AnnData to extract data from.
            - input_layer: Name of the layer in AnnData to extract data from.
            - input_obs: Column in AnnData.obs to extract data from.
            - input_obsm: Key in AnnData.obsm to extract data from.
            - input_var: Column in AnnData.var to extract data from.
            - input_varm: Key in AnnData.varm to extract data from.
            - input_uns: Key in AnnData.uns to extract data from.

        The result will be stored back into the AnnData object based on the following parameters:
            - output_layer: Name of the layer in AnnData to store the result.
            - output_obs: Column in AnnData.obs to store the result.
            - output_obsm: Key in AnnData.obsm to store the result.
            - output_var: Column in AnnData.var to store the result.
            - output_varm: Key in AnnData.varm to store the result.
            - output_uns: Key in AnnData.uns to store the result.
        """
        if wrapper.__doc__:
            wrapper.__doc__ += additional_doc
        else:
            wrapper.__doc__ = additional_doc

        return wrapper

    return decorator
