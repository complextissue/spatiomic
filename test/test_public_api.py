"""Ensure each submodule's ``__all__`` matches its actual public exports."""

import importlib

import pytest

SUBMODULES = [
    "cluster",
    "data",
    "dimension",
    "neighbor",
    "plot",
    "process",
    "segment",
    "spatial",
    "tool",
]


@pytest.mark.cpu
@pytest.mark.parametrize("submodule", SUBMODULES)
def test_all_names_are_importable(submodule: str) -> None:
    """Every name declared in a submodule's ``__all__`` must be a real attribute of that submodule."""
    module = importlib.import_module(f"spatiomic.{submodule}")
    missing = [name for name in getattr(module, "__all__", []) if not hasattr(module, name)]
    assert not missing, f"spatiomic.{submodule}.__all__ lists names that do not exist: {missing}"
