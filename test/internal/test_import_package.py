"""Test import_package."""

from spatiomic._internal._import_package import import_package


def test_import_package() -> None:
    """Test import_package."""
    # test importing an existing package
    package = import_package("numpy")
    # check that the package has the __name__ attribute and that it is "numpy"
    assert hasattr(package, "__name__") and package.__name__ == "numpy"  # type: ignore

    # test importing an existing subpackage
    package = import_package("numpy.random")
    # check that the subpackage has the __name__ attribute and that it is "numpy"
    assert hasattr(package, "__name__") and package.__name__ == "numpy.random"  # type: ignore

    # test importing an existing package with an attribute
    package = import_package("numpy.random", package_attribute="randint")
    # check that the package is a function
    assert callable(package)

    # test importing a non-existent package with no alternative
    try:
        import_package("nonexistent_package")
    except ImportError as e:
        assert str(e) == "Could not import nonexistent_package. Please install it in order to use this functionality."

    # test importing a non-existent package with an alternative
    alternative = object()
    result = import_package("nonexistent_package", alternative=alternative)
    assert result is alternative
