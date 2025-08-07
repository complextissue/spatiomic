"""Configure the Sphinx documentation builder."""

# -- Path setup --------------------------------------------------------------
import os
import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path
from typing import Any, Dict, List

DOCS_DIR = Path(__file__).parent
PROJECT_ROOT = DOCS_DIR.parent.parent
sys.path.insert(0, os.fspath(PROJECT_ROOT))

# -- Project information -----------------------------------------------------

info = metadata("spatiomic")
project = "spatiomic"
author = "Malte Kuehl"
copyright = f"{datetime.now():%Y}, {author}."  # noqa: A001
version = info["Version"]
release = info["Version"]

# -- General configuration ---------------------------------------------------

templates_path = ["_templates"]
exclude_patterns = [".DS_Store", "_gtm.py"]
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}
master_doc = "index"
default_role = "literal"

extensions = [
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    # "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    "sphinx_design",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_autodoc_typehints",
    "sphinx.ext.inheritance_diagram",
    "sphinxcontrib.mermaid",
    "nbsphinx",
    "myst_parser",
]
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# don't run the notebooks
nbsphinx_execute = "never"

autoapi_type = "python"
autoapi_add_toctree_entry = False
autoapi_ignore: List[str] = ["_*.py"]
autoapi_dirs = [os.path.join(PROJECT_ROOT, "spatiomic")]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]
autoapi_member_order = "alphabetical"
autoapi_python_class_content = "init"  # ensures that the __init__ method is also displayed

autodoc_typehints = "description"

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_rtype = True
napoleon_use_param = True

intersphinx_mapping = dict(  # noqa: C408
    matplotlib=("https://matplotlib.org/stable/", None),
    numpy=("https://numpy.org/doc/stable/", None),
    pandas=("https://pandas.pydata.org/pandas-docs/stable/", None),
    pytest=("https://docs.pytest.org/en/latest/", None),
    python=("https://docs.python.org/3", None),
    scipy=("https://docs.scipy.org/doc/scipy/", None),
    seaborn=("https://seaborn.pydata.org/", None),
    sklearn=("https://scikit-learn.org/stable/", None),
    anndata=("https://anndata.readthedocs.io/en/latest/", None),
)

viewcode_follow_imported_members = True
viewcode_line_numbers = False

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"

html_title = "spatiomic"

html_static_path = ["_static"]
html_logo = "_static/logo.png"

html_css_files = [
    "css/custom.css",
]

html_context: Dict[str, Any] = {
    "display_github": True,
    "github_user": "complextissue",
    "github_repo": "spatiomic",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
    "github_button": True,
    "show_powered_by": False,
}
html_show_sphinx = False

pygments_style = "monokai"

plot_include_source = True
plot_formats = [("svg", 90)]
plot_html_show_formats = False
plot_html_show_source_link = False
plot_working_directory = DOCS_DIR.parent
