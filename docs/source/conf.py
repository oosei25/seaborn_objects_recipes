from __future__ import annotations
import os
import sys
from datetime import datetime

# Path
sys.path.insert(0, os.path.abspath("../../"))

project = "seaborn_objects_recipes"
author = "Ofosu Osei"
copyright = f"{datetime.now():%Y}, {author}"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "numpydoc",
]

# Theme
html_theme = "pydata_sphinx_theme"
html_title = "seaborn_objects_recipes"
html_static_path = []
html_theme_options = {
    "show_prev_next": False,
    "github_url": "https://github.com/oosei25/seaborn_objects_recipes",
    "use_edit_page_button": True,
}
html_context = {
    "github_user": "oosei25",
    "github_repo": "seaborn_objects_recipes",
    "github_version": "main",
    "doc_path": "docs/source",
}

# Autodoc / Autosummary
autosummary_generate = True
autodoc_default_options = {"members": True, "inherited-members": True, "show-inheritance": True}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
}

# sphinx-gallery
from sphinx_gallery.sorting import ExampleTitleSortKey

sphinx_gallery_conf = {
    "examples_dirs": ["examples"],          # where example scripts live
    "gallery_dirs": ["auto_examples"],      # where to build the gallery
    "filename_pattern": r"^plot_",
    "within_subsection_order": ExampleTitleSortKey,
    "remove_config_comments": True,
    "download_all_examples": False,
    # Binder – “launch binder” buttons
    "binder": {
        "org": "oosei25",
        "repo": "seaborn_objects_recipes",
        "branch": "main",
        "binderhub_url": "https://mybinder.org",
        "dependencies": "../../.binder/requirements.txt",
    },
}

# MyST
myst_enable_extensions = ["colon_fence", "deflist", "linkify"]

# Misc
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
