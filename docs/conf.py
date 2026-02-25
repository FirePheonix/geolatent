import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "GeoLatent"
author = "GeoLatent Contributors"
copyright = "2026, GeoLatent Contributors"
release = "0.1.0"

extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
]

# Auto-generate API docs from source
autoapi_dirs = ["../geolatent"]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_keep_files = False
autoapi_add_toctree_entry = True
autoapi_python_class_content = "both"

# Napoleon — numpy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# Cross-package links
intersphinx_mapping = {
    "python":   ("https://docs.python.org/3", None),
    "numpy":    ("https://numpy.org/doc/stable", None),
    "sklearn":  ("https://scikit-learn.org/stable", None),
    "plotly":   ("https://plotly.com/python-api-reference", None),
}

# MyST
myst_enable_extensions = ["colon_fence", "deflist"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

html_theme = "pydata_sphinx_theme"
html_title = "GeoLatent"

html_theme_options = {
    "logo": {
        "text": "GeoLatent",
    },
    "github_url": "https://github.com/FirePheonix/geolatent",
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/FirePheonix/geolatent",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/geolatent/",
            "icon": "fa-brands fa-python",
        },
    ],
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
