from pathlib import Path
import sys


project = "pySNOM"
copyright = "2026, pySNOM contributors"
author = "pySNOM contributors"
release = "0.3.1"

# Import the package directly from this checkout for autodoc.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_title = "pySNOM documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "github_url": "https://github.com/Quasars/pySNOM",
    "navbar_align": "content",
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "navbar_persistent": ["search-button"],
    "show_nav_level": 2,
    "show_toc_level": 2,
    "navigation_depth": 4,
    "show_version_warning_banner": False,
}
html_sidebars = {
    "**": ["sidebar-nav-bs", "sidebar-ethical-ads"],
}

# Parse NumPy-style sections in Python docstrings.
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
