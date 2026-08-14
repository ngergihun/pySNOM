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
    'numpydoc',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
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
    "show_nav_level": 0,
    "collapse_navigation": True,
    "show_toc_level": 2,
    "navigation_depth": 4,
}
html_sidebars = {
    "**": ["sidebar-nav-bs", "sidebar-ethical-ads"],
}

autosummary_generate = True
autosummary_imported_members = False
numpydoc_class_members_toctree = False

add_function_parentheses = True
numpydoc_use_plots = True

plot_html_show_formats = False
plot_html_show_source_link = False

