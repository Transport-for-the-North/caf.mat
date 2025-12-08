# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Built-Ins
import os
import pathlib
import re
import sys

dir_path = pathlib.Path(__file__).parents[2]
source = dir_path / "src"
sys.path.insert(0, str(source.absolute()))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "caf.mat"
copyright = "2024, Transport for the North"
author = "Transport for the North"

# Local Imports
import caf.mat

version = str(caf.mat.__version__)
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates", "_templates/autosummary"]
exclude_patterns = []


numpydoc_show_class_members = False

# Change autodoc settings
autodoc_member_order = "groupwise"
autoclass_content = "both"
autodoc_default_options = {
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": False,
    "private-members": False,
    "exclude-members": "__module__, __weakref__, __dict__",
}
autodoc_typehints = "description"

# Auto summary options
autosummary_generate = True

modindex_common_prefix = ["caf.", "caf.mat."]

# Sphinx gallery settings
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",  # path to your example scripts
    "gallery_dirs": "examples",  # path to where to save gallery generated output
    # Regex pattern of filenames to be ran so the output can be included
    "filename_pattern": rf"{re.escape(os.sep)}run_.*\.py",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
