import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'phlearn'
copyright = '2022, Camilla Sterud, Sølve Eidnes, Eivind Bøhn, Signe Riemer-Sørensen, Alexander Stasik'
author = 'Camilla Sterud, Sølve Eidnes, Eivind Bøhn, Signe Riemer-Sørensen, Alexander Stasik'
release = '0.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'numpydoc'
]

numpydoc_class_members_toctree = False
numpydoc_show_inherited_class_members = False
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'special-members': '__init__',
    'imported-members': True,
    'show-inheritance': True,
}