# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'Predspot'
copyright = '2024, Adelson Araujo'
author = 'Adelson Araujo'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.todo',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_baseurl = 'https://adaj.github.io/predspot/'

html_theme_options = {
    'description': 'Crime hotspot prediction using machine learning',
    'github_user': 'adaj',
    'github_repo': 'predspot',
    'github_button': True,
    'github_type': 'star',
    'github_count': True,
    'fixed_sidebar': True,
    'page_width': '1000px',
    'sidebar_width': '250px',
    'show_powered_by': True,
    'show_relbars': True,
    'sidebar_collapse': True,
    'sidebar_includehidden': True,
    'extra_nav_links': {
        'Project Github': 'https://github.com/adaj/predspot',
        'Documentation': 'https://adaj.github.io/predspot/',
    }
} 