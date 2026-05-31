# Sphinx configuration for RetailGenius Churn Prediction project

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "RetailGenius Churn Prediction"
author = "Group 7 — EPITA International Programs"
release = "1.0.0"
copyright = "2025-2026, Group 7"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

html_theme = "sphinx_rtd_theme"
autodoc_member_order = "bysource"
