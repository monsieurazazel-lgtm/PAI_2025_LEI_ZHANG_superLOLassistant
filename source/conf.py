# Configuration file for the Sphinx documentation builder.
import os
import sys

# -- Chemin du code source --------------------------------------------------
# On remonte de deux crans pour atteindre la racine du projet depuis 'source/'
# Cela permet à Sphinx d'importer tes fichiers .py
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..')) 

# -- Project information -----------------------------------------------------
project = 'SuperLOLassistant'   
copyright = '2026, ShutianLEI/LixiangZHANG'
author = 'ShutianLEI/LixiangZHANG'

# -- General configuration ---------------------------------------------------

# Ajout des extensions cruciales pour Python
extensions = [
    'sphinx.ext.autodoc',      # Extrait les docstrings du code 
    'sphinx.ext.napoleon',     # Supporte le format Google/Numpy [cite: 330, 336]
    'sphinx.ext.viewcode',     # Ajoute des liens vers le code source
    'sphinx.ext.autosummary',  # Génère des tableaux récapitulatifs [cite: 336, 339]
]

# Générer automatiquement les pages de résumé
autosummary_generate = True 

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'fr'

# -- Options for HTML output -------------------------------------------------

# Tu peux garder 'alabaster' ou tester 'sphinx_rtd_theme' si tu l'installes
html_theme = 'alabaster' 
html_static_path = ['_static']

# Configuration Napoleon pour coller à tes exercices
napoleon_google_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True