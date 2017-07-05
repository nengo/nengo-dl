import os
import sys

# we prepend the current directory to the path so that when
# sphinxcontrib-versioning copies branches to different subdirectories we
# import the copied version of nengo_dl (the version associated with the docs
# being built)
sys.path = [os.path.join(os.path.dirname(__file__), "..")] + sys.path

import nengo_dl
import sphinx_rtd_theme

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'numpydoc',
    'nengo.utils.docutils',
    'nbsphinx',
    'IPython.sphinxext.ipython_console_highlighting',
]

# -- sphinx.ext.autodoc
autoclass_content = 'both'  # class and __init__ docstrings are concatenated
autodoc_default_flags = ['members']
autodoc_member_order = 'bysource'  # default is alphabetical

# -- sphinx.ext.intersphinx
intersphinx_mapping = {
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'nengo': ('http://pythonhosted.org/nengo', None),
}

# -- sphinx.ext.todo
todo_include_todos = True
# -- numpydoc config
numpydoc_show_class_members = False

# -- sphinx-versioning
scv_whitelist_branches = ('master',)
scv_show_banner = True
scv_banner_recent_tag = True

# -- sphinx
exclude_patterns = ['_build', '**.ipynb_checkpoints']
source_suffix = '.rst'
source_encoding = 'utf-8'
master_doc = 'index'

# Need to include https Mathjax path for sphinx < v1.3
mathjax_path = ("https://cdn.mathjax.org/mathjax/latest/MathJax.js"
                "?config=TeX-AMS-MML_HTMLorMML")

project = u'NengoDL'
authors = u'Applied Brain Research'
copyright = nengo_dl.__copyright__
# version = '.'.join(nengo_dl.__version__.split('.')[:2])  # Short X.Y version
release = nengo_dl.__version__  # Full version, with tags
pygments_style = 'default'

# -- Options for HTML output --------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_title = "NengoDL documentation"
html_static_path = ['_static']
html_context = {
    'css_files': [os.path.join('_static', 'custom.css')],
}
html_use_smartypants = True
htmlhelp_basename = 'Nengodoc'
html_last_updated_fmt = ''  # Suppress 'Last updated on:' timestamp
html_show_sphinx = False

# -- Options for LaTeX output -------------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '11pt',
    # 'preamble': '',
}

latex_documents = [
    # (source start file, target, title, author, documentclass [howto/manual])
    ('index', 'nengo_dl.tex', html_title, authors, 'manual'),
]

# -- Options for manual page output -------------------------------------------

man_pages = [
    # (source start file, name, description, authors, manual section).
    ('index', 'nengo_dl', html_title, [authors], 1)
]

# -- Options for Texinfo output -----------------------------------------------

texinfo_documents = [
    # (source start file, target, title, author, dir menu entry,
    #  description, category)
    ('index', 'nengo_dl', html_title, authors, 'NengoDL',
     'Nengo with deep learning integration', 'Miscellaneous'),
]
