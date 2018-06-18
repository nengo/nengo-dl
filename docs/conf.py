import os

import sphinx_rtd_theme

import nengo_dl

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'numpydoc',
    'nbsphinx',
    'IPython.sphinxext.ipython_console_highlighting',
]

templates_path = ["_templates"]

# -- sphinx.ext.autodoc
autoclass_content = 'both'  # class and __init__ docstrings are concatenated
autodoc_default_flags = ['members']
autodoc_member_order = 'bysource'  # default is alphabetical

# -- sphinx.ext.intersphinx
intersphinx_mapping = {
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'nengo': ('https://www.nengo.ai/nengo/', None),
}

# -- numpydoc config
numpydoc_show_class_members = False

# -- nbsphinx
nbsphinx_timeout = 300

# -- sphinx
exclude_patterns = ['_build', '**.ipynb_checkpoints']
source_suffix = '.rst'
source_encoding = 'utf-8'
master_doc = 'index'
suppress_warnings = ['image.nonlocal_uri']
linkcheck_ignore = [r'http://localhost:\d+']
linkcheck_anchors = True
nitpicky = True

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
htmlhelp_basename = 'NengoDLdoc'
html_last_updated_fmt = ''  # default output format
html_show_sphinx = False
html_favicon = os.path.join("_static", "favicon.ico")
html_logo = os.path.join("_static", "logo.png")
