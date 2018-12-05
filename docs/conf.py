import errno
import os

import nengo_dl

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "nengo_sphinx_theme",
    "numpydoc",
    "nbsphinx",
    "sphinx_click.ext",
]

templates_path = ["_templates"]

# -- sphinx.ext.autodoc
autoclass_content = "both"  # class and __init__ docstrings are concatenated
autodoc_default_options = {"members": None}
autodoc_member_order = "bysource"  # default is alphabetical

# -- sphinx.ext.intersphinx
intersphinx_mapping = {
    "numpy": ("https://docs.scipy.org/doc/numpy", None),
    "nengo": ("https://www.nengo.ai/nengo/", None),
}

# -- numpydoc config
numpydoc_show_class_members = False

# -- nbsphinx
nbsphinx_timeout = 300

# -- sphinx
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
source_suffix = ".rst"
source_encoding = "utf-8"
master_doc = "index"
suppress_warnings = ["image.nonlocal_uri"]
linkcheck_ignore = [r"http://localhost:\d+"]
linkcheck_anchors = True
nitpicky = True
default_role = "py:obj"

project = u"NengoDL"
authors = u"Applied Brain Research"
copyright = nengo_dl.__copyright__
# version = ".".join(nengo_dl.__version__.split(".")[:2])  # Short X.Y version
release = nengo_dl.__version__  # Full version, with tags
pygments_style = "friendly"

# -- Options for HTML output --------------------------------------------------

html_theme = "nengo_sphinx_theme"
html_title = "NengoDL documentation"
html_static_path = ["_static"]
htmlhelp_basename = "NengoDLdoc"
html_last_updated_fmt = ""  # default output format
html_show_sphinx = False
html_favicon = os.path.join("_static", "favicon.ico")
html_theme_options = {
    "sidebar_toc_depth": 4,
    "sidebar_logo_width": 200,
    "nengo_logo": "nengo-dl-full-light.svg",
}

# create redirect pages (from_page, to_page)
# TODO: we can remove these redirects after a few releases
redirects = [
    ("frontend.html", "user-guide.html"),
    ("backend.html", "reference.html#developers"),
    ("builder.html", "reference.html#builder"),
    ("extra_objects.html", "reference.html#neuron-types"),
    ("graph_optimizer.html", "reference.html#graph-optimization"),
    ("operators.html", "reference.html#operator-builders"),
    ("learning_rules.html", "reference.html#operator-builders"),
    ("neurons.html", "reference.html#operator-builders"),
    ("op_builders.html", "reference.html#operator-builders"),
    ("processes.html", "reference.html#operator-builders"),
    ("tensor_node_builders.html", "reference.html#operator-builders"),
    ("signals.html", "reference.html#signals"),
    ("tensor_graph.html", "reference.html#graph-construction"),
    ("utils.html", "reference.html#utilities"),
    ("tensor_node.html", "tensor-node.html"),
    ("examples/nef_init.html", "examples/nef-init.html"),
    ("examples/pretrained_model.html", "examples/pretrained-model.html"),
    ("examples/spa_memory.html", "examples/spa-memory.html"),
    ("examples/spa_retrieval.html", "examples/spa-retrieval.html"),
    ("examples/spiking_mnist.html", "examples/spiking-mnist.html"),
]


def setup(app):
    def mkdir_p(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise

    def redirect_pages(app, docname):
        redirects = app.config.redirects
        if app.builder.name == "html":
            for src, dst in redirects:
                srcfile = os.path.join(app.outdir, src)
                dsturl = "/".join(
                    [".." for _ in range(src.count("/"))] + [dst])
                mkdir_p(os.path.dirname(srcfile))
                with open(srcfile, "w") as fp:
                    fp.write("\n".join([
                        '<!DOCTYPE html>',
                        '<html>',
                        ' <head><title>This page has moved</title></head>',
                        ' <body>',
                        '  <script type="text/javascript">',
                        '   window.location.replace("{0}");',
                        '  </script>',
                        '  <noscript>',
                        '   <meta http-equiv="refresh" content="0; url={0}">',
                        '  </noscript>',
                        ' </body>',
                        '</html>',
                    ]).format(dsturl))

    app.add_config_value("redirects", [], "")
    app.connect("build-finished", redirect_pages)
