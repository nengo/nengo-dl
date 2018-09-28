#!/usr/bin/env bash
set -e -v

if [[ $1 == "install" ]]; then
  pip install -e .[docs]
  conda install -q pandoc
elif [[ $1 == "script" ]]; then
  pylint nengo_dl --rcfile=setup.cfg
  codespell -q 3
  sphinx-build -b linkcheck docs docs/_build -W -D nbsphinx_execute=never
fi
