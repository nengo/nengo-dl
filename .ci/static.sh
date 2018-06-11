#!/usr/bin/env bash
set -e -v

if [[ $1 == "script" ]]; then
  codespell -q 3
  pylint nengo_dl --rcfile=setup.cfg
fi
