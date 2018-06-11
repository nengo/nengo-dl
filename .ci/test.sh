#!/usr/bin/env bash
set -e -v

if [[ $1 == "script" ]]; then
  pytest -n 2 --pyargs nengo
  pytest -n 2 --durations 20 nengo_dl
fi
