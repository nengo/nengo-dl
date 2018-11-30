#!/usr/bin/env bash
set -e -v

if [[ $1 == "install" ]]; then
  pip install git+https://github.com/drasmuss/spaun2.0.git
elif [[ $1 == "script" ]]; then
  pytest $TEST_ARGS -v -n 2 --color=yes --pyargs nengo
  pytest $TEST_ARGS -v -n 2 --color=yes --durations 20 nengo_dl
fi
