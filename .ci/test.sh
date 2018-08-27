#!/usr/bin/env bash
set -e -v

if [[ $1 == "script" ]]; then
  pytest -v -n 2 --color=yes --pyargs nengo
  pytest -v -n 2 --color=yes --durations 20 $TEST_ARGS nengo_dl
fi
