#!/usr/bin/env bash
set -e -v

if [[ $1 == "script" ]]; then
  pytest $TEST_ARGS -v -n 2 --color=yes --pyargs nengo
  pytest $TEST_ARGS -v -n 2 --color=yes --durations 20 nengo_dl
fi
