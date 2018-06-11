#!/usr/bin/env bash
set -e -v

if [[ $1 == "script" ]]; then
  coverage run -m pytest --pyargs nengo
  coverage run -a -m pytest nengo_dl
  coverage report -m
elif [[ $1 == "after_success" ]]; then
  eval "bash <(curl -s https://codecov.io/bash)"
fi
