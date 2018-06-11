#!/usr/bin/env bash
set -e -v

if [[ $1 == "install" ]]; then
  pip install -e .[docs]
elif [[ $1 == "before_script" ]]; then
  export DISPLAY=:99.0
  sh -e /etc/init.d/xvfb start
  sleep 3
elif [[ $1 == "script" ]]; then
  python docs/whitepaper/whitepaper2018_code.py > /dev/null
  pytest -v --durations 20 --nbval-lax docs/examples
fi
