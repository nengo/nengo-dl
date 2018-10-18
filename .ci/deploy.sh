#!/usr/bin/env bash
set -e -v

if [[ $1 == "install" ]]; then
  pip install check-manifest collective.checkdocs
elif [[ $1 == "script" ]]; then
  check-manifest
  python setup.py checkdocs
  if [[ "$TRAVIS_TAG" == "" ]]; then
    TAG="v$(cut -d'-' -f3 <<<$TRAVIS_BRANCH)"
  else
    TAG=$TRAVIS_TAG
  fi
  python -c "from nengo_dl import version; assert not version.dev, 'this is a dev version'"
  python -c "from nengo_dl import version; assert 'v' + version.version == '$TAG', 'version does not match tag'"
  python -c "from nengo_dl import version; assert any(line.startswith(version.version) and 'unreleased' not in line for line in open('CHANGES.rst').readlines()), 'changelog not updated'"
fi
