#!/bin/bash

# build wheels
for dep_target in 13.0 14.0; do
  export MACOSX_DEPLOYMENT_TARGET=${dep_target}
  pypy3.10 -m pip wheel --no-deps --wheel-dir dist/ .
  python3.10 -m pip wheel --no-deps --wheel-dir dist/ .
  python3.11 -m pip wheel --no-deps --wheel-dir dist/ .
  python3.12 -m pip wheel --no-deps --wheel-dir dist/ .
done

# audit wheels
delocate-wheel dist/*.whl
