#!/bin/bash

# build wheels
python3.10 -m pip wheel --no-deps --wheel-dir dist/ .
python3.11 -m pip wheel --no-deps --wheel-dir dist/ .

# audit wheels
delocate-wheel dist/*.whl
