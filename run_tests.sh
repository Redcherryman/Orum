#!/bin/sh
# Run the Orum v0.001 test suite (stdlib unittest; also pytest-compatible).
cd "$(dirname "$0")" || exit 1
exec python3 -m unittest discover -s tests
