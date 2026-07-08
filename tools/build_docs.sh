#!/usr/bin/env bash
# Build the Sphinx documentation.
#
# The build uses -W (via the `docs` Make target), so any broken cross-reference
# or warning fails the commit.
set -euo pipefail

make docs
