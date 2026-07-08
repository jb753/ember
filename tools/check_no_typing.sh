#!/bin/bash
# Check that no type annotations are used in the source code.
# Type information belongs in docstrings, not in the code itself.

set -e

ERRORS=$(grep -rn --include="*.py" \
    -e "^\s*import typing" \
    -e "^\s*from typing import" \
    -e "^\s*import numpy\.typing" \
    -e "^\s*from numpy\.typing import" \
    -e "^\s*def .* -> " \
    -e "^\s*def .*[a-zA-Z_]: [a-zA-Z]" \
    src/ember/ 2>/dev/null \
    | grep -v "#.*[a-zA-Z_]: [a-zA-Z]" \
    || true)

if [ -n "$ERRORS" ]; then
    echo "Type annotations found in source code:"
    echo "$ERRORS"
    echo ""
    echo "Use docstrings for type information instead."
    exit 1
fi

echo "No type annotations found."
