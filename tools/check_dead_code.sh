#!/bin/bash
# Check for untested code, unused vars etc using vulture

ERRORS=$(uv run -q vulture src tests scripts --min-confidence 60 --ignore-names "Test*,pytestmark" 2>&1  || true)

if [ -n "$ERRORS" ]; then
    echo "Dead code found:"
    echo "$ERRORS"
    echo "Instructions:"
    echo "1) Check the git staging area, there may be tests that are not added to the commit yet that the check does not pick up."
    echo "2) Delete unused variables right away, use '_' for superflous iterators or unpacked parts of tuples. Do not use descriptive names with a leading _ if the variable is not used."
    echo "3) Added methods, functions etc must come with tests. If you have written new code, also write basic tests for it."
    echo "4) DO NOT create a whitelist without user confirmation."
    exit 1
fi
