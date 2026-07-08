#!/bin/bash
# Detect stale files under build/lib*/ember with no corresponding source file.
#
# setuptools' build/ directory is an incremental cache: it is never pruned,
# so a module deleted from src/ember can silently persist there and leak
# back into a wheel built later without a clean first (this happened once:
# a purged src/ember/cases package resurfaced in a locally built wheel).
# This only fires when build/ already exists from a prior local
# 'uv build --wheel' -- it never triggers a build itself, so it stays cheap
# on every commit.

STALE=()
for build_dir in build/lib*/ember; do
    [ -d "$build_dir" ] || continue
    while IFS= read -r -d '' f; do
        rel="${f#"$build_dir"/}"
        if [ ! -f "src/ember/$rel" ]; then
            STALE+=("$build_dir/$rel")
        fi
    done < <(find "$build_dir" -name '*.py' -print0)
done

if [ "${#STALE[@]}" -gt 0 ]; then
    echo "Stale build artifacts found (no matching file in src/ember):"
    printf '  %s\n' "${STALE[@]}"
    echo "These can leak into a wheel built from this tree without a clean first."
    echo "Run: rm -rf build dist src/*.egg-info"
    exit 1
fi
