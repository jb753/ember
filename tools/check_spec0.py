"""Check that the Python version floor and wheel build matrix track numpy.

ember follows SPEC 0's Python support window (see pyproject.toml), and
numpy is a strict SPEC-0 anchor project, so it is used as the reference:

  - `project.requires-python` in pyproject.toml must equal numpy's lowest
    supported Python minor version.
  - `tool.cibuildwheel.build` must build wheels for exactly the set of
    Python minor versions numpy supports, including the newest.

Requires network access to query PyPI for numpy's current metadata.
"""

import json
import re
import sys
import tomllib
import urllib.request
from pathlib import Path

NUMPY_METADATA_URL = "https://pypi.org/pypi/numpy/json"
CLASSIFIER_RE = re.compile(r"^Programming Language :: Python :: 3\.(\d+)$")
CP_TAG_RE = re.compile(r"cp3(\d+)")


def numpy_supported_versions():
    with urllib.request.urlopen(NUMPY_METADATA_URL) as resp:
        data = json.load(resp)
    versions = set()
    for classifier in data["info"]["classifiers"]:
        m = CLASSIFIER_RE.match(classifier)
        if m:
            versions.add(int(m.group(1)))
    return versions


def our_requires_python_floor(pyproject):
    spec = pyproject["project"]["requires-python"]
    m = re.fullmatch(r">=3\.(\d+)", spec)
    if not m:
        raise ValueError(f"Unsupported requires-python format: {spec!r}")
    return int(m.group(1))


def cibuildwheel_versions(pyproject):
    build = pyproject["tool"]["cibuildwheel"]["build"]
    return {int(m.group(1)) for m in CP_TAG_RE.finditer(build)}


def main():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    numpy_versions = numpy_supported_versions()
    our_floor = our_requires_python_floor(pyproject)
    wheel_versions = cibuildwheel_versions(pyproject)

    errors = []

    numpy_floor = min(numpy_versions)
    if our_floor != numpy_floor:
        errors.append(
            f"requires-python floor is 3.{our_floor}, but numpy's floor is "
            f"3.{numpy_floor}"
        )

    if wheel_versions != numpy_versions:
        errors.append(
            "cibuildwheel build versions "
            f"{sorted(f'3.{v}' for v in wheel_versions)} do not match "
            f"numpy's supported versions {sorted(f'3.{v}' for v in numpy_versions)}"
        )

    if errors:
        print("SPEC 0 check failed:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)

    print(
        f"SPEC 0 check passed: 3.{our_floor} floor, wheels for {sorted(f'3.{v}' for v in numpy_versions)}"
    )


if __name__ == "__main__":
    main()
