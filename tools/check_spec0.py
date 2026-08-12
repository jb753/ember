"""Check that the Python version floor and wheel build matrix track numpy.

ember follows SPEC 0's Python support window (see pyproject.toml), and numpy
is a strict SPEC-0 anchor project, so it sets the target:

  - `project.requires-python` must equal the support floor.
  - `tool.cibuildwheel.build` must build wheels for exactly the set of Python
    minor versions we are able to support.

Being *behind* numpy is a failure, but so is running ahead of our own
dependencies. A wheel we build for a Python that h5py or pykdtree has no wheel
for is a wheel nobody can install: pip falls back to a source build of the
dependency, which needs system HDF5 headers and a compiler and fails in both
the cibuildwheel test container and the verify-install job. numpy routinely
declares support for a new Python months before the rest of the compiled
scientific stack ships binaries for it.

So the supported set is numpy's set intersected with the Python versions every
compiled runtime dependency actually publishes wheels for, and the floor is the
highest floor among numpy and those dependencies. Dependency versions come from
uv.lock, which is what CI installs. When the newest Python numpy supports is
held back by a dependency, that is reported but is not an error -- it is the
correct state, and the check starts requiring the newer Python as soon as the
laggard ships wheels for it.

Requires network access to query PyPI.
"""

import json
import re
import sys
import tomllib
import urllib.request
from pathlib import Path

PYPI_URL = "https://pypi.org/pypi/{name}/json"
CLASSIFIER_RE = re.compile(r"^Programming Language :: Python :: 3\.(\d+)$")
CP_TAG_RE = re.compile(r"cp3(\d+)")
# Wheel filenames are name-version-[build-]pytag-abitag-platform.whl; the
# python tag is the third hyphen-separated field only when there is no build
# tag, so pull the tag out by pattern rather than by position.
WHEEL_PY_TAG_RE = re.compile(r"-(?:cp|pp|py)3(\d+)-")


def _pypi(name, _cache={}):
    if name not in _cache:
        with urllib.request.urlopen(PYPI_URL.format(name=name)) as resp:
            _cache[name] = json.load(resp)
    return _cache[name]


def latest_version(name):
    """The newest non-prerelease version of `name` on PyPI."""
    return _pypi(name)["info"]["version"]


def numpy_supported_versions():
    """Python minor versions numpy declares support for, from its classifiers."""
    versions = set()
    for classifier in _pypi("numpy")["info"]["classifiers"]:
        m = CLASSIFIER_RE.match(classifier)
        if m:
            versions.add(int(m.group(1)))
    return versions


def wheel_versions(name, version):
    """Python minor versions `name == version` publishes wheels for.

    Returns None for a package whose wheels are version-independent
    (``py3-none-any``), which constrains nothing.
    """
    files = _pypi(name)["releases"].get(version, [])
    wheels = [f["filename"] for f in files if f["filename"].endswith(".whl")]
    if any("-py3-none-any.whl" in w or "-py2.py3-none-any.whl" in w for w in wheels):
        return None
    return {int(m.group(1)) for w in wheels for m in WHEEL_PY_TAG_RE.finditer(w)}


def locked_versions():
    """Package name -> version, as resolved in uv.lock."""
    lock = tomllib.loads(Path("uv.lock").read_text())
    return {p["name"]: p["version"] for p in lock["package"] if "version" in p}


def runtime_dependencies(pyproject):
    """Distribution names of the project's runtime dependencies."""
    return [
        re.split(r"[<>=~!\[; ]", spec, maxsplit=1)[0].strip()
        for spec in pyproject["project"]["dependencies"]
    ]


def our_requires_python_floor(pyproject):
    spec = pyproject["project"]["requires-python"]
    m = re.fullmatch(r">=3\.(\d+)", spec)
    if not m:
        raise ValueError(f"Unsupported requires-python format: {spec!r}")
    return int(m.group(1))


def cibuildwheel_versions(pyproject):
    build = pyproject["tool"]["cibuildwheel"]["build"]
    return {int(m.group(1)) for m in CP_TAG_RE.finditer(build)}


def _fmt(versions):
    return sorted(f"3.{v}" for v in versions)


def main():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    numpy_versions = numpy_supported_versions()
    locked = locked_versions()

    # What each compiled runtime dependency can be installed on, both as
    # currently locked and as it would be if upgraded to its newest release.
    # The pair is what stops a stale lock, or a specifier nobody revisits,
    # from silently capping the support window: the locked versions set what
    # we can claim today, the newest ones set what we ought to be claiming.
    dep_support = {}
    dep_upgrade = {}
    for name in runtime_dependencies(pyproject):
        version = locked.get(name)
        if version is None:
            print(f"SPEC 0 check failed:\n  {name} is not present in uv.lock")
            sys.exit(1)
        supported = wheel_versions(name, version)
        if supported:
            dep_support[f"{name} {version}"] = supported
            newest = latest_version(name)
            if newest != version:
                upgraded = wheel_versions(name, newest)
                if upgraded and upgraded - supported:
                    dep_upgrade[name] = (version, newest, upgraded - supported)

    supported = set(numpy_versions)
    for versions in dep_support.values():
        supported &= versions
    if not supported:
        print(
            "SPEC 0 check failed:\n  no Python version is supported by numpy and "
            "every compiled dependency at once"
        )
        sys.exit(1)

    # The floor is the most restrictive of numpy's and the dependencies'.
    floor = max([min(numpy_versions)] + [min(v) for v in dep_support.values()])
    expected_build = {v for v in supported if v >= floor}

    our_floor = our_requires_python_floor(pyproject)
    build = cibuildwheel_versions(pyproject)

    # What we could support if every dependency were upgraded to its newest
    # release. Anything in here and not in expected_build is support we are
    # leaving on the table, and the fix is ours to make (uv lock --upgrade, or
    # widening the specifier), so it is an error rather than a note.
    achievable = set(numpy_versions)
    for dep, versions in dep_support.items():
        name = dep.rsplit(" ", 1)[0]
        gained = dep_upgrade.get(name, (None, None, set()))[2]
        achievable &= versions | gained
    achievable = {v for v in achievable if v >= floor}

    errors = []
    if our_floor != floor:
        errors.append(f"requires-python floor is 3.{our_floor}, but should be 3.{floor}")
    if build != expected_build:
        errors.append(
            f"cibuildwheel build versions {_fmt(build)} do not match the "
            f"supportable versions {_fmt(expected_build)}"
        )
    if achievable - expected_build:
        stale = ", ".join(
            f"{name} {old} -> {new}" for name, (old, new, _) in sorted(dep_upgrade.items())
        )
        errors.append(
            f"{_fmt(achievable - expected_build)} would be supportable after "
            f"upgrading: {stale}"
        )

    if errors:
        print("SPEC 0 check failed:")
        for e in errors:
            print(f"  {e}")
        # Name the laggards, so the fix is obvious rather than a puzzle.
        for newest in sorted(numpy_versions - expected_build, reverse=True):
            blockers = [
                dep for dep, versions in dep_support.items() if newest not in versions
            ]
            if blockers:
                print(
                    f"  note: numpy supports 3.{newest} but no wheels for it from: "
                    + ", ".join(sorted(blockers))
                )
        sys.exit(1)

    print(f"SPEC 0 check passed: 3.{floor} floor, wheels for {_fmt(expected_build)}")
    for held in sorted(numpy_versions - expected_build, reverse=True):
        blockers = [dep for dep, v in dep_support.items() if held not in v]
        if blockers:
            print(
                f"  held back from 3.{held} (numpy supports it) by: "
                + ", ".join(sorted(blockers))
            )


if __name__ == "__main__":
    main()
