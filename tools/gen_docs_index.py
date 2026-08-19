"""Regenerate the root index, version list and .nojekyll of the docs site.

The published docs are one directory per version, each built once when its ref
is pushed and never rebuilt. Nothing that lives inside a version directory can
therefore describe the set of versions -- 0.2.0's pages were written before
0.3.0 existed. So the shared state lives at the site root instead, and this
script rewrites it from whatever directories are actually present after a
deploy has synced its own:

  versions.json  read at page load by the sidebar picker (docs/_templates)
  index.html     redirects the site root to the newest release
  .nojekyll      stops GitHub Pages running Jekyll, which silently discards
                 the _static/ and _sources/ directories Sphinx emits

Directories are ordered dev first, then releases newest first, then anything
else alphabetically. Only a non-prerelease version is ever a redirect target,
so a throwaway or prerelease deploy is listed but never becomes the landing
page.

Usage: gen_docs_index.py <site-dir>
"""

import json
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version

DEV = "dev"

# GitHub Pages serves static files and cannot answer with a 3xx, so the root
# has to redirect from the page itself. location.replace runs before the body
# paints, so nothing flashes, and it leaves no history entry -- a meta refresh
# alone can trap the back button on the page it just left. The refresh stays as
# the fallback for a browser running without scripts.
REDIRECT = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>ember documentation</title>
    <script>location.replace("./{name}/index.html");</script>
    <meta http-equiv="refresh" content="0; url=./{name}/index.html">
    <link rel="canonical" href="./{name}/index.html">
  </head>
  <body></body>
</html>
"""


def find_versions(site):
    """Return the names of subdirectories of `site` holding a built doc set."""
    return sorted(
        path.name
        for path in site.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and (path / "index.html").exists()
    )


def parse(name):
    """Return the Version for a directory name, or None if it isn't one."""
    try:
        return Version(name)
    except InvalidVersion:
        return None


def order(names):
    """Sort names dev first, then releases newest first, then the rest."""
    dev = [name for name in names if name == DEV]
    releases = sorted(
        (n for n in names if n != DEV and parse(n) is not None),
        key=parse,
        reverse=True,
    )
    other = sorted(n for n in names if n != DEV and parse(n) is None)
    return dev + releases + other


def newest_release(names):
    """Return the highest non-prerelease version name, or None."""
    stable = [
        name
        for name in names
        if name != DEV and parse(name) is not None and not parse(name).is_prerelease
    ]
    if not stable:
        return None
    return max(stable, key=parse)


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2

    site = Path(argv[1])
    names = find_versions(site)
    if not names:
        print(f"no version directories found in {site}", file=sys.stderr)
        return 1

    ordered = order(names)
    (site / "versions.json").write_text(
        json.dumps(
            {"versions": [{"name": name, "url": f"{name}/"} for name in ordered]},
            indent=2,
        )
        + "\n"
    )

    # Fall back to the first listed directory when no release has been
    # published yet, so a site holding only `dev` still has a working root.
    landing = newest_release(names) or ordered[0]
    (site / "index.html").write_text(REDIRECT.format(name=landing))

    (site / ".nojekyll").touch()

    print(f"versions: {', '.join(ordered)}")
    print(f"root redirects to: {landing}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
