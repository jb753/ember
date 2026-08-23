import os
import sys
import datetime
import importlib.metadata

sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath("_ext"))

project = "ember"
author = "James Brind"
release = importlib.metadata.version("ember-cfd")
version = release
python_version = importlib.metadata.metadata("ember-cfd")[
    "Requires-Python"
].removeprefix(">=")

start_year = 2023
current_year = datetime.datetime.now().year
copyright_years = (
    str(start_year) if current_year == start_year else f"{start_year}–{current_year}"
)
copyright = f"{copyright_years}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "autodocsumm",
    "tikz",
    "trim_module_prefix",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
]

bibtex_bibfiles = ["refs.bib"]

# -- sphinx-gallery ----------------------------------------------------------
# Example scripts live in ``../examples`` and the rendered gallery is written
# to ``auto_examples`` within the build.
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    # Only execute examples/plot_*.py on a normal build. examples/run_*.py
    # cases are expensive (many seconds each) and are only re-run when
    # explicitly requested, e.g. `make docs-full` -- see examples/README.txt.
    # Their generated docs/auto_examples/run_* output is committed to git (see
    # .gitignore) so a clean checkout, e.g. Read the Docs, reuses it via
    # sphinx-gallery's md5 cache instead of re-running it. After changing a
    # run_*.py example: `make docs-full`, then `git add` the regenerated
    # docs/auto_examples/run_* files and commit them.
    "filename_pattern": r"[\\/]plot_",
    "within_subsection_order": "FileNameSortKey",
    "matplotlib_animations": False,
    "remove_config_comments": True,
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

add_module_names = True
# Show the full dotted path in sidebar TOC object entries, matching the
# fully-qualified headings produced by ``add_module_names``.
toc_object_entries_show_parents = "all"
autoclass_content = "init"
autodoc_member_order = "bysource"
autodoc_default_options = {
    # "autosummary": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# Fail the build on broken cross-references (enforced via `-W` in the Makefile).
nitpicky = True
suppress_warnings = [
    # Sphinx >=9 resolves bare-word napoleon type annotations (e.g. "shape")
    # against every same-named attribute in the project, so a word like
    # "shape" that happens to be an attribute on both Block and Patch now
    # triggers an ambiguous-reference warning. These aren't real ambiguities:
    # napoleon type annotations are prose, not cross-references.
    "ref.python",
]
nitpick_ignore_regex = [
    # Napoleon renders parameter *type* strings as class references, e.g.
    # "array-like", "shape (ni, nj)", dimension tokens and integer literals.
    # None of these are real targets; they never contain a dotted path.
    (r"py:.*", r"^[^.]*$"),
    # "default 1.0" and similar default-value annotations leak through with a dot.
    (r"py:class", r"default.*"),
]
nitpick_ignore = [
    # Objects that do not yet have a documentation page. Remove an entry once
    # the corresponding module gains an autodoc page so the link resolves.
    ("py:class", "ember.collections._LabelledList"),
    # Private base class, so no page of its own; ``:show-inheritance:`` on
    # PerfectFluid still emits a reference to it from the rendered "Bases:" line.
    ("py:class", "ember.fluid._Fluid"),
    ("py:meth", "ember._struct.StructuredData.__init__"),
    # Documented on the private base, which has no page of its own; Block.flat
    # points readers at it for the ordering rules.
    ("py:attr", "ember._struct.StructuredData.flat"),
    ("py:func", "ember._struct.cached_array"),
    # Private helper, so no page of its own; the multigrid docstrings name
    # it as what sizes Block.scratch, which is the invariant they turn on.
    ("py:func", "ember.block._scratch_len"),
]

html_theme = "alabaster"
html_static_path = ["_static"]
html_theme_options = {
    "description": f"Version {release}",
    "fixed_sidebar": True,
}

# Docs are published as one directory per version, so a build has to know the
# slug it will be served under: the rolling master build lives at "dev", not at
# setuptools-scm's "0.2.1.devN+g<sha>". The picker template reads this to mark
# the current entry.
html_context = {"ember_version_slug": os.environ.get("EMBER_DOCS_VERSION", release)}

# Setting html_sidebars replaces alabaster's theme-level default wholesale, so
# its stock blocks are re-listed here around the version picker.
html_sidebars = {
    "**": [
        "about.html",
        "versions.html",
        "searchfield.html",
        "navigation.html",
        "relations.html",
        "donate.html",
    ]
}

# html_baseurl is deliberately unset. Without it Sphinx emits only relative
# links, so one build tree serves correctly both from a GitHub Pages project
# subpath and from the site apex, with no rebuild in between.

rst_epilog = rf"""
.. |ProjectVersion| replace:: {release}
.. |PythonVersion| replace:: {python_version}

.. |m2| replace:: m\ :sup:`2`
.. |m3| replace:: m\ :sup:`3`
.. |ms2| replace:: m\ :sup:`2`\ /s
.. |Jm3| replace:: J/m\ :sup:`3`
.. |kgm2s| replace:: kg/m\ :sup:`2`\ /s
.. |kgm3| replace:: kg/m\ :sup:`3`
.. |JkgK| replace:: J/kg/K
.. |Jkg| replace:: J/kg
.. |ms| replace:: m/s
.. |Pa| replace:: Pa
.. |K| replace:: K
.. |rads| replace:: rad/s
.. |rpm| replace:: rpm
.. |rad| replace:: rad
.. |m| replace:: m
.. |deg| replace:: deg
.. |minus| replace:: --
"""
