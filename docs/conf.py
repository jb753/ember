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
    ("py:class", "ember.mixing_communicator.MixingCommunicator"),
    ("py:class", "ember.nonmatch_communicator.NonMatchCommunicator"),
    ("py:class", "ember.collections._LabelledList"),
    ("py:class", "ember.fluid._Fluid"),
    # Compiled Fortran extension: no autodoc page, so :func: refs to its kernels
    # (used throughout the solver docstrings) cannot resolve.
    ("py:func", "ember.fortran.smooth3d_const"),
]

html_theme = "alabaster"
html_static_path = ["_static"]
html_theme_options = {
    "description": f"Version {release}",
    "fixed_sidebar": True,
}

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
