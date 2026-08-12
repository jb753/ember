"""Sphinx extension: drop the leading "ember." package segment from autodoc
object headings and sidebar TOC entries, keeping the rest of the module path
(e.g. "ember.util.dot" displays as "util.dot").

Scope: this only touches two things Sphinx's Python domain generates itself --
the ``desc_addname`` node (the module-prefix text rendered before an object's
own name in its definition heading) and the ``_toc_name`` attribute stashed on
each ``desc_signature`` node (consumed when building sidebar/localtoc object
entries). It does not touch prose, cross-reference link text, or code/doctest
blocks -- those are different node types entirely (``Text``, ``reference``,
``literal_block``/``doctest_block``) that autodoc's signature builder never
constructs.
"""

from docutils import nodes
from sphinx import addnodes

_PREFIX = "ember."


def _strip_prefix(text):
    return text[len(_PREFIX) :] if text.startswith(_PREFIX) else text


def _trim(app, doctree):
    for signode in doctree.findall(addnodes.desc_signature):
        toc_name = signode.get("_toc_name")
        if toc_name:
            signode["_toc_name"] = _strip_prefix(toc_name)
        for addname in signode.findall(addnodes.desc_addname):
            old_text = addname.astext()
            new_text = _strip_prefix(old_text)
            if new_text != old_text:
                addname.children = [nodes.Text(new_text)]


def setup(app):
    # Sphinx's built-in TocTreeCollector also listens on 'doctree-read' (at
    # the default priority 500) to snapshot each object's `_toc_name` into
    # the environment's cached toctree/sidebar data. That snapshot must see
    # our trimmed text, so connect at a lower priority to run first.
    app.connect("doctree-read", _trim, priority=100)
    return {"version": "0.1", "parallel_read_safe": True}
