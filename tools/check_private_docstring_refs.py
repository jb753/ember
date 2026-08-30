"""Check that public docstrings do not cross-reference private members.

A docstring on a public module, class or function is user-facing: it renders
into the published API docs, where a private member has no page to point at.
Sphinx does not document private members (``autodoc_default_options`` in
``docs/conf.py`` enables no ``private-members``), so a ``:meth:`Foo._bar```
in public prose resolves to nothing, and with ``nitpicky = True`` plus the
``-W`` in the ``docs`` Makefile target that unresolved reference fails the
build. That failure only surfaces on master, because ``.github/workflows/
docs.yml`` triggers on push to master and not on pull requests -- so this
check exists to catch it at PR time instead.

Private docstrings may reference whatever they like: they are notes to the
next maintainer, not published prose.
"""

import ast
import re
import sys
from pathlib import Path

# Python-domain roles that resolve to a documented object. Roles that render
# text rather than a link (``:ref:``, ``:doc:``, ``:term:``) are not checked.
XREF_ROLES = {
    "attr",
    "class",
    "data",
    "exc",
    "func",
    "meth",
    "mod",
    "obj",
}

# :role:`target`, :role:`~target`, :role:`title <target>`, :py:role:`target`.
ROLE_RE = re.compile(r":(?:py:)?(\w+):`([^`]+)`")
# ``literal`` and `literal`, with any preceding role already consumed above.
# De-linking a private cross-reference into a literal is not a fix -- the name
# is still in user-facing prose -- so literals are checked the same way.
LITERAL_RE = re.compile(r"(?<![:`\w])(``?)([^`\n]+?)\1(?!`)")
# A single leading underscore, but not a dunder like __init__ or __call__.
PRIVATE_RE = re.compile(r"^_(?!_.*__$)")

# Naming conventions, not members. Docstrings discuss these as suffixes ("the
# ``_nd`` nondimensional suffix", "``_mid`` rather than the ``_ref`` suffix"),
# and the definition index cannot tell that apart from a reference because
# unrelated classes do happen to own attributes of the same name.
SUFFIX_CONVENTIONS = {"_nd", "_mid", "_ref", "_rel"}


def _target_of(raw):
    """The object a role body points at, stripped of title and ~ prefix."""
    body = raw.strip()
    if "<" in body and body.endswith(">"):
        body = body[body.index("<") + 1 : -1].strip()
    return body.lstrip("~").strip()


def _private_part(target, defined):
    """The first component of ``target`` naming a real private definition.

    Matching the ``_foo`` shape alone is not enough. The codebase uses a
    leading underscore for naming *conventions* too -- the ``_nd``
    nondimensional suffix, ``_mid``/``_ref``/``_rel`` -- and those are prose
    about a convention, not references to a member. So a component only counts
    when something in ``src/ember`` actually defines that name.
    """
    for part in target.split("."):
        if part in SUFFIX_CONVENTIONS:
            continue
        if PRIVATE_RE.match(part) and part in defined:
            return part
    return None


def private_definitions(paths):
    """Every private name actually defined across ``paths``.

    Functions, classes, module-level assignments and ``self._x`` attributes.
    """
    names = set()
    for path in paths:
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                names.add(node.name)
            elif isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
    return {n for n in names if PRIVATE_RE.match(n)}


def _docstring_violations(path, qualname, node, offset_name, defined):
    """Private cross-references in ``node``'s docstring, as report lines."""
    raw = ast.get_docstring(node, clean=False)
    if not raw:
        return []

    # node.body[0] is the docstring Expr; its lineno is where the string
    # literal starts, so line N of the docstring text sits that many lines
    # further down. Good enough to click on, which is the point.
    base = node.body[0].lineno

    out = []
    spans = []
    for match in ROLE_RE.finditer(raw):
        spans.append(match.span())
        role, body = match.group(1), match.group(2)
        if role not in XREF_ROLES:
            continue
        target = _target_of(body)
        private = _private_part(target, defined)
        if private is None:
            continue
        line = base + raw.count("\n", 0, match.start())
        out.append(
            f"  {path}:{line}: {offset_name} {qualname} references private"
            f" '{private}' via :{role}:`{body}`"
        )

    for match in LITERAL_RE.finditer(raw):
        # Skip the backtick runs already reported as a role above.
        if any(s <= match.start() < e for s, e in spans):
            continue
        private = _private_part(_target_of(match.group(2)), defined)
        if private is None:
            continue
        line = base + raw.count("\n", 0, match.start())
        out.append(
            f"  {path}:{line}: {offset_name} {qualname} names private"
            f" '{private}' in literal {match.group(0)}"
        )
    return out


def check_file(path, defined):
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as e:
        return [f"  {path}: SyntaxError: {e}"]

    violations = []

    # A module under a private filename (_struct.py) is itself private;
    # __init__.py is the package's public face.
    if PRIVATE_RE.match(path.stem) and path.stem != "__init__":
        return []
    violations.extend(_docstring_violations(path, path.stem, tree, "module", defined))

    # Walk with a scope stack so a public method on a private class, or a
    # nested helper, is correctly treated as private.
    def walk(node, scope):
        for child in node.body:
            if not isinstance(
                child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                continue
            name = child.name
            public = scope is not None and not PRIVATE_RE.match(name)
            qualname = f"{scope}.{name}" if scope else name
            if public:
                kind = "class" if isinstance(child, ast.ClassDef) else "function"
                violations.extend(
                    _docstring_violations(path, qualname, child, kind, defined)
                )
            walk(child, qualname if public else None)

    walk(tree, "")
    return violations


def main():
    paths = (
        [Path(p) for p in sys.argv[1:]]
        if sys.argv[1:]
        else sorted(Path("src/ember").glob("**/*.py"))
    )
    defined = private_definitions(sorted(Path("src/ember").glob("**/*.py")))
    all_violations = []
    for path in paths:
        all_violations.extend(check_file(path, defined))

    if all_violations:
        print("Public docstrings referencing private members:")
        for v in all_violations:
            print(v)
        sys.exit(1)


if __name__ == "__main__":
    main()
