"""Flag comment blocks and docstrings that have grown too long.

Long prose next to code tends to accumulate history --- what an earlier version
did, what was measured once, why an alternative was dropped --- that no longer
helps someone reading the code as it is. This check does not judge content; it
finds the longest blocks so they can be reviewed and cut back.

Two limits:

* a comment block is a run of consecutive full-line ``#`` comments (a blank
  line ends it; a trailing comment after code does not count), limited to
  ``--max-comment`` lines;
* a docstring is that of a function or method, or an attribute docstring (a
  string statement straight after an assignment, as used for dataclass fields
  and module constants), limited to ``--max-docstring`` lines. Module and
  class docstrings are exempt: the docstring at the top of a class carries the
  overview of what it is for, which is the one place length earns its keep.

Only a docstring's prose counts. Counting stops at the first numpy-style
reference section --- ``Parameters``, ``Returns`` and the like, a header
underlined with dashes --- because a thoroughly documented argument list is
not what this is looking for. ``Notes`` is prose and still counts.

Lines are counted from the opening line inclusive.
"""

import argparse
import ast
import io
import itertools
import sys
import tokenize
from pathlib import Path

MAX_COMMENT = 8
"""Default longest comment block allowed [lines]."""

MAX_DOCSTRING = 32
"""Default longest docstring prose allowed [lines]."""

REFERENCE_SECTIONS = {
    "Parameters",
    "Other Parameters",
    "Returns",
    "Yields",
    "Receives",
    "Raises",
    "Warns",
    "Warnings",
    "See Also",
    "Attributes",
    "Methods",
    "References",
    "Examples",
}
"""numpy docstring sections that document an interface rather than explain it."""


def comment_blocks(source):
    """Yield ``(first_line, n_lines)`` for each run of full-line comments."""
    lines = source.splitlines()
    comment_lines = []
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    try:
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            row, col = token.start
            # Full-line only: nothing but whitespace before the `#`.
            if lines[row - 1][:col].strip():
                continue
            comment_lines.append(row)
    except tokenize.TokenError:
        return

    start = previous = None
    for row in comment_lines:
        if start is not None and row == previous + 1:
            previous = row
            continue
        if start is not None:
            yield start, previous - start + 1
        start = previous = row
    if start is not None:
        yield start, previous - start + 1


def _prose_lines(node):
    """Return how many lines of the string statement `node` come before its
    first reference section, or all of them if it has none."""
    lines = node.value.value.splitlines()
    for i, (line, underline) in enumerate(itertools.pairwise(lines)):
        header = line.strip()
        rule = underline.strip()
        if header in REFERENCE_SECTIONS and rule and set(rule) == {"-"}:
            # `i` lines of the string precede the header, the first of them
            # being the opening line.
            return i
    return node.end_lineno - node.lineno + 1


def _is_string_statement(node):
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def docstrings(tree):
    """Yield ``(first_line, n_lines, owner)`` for every limited docstring."""
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue

        # The docstring proper of a function or method. A class's own is
        # exempt, unlike the attribute docstrings in its body below.
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and body
            and _is_string_statement(body[0])
        ):
            doc = body[0]
            yield doc.lineno, _prose_lines(doc), node.name

        # Attribute docstrings: a string statement right after an assignment,
        # in a module or a class body. The module's own docstring is skipped
        # because nothing precedes it.
        if not isinstance(node, (ast.Module, ast.ClassDef)):
            continue
        for before, stmt in itertools.pairwise(body):
            if not _is_string_statement(stmt):
                continue
            if isinstance(before, ast.Assign):
                targets = before.targets
            elif isinstance(before, ast.AnnAssign):
                targets = [before.target]
            else:
                continue
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            owner = names[0] if names else "<attribute>"
            if isinstance(node, ast.ClassDef):
                owner = f"{node.name}.{owner}"
            yield stmt.lineno, _prose_lines(stmt), owner


def check_file(path, max_comment, max_docstring):
    """Return a list of ``(path, line, message)`` for blocks over the limits."""
    source = Path(path).read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError as err:
        return [(path, err.lineno or 0, f"SyntaxError: {err.msg}")]

    found = []
    for line, n in comment_blocks(source):
        if n > max_comment:
            found.append((path, line, f"comment block of {n} lines > {max_comment}"))
    for line, n, owner in docstrings(tree):
        if n > max_docstring:
            found.append(
                (
                    path,
                    line,
                    f"docstring of {owner} is {n} prose lines > {max_docstring}",
                )
            )
    return sorted(found, key=lambda item: item[1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="files to check")
    parser.add_argument("--max-comment", type=int, default=MAX_COMMENT)
    parser.add_argument("--max-docstring", type=int, default=MAX_DOCSTRING)
    args = parser.parse_args(argv)

    paths = (
        [Path(p) for p in args.paths]
        if args.paths
        else sorted(Path("src/ember").glob("**/*.py"))
    )

    found = []
    for path in paths:
        found.extend(check_file(path, args.max_comment, args.max_docstring))

    for path, line, message in found:
        print(f"{path}:{line}: {message}")

    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
