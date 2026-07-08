"""Sphinx extension: compile .tikz files to SVG and emit image nodes.

Usage in RST::

    .. tikz:: _tikz/cell_indexing.tikz
       :alt: Cell indexing diagram

The path is relative to the Sphinx source directory (confdir).
The SVG is written to ``_tikz_out/`` inside the build directory and
copied into the HTML output tree via Builder.outdir.
"""

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.util import logging

logger = logging.getLogger(__name__)


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


def _compile(tikz_path: Path, out_svg: Path) -> None:
    """Compile a standalone .tikz file to SVG via pdflatex + pdf2svg."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tex = tmp / "fig.tex"
        shutil.copy(tikz_path, tex)
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig.tex"],
            cwd=tmp,
            check=True,
            capture_output=True,
        )
        pdf = tmp / "fig.pdf"
        subprocess.run(
            ["pdf2svg", str(pdf), str(out_svg)],
            check=True,
            capture_output=True,
        )


class TikzDirective(Directive):
    required_arguments = 1  # path to .tikz file
    optional_arguments = 0
    option_spec = {
        "alt": directives.unchanged,
        "width": directives.unchanged,
        "align": directives.unchanged,
    }
    has_content = False

    def run(self):
        env = self.state.document.settings.env
        rel_path = self.arguments[0]
        tikz_path = Path(env.app.srcdir) / rel_path

        if not tikz_path.exists():
            raise self.error(f"tikz: file not found: {tikz_path}")

        # Write SVG into _static/_tikz/ so Sphinx copies it automatically
        out_dir = Path(env.app.srcdir) / "_static" / "_tikz"
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = tikz_path.stem
        tag = _sha1(tikz_path)
        svg_name = f"{stem}_{tag}.svg"
        out_svg = out_dir / svg_name

        if not out_svg.exists():
            logger.info(f"tikz: compiling {tikz_path.name} ...")
            try:
                _compile(tikz_path, out_svg)
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or b"").decode()
                raise self.error(f"tikz: compilation failed:\n{stderr}") from exc

        # URI relative to the HTML output root (_static is always copied)
        uri = f"_static/_tikz/{svg_name}"

        opts = {"uri": uri}
        if "alt" in self.options:
            opts["alt"] = self.options["alt"]
        if "width" in self.options:
            opts["width"] = self.options["width"]
        if "align" in self.options:
            opts["align"] = self.options["align"]

        node = nodes.image(**opts)
        return [node]


def setup(app):
    app.add_directive("tikz", TikzDirective)
    return {"version": "0.1", "parallel_read_safe": True}
