"""Smoke test that tools/run_duct.py's CLI still matches the current API.

This is not a solver test -- convergence of solver internals is covered
elsewhere. This guards against the kind of drift that broke the script
before: a library-side rename (e.g. SolverConfig -> Solver) landing without
updating this CLI wrapper, which run_duct.py's own try/except does not catch
since it only guards against RuntimeError/FloatingPointError from a diverged
run, not an AttributeError from a stale call site.
"""

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_duct", Path(__file__).parent.parent / "tools" / "run_duct.py"
)
run_duct = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_duct)


def test_run_duct_runs(monkeypatch):
    # Smallest grid the CLI can build (fixed nj=65, nk=57 cross-section);
    # 5 steps is nowhere near enough to converge, so run_duct.py is expected
    # to exit 2 (ran fine, did not converge) -- exit 1 would mean it diverged,
    # any other exception means the script no longer matches the library API.
    monkeypatch.setattr(
        sys, "argv", ["run_duct.py", "--n-step", "5", "--ncell", "100000"]
    )
    try:
        run_duct.main()
    except SystemExit as exc:
        assert exc.code in (1, 2)
