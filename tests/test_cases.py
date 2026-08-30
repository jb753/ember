"""Tests for the case-construction scaffolding in :mod:`ember.cases`."""

import numpy as np
import pytest

from ember import util
from ember.cases import build_duct_grid, er_for_duct_yplus

# Smallest duct that still clears build_duct_grid's ni >= 25 floor at these
# cross-stream counts, so the probe build stays cheap.
SETTINGS = dict(ncell=30_000, nj=33, nk=29)
SIDE = 0.1  # build_duct_grid's own default cross-section side


def first_spacing(ER, nj=SETTINGS["nj"]):
    """Height of the first cell off the wall for a clustering ratio."""
    return SIDE * float(np.diff(util.cluster_symmetric(nj, ER))[0])


def test_er_for_duct_yplus_clusters():
    """The solved ratio refines the wall relative to a uniform mesh."""
    ER = er_for_duct_yplus(30.0, **SETTINGS)
    assert ER > 1.0
    assert first_spacing(ER) < SIDE / (SETTINGS["nj"] - 1)


def test_er_for_duct_yplus_scales_with_target():
    """Wall spacing is proportional to the y+ target it was solved for."""
    d30 = first_spacing(er_for_duct_yplus(30.0, **SETTINGS))
    d60 = first_spacing(er_for_duct_yplus(60.0, **SETTINGS))
    assert d60 / d30 == pytest.approx(2.0, rel=1e-3)


def test_er_for_duct_yplus_correlations_differ():
    """The two skin-friction correlations do not give the same mesh."""
    er_white = er_for_duct_yplus(30.0, **SETTINGS)
    er_prandtl = er_for_duct_yplus(30.0, correlation="prandtl", **SETTINGS)
    assert er_white != er_prandtl
    with pytest.raises(ValueError, match="unknown correlation"):
        er_for_duct_yplus(30.0, correlation="nope", **SETTINGS)


def test_er_for_duct_yplus_unreachable():
    """A target coarser than the uniform mesh cannot be clustered onto."""
    with pytest.raises(ValueError, match="COARSER"):
        er_for_duct_yplus(1e6, **SETTINGS)


def test_er_for_duct_yplus_builds():
    """The solved ratio feeds straight back into a clustered duct grid."""
    ER = er_for_duct_yplus(30.0, **SETTINGS)
    b = build_duct_grid(cluster=True, ER=ER, **SETTINGS)[0]
    dr = float(b.r[0, 1, 0] - b.r[0, 0, 0])
    assert dr == pytest.approx(first_spacing(ER), rel=1e-4)
