"""Tests for throttle state recording in ConvergenceHistory."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from ember.convergence_history import ConvergenceHistory
from ember.fluid import PerfectFluid
from ember.grid import ConvergenceStep

_DATA = Path(__file__).parent / "data"


P0 = 101325.0
_FLUID = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)


def _throttle_conv(mdot_target, mdot_throttle, P_throttle):
    """A ConvergenceStep carrying only throttle state (flow monitors left NaN)."""
    return ConvergenceStep(
        residual=np.full(5, np.nan, np.float32),
        mdot=np.full(2, np.nan, np.float32),
        ho=np.full(2, np.nan, np.float32),
        s=np.full(2, np.nan, np.float32),
        mdot_target=mdot_target,
        mdot_throttle=mdot_throttle,
        P_throttle=P_throttle,
    )


@pytest.fixture
def hist_with_throttle():
    """Create a minimal ConvergenceHistory for testing throttle recording."""
    # Create a small history directly (not from_grid, to avoid grid setup complications)
    hist = ConvergenceHistory(shape=(10,))
    hist._set_metadata_by_key("n_block", 1)
    hist._set_metadata_by_key("n_node", 100)
    hist._set_metadata_by_key("A_in", np.float32(0.5))
    hist._set_metadata_by_key("A_out", np.float32(0.5))
    hist._set_metadata_by_key("is_rotating", False)
    hist._set_metadata_by_key("fluid", _FLUID)
    hist._set_metadata_by_key("_time_start", 0.0)
    hist._set_metadata_by_key("i_log", -1)
    # Mark all data keys as initialized
    for k in hist._data_keys:
        hist._versions[k] += 1
    hist.record_step(0)
    return hist


def test_throttle_shape(hist_with_throttle):
    """Test throttle compound property shape."""
    hist = hist_with_throttle
    # Record a few more steps
    for i in range(1, 3):
        hist.record_step(i)
        hist.record_convergence(_throttle_conv(1.0, 0.95, 101325.0))
    # throttle shape should be (n_steps_allocated, 3)
    throttle = hist.throttle
    assert throttle.shape[1] == 3, f"Expected second dimension 3, got {throttle.shape}"
    assert throttle.shape[0] == 10, (
        f"Expected first dimension 10 (allocated), got {throttle.shape[0]}"
    )


def test_record_throttle_inactive(hist_with_throttle):
    """Test that Throt line is absent when mdot_target=0."""
    hist = hist_with_throttle
    hist.record_convergence(_throttle_conv(0.0, 0.5, 101325.0))
    msg = hist.format_message()
    assert "Throt" not in msg, "Throt line should not appear when mdot_target=0"


def test_record_throttle_active(hist_with_throttle):
    """Test that Throt line appears when mdot_target > 0."""
    hist = hist_with_throttle
    hist.record_convergence(_throttle_conv(1.0, 0.95, 101325.0))
    msg = hist.format_message()
    assert "Throt" in msg, "Throt line should appear when mdot_target > 0"
    assert "mdot=0.9500/1.0000 kg/s" in msg


def test_record_throttle_err_sign(hist_with_throttle):
    """Test that error sign is correct when mdot_throttle < mdot_target."""
    hist = hist_with_throttle
    mdot_target = 1.0
    mdot_throttle = 0.9  # 10% below target
    hist.record_convergence(_throttle_conv(mdot_target, mdot_throttle, 101325.0))
    msg = hist.format_message()
    expected_err = (mdot_throttle - mdot_target) / mdot_target  # -0.1
    assert f"err={expected_err:+.3f}" in msg, f"Error sign incorrect in message: {msg}"
    assert "err=-0.100" in msg


def test_record_throttle_nan_passthrough(hist_with_throttle):
    """Test that NaN values are stored and don't crash format_message."""
    hist = hist_with_throttle
    hist.record_convergence(_throttle_conv(1.0, np.nan, np.nan))
    msg = hist.format_message()
    # Should not crash and should still contain Thr line
    assert "Thr" in msg


# ---------------------------------------------------------------------------
# from_ts3
# ---------------------------------------------------------------------------


def _make_duct_grid():
    """Single-block grid with inlet/outlet and a uniform flow field.

    Supplies from_ts3 with reference scales: a known mean axial velocity sets
    V_ref and a known outlet temperature sets T_ref. The geometry is otherwise
    arbitrary; the per-step history comes from the log, not the grid.
    """
    import ember.util
    from ember.block import Block
    from ember.grid import Grid
    from ember.patch import InletPatch, OutletPatch

    # Gas properties match the log_duct.txt header so from_ts3's grid/log
    # consistency check passes (cp=1005, ga=1.4, viscosity=0.001, prandtl=0.72).
    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.0e-3, Pr=0.72)

    shape = (4, 4, 4)
    block = Block(shape=shape)
    xrt = ember.util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], shape)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_fluid(fluid)
    block.set_rpm(0.0)
    block.set_Nb(1)
    block.set_P_T(np.full(shape, 1.0e5), np.full(shape, 300.0))
    block.set_Vxrt(
        np.stack([np.full(shape, 100.0), np.zeros(shape), np.zeros(shape)], axis=-1)
    )
    inlet = InletPatch(i=0, j=(0, -1), k=(0, -1), label="inlet")
    outlet = OutletPatch(i=-1, j=(0, -1), k=(0, -1), label="outlet")
    block.patches.append(inlet)
    block.patches.append(outlet)
    inlet.set_Po_To_Alpha_Beta(
        Po=np.full(inlet.shape, 1.0e5),
        To=np.full(inlet.shape, 300.0),
        Alpha=np.zeros(inlet.shape),
        Beta=np.zeros(inlet.shape),
    )
    outlet.set_P(0.9e5)
    return Grid([block])


@pytest.fixture(scope="module")
def ts3_hist():
    """ConvergenceHistory parsed from the duct log file with a grid for scales."""
    return ConvergenceHistory.from_ts3(_DATA / "log_duct.txt", _make_duct_grid())


def test_from_ts3_step_count(ts3_hist):
    """from_ts3 parses all 399 STEP blocks."""
    assert ts3_hist.i_log + 1 == 399


def test_from_ts3_steps_monotonic(ts3_hist):
    """Steps array starts at 50 and is strictly increasing."""
    n = ts3_hist.i_log + 1
    steps = ts3_hist.i_step[:n]
    assert steps[0] == 50
    assert np.all(np.diff(steps) > 0)


def test_from_ts3_metadata(ts3_hist):
    """Grid-derived metadata: single non-rotating row, no kinetic-energy scales."""
    assert ts3_hist._get_metadata_by_key("n_row") == 1
    assert "V_ref" not in ts3_hist._metadata
    assert "T_ref" not in ts3_hist._metadata


def test_from_ts3_mdot_finite(ts3_hist):
    """mdot_in and mdot_out are finite for all recorded steps."""
    n = ts3_hist.i_log + 1
    assert np.all(np.isfinite(ts3_hist.mdot_in[:n]))
    assert np.all(np.isfinite(ts3_hist.mdot_out[:n]))


def test_from_ts3_first_step_mdot(ts3_hist):
    """First-step mass flows match the log file values."""
    assert ts3_hist.mdot_in[0] == pytest.approx(1.38360, rel=1e-4)
    assert ts3_hist.mdot_out[0] == pytest.approx(1.38339, rel=1e-4)


def test_from_ts3_residual_drho_finite(ts3_hist):
    """drho residual is finite; other residual components are NaN (not in TS3 log)."""
    n = ts3_hist.i_log + 1
    assert np.all(np.isfinite(ts3_hist.residual[:n, 0]))


def test_from_ts3_ho_s_finite(ts3_hist):
    """ho_in, ho_out, s_in, s_out are finite for all steps."""
    n = ts3_hist.i_log + 1
    for attr in ("ho_in", "ho_out", "s_in", "s_out"):
        assert np.all(np.isfinite(getattr(ts3_hist, attr)[:n])), attr


def test_from_ts3_zeta_positive(ts3_hist):
    """Total pressure loss coefficient zeta is positive for all steps."""
    n = ts3_hist.i_log + 1
    assert np.all(ts3_hist.zeta[:n] > 0)


def test_from_ts3_fluid_mismatch_raises():
    """A grid whose fluid disagrees with the log header is rejected."""
    grid = _make_duct_grid()
    # Re-set the fluid to a different cp than the log header records.
    grid[0].set_fluid(PerfectFluid(cp=1100.0, gamma=1.4, mu=1.0e-3, Pr=0.72))
    with pytest.raises(ValueError, match="does not match grid"):
        ConvergenceHistory.from_ts3(_DATA / "log_duct.txt", grid)


# ---------------------------------------------------------------------------
# read_cnv / write_cnv round-trip
# ---------------------------------------------------------------------------


def test_read_cnv_returns_instance():
    """read_cnv returns a ConvergenceHistory instance."""
    hist = ConvergenceHistory.read_cnv(_DATA / "duct.cnv")
    assert isinstance(hist, ConvergenceHistory)


def test_read_cnv_roundtrip_steps():
    """Round-trip write_cnv/read_cnv preserves step count and values."""
    orig = ConvergenceHistory.read_cnv(_DATA / "duct.cnv")
    with tempfile.NamedTemporaryFile(suffix=".cnv") as f:
        orig.write_cnv(f.name)
        reloaded = ConvergenceHistory.read_cnv(f.name)
    n = orig.i_log + 1
    assert reloaded.i_log + 1 == n
    assert np.array_equal(reloaded.i_step[:n], orig.i_step[:n])


def test_read_cnv_roundtrip_mdot():
    """Round-trip preserves mdot arrays."""
    orig = ConvergenceHistory.read_cnv(_DATA / "duct.cnv")
    with tempfile.NamedTemporaryFile(suffix=".cnv") as f:
        orig.write_cnv(f.name)
        reloaded = ConvergenceHistory.read_cnv(f.name)
    n = orig.i_log + 1
    assert np.array_equal(reloaded.mdot_in[:n], orig.mdot_in[:n])
    assert np.array_equal(reloaded.mdot_out[:n], orig.mdot_out[:n])


def test_read_cnv_roundtrip_residual():
    """Round-trip preserves drho residual."""
    orig = ConvergenceHistory.read_cnv(_DATA / "duct.cnv")
    with tempfile.NamedTemporaryFile(suffix=".cnv") as f:
        orig.write_cnv(f.name)
        reloaded = ConvergenceHistory.read_cnv(f.name)
    n = orig.i_log + 1
    assert np.array_equal(reloaded.residual[:n, 0], orig.residual[:n, 0])


def test_read_cnv_roundtrip_metadata():
    """Round-trip preserves grid metadata (node count, areas)."""
    orig = ConvergenceHistory.read_cnv(_DATA / "duct.cnv")
    with tempfile.NamedTemporaryFile(suffix=".cnv") as f:
        orig.write_cnv(f.name)
        reloaded = ConvergenceHistory.read_cnv(f.name)
    assert reloaded._get_metadata_by_key("n_node") == orig._get_metadata_by_key(
        "n_node"
    )
    assert reloaded._get_metadata_by_key("A_in") == orig._get_metadata_by_key("A_in")


def test_read_cnv_compressed_roundtrip():
    """Compressed write_cnv round-trip is readable by read_cnv."""
    orig = ConvergenceHistory.read_cnv(_DATA / "duct.cnv")
    with tempfile.NamedTemporaryFile(suffix=".cnv.gz") as f:
        orig.write_cnv(f.name, compress=True)
        reloaded = ConvergenceHistory.read_cnv(f.name)
    assert reloaded.i_log == orig.i_log


# ---------------------------------------------------------------------------
# to_json
# ---------------------------------------------------------------------------


def test_to_json_creates_files(hist_with_throttle, tmp_path):
    """to_json writes the three expected JSON files."""

    hist = hist_with_throttle
    hist.record_convergence(
        ConvergenceStep(
            residual=np.full(5, 1e-3, dtype=np.float32),
            mdot=np.array([1.0, 1.0], dtype=np.float32),
            ho=np.array([300000.0, 300000.0], dtype=np.float32),
            s=np.array([1000.0, 1010.0], dtype=np.float32),
        )
    )
    hist.to_json(str(tmp_path))
    for name in ("convergence_err_mdot", "convergence_work", "convergence_loss"):
        assert (tmp_path / f"{name}.json").exists(), f"{name}.json not created"


def test_to_json_format(hist_with_throttle, tmp_path):
    """Each JSON file contains a list of {'x': ..., 'y': ...} dicts."""
    import json

    hist = hist_with_throttle
    hist.record_convergence(
        ConvergenceStep(
            residual=np.full(5, 1e-3, dtype=np.float32),
            mdot=np.array([1.0, 1.0], dtype=np.float32),
            ho=np.array([300000.0, 300000.0], dtype=np.float32),
            s=np.array([1000.0, 1010.0], dtype=np.float32),
        )
    )
    hist.to_json(str(tmp_path))
    n = hist.i_log + 1
    for name in ("convergence_err_mdot", "convergence_work", "convergence_loss"):
        rows = json.loads((tmp_path / f"{name}.json").read_text())
        assert isinstance(rows, list)
        assert len(rows) == n
        assert all("x" in r and "y" in r for r in rows)


def test_to_json_x_matches_steps(hist_with_throttle, tmp_path):
    """x values in convergence_err_mdot.json match recorded i_step values."""
    import json

    hist = hist_with_throttle
    for i in range(1, 4):
        hist.record_step(i * 10)
    hist.record_convergence(
        ConvergenceStep(
            residual=np.full(5, 1e-3, dtype=np.float32),
            mdot=np.array([1.0, 1.0], dtype=np.float32),
            ho=np.array([300000.0, 300000.0], dtype=np.float32),
            s=np.array([1000.0, 1010.0], dtype=np.float32),
        )
    )
    hist.to_json(str(tmp_path))
    n = hist.i_log + 1
    rows = json.loads((tmp_path / "convergence_err_mdot.json").read_text())
    x_vals = [r["x"] for r in rows]
    expected = [float(hist.i_step[i]) for i in range(n)]
    assert x_vals == expected


# ---------------------------------------------------------------------------
# err_mdot_row
# ---------------------------------------------------------------------------


def _make_hist_with_row_flows(n_row, steps=3):
    """ConvergenceHistory with n_row rows and row flow data recorded."""
    hist = ConvergenceHistory(shape=(10,))
    hist._set_metadata_by_key("n_block", 1)
    hist._set_metadata_by_key("n_node", 100)
    hist._set_metadata_by_key("A_in", np.float32(0.5))
    hist._set_metadata_by_key("A_out", np.float32(0.5))
    hist._set_metadata_by_key("is_rotating", False)
    hist._set_metadata_by_key("fluid", _FLUID)
    hist._set_metadata_by_key("_time_start", 0.0)
    hist._set_metadata_by_key("i_log", -1)
    hist._set_metadata_by_key("n_row", n_row)
    for k in hist._data_keys:
        hist._versions[k] += 1
    for i in range(steps):
        hist.record_step(i)
        mdot_rows = [
            (10.0 + i * 0.1 * (r + 1), 9.8 + i * 0.1 * (r + 1)) for r in range(n_row)
        ]
        mdot = np.array([v for pair in mdot_rows for v in pair], dtype=np.float32)
        hist.record_convergence(
            ConvergenceStep(
                residual=np.full(5, np.nan, dtype=np.float32),
                mdot=mdot,
                ho=np.zeros(len(mdot), dtype=np.float32),
                s=np.zeros(len(mdot), dtype=np.float32),
            )
        )
    return hist


def test_err_mdot_row_shape_one_row():
    """err_mdot_row returns shape (n_log, 1) for a single-row history."""
    hist = _make_hist_with_row_flows(n_row=1, steps=3)
    err = hist.err_mdot_row
    assert err.shape == (3, 1)


def test_err_mdot_row_shape_two_rows():
    """err_mdot_row returns shape (n_log, 2) for a two-row history."""
    hist = _make_hist_with_row_flows(n_row=2, steps=4)
    err = hist.err_mdot_row
    assert err.shape == (4, 2)


def test_err_mdot_row_values():
    """err_mdot_row computes (dn - up) / avg correctly."""
    hist = ConvergenceHistory(shape=(10,))
    hist._set_metadata_by_key("n_block", 1)
    hist._set_metadata_by_key("n_node", 100)
    hist._set_metadata_by_key("A_in", np.float32(0.5))
    hist._set_metadata_by_key("A_out", np.float32(0.5))
    hist._set_metadata_by_key("is_rotating", False)
    hist._set_metadata_by_key("fluid", _FLUID)
    hist._set_metadata_by_key("_time_start", 0.0)
    hist._set_metadata_by_key("i_log", -1)
    hist._set_metadata_by_key("n_row", 1)
    for k in hist._data_keys:
        hist._versions[k] += 1
    hist.record_step(0)
    hist.record_convergence(
        ConvergenceStep(
            residual=np.full(5, np.nan, dtype=np.float32),
            mdot=np.array([10.0, 9.0], dtype=np.float32),
            ho=np.zeros(2, dtype=np.float32),
            s=np.zeros(2, dtype=np.float32),
        )
    )
    err = hist.err_mdot_row
    expected = (9.0 - 10.0) / ((10.0 + 9.0) / 2.0)
    assert np.isclose(err[0, 0], expected)


def test_err_mdot_row_no_metadata_returns_nan():
    """err_mdot_row returns NaN array when n_row metadata is absent."""
    hist = ConvergenceHistory(shape=(10,))
    hist._set_metadata_by_key("n_block", 1)
    hist._set_metadata_by_key("n_node", 100)
    hist._set_metadata_by_key("A_in", np.float32(0.5))
    hist._set_metadata_by_key("A_out", np.float32(0.5))
    hist._set_metadata_by_key("is_rotating", False)
    hist._set_metadata_by_key("fluid", _FLUID)
    hist._set_metadata_by_key("_time_start", 0.0)
    hist._set_metadata_by_key("i_log", -1)
    for k in hist._data_keys:
        hist._versions[k] += 1
    hist.record_step(0)
    err = hist.err_mdot_row
    assert err.shape[1] == 0
    assert np.all(np.isnan(err))
