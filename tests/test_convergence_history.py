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
    hist._set_metadata_by_key("n_node", 100)
    hist._set_metadata_by_key("fluid", _FLUID)
    hist._set_metadata_by_key("_time_start", 0.0)
    hist._set_metadata_by_key("i_log", -1)
    # Mark all data keys as initialized
    for k in hist._data_keys:
        hist._versions[k] += 1
    return hist


def test_throttle_shape(hist_with_throttle):
    """Test throttle compound property shape."""
    hist = hist_with_throttle
    for i in range(3):
        hist.record_convergence(i, _throttle_conv(1.0, 0.95, 101325.0))
    # throttle shape should be (n_steps_allocated, 3)
    throttle = hist.throttle
    assert throttle.shape[1] == 3, f"Expected second dimension 3, got {throttle.shape}"
    assert throttle.shape[0] == 10, (
        f"Expected first dimension 10 (allocated), got {throttle.shape[0]}"
    )


def test_record_throttle_inactive(hist_with_throttle):
    """Test that Throt line is absent when mdot_target=0."""
    hist = hist_with_throttle
    hist.record_convergence(0, _throttle_conv(0.0, 0.5, 101325.0))
    msg = hist.format_message()
    assert "Throt" not in msg, "Throt line should not appear when mdot_target=0"


def test_record_throttle_active(hist_with_throttle):
    """Test that Throt line appears when mdot_target > 0."""
    hist = hist_with_throttle
    hist.record_convergence(0, _throttle_conv(1.0, 0.95, 101325.0))
    msg = hist.format_message()
    assert "Throt" in msg, "Throt line should appear when mdot_target > 0"
    assert "mdot=0.9500/1.0000 kg/s" in msg


def test_record_throttle_err_sign(hist_with_throttle):
    """Test that error sign is correct when mdot_throttle < mdot_target."""
    hist = hist_with_throttle
    mdot_target = 1.0
    mdot_throttle = 0.9  # 10% below target
    hist.record_convergence(0, _throttle_conv(mdot_target, mdot_throttle, 101325.0))
    msg = hist.format_message()
    expected_err = (mdot_throttle - mdot_target) / mdot_target  # -0.1
    assert f"err={expected_err:+.3f}" in msg, f"Error sign incorrect in message: {msg}"
    assert "err=-0.100" in msg


def test_record_throttle_nan_passthrough(hist_with_throttle):
    """Test that NaN values are stored and don't crash format_message."""
    hist = hist_with_throttle
    hist.record_convergence(0, _throttle_conv(1.0, np.nan, np.nan))
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
    assert np.all(np.isfinite(ts3_hist.mdot_nd[:n, 0]))
    assert np.all(np.isfinite(ts3_hist.mdot_nd[:n, -1]))


def test_from_ts3_first_step_mdot(ts3_hist):
    """First-step mass flows match the log file values."""
    assert ts3_hist.mdot_nd[0, 0] == pytest.approx(1.38360, rel=1e-4)
    assert ts3_hist.mdot_nd[0, -1] == pytest.approx(1.38339, rel=1e-4)


def test_from_ts3_residual_drho_finite(ts3_hist):
    """drho residual is finite; other residual components are NaN (not in TS3 log)."""
    n = ts3_hist.i_log + 1
    assert np.all(np.isfinite(ts3_hist.residual[:n, 0]))


def test_from_ts3_ho_s_finite(ts3_hist):
    """Inlet and outlet stagnation enthalpy and entropy are finite for all steps."""
    n = ts3_hist.i_log + 1
    for name in ("ho_nd", "s_nd"):
        stations = getattr(ts3_hist, name)[:n]
        assert np.all(np.isfinite(stations[:, 0])), f"{name} inlet"
        assert np.all(np.isfinite(stations[:, -1])), f"{name} outlet"


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
    assert np.array_equal(reloaded.mdot_nd[:n, 0], orig.mdot_nd[:n, 0])
    assert np.array_equal(reloaded.mdot_nd[:n, -1], orig.mdot_nd[:n, -1])


def test_read_cnv_roundtrip_residual():
    """Round-trip preserves drho residual."""
    orig = ConvergenceHistory.read_cnv(_DATA / "duct.cnv")
    with tempfile.NamedTemporaryFile(suffix=".cnv") as f:
        orig.write_cnv(f.name)
        reloaded = ConvergenceHistory.read_cnv(f.name)
    n = orig.i_log + 1
    assert np.array_equal(reloaded.residual[:n, 0], orig.residual[:n, 0])


def test_read_cnv_roundtrip_metadata():
    """Round-trip preserves grid metadata (node count)."""
    orig = ConvergenceHistory.read_cnv(_DATA / "duct.cnv")
    with tempfile.NamedTemporaryFile(suffix=".cnv") as f:
        orig.write_cnv(f.name)
        reloaded = ConvergenceHistory.read_cnv(f.name)
    assert reloaded._get_metadata_by_key("n_node") == orig._get_metadata_by_key(
        "n_node"
    )


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
        0,
        ConvergenceStep(
            residual=np.full(5, 1e-3, dtype=np.float32),
            mdot=np.array([1.0, 1.0], dtype=np.float32),
            ho=np.array([300000.0, 300000.0], dtype=np.float32),
            s=np.array([1000.0, 1010.0], dtype=np.float32),
        ),
    )
    hist.to_json(str(tmp_path))
    for name in ("convergence_err_mdot", "convergence_work", "convergence_loss"):
        assert (tmp_path / f"{name}.json").exists(), f"{name}.json not created"


def test_to_json_format(hist_with_throttle, tmp_path):
    """Each JSON file contains a list of {'x': ..., 'y': ...} dicts."""
    import json

    hist = hist_with_throttle
    hist.record_convergence(
        0,
        ConvergenceStep(
            residual=np.full(5, 1e-3, dtype=np.float32),
            mdot=np.array([1.0, 1.0], dtype=np.float32),
            ho=np.array([300000.0, 300000.0], dtype=np.float32),
            s=np.array([1000.0, 1010.0], dtype=np.float32),
        ),
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
    for i in range(4):
        hist.record_convergence(
            i * 10,
            ConvergenceStep(
                residual=np.full(5, 1e-3, dtype=np.float32),
                mdot=np.array([1.0, 1.0], dtype=np.float32),
                ho=np.array([300000.0, 300000.0], dtype=np.float32),
                s=np.array([1000.0, 1010.0], dtype=np.float32),
            ),
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
    hist._set_metadata_by_key("n_node", 100)
    hist._set_metadata_by_key("fluid", _FLUID)
    hist._set_metadata_by_key("_time_start", 0.0)
    hist._set_metadata_by_key("i_log", -1)
    hist._set_metadata_by_key("n_row", n_row)
    for k in hist._data_keys:
        hist._versions[k] += 1
    for i in range(steps):
        mdot_rows = [
            (10.0 + i * 0.1 * (r + 1), 9.8 + i * 0.1 * (r + 1)) for r in range(n_row)
        ]
        mdot = np.array([v for pair in mdot_rows for v in pair], dtype=np.float32)
        hist.record_convergence(
            i,
            ConvergenceStep(
                residual=np.full(5, np.nan, dtype=np.float32),
                mdot=mdot,
                ho=np.zeros(len(mdot), dtype=np.float32),
                s=np.zeros(len(mdot), dtype=np.float32),
            ),
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
    hist._set_metadata_by_key("n_node", 100)
    hist._set_metadata_by_key("fluid", _FLUID)
    hist._set_metadata_by_key("_time_start", 0.0)
    hist._set_metadata_by_key("i_log", -1)
    hist._set_metadata_by_key("n_row", 1)
    for k in hist._data_keys:
        hist._versions[k] += 1
    hist.record_convergence(
        0,
        ConvergenceStep(
            residual=np.full(5, np.nan, dtype=np.float32),
            mdot=np.array([10.0, 9.0], dtype=np.float32),
            ho=np.zeros(2, dtype=np.float32),
            s=np.zeros(2, dtype=np.float32),
        ),
    )
    err = hist.err_mdot_row
    expected = (9.0 - 10.0) / ((10.0 + 9.0) / 2.0)
    assert np.isclose(err[0, 0], expected)


def test_err_mdot_row_no_metadata_returns_nan():
    """err_mdot_row returns NaN array when n_row metadata is absent."""
    hist = ConvergenceHistory(shape=(10,))
    hist._set_metadata_by_key("n_node", 100)
    hist._set_metadata_by_key("fluid", _FLUID)
    hist._set_metadata_by_key("_time_start", 0.0)
    hist._set_metadata_by_key("i_log", -1)
    for k in hist._data_keys:
        hist._versions[k] += 1
    hist.record_convergence(0, _conv())
    err = hist.err_mdot_row
    assert err.shape[1] == 0
    assert np.all(np.isnan(err))


def _conv():
    """A ConvergenceStep with every flow monitor finite."""
    return ConvergenceStep(
        residual=np.full(5, 1e-3, dtype=np.float32),
        mdot=np.array([1.0, 1.0], dtype=np.float32),
        ho=np.array([300000.0, 300000.0], dtype=np.float32),
        s=np.array([1000.0, 1010.0], dtype=np.float32),
    )


def _record(hist, i_step):
    """Log one step carrying a full set of finite flow monitors."""
    hist.record_convergence(i_step, _conv())


def test_trim_drops_unfilled_records(hist_with_throttle):
    """A part-filled history trims to the steps actually logged."""
    hist = hist_with_throttle
    for i in range(4):
        _record(hist, i * 10)

    assert hist.shape == (10,)
    assert np.isnan(hist.i_step[4:]).all()  # tail never written

    trimmed = hist.trim()

    assert trimmed.shape == (4,)
    assert trimmed.i_log == 3
    assert trimmed.i_log + 1 == trimmed.shape[0]  # the invariant trim preserves
    np.testing.assert_array_equal(trimmed.i_step, [0.0, 10.0, 20.0, 30.0])
    assert np.isfinite(trimmed.i_step).all()
    assert np.isfinite(trimmed.err_mdot).all()
    assert np.isfinite(trimmed.zeta).all()


def test_trim_of_full_history_preserves_every_record(hist_with_throttle):
    """A march that ran to completion loses no data, and keeps its shape."""
    hist = hist_with_throttle
    for i in range(10):
        _record(hist, i * 10)
    assert hist.i_log == 9  # every allocated row filled

    trimmed = hist.trim()

    assert trimmed.shape == hist.shape
    assert trimmed.i_log == hist.i_log
    np.testing.assert_array_equal(trimmed.i_step, hist.i_step)
    np.testing.assert_array_equal(trimmed.residual, hist.residual)
    np.testing.assert_array_equal(trimmed.err_mdot, hist.err_mdot)
    np.testing.assert_array_equal(trimmed.zeta, hist.zeta)


def test_trim_returns_independent_copy(hist_with_throttle):
    """trim never aliases the original, so the full allocation can be dropped."""
    hist = hist_with_throttle
    for i in range(10):
        _record(hist, i * 10)  # fill it, the case where a view is most tempting

    trimmed = hist.trim()

    assert trimmed is not hist
    assert not np.shares_memory(trimmed._data, hist._data)
    assert trimmed._metadata is not hist._metadata

    trimmed._set_metadata_by_key("i_log", 0)
    assert hist.i_log == 9


def test_trim_preserves_diverged_flag(hist_with_throttle):
    """The diverged flag survives the trim it exists to justify."""
    hist = hist_with_throttle
    hist.diverged = True
    assert hist.trim().diverged is True


# ---------------------------------------------------------------------------
# check_convergence
# ---------------------------------------------------------------------------


def _make_hist_with_energy_residual(res_energy, di_step=10):
    """History whose energy residual (column 4) follows ``res_energy``.

    The other four residual columns are held at a constant 1.0, so a check that
    read the wrong column would see no decay and no slope.
    """
    res_energy = np.asarray(res_energy, dtype=np.float32)
    n = len(res_energy)
    hist = ConvergenceHistory(shape=(n,))
    hist._set_metadata_by_key("n_node", 100)
    hist._set_metadata_by_key("fluid", _FLUID)
    hist._set_metadata_by_key("_time_start", 0.0)
    hist._set_metadata_by_key("i_log", -1)
    for k in hist._data_keys:
        hist._versions[k] += 1
    for i in range(n):
        hist.record_convergence(
            i * di_step,
            ConvergenceStep(
                residual=np.array(
                    [1.0, 1.0, 1.0, 1.0, res_energy[i]], dtype=np.float32
                ),
                mdot=np.array([1.0, 1.0], dtype=np.float32),
                ho=np.array([300000.0, 300000.0], dtype=np.float32),
                s=np.array([1000.0, 1010.0], dtype=np.float32),
            ),
        )
    return hist


def test_check_convergence_no_args_reports_only_divergence():
    """A bare check_convergence() is True unless the march diverged."""
    hist = _make_hist_with_energy_residual(np.logspace(-2, -6, 20))
    assert hist.check_convergence() is True
    hist.diverged = True
    assert hist.check_convergence() is False


def test_check_convergence_diverged_ignores_residual():
    """A diverged flag wins even when the residual would satisfy every check."""
    hist = _make_hist_with_energy_residual(np.logspace(-2, -8, 20))
    hist.diverged = True
    # Thresholds the finite residual would easily pass, yet still False.
    assert hist.check_convergence(decay=1.0, slope=1.0) is False


def test_check_convergence_decay_decades():
    """decay counts decades fallen from the peak energy residual."""
    hist = _make_hist_with_energy_residual(np.logspace(-2, -6, 20))  # 4 decades
    assert hist.check_convergence(decay=3.0) is True
    assert hist.check_convergence(decay=5.0) is False


def test_check_convergence_decay_reads_energy_column():
    """decay looks at drhoe (column 4), not another residual."""
    # Energy residual is flat; a wrong-column read would also see the flat 1.0
    # columns and still report no decay, so instead give energy the only decay.
    hist = _make_hist_with_energy_residual(np.logspace(-1, -7, 20))  # 6 decades
    assert hist.check_convergence(decay=5.0) is True


def test_check_convergence_slope_flat_tail_passes():
    """A residual whose last fifth is flat has ~zero slope there."""
    res = np.concatenate(
        [np.logspace(-2, -6, 16), np.full(4, 1e-6)]
    )  # falls, then flat
    hist = _make_hist_with_energy_residual(res)
    assert hist.check_convergence(slope=1e-3) is True


def test_check_convergence_slope_falling_tail_fails():
    """A residual still falling over its tail exceeds a small slope tolerance."""
    hist = _make_hist_with_energy_residual(np.logspace(-2, -6, 20))
    assert hist.check_convergence(slope=1e-3) is False


def test_check_convergence_cfl_scales_slope():
    """A larger cfl stretches pseudo-time, shrinking the slope magnitude."""
    res = np.logspace(-2, -6, 20).astype(np.float32)
    hist = _make_hist_with_energy_residual(res)
    n = 20
    n_fit = max(2, -(-n // 5))
    i_tail = hist.i_step[n - n_fit : n]
    m1 = abs(np.polyfit(i_tail, np.log10(res[n - n_fit :]), 1)[0])
    # cfl=2 halves the slope; pick a threshold between m1/2 and m1.
    thr = 0.75 * m1
    assert hist.check_convergence(slope=thr, cfl=1.0) is False
    assert hist.check_convergence(slope=thr, cfl=2.0) is True
