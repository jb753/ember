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


def test_record_convergence_explicit_time(hist_with_throttle):
    """An explicit time (seconds) is stored verbatim, bypassing the wall clock."""
    hist = hist_with_throttle
    # 2.5 s at _TIME_SCALE = 1e-3 s/unit -> 2500 stored units.
    hist.record_convergence(0, _throttle_conv(1.0, 0.95, P0), time=2.5)
    assert hist.time[0] == pytest.approx(2.5 / ConvergenceHistory._TIME_SCALE)


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


def _hist_with_zeta(zeta_series, n_step_log=100):
    """A fully-logged history whose entropy rise follows ``zeta_series``.

    zeta = s_out - s_in, so hold the inlet entropy at 0 and drive the outlet
    with the requested series. i_step advances by ``n_step_log`` per record.
    """
    n = len(zeta_series)
    hist = ConvergenceHistory(shape=(n,))
    hist._set_metadata_by_key("n_node", 100)
    hist._set_metadata_by_key("fluid", _FLUID)
    hist._set_metadata_by_key("_time_start", 0.0)
    hist._set_metadata_by_key("i_log", -1)
    for k in hist._data_keys:
        hist._versions[k] += 1
    for i, z in enumerate(zeta_series):
        hist.record_convergence(
            i * n_step_log,
            ConvergenceStep(
                residual=np.full(5, 1.0, dtype=np.float32),
                mdot=np.array([1.0, 1.0], dtype=np.float32),
                ho=np.zeros(2, dtype=np.float32),
                s=np.array([0.0, z], dtype=np.float32),
            ),
        )
    return hist


def test_find_settling_record_ramp_then_plateau():
    """Settling record is the first plateau step once zeta stops climbing."""
    # Rise 0..5 then flat: tail-mean target = 5, band = 0.01*5. The last record
    # outside the band is the last "4" at index 4, so settling lands at index 5.
    hist = _hist_with_zeta([0, 1, 2, 3, 4, 5, 5, 5, 5, 5])
    idx = hist.find_settling_record()
    assert idx == 5
    assert int(hist.i_step[idx]) == 500


def test_find_settling_record_flat_returns_zero():
    """A flat zeta has zero swing (degenerate band) and settles from the start."""
    hist = _hist_with_zeta([3, 3, 3, 3, 3])
    assert hist.find_settling_record() == 0


def test_find_settling_record_overshoot_uses_last_exit():
    """An overshoot that dips into the band and back out settles at the last exit."""
    # zeta touches the target (5) at index 1, overshoots to 10 at index 2, then
    # settles. First-entry would wrongly report index 1; last-exit reports 3.
    hist = _hist_with_zeta([0, 5, 10, 5, 5, 5])
    assert hist.find_settling_record() == 3


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
