r"""Convergence history recorded by the time-marching loop.

This module defines :class:`ConvergenceHistory`, a 1D time series of flow
monitors that :func:`ember.solver.run` fills as it marches. It holds one
*record* per logged step, carrying the residuals, and the mass flow, stagnation
enthalpy and entropy at each station. The residuals are whole-field block
means; only the mass flow, enthalpy, and entropy are per-station quantities.

Throughout, *record* means one entry in the history, and *row* is reserved for
a blade row of the machine.

A history is *returned to you* by :func:`ember.solver.run`; it is not something
you build directly. :meth:`ConvergenceHistory.from_grid`,
:meth:`ConvergenceHistory.record_convergence` exist for the solver to call, and
are of no use outside it.

Reading a history
=================

Run a simulation, then read the monitors as plain numpy arrays::

    hist = ember.solver.run(grid, conf)

    hist.i_step            # solver step index of each record
    hist.residual[:, 4]    # energy residual, drhoe
    hist.err_mdot          # mass flow conservation error
    hist.zeta              # entropy rise, outlet minus inlet station

Every property returns a single-precision float32 array of length ``i_log +
1``, one entry per record.

The residuals, enthalpy and entropy are stored non-dimensionally on the fluid reference
scales as described in :ref:`reference-scales`.

Monitors
========

Whole-field, one value per record:

* :attr:`ConvergenceHistory.residual` -- block-mean ``|residual_nd|`` per
  conserved variable
* :attr:`ConvergenceHistory.err_mdot` -- mass flow conservation error, inlet to
  outlet
* :attr:`ConvergenceHistory.err_mdot_row` -- the same, resolved per blade row
* :attr:`ConvergenceHistory.psi` -- stagnation enthalpy rise across the machine
* :attr:`ConvergenceHistory.zeta` -- entropy rise across the machine
* :attr:`ConvergenceHistory.tpnps` -- wall-clock time per node per step

Through-flow stations. Each blade row contributes an upstream and a downstream
face, ordered inlet to outlet as ``[row0_up, row0_dn, row1_up, row1_dn]``, so
station ``0`` is the inlet and station ``-1`` the outlet. All three are
non-dimensional, hence the ``_nd`` suffix:

* :attr:`ConvergenceHistory.mdot_nd`, :attr:`ConvergenceHistory.ho_nd`,
  :attr:`ConvergenceHistory.s_nd`

Throttle state, when an outlet is running a PID throttle. Unlike the stations
above, these are dimensional:

* :attr:`ConvergenceHistory.throttle`, :attr:`ConvergenceHistory.mdot_target`,
  :attr:`ConvergenceHistory.mdot_throttle`
* :attr:`ConvergenceHistory.dP_P`, :attr:`ConvergenceHistory.dP_I`,
  :attr:`ConvergenceHistory.dP_D`

Bookkeeping:

* :attr:`ConvergenceHistory.i_step` -- solver step index of each record
* :attr:`ConvergenceHistory.time` -- elapsed wall-clock time
* :attr:`ConvergenceHistory.i_log` -- index of the last written record
* :attr:`ConvergenceHistory.diverged` -- True if the march blew up
* :attr:`ConvergenceHistory.now` -- view of the record currently being written
* :attr:`ConvergenceHistory.n_node` -- node count of the grid behind the history

Storage
=======

:meth:`ConvergenceHistory.from_grid` allocates ``n_log`` records up front and
fills them with NaN, one per log step; the solver passes
``n_log = ceil(n_step / n_step_log)`` for ``n_step`` marched steps logged every
``n_step_log``. At each log step,
:func:`ember.solver.run` calls :meth:`ConvergenceHistory.record_convergence`
to fill one record. A march that runs to completion fills every record; one that
diverges breaks from the loop early before logging an invalid value, sets
:attr:`ConvergenceHistory.diverged`, and trims the history to drop the
preallocated NaN for steps that were never reached. This means that we can
always assume that history arrays are finite and valid for plotting or
reduction, without having to mask out NaN values.

Reading from a file
===================

A history is read back from disk with :meth:`ConvergenceHistory.read_cnv`, which
unpickles a history written by :meth:`ConvergenceHistory.write_cnv`.

"""

import gzip
import json
import pickle

import numpy as np
import time as _time
from ember.struct import StructuredData

f32 = np.float32


class ConvergenceHistory(StructuredData):
    # No class docstring: docs/conf.py sets autoclass_content = "init", so
    # Sphinx renders the inherited StructuredData.__init__ docstring in its
    # place and a class docstring here would never appear. The prose lives in
    # the module docstring above, as it does for Block.

    _TIME_SCALE = 1e-3  # seconds per stored unit (i.e. milliseconds)

    # A history built by any route other than a completed solver run has not
    # observed a divergence, so `diverged` reads False without being set.
    _defaults = {"diverged": False}

    # Station ordering and reference scales are documented in the module
    # docstring. Width 4 covers n_row <= 2.
    _data_keys = (
        "mdot_st0",
        "mdot_st1",
        "mdot_st2",
        "mdot_st3",
        "ho_st0",
        "ho_st1",
        "ho_st2",
        "ho_st3",
        "s_st0",
        "s_st1",
        "s_st2",
        "s_st3",
        "drho",
        "drhoVx",
        "drhoVr",
        "drhorVt",
        "drhoe",
        "i_step",
        "time",
        "mdot_target",
        "mdot_throttle",
        "P_throttle",
        "dP_P",
        "dP_I",
        "dP_D",
    )

    def __post_init__(self):
        super().__post_init__()
        # Mark all data keys as initialized (we write to slices, not whole arrays)
        for k in self._data_keys:
            self._versions[k] += 1

    @property
    def _n_station(self):
        """Number of through-flow stations, ``2 * n_row`` (defaults to 2)."""
        return 2 * (self._metadata.get("n_row") or 1)

    def _station_array(self, prefix):
        """Stack the per-station data keys ``<prefix>0..N-1`` on the last axis."""
        return self._get_data_by_keys(
            tuple(f"{prefix}{i}" for i in range(self._n_station))
        )

    @staticmethod
    def _set_grid_metadata(out, grid):
        """Set the node count on `out` from a grid.

        Shared by from_grid and other readers (e.g. the ember-cfd-ts log
        parser); callers set `fluid` and any per-step data themselves.

        The convergence monitors are non-dimensionalised entirely by the fluid
        reference scales (carried on `fluid`), so no separate kinetic-energy
        reference velocity/temperature is stored here.
        """
        out._set_metadata_by_key("n_node", grid.size)

    @classmethod
    def from_grid(cls, n_log, grid):
        """Initialize an empty convergence history sized for a solver run.

        Parameters
        ----------
        n_log : int
            Number of records to allocate; one is filled per log step. The
            solver derives this as ``ceil(n_step / n_step_log)`` -- floor
            division would under-allocate when ``n_step`` is not a multiple of
            ``n_step_log``.
        grid : Grid
            Grid object containing blocks with patches

        Returns
        -------
        ConvergenceHistory
            Configured instance ready to record data
        """
        out = cls(shape=(n_log,))

        # Reference scales and counts derived from the grid geometry/flow.
        cls._set_grid_metadata(out, grid)

        # Fluid from the first outlet patch (the grid is the reference frame the
        # history is projected onto).
        outlet_block = grid.patches.outlet[0].block_view
        out._set_metadata_by_key("fluid", outlet_block.fluid)

        # Initialize timer reference
        out._set_metadata_by_key(
            "_time_start", np.array(_time.perf_counter(), dtype=np.float64)
        )

        # Initialize log index to -1 (incremented to 0 by the first record_convergence)
        out._set_metadata_by_key("i_log", -1)

        rows = grid.rows
        n_row = len(rows)
        if n_row > 2:
            raise NotImplementedError(
                f"Per-row mass flow tracking supports n_row <= 2, got {n_row}"
            )
        out._set_metadata_by_key("n_row", n_row)

        return out

    @classmethod
    def read_cnv(cls, filename):
        """Read convergence history from CNV binary format file.

        Automatically detects gzip-compressed files.

        Parameters
        ----------
        filename : str
            Input CNV file to read

        Returns
        -------
        ConvergenceHistory
        """
        try:
            with gzip.open(filename, "rb") as f:
                return pickle.load(f)
        except OSError:
            with open(filename, "rb") as f:
                return pickle.load(f)

    def format_message(self, n_step=None):
        """Format convergence message for current log step.

        Parameters
        ----------
        n_step : int, optional
            Total steps in this march. When given, a timing line (tpnps,
            elapsed, estimated remaining) is inserted after the step header.

        Returns
        -------
        str
            Formatted convergence status message
        """
        now = self.now
        i_step = int(now.i_step)

        out = f"Step {i_step:4d}:\n"

        if n_step is not None:
            out += self.format_timing(i_step, n_step) + "\n"

        # Second line: stagnation conditions. ho/s are stored non-dimensional
        # (by u_ref / Rgas_ref), which is exactly what set_h_s expects.
        fluid = self._get_metadata_by_key("fluid")
        ho, s = now.ho_nd, now.s_nd
        rho_o_in, u_o_in = fluid.set_h_s(ho[..., 0], s[..., 0])
        rho_o_out, u_o_out = fluid.set_h_s(ho[..., -1], s[..., -1])
        To_in = fluid.get_T(rho_o_in, u_o_in) * fluid.T_ref
        To_out = fluid.get_T(rho_o_out, u_o_out) * fluid.T_ref
        Po_in = fluid.get_P(rho_o_in, u_o_in) * fluid.P_ref
        Po_out = fluid.get_P(rho_o_out, u_o_out) * fluid.P_ref
        out += "  In/Out:"
        out += f"  To={To_in:.1f}/{To_out:.1f} K"
        out += f"  Po={Po_in / 1e3:.3f}/{Po_out / 1e3:.3f} kPa\n"

        # Throttle line (only when active)
        if float(now.mdot_target) > 0:
            err = (float(now.mdot_throttle) - float(now.mdot_target)) / float(
                now.mdot_target
            )
            out += (
                f"  Throt :"
                f"  mdot={float(now.mdot_throttle):.4f}/{float(now.mdot_target):.4f} kg/s"
                f"  err={err:+.3f}"
                f"  dP={float(now.dP_P):+.1f}/{float(now.dP_I):+.1f}/{float(now.dP_D):+.1f}\n"
            )

        # Third line: non-dimensional metrics
        out += "  m ho s:"
        out += f"  ε={now.err_mdot:<6.4f}"
        out += f"  ψ={now.psi:6.4f}"
        out += f"  ζ={now.zeta:6.4f}\n"

        # Per-row and mix-plane mdot errors (only when n_row metadata present)
        n_row = self._metadata.get("n_row")
        if n_row:
            mdot = [
                float(now._get_data_by_keys((f"mdot_st{i}",))) for i in range(2 * n_row)
            ]

            def _err(a, b):
                return (b - a) / (0.5 * (a + b))

            row_errs = [_err(mdot[2 * r], mdot[2 * r + 1]) for r in range(n_row)]
            out += "  Rows  :"
            for r, e in enumerate(row_errs):
                out += f"  row{r} ε={e:+.4f}"
            if n_row > 1:
                mix_err = _err(mdot[1], mdot[2])
                out += f"  mix ε={mix_err:+.4f}"
            out += "\n"

        # Residuals line
        res = now.residual
        out += f"  Resid : {res[0]:9.2e} {res[1]:9.2e} {res[2]:9.2e} {res[3]:9.2e} {res[4]:9.2e}\n"

        # Drop the trailing newline so callers (logger) own line separation.
        return out.rstrip("\n")

    def format_timing(self, i_step, n_step):
        """Format timing line: tpnps, elapsed, and estimated remaining.

        Parameters
        ----------
        i_step : int
            Current step index within this march.
        n_step : int
            Total steps in this march.
        """
        # tpnps already divides wall time by this history's own grid node count
        # (from_grid stores n_node = grid.size), so it is the true per-node cost
        # of the grid being marched. Under FMG each level keeps its own history,
        # so no cross-level rescaling is needed here.
        tpnps = self.tpnps
        if np.isnan(tpnps):
            return "  Timing: insufficient data"

        elapsed_ms = float(self.now.time) * self._TIME_SCALE * 1e3
        steps_left = n_step - i_step - 1
        remaining_ms = steps_left * tpnps * self.n_node / 1e3
        elapsed_min = elapsed_ms / 60e3

        return (
            f"  Timing:"
            f"  tpnps={tpnps:.3f} µs"
            f"  Elapsed/Remaining={elapsed_min:.1f}/{remaining_ms / 60e3:.1f} min"
        )

    def record_convergence(self, i_step, conv, time=None):
        """Append one fully populated record, holding solver step ``i_step``.

        Advances :attr:`i_log` onto the next allocated record and writes every
        column of it: the step index, the time, and the monitors carried by
        ``conv``. A record is never left half-written, so the only NaN a reader
        can meet is an untouched record past ``i_log``.

        Parameters
        ----------
        i_step : int
            Index of the solver step being recorded.
        conv : ember.grid.ConvergenceStep
            One step's monitors, from :meth:`ember.grid.Grid.get_convergence`.
            The ``mdot``, ``ho`` and ``s`` station vectors are unpacked into one
            scalar column per station, so at most 4 stations (``n_row <= 2``)
            can be recorded.
        time : float, optional
            Elapsed time for this record, in seconds. Defaults to the wall-clock
            time since the history was created (the live-solver behaviour); a
            reader replaying a log passes the log's own time instead.
        """
        n = len(conv.mdot)
        if n > 4:
            raise NotImplementedError(
                f"Station tracking supports n_row <= 2 (<= 4 stations), got {n}"
            )

        # Advance onto the next allocated record, then fill every column of it.
        self._set_metadata_by_key("i_log", self._get_metadata_by_key("i_log") + 1)
        now = self.now

        now._set_data_by_keys(("i_step",), i_step)
        if time is None:
            t_start = self._get_metadata_by_key("_time_start")  # float64 array
            time = np.float64(_time.perf_counter()) - t_start  # elapsed s, in f64
        now._set_data_by_keys(("time",), f32(time / self._TIME_SCALE))

        now._set_data_by_keys(
            ("drho", "drhoVx", "drhoVr", "drhorVt", "drhoe"), conv.residual
        )
        now._set_data_by_keys(tuple(f"mdot_st{i}" for i in range(n)), conv.mdot)
        now._set_data_by_keys(tuple(f"ho_st{i}" for i in range(n)), conv.ho)
        now._set_data_by_keys(tuple(f"s_st{i}" for i in range(n)), conv.s)
        now._set_data_by_keys(("mdot_target",), conv.mdot_target)
        now._set_data_by_keys(("mdot_throttle",), conv.mdot_throttle)
        now._set_data_by_keys(("P_throttle",), conv.P_throttle)
        now._set_data_by_keys(("dP_P",), conv.dP_P)
        now._set_data_by_keys(("dP_I",), conv.dP_I)
        now._set_data_by_keys(("dP_D",), conv.dP_D)

    def check_convergence(self, decay=0.0, slope=0.0, cfl=1.0):
        r"""True when every enabled convergence criterion is met.

        Three independent signals reduce the history to a single verdict, and
        the result is their logical AND. Each criterion is disabled by passing
        its no-op threshold, so a bare :meth:`check_convergence` checks
        divergence alone.

        * **Divergence** reads :attr:`diverged` only; it never touches the
          residual. A diverged march is never converged.
        * **Decay** and **slope** read the energy residual ``drhoe`` (column 4
          of :attr:`residual`), the strictest conserved-variable residual and
          the one that lags in a stalled march, over the ``i_log + 1`` written
          records.

        Parameters
        ----------
        decay : float, optional
            Required fall of the residual from its peak over the whole march, in
            decades: converged needs ``log10(r.max() / r[-1]) >= decay``. The
            default ``0`` disables the check (0 decades of fall is always met).
        slope : float, optional
            Maximum allowed magnitude of the residual slope, in decades of
            residual per unit pseudo-time, where pseudo-time is ``i_step * cfl``.
            Fitted over the last 20% of records so it reflects the recent tail
            rather than the startup transient. Converged needs
            ``abs(d log10(r) / d(i_step * cfl)) <= slope``. The default ``0``
            disables the check.
        cfl : float, optional
            CFL number used to march, scaling the pseudo-time step so the slope
            is comparable across runs with different step sizes. Only affects
            the ``slope`` criterion.

        Returns
        -------
        bool
        """
        # Divergence: always checked, cannot be disabled. Reads the flag only.
        if self.diverged:
            return False

        n = self.i_log + 1
        r = self.residual[:n, 4]  # energy residual, drhoe

        # Decay: decades fallen from the peak residual over the whole calc.
        if decay > 0.0 and np.log10(r.max() / r[-1]) < decay:
            return False

        # Slope: decades of residual per unit pseudo-time (i_step scaled by
        # cfl), fitted over the final fifth of the march.
        if slope > 0.0:
            if n < 2:
                return False  # need >= 2 records to fit a line
            n_fit = max(2, -(-n // 5))  # ceil(n / 5), at least 2 records
            t = self.i_step[n - n_fit : n] * cfl
            m = np.polyfit(t, np.log10(r[n - n_fit :]), 1)[0]
            if abs(m) > slope:
                return False

        return True

    def find_settling_record(self, tol=0.01):
        r"""Record index at which the entropy rise :attr:`zeta` has settled.

        Complements :meth:`check_convergence`: where that reduces the *residual*
        history to a converged/not verdict, this locates when the *solution
        output* stopped moving. It returns the record index at which
        :attr:`zeta` has come within a fraction ``tol`` of its total change and
        stays there for the rest of the march.

        The asymptote is estimated as the mean of ``zeta`` over the final fifth
        of records (the same last-20% window :meth:`check_convergence` fits its
        slope over), and the band is ``tol`` of the total swing
        ``abs(zeta[0] - target)`` about it. The settling record is the one
        *after* the last record still outside that band -- keying on the last
        exit rather than the first entry makes it robust to any overshoot that
        dips through the band and back out. Because the asymptote is the tail
        mean, the result is only physically meaningful once the march has
        actually levelled out (e.g. a run :meth:`check_convergence` accepts).

        The return is a *record index* into the history arrays, not a solver
        step number, so both the step it settled at and the wall-clock time to
        get there are one indexing away::

            idx  = hist.find_settling_record()
            step = int(hist.i_step[idx])                  # solver step
            wall = float(hist.time[idx] - hist.time[0])   # ms to settle

        (``hist.time[0]`` is the one-iteration startup offset, not zero, so
        subtract it for elapsed march time.)

        Parameters
        ----------
        tol : float, optional
            Settling band half-width, as a fraction of the total ``zeta`` swing.
            Default ``0.01`` (1%).

        Returns
        -------
        int
            Record index of the settling point. Falls back to ``0`` when
            ``zeta`` is within the band from the start, or is flat (zero swing).
        """
        n = self.i_log + 1
        zeta = self.zeta[:n]
        n_tail = max(1, -(-n // 5))  # ceil(n / 5), the final fifth, at least 1
        target = zeta[n - n_tail :].mean()  # asymptote estimate
        band = tol * abs(zeta[0] - target)  # tol of the total swing
        outside = np.abs(zeta - target) > band
        if band > 0 and outside.any():
            # Record after the last one still outside; clamp in case the tail
            # itself never settles (then report the last record).
            return int(min(np.nonzero(outside)[0][-1] + 1, n - 1))
        return 0

    def to_json(self, directory="."):
        """Write convergence history to three JSON files in directory.

        Writes err_mdot.json, work.json, and convergence_loss.json, each
        containing a list of {"x": i_step, "y": value} objects.

        Parameters
        ----------
        directory : str or path-like, optional
            Output directory (default current directory).
        """
        import os

        n = self.i_log + 1
        x = [float(self.i_step[i]) for i in range(n)]
        series = {
            "convergence_err_mdot": [float(self.err_mdot[i]) for i in range(n)],
            "convergence_work": [float(self.psi[i]) for i in range(n)],
            "convergence_loss": [float(self.zeta[i]) for i in range(n)],
        }
        for name, y in series.items():
            points = [{"x": xi, "y": yi} for xi, yi in zip(x, y)]
            with open(os.path.join(directory, f"{name}.json"), "w") as f:
                json.dump(points, f)

    def trim(self):
        """Copy of the records actually written, dropping the unfilled ones.

        :meth:`from_grid` allocates enough records for the requested step count
        and leaves them NaN until :meth:`record_convergence` fills them, so a
        march that broke early on divergence leaves a NaN tail past ``i_log``.
        The result holds ``i_log + 1`` records, the only length at which
        ``i_log`` stays consistent with the number of records, and it can be
        plotted or reduced without masking the tail out first.

        The copy is independent of the original, which is also what makes the
        result safe to keep once the full allocation is dropped. It has no
        spare records, so :meth:`record_convergence` cannot be called on it:
        trim once the march is over.

        Returns
        -------
        ConvergenceHistory
            A new history containing only the logged steps.
        """
        return self[: self.i_log + 1].copy()

    def write_cnv(self, filename, compress=False):
        """Write convergence history to CNV binary format file.

        Parameters
        ----------
        filename : str
            Output filename
        compress : bool, optional
            If True, compress using gzip (default False)
        """
        opener = gzip.open if compress else open
        with opener(filename, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @property
    def dP_D(self):
        r"""Derivative term of the throttle PID correction [Pa], record array.

        The three terms :attr:`dP_P`, :attr:`dP_I` and :attr:`dP_D` sum to the
        total correction :math:`\Delta p_\mathrm{throttle}` that the outlet adds
        to its base static pressure. See
        :meth:`ember.outlet.OutletPatch.get_throttle_stats`.
        """
        return self._get_data_by_keys(("dP_D",))

    @property
    def dP_I(self):
        r"""Integral term of the throttle PID correction [Pa], record array."""
        return self._get_data_by_keys(("dP_I",))

    @property
    def dP_P(self):
        r"""Proportional term of the throttle PID correction [Pa], record array."""
        return self._get_data_by_keys(("dP_P",))

    @property
    def diverged(self):
        """True if the run that produced this history blew up, scalar.

        Set by :func:`ember.solver.run` when :meth:`ember.grid.Grid.check_nan`
        raises :class:`ember.grid.DivergenceError`, in which case the step loop
        broke early and only ``i_log + 1`` records were written.
        """
        return self._get_metadata_by_key("diverged")

    @diverged.setter
    def diverged(self, value):
        self._set_metadata_by_key("diverged", bool(value))

    @property
    def err_mdot(self):
        r"""Mass flow conservation error
        :math:`(\dot m_\mathrm{out} - \dot m_\mathrm{in}) / \bar{\dot m}` [-],
        record array.

        Taken between the first and last station, so it spans the whole machine.
        Zero for a perfectly converged march; the sign says whether the outlet
        passes more or less than the inlet.
        """
        mdot = self.mdot_nd
        mdot_in, mdot_out = mdot[..., 0], mdot[..., -1]
        return (mdot_out - mdot_in) / ((mdot_in + mdot_out) / 2)

    @property
    def err_mdot_row(self):
        r"""Mass flow conservation error per blade row [-], shape ``(n_log, n_row)``.

        As :attr:`err_mdot`, but taken across each blade row separately:
        ``err[i, r] = (mdot_dn_r - mdot_up_r) / mdot_avg_r`` for record ``i``
        and row ``r``.

        Returns an empty NaN array if the ``n_row`` metadata is absent, as it is
        for histories written before that key existed.
        """
        n_row = self._metadata.get("n_row")
        if n_row is None:
            return np.full((self.i_log + 1, 0), np.nan)
        n = self.i_log + 1
        out = np.empty((n, n_row))
        for r in range(n_row):
            up = self._get_data_by_keys((f"mdot_st{2 * r}",))[:n]
            dn = self._get_data_by_keys((f"mdot_st{2 * r + 1}",))[:n]
            avg = (up + dn) / 2.0
            out[:, r] = (dn - up) / avg
        return out

    @property
    def ho_nd(self):
        r"""Stagnation enthalpy :math:`h_0/u_\mathrm{ref}` [-] at each station.

        Record array of shape ``(n_log, 2*n_row)``; station ``0`` is the inlet
        and ``-1`` the outlet. Carries an offset dependent
        on the arbitrary datum where :math:`u = s = 0` at
        :math:`(p_\mathrm{dtm}, T_\mathrm{dtm})`; only changes are physically
        meaningful, which is why :attr:`psi` is a difference between stations.
        See :ref:`datum-state`.
        """
        return self._station_array("ho_st")

    @property
    def i_log(self):
        """Index of the last written record; ``-1`` before any is written, scalar.

        A trimmed history has ``i_log + 1`` records, so ``i_log`` is also its
        last valid index.
        """
        return self._get_metadata_by_key("i_log")

    @property
    def i_step(self):
        r"""Solver step index :math:`i_\mathrm{step}` [-], record array.

        The step the march had reached when each record was written, so entries
        advance by ``n_step_log``, not by one.
        """
        return self._get_data_by_keys(("i_step",))

    @property
    def mdot_nd(self):
        r"""Mass flow :math:`\dot m` [-] at each station, non-dimensional.

        Record array of shape ``(n_log, 2*n_row)``; station ``0`` is the inlet
        and ``-1`` the outlet. Scaled by the fluid mass-flux scale. Not to be
        confused with :attr:`mdot_target` and :attr:`mdot_throttle`, which are
        dimensional [kg/s].
        """
        return self._station_array("mdot_st")

    @property
    def mdot_target(self):
        r"""Throttle mass flow setpoint :math:`\dot m_\mathrm{target}` [kg/s], record array.

        Zero when no outlet is running a throttle.
        """
        return self._get_data_by_keys(("mdot_target",))

    @property
    def mdot_throttle(self):
        r"""Mass flow measured at the throttled outlet :math:`\dot m` [kg/s], record array."""
        return self._get_data_by_keys(("mdot_throttle",))

    @property
    def n_node(self):
        """Node count of the grid that produced this history [-], scalar."""
        return self._get_metadata_by_key("n_node")

    @property
    def now(self):
        """View of the record at :attr:`i_log`, a single-record history.

        During a march this is the record being written. Once
        :func:`ember.solver.run` has trimmed the history it is the last one.
        """
        return self[self.i_log]

    @property
    def psi(self):
        r"""Stagnation enthalpy rise :math:`\psi` [-] across the machine, record array.

        Outlet station minus inlet station. Both terms are already non-dimensional (scaled by ``u_ref``), so this is
        the inlet-to-outlet stagnation enthalpy change on the fluid reference
        scale -- no separate kinetic-energy normalisation.
        """
        ho = self.ho_nd
        return ho[..., -1] - ho[..., 0]

    @property
    def residual(self):
        r"""Residual of each conserved variable [-], shape ``(n_log, 5)``.

        Ordered ``(drho, drhoVx, drhoVr, drhorVt, drhoe)``. Each entry is a block mean of :math:`|\mathtt{residual\_nd}|` for one
        conserved variable (see :meth:`ember.grid.Grid.get_convergence`), so
        the values are non-negative by construction.

        In a real march they are also strictly positive, and callers may plot
        them on a log axis without masking. A mean of non-negative floats is
        zero only if every cell residual is exactly zero, since the sum is
        never smaller than its largest term and so cannot underflow. Float32
        round-off in the flux balance and boundary conditions keeps a converged
        residual at a floor many orders of magnitude above the smallest normal
        float, rather than descending to zero: a uniform inviscid duct marched
        to a standstill still reports ``~2.6e-6``.

        The exception is a history rebuilt from an external solver log (e.g. by
        the ember-cfd-ts parser), which may recover only the density residual and
        leave the other four NaN.
        """
        return self._get_data_by_keys(("drho", "drhoVx", "drhoVr", "drhorVt", "drhoe"))

    @property
    def s_nd(self):
        r"""Specific entropy :math:`s/R_\mathrm{ref}` [-] at each station.

        Record array of shape ``(n_log, 2*n_row)``; station ``0`` is the inlet
        and ``-1`` the outlet. Carries the same
        arbitrary datum offset as :attr:`ho_nd`, so only changes between
        stations are meaningful. See :ref:`datum-state`.
        """
        return self._station_array("s_st")

    @property
    def throttle(self):
        r"""Throttle state ``(mdot_target, mdot_throttle, dP_throttle)``, shape ``(n_log, 3)``.

        The first two are mass flows [kg/s]; the third is the total PID pressure
        correction :math:`\Delta p_\mathrm{throttle}` [Pa], the sum of
        :attr:`dP_P`, :attr:`dP_I` and :attr:`dP_D`.
        """
        return self._get_data_by_keys(("mdot_target", "mdot_throttle", "P_throttle"))

    @property
    def time(self):
        """Elapsed wall-clock time since the march began [ms], record array."""
        return self._get_data_by_keys(("time",))

    @property
    def tpnps(self):
        r"""Wall-clock time per node per step [:math:`\mu\mathrm{s}`], scalar.

        Measured over the interval between the last two records, so it is NaN
        until a second record exists.
        """
        if self.i_log < 1:
            return np.nan
        dt_s = (self.time[self.i_log] - self.time[self.i_log - 1]) * self._TIME_SCALE
        di_step = self.i_step[self.i_log] - self.i_step[self.i_log - 1]
        return dt_s / di_step / self.n_node * 1e6

    @property
    def zeta(self):
        r"""Entropy rise :math:`\zeta` [-] across the machine, record array.

        Outlet station minus inlet station. Both terms are already non-dimensional (scaled by ``Rgas_ref``), so this
        is the inlet-to-outlet entropy generation on the fluid reference scale.
        It remains positive for an irreversible process (Gouy-Stodola), but is
        no longer normalised by a reference kinetic energy.
        """
        s = self.s_nd
        return s[..., -1] - s[..., 0]
