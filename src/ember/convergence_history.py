"""Convergence history tracking for the time-stepping loop.

Stores a 1D time-series of stagnation conditions, residuals, and CFL numbers
at inlet and outlet for convergence monitoring and post-processing.
"""

import gzip
import json
import pickle
import re

import numpy as np
import time as _time
import ember.average
from ember.struct import StructuredData
from ember.fluid import PerfectFluid

f32 = np.float32


class ConvergenceHistory(StructuredData):
    """Simplified convergence history storage for flow monitoring.

    Shape is (n_step,) - a simple 1D time series.

    Stores mass flow rate and specific properties at inlet and outlet over time.
    """

    _TIME_SCALE = 1e-3  # seconds per stored unit (i.e. milliseconds)

    # A history built by any route other than a completed solver run has not
    # observed a divergence, so `diverged` reads False without being set.
    _defaults = {"diverged": False}

    # Through-flow stations: each blade row contributes an upstream and a
    # downstream face, ordered inlet->outlet as
    # [row0_up, row0_dn, row1_up, row1_dn, ...]. Width 4 covers n_row <= 2.
    # mdot/ho/s are stored *non-dimensional* (fluid reference scales, the same
    # convention as Block.residual_nd): mdot by the mass-flux scale, ho by
    # u_ref, s by Rgas_ref.
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
        "cfl_rho",
        "cfl_rhoVx",
        "cfl_rhoVr",
        "cfl_rhorVt",
        "cfl_rhoe",
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
        """Set reference scales and counts on `out` from a grid.

        Derives the metadata that depends only on grid geometry and the current
        flow field: block/node counts, inlet/outlet areas, and whether the grid
        rotates. Shared by from_grid and from_ts3; callers set `fluid` and any
        per-step data themselves.

        The convergence monitors are non-dimensionalised entirely by the fluid
        reference scales (carried on `fluid`), so no separate kinetic-energy
        reference velocity/temperature is stored here.
        """
        out._set_metadata_by_key("n_block", len(grid))
        out._set_metadata_by_key("n_node", grid.size)

        # Total inlet/outlet areas, summed over the full annulus (× Nb).
        A_in = 0.0
        A_out = 0.0
        for b in grid:
            for p in b.patches.inlet:
                A_in += (
                    np.linalg.norm(ember.average.total_area(b[p.slice].squeeze()))
                    * b.Nb
                )
            for p in b.patches.outlet:
                A_out += (
                    np.linalg.norm(ember.average.total_area(b[p.slice].squeeze()))
                    * b.Nb
                )
        out._set_metadata_by_key("A_in", f32(A_in))
        out._set_metadata_by_key("A_out", f32(A_out))

    @classmethod
    def from_grid(cls, n_step, grid):
        """Initialize convergence history from grid.

        Parameters
        ----------
        n_step : int
            Number of time steps to allocate
        grid : Grid
            Grid object containing blocks with patches

        Returns
        -------
        ConvergenceHistory
            Configured instance ready to record data
        """
        # Create instance with 1D shape
        out = cls(shape=(n_step,))

        # Reference scales and counts derived from the grid geometry/flow.
        cls._set_grid_metadata(out, grid)

        # Fluid from the first outlet patch (from_ts3 overrides this with the
        # fluid recorded in the log header instead).
        outlet_block = grid.patches.outlet[0].block_view
        out._set_metadata_by_key("fluid", outlet_block.fluid)

        # Initialize timer reference
        out._set_metadata_by_key(
            "_time_start", np.array(_time.perf_counter(), dtype=np.float64)
        )

        # Initialize log index to -1 (will be incremented to 0 on first record_step)
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
    def from_ts3(cls, filename, grid):
        """Reconstruct a ConvergenceHistory from a TS3 text log file.

        The per-step history (residuals, mass flows, stagnation conditions) is
        parsed from the log; the reference scales needed to non-dimensionalize
        it (areas, V_ref, T_ref, node count) are derived from `grid`, which the
        log does not record. The fluid is taken from the log header, the
        authoritative record of what TS3 actually ran with.

        Parameters
        ----------
        filename : str
            Path to TS3 log file (e.g. log_duct.txt)
        grid : ember.grid.Grid
            The grid that was solved, for reference scales (V_ref, T_ref, areas).

        Returns
        -------
        ConvergenceHistory
        """
        with open(filename, "r") as f:
            text = f.read()

        # --- Parse header (before main loop) ---
        header_match = re.search(
            r"APPLICATION VARIABLES:(.*?)STARTING THE MAIN TIME STEPPING LOOP",
            text,
            re.DOTALL,
        )
        if header_match is None:
            raise ValueError("Could not find APPLICATION VARIABLES header in log")
        header_text = header_match.group(1)

        def _get_var(name, txt):
            m = re.search(r"^\s+" + name + r":\s+([\d.Ee+-]+)", txt, re.MULTILINE)
            if m is None:
                raise ValueError(f"Could not find '{name}' in header")
            return float(m.group(1))

        cp = _get_var("cp", header_text)
        ga = _get_var("ga", header_text)
        mu = _get_var("viscosity", header_text)
        Pr = _get_var("prandtl", header_text)

        # Guard against a grid/log mismatch: the gas properties recorded in the
        # log header must agree with the solved grid's fluid (grid[0].cp etc.
        # are spatially constant for a perfect gas, so compare their means).
        block0 = grid[0]
        for name, log_val, grid_val in (
            ("cp", cp, block0.cp),
            ("ga", ga, block0.gamma),
            ("viscosity", mu, block0.mu),
            ("prandtl", Pr, block0.Pr),
        ):
            grid_val = float(np.mean(grid_val))
            if not np.isclose(log_val, grid_val, rtol=1e-3):
                raise ValueError(
                    f"TS3 log {name}={log_val:g} does not match grid "
                    f"{name}={grid_val:g}; wrong grid for this log?"
                )

        fluid = PerfectFluid(cp=cp, gamma=ga, mu=mu, Pr=Pr, T_dtm=1.0)

        # --- Split into per-step blocks (after main loop start) ---
        body = text[header_match.end() :]

        # Collect all timing values for mean dt estimation
        timing_vals = [
            float(v) for v in re.findall(r"TIME FOR \d+ STEPS = ([\d.]+)", body)
        ]
        mean_dt = np.mean(timing_vals) if timing_vals else 0.0

        # Parse step blocks: each starts with "STEP No. <n>"
        step_re = re.compile(r"STEP No\.\s+(\d+)")
        davg_re = re.compile(r"TOTAL DAVG\s+([\d.E+\-]+)")
        flows_re = re.compile(r"INLET FLOW =\s+([\d.]+)\s+OUTLET FLOW =\s+([\d.]+)")
        stagP_re = re.compile(
            r"AVG INLET STAG P =\s+([\d.]+)\s+AVG OUTLET STAG P =\s+([\d.]+)"
        )
        stagT_re = re.compile(
            r"AVG INLET STAG T =\s+([\d.]+)\s+AVG OUTLET STAG T =\s+([\d.]+)"
        )

        # Find positions of all "STEP No." matches
        step_starts = [m.start() for m in step_re.finditer(body)]

        step_blocks = []
        for idx, start in enumerate(step_starts):
            end = step_starts[idx + 1] if idx + 1 < len(step_starts) else len(body)
            chunk = body[start:end]

            i_step_m = step_re.match(chunk)
            davg_m = davg_re.search(chunk)
            flows_m = flows_re.search(chunk)
            stagP_m = stagP_re.search(chunk)
            stagT_m = stagT_re.search(chunk)

            if not all([i_step_m, davg_m, flows_m, stagP_m, stagT_m]):
                continue  # skip incomplete blocks

            step_blocks.append(
                {
                    "i_step": int(i_step_m.group(1)),
                    "davg": float(davg_m.group(1)),
                    "mdot_in": float(flows_m.group(1)),
                    "mdot_out": float(flows_m.group(2)),
                    "Po_in": float(stagP_m.group(1)),
                    "Po_out": float(stagP_m.group(2)),
                    "To_in": float(stagT_m.group(1)),
                    "To_out": float(stagT_m.group(2)),
                }
            )

        n_log = len(step_blocks)
        out = cls(shape=(n_log,))

        # --- Set metadata ---
        # Reference scales and counts from the grid (the log does not record
        # them); fluid from the log header (what TS3 actually ran with).
        cls._set_grid_metadata(out, grid)
        out._set_metadata_by_key("fluid", fluid)
        out._set_metadata_by_key(
            "_time_start", np.array(_time.perf_counter(), dtype=np.float64)
        )
        out._set_metadata_by_key("i_log", n_log - 1)

        # The log records only overall inlet/outlet flows, so map them to the
        # first and last through-flow stations; any interior stations stay NaN.
        n_row = len(grid.rows)
        out._set_metadata_by_key("n_row", n_row)
        st_in, st_out = 0, 2 * n_row - 1
        # mdot is non-dimensionalised by the fluid mass-flux scale, matching the
        # live grid path (Grid._station_stats); ho/s from get_h/get_s on nd
        # inputs are already non-dimensional.
        mdot_ref = fluid.rhoV_ref * grid[0].L_ref ** 2

        # NaN arrays for residuals/CFLs we cannot recover
        nan5 = np.full(5, np.nan, dtype=f32)

        for i, blk in enumerate(step_blocks):
            # Navigate to this index via a temporary slice view
            view = out[i]

            view._set_data_by_keys(("i_step",), blk["i_step"])
            view._set_data_by_keys(("time",), f32(i * mean_dt / cls._TIME_SCALE))

            # Convert dimensional Po, To → non-dimensional ho, s
            for side, st in (("in", st_in), ("out", st_out)):
                Po = blk[f"Po_{side}"]
                To = blk[f"To_{side}"]
                rho_nd, u_nd = fluid.set_P_T(Po / fluid.P_ref, To / fluid.T_ref)
                ho_nd = fluid.get_h(rho_nd, u_nd)
                s_nd = fluid.get_s(rho_nd, u_nd)
                view._set_data_by_keys((f"ho_st{st}",), f32(ho_nd))
                view._set_data_by_keys((f"s_st{st}",), f32(s_nd))
                view._set_data_by_keys(
                    (f"mdot_st{st}",), f32(blk[f"mdot_{side}"] / mdot_ref)
                )

            # DAVG → drho; other residuals NaN
            view._set_data_by_keys(
                ("drho", "drhoVx", "drhoVr", "drhorVt", "drhoe"),
                np.array([blk["davg"], np.nan, np.nan, np.nan, np.nan], dtype=f32),
            )
            view._set_data_by_keys(
                ("cfl_rho", "cfl_rhoVx", "cfl_rhoVr", "cfl_rhorVt", "cfl_rhoe"),
                nan5,
            )

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

    def format_message(self, i_finest=None, n_step=None, n_levels=None, show_cfl=True):
        """Format convergence message for current log step.

        Parameters
        ----------
        i_finest, n_step, n_levels : int, optional
            When provided, a timing line is inserted after the step header.
        show_cfl : bool, optional
            Append the per-equation CFL line (default True). Set False for
            fixed-CFL marches (e.g. the explicit solver loop) that never populate
            ``now.cfl``.

        Returns
        -------
        str
            Formatted convergence status message
        """
        now = self.now
        i_step = int(now.i_step)

        level_str = f" Level {i_finest}" if i_finest is not None else ""
        out = f"Step {i_step:4d}{level_str}:\n"

        if i_finest is not None and n_step is not None and n_levels is not None:
            out += self.format_timing(i_step, i_finest, n_step, n_levels) + "\n"

        # Second line: stagnation conditions. ho/s are stored non-dimensional
        # (by u_ref / Rgas_ref), which is exactly what set_h_s expects.
        fluid = self._get_metadata_by_key("fluid")
        rho_o_in, u_o_in = fluid.set_h_s(now.ho_in, now.s_in)
        rho_o_out, u_o_out = fluid.set_h_s(now.ho_out, now.s_out)
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

        # CFL line (skipped for fixed-CFL marches that never populate now.cfl)
        if show_cfl:
            cfl = now.cfl
            out += f"  CFL   :  {cfl[0]:<8.2f} {cfl[1]:<8.2f} {cfl[2]:<8.2f} {cfl[3]:<8.2f} {cfl[4]:<8.2f}"

        # Drop any trailing newline so callers (logger) own line separation; with
        # show_cfl the CFL line already ends without one, so this is a no-op there.
        return out.rstrip("\n")

    def format_timing(self, i_step, i_finest, n_step, n_levels):
        """Format timing line: tpnps at current level, elapsed, and estimated remaining.

        Parameters
        ----------
        i_step : int
            Current global step index
        i_finest : int
            Index of the finest currently active grid level (0 = finest)
        n_step : int
            Steps per FMG phase (conf.n_step); equals total steps when no FMG
        n_levels : int
            Total number of multigrid levels (conf.n_levels)
        """
        # self.tpnps divides wall time by n_node (finest grid count), so it
        # gives us/finest-node/step regardless of which level is actually active.
        # True tpnps at the current coarse level = tpnps_stored * 8^i_finest
        # because the current level has n_node/8^i_finest nodes.
        tpnps_stored = self.tpnps
        if np.isnan(tpnps_stored):
            return "  Timing: insufficient data"

        tpnps_level = tpnps_stored * (8**i_finest)

        elapsed_ms = float(self.now.time) * self._TIME_SCALE * 1e3

        # Estimate remaining time using tpnps_level (true cost at current level)
        # and n_node_current. Future phases at finer level i cost 8^(i_finest-i)
        # times more per step than the current level.
        n_node_current = self.n_node / (8**i_finest)
        steps_left = n_step - (i_step % n_step) - 1
        equiv = float(steps_left)
        for i in range(i_finest - 1, -1, -1):
            equiv += n_step * (8 ** (i_finest - i))
        remaining_ms = equiv * tpnps_level * n_node_current / 1e3

        elapsed_min = elapsed_ms / 60e3

        return (
            f"  Timing:"
            f"  tpnps={tpnps_level:.3f} µs"
            f"  Elapsed/Remaining={elapsed_min:.1f}/{remaining_ms / 60e3:.1f} min"
        )

    def record_convergence(self, conv):
        """Record the per-step convergence monitors at the current log step.

        Parameters
        ----------
        conv : ember.grid.ConvergenceStep
            One step's monitors, from :meth:`ember.grid.Grid.get_convergence`.
            The ``mdot``, ``ho`` and ``s`` station vectors are unpacked into one
            scalar column per station, so at most 4 stations (``n_row <= 2``)
            can be recorded.
        """
        n = len(conv.mdot)
        if n > 4:
            raise NotImplementedError(
                f"Station tracking supports n_row <= 2 (<= 4 stations), got {n}"
            )
        self.now._set_data_by_keys(
            ("drho", "drhoVx", "drhoVr", "drhorVt", "drhoe"), conv.residual
        )
        self.now._set_data_by_keys(tuple(f"mdot_st{i}" for i in range(n)), conv.mdot)
        self.now._set_data_by_keys(tuple(f"ho_st{i}" for i in range(n)), conv.ho)
        self.now._set_data_by_keys(tuple(f"s_st{i}" for i in range(n)), conv.s)
        self.now._set_data_by_keys(("mdot_target",), conv.mdot_target)
        self.now._set_data_by_keys(("mdot_throttle",), conv.mdot_throttle)
        self.now._set_data_by_keys(("P_throttle",), conv.P_throttle)
        self.now._set_data_by_keys(("dP_P",), conv.dP_P)
        self.now._set_data_by_keys(("dP_I",), conv.dP_I)
        self.now._set_data_by_keys(("dP_D",), conv.dP_D)

    def record_step(self, i_step):
        """Record a new step in the history.

        Parameters
        ----------
        i_step : int
            The step index to record

        Returns
        -------
        int
            The log index where this step was recorded
        """
        # Increment log index
        i_log = self._get_metadata_by_key("i_log") + 1
        self._set_metadata_by_key("i_log", i_log)

        # Record the step index and time at current position
        self.now._set_data_by_keys(("i_step",), i_step)
        t_start = self._get_metadata_by_key("_time_start")  # float64 array
        t_raw_f64 = np.float64(_time.perf_counter()) - t_start  # subtraction in f64
        time_now = f32(t_raw_f64 / self._TIME_SCALE)  # cast to f32 after scaling
        self.now._set_data_by_keys(("time",), time_now)

        return i_log

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
            rows = [{"x": xi, "y": yi} for xi, yi in zip(x, y)]
            with open(os.path.join(directory, f"{name}.json"), "w") as f:
                json.dump(rows, f)

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
    def A_in(self):
        """Total inlet area [m^2]."""
        return self._get_metadata_by_key("A_in")

    @property
    def A_out(self):
        """Total outlet area [m^2]."""
        return self._get_metadata_by_key("A_out")

    @property
    def cfl(self):
        """CFL numbers for conserved variables [shape (..., 5)]."""
        return self._get_data_by_keys(
            ("cfl_rho", "cfl_rhoVx", "cfl_rhoVr", "cfl_rhorVt", "cfl_rhoe")
        )

    @property
    def dP_D(self):
        """Derivative PID contribution [Pa]."""
        return self._get_data_by_keys(("dP_D",))

    @property
    def dP_I(self):
        """Integral PID contribution [Pa]."""
        return self._get_data_by_keys(("dP_I",))

    @property
    def dP_P(self):
        """Proportional PID contribution [Pa]."""
        return self._get_data_by_keys(("dP_P",))

    @property
    def diverged(self):
        """True if the run that produced this history blew up.

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
        """Mass flow rate error at each logged step."""
        mdot_avg = (self.mdot_in + self.mdot_out) / 2
        dmdot = self.mdot_out - self.mdot_in
        return dmdot / mdot_avg

    @property
    def err_mdot_row(self):
        """Per-row mass flow conservation error, shape (n_log, n_row).

        err[i, r] = (mdot_dn_r - mdot_up_r) / mdot_avg_r

        Returns NaN array if n_row metadata is absent (old histories).
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
    def ho(self):
        """Non-dimensional stagnation enthalpy at each station [shape (..., 2*n_row)]."""
        return self._station_array("ho_st")

    @property
    def ho_in(self):
        r"""Inlet non-dimensional specific stagnation enthalpy (first station).

        Non-dimensionalised by ``u_ref``. Carries an offset dependent on the
        arbitrary datum where :math:`u = s = 0` at
        :math:`(p_\mathrm{dtm}, T_\mathrm{dtm})`; only changes are physically
        meaningful. See :ref:`datum-state`.
        """
        return self._get_data_by_keys(("ho_st0",))

    @property
    def ho_out(self):
        r"""Outlet non-dimensional specific stagnation enthalpy (last station).

        Non-dimensionalised by ``u_ref``. Carries an offset dependent on the
        arbitrary datum where :math:`u = s = 0` at
        :math:`(p_\mathrm{dtm}, T_\mathrm{dtm})`; only changes are physically
        meaningful. See :ref:`datum-state`.
        """
        return self._get_data_by_keys((f"ho_st{self._n_station - 1}",))

    @property
    def i_log(self):
        """Current log index (-1 means no steps recorded yet)."""
        return self._get_metadata_by_key("i_log")

    @property
    def i_step(self):
        """Step index counter."""
        return self._get_data_by_keys(("i_step",))

    @property
    def mdot(self):
        """Non-dimensional mass flow at each station [shape (..., 2*n_row)]."""
        return self._station_array("mdot_st")

    @property
    def mdot_in(self):
        """Inlet non-dimensional mass flow (first station)."""
        return self._get_data_by_keys(("mdot_st0",))

    @property
    def mdot_out(self):
        """Outlet non-dimensional mass flow (last station)."""
        return self._get_data_by_keys((f"mdot_st{self._n_station - 1}",))

    @property
    def mdot_target(self):
        """Throttle setpoint [kg/s]; 0 = inactive."""
        return self._get_data_by_keys(("mdot_target",))

    @property
    def mdot_throttle(self):
        """Actual mdot at outlet [kg/s]."""
        return self._get_data_by_keys(("mdot_throttle",))

    @property
    def n_node(self):
        """Total number of nodes in the grid."""
        return self._get_metadata_by_key("n_node")

    @property
    def now(self):
        """View of the current step (at i_log position)."""
        return self[self.i_log]

    @property
    def P_throttle(self):
        """Pressure on throttle curve [Pa]."""
        return self._get_data_by_keys(("P_throttle",))

    @property
    def psi(self):
        r"""Non-dimensional stagnation-enthalpy rise, ``ho_out - ho_in``.

        Both terms are already non-dimensional (scaled by ``u_ref``), so this is
        the inlet-to-outlet stagnation enthalpy change on the fluid reference
        scale -- no separate kinetic-energy normalisation.
        """
        return self.ho_out - self.ho_in

    @property
    def residual(self):
        """Residuals for conserved variables [shape (..., 5)]."""
        return self._get_data_by_keys(("drho", "drhoVx", "drhoVr", "drhorVt", "drhoe"))

    @property
    def s(self):
        """Non-dimensional specific entropy at each station [shape (..., 2*n_row)]."""
        return self._station_array("s_st")

    @property
    def s_in(self):
        """Inlet non-dimensional specific entropy (first station)."""
        return self._get_data_by_keys(("s_st0",))

    @property
    def s_out(self):
        """Outlet non-dimensional specific entropy (last station)."""
        return self._get_data_by_keys((f"s_st{self._n_station - 1}",))

    @property
    def throttle(self):
        """Throttle state [shape (n_step, 3)]."""
        return self._get_data_by_keys(("mdot_target", "mdot_throttle", "P_throttle"))

    @property
    def time(self):
        """Elapsed time [units of _TIME_SCALE seconds]."""
        return self._get_data_by_keys(("time",))

    @property
    def tpnps(self):
        r"""Time per node per step [:math:`\mu\mathrm{s}`], from the last recorded interval."""
        if self.i_log < 1:
            return np.nan
        dt_s = (self.time[self.i_log] - self.time[self.i_log - 1]) * self._TIME_SCALE
        di_step = self.i_step[self.i_log] - self.i_step[self.i_log - 1]
        return dt_s / di_step / self.n_node * 1e6

    @property
    def zeta(self):
        r"""Non-dimensional entropy rise, ``s_out - s_in``.

        Both terms are already non-dimensional (scaled by ``Rgas_ref``), so this
        is the inlet-to-outlet entropy generation on the fluid reference scale.
        It remains positive for an irreversible process (Gouy-Stodola), but is
        no longer normalised by a reference kinetic energy.
        """
        return self.s_out - self.s_in
