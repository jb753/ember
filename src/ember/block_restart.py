"""Restart snapshot for a single Block: save and restore solver state."""

import numpy as np
from dataclasses import dataclass
import ember.fortran
from ember.block_util import interp_from_conserved


def _frozen_copy(a):
    out = np.array(a, copy=True)
    out.flags.writeable = False
    return out


def _cons_filt_refs(block):
    """Per-component dimensional reference scales for conserved_filt_nd."""
    f = block.fluid
    return np.array(
        [
            f.rho_ref,
            f.rho_ref * f.V_ref,
            f.rho_ref * f.V_ref,
            f.rho_ref * block.L_ref * f.V_ref,
            f.rho_ref * f.V_ref**2,
        ],
        dtype=np.float32,
    )


def _index_interp(arr, target_shape):
    """Trilinearly interpolate `arr` onto `target_shape` in index space.

    Spatial axes are the first 3. Trailing axes (component dim) are
    preserved. If a 3D array is passed, the kernel's trailing singleton
    component axis is squeezed away.
    """
    spatial_in = arr.shape[:3]
    spatial_out = target_shape[:3]
    if spatial_in == spatial_out:
        return arr
    coords = [
        np.linspace(0, spatial_in[d] - 1, spatial_out[d], dtype=np.float32)
        for d in range(3)
    ]
    out = ember.fortran.map_coordinates_3d(
        arr.astype(np.float32), coords[0], coords[1], coords[2]
    )
    if arr.ndim == 3:
        out = out[..., 0]
    return out


@dataclass(frozen=True)
class BlockRestart:
    """Immutable per-block snapshot for restarting a solution.

    Fields
    ------
    conserved : ndarray
        Dimensional conserved variables of shape (ni, nj, nk, 5).
    conserved_filt_lag : ndarray or None
        Dimensional body-force controller lag (conserved_filt - conserved_cell)
        of shape (ni-1, nj-1, nk-1, 5). Stored as the lag rather than the raw
        filtered field so the small steady controller offset survives the
        save/restore round-trip without catastrophic cancellation; on restore
        the filtered field is reconstructed as conserved_cell + lag. Stored
        dimensional so reference scales can differ between save and restore.
        None if conserved_filt was never allocated.
    cfl : tuple of (ndarray or None), or None
        Cell-centred CFL fields across all multigrid levels, indexed
        fine-to-coarse. ``cfl[0]`` is the finest level, shape
        (ni-1, nj-1, nk-1, 5). Coarser levels follow at indices 1+.
        An individual element is None if that level had no cfl
        allocated. The whole field is None if no level had cfl data.
    outlet : tuple of ndarray
        One read-only array per OutletPatch, in `block.patches.outlet`
        order. Each array is `_P_target_nd / _P_raw_nd` (a unitless
        pressure profile). On restore the destination patch's
        `_P_raw` sets the mean target pressure.
    outlet_rho_soln : tuple of (ndarray or None)
        One entry per OutletPatch, in `block.patches.outlet` order. The
        patch's `_rho_nd_soln` density-relaxation reference, or None if it
        was never seeded.
    outlet_P_last : tuple of (ndarray or None)
        One entry per OutletPatch, in `block.patches.outlet` order. The
        patch's `_P_last_nd` spanwise-adjustment relaxation profile, or
        None if `set_adjustment` was not active.
    mixing : tuple of ndarray
        One read-only array per MixingPatch, in `block.patches.mixing`
        order. Each is `_target` dimensionalized so reference
        scales between save and restore can differ; stack along last
        axis is [ho, s, Vr, Vt, P].
    mixing_rho_soln : tuple of (ndarray or None)
        One entry per MixingPatch, in `block.patches.mixing` order. The
        patch's `_rho_nd_soln` density-relaxation reference, or None if it
        was never seeded.
    inlet_rho_soln : tuple of (ndarray or None)
        One entry per InletPatch, in `block.patches.inlet` order. The
        patch's `_P_nd_soln` pressure-relaxation reference, or None if it
        was never seeded.
    """

    conserved: np.ndarray
    conserved_filt_lag: np.ndarray | None = None
    cfl: tuple | None = None
    outlet: tuple = ()
    outlet_rho_soln: tuple = ()
    outlet_P_last: tuple = ()
    mixing: tuple = ()
    mixing_rho_soln: tuple = ()
    inlet_rho_soln: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "conserved", _frozen_copy(self.conserved))
        if self.conserved_filt_lag is not None:
            object.__setattr__(
                self, "conserved_filt_lag", _frozen_copy(self.conserved_filt_lag)
            )
        if self.cfl is not None:
            object.__setattr__(
                self,
                "cfl",
                tuple(_frozen_copy(a) if a is not None else None for a in self.cfl),
            )
        object.__setattr__(self, "outlet", tuple(_frozen_copy(a) for a in self.outlet))
        object.__setattr__(
            self,
            "outlet_rho_soln",
            tuple(
                _frozen_copy(a) if a is not None else None for a in self.outlet_rho_soln
            ),
        )
        object.__setattr__(
            self,
            "outlet_P_last",
            tuple(
                _frozen_copy(a) if a is not None else None for a in self.outlet_P_last
            ),
        )
        object.__setattr__(self, "mixing", tuple(_frozen_copy(a) for a in self.mixing))
        object.__setattr__(
            self,
            "mixing_rho_soln",
            tuple(
                _frozen_copy(a) if a is not None else None for a in self.mixing_rho_soln
            ),
        )
        object.__setattr__(
            self,
            "inlet_rho_soln",
            tuple(
                _frozen_copy(a) if a is not None else None for a in self.inlet_rho_soln
            ),
        )


def make_restart(grid):
    """Return a list of BlockRestart snapshots, one per block in grid.

    Coarse-level CFL data is read from the ``_cfl_coarse_restart`` stash, so the
    snapshot covers all multigrid levels without needing access to the coarse
    grids directly. dt_vol is not snapshotted — it is recomputed from the
    restored field on the next run. No current solver loop populates this
    stash (it predates ``scree.py``'s multigrid), so ``cfl`` presently always
    falls back to the fine level's own working CFL.

    The conserved field is taken from the ``_conserved_inst_restart`` stash —
    the instantaneous converged solution rather than ``block.conserved``,
    which time-averaging overwrites (see :meth:`ember.grid.Grid.finalise_average`).
    The time average is the mean of the converged limit cycle and is not
    itself on that cycle, so restarting from it injects a spurious residual.
    No current solver loop populates this stash either, so ``block.conserved``
    is used instead until one does.

    Parameters
    ----------
    grid : Grid
        The finest-level grid.

    Returns
    -------
    list of BlockRestart
    """
    restarts = []
    for block in grid:
        P_ref = block.fluid.P_ref
        outlet = tuple(
            p._P_target_nd / (p._P_raw / P_ref) for p in block.patches.outlet
        )
        outlet_rho_soln = tuple(p._rho_nd_soln for p in block.patches.outlet)
        outlet_P_last = tuple(p._P_last_nd for p in block.patches.outlet)

        refs = block._mixing_refs()
        mixing = tuple(p._target * refs for p in block.patches.mixing)
        mixing_rho_soln = tuple(p._P_nd_soln for p in block.patches.mixing)

        inlet_rho_soln = tuple(p._P_nd_soln for p in block.patches.inlet)

        # conserved_filt_nd is a cached Block property; read its store entry
        # directly so a block that never allocated it still saves None (no lag).
        _filt_entry = block._store.get("conserved_filt_nd")
        cons_filt_nd = None if _filt_entry is None else _filt_entry[1]
        if cons_filt_nd is not None:
            lag_nd = cons_filt_nd - block.conserved_cell_nd
            cons_filt_lag_dim = lag_nd * _cons_filt_refs(block)
        else:
            cons_filt_lag_dim = None

        cfl_fine = block.working.get("cfl")
        cfl_coarse = block.working.get("_cfl_coarse_restart", ())
        cfl_levels = (cfl_fine,) + tuple(cfl_coarse)
        cfl = None if all(c is None for c in cfl_levels) else cfl_levels

        # finalise overwrites block.conserved with the time average; prefer the
        # instantaneous converged field it stashes for restart. Falls back to
        # block.conserved when the grid was never run through finalise.
        conserved = block.working.get("_conserved_inst_restart")
        if conserved is None:
            conserved = block.conserved

        restarts.append(
            BlockRestart(
                conserved=conserved,
                conserved_filt_lag=cons_filt_lag_dim,
                cfl=cfl,
                outlet=outlet,
                outlet_rho_soln=outlet_rho_soln,
                outlet_P_last=outlet_P_last,
                mixing=mixing,
                mixing_rho_soln=mixing_rho_soln,
                inlet_rho_soln=inlet_rho_soln,
            )
        )
    return restarts


def apply_restart(block, restart):
    """Apply a BlockRestart to block.

    Conserved variables are always restored; cfl is restored only if the
    source had it. dt_vol is not restored — it is a fast local quantity
    recomputed from the restored field by `Grid.update_timestep`. Outlet
    pressure profiles are interpolated in index space then rescaled by the
    destination patch's `_P_raw`, so the user can change the mean target
    pressure between save and restore.

    The mixing-plane cross-plane `_target` is deliberately NOT restored: it is
    left unset so `MixingPatch.get_target`/`apply` lazily re-seed it from the
    interpolated interior pitch mean on first use, which is consistent with
    the field on this grid (restoring the saved target leaves a step-0
    inconsistency that makes the reflective plane ring; see the body).

    The outlet spanwise-adjustment relaxation profile (`_P_last_nd`) is
    restored when present, index-interpolated if shapes differ.

    Per-patch density/pressure relaxation anchors (`_rho_nd_soln` on outlet and
    mixing, `_P_nd_soln` on inlet) are NOT restored: they are start-of-step
    references that `Grid.update_bconds` overwrites via each patch's
    `update_soln()` before any `apply()` reads them, so restoring them has no
    effect (see body).

    The flux-kernel pressure datum `Block.P_offset_nd` is no longer saved or
    restored: it is a cached property keyed on the conserved state, so it
    re-derives from the restored field on first access.

    Parameters
    ----------
    block : Block
    restart : BlockRestart
    """
    interp_from_conserved(block, restart.conserved)

    if restart.conserved_filt_lag is not None:
        refs = _cons_filt_refs(block)
        lag_nd = restart.conserved_filt_lag / refs
        target_shape = block.shape_cell + (5,)
        if lag_nd.shape != target_shape:
            lag_nd = _index_interp(lag_nd.astype(np.float32), target_shape)
        cons_filt = block.conserved_filt_nd  # read-only cached buffer
        cons_filt.flags.writeable = True
        cons_filt[...] = block.conserved_cell_nd + lag_nd
        cons_filt.flags.writeable = False

    if restart.cfl is not None and restart.cfl[0] is not None:
        cfl0 = restart.cfl[0]
        target_cfl_shape = block.shape_cell + (5,)
        if cfl0.shape == target_cfl_shape:
            block.working.cfl[...] = cfl0
        else:
            block.working.cfl.fill(0.0)

    if restart.cfl is not None and len(restart.cfl) > 1:
        block.working._store["_cfl_coarse_restart"] = restart.cfl[1:]

    P_ref = block.fluid.P_ref
    for patch, P_shape, P_last in zip(
        block.patches.outlet,
        restart.outlet,
        restart.outlet_P_last,
        strict=True,
    ):
        target_shape = patch.block_view.shape
        if P_shape.shape != target_shape:
            P_shape = _index_interp(P_shape, target_shape)
        patch._P_target_nd = (P_shape * (patch._P_raw / P_ref)).astype(np.float32)
        if P_last is not None:
            if P_last.shape != target_shape:
                P_last = _index_interp(P_last.astype(np.float32), target_shape)
            patch._P_last_nd = np.array(P_last, dtype=np.float32)

    # The mixing-plane cross-plane target is intentionally NOT restored from the
    # snapshot. The saved target and the conserved field arrive through
    # different interpolation paths (separate index-interp + dimensional
    # round-trip vs interp_from_conserved), so their pitch means disagree on
    # this grid at step 0, and when the guess comes from a different solution
    # the target is foreign to the restored field. Either way the reflective
    # plane is kicked on the first step and rings. Leaving `_target` at its
    # unset (None) value lets MixingPatch.get_target/apply lazily re-seed it
    # from the interpolated interior pitch mean on first use, which is
    # consistent with the field on this grid by construction. (restart.mixing
    # is still saved for diagnostics/back-compat.)
    #
    # The per-patch density/pressure relaxation anchors (_rho_nd_soln on outlet,
    # _P_nd_soln on inlet and mixing) are likewise NOT restored. They are
    # start-of-step relaxation references, refreshed every timestep by each
    # patch's update_soln() in Grid.update_bconds before any apply() reads
    # them, so a restored value is overwritten before it can take effect.
    # update_soln() re-anchors them to the boundary-face field that
    # interp_from_conserved has already populated from restart.conserved,
    # which is exactly what restoring would have produced. (*_rho_soln are
    # still saved for diagnostics/back-compat.)
