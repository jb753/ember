"""Restart snapshot for a single Block: save and restore solver state."""

import numpy as np
from dataclasses import dataclass
import ember.fortran
from ember.block_util import interp_from_conserved


def _frozen_copy(a):
    out = np.array(a, copy=True)
    out.flags.writeable = False
    return out


def _cons_refs(block):
    """Per-component dimensional reference scales for a stack of conserved variables."""
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
    """

    conserved: np.ndarray
    conserved_filt_lag: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "conserved", _frozen_copy(self.conserved))
        if self.conserved_filt_lag is not None:
            object.__setattr__(
                self, "conserved_filt_lag", _frozen_copy(self.conserved_filt_lag)
            )


def make_restart(grid):
    """Return a list of BlockRestart snapshots, one per block in grid.

    dt_vol is not snapshotted — it is recomputed from the restored field on the
    next run.

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
        # conserved_filt_nd is a cached Block property; read its store entry
        # directly so a block that never allocated it still saves None (no lag).
        _filt_entry = block._store.get("conserved_filt_nd")
        cons_filt_nd = None if _filt_entry is None else _filt_entry[1]
        if cons_filt_nd is not None:
            lag_nd = cons_filt_nd - block.conserved_cell_nd
            cons_filt_lag_dim = lag_nd * _cons_refs(block)
        else:
            cons_filt_lag_dim = None

        restarts.append(
            BlockRestart(
                conserved=block.conserved,
                conserved_filt_lag=cons_filt_lag_dim,
            )
        )
    return restarts


def apply_restart(block, restart):
    """Apply a BlockRestart to block.

    Conserved variables are always restored. dt_vol is not restored — it is a
    fast local quantity recomputed from the restored field by
    `Grid.update_timestep`.

    No inlet, outlet, or mixing-plane state is restored at all. Those patches
    are characteristic conditions carrying their own marched face state
    (`_prim_prev`), the mixing plane's exchanged cross-plane target, and, where
    the user has not prescribed them, backflow target rows seeded from the flow
    at the first timestep. All are re-derived from the interpolated field, so a
    restarted boundary re-converges over roughly `1/sigma` steps rather than
    resuming exactly where it left off. That is the accepted cost of holding no
    boundary state in the snapshot; the interior field, which is what the
    restart is for, is unaffected.

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
        refs = _cons_refs(block)
        lag_nd = restart.conserved_filt_lag / refs
        target_shape = block.shape_cell + (5,)
        if lag_nd.shape != target_shape:
            lag_nd = _index_interp(lag_nd.astype(np.float32), target_shape)
        cons_filt = block.conserved_filt_nd  # read-only cached buffer
        cons_filt.flags.writeable = True
        cons_filt[...] = block.conserved_cell_nd + lag_nd
        cons_filt.flags.writeable = False
