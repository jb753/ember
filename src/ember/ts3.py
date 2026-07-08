"""Turbostream 3 (TS3) HDF5 file format I/O utilities.

This module provides comprehensive read and write functionality for the TS3 file format,
which is the native HDF5-based file format used by the Turbostream CFD solver. The module
handles bidirectional conversion between Ember's Grid objects and TS3 files, including
all grid coordinates, flow field data, patch boundary conditions, block variables, and
application variables. Key features include automatic fluid property extraction, patch
type mapping between TS3 and Ember conventions, support for inlet/outlet/periodic/cooling
patches, and validation of all written data. The TS3Writer class provides a flexible
workflow for extracting data from grids, modifying solver parameters, validating
configurations, and writing complete restart files compatible with Turbostream.
"""

import os
import time
import h5py
import numpy as np
import logging
import numbers
from glob import glob
from pathlib import Path
from ember.grid import Grid
from ember.block import Block
from ember.fluid import PerfectFluid
from ember.patch import (
    InletPatch,
    OutletPatch,
    PeriodicPatch,
    CoolingPatch,
    RotatingPatch,
    MixingPatch,
    ProbePatch,
    InviscidPatch,
)
from ember import util
from ember import util_yaml

logger = logging.getLogger(__name__)

# Map TS3 patch kind integers to ember patch classes
PATCH_BY_KIND = {
    0: InletPatch,
    1: OutletPatch,
    # 19: OutletPatch,  # outlet2d
    2: MixingPatch,
    # 17: PorousPatch,  # Not implemented in ember yet
    # 16: PeriodicPatch,  # periodic_cartesian
    5: PeriodicPatch,
    7: InviscidPatch,
    8: ProbePatch,
    # 15: NonMatchPatch,  # Not implemented in ember yet
    6: CoolingPatch,
}
# Map ember patch classes to the canonical TS3 kind integer used when writing.
# Defined explicitly rather than inverting PATCH_BY_KIND: several TS3 kinds may
# map to one ember class on read (e.g. the commented-out aliases above), but each
# class has a single kind to write, so a naive inversion would silently collapse.
KIND_BY_PATCH = {
    InletPatch: 0,
    OutletPatch: 1,
    MixingPatch: 2,
    PeriodicPatch: 5,
    CoolingPatch: 6,
    ProbePatch: 8,
    InviscidPatch: 7,
}

# Patch attributes that must be present
# Where we should not set a default, use None
DEFAULT_PA = {
    "pid": None,
    "bid": None,
    "ist": None,
    "jst": None,
    "kst": None,
    "ien": None,
    "jen": None,
    "ken": None,
    "idir": None,
    "jdir": None,
    "kdir": None,
    "kind": None,
    "nface": 0,
    "nt": 1,
    "nxbid": None,
    "nxpid": None,
}

# Block attributes that must be present
# Where we should not set a default, use -1
DEFAULT_BA = {
    "bid": -1,
    "nc": 0,
    "np": -1,
    "ncl": 0,
    "ni": -1,
    "nj": -1,
    "nk": -1,
    "procid": 0,
    "threadid": 0,
}

# Application variables that must be present
# Where we should not set a default, use NaN
DEFAULT_AV = {
    "adaptive_smoothing": 1,
    "cfl": 0.4,
    "cfl_en_ko": 0.4,
    "cfl_ko": 0.4,
    "cfl_st_ko": 0.01,
    "cp": np.nan,
    "cp0_0": 1005.0,
    "cp0_1": 1005.0,
    "cp1_0": 0.0,
    "cp1_1": 0.0,
    "cp2_0": 0.0,
    "cp2_1": 0.0,
    "cp3_0": 0.0,
    "cp3_1": 0.0,
    "cp4_0": 0.0,
    "cp4_1": 0.0,
    "cp5_0": 0.0,
    "cp5_1": 0.0,
    "dampin": 25.0,
    "dts": 0,
    "dts_conv": 0.0,
    "fac_sa_smth": 4.0,
    "fac_sa_step": 1.0,
    "fac_st0": 1.0,
    "fac_st0_option": 0,
    "fac_st1": 1.0,
    "fac_st2": 1.0,
    "fac_st3": 1.0,
    "fac_stmix": 0.0,
    "fac_wall": 1.0,
    "facsafe": 0.2,
    "facsecin": 0.005,
    "frequency": 0.0,
    "ga": np.nan,
    "if_ale": 0,
    "if_no_mg": 0,
    "ifgas": 0,
    "ifsuperfac": 0,
    "ilos": 1,
    "ko_dist": 1e-4,
    "ko_restart": 0,
    "nchange": 1000,
    "ncycle": 0,
    "nlos": 5,
    "nomatch_int": 1,
    "nspecies": 1,
    "nstep": np.nan,
    "nstep_cycle": 0,
    "nstep_inner": 0,
    "nstep_save": 0,
    "nstep_save_probe": 0,
    "nstep_save_start": 0,
    "nstep_save_start_probe": 0,
    "poisson_cfl": 0.7,
    "poisson_limit": 0,
    "poisson_nsmooth": 10,
    "poisson_nstep": 0,
    "poisson_restart": 2,
    "poisson_sfin": 0.02,
    "prandtl": np.nan,
    "precon": 0,
    "pref": 1e5,
    "restart": 1,
    "rfmix": 0.0,
    "rfvis": 0.2,
    "rg_cp0": 1005.0,
    "rg_cp1": 0.0,
    "rg_cp2": 0.0,
    "rg_cp3": 0.0,
    "rg_cp4": 0.0,
    "rg_cp5": 0.0,
    "rg_rgas": 287.15,
    "rgas_0": 287.15,
    "rgas_1": 287.15,
    "sa_ch1": 1.0,
    "sa_ch2": 1.0,
    "sa_helicity_option": 0,
    "schmidt_0": 1.0,
    "schmidt_1": 1.0,
    "sf_scalar": 0.05,
    "sfin": 0.5,
    "sfin_ko": 0.05,
    "sfin_sa": 0.05,
    "smooth_scale_directional_option": 0,
    "smooth_scale_dts_option": 0,
    "smooth_scale_precon_option": 0,
    "tref": 300.0,
    "turb_vis_damp": 1.0,
    "turbvis_lim": 3000.0,
    "use_temperature_sensor": 0,
    "viscosity": np.nan,
    "viscosity_a1": 0.0,
    "viscosity_a2": 0.0,
    "viscosity_a3": 0.0,
    "viscosity_a4": 0.0,
    "viscosity_a5": 0.0,
    "viscosity_law": 0,
    "wall_law": 0,
    "write_egen": 0,
    "write_force": 0,
    "write_tdamp": 0,
    "write_yplus": 1,
}


# Block variables that must be present
# Where we should not set a default, use NaN
DEFAULT_BV = {
    "dampin_mul": 1.0,
    "fac_st0": 1.0,
    "facsecin_mul": 1.0,
    "fblade": 1.0,
    "fl_ibpa": 0.0,
    "fmgrid": 0.2,
    "fracann": 1.0,
    "free_turb": 0.05,
    "fsturb": 1.0,
    "ftype": 0,
    "itrans": 0,
    "itrans_j1_en": 0,
    "itrans_j1_frac": 0.0,
    "itrans_j1_st": 0,
    "itrans_j2_en": 0,
    "itrans_j2_frac": 0.0,
    "itrans_j2_st": 0,
    "itrans_k1_en": 0,
    "itrans_k1_frac": 0.0,
    "itrans_k1_st": 0,
    "itrans_k2_en": 0,
    "itrans_k2_frac": 0.0,
    "itrans_k2_st": 0,
    "jtrans": 0,
    "jtrans_i1_en": 0,
    "jtrans_i1_frac": 0.0,
    "jtrans_i1_st": 0,
    "jtrans_i2_en": 0,
    "jtrans_i2_frac": 0.0,
    "jtrans_i2_st": 0,
    "jtrans_k1_en": 0,
    "jtrans_k1_frac": 0.0,
    "jtrans_k1_st": 0,
    "jtrans_k2_en": 0,
    "jtrans_k2_frac": 0.0,
    "jtrans_k2_st": 0,
    "ktrans": 0,
    "ktrans_i1_en": 0,
    "ktrans_i1_frac": 0.0,
    "ktrans_i1_st": 0,
    "ktrans_i2_en": 0,
    "ktrans_i2_frac": 0.0,
    "ktrans_i2_st": 0,
    "ktrans_j1_en": 0,
    "ktrans_j1_frac": 0.0,
    "ktrans_j1_st": 0,
    "ktrans_j2_en": 0,
    "ktrans_j2_frac": 0.0,
    "ktrans_j2_st": 0,
    "nblade": 1,
    "ndup_phaselag": 1,
    "nimixl": 0,
    "poisson_fmgrid": 0.0,
    "pstatin": 800000.0,
    "pstatout": 800000.0,
    "rpm": 0.0,
    "rpmi1": 0.0,
    "rpmi2": 0.0,
    "rpmj1": 0.0,
    "rpmj2": 0.0,
    "rpmk1": 0.0,
    "rpmk2": 0.0,
    "sfin_mul": 1.0,
    "srough_i0": 0.0,
    "srough_i1": 0.0,
    "srough_j0": 0.0,
    "srough_j1": 0.0,
    "srough_k0": 0.0,
    "srough_k1": 0.0,
    "superfac": 0.0,
    "tstagin": 1200.0,
    "tstagout": 1200.0,
    "turb_intensity": 5.0,
    "vgridin": 50.0,
    "vgridout": 50.0,
    # Mixing-length limit is baked into wdist by Grid.calculate_wdist, so
    # Turbostream's own cap is disabled: xllim large enough to never bind,
    # xllim_free zero. See Grid.calculate_wdist(limit_pitch=...).
    "xllim": 1.0e6,
    "xllim_free": 0.0,
}


def _unflip(x, shape=None):
    """Make the shape of a TS3 hdf5 array [ni, nj, nk]

    Although the TS3 hdf5 reports the shape of the data as ni x nj x nk,
    this is not actually true and the underlying data is stored in nk x nj x
    ni order. So we reshape and swap the axes back."""
    if shape and len(shape) == 4:
        ni, nj, nk, nt = shape
        return np.transpose(np.reshape(x, (nt, nk, nj, ni)))[..., 0]
    else:
        if not shape:
            ni, nj, nk = x.shape
        else:
            ni, nj, nk = shape
        return np.swapaxes(np.reshape(x, (nk, nj, ni)), 0, 2)


def scalar(x):
    """Extract scalar value from HDF5 dataset."""
    return np.squeeze(x).item()


def _write_variable(group, name, suffix, val):
    """Save a scalar to an hdf5 file."""
    key = name + suffix
    if val is None:
        raise Exception(f"Unspecified value for variable {name}")
    # numbers.Integral catches both Python int and numpy integer scalars, which
    # bool isinstance(val, int) would miss (it would mis-write them as f4)
    if isinstance(val, numbers.Integral):
        dtype = np.dtype("i4")
    else:
        dtype = np.dtype("f4")
    try:
        group.create_dataset(key, data=np.reshape(val, (1,)), dtype=dtype)
    except Exception:
        raise Exception(f"Could not write key={key}, val={val}")


def _write_property(group, name, suffix, val, flat=False):
    """Save an array to an hdf5 file."""
    key = name + suffix
    dtype = np.dtype("f4")
    if val is None:
        raise Exception(f"Unspecified value for variable {name}")
    if np.isnan(val).any():
        raise Exception(f"NaN in variable {name}")

    val_out = np.ones(val.shape, dtype=np.float32)
    val_out.flat = val.transpose().flat

    if flat:
        val_out = val_out.flatten()

    group.create_dataset(key, data=val_out, dtype=dtype)


def read_conserved(grid, fname):
    """Read the conserved flow field from a TS3 file into an existing grid.

    Loads only the per-node flow field (density, momentum, total energy) from a
    Turbostream 3 HDF5 file and stores it on the blocks of an existing
    :class:`~ember.grid.Grid`, in place. Coordinates, patches and fluid
    properties on ``grid`` are left untouched; only the flow field is
    overwritten. This avoids allocating a whole new grid (cf. :func:`read_ts3`)
    when reading a solver result back onto the grid that produced it.

    Datum handling
    --------------
    Turbostream stores total energy on its own thermodynamic datum (internal
    energy near zero at ~1 K), whereas ``grid`` may carry a general datum. The
    file's density and internal energy are therefore converted to dimensional
    pressure and temperature using a throwaway fluid placed on the TS3 datum,
    and stored via :meth:`~ember.block.Block.set_P_T`, which re-expresses the
    state on each block's own (unchanged) datum. The throwaway fluid is built
    with unity reference scales so the conversion is a direct dimensional one.

    Parameters
    ----------
    grid : ember.grid.Grid
        Grid to receive the flow field. Must have the same number of blocks,
        each of the same shape, as the file; coordinates must already be set.
    fname : str
        TS3 HDF5 file to read the flow field from.

    Returns
    -------
    grid : ember.grid.Grid
        The same grid, mutated in place, for chaining.

    Raises
    ------
    ValueError
        If the grid block count does not match the file, or the file holds a
        diverged solution (non-positive or non-finite density, etc.).
    """
    with h5py.File(fname, "r") as f:
        nb = int(scalar(f.attrs["nb"]))
        if len(grid) != nb:
            raise ValueError(f"Grid has {len(grid)} blocks but file {fname} has {nb}")

        for ib, block in enumerate(grid):
            b = f[f"block{ib}"]

            # Dimensional flow field from the file
            rho = _unflip(b["ro_bp"])
            Vx = _unflip(b["rovx_bp"]) / rho
            Vr = _unflip(b["rovr_bp"]) / rho
            Vt = _unflip(b["rorvt_bp"]) / rho / block.r
            u = _unflip(b["roe_bp"]) / rho - 0.5 * (Vx**2 + Vr**2 + Vt**2)

            # Turbostream measures internal energy from a datum at T = 0, where
            # u = cv * T, so T = u / cv and P = rho * R * T. set_P_T then stores
            # these on the block's own datum. The velocity and thermodynamic
            # setters preserve each other, so order is free.
            T = u / block.cv
            P = rho * block.Rgas * T
            block.set_Vxrt(np.stack([Vx, Vr, Vt], axis=-1))
            block.set_P_T(P, T)

    return grid


def read_mu_turb(grid, fname):
    """Read turbulent viscosity from a TS3 file into an existing grid.

    Loads the per-node turbulent viscosity (``trans_dyn_vis_bp``) from a
    Turbostream 3 HDF5 file and stores it on the blocks of an existing
    :class:`~ember.grid.Grid`, in place. Kept separate from
    :func:`read_conserved` because Turbostream writes turbulent viscosity only
    to the instantaneous output file, not the averaged one, so the two fields
    are typically read from different files.

    Unlike the conserved field, turbulent viscosity is datum-independent and is
    stored directly via :meth:`~ember.block.Block.set_mu_turb`.

    Parameters
    ----------
    grid : ember.grid.Grid
        Grid to receive the turbulent viscosity. Must have the same number of
        blocks, each of the same shape, as the file.
    fname : str
        TS3 HDF5 file to read ``trans_dyn_vis_bp`` from.

    Returns
    -------
    grid : ember.grid.Grid
        The same grid, mutated in place, for chaining.

    Raises
    ------
    ValueError
        If the grid block count does not match the file, or the file holds a
        non-finite or negative turbulent viscosity.
    KeyError
        If the file does not contain ``trans_dyn_vis_bp``.
    """
    with h5py.File(fname, "r") as f:
        nb = int(scalar(f.attrs["nb"]))
        if len(grid) != nb:
            raise ValueError(f"Grid has {len(grid)} blocks but file {fname} has {nb}")

        for ib, block in enumerate(grid):
            block.set_mu_turb(_unflip(f[f"block{ib}"]["trans_dyn_vis_bp"]))

    return grid


def read_ts3(fname):
    """Read a Turbostream 3 input file and return a Grid object.

    This function loads not only the flow field, but also coordinates
    and patch information from the file.

    Parameters
    ----------
    fname : str
        The name of the hdf5 file to read.

    Returns
    -------
    g : ember.grid.Grid
        The Grid object containing the data from the file.
    """

    with h5py.File(fname, "r") as f:
        logger.info(f"Reading TS3 input file {fname}")

        # Get gas properties from application vars and initialise a fluid.
        # These are data items of the root group. A grid written without a fluid
        # omits them (they default to NaN, which write_av skips), so their
        # absence is the "no fluid" signal: build a fluid only when present.
        if all(f"{k}_av" in f for k in ("cp", "ga", "viscosity", "prandtl")):
            cp, ga, mu = (scalar(f[f"{k}_av"]) for k in ("cp", "ga", "viscosity"))
            Pr = scalar(f["prandtl_av"])
            logger.info(
                f"Fluid properties: cp = {cp:.0f}, ga = {ga:.3f}, "
                f"mu = {mu:.3g}, Pr = {Pr:.3f}"
            )
            # The flow field is reconstructed below by converting the file's
            # conserved variables to dimensional P and T (on Turbostream's
            # u = cv * T datum) and storing via set_P_T, so the fluid can carry
            # ember's default datum rather than a TS3-specific one.
            fluid = PerfectFluid(cp=cp, gamma=ga, mu=mu, Pr=Pr)
        else:
            logger.info("No fluid properties in file; reading geometry only.")
            fluid = None

        # Get number of blocks from root group
        nb = int(scalar(f.attrs["nb"]))
        logger.info(f"Number of blocks: {nb}")

        # Loop over blocks
        blocks = []
        for ib in range(nb):
            b = f[f"block{ib}"]

            # Shape from attributes
            ni, nj, nk = [int(scalar(b.attrs[k])) for k in ("ni", "nj", "nk")]
            npatch = int(scalar(b.attrs["np"]))

            # Now read the block variables we need
            rpm = scalar(b["rpm_bv"])
            Nb = int(scalar(b["nblade_bv"]))
            logger.info(
                f"bid {ib}: shape={ni}x{nj}x{nk}, rpm={rpm:.0f}, Nb={Nb:.0f}, np={npatch}"
            )

            # Read block properties at all nodes

            # Coordinates
            x = _unflip(b["x_bp"])
            r = _unflip(b["r_bp"])
            t = _unflip(b["rt_bp"]) / r

            # Create the ember block object
            block = Block(shape=(ni, nj, nk))
            block.set_x(x)
            block.set_r(r)
            block.set_t(t)
            block.set_rpm(rpm)
            block.set_Nb(Nb)
            block.set_label(str(ib))

            # The following bp are optional

            # Flow field. Requires a fluid; without one the file is geometry
            # only, so skip. Turbostream measures internal energy from a datum
            # at T = 0 (u = cv * T), so convert the file's conserved variables
            # to dimensional velocity, P and T and store via set_Vxrt/set_P_T,
            # which re-express the state on the block's own datum. This mirrors
            # read_conserved and matches the u = cv * T convention used on write.
            if fluid is not None:
                block.set_fluid(fluid)
                try:
                    rho = _unflip(b["ro_bp"])
                    Vx = _unflip(b["rovx_bp"]) / rho
                    Vr = _unflip(b["rovr_bp"]) / rho
                    Vt = _unflip(b["rorvt_bp"]) / rho / r
                    u = _unflip(b["roe_bp"]) / rho - 0.5 * (Vx**2 + Vr**2 + Vt**2)
                    # cv and Rgas from the file's gas properties; T = u / cv on
                    # the u = cv * T datum, P = rho * Rgas * T.
                    cv = cp / ga
                    Rgas = cp - cv
                    T = u / cv
                    P = rho * Rgas * T
                    block.set_Vxrt(np.stack([Vx, Vr, Vt], axis=-1))
                    block.set_P_T(P, T)
                except KeyError:
                    pass

            try:
                block.set_mu_turb(_unflip(b["trans_dyn_vis_bp"]))
            except KeyError:
                pass

            try:
                block.set_wdist(_unflip(b["phi_bp"]))
            except KeyError:
                pass

            # Now read the patches
            patches = []
            for ip in range(npatch):
                p = b[f"patch{ip}"]

                # Check bid and pid
                assert p.attrs["bid"] == ib
                assert p.attrs["pid"] == ip

                # Start and end indices of the patch
                ist, ien, jst, jen, kst, ken = (
                    int(p.attrs[k]) for k in ("ist", "ien", "jst", "jen", "kst", "ken")
                )
                nt = int(p.attrs["nt"])

                # Patch shape
                di = ien - ist
                dj = jen - jst
                dk = ken - kst
                pshape = (di, dj, dk)

                # Subtract 1 to make the end indices inclusive (ember uses inclusive indices)
                ien -= 1
                jen -= 1
                ken -= 1

                # Select what subclass of Patch to use
                kind = p.attrs["kind"]
                if kind not in PATCH_BY_KIND:
                    logger.warning(
                        f"Skipping unknown patch kind {kind} in block {ib}, patch {ip}"
                    )
                    continue

                patch = PATCH_BY_KIND[kind](i=(ist, ien), j=(jst, jen), k=(kst, ken))

                # Now process the patch variables and properties according to subclass

                # Inlet
                if isinstance(patch, InletPatch):
                    if nt > 1:
                        tshape = (*pshape, nt)
                    else:
                        tshape = pshape
                    pstag = _unflip(p["pstag_pp"], tshape)
                    tstag = _unflip(p["tstag_pp"], tshape)
                    yaw = _unflip(p["yaw_pp"], tshape)
                    pitch = _unflip(p["pitch_pp"], tshape)

                    patch.set_Po_To_Alpha_Beta(
                        Po=pstag, To=tstag, Alpha=yaw, Beta=pitch
                    )

                # Outlet
                elif isinstance(patch, OutletPatch):
                    pout = float(p["pout_pv"][0])
                    patch.set_P(pout)

                # Cooling
                elif isinstance(patch, CoolingPatch):
                    cool_type = int(p["cool_type_pv"][0])
                    cool_mass = float(p["cool_mass_pv"][0])
                    cool_pstag = float(p["cool_pstag_pv"][0])
                    cool_tstag = float(p["cool_tstag_pv"][0])
                    cool_sangle = float(p["cool_sangle_pv"][0])
                    cool_xangle = float(p["cool_xangle_pv"][0])
                    cool_mach = float(p["cool_mach_pv"][0])

                    # Use ember's set_cool method
                    patch.set_cool(
                        type=cool_type,
                        mass=cool_mass,
                        pstag=cool_pstag,
                        tstag=cool_tstag,
                        sangle=cool_sangle,
                        xangle=cool_xangle,
                        mach=cool_mach,
                        angle_def=1,
                    )

                patches.append(patch)

            # Convert rpm block variables to RotatingPatch
            if rpmi1 := b["rpmi1_bv"][0]:
                patches.append(RotatingPatch(i=0))
                patches[-1].set_rpm(rpmi1)
            if rpmi2 := b["rpmi2_bv"][0]:
                patches.append(RotatingPatch(i=-1))
                patches[-1].set_rpm(rpmi2)
            if rpmj1 := b["rpmj1_bv"][0]:
                patches.append(RotatingPatch(j=0))
                patches[-1].set_rpm(rpmj1)
            if rpmj2 := b["rpmj2_bv"][0]:
                patches.append(RotatingPatch(j=-1))
                patches[-1].set_rpm(rpmj2)
            if rpmk1 := b["rpmk1_bv"][0]:
                patches.append(RotatingPatch(k=0))
                patches[-1].set_rpm(rpmk1)
            if rpmk2 := b["rpmk2_bv"][0]:
                patches.append(RotatingPatch(k=-1))
                patches[-1].set_rpm(rpmk2)

            # Add patches to block
            block.patches.extend(patches)

            blocks.append(block)

        # Create the grid object
        g = Grid(blocks)

        logger.info("Finished reading TS3 grid.")

        return g


def write_ts3(grid, filename, strict=False):
    """Write grid to TS3 format file.

    Total energy is always written on Turbostream's thermodynamic datum, where
    internal energy is measured from T = 0 (u = cv * T); see
    :meth:`TS3Writer.get_blocks`.

    The mixing-length limit is baked into the wall distance by
    :meth:`Grid.calculate_wdist` (via its ``limit_pitch`` argument), so
    Turbostream's own cap is disabled through the ``xllim``/``xllim_free``
    defaults in :data:`DEFAULT_BV`.

    Parameters
    ----------
    grid : Grid
        Grid object to write
    filename : str
        Output filename for TS3 file
    strict : bool, optional
        Whether to validate all variables

    """
    if len(grid) == 0:
        raise ValueError("Cannot write TS3 file: grid contains no blocks")

    writer = TS3Writer()
    writer.get(grid, strict=strict)

    writer.check()
    writer.write(filename)
    writer.write_probe_meta(filename)


class TS3Writer:
    """Create a TS3Writer to write TS3 input files.

    The workflow is:
    - get(grid) to extract data from a Grid object
    - set_av(...), set_bv(...) to modify any stored variables
    - check() to validate
    - write(f) to write application variables to a fname

    """

    def set_av(self, **kwargs):
        """Set application variables, with type casting

        Parameters
        ----------
        **kwargs : dict
            Application variables to set. See DEFAULT_AV for valid names.
        """
        if not hasattr(self, "av"):
            raise RuntimeError(
                "set_av() called before get(): application variables are "
                "populated from a grid by get(), call it first"
            )
        for name, val in kwargs.items():
            if name not in self.av:
                raise Exception(f"Unknown application variable: {name}")
            cast = type(self.av[name])(val)
            # NaN is reserved as the "unset" marker (omitted on write); a value
            # explicitly set to NaN would silently vanish, so reject it.
            if isinstance(cast, float) and np.isnan(cast):
                raise Exception(f"Application variable {name} cannot be set to NaN")
            self.av[name] = cast

    def set_bv(self, bid, **kwargs):
        """Set block variables for a given block ID, with type casting

        Parameters
        ----------
        bid : int
            The block ID to set variables for.
        **kwargs : dict
            Block variables to set. See TS3 documentation for valid names.
        """
        if not hasattr(self, "bv"):
            raise RuntimeError(
                "set_bv() called before get(): block variables are "
                "populated from a grid by get(), call it first"
            )
        for name, val in kwargs.items():
            if name not in DEFAULT_BV:
                raise Exception(f"Unknown block variable: {name}")
            cast = type(DEFAULT_BV[name])(val)
            # NaN is reserved as the "unset" marker (omitted on write); a value
            # explicitly set to NaN would silently vanish, so reject it.
            if isinstance(cast, float) and np.isnan(cast):
                raise Exception(f"Block variable {name} cannot be set to NaN")
            self.bv[bid][name] = cast

    def set_pv(self, bid, pid, **kwargs):
        """Set patch variables for a given block/patch ID, with type casting.

        Unlike set_av/set_bv there is no DEFAULT_PV: the valid patch variables
        differ by patch kind (inlet, outlet, ...) and are populated per kind by
        get_patches. So only names already present for this patch are accepted;
        an unknown name is rejected, and the value is cast to the existing
        value's type.

        Parameters
        ----------
        bid : int
            The block ID of the patch.
        pid : int
            The patch ID within the block.
        **kwargs : dict
            Patch variables to set. Must already exist on the patch.
        """
        if not hasattr(self, "pv"):
            raise RuntimeError(
                "set_pv() called before get(): patch variables are "
                "populated from a grid by get(), call it first"
            )
        pv = self.pv[bid][pid]
        for name, val in kwargs.items():
            if name not in pv:
                raise Exception(
                    f"Unknown patch variable {name} for block {bid} patch {pid}"
                )
            cast = type(pv[name])(val)
            # NaN is reserved as the "unset" marker (omitted on write); a value
            # explicitly set to NaN would silently vanish, so reject it.
            if isinstance(cast, float) and np.isnan(cast):
                raise Exception(f"Patch variable {name} cannot be set to NaN")
            pv[name] = cast

    def get_av(self, grid, strict):
        """Extract application variables from a Grid object.

        Parameters
        ----------
        grid : ember.grid.Grid
            The Grid object to extract application variables from.
        strict : bool, optional
            Whether to require all fluid properties to be available (default True)
        """
        self.av = DEFAULT_AV.copy()

        try:
            b0 = grid[0][0, 0, 0]
            self.set_av(
                cp=b0.cp,
                ga=b0.gamma,
                viscosity=b0.mu,
                prandtl=b0.Pr,
            )
        except ValueError:
            if strict:
                raise ValueError("Requires working fluid set in strict mode")

    def get_blocks(self, grid, strict):
        """Extract references to block properties from a Grid object.

        Parameters
        ----------
        grid : Grid
            Grid object to extract block data from
        strict : bool, optional
            Whether to require all flow variables to be set (default True)

        Do not make copies of the data to save memory."""

        self.ba = []
        self.bp = []
        self.bv = []
        for bid, block in enumerate(grid):
            #
            # Attributes
            self.ba.append(DEFAULT_BA.copy())
            ba = self.ba[bid]
            ba["bid"] = bid
            ba["ni"], ba["nj"], ba["nk"] = block.shape
            ba["np"] = len(
                [p for p in block.patches if not isinstance(p, RotatingPatch)]
            )

            # Variables
            self.bv.append(DEFAULT_BV.copy())
            self.set_bv(
                bid,
                nblade=block.Nb,
                fblade=float(block.Nb),
                rpm=block.rpm,
                fracann=1.0 / float(block.Nb),
            )
            # Set rotating patch rpms
            for patch in block.patches.rotating:
                if patch.const_dim == 0:
                    if patch.ist:
                        self.set_bv(bid, rpmi2=patch.rpm)
                    else:
                        self.set_bv(bid, rpmi1=patch.rpm)
                elif patch.const_dim == 1:
                    if patch.jst:
                        self.set_bv(bid, rpmj2=patch.rpm)
                    else:
                        self.set_bv(bid, rpmj1=patch.rpm)
                elif patch.const_dim == 2:
                    if patch.kst:
                        self.set_bv(bid, rpmk2=patch.rpm)
                    else:
                        self.set_bv(bid, rpmk1=patch.rpm)

            # Properties
            self.bp.append({})

            # Coords must always be set
            self.bp[bid]["x"] = block.x
            self.bp[bid]["r"] = block.r
            self.bp[bid]["rt"] = block.r * block.t  # Must calculate and store

            # Mapping of TS3 variable names to ember block attributes.
            # roe is handled separately below (re-expressed on the TS3 datum).
            flow_vars = [
                ("ro", "rho"),
                ("rovx", "rhoVx"),
                ("rovr", "rhoVr"),
                ("rorvt", "rhorVt"),
                ("phi", "wdist"),
                ("trans_dyn_vis", "mu_turb"),
            ]

            # In non-strict mode, simply omit uninitialized variables from bp dict
            for ts3_name, key in flow_vars:
                try:
                    self.bp[bid][ts3_name] = getattr(block, key)
                except ValueError:
                    if strict:
                        raise ValueError(
                            f"Block {bid}: Flow variable '{key}' is not set"
                        )

            # Total energy is always written on Turbostream's datum, which
            # measures internal energy from T = 0 (u = cv * T). Re-express it
            # without touching the grid's own datum:
            #   roe = rho * (cv * T + 0.5 * V^2)
            # The block's velocity (rhoVx etc.) is read directly above, so a
            # block with coordinates but no flow field simply omits roe.
            try:
                self.bp[bid]["roe"] = block.rho * (
                    block.cv * block.T + 0.5 * block.V**2
                )
            except ValueError:
                if strict:
                    raise ValueError(f"Block {bid}: Flow variable 'rhoe' is not set")

    def get_ga(self, grid):
        """Extract grid attributes from a Grid object."""
        self.ga = dict(nb=len(grid), ntb=0)

    def get_patches(self, grid, strict):
        """Extract patches from a Grid object."""

        # Store connectivity information
        conn = grid.connectivity.pair()

        # Store a mapping of original pids to new pids excluding rotating patches
        pid_new = {}
        for bid, block in enumerate(grid):
            pid_true = 0
            for pid, patch in enumerate(block.patches):
                if isinstance(patch, RotatingPatch):
                    continue
                pid_new[bid, pid] = pid_true
                pid_true += 1

        # Now loop over blocks and get attributes, variables, properties
        self.pa = []
        self.pv = []
        self.pp = []
        # Probe metadata, keyed by {bid: {pid: {...}}}, written as a sidecar
        self.probe_meta = {}
        for bid, block in enumerate(grid):
            # Preallocate
            self.pa.append({})
            self.pv.append({})
            self.pp.append({})

            # Loop over patches
            for pid, patch in enumerate(block.patches):
                #
                # Skip rotating patches
                if isinstance(patch, RotatingPatch):
                    continue

                # Use corrected pid numbering
                pid_true = pid_new[bid, pid]

                # Patch attributes
                self.pa[bid][pid_true] = pa = DEFAULT_PA.copy()

                # Indices
                pa["pid"] = pid_true
                pa["bid"] = bid

                # Select kind by subclass
                pa["kind"] = KIND_BY_PATCH[type(patch)]

                # Start indices and exclusive end indices
                pa["ist"], pa["jst"], pa["kst"] = patch.ijk_lim_abs[:, 0]
                pa["ien"], pa["jen"], pa["ken"] = patch.ijk_lim_abs[:, 1] + 1

                # Matching directions
                # Set dummy values first, replace if periodic/mixing and connected
                pa.update({"idir": 0, "jdir": 0, "kdir": 0, "nxbid": 0, "nxpid": 0})
                if isinstance(patch, (PeriodicPatch, MixingPatch)):
                    if (bid, pid) in conn:
                        ((nxbid, nxpid), _) = conn[bid, pid]
                        pa["nxbid"] = nxbid
                        pa["nxpid"] = pid_new[nxbid, nxpid]
                        # A mixing patch must be paired to another mixing patch.
                        if isinstance(patch, MixingPatch):
                            nxpatch = grid[nxbid].patches[nxpid]
                            assert isinstance(nxpatch, MixingPatch), (
                                f"MixingPatch {pid} on block {bid} is paired to a "
                                f"{type(nxpatch).__name__} (patch {nxpid} on block "
                                f"{nxbid}), not a MixingPatch"
                            )
                        # Only periodic patches need matched index directions;
                        # mixing planes are circumferentially averaged.
                        if isinstance(patch, PeriodicPatch):
                            (perm, flip) = conn[bid, pid][1]
                            pa["idir"], pa["jdir"], pa["kdir"] = util.perm_flip_to_dirs(
                                perm, flip, patch.const_dim
                            )
                    elif strict:
                        raise Exception(f"Patch {pid} on {bid} has no connectivity")

                # Convert to plain int for TS3
                for k, v in pa.items():
                    pa[k] = int(v)

                # Now patch variables
                self.pv[bid][pid_true] = pv = {}
                if isinstance(patch, InletPatch):
                    pv.update(
                        rfin=0.5,
                        sfinlet=0.0,
                    )
                elif isinstance(patch, OutletPatch):
                    # Note Turbostream uses 'PDI' control, not 'PID'
                    pv.update(
                        ipout=3,
                        throttle_type=1 if patch.mdot_target else 0,
                        throttle_target=patch.mdot_target,
                        throttle_k0=patch.K_pid[0],
                        throttle_k1=patch.K_pid[2],
                        throttle_k2=patch.K_pid[1],
                        fthrottle=0.0,
                        pout=patch.P,
                        pout_st=0.0,
                        pout_en=0.0,
                        pout_nchange=0,
                    )
                elif isinstance(patch, ProbePatch):
                    pv["probe_append"] = 1
                    # Record metadata so probe output can be reshaped and
                    # rebuilt on read without the (often-deleted) input hdf5.
                    # Values come straight from ember (no Cut needed).
                    self.probe_meta.setdefault(bid, {})[pid_true] = {
                        "shape": [int(s) for s in patch.shape],
                        "Omega": float(block.Omega),
                        "Nb": int(block.Nb),
                        "label": patch.label,
                        "cp": float(self.av["cp"]),
                        "ga": float(self.av["ga"]),
                        "mu": float(self.av["viscosity"]),
                        "fs": _sampling_frequency(self.av),
                        # Raw sampling vars so the expected time-sample count
                        # can be recomputed and validated on read.
                        "ncycle": int(self.av["ncycle"]),
                        "nstep_cycle": int(self.av["nstep_cycle"]),
                        "nstep_save_probe": int(self.av["nstep_save_probe"]),
                        "nstep_save_start_probe": int(
                            self.av["nstep_save_start_probe"]
                        ),
                    }

                # Patch properties
                self.pp[bid][pid_true] = pp = {}
                if isinstance(patch, InletPatch):
                    x = np.ones(patch.shape)
                    pp["pstag"] = patch.Po * x
                    pp["tstag"] = patch.To * x
                    pp["pitch"] = patch.Beta * x
                    pp["yaw"] = patch.Alpha * x
                    pp["fsturb_mul"] = x  # Inlet turbulent viscosity ratio

        # Verify mixing patches are written paired to another mixing patch.
        # Operates on the resolved self.pa (i.e. exactly what will be written),
        # so it also catches a target slot that ended up non-mixing or a
        # non-reciprocal link.
        kind_mix = KIND_BY_PATCH[MixingPatch]
        for bid, patches in enumerate(self.pa):
            for pid_true, pa in patches.items():
                if pa["kind"] != kind_mix:
                    continue
                nxbid, nxpid = pa["nxbid"], pa["nxpid"]
                nxpa = self.pa[nxbid].get(nxpid)
                assert nxpa is not None, (
                    f"MixingPatch (block {bid}, patch {pid_true}) points to "
                    f"missing patch (block {nxbid}, patch {nxpid})"
                )
                assert nxpa["kind"] == kind_mix, (
                    f"MixingPatch (block {bid}, patch {pid_true}) is paired to "
                    f"a non-mixing patch (block {nxbid}, patch {nxpid}, "
                    f"kind {nxpa['kind']})"
                )
                # Link must be reciprocal.
                assert (nxpa["nxbid"], nxpa["nxpid"]) == (bid, pid_true), (
                    f"MixingPatch pairing not reciprocal: (block {bid}, patch "
                    f"{pid_true}) -> (block {nxbid}, patch {nxpid}) -> (block "
                    f"{nxpa['nxbid']}, patch {nxpa['nxpid']})"
                )

    def check(self):
        """Validate all variables, raising an exception if any problems."""
        self.check_av()
        self.check_bv()

    def check_av(self):
        """Validate application variables, raising an exception if any problems."""

        av = self.av

        # All should be non-negative values
        # NaN marks a variable that is deliberately unset (omitted on write),
        # so skip it rather than tripping the comparison.
        for k, v in av.items():
            if np.isnan(v):
                continue
            assert v >= 0.0, f"{k} must be >= 0.0"

        # ga is NaN when no fluid was set (omitted on write); only validate it
        # when present.
        assert np.isnan(av["ga"]) or av["ga"] > 1.0, "ga must be > 1.0"
        assert 0.0 < av["cfl"] <= 0.4, "Expected 0.0 < cfl <= 0.4"
        if not np.isnan(av["nstep"]):
            assert av["nstep_save_start"] < av["nstep"], (
                "nstep_save_start must be < nstep"
            )

        assert av["ilos"] in (0, 1, 2), "ilos must be {0, 1, 2}"
        assert av["dts"] in (0, 1), "dts must be {0, 1}"

    def check_bv(self):
        """Validate block variables, raising an exception if any problems."""

        for bid, bv in enumerate(self.bv):
            # Non-negative values
            for k in (
                "fmgrid",
                "fblade",
                "fracann",
                "free_turb",
                "fsturb",
                "nblade",
                "ndup_phaselag",
                "poisson_fmgrid",
            ):
                assert bv[k] >= 0.0, f"Block {bid}: {k} must be >= 0.0"

    def get(self, grid, strict=True):
        """Extract all variables from a Grid object.

        Parameters
        ----------
        grid : ember.grid.Grid
            The Grid object to extract variables from.
        strict : bool, optional
            Whether to require all flow variables to be set (default True)
        """
        self.get_ga(grid)
        self.get_av(grid, strict)
        self.get_patches(grid, strict)
        self.get_blocks(grid, strict)

    def write(self, fname):
        """Save to a TS3 hdf5 file."""

        with h5py.File(fname, "w") as f:
            self.write_ga(f)
            self.write_av(f)
            self.write_bv(f)
            self.write_ba(f)
            self.write_bp(f)
            self.write_pa(f)
            self.write_pv(f)
            self.write_pp(f)

    def write_av(self, f):
        """Save application variables to an hdf5 file.

        Variables left at the NaN "unset" marker are omitted from the file."""
        for name, val in self.av.items():
            if isinstance(val, float) and np.isnan(val):
                continue
            _write_variable(f, name, "_av", val)

    def write_ba(self, f):
        """Save block attributes to an hdf5 file."""
        for bid, ba in enumerate(self.ba):
            block_group = f.require_group(f"block{bid}")
            block_group.attrs.update(ba)

    def write_bp(self, f):
        """Save block properties to an hdf5 file."""
        for bid, bp in enumerate(self.bp):
            block_group = f.require_group(f"block{bid}")
            for name, val in bp.items():
                _write_property(block_group, name, "_bp", val)

    def write_bv(self, f):
        """Save block variables to an hdf5 file."""
        for bid, bv in enumerate(self.bv):
            block_group = f.require_group(f"block{bid}")
            for name, val in bv.items():
                _write_variable(block_group, name, "_bv", val)

    def write_ga(self, f):
        """Save grid attributes to an hdf5 file."""
        f.attrs.update(self.ga)

    def write_pa(self, f):
        for bid, pa in enumerate(self.pa):
            block_group = f.require_group(f"block{bid}")
            for pid, pattr in pa.items():
                patch_group = block_group.require_group(f"patch{pid}")
                patch_group.attrs.update(pattr)

    def write_pp(self, f):
        for bid, pp in enumerate(self.pp):
            block_group = f.require_group(f"block{bid}")
            for pid, pprop in pp.items():
                patch_group = block_group.require_group(f"patch{pid}")
                for name, val in pprop.items():
                    _write_property(patch_group, name, "_pp", val, flat=True)

    def write_probe_meta(self, fname):
        """Write probe metadata sidecar next to the TS3 hdf5 file.

        Writes ``probe_meta.yaml`` in the same directory as ``fname`` when the
        grid contains probe patches; a no-op otherwise. The metadata lets probe
        output (flat ``.dat`` files) be reshaped back to patch dimensions and
        rebuilt into a flow field on read.
        """
        if not getattr(self, "probe_meta", None):
            return
        meta_path = Path(fname).parent / "probe_meta.yaml"
        logger.info(f"Writing probe metadata to {meta_path}")
        util_yaml.write_yaml(self.probe_meta, meta_path)

    def write_pv(self, f):
        for bid, pv in enumerate(self.pv):
            block_group = f.require_group(f"block{bid}")
            for pid, pvar in pv.items():
                patch_group = block_group.require_group(f"patch{pid}")
                for name, val in pvar.items():
                    _write_variable(patch_group, name, "_pv", val)


def _parse_bid_pid(fname):
    """Parse trailing ``_<bid>_<pid>`` from a probe file name."""
    stem = os.path.splitext(os.path.basename(fname))[0]
    bid, pid = (int(x) for x in stem.split("_")[-2:])
    return bid, pid


def read_probe_metadata(dname):
    """Map probe data files in a directory to their metadata.

    Globs probe data files (``.dat``/``.hdf5``/``.npz``) in ``dname``, reads the
    ``probe_meta.yaml`` sidecar written alongside the TS3 input file, and
    returns a dict keyed by the normalised ``.dat`` file path.

    Parameters
    ----------
    dname : str or Path
        Directory containing Turbostream 3 probe files and ``probe_meta.yaml``.

    Returns
    -------
    dict
        ``{dat_path: {"shape": ..., "Omega": ..., "Nb": ..., "label": ...}}``.
        Empty if no probe files are present.
    """
    dname = str(dname)

    # Collect probe files in any supported format, normalised to .dat names
    fnames = (
        glob(os.path.join(dname, "*_probe_*_*.hdf5"))
        + glob(os.path.join(dname, "*.npz"))
        + glob(os.path.join(dname, "*.dat"))
    )
    fnames = sorted(os.path.splitext(f)[0] + ".dat" for f in set(fnames))
    if not fnames:
        return {}

    probe_meta = util_yaml.read_yaml(os.path.join(dname, "probe_meta.yaml"))

    out = {}
    for f in fnames:
        bid, pid = _parse_bid_pid(f)
        out[f] = probe_meta[bid][pid]
    return out


def _read_probe_meta_entry(path):
    """Return the ``probe_meta.yaml`` entry dict for a single probe file.

    The sidecar is read from the same directory as ``path`` and the trailing
    ``_<bid>_<pid>`` of the file name selects the entry.

    Raises
    ------
    FileNotFoundError
        If ``probe_meta.yaml`` is missing.
    KeyError
        If the sidecar has no entry for this file's bid/pid.
    """
    path = str(path)
    meta_path = os.path.join(os.path.dirname(path), "probe_meta.yaml")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No probe_meta.yaml in {os.path.dirname(path) or '.'}; "
            "cannot determine probe metadata."
        )

    bid, pid = _parse_bid_pid(path)
    probe_meta = util_yaml.read_yaml(meta_path)
    try:
        return probe_meta[bid][pid]
    except (KeyError, TypeError):
        raise KeyError(
            f"probe_meta.yaml has no entry for bid={bid}, pid={pid} "
            f"(from {os.path.basename(path)})."
        ) from None


def _get_probe_metadata(path):
    """Return ``(shape, Omega)`` for a single probe file from its sidecar.

    ``shape`` is the spatial patch shape with a trailing ``-1`` time dimension
    appended; ``Omega`` is the probe's block rotation rate.
    """
    entry = _read_probe_meta_entry(path)
    shape = tuple(int(s) for s in entry["shape"]) + (-1,)
    return shape, float(entry["Omega"])


def _sampling_frequency(av):
    """Probe sampling frequency from application variables.

    ``fs = frequency * nstep_cycle / nstep_save_probe``, guarded to ``0.0``
    when ``nstep_save_probe`` is zero (its default).
    """
    nstep_save_probe = float(av["nstep_save_probe"])
    if nstep_save_probe == 0.0:
        return 0.0
    return float(av["frequency"]) * float(av["nstep_cycle"]) / nstep_save_probe


def _expected_nt(entry):
    """Expected probe time-sample count from a probe_meta entry, or None.

    ``nt = (ncycle*nstep_cycle - nstep_save_start_probe) // nstep_save_probe``.
    Returns ``None`` (validation not possible) when ``nstep_save_probe`` is zero
    or any sampling variable is missing (e.g. a legacy sidecar).
    """
    keys = ("ncycle", "nstep_cycle", "nstep_save_probe", "nstep_save_start_probe")
    if not all(k in entry for k in keys):
        return None
    ncycle, nstep_cycle, nstep_save_probe, nstep_save_start_probe = (
        int(entry[k]) for k in keys
    )
    if nstep_save_probe == 0:
        return None
    return (ncycle * nstep_cycle - nstep_save_start_probe) // nstep_save_probe


def _save_h5(fname, conserved):
    """Save conserved variables to a compressed HDF5 cache."""
    with h5py.File(fname, "w") as f:
        f.create_dataset(
            "conserved", data=conserved, compression="gzip", compression_opts=9
        )


def _load_h5(fname):
    """Load conserved variables from an HDF5 cache."""
    with h5py.File(fname, "r") as f:
        return f["conserved"][:]


def _get_file_mtimes(fname, cache_fname):
    """Modification times of the ``.dat`` and cache files, ``0`` when absent."""
    dat_mtime = os.path.getmtime(fname) if os.path.exists(fname) else 0
    cache_mtime = os.path.getmtime(cache_fname) if os.path.exists(cache_fname) else 0
    return dat_mtime, cache_mtime


def _build_flowfield(conserved, cp, ga, mu, Omega):
    """Build an ember ``Block`` flow field from probe conserved variables.

    Parameters
    ----------
    conserved : ndarray, shape (8, ...)
        Rows are ``x, r, rt, ro, rovx, rovr, rorvt, roe``.
    cp, ga, mu : float
        Gas properties.
    Omega : float
        Block rotation rate (rad/s).

    Returns
    -------
    Block
        Flow field with coordinates, velocities, and thermodynamic state set.
    """
    x, r, rt, ro, rovx, rovr, rorvt, roe = conserved

    block = Block(shape=x.shape)
    block.set_x(x)
    block.set_r(r)
    block.set_t(rt / r)
    # Pr does not affect any returned quantity and is absent from the sidecar.
    # T_dtm=1.0 mirrors read_ts3 (TS3 sets internal energy datum near 1 K).
    block.set_fluid(PerfectFluid(cp=cp, gamma=ga, mu=mu, Pr=1.0, T_dtm=1.0))
    block.set_conserved(np.stack((ro, rovx, rovr, rorvt, roe), axis=-1))
    block.set_Omega(Omega)
    return block


def _load_conserved_data(
    fname, cache_fname, shape, dat_mtime, cache_mtime, skip_age_check=False
):
    """Load probe conserved variables from the HDF5 cache or the ``.dat`` file.

    Returns an array of shape ``(8,) + shape``. When parsing the ``.dat`` it is
    cached as HDF5; the raw ``.dat`` is then deleted if ``skip_age_check`` is
    set or it is older than 48 hours.
    """
    # Load from cache if it exists and is newer than the dat file
    if os.path.exists(cache_fname) and cache_mtime > dat_mtime:
        conserved = _load_h5(cache_fname)
        if conserved.shape != (8,) + shape:
            conserved = conserved.reshape((8,) + shape, order="F")
        # On successful load, delete redundant dat file if present
        if os.path.exists(fname):
            os.remove(fname)
        return conserved

    # Otherwise parse the dat file and cache it
    conserved = (
        np.loadtxt(fname, skiprows=1)
        .T.reshape((8,) + shape, order="F")
        .astype(np.float32)
    )
    _save_h5(cache_fname, conserved)

    if skip_age_check:
        os.remove(fname)
    else:
        age_hours = (time.time() - dat_mtime) / 3600
        if age_hours > 48:
            os.remove(fname)
        else:
            logger.debug(f"Keeping raw probe dat file (age {age_hours:.1f} h)")

    return conserved


def read_probe_dat(fname, skip_age_check=False, validate=True):
    """Load a Turbostream 3 probe ``.dat`` file into an ember ``Block``.

    Metadata (spatial shape, rotation rate, gas properties, sampling frequency)
    is read from the ``probe_meta.yaml`` sidecar, so no ``input.hdf5`` is
    required. The probe data is cached as HDF5 next to the ``.dat`` for fast
    repeat reads.

    Parameters
    ----------
    fname : str or Path
        Path to a probe ``.dat`` file (a cache extension is also accepted).
    skip_age_check : bool, optional
        If True, delete the raw ``.dat`` after caching regardless of age.
    validate : bool, optional
        If True (default), check the number of time samples against the count
        implied by the sidecar's sampling variables, raising ``ValueError`` on
        a mismatch. Skipped automatically when those variables are absent.

    Returns
    -------
    F : Block
        Flow field of shape ``spatial + (nstep,)``.
    fs : float
        Sampling frequency in Hz.
    """
    # Normalise any cache extension back to .dat
    fname = str(fname)
    stem, ext = os.path.splitext(fname)
    if ext in (".hdf5", ".h5", ".npz"):
        fname = stem + ".dat"

    entry = _read_probe_meta_entry(fname)
    # Fixing the time dimension makes the reshape validate the sample count;
    # -1 lets numpy infer it (no check).
    nt = _expected_nt(entry)
    time_dim = nt if (validate and nt is not None) else -1
    shape = tuple(int(s) for s in entry["shape"]) + (time_dim,)
    Omega = float(entry["Omega"])
    cp, ga, mu = (float(entry[k]) for k in ("cp", "ga", "mu"))
    fs = float(entry["fs"])

    cache_fname = os.path.splitext(fname)[0] + ".hdf5"
    dat_mtime, cache_mtime = _get_file_mtimes(fname, cache_fname)
    conserved = _load_conserved_data(
        fname, cache_fname, shape, dat_mtime, cache_mtime, skip_age_check
    )

    F = _build_flowfield(conserved, cp, ga, mu, Omega)
    return F, fs


def read_probe_dat_dir(dname, label=None, exact=False, validate=True):
    """Load every probe ``.dat`` in a directory into a list of ember ``Block``s.

    Each probe is loaded via :func:`read_probe_dat`, so the directory needs a
    ``probe_meta.yaml`` sidecar but no ``input.hdf5``. Probes may have different
    shapes; returning a list (rather than a stacked array) keeps that general.

    Parameters
    ----------
    dname : str or Path
        Directory containing Turbostream 3 probe ``.dat`` files and
        ``probe_meta.yaml``.
    label : str, optional
        If given, keep only probes whose label matches (substring match, or
        exact when ``exact`` is True).
    exact : bool, optional
        Require the label to match exactly rather than as a substring.
    validate : bool, optional
        Passed through to :func:`read_probe_dat` for time-sample validation.

    Returns
    -------
    blocks : list of Block
        One flow field per probe, each of shape ``spatial + (nstep,)``.
    fs : float or None
        Sampling frequency in Hz, or ``None`` when no probes are found and no
        ``label`` filter was given.

    Raises
    ------
    ValueError
        If ``label`` is given but matches no probe; the message lists the
        available labels.
    """
    metadata = read_probe_metadata(dname)

    if label:
        all_metadata = metadata
        if exact:
            metadata = {f: m for f, m in metadata.items() if label == m["label"]}
        else:
            metadata = {f: m for f, m in metadata.items() if label in m["label"]}
        logger.info(f'Filtered by label "{label}", found {len(metadata)} probes.')

    fnames = list(metadata)
    if not fnames:
        if label:
            available = sorted(set(m["label"] for m in all_metadata.values()))
            raise ValueError(
                f'No probes found with label "{label}". Available labels: {available}'
            )
        return [], None

    blocks = []
    fs = None
    for f in fnames:
        try:
            F, fs = read_probe_dat(f, validate=validate)
        except Exception as e:
            logger.error(f"Failed to read {f}: {e}")
            continue
        blocks.append(F)

    return blocks, fs
