"""Cooling flow boundary condition patch for EMBER CFD.

CoolingPatch represents cooling flow injection through a boundary.

See Also
--------
ember.patch.Patch : Base class for all patches
"""

import numpy as np

from ember.basepatch import Patch


class CoolingPatch(Patch):
    """Cooling flow through a boundary."""

    _collection_name = "cooling"

    def _setup(self):
        self.type = np.nan
        self.mass = np.nan
        self.pstag = np.nan
        self.tstag = np.nan
        self.sangle = np.nan
        self.xangle = np.nan
        self.mach = np.nan
        self.angle_def = 1

    def set_cool(
        self,
        type=None,
        mass=None,
        pstag=None,
        tstag=None,
        sangle=None,
        xangle=None,
        mach=None,
        angle_def=None,
    ):
        """Set cooling parameters."""
        if type is not None:
            type_arr = np.asarray(type)
            if type_arr.ndim > 0:
                raise ValueError("type must be a scalar, not an array")
            self.type = int(type)
        if mass is not None:
            mass_arr = np.asarray(mass)
            if mass_arr.ndim > 0:
                raise ValueError("mass must be a scalar, not an array")
            self.mass = np.float32(mass)
        if pstag is not None:
            pstag_arr = np.asarray(pstag)
            if pstag_arr.ndim > 0:
                raise ValueError("pstag must be a scalar, not an array")
            self.pstag = np.float32(pstag)
        if tstag is not None:
            tstag_arr = np.asarray(tstag)
            if tstag_arr.ndim > 0:
                raise ValueError("tstag must be a scalar, not an array")
            self.tstag = np.float32(tstag)
        if sangle is not None:
            sangle_arr = np.asarray(sangle)
            if sangle_arr.ndim > 0:
                raise ValueError("sangle must be a scalar, not an array")
            self.sangle = np.float32(sangle)
        if xangle is not None:
            xangle_arr = np.asarray(xangle)
            if xangle_arr.ndim > 0:
                raise ValueError("xangle must be a scalar, not an array")
            self.xangle = np.float32(xangle)
        if mach is not None:
            mach_arr = np.asarray(mach)
            if mach_arr.ndim > 0:
                raise ValueError("mach must be a scalar, not an array")
            self.mach = np.float32(mach)
        if angle_def is not None:
            angle_def_arr = np.asarray(angle_def)
            if angle_def_arr.ndim > 0:
                raise ValueError("angle_def must be a scalar, not an array")
            self.angle_def = int(angle_def)
        return self
