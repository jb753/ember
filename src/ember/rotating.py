"""Rotating wall boundary condition patch for EMBER CFD.

RotatingPatch represents a rotating wall boundary with specified angular velocity.

See Also
--------
ember.patch.Patch : Base class for all patches
"""

import numpy as np

from ember.basepatch import Patch


class RotatingPatch(Patch):
    """Rotating wall boundary with a prescribed angular velocity.

    Overrides the wall angular velocity on the patch face used by the Fortran
    flux routines to compute wall-relative velocities. Faces without a
    :class:`RotatingPatch` default to the block angular velocity
    :py:attr:`~ember.block.Block.Omega`.

    Angular velocity must be set via :meth:`set_Omega` or :meth:`set_rpm`
    before the solver runs.
    """

    _collection_name = "rotating"

    def _copy(self, c):
        c._Omega = self._Omega

    def _setup(self):
        self._Omega = np.nan

    def set_Omega(self, Omega):
        """Set the wall angular velocity.

        Parameters
        ----------
        Omega : float
            Angular velocity [rad/s]. Must be a scalar.
        """
        Omega_arr = np.asarray(Omega)
        if Omega_arr.ndim > 0:
            raise ValueError("Omega must be a scalar, not an array")
        self._Omega = np.float32(Omega)

    def set_rpm(self, rpm):
        r"""Set the wall angular velocity from revolutions per minute.

        Converts via :math:`\Omega = \mathrm{rpm} \cdot 2\pi / 60`.

        Parameters
        ----------
        rpm : float
            Rotational speed [rev/min].
        """
        self.set_Omega(rpm * 2.0 * np.pi / 60.0)

    @property
    def Omega(self):
        """Angular velocity [rad/s]."""
        return self._Omega

    @property
    def rpm(self):
        """Angular velocity in revolutions per minute [rpm]."""
        return self._Omega * 60.0 / (2.0 * np.pi)
