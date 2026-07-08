r"""Working fluid interface and equation of state implementations.

This module defines an interface for computing thermodynamic properties of
working fluids enabling manipulations, of flow fields independent of the underlying equations
of state. The
abstraction cleanly separates thermodynamic relations from the flow solver,
allowing easy extension to real gas models or tabulated properties.

Currently, the only implementation of the interface is :class:`PerfectFluid` for ideal gases with
constant specific heats.

A fluid instance is immutable and only stores intrinsic fluid properties that never change, such
as specific heats for a perfect gas, or the fluid species for a real
fluid. These must passed into the constructor on initialisation.

Get and set methods
===================

The chosen basic state variables are density and internal energy, :math:`(\rho, u)`, because these are the most natural in a conservative computational fluid dynamics solver. An equation of state must provide two types of methods: `get_?` and `set_?_?`.

- `set_x_y(x,y)`: Take thermodynamic properties :math:`(x, y)` and return :math:`(\rho, u)`.
- `get_z(rho, u, out=None)`: Take :math:`(rho, u)` and return thermodynamic property :math:`z`.

All methods support both scalar and array inputs, where the inputs must be broadcastable against each other. The output will have the broadcasted shape. Constructor inputs are cast to single-precision floats, and outputs will remain single-precision if all inputs are single-precision. Supplying an `out` keyword argument to `get_?` methods allows the output to be written into a pre-allocated array, following NumPy conventions, which may improve performance by avoiding temporary array allocations.

.. _datum-state:

Datum state
===========

Only changes in internal energy, enthalpy, and entropy are physically meaningful, therefore we have the freedom to set the physical state at which these properties are zero to improve numerics and reduce precision errors due to subtracting two large floats. We define a thermodynamic datum :math:`(p_\mathrm{dtm}, T_\mathrm{dtm})` is where :math:`u = s = 0`
simultaneously. Enthalpy at the datum is not zero because of the pressure term in :math:`h = u + p/\rho`.

It is possible to shift the datum state of a fluid instance using the `change_datum` method, which returns a new instance with the same properties but shifted datum. The current datum is accessible via the :attr:`~PerfectFluid.P_dtm` and :attr:`~PerfectFluid.T_dtm` attributes. So the following code simply returns an identical copy of the original `fluid` instance:

.. code:: python

    fluid.change_datum(fluid.P_dtm, fluid.T_dtm)

.. _reference-scales:

Reference scales
================

The constructors for fluid instances take optional reference scales for non-dimensionalisation, which default to unity such that all quantities are in SI units. If reference scales are provided, then internally the class uses them to form a consistent system of non-dimensional quantities, and all inputs and outputs are taken as non-dimensional. The advantage of setting reference scales is improved numerical precision when working with non-dimensional quantities all of order unity.

The user specifies:

- :math:`\rho_\mathrm{ref}\,`: density [kg/m\ :sup:`3`]
- :math:`V_\mathrm{ref}\,`: velocity [m/s]
- :math:`R_\mathrm{ref}\,`: gas constant [J/kg/K]

and the class forms the following derived reference scales:

- :math:`p_\mathrm{ref} = \rho V_\mathrm{ref}^2\,`: dynamic pressure [Pa]
- :math:`u_\mathrm{ref} = V_\mathrm{ref}^2\,`: specific energy [J/kg]
- :math:`T_\mathrm{ref} = V_\mathrm{ref}^2 / R_\mathrm{ref}\,`: temperature [K].

Equations of state are unchanged when all quantities are scaled consistently. For example, taking the ideal gas law :math:`p = \rho R T` and dividing through by the reference pressure :math:`\rho_\mathrm{ref} V_\mathrm{ref}^2` gives

.. math:: \frac{p}{\rho_\mathrm{ref} V_\mathrm{ref}^2} = \frac{\rho}{\rho_\mathrm{ref}} \frac{R}{R_\mathrm{ref}} \frac{T}{V_\mathrm{ref}^2 / R_\mathrm{ref}} = \frac{\rho}{\rho_\mathrm{ref}} \frac{R}{R_\mathrm{ref}} \frac{T}{T_\mathrm{ref}}

Transport properties such as viscosity and thermal conductivity are an exception to this scaling, and would require an additional reference length to make fully non-dimensional. So when references are provided, transport properties have dimensions of [m].

We can get a new instance with different reference scales using the :meth:`PerfectFluid.change_ref` method.

"""

import numpy as np
from abc import ABC, abstractmethod
from ember import util


class _Fluid(ABC):
    """Interface for converting density and internal energy to and from other thermodynamic properties.

    Constructors should cast all input parameters to single-precision floats; the output types of all methods are not explicitly cast, but will be single-precision if all inputs are single-precision.

    """

    def __init__(self, rho_ref=1.0, V_ref=1.0, Rgas_ref=1.0):
        if rho_ref <= 0:
            raise ValueError(f"rho_ref must be positive, got {rho_ref}")
        if V_ref <= 0:
            raise ValueError(f"V_ref must be positive, got {V_ref}")

        self._rho_ref = np.float32(rho_ref)
        self._V_ref = np.float32(V_ref)
        self._Rgas_ref = np.float32(Rgas_ref)
        self._P_ref = np.float32(rho_ref * V_ref**2)
        self._u_ref = np.float32(V_ref**2)
        self._T_ref = np.float32(V_ref**2 / Rgas_ref)
        self._rhoV_ref = np.float32(rho_ref * V_ref)

    @abstractmethod
    def set_h_s(self, h, s):
        raise NotImplementedError()

    @abstractmethod
    def set_P_h(self, P, h):
        raise NotImplementedError()

    @abstractmethod
    def set_P_rho(self, P, rho):
        raise NotImplementedError()

    @abstractmethod
    def set_P_s(self, P, s):
        raise NotImplementedError()

    @abstractmethod
    def set_P_T(self, P, T):
        raise NotImplementedError()

    @abstractmethod
    def set_rho_s(self, rho, s):
        raise NotImplementedError()

    @abstractmethod
    def set_T_rho(self, T, rho):
        raise NotImplementedError()

    @abstractmethod
    def set_T_s(self, T, s):
        raise NotImplementedError()

    @abstractmethod
    def get_a(self, rho, u, out=None):
        raise NotImplementedError()

    @abstractmethod
    def get_cp(self, rho, u, out=None):
        raise NotImplementedError()

    @abstractmethod
    def get_cv(self, rho, u, out=None):
        raise NotImplementedError()

    @abstractmethod
    def get_dhdP_rho(self, rho, u, out=None):
        """Derivative of enthalpy with respect to pressure at constant density."""
        raise NotImplementedError()

    @abstractmethod
    def get_dhdrho_P(self, rho, u, out=None):
        """Derivative of enthalpy with respect to density at constant pressure."""
        raise NotImplementedError()

    @abstractmethod
    def get_dsdP_rho(self, rho, u, out=None):
        """Derivative of entropy with respect to pressure at constant density."""
        raise NotImplementedError()

    @abstractmethod
    def get_dsdrho_P(self, rho, u, out=None):
        """Derivative of entropy with respect to density at constant pressure."""
        raise NotImplementedError()

    @abstractmethod
    def get_dudP_rho(self, rho, u, out=None):
        """Derivative of internal energy with respect to pressure at constant density."""
        raise NotImplementedError()

    @abstractmethod
    def get_dudrho_P(self, rho, u, out=None):
        """Derivative of internal energy with respect to density at constant pressure."""
        raise NotImplementedError()

    @abstractmethod
    def get_gamma(self, rho, u, out=None):
        raise NotImplementedError()

    @abstractmethod
    def get_h(self, rho, u, out=None):
        raise NotImplementedError()

    @abstractmethod
    def get_mu(self, rho, u, out=None):
        """Quasi-dimensional dynamic viscosity, mu / (rho_ref * V_ref) [m]."""
        raise NotImplementedError()

    @abstractmethod
    def get_P(self, rho, u, out=None):
        raise NotImplementedError()

    @abstractmethod
    def get_Pr(self, rho, u, out=None):
        raise NotImplementedError()

    @abstractmethod
    def get_Rgas(self, rho, u, out=None):
        raise NotImplementedError()

    @abstractmethod
    def get_s(self, rho, u, out=None):
        raise NotImplementedError()

    @abstractmethod
    def get_T(self, rho, u, out=None):
        raise NotImplementedError()

    @abstractmethod
    def change_datum(self, P_dtm_new, T_dtm_new):
        """Return (fluid_new, rho_nd_new, u_nd_new) with datum shifted to (P_dtm_new, T_dtm_new).

        rho_nd is unchanged for a perfect gas; may differ for real-gas EOS.
        """
        raise NotImplementedError()

    def change_ref(self, rho_ref=None, V_ref=None, Rgas_ref=None):
        """Return a new instance with different reference scales."""
        raise NotImplementedError("Subclasses must implement change_ref")

    @property
    def P_dtm(self):
        r"""Datum pressure :math:`p_\mathrm{dtm}` where :math:`u = s = 0` [Pa]."""
        return self._P_dtm

    @property
    def P_ref(self):
        r"""Reference pressure for nondimensionalisation, :math:`p_\mathrm{ref}` [Pa].

        See :ref:`reference-scales`; derived from

        .. math:: p_\mathrm{ref} = \rho_\mathrm{ref} V_\mathrm{ref}^2
        """
        return self._P_ref

    @property
    def Rgas_ref(self):
        r"""Reference gas constant for nondimensionalisation, :math:`R_\mathrm{ref}` [J/kg/K].

        User-specified; see :ref:`reference-scales`.
        """
        return self._Rgas_ref

    @property
    def rho_ref(self):
        r"""Reference density for nondimensionalisation, :math:`\rho_\mathrm{ref}` [kg/m³].

        User-specified; see :ref:`reference-scales`.
        """
        return self._rho_ref

    @property
    def rhoV_ref(self):
        r"""Reference mass flux for nondimensionalisation, :math:`(\rho V)_\mathrm{ref}` [kg/m²/s].

        See :ref:`reference-scales`; derived from

        .. math:: (\rho V)_\mathrm{ref} = \rho_\mathrm{ref} V_\mathrm{ref}
        """
        return self._rhoV_ref

    @property
    def T_dtm(self):
        r"""Datum temperature :math:`T_\mathrm{dtm}` where :math:`u = s = 0` [K]."""
        return self._T_dtm

    @property
    def T_ref(self):
        r"""Reference temperature for nondimensionalisation, :math:`T_\mathrm{ref}` [K].

        See :ref:`reference-scales`; derived from

        .. math:: T_\mathrm{ref} = V_\mathrm{ref}^2 / R_\mathrm{ref}
        """
        return self._T_ref

    @property
    def u_ref(self):
        r"""Reference specific energy for nondimensionalisation, :math:`u_\mathrm{ref}` [J/kg].

        See :ref:`reference-scales`; derived from

        .. math:: u_\mathrm{ref} = V_\mathrm{ref}^2
        """
        return self._u_ref

    @property
    def V_ref(self):
        r"""Reference velocity for nondimensionalisation, :math:`V_\mathrm{ref}` [m/s].

        User-specified; see :ref:`reference-scales`.
        """
        return self._V_ref


class PerfectFluid(_Fluid):
    def __init__(
        self,
        cp,
        gamma,
        mu,
        Pr,
        P_dtm=1e5,
        T_dtm=300.0,
        rho_ref=1.0,
        V_ref=1.0,
        Rgas_ref=1.0,
    ):
        """Perfect gas with constant specific heats.

        Parameters
        ----------
        cp : float
            Specific heat at constant pressure [J/kg/K].
        gamma : float
            Ratio of specific heats [--].
        mu : float
            Dynamic viscosity [kg/m/s].
        Pr : float
            Prandtl number [--].
        P_dtm : float, optional
            Datum pressure where u = 0 and s = 0 [Pa].
        T_dtm : float, optional
            Datum temperature where u = 0 and s = 0 [K].
        rho_ref : float, optional
            Reference density for non-dimensionalisation.
        V_ref : float, optional
            Reference velocity for non-dimensionalisation.
        Rgas_ref : float, optional
            Reference gas constant for non-dimensionalisation.

        """

        super().__init__(rho_ref, V_ref, Rgas_ref)

        # Store dimensional fluid properties for change_datum / change_ref
        self._cp = np.float32(cp)
        self._gamma = np.float32(gamma)
        self._mu = np.float32(mu)
        self._mu_nd = np.float32(mu / (rho_ref * V_ref))
        self._Pr = np.float32(Pr)
        self._P_dtm = np.float32(P_dtm)
        self._T_dtm = np.float32(T_dtm)

        # Check for nonsense values
        if self._cp <= 0.0:
            raise ValueError(f"cp={cp} must be positive.")
        if self._gamma <= 1.0:
            raise ValueError(f"gamma={gamma} must be greater than 1.")
        if self._mu <= 0.0:
            raise ValueError(f"mu={mu} must be positive.")
        if Pr <= 0.0:
            raise ValueError(f"Pr={Pr} must be positive.")
        if T_dtm <= 0.0:
            raise ValueError(f"T_dtm={T_dtm} must be positive.")
        if P_dtm <= 0.0:
            raise ValueError(f"P_dtm={P_dtm} must be positive.")

        # Derived dimensional
        self._cv = self._cp / self._gamma

        # Nondim stored properties
        self._cp_nd = np.float32(cp / Rgas_ref)
        self._cv_nd = self._cp_nd / self._gamma
        self._Rgas_nd = self._cp_nd - self._cv_nd
        self._T_dtm_nd = np.float32(T_dtm * Rgas_ref / V_ref**2)
        self._P_dtm_nd = np.float32(P_dtm / (rho_ref * V_ref**2))
        self._rho_dtm_nd = self._P_dtm_nd / (self._Rgas_nd * self._T_dtm_nd)

        self._gamma_m1 = self._gamma - np.float32(1.0)
        self._ga_gam1 = self._gamma / self._gamma_m1

    @staticmethod
    def _const_nd(rho_nd, u_nd, value, out):
        """Return a constant broadcast to the shape of (rho_nd, u_nd)."""
        if out is None:
            return util.full(np.broadcast(rho_nd, u_nd).shape, value)
        out[...] = value
        return out

    def set_h_s(self, h, s):
        r"""Density and internal energy from specific enthalpy and entropy.

        Temperature is recovered from :math:`h`, then pressure from :math:`s`
        , then :math:`\rho` and :math:`u` follow from
        :meth:`set_P_T`:

        .. math::

            T = \frac{h}{c_p} + \frac{T_\mathrm{dtm}}{\gamma}, \qquad
            p = p_\mathrm{dtm} \exp\!\left(\frac{c_p\ln(T/T_\mathrm{dtm}) - s}{R}\right)

        Parameters
        ----------
        h : array_like
            Specific enthalpy [J/kg].
        s : array_like
            Specific entropy [J/kg/K].

        Returns
        -------
        rho : ndarray
            Density [kg/m³].
        u : ndarray
            Specific internal energy [J/kg].
        """
        T = h / self._cp_nd + self._T_dtm_nd / self._gamma
        P = self._P_dtm_nd * np.exp(
            (self._cp_nd * np.log(T / self._T_dtm_nd) - s) / self._Rgas_nd
        )
        return self.set_P_T(P, T)

    def set_P_h(self, P, h):
        r"""Density and internal energy from pressure and specific enthalpy.

        Temperature is recovered from :math:`h = c_p T - R T_\mathrm{dtm}`:

        .. math::

            T = \frac{h}{c_p} + \frac{T_\mathrm{dtm}}{\gamma}

        Then :math:`\rho` and :math:`u` follow from :meth:`set_P_T`.

        Parameters
        ----------
        P : array_like
            Pressure [Pa].
        h : array_like
            Specific enthalpy [J/kg].

        Returns
        -------
        rho : ndarray
            Density [kg/m³].
        u : ndarray
            Specific internal energy [J/kg].
        """
        T = h / self._cp_nd + self._T_dtm_nd / self._gamma
        return self.set_P_T(P, T)

    def set_P_rho(self, P, rho):
        r"""Density and internal energy from pressure and density.

        Temperature follows from the ideal gas law, giving:

        .. math::

            u = c_v\!\left(\frac{p}{\rho R} - T_\mathrm{dtm}\right)

        Parameters
        ----------
        P : array_like
            Pressure [Pa].
        rho : array_like
            Density [kg/m³].

        Returns
        -------
        rho : ndarray
            Density [kg/m³] (returned unchanged).
        u : ndarray
            Specific internal energy [J/kg].
        """
        u = self._cv_nd * (P / (self._Rgas_nd * rho) - self._T_dtm_nd)
        return rho, u

    def set_P_s(self, P, s):
        r"""Density and internal energy from pressure and specific entropy.

        Inverting the Gibbs relation gives temperature, then :math:`\rho` and
        :math:`u` follow from :meth:`set_P_T`:

        .. math::

            T = T_\mathrm{dtm} \exp\!\left(\frac{s + R\ln(p/p_\mathrm{dtm})}{c_p}\right)

        Parameters
        ----------
        P : array_like
            Pressure [Pa].
        s : array_like
            Specific entropy [J/kg/K].

        Returns
        -------
        rho : ndarray
            Density [kg/m³].
        u : ndarray
            Specific internal energy [J/kg].
        """
        # T = T_dtm * exp((s + R*ln(P/P_dtm)) / cp).
        T = P / self._P_dtm_nd
        if not isinstance(T, np.ndarray) or T.ndim == 0:
            # Scalar/0-d setup path (cold): plain expression -- in-place ufuncs
            # with out= reject numpy scalars, and allocation is irrelevant.
            T = self._T_dtm_nd * np.exp((s + self._Rgas_nd * np.log(T)) / self._cp_nd)
            return self.set_P_T(P, T)
        # Array hot path: fold the chain into the single T buffer with out=,
        # collapsing the ~6 temporaries the expression form allocated. T is fresh
        # (P / P_dtm never aliases the caller's P or s), so the steps are safe.
        np.log(T, out=T)
        T *= self._Rgas_nd
        T += s
        T /= self._cp_nd
        np.exp(T, out=T)
        T *= self._T_dtm_nd
        return self.set_P_T(P, T)

    def set_P_T(self, P, T):
        r"""Density and internal energy from pressure and temperature.

        From the ideal gas law and the calorific equation of state:

        .. math::

            \rho = \frac{p}{RT}, \qquad u = c_v(T - T_\mathrm{dtm})

        Parameters
        ----------
        P : array_like
            Pressure [Pa].
        T : array_like
            Temperature [K].

        Returns
        -------
        rho : ndarray
            Density [kg/m³].
        u : ndarray
            Specific internal energy [J/kg].
        """
        u = self._cv_nd * (T - self._T_dtm_nd)
        rho = P / (self._Rgas_nd * T)
        return rho, u

    def set_rho_s(self, rho, s):
        r"""Density and internal energy from density and specific entropy.

        Inverting the entropy relation for a perfect gas at fixed density gives
        temperature directly:

        .. math::

            T = T_\mathrm{dtm} \exp\!\left(\frac{s}{c_v} + (\gamma-1)\ln\!\frac{\rho}{\rho_\mathrm{dtm}}\right)

        where :math:`\rho_\mathrm{dtm} = p_\mathrm{dtm}/(R T_\mathrm{dtm})`, then:

        .. math::

            u = c_v(T - T_\mathrm{dtm})

        Parameters
        ----------
        rho : array_like
            Density [kg/m³].
        s : array_like
            Specific entropy [J/kg/K].

        Returns
        -------
        rho : ndarray
            Density [kg/m³] (returned unchanged).
        u : ndarray
            Specific internal energy [J/kg].
        """
        T = self._T_dtm_nd * np.exp(
            s / self._cv_nd + self._gamma_m1 * np.log(rho / self._rho_dtm_nd)
        )
        u = self._cv_nd * (T - self._T_dtm_nd)
        return rho, u

    def set_T_rho(self, T, rho):
        r"""Density and internal energy from temperature and density.

        From the calorific equation of state:

        .. math::

            u = c_v(T - T_\mathrm{dtm})

        Parameters
        ----------
        T : array_like
            Temperature [K].
        rho : array_like
            Density [kg/m³].

        Returns
        -------
        rho : ndarray
            Density [kg/m³] (returned unchanged).
        u : ndarray
            Specific internal energy [J/kg].
        """
        u = self._cv_nd * (T - self._T_dtm_nd)
        return rho, u

    def set_T_s(self, T, s):
        r"""Density and internal energy from temperature and specific entropy.

        Inverting the Gibbs relation gives pressure, then :math:`\rho` and
        :math:`u` follow from :meth:`set_P_T`:

        .. math::

            p = p_\mathrm{dtm} \exp\!\left(\frac{c_p\ln(T/T_\mathrm{dtm}) - s}{R}\right)

        Parameters
        ----------
        T : array_like
            Temperature [K].
        s : array_like
            Specific entropy [J/kg/K].

        Returns
        -------
        rho : ndarray
            Density [kg/m³].
        u : ndarray
            Specific internal energy [J/kg].
        """
        P = self._P_dtm_nd * np.exp(
            (self._cp_nd * np.log(T / self._T_dtm_nd) - s) / self._Rgas_nd
        )
        return self.set_P_T(P, T)

    def get_a(self, rho, u, out=None):
        r"""Speed of sound from density and internal energy.

        For a perfect gas, :math:`a^2 = \gamma R T`, combined with
        the definition of internal energy :math:`u = c_v (T - T_\mathrm{dtm})` gives

        .. math:: a = \sqrt{\gamma R \left(\frac{u}{c_v} + T_\mathrm{dtm}\right)}

        Parameters
        ----------
        rho : array_like
            Density [kg/m³].
        u : array_like
            Specific internal energy [J/kg].
        out : ndarray, optional
            Pre-allocated output array.

        Returns
        -------
        a : ndarray
            Speed of sound [m/s].
        """
        if out is None:
            return np.sqrt(
                self._gamma * self._Rgas_nd * (u / self._cv_nd + self._T_dtm_nd)
            )
        np.divide(u, self._cv_nd, out=out)
        out += self._T_dtm_nd
        out *= self._gamma * self._Rgas_nd
        np.sqrt(out, out=out)
        return out

    def get_cp(self, rho, u, out=None):
        r"""Specific heat at constant pressure (constant for a perfect gas).

        .. math:: c_p = \frac{\gamma R}{\gamma - 1}

        Parameters
        ----------
        rho : array_like
            Density [kg/m³].
        u : array_like
            Specific internal energy [J/kg].
        out : ndarray, optional
            Pre-allocated output array.

        Returns
        -------
        cp : ndarray
            Specific heat at constant pressure [J/kg/K].
        """
        return self._const_nd(rho, u, self._cp_nd, out)

    def get_cv(self, rho, u, out=None):
        r"""Specific heat at constant volume (constant for a perfect gas).

        .. math:: c_v = \frac{R}{\gamma - 1} = \frac{c_p}{\gamma}

        Parameters
        ----------
        rho : array_like
            Density [kg/m³].
        u : array_like
            Specific internal energy [J/kg].
        out : ndarray, optional
            Pre-allocated output array.

        Returns
        -------
        cv : ndarray
            Specific heat at constant volume [J/kg/K].
        """
        return self._const_nd(rho, u, self._cv_nd, out)

    def get_dhdP_rho(self, rho, u, out=None):
        r"""Derivative of specific enthalpy with respect to pressure at constant density.

        From :math:`h = \gamma u + R T_\mathrm{dtm}` and
        :math:`p = \rho R T`, differentiating at constant :math:`\rho`:

        .. math:: \left.\frac{\partial h}{\partial p}\right|_\rho = \frac{\gamma}{\rho(\gamma-1)}

        Parameters
        ----------
        rho : array_like
            Density [kg/m³].
        u : array_like
            Specific internal energy [J/kg].
        out : ndarray, optional
            Pre-allocated output array.

        Returns
        -------
        dhdP_rho : ndarray
            Derivative :math:`(\partial h/\partial p)_\rho` [m³/kg].
        """
        if out is None:
            return self._ga_gam1 / rho
        np.divide(self._ga_gam1, rho, out=out)
        return out

    def get_dhdrho_P(self, rho, u, out=None):
        r"""Derivative of specific enthalpy with respect to density at constant pressure.

        From :math:`h = c_p T` and the ideal gas law :math:`T = p / (\rho R)`,
        differentiating at constant :math:`p`:

        .. math:: \left.\frac{\partial h}{\partial \rho}\right|_p = -\frac{c_p T}{\rho}

        Parameters
        ----------
        rho : array_like
            Density [kg/m³].
        u : array_like
            Specific internal energy [J/kg].
        out : ndarray, optional
            Pre-allocated output array.

        Returns
        -------
        dhdrho_P : ndarray
            Derivative :math:`(\partial h/\partial \rho)_p` [J·m³/kg²].
        """
        out = self.get_T(rho, u, out=out)
        out *= -self._cp_nd
        out /= rho
        return out

    def get_dsdP_rho(self, rho, u, out=None):
        r"""Derivative of specific entropy with respect to pressure at constant density.

        From the Gibbs relation for a perfect gas, differentiating at constant
        :math:`\rho` (so :math:`T \propto p`):

        .. math:: \left.\frac{\partial s}{\partial p}\right|_\rho = \frac{c_v}{p}

        Parameters
        ----------
        rho : array_like
            Density [kg/m³].
        u : array_like
            Specific internal energy [J/kg].
        out : ndarray, optional
            Pre-allocated output array.

        Returns
        -------
        dsdP_rho : ndarray
            Derivative :math:`(\partial s/\partial p)_\rho` [J/kg/K/Pa].
        """
        out = self.get_P(rho, u, out=out)
        np.divide(self._cv_nd, out, out=out)
        return out

    def get_dsdrho_P(self, rho, u, out=None):
        r"""Derivative of specific entropy with respect to density at constant pressure.

        From the Gibbs relation for a perfect gas, differentiating at constant
        :math:`p` (so :math:`T \propto 1/\rho`):

        .. math:: \left.\frac{\partial s}{\partial \rho}\right|_p = -\frac{c_p}{\rho}

        Parameters
        ----------
        rho : array_like
            Density [kg/m³].
        u : array_like
            Specific internal energy [J/kg].
        out : ndarray, optional
            Pre-allocated output array.

        Returns
        -------
        dsdrho_P : ndarray
            Derivative :math:`(\partial s/\partial \rho)_p` [J·m³/kg²/K].
        """
        if out is None:
            return -self._cp_nd / rho
        np.divide(-self._cp_nd, rho, out=out)
        return out

    def get_dudP_rho(self, rho, u, out=None):
        r"""Derivative of specific internal energy with respect to pressure at constant density.

        From :math:`u = c_v(T - T_\mathrm{dtm})` and :math:`p = \rho R T`,
        differentiating at constant :math:`\rho`:

        .. math:: \left.\frac{\partial u}{\partial p}\right|_\rho = \frac{1}{\rho(\gamma-1)}

        Parameters
        ----------
        rho : array_like
            Density [kg/m³].
        u : array_like
            Specific internal energy [J/kg].
        out : ndarray, optional
            Pre-allocated output array.

        Returns
        -------
        dudP_rho : ndarray
            Derivative :math:`(\partial u/\partial p)_\rho` [m³/kg].
        """
        if out is None:
            return 1.0 / (rho * self._gamma_m1)
        np.multiply(rho, self._gamma_m1, out=out)
        np.reciprocal(out, out=out)
        return out

    def get_dudrho_P(self, rho, u, out=None):
        r"""Derivative of specific internal energy with respect to density at constant pressure.

        From :math:`u = c_v(T - T_\mathrm{dtm})` and :math:`T = p/(\rho R)`,
        differentiating at constant :math:`p`:

        .. math:: \left.\frac{\partial u}{\partial \rho}\right|_p = -\frac{p}{\rho^2(\gamma-1)}

        Parameters
        ----------
        rho : array_like
            Density [kg/m³].
        u : array_like
            Specific internal energy [J/kg].
        out : ndarray, optional
            Pre-allocated output array.

        Returns
        -------
        dudrho_P : ndarray
            Derivative :math:`(\partial u/\partial \rho)_p` [J·m³/kg²].
        """
        out = self.get_P(rho, u, out=out)
        out /= rho**2 * (self._gamma - 1.0)
        np.negative(out, out=out)
        return out

    def get_gamma(self, rho, u, out=None):
        r"""Ratio of specific heats (constant for a perfect gas).

        .. math:: \gamma = \frac{c_p}{c_v}

        Parameters
        ----------
        rho : array_like
            Density [kg/m³].
        u : array_like
            Specific internal energy [J/kg].
        out : ndarray, optional
            Pre-allocated output array.

        Returns
        -------
        gamma : ndarray
            Ratio of specific heats [--].
        """
        return self._const_nd(rho, u, self._gamma, out)

    def get_h(self, rho, u, out=None):
        r"""Specific enthalpy from density and internal energy.

        Using the definitions of enthalpy :math:`h = u + p/\rho`, internal energy :math:`u = c_v(T-T_\mathrm{dtm})`, and the ideal gas law :math:`p = \rho R T` gives

        .. math:: h = \gamma u + R T_\mathrm{dtm}

        Enthalpy carries an offset dependent on the arbitrary datum state where
        :math:`u = s = 0` at :math:`(p_\mathrm{dtm}, T_\mathrm{dtm})`; only
        changes in :math:`h` are physically meaningful, so :math:`h \neq c_p T`.
        See :ref:`datum-state`.

        Parameters
        ----------
        rho : array_like
            Density [kg/m³].
        u : array_like
            Specific internal energy [J/kg].
        out : ndarray, optional
            Pre-allocated output array.

        Returns
        -------
        h : ndarray
            Specific enthalpy [J/kg].
        """
        out = np.multiply(self._gamma, u, out=out)
        out += self._T_dtm_nd * self._Rgas_nd
        return out

    def get_mu(self, rho, u, out=None):
        r"""Dynamic viscosity (constant for a perfect gas).

        If reference scales are set, then this method returns a quasi-dimensional viscosity in units of [m] --- see `Reference Scales`_ for details.

        Parameters
        ----------
        rho : array_like
            Density [kg/m³].
        u : array_like
            Specific internal energy [J/kg].
        out : ndarray, optional
            Pre-allocated output array.

        Returns
        -------
        mu : ndarray
            Dynamic viscosity [kg/m/s].
        """
        return self._const_nd(rho, u, self._mu_nd, out)

    def get_P(self, rho, u, out=None):
        r"""Pressure from density and internal energy.

        From the ideal gas law and the definition of internal energy for a perfect gas,
        :math:`u = c_v (T - T_\mathrm{dtm})`

        .. math:: p = \rho R \left(\frac{u}{c_v} + T_\mathrm{dtm}\right)

        Parameters
        ----------
        rho : array_like
            Density [kg/m³].
        u : array_like
            Specific internal energy [J/kg].
        out : ndarray, optional
            Pre-allocated output array.

        Returns
        -------
        P : ndarray
            Pressure [Pa].
        """
        out = np.divide(u, self._cv_nd, out=out)
        out += self._T_dtm_nd
        out *= rho
        out *= self._Rgas_nd
        return out

    def get_Pr(self, rho, u, out=None):
        r"""Prandtl number (constant for a perfect gas).

        .. math:: \mathit{Pr} = \frac{\mu c_p}{\kappa}

        Parameters
        ----------
        rho : array_like
            Density [kg/m³].
        u : array_like
            Specific internal energy [J/kg].
        out : ndarray, optional
            Pre-allocated output array.

        Returns
        -------
        Pr : ndarray
            Prandtl number [--].
        """
        return self._const_nd(rho, u, self._Pr, out)

    def get_Rgas(self, rho, u, out=None):
        r"""Specific gas constant (constant for a perfect gas).

        .. math:: R = c_p - c_v = \frac{(\gamma - 1)\, c_p}{\gamma}

        Parameters
        ----------
        rho : array_like
            Density [kg/m³].
        u : array_like
            Specific internal energy [J/kg].
        out : ndarray, optional
            Pre-allocated output array.

        Returns
        -------
        R : ndarray
            Specific gas constant [J/kg/K].
        """
        return self._const_nd(rho, u, self._Rgas_nd, out)

    def get_s(self, rho, u, out=None):
        r"""Specific entropy from density and internal energy.

        Using the Gibbs relation for a perfect gas, with the datum state
        :math:`(p_\mathrm{dtm}, T_\mathrm{dtm})` defining zero entropy

        .. math::

            s = c_p \ln\!\frac{T}{T_\mathrm{dtm}} - R \ln\!\frac{p}{p_\mathrm{dtm}}

        where :math:`T = u/c_v + T_\mathrm{dtm}` and :math:`p = \rho R T`.

        Entropy is defined relative to the arbitrary datum state where
        :math:`u = s = 0` at :math:`(p_\mathrm{dtm}, T_\mathrm{dtm})`; only
        changes in :math:`s` are physically meaningful. See :ref:`datum-state`.

        Parameters
        ----------
        rho : array_like
            Density [kg/m³].
        u : array_like
            Specific internal energy [J/kg].
        out : ndarray, optional
            Pre-allocated output array.

        Returns
        -------
        s : ndarray
            Specific entropy [J/kg/K].
        """
        # s = cp*ln(T/T_dtm) - R*ln(P/P_dtm), with T = u/cv + T_dtm and P = rho*R*T.
        T = u / self._cv_nd
        P = rho * self._Rgas_nd
        if not isinstance(T, np.ndarray) or T.ndim == 0:
            # Scalar/0-d setup path (cold): keep the plain expression -- in-place
            # ufuncs with out= reject numpy scalars, and allocation is irrelevant.
            T = T + self._T_dtm_nd
            P = P * T
            return np.subtract(
                self._cp_nd * np.log(T / self._T_dtm_nd),
                self._Rgas_nd * np.log(P / self._P_dtm_nd),
                out=out,
            )
        # Array hot path: fold the two log terms in place in the T and P work
        # buffers, collapsing the ~6 temporaries the expression form allocated.
        # T (=u/cv) and P (=rho*R) are fresh (never alias the inputs), so the
        # in-place steps are safe; P captures T before T is reused for term 1.
        T += self._T_dtm_nd
        P *= T
        T /= self._T_dtm_nd
        np.log(T, out=T)
        T *= self._cp_nd  # T = cp*ln(T/T_dtm)
        P /= self._P_dtm_nd
        np.log(P, out=P)
        P *= self._Rgas_nd  # P = R*ln(P/P_dtm)
        return np.subtract(T, P, out=out)

    def get_T(self, rho, u, out=None):
        r"""Temperature from density and internal energy.

        Rearranging the definition of internal energy for a perfect gas :math:`u = c_v (T - T_\mathrm{dtm})` gives

        .. math:: T = \frac{u}{c_v} + T_\mathrm{dtm}

        Parameters
        ----------
        rho : array_like
            Density [kg/m³].
        u : array_like
            Specific internal energy [J/kg].
        out : ndarray, optional
            Pre-allocated output array.

        Returns
        -------
        T : ndarray
            Temperature [K].
        """
        out = np.divide(u, self._cv_nd, out=out)
        out += self._T_dtm_nd
        return out

    def change_datum(self, P_dtm, T_dtm):
        """Get a new :class:`PerfectFluid` with shifted datum.

        The new instance will have zero internal energy and entropy at the specified pressure and temperature.

        Parameters
        ----------
        P_dtm : float
            New datum pressure [Pa].
        T_dtm : float
            New datum temperature [K].

        Returns
        -------
        fluid_new : PerfectFluid
            New fluid instance with shifted and entropy datum.

        """
        fluid_new = self.__class__(
            cp=self._cp,
            gamma=self._gamma,
            mu=self._mu,
            Pr=self._Pr,
            P_dtm=P_dtm,
            T_dtm=T_dtm,
            rho_ref=self.rho_ref,
            V_ref=self.V_ref,
            Rgas_ref=self.Rgas_ref,
        )
        return fluid_new

    def change_ref(self, rho_ref=None, V_ref=None, Rgas_ref=None):
        """Make a new :class:`PerfectFluid` with different reference scales.

        Omitted reference scales default to the current instance's reference scales, so only the scales that need to be changed must be specified.

        Parameters
        ----------
        rho_ref : float, optional
            New reference density for non-dimensionalisation.
        V_ref : float, optional
            New reference velocity for non-dimensionalisation.
        Rgas_ref : float, optional
            New reference gas constant for non-dimensionalisation.

        Returns
        -------
        fluid_new : PerfectFluid
            New fluid instance with the same properties but different reference scales.

        """
        return self.__class__(
            cp=self._cp,
            gamma=self._gamma,
            mu=self._mu,
            Pr=self._Pr,
            P_dtm=self._P_dtm,
            T_dtm=self._T_dtm,
            rho_ref=rho_ref if rho_ref is not None else self.rho_ref,
            V_ref=V_ref if V_ref is not None else self.V_ref,
            Rgas_ref=Rgas_ref if Rgas_ref is not None else self.Rgas_ref,
        )
