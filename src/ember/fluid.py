r"""Working fluid interface and equation of state implementations.

This module defines an interface for computing thermodynamic properties of
working fluids enabling manipulations of flow fields independent of the
underlying equations of state. The abstraction cleanly separates thermodynamic
relations from the flow solver, allowing easy extension to real gas models or
tabulated properties.

Two implementations are provided: :class:`PerfectFluid`, for ideal gases with
constant specific heats, and :class:`RealFluid`, a thermodynamically consistent
real gas built from a fitted entropy surface. Coefficients for the latter are
produced offline by :mod:`ember.realgas_fit`.

A fluid instance is immutable and only stores intrinsic fluid properties that never change, such
as specific heats for a perfect gas, or the fluid species for a real
fluid. These must passed into the constructor on initialisation. Storage of the actual flow solution is the responsibility of a :class:`~ember.block.Block`.

Get and set methods
===================

The basic state variables are density and internal energy, :math:`(\rho, u)`, because these are the most natural in a conservative computational fluid dynamics solver. An equation of state must provide two types of methods: `get_?` and `set_?_?`.

- `set_x_y(x,y)`: Take thermodynamic properties :math:`(x, y)` and return :math:`(\rho, u)`.
- `get_z(rho, u, out=None)`: Take :math:`(\rho, u)` and return thermodynamic property :math:`z`.

All methods support both scalar and array inputs, where the inputs must be broadcastable against each other. The output will have the broadcasted shape. Constructor inputs are cast to single-precision floats, and outputs will remain single-precision if all inputs are single-precision. Supplying an `out` keyword argument to `get_?` methods allows the output to be written into a pre-allocated array, following NumPy conventions, which may improve performance by avoiding temporary array allocations.

.. _datum-state:

Datum state
===========

Only changes in internal energy, enthalpy, and entropy are physically
meaningful. Therefore, we have freedom to set the physical state at which
these properties are zero, to improve numerics and reduce precision errors due
to subtracting two large floats. We define a thermodynamic datum
:math:`(p_\mathrm{dtm}, T_\mathrm{dtm})` where :math:`u = s = 0`
simultaneously. Enthalpy at the datum is not zero because of the pressure term
in :math:`h = u + p/\rho`.

It is possible to shift the datum state of a fluid instance using
``change_datum``, which returns a new instance with the same properties but
shifted datum. The current datum is accessible via
:attr:`~PerfectFluid.P_dtm` and :attr:`~PerfectFluid.T_dtm` attributes.


.. _reference-scales:

Reference scales
================

The constructors for fluid instances take optional reference scales for non-dimensionalisation, which default to unity such that all quantities are in SI units. If reference scales are provided, all inputs and outputs are taken as non-dimensional. The advantage of setting reference scales is improved numerical precision when working with non-dimensional quantities all of order unity.

The user specifies:

- :math:`\rho_\mathrm{ref}\,`: density [kg/m\ :sup:`3`], :attr:`~PerfectFluid.rho_ref`
- :math:`V_\mathrm{ref}\,`: velocity [m/s], :attr:`~PerfectFluid.V_ref`
- :math:`R_\mathrm{ref}\,`: gas constant [J/kg/K], :attr:`~PerfectFluid.Rgas_ref`

and the class forms the following derived reference scales:

- :math:`p_\mathrm{ref} = \rho_\mathrm{ref} V_\mathrm{ref}^2\,`: dynamic pressure [Pa], :attr:`~PerfectFluid.P_ref`
- :math:`u_\mathrm{ref} = V_\mathrm{ref}^2\,`: specific energy [J/kg], :attr:`~PerfectFluid.u_ref`
- :math:`T_\mathrm{ref} = V_\mathrm{ref}^2 / R_\mathrm{ref}\,`: temperature [K], :attr:`~PerfectFluid.T_ref`
- :math:`(\rho V)_\mathrm{ref} = \rho_\mathrm{ref} V_\mathrm{ref}\,`: mass flux [kg/m\ :sup:`2`/s], :attr:`~PerfectFluid.rhoV_ref`

Equations of state are unchanged when all quantities are scaled consistently. For example, taking the ideal gas law :math:`p = \rho R T` and dividing through by the reference pressure :math:`\rho_\mathrm{ref} V_\mathrm{ref}^2` gives

.. math:: \frac{p}{\rho_\mathrm{ref} V_\mathrm{ref}^2} = \frac{\rho}{\rho_\mathrm{ref}} \frac{R}{R_\mathrm{ref}} \frac{T}{V_\mathrm{ref}^2 / R_\mathrm{ref}} = \frac{\rho}{\rho_\mathrm{ref}} \frac{R}{R_\mathrm{ref}} \frac{T}{T_\mathrm{ref}}

Transport properties such as viscosity and thermal conductivity are an exception to this scaling, and would require an additional reference length to make fully non-dimensional. So when references are provided, transport properties have dimensions of meters.

We can get a new instance with different reference scales using the :meth:`PerfectFluid.change_ref` method.

"""

import inspect

import numpy as np
from abc import ABC, abstractmethod
from ember import util
import ember.fortran
import ember.realgas_fit

_leg = np.polynomial.legendre

# Highest Legendre order the real-gas kernel can hold. It sizes its basis
# buffers at compile time rather than allocating them per call, so a surface
# fitted beyond this has to take the numpy path -- and MUST, since the kernel
# would otherwise write past the end of them. Mirrors MAXORD in
# _fortran/fluid_real.f90; the two are pinned together by a test.
_REAL_KERNEL_MAXORD = 31


def _last_nonzero_rows(coef):
    """Rows worth visiting in each column of a coefficient surface.

    A total-order fit keeps only the terms with i + j <= order, so most of the
    surface is exactly zero and the kernel would otherwise multiply by it.
    Returned per column rather than as a mask because the zeros sit in a
    trailing block: shortening the loop skips them without a branch, and an
    interior zero is left alone, so this makes no assumption about the basis
    beyond what is actually there. A dense surface gives back its full extent.

    Parameters
    ----------
    coef : ndarray
        Two-dimensional coefficient array.

    Returns
    -------
    ndarray of int32
        One count per column, zero where the column is entirely zero.

    """
    nz = coef != 0.0
    nrow = nz.shape[0]
    counts = np.where(nz.any(axis=0), nrow - np.argmax(nz[::-1], axis=0), 0)
    return np.asfortranarray(counts.astype(np.int32))


def _plain(value):
    """Strip a value down to what a plain JSON or YAML dumper can write.

    Arrays become nested lists and every scalar becomes a Python float, because
    the two things that quietly break a dumped fluid are numpy scalars, which
    :mod:`json` refuses outright, and tuples, which PyYAML tags
    ``!!python/tuple`` and so writes a file only Python can read back.
    """
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, str):
        return value
    return float(value)


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
    def _kwargs(self):
        """Return the constructor arguments that define this fluid.

        Every value is a plain Python float, tuple or array --- never a
        ``numpy`` scalar --- so that a caller can splat the result straight back
        into the constructor and get an identical fluid. Derived quantities are
        absent: they are all functions of these.

        Returns
        -------
        kwargs : dict
            Keyword arguments reproducing this instance.
        """
        raise NotImplementedError()

    @staticmethod
    def _const_nd(rho_nd, u_nd, value, out):
        """Return a constant broadcast to the shape of (rho_nd, u_nd)."""
        if out is None:
            return util.full(np.broadcast(rho_nd, u_nd).shape, value)
        out[...] = value
        return out

    @classmethod
    def from_dict(cls, data):
        """Build a fluid from a dict written by :meth:`to_dict`.

        Called on :class:`_Fluid` it dispatches on the ``type`` key, so a saved
        fluid can be read back without the caller knowing which equation of
        state wrote it. Called on a concrete class it checks that ``type``
        names that class, so loading the wrong file is an error rather than a
        constructor failure several arguments deep.

        Parameters
        ----------
        data : dict
            As returned by :meth:`to_dict`. Not modified.

        Returns
        -------
        fluid : _Fluid
            A new instance of the class named by ``data["type"]``.
        """
        data = dict(data)

        name = data.pop("type", None)
        if name is None:
            raise ValueError(
                f"A fluid dict needs a 'type' key, one of {sorted(_FLUID_TYPES)}."
            )

        if cls is _Fluid:
            if name not in _FLUID_TYPES:
                raise ValueError(
                    f"Unknown fluid type {name!r}. "
                    f"Available types: {sorted(_FLUID_TYPES)}."
                )
            return _FLUID_TYPES[name].from_dict({"type": name, **data})

        if name != cls.__name__:
            raise ValueError(f"{cls.__name__}.from_dict was given a {name!r} dict.")

        # Rejected by name rather than ignored. A misspelled key is otherwise
        # silently the constructor default, which for `Pr` is a perfectly
        # plausible fluid and so never gets noticed.
        allowed = set(inspect.signature(cls).parameters)
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(
                f"Unknown key(s) for {cls.__name__}: {sorted(unknown)}. "
                f"Valid keys: {sorted(allowed)}."
            )

        return cls(**data)

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

    def get_P_h_T(self, rho, u, out_P=None, out_h=None, out_T=None):
        """Pressure, enthalpy and temperature together, from density and energy.

        A batched form of :meth:`get_P`, :meth:`get_h` and :meth:`get_T`, for
        callers that want all three from the same state -- the solver does, once
        per Runge-Kutta stage. Evaluating them separately re-reads ``rho`` and
        ``u`` three times, and for most equations of state the three share
        nearly all of their work.

        Deliberately NOT abstract: this base implementation simply delegates to
        the three single-property methods, so every fluid -- present or future
        -- is correct without implementing anything. A subclass may override it
        with a fused evaluation purely as an optimisation, and is not obliged
        to. :class:`PerfectFluid` does.

        Parameters
        ----------
        rho : array_like
            Density [kg/m³].
        u : array_like
            Specific internal energy [J/kg].
        out_P, out_h, out_T : ndarray, optional
            Pre-allocated output arrays.

        Returns
        -------
        tuple of ndarray
            ``(P, h, T)``.

        """
        return (
            self.get_P(rho, u, out=out_P),
            self.get_h(rho, u, out=out_h),
            self.get_T(rho, u, out=out_T),
        )

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
    def change_datum(self, P_dtm, T_dtm):
        """Return a new instance with datum shifted to (P_dtm, T_dtm).

        A pure factory: the returned fluid carries the same properties on the
        new datum, and no field values are transformed. To move a *stored* flow
        field onto another datum, pass the new fluid to
        :meth:`ember.block.Block.set_fluid`, which reads the state out through
        the old fluid and re-expresses it through the new one.

        Note that ``T_dtm = 0`` is not representable: the datum sets
        ``u = s = 0`` simultaneously, and entropy carries a
        :math:`\\ln(T/T_\\mathrm{dtm})` term that is singular there.
        """
        raise NotImplementedError()

    def change_ref(self, rho_ref=None, V_ref=None, Rgas_ref=None):
        """Return a new instance with different reference scales."""
        raise NotImplementedError("Subclasses must implement change_ref")

    def to_dict(self):
        """Return a portable record of this fluid, ready for :meth:`from_dict`.

        The dict holds plain floats, strings and nested lists only, so it can be
        written with :mod:`json` or a plain YAML dumper and read by anything.
        That is the point of it: a :class:`RealFluid` surface costs a CoolProp
        table and an offline fit to produce, and until now the only way to keep
        one was to pickle the grid it happened to be attached to.

        Everything the constructor takes is included, the reference scales and
        the datum among them, so the round trip is exact. A fluid reloaded into
        another run therefore arrives with the scales of the run that wrote it;
        :meth:`change_ref` is how to move it onto new ones.

        Returns
        -------
        data : dict
            Constructor arguments plus a ``type`` key naming the class.
        """
        return {"type": type(self).__name__, **_plain(self._kwargs())}

    @property
    def P_dtm(self):
        r"""Datum pressure :math:`p_\mathrm{dtm}` where :math:`u = s = 0` [Pa].

        User-selectable as discussed in  :ref:`datum-state`.
        """
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
        r"""Datum temperature :math:`T_\mathrm{dtm}` where :math:`u = s = 0` [K].

        User-selectable as discussed in  :ref:`datum-state`.
        """
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

    def _kwargs(self):
        """Constructor arguments reproducing this fluid; see :meth:`_Fluid._kwargs`."""
        return {
            "cp": float(self._cp),
            "gamma": float(self._gamma),
            "mu": float(self._mu),
            "Pr": float(self._Pr),
            "P_dtm": float(self._P_dtm),
            "T_dtm": float(self._T_dtm),
            "rho_ref": float(self.rho_ref),
            "V_ref": float(self.V_ref),
            "Rgas_ref": float(self.Rgas_ref),
        }

    def _rebuild(self, **over):
        """New instance with the same properties and selected overrides."""
        return self.__class__(**{**self._kwargs(), **over})

    def set_h_s(self, h, s):
        r"""Density and internal energy from specific enthalpy and entropy.

        Temperature is recovered from :math:`h`, then pressure from :math:`s`
        , then :math:`\rho` and :math:`u` follow from
        :meth:`set_P_T`,

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

        Temperature is recovered from :math:`h`,

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

        Temperature follows from the ideal gas law, giving,

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
        :math:`u` follow from :meth:`set_P_T`,

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

        From the ideal gas law and the definition of internal energy,

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

        From the definition of internal energy,

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
        if not isinstance(out, np.ndarray):
            # Scalar/0-d setup path (cold): in-place ufuncs with out= reject
            # numpy scalars, so fall back to the plain expression.
            return self._cv_nd / out
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
        if not isinstance(out, np.ndarray):
            # Scalar/0-d setup path (cold): in-place ufuncs with out= reject
            # numpy scalars, so fall back to the plain expression.
            return -out / (rho**2 * (self._gamma - 1.0))
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

    def get_P_h_T(self, rho, u, out_P=None, out_h=None, out_T=None):
        """Batched evaluation of pressure, enthalpy and temperature.

        The base class calls :meth:`get_P`, :meth:`get_h` and :meth:`get_T` in sequence, so this method is an optional override for subclasses that can compute all three in a single pass over the state. The CFD solver calls
        this method once per Runge-Kutta stage, so a fused evaluation can save time.

        Parameters
        ----------
        rho : array_like
            Density [kg/m³].
        u : array_like
            Specific internal energy [J/kg].
        out_P, out_h, out_T : ndarray, optional
            Pre-allocated output arrays.

        Returns
        -------
        tuple of ndarray
            ``(P, h, T)``.
        """
        outs = (out_P, out_h, out_T)
        arrs = (rho, u) + outs
        usable = (
            all(o is not None for o in outs)
            and all(isinstance(a, np.ndarray) for a in arrs)
            and all(a.dtype == np.float32 for a in arrs)
            and all(a.shape == np.shape(rho) for a in arrs)
            and all(a.flags["F_CONTIGUOUS"] or a.flags["C_CONTIGUOUS"] for a in arrs)
            # Flattening without copying is not enough: they must flatten in
            # the same order. This kernel pairs the arrays element by element,
            # so a C-ordered output against an F-ordered input would take each
            # answer to the wrong node -- silently, and only in two dimensions
            # or more, where the two traversals differ.
            and all(np.isfortran(a) == np.isfortran(rho) for a in arrs)
        )
        if not usable:
            return super().get_P_h_T(rho, u, out_P, out_h, out_T)

        # order="A" ravels without copying for either contiguity, so these stay
        # views and the kernel's writes land in the caller's arrays.
        ember.fortran.set_p_h_t_perfect(
            rho=np.ravel(rho, order="A"),
            u=np.ravel(u, order="A"),
            cv=self._cv_nd,
            t_dtm=self._T_dtm_nd,
            rgas=self._Rgas_nd,
            gamma=self._gamma,
            p=np.ravel(out_P, order="A"),
            h=np.ravel(out_h, order="A"),
            t=np.ravel(out_T, order="A"),
        )
        return out_P, out_h, out_T

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

        The new instance will have zero internal energy and entropy at the specified pressure and temperature. See :ref:`datum-state`.

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
        return self._rebuild(P_dtm=P_dtm, T_dtm=T_dtm)

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
        return self._rebuild(
            rho_ref=rho_ref if rho_ref is not None else self.rho_ref,
            V_ref=V_ref if V_ref is not None else self.V_ref,
            Rgas_ref=Rgas_ref if Rgas_ref is not None else self.Rgas_ref,
        )


class RealFluid(_Fluid):
    r"""Real gas defined by a fitted entropy surface.

    Implements the thermodynamically consistent equation of state of Wheeler
    (2024), *Computers and Fluids* 268:106088. A polynomial surface is fitted
    offline to the compressibility factor and integrated analytically to give
    entropy; temperature and pressure are then *derived* from that one surface
    rather than fitted separately, so the thermodynamic relations between them
    hold exactly. Use :mod:`ember.realgas_fit` to produce the coefficients.

    The consistency is the point. An interpolated lookup table satisfies its own
    tabulated values but not the relations between them, and the resulting
    mismatch acts like a non-equilibrium process, creating spurious entropy
    wherever gradients are steep. Wheeler shows a 62k-point table was needed to
    bring turbine entropy rise within 1.4 percent of this method.

    Evaluation
    ----------
    Everything descends from the entropy surface and its partials, via the
    combined first and second law :math:`\mathrm{d}u = T\,\mathrm{d}s +
    (p/\rho^2)\,\mathrm{d}\rho`:

    .. math::

        T = \left[\left(\frac{\partial s}{\partial u}\right)_\rho\right]^{-1},
        \qquad
        p = -\rho^2 T \left(\frac{\partial s}{\partial \rho}\right)_u

    The inverse ``set_*`` methods have no closed form and are solved by Newton
    iteration; see :meth:`set_P_rho` and :meth:`set_P_T`.

    Domain of validity
    ------------------
    The polynomials mean nothing outside the box they were fitted over, so
    ``rho_lim`` and ``u_lim`` are enforced: iterates are clamped to the box, and
    a solve that cannot meet its tolerance, or that returns no number at all,
    raises rather than returning a state the fit does not describe.

    A density passed in rather than solved for -- by :meth:`set_rho_s`,
    :meth:`set_P_rho` and :meth:`set_T_rho` -- is checked against the box
    before the iteration starts. Nothing later could catch it: the fitted
    surface extrapolates smoothly, so the solve would converge on a state that
    is self-consistent with a polynomial nobody fitted out there.

    Limitations
    -----------
    Transport properties are constant, as for :class:`PerfectFluid`. Wheeler's
    fitted viscosity and conductivity surfaces are not implemented, so ``mu``
    and ``Pr`` are supplied directly. The specific heat carried into the
    conductivity is not among them: :attr:`~ember.block.Block.cp_nd` reads
    :meth:`get_cp` at every node, so the heat flux sees the same state
    dependence as the rest of the surface.

    Parameters
    ----------
    alpha : array_like
        Two-dimensional Legendre coefficients of the compressibility factor
        :math:`Z(\hat\rho, \hat u)` [--], in the normalised coordinates of
        :ref:`normalised-coordinates`.
    beta : array_like
        One-dimensional Legendre coefficients of :math:`s/R` along the reference
        isochor [--].
    rho_lim : tuple
        ``(min, max)`` density bounds of the fit box [kg/m³].
    u_lim : tuple
        ``(min, max)`` internal energy bounds of the fit box [J/kg], on the same
        datum as the data ``beta`` was fitted to.
    rho_isochor : float
        Density of the isochor the entropy integral starts from [kg/m³]. This is
        Wheeler's :math:`\rho_\mathrm{ref}`, and is unrelated to
        :attr:`rho_ref`, the non-dimensionalisation scale.
    Rgas : float
        Specific gas constant [J/kg/K]. Converts the two dimensionless
        coefficient arrays into entropy, and is what :meth:`get_Rgas` reports.
    mu : float
        Dynamic viscosity [kg/m/s].
    Pr : float
        Prandtl number [--].
    P_dtm : float, optional
        Datum pressure where u = 0 and s = 0 [Pa]. Must lie in the fit box.
    T_dtm : float, optional
        Datum temperature where u = 0 and s = 0 [K]. Must lie in the fit box.
    rho_ref : float, optional
        Reference density for non-dimensionalisation.
    V_ref : float, optional
        Reference velocity for non-dimensionalisation.
    Rgas_ref : float, optional
        Reference gas constant for non-dimensionalisation.

    """

    # Newton iteration limit and relative tolerance for the set_* inversions.
    # Newton converges quadratically from the perfect-gas guess, so the limit is
    # a backstop rather than a working iteration count; the tolerance sits just
    # above where float32 arithmetic stalls.
    _NEWTON_ITER = 40
    _NEWTON_RTOL = 1e-6
    # Acceptance tolerance for the converged solve, loose enough that a
    # float32 surface evaluation cannot trip it but tight enough that a
    # genuinely unreachable state is reported rather than returned.
    _VERIFY_RTOL = 1e-4

    # Ratio below which a Newton step counts as still making progress. These
    # solves converge quadratically, so a step that is genuinely closing on the
    # answer shrinks by orders of magnitude at a time and is nowhere near this;
    # only a step that has hit the arithmetic's floor sits beside its
    # predecessor. Needed because _NEWTON_RTOL alone cannot end the loop: the
    # step is a maximum over every node, so the noisiest one decides, and past a
    # few thousand nodes that sample never drops below a tolerance finer than
    # float32 resolves. The solve is right long before then; it just kept going.
    _NEWTON_STALL = 0.9

    # Step below which the stall test is allowed to end the loop. That test
    # reads a step which failed to shrink as a step at the arithmetic's floor,
    # which is only true once Newton is in its quadratic regime. On the
    # approach from a cold seed it is not: the step is relative in density, so
    # a solve walking from the box centre out to the low-density edge takes two
    # large steps of much the same size, and a step cut short by the box clamp
    # can be larger than the one before it. Either reads as a stall and stops a
    # solve that was closing on the answer -- which is how a datum inside the
    # box came back reported as outside it. A solve at the floor is orders
    # below this, so the gate costs the early exit nothing.
    _NEWTON_SETTLED = 1e-2

    def __init__(
        self,
        alpha,
        beta,
        rho_lim,
        u_lim,
        rho_isochor,
        Rgas,
        mu,
        Pr,
        P_dtm=1e5,
        T_dtm=300.0,
        rho_ref=1.0,
        V_ref=1.0,
        Rgas_ref=1.0,
    ):
        super().__init__(rho_ref, V_ref, Rgas_ref)

        self._alpha = np.atleast_2d(np.asarray(alpha, dtype=np.float64))
        self._beta = np.atleast_1d(np.asarray(beta, dtype=np.float64))
        self._rho_lim = (float(rho_lim[0]), float(rho_lim[1]))
        self._u_lim = (float(u_lim[0]), float(u_lim[1]))
        self._rho_isochor = float(rho_isochor)
        self._Rgas = np.float32(Rgas)
        self._mu = np.float32(mu)
        self._mu_nd = np.float32(mu / (rho_ref * V_ref))
        self._Rgas_nd = np.float32(Rgas / Rgas_ref)
        self._Pr = np.float32(Pr)
        self._P_dtm = np.float32(P_dtm)
        self._T_dtm = np.float32(T_dtm)

        if not self._rho_lim[1] > self._rho_lim[0] > 0.0:
            raise ValueError(f"rho_lim must be increasing and positive, got {rho_lim}")
        if not self._u_lim[1] > self._u_lim[0]:
            raise ValueError(f"u_lim must be increasing, got {u_lim}")
        if not self._rho_lim[0] <= self._rho_isochor <= self._rho_lim[1]:
            raise ValueError(
                f"rho_isochor={rho_isochor} must lie within rho_lim={rho_lim}"
            )
        if self._beta.size > self._alpha.shape[1]:
            raise ValueError(
                f"beta has {self._beta.size} coefficients but alpha carries "
                f"{self._alpha.shape[1]}; the entropy surface has one column "
                "per internal-energy order in alpha, so there is nowhere to "
                "put the rest."
            )
        if Rgas <= 0.0:
            raise ValueError(f"Rgas={Rgas} must be positive.")
        if mu <= 0.0:
            raise ValueError(f"mu={mu} must be positive.")
        if Pr <= 0.0:
            raise ValueError(f"Pr={Pr} must be positive.")
        if T_dtm <= 0.0:
            raise ValueError(f"T_dtm={T_dtm} must be positive.")
        if P_dtm <= 0.0:
            raise ValueError(f"P_dtm={P_dtm} must be positive.")

        # Two-pass construction. The datum is the internal energy and entropy at
        # (P_dtm, T_dtm), which can only be found by inverting the surface --
        # but the surface needs the datum to be built. So build it once in raw
        # units with no datum shift, invert to locate the datum, then rebuild
        # with the datum and reference scales folded in.
        self._companion = None
        self._configure(0.0, 0.0, 1.0, 1.0, 1.0, dtype=np.float64)
        rho_dtm, u_dtm = self.set_P_T(float(P_dtm), float(T_dtm))
        s_dtm = self._entropy(rho_dtm, u_dtm)
        self._configure(
            float(u_dtm), float(s_dtm), rho_ref, V_ref, Rgas_ref, dtype=np.float32
        )

        self._companion = self._build_companion(P_dtm, T_dtm)

    def _build_companion(self, P_dtm, T_dtm):
        """Perfect gas approximating this fluid, used to seed the Newton solves.

        The ``set_*`` interface is stateless -- the nonreflecting boundary calls
        :meth:`set_P_rho` with no previous internal energy to warm-start from --
        so the initial guess has to come from somewhere. A perfect gas matched
        to this fluid at the centre of its box has analytic inverses and lands
        within a few tens of percent even at a compressibility factor well away
        from one, which Newton clears in a handful of iterations.
        """
        rho_c = 0.5 * (self._rho_box_nd[0] + self._rho_box_nd[1])
        u_c = 0.5 * (self._u_box_nd[0] + self._u_box_nd[1])
        gamma = float(self.get_gamma(rho_c, u_c))
        cp = float(self.get_cp(rho_c, u_c)) * float(self.Rgas_ref)
        return PerfectFluid(
            cp=cp,
            gamma=gamma,
            mu=float(self._mu),
            Pr=float(self._Pr),
            P_dtm=float(P_dtm),
            T_dtm=float(T_dtm),
            rho_ref=float(self.rho_ref),
            V_ref=float(self.V_ref),
            Rgas_ref=float(self.Rgas_ref),
        )

    def _configure(self, u_dtm_abs, s_dtm, rho_ref, V_ref, Rgas_ref, dtype):
        """Compose the fit box, datum and reference scales into one surface.

        The normalised coordinates are affine in density and internal energy, so
        the datum shift and the reference scaling are just further affine maps
        and collapse into the same two constants per variable. The hot path
        therefore never applies them separately: one multiply-add takes a stored
        non-dimensional state straight to the normalised coordinate.

        Folding the datum in here, rather than subtracting it from the result,
        is what keeps single precision usable. Entropy over a typical table
        varies by a few percent of its absolute level, so forming
        ``s = (large) - (large)`` once per node per Runge-Kutta stage would lose
        most of the significant digits.
        """
        u_ref = V_ref**2
        rho_m = 0.5 * (self._rho_lim[1] + self._rho_lim[0])
        rho_f = 0.5 * (self._rho_lim[1] - self._rho_lim[0])
        u_m = 0.5 * (self._u_lim[1] + self._u_lim[0])
        u_f = 0.5 * (self._u_lim[1] - self._u_lim[0])

        # rho_hat = rho_nd*_xa + _xb, u_hat = u_nd*_ya + _yb
        self._xa = dtype(rho_ref / rho_f)
        self._xb = dtype(-rho_m / rho_f)
        self._ya = dtype(u_ref / u_f)
        self._yb = dtype((u_dtm_abs - u_m) / u_f)

        self._rho_box_nd = (
            dtype(self._rho_lim[0] / rho_ref),
            dtype(self._rho_lim[1] / rho_ref),
        )
        self._u_box_nd = (
            dtype((self._u_lim[0] - u_dtm_abs) / u_ref),
            dtype((self._u_lim[1] - u_dtm_abs) / u_ref),
        )

        # Characteristic magnitudes, used to scale convergence tests where a
        # target may legitimately pass through zero at the datum.
        self._u_scale = dtype(0.5 * (self._u_lim[1] - self._u_lim[0]) / u_ref)
        self._s_scale = dtype(float(self._Rgas) / Rgas_ref)

        # Density integral of the compressibility surface, in closed form.
        c = rho_m / rho_f
        x0 = (self._rho_isochor - rho_m) / rho_f
        D, Lam = ember.realgas_fit.entropy_integral(self._alpha, c, x0)

        # Assemble the non-dimensional entropy surface,
        #     s = legval2d(x, y, Sc) + legval(y, Sl)*log(rho)
        # The log argument is rho_hat + c = rho/rho_f, which differs from the
        # stored rho by a constant factor; that constant, the isochor entropy
        # polynomial and the datum offset are all functions of internal energy
        # alone, so they fold into the constant-density row of Sc.
        k = float(self._Rgas) / Rgas_ref
        Sl = -k * Lam
        Sc = -k * D
        # Sc has one column per internal-energy order in alpha, and the
        # constructor has already refused a beta longer than that, so this
        # pads a short isochor polynomial and never truncates a long one.
        beta = np.zeros(Sc.shape[1])
        beta[: self._beta.size] = self._beta
        Sc[0, :] += k * beta + Sl * np.log(rho_ref / rho_f)
        Sc[0, 0] -= s_dtm / Rgas_ref

        self._Sc = Sc.astype(dtype)
        # The two first-derivative surfaces go to the Fortran kernel in
        # get_P_h_T, which wants them column-major; f2py would otherwise copy
        # them on every call. The copy would be tiny, but so is this.
        self._Sc_x = np.asfortranarray(_leg.legder(Sc, axis=0).astype(dtype))
        self._Sc_y = np.asfortranarray(_leg.legder(Sc, axis=1).astype(dtype))
        self._nzx = _last_nonzero_rows(self._Sc_x)
        self._nzy = _last_nonzero_rows(self._Sc_y)
        self._Sc_xx = _leg.legder(Sc, 2, axis=0).astype(dtype)
        self._Sc_xy = _leg.legder(_leg.legder(Sc, axis=0), axis=1).astype(dtype)
        self._Sc_yy = _leg.legder(Sc, 2, axis=1).astype(dtype)
        self._Sl = Sl.astype(dtype)
        self._Sl_y = _leg.legder(Sl).astype(dtype)
        self._Sl_yy = _leg.legder(Sl, 2).astype(dtype)

        # The same six surfaces again, padded to a common extent and stacked,
        # for the kernel behind _partials2. Differentiating shortens a
        # different axis each time, so passing them separately would mean six
        # pairs of bounds saying nothing the counts do not already say. The
        # padding is free: those zeros fall past the per-column counts, exactly
        # like the ones a total-order fit leaves.
        surfaces = (
            self._Sc,
            self._Sc_x,
            self._Sc_y,
            self._Sc_xx,
            self._Sc_xy,
            self._Sc_yy,
        )
        nrow = max(c.shape[0] for c in surfaces)
        ncol = max(c.shape[1] for c in surfaces)
        stack = np.zeros((nrow, ncol, len(surfaces)), dtype=dtype)
        counts = np.zeros((ncol, len(surfaces)), dtype=np.int32)
        for m, coef in enumerate(surfaces):
            stack[: coef.shape[0], : coef.shape[1], m] = coef
            counts[: coef.shape[1], m] = _last_nonzero_rows(coef)
        self._Sc_stack = np.asfortranarray(stack)
        self._Sc_stack_nz = np.asfortranarray(counts)

        lines = (self._Sl, self._Sl_y, self._Sl_yy)
        stack1 = np.zeros((ncol, len(lines)), dtype=dtype)
        for m, coef in enumerate(lines):
            stack1[: coef.size, m] = coef
        self._Sl_stack = np.asfortranarray(stack1)

        # Which of those six surfaces a scalar solve needs when matching each
        # property, as indices into the stack, with the code that names the
        # combination to close with. A solve wants the value and its energy
        # derivative and nothing else, which is two of the six for entropy and
        # temperature and four for pressure and enthalpy; the rest would be
        # evaluated and discarded. Order matches set_f_fu_real's expectations.
        self._solve_plan = {
            "s": (np.asfortranarray(np.array([1, 3], np.int32)), 1),
            "T": (np.asfortranarray(np.array([3, 6], np.int32)), 2),
            "P": (np.asfortranarray(np.array([2, 3, 5, 6], np.int32)), 3),
        }

    def _entropy(self, rho, u):
        """Non-dimensional entropy alone, without the partials."""
        x, y = self._hats(rho, u)
        return _leg.legval2d(x, y, self._Sc) + _leg.legval(y, self._Sl) * np.log(rho)

    def _floor(self, prop):
        """Magnitude below which a target counts as zero for convergence tests.

        Pressure and temperature are strictly positive, so their own value is a
        sound scale. Entropy and enthalpy are measured from an arbitrary datum
        and pass through zero there, so they need a characteristic scale of
        their own or the relative test becomes meaningless.
        """
        if prop == "s":
            return self._s_scale
        if prop == "h":
            return self._u_scale
        return 0.0

    def _guess_2d(self, method, a, b):
        """Seed a two-dimensional solve with (rho, u) from the companion gas."""
        if self._companion is None:
            # Only during construction, before the companion can be built.
            rho0 = 0.5 * (self._rho_box_nd[0] + self._rho_box_nd[1])
            u0 = 0.5 * (self._u_box_nd[0] + self._u_box_nd[1])
            return np.broadcast_to(rho0, np.broadcast(a, b).shape), u0
        # The companion is a perfect gas, so a target this fit cannot reach may
        # be one the companion cannot reach either -- a negative temperature
        # under a logarithm, say. A guess is only a starting point and an
        # unreachable state is caught on acceptance, so the seed is allowed to
        # come back as nan without complaining on the way.
        with np.errstate(invalid="ignore", divide="ignore"):
            rho0, u0 = getattr(self._companion, method)(a, b)
        return np.clip(rho0, *self._rho_box_nd), np.clip(u0, *self._u_box_nd)

    def _guess_u(self, method, *args):
        """Seed a scalar solve with an internal energy from the companion gas."""
        if self._companion is None:
            return 0.5 * (self._u_box_nd[0] + self._u_box_nd[1])
        with np.errstate(invalid="ignore", divide="ignore"):
            u0 = getattr(self._companion, method)(*args)[1]
        return np.clip(u0, *self._u_box_nd)

    def _kernel_P_h_T(self, rho, u, outs=(None, None, None)):
        """Pressure, enthalpy and temperature from the Fortran kernel.

        Returns ``None`` when the kernel cannot take the call, leaving the
        caller on the numpy path. Each of ``outs`` is written into if it was
        given and allocated if it was not, so a caller wanting one property
        supplies its buffer and lets the other two be found -- the answer lands
        in the caller's own array either way, with nothing copied afterwards.

        Shared by :meth:`get_P_h_T` and by :meth:`get_P` and :meth:`get_T`,
        which ask for one property and are handed three. That is not the waste
        it looks: all three come off the same two polynomial surfaces, walking
        those is the entire cost, and forming the other two afterwards is a
        division apiece. Three from the kernel is still two orders cheaper than
        one from numpy.
        """
        arrs = (rho, u) + tuple(o for o in outs if o is not None)
        usable = (
            self._kernel_fits()
            and all(isinstance(a, np.ndarray) for a in arrs)
            and all(a.dtype == np.float32 for a in arrs)
            and all(a.shape == np.shape(rho) for a in arrs)
            and all(a.flags["F_CONTIGUOUS"] or a.flags["C_CONTIGUOUS"] for a in arrs)
            # Every array must flatten in the same order, not merely flatten
            # without copying. The kernel pairs them up element by element, so
            # a C-ordered output against an F-ordered input would take each
            # answer to the wrong node -- quietly, and only in two dimensions
            # or more, where the two traversals differ.
            and all(np.isfortran(a) == np.isfortran(rho) for a in arrs)
        )
        if not usable:
            return None
        # empty_like, so anything allocated here inherits the input's dtype and
        # memory order and satisfies the check above by construction.
        outs = [np.empty_like(rho) if o is None else o for o in outs]
        # order="A" ravels without copying for either contiguity, so these stay
        # views and the kernel's writes land in the caller's arrays.
        ember.fortran.set_p_h_t_real(
            rho=np.ravel(rho, order="A"),
            u=np.ravel(u, order="A"),
            scx=self._Sc_x,
            nzx=self._nzx,
            scy=self._Sc_y,
            nzy=self._nzy,
            sl=self._Sl,
            sly=self._Sl_y,
            xa=self._xa,
            xb=self._xb,
            ya=self._ya,
            yb=self._yb,
            p=np.ravel(outs[0], order="A"),
            h=np.ravel(outs[1], order="A"),
            t=np.ravel(outs[2], order="A"),
        )
        return tuple(outs)

    def _kernel_fits(self):
        """Whether this surface's order fits the Fortran kernel's basis buffers.

        The kernel sizes them at compile time, so this is a bound it cannot
        check for itself and cannot be allowed to exceed: overrunning them
        would corrupt the stack rather than raise. Practical fits sit an order
        of magnitude below the limit -- a least-squares Legendre fit loses
        conditioning long before it -- so this is a guard, not a real
        restriction, and a surface past it simply takes the numpy path.
        """
        na = max(self._Sc_x.shape[0], self._Sc_y.shape[0])
        nb = max(
            self._Sc_x.shape[1],
            self._Sc_y.shape[1],
            self._Sl.size,
            self._Sl_y.size,
        )
        return max(na, nb) <= _REAL_KERNEL_MAXORD + 1

    @staticmethod
    def _isentropic_exponent(st, rho):
        """Isentropic exponent from a state dict, k = (dP/drho)_s * rho/P."""
        dPdrho_s = st["P_r"] - st["P_u"] * st["s_r"] / st["s_u"]
        return dPdrho_s * rho / st["P"]

    def _hats(self, rho, u):
        """Normalised coordinates from a non-dimensional state.

        The two are broadcast against each other before returning:
        ``legval2d`` requires its coordinate arrays to have identical shapes and
        will not broadcast them itself, whereas this interface promises that any
        broadcastable pair of inputs works.
        """
        x = rho * self._xa + self._xb
        y = u * self._ya + self._yb
        return np.broadcast_arrays(x, y)

    def _partials1(self, rho, u):
        """First partials of entropy with respect to (rho, u).

        Split from :meth:`_partials2` because pressure, temperature and enthalpy
        need only these, and the solver asks for them once per Runge-Kutta
        stage. Second derivatives cost three more surface evaluations and are
        wanted only by the specific heats and the Newton solves.

        Entropy itself is not among them. Every caller here wants a partial and
        nothing else, and the level costs a whole surface evaluation over arrays
        the size of a block -- a third of the work of this method, once per
        stage, for a number that was thrown away. :meth:`_entropy` is the one
        that returns it, and :meth:`_partials2` for the solves that need both.
        """
        x, y = self._hats(rho, u)
        lnr = np.log(rho)
        M = _leg.legval(y, self._Sl)
        My = _leg.legval(y, self._Sl_y)

        s_r = _leg.legval2d(x, y, self._Sc_x) * self._xa + M / rho
        s_u = (_leg.legval2d(x, y, self._Sc_y) + My * lnr) * self._ya
        return s_r, s_u

    def _partials2(self, rho, u):
        """Entropy and its first and second partials with respect to (rho, u)."""
        rho_b, u_b = np.broadcast_arrays(rho, u)
        if (
            self._kernel_fits()
            and rho_b.dtype == np.float32
            and u_b.dtype == np.float32
            # All eight arrays have to flatten in the same order. The kernel
            # pairs rho against u element by element and the six results are
            # laid back out the same way, so a pair walked in Fortran order
            # against results written in C order takes every answer to the
            # wrong node -- quietly, and only in two dimensions or more, where
            # the two traversals differ. A block's fields are Fortran
            # contiguous and hit exactly that. :meth:`_kernel_P_h_T` makes the
            # same check for the same reason; the results are allocated here
            # rather than passed in, so matching them to the input settles it.
            and np.isfortran(rho_b) == np.isfortran(u_b)
            and all(
                a.flags["C_CONTIGUOUS"] or a.flags["F_CONTIGUOUS"] for a in (rho_b, u_b)
            )
        ):
            # empty_like, so each result inherits the input's memory order and
            # ravels below to a view the kernel can write straight through.
            outs = [np.empty_like(rho_b) for _ in range(6)]
            ember.fortran.set_partials2_real(
                rho=np.ravel(rho_b, order="A"),
                u=np.ravel(u_b, order="A"),
                sc2=self._Sc_stack,
                nz2=self._Sc_stack_nz,
                sc1=self._Sl_stack,
                xa=self._xa,
                xb=self._xb,
                ya=self._ya,
                yb=self._yb,
                s=np.ravel(outs[0], order="A"),
                s_r=np.ravel(outs[1], order="A"),
                s_u=np.ravel(outs[2], order="A"),
                s_rr=np.ravel(outs[3], order="A"),
                s_ru=np.ravel(outs[4], order="A"),
                s_uu=np.ravel(outs[5], order="A"),
            )
            return tuple(outs)

        x, y = self._hats(rho, u)
        lnr = np.log(rho)
        M = _leg.legval(y, self._Sl)
        My = _leg.legval(y, self._Sl_y)
        Myy = _leg.legval(y, self._Sl_yy)

        s = _leg.legval2d(x, y, self._Sc) + M * lnr
        s_r = _leg.legval2d(x, y, self._Sc_x) * self._xa + M / rho
        s_u = (_leg.legval2d(x, y, self._Sc_y) + My * lnr) * self._ya
        s_rr = _leg.legval2d(x, y, self._Sc_xx) * self._xa**2 - M / rho**2
        s_ru = (_leg.legval2d(x, y, self._Sc_xy) * self._xa + My / rho) * self._ya
        s_uu = (_leg.legval2d(x, y, self._Sc_yy) + Myy * lnr) * self._ya**2
        return s, s_r, s_u, s_rr, s_ru, s_uu

    @staticmethod
    def _pick(prop, st, rho, u):
        """Value and (rho, u) partials of one property, from a state dict.

        ``prop`` is one of ``"P"``, ``"T"``, ``"s"`` or ``"h"``: which property
        a solve is being asked to match. Not ``kind``, which is what half this
        codebase is written in and where the word already means precision.
        """
        if prop == "h":
            P, P_r, P_u = st["P"], st["P_r"], st["P_u"]
            return u + P / rho, P_r / rho - P / rho**2, 1.0 + P_u / rho
        return st[prop], st[f"{prop}_r"], st[f"{prop}_u"]

    def _state(self, rho, u):
        """Pressure, temperature, entropy and their first partials, in one pass.

        Temperature and pressure follow from the entropy partials by the
        combined first and second law; their own partials then follow by
        differentiating those expressions, which is where the second
        derivatives of the surface are needed.
        """
        s, s_r, s_u, s_rr, s_ru, s_uu = self._partials2(rho, u)
        T = 1.0 / s_u
        Tsq = T * T
        T_r = -Tsq * s_ru
        T_u = -Tsq * s_uu
        P = -(rho**2) * T * s_r
        P_r = -2.0 * rho * T * s_r - rho**2 * (T_r * s_r + T * s_rr)
        P_u = -(rho**2) * (T_u * s_r + T * s_ru)
        return {
            "s": s,
            "s_r": s_r,
            "s_u": s_u,
            "T": T,
            "T_r": T_r,
            "T_u": T_u,
            "P": P,
            "P_r": P_r,
            "P_u": P_u,
        }

    def _kwargs(self):
        """Constructor arguments reproducing this fluid; see :meth:`_Fluid._kwargs`."""
        return {
            "alpha": self._alpha,
            "beta": self._beta,
            "rho_lim": self._rho_lim,
            "u_lim": self._u_lim,
            "rho_isochor": self._rho_isochor,
            "Rgas": float(self._Rgas),
            "mu": float(self._mu),
            "Pr": float(self._Pr),
            "P_dtm": float(self._P_dtm),
            "T_dtm": float(self._T_dtm),
            "rho_ref": float(self.rho_ref),
            "V_ref": float(self.V_ref),
            "Rgas_ref": float(self.Rgas_ref),
        }

    def _rebuild(self, **over):
        """New instance with the same fitted surface and selected overrides."""
        return self.__class__(**{**self._kwargs(), **over})

    @staticmethod
    def _write(val, out):
        """Return a value, filling a pre-allocated array if one was given."""
        if out is None:
            return val
        out[...] = val
        return out

    def _solve_2d(self, prop1, val1, prop2, val2, method):
        """Newton solve for (rho, u) matching two properties.

        The 2x2 Jacobian is inverted in closed form. The alternative -- fixing
        one variable and calling a scalar solve inside an outer iteration --
        costs the product of the two iteration counts and needs the inner solve
        converged tightly for the outer derivative to mean anything, while
        arriving at the same Jacobian algebra regardless.

        The state handed back is the one the acceptance test looked at. The
        loop breaks before applying its last correction rather than after, so
        no evaluation is spent re-measuring a state that has just moved: a step
        small enough to end the loop is a step at the arithmetic's floor, and
        taking it would change the answer by an ulp while costing a full walk
        of every surface -- one iteration in five, on top of the four that did
        the work.
        """
        val1 = np.asarray(val1)
        val2 = np.asarray(val2)
        rho0, u0 = self._guess_2d(method, val1, val2)
        shape = np.broadcast(val1, val2, rho0, u0).shape
        # Iterate in the callers' own precision, so single-precision inputs give
        # single-precision results as the rest of this interface promises. The
        # float32 floor is what _NEWTON_RTOL is sized against.
        dtype = np.result_type(np.float32, val1, val2)
        rho = np.array(np.broadcast_to(rho0, shape), dtype=dtype, copy=True)
        u = np.array(np.broadcast_to(u0, shape), dtype=dtype, copy=True)

        step_prev = np.inf
        for _ in range(self._NEWTON_ITER):
            st = self._state(rho, u)
            f1, f1r, f1u = self._pick(prop1, st, rho, u)
            f2, f2r, f2u = self._pick(prop2, st, rho, u)
            r1 = f1 - val1
            r2 = f2 - val2
            det = f1r * f2u - f1u * f2r
            drho = (r1 * f2u - r2 * f1u) / det
            du = (r2 * f1r - r1 * f2r) / det
            rho_new = np.clip(rho - drho, *self._rho_box_nd)
            u_new = np.clip(u - du, *self._u_box_nd)
            step = np.max(
                np.abs(rho_new - rho) / np.abs(rho) + np.abs(u_new - u) / self._u_scale
            )
            if step < self._NEWTON_RTOL or (
                step < self._NEWTON_SETTLED and step > self._NEWTON_STALL * step_prev
            ):
                break
            rho, u = rho_new, u_new
            step_prev = step

        self._verify(prop1, f1, val1, method)
        self._verify(prop2, f2, val2, method)
        return rho, u

    def _solve_u(self, rho, val, prop, method, *guess_args):
        """Newton solve for u at fixed density, matching one property.

        Monotone in ``u`` for every property this is used with -- entropy
        because ``(ds/du)_rho = 1/T > 0``, temperature because ``cv > 0``, and
        pressure because heating at constant volume raises it -- so the
        iteration cannot walk off in the wrong direction.

        The state handed back is the one the acceptance test looked at. The
        loop breaks before applying its last correction rather than after, so
        no evaluation is spent re-measuring a state that has just moved: a step
        small enough to end the loop is a step at the arithmetic's floor, and
        taking it would change the answer by an ulp while costing a full walk
        of every surface -- one iteration in five, on top of the four that did
        the work.
        """
        # Before the guess, which would otherwise take the log of a density
        # that should never have got this far.
        self._check_rho_box(rho, method)
        val = np.asarray(val)
        u0 = self._guess_u(method, *guess_args)
        shape = np.broadcast(np.asarray(rho), val, u0).shape
        dtype = np.result_type(np.float32, np.asarray(rho), val)
        rho_b = np.broadcast_to(np.asarray(rho, dtype=dtype), shape)
        u = np.array(np.broadcast_to(u0, shape), dtype=dtype, copy=True)

        # Decided once: nothing it depends on changes as the loop runs.
        plan = self._solve_plan.get(prop)
        rho_c = f_buf = fu_buf = None
        if plan is not None and self._kernel_fits() and dtype == np.float32:
            rho_c = np.ascontiguousarray(rho_b, dtype=np.float32)
            u = np.ascontiguousarray(u)
            f_buf, fu_buf = np.empty_like(u), np.empty_like(u)

        step_prev = np.inf
        for _ in range(self._NEWTON_ITER):
            if f_buf is None:
                st = self._state(rho_b, u)
                f, _, f_u = self._pick(prop, st, rho_b, u)
            else:
                ember.fortran.set_f_fu_real(
                    rho=np.ravel(rho_c, order="A"),
                    u=np.ravel(u, order="A"),
                    sc2=self._Sc_stack,
                    nz2=self._Sc_stack_nz,
                    sel=plan[0],
                    sc1=self._Sl_stack,
                    xa=self._xa,
                    xb=self._xb,
                    ya=self._ya,
                    yb=self._yb,
                    which=plan[1],
                    f=np.ravel(f_buf, order="A"),
                    f_u=np.ravel(fu_buf, order="A"),
                )
                f, f_u = f_buf, fu_buf
            u_new = np.clip(u - (f - val) / f_u, *self._u_box_nd)
            step = np.max(np.abs(u_new - u)) / self._u_scale
            if step < self._NEWTON_RTOL or (
                step < self._NEWTON_SETTLED and step > self._NEWTON_STALL * step_prev
            ):
                break
            u = u_new
            step_prev = step

        self._verify(prop, f, val, method)
        return rho, u

    def _verify(self, prop, got, want, method):
        """Raise unless a Newton solve met its tolerance everywhere.

        A state the fit cannot reach is a physical error -- most often a
        requested condition outside the box the coefficients were fitted over --
        so it is reported rather than returned as a silently wrong state.
        """
        scale = np.abs(want) + self._floor(prop)
        resid = np.asarray(got) - np.asarray(want)
        # A diverged solve returns nan, and every comparison against nan is
        # false -- including the tolerance test below. Whether the answer is a
        # number at all has to be asked separately, or the worst failure there
        # is becomes the one failure that reports success.
        adrift = ~np.isfinite(resid)
        rel = np.abs(resid) / np.where(scale > 0, scale, 1.0)
        bad = adrift | (rel > self._VERIFY_RTOL)
        if not np.any(bad):
            return
        nbad = int(np.count_nonzero(bad))
        n_adrift = int(np.count_nonzero(adrift))
        if n_adrift:
            detail = f"{n_adrift} did not return a finite value"
        else:
            detail = f"worst relative residual {float(np.max(rel)):.3e}"
        raise RuntimeError(
            f"{type(self).__name__}.{method} failed to converge at {nbad} of "
            f"{np.size(bad)} states matching {prop}: {detail}. The requested "
            f"state is most likely outside the fit box "
            f"rho_lim={self._rho_lim}, u_lim={self._u_lim}."
        )

    def _check_rho_box(self, rho, method):
        """Raise unless every density lies inside the fitted box.

        Where density is given rather than solved for, nothing downstream can
        catch it being wrong. The fitted surface extrapolates smoothly, so the
        solve converges and :meth:`_verify` passes on a state that is perfectly
        self-consistent with a polynomial nobody fitted out there.
        """
        rho = np.asarray(rho)
        lo, hi = self._rho_box_nd
        bad = ~np.isfinite(rho) | (rho < lo) | (rho > hi)
        if not np.any(bad):
            return
        nbad = int(np.count_nonzero(bad))
        raise RuntimeError(
            f"{type(self).__name__}.{method} was given {nbad} of "
            f"{np.size(bad)} densities outside the fit box "
            f"rho_lim={self._rho_lim}, which is "
            f"[{float(lo):.6g}, {float(hi):.6g}] non-dimensional."
        )

    def set_h_s(self, h, s):
        """Density and internal energy from specific enthalpy and entropy.

        Solved by two-dimensional Newton iteration; see :meth:`set_P_T`.

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
        return self._solve_2d("h", h, "s", s, "set_h_s")

    def set_P_h(self, P, h):
        """Density and internal energy from pressure and specific enthalpy.

        Solved by two-dimensional Newton iteration; see :meth:`set_P_T`.

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
        return self._solve_2d("P", P, "h", h, "set_P_h")

    def set_P_rho(self, P, rho):
        r"""Density and internal energy from pressure and density.

        Density is already known, so this is a scalar Newton solve for the
        internal energy satisfying :math:`p(\rho, u) = p`, seeded from the
        companion perfect gas. The residual is monotone in :math:`u`, since
        heating at constant volume raises pressure.

        This is on the per-step nonreflecting boundary path, so it is solved for
        the whole array at once rather than node by node.

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
        return self._solve_u(rho, P, "P", "set_P_rho", P, rho)

    def set_P_s(self, P, s):
        """Density and internal energy from pressure and specific entropy.

        Solved by two-dimensional Newton iteration; see :meth:`set_P_T`.

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
        return self._solve_2d("P", P, "s", s, "set_P_s")

    def set_P_T(self, P, T):
        r"""Density and internal energy from pressure and temperature.

        Neither variable is known directly, so both are found together by
        Newton iteration on the 2x2 system :math:`p(\rho, u) = p`,
        :math:`T(\rho, u) = T`, with the Jacobian assembled analytically from
        the entropy surface and inverted in closed form.

        Conditioning degrades near the critical point, where
        :math:`(\partial p/\partial\rho)_T \rightarrow 0` and the Jacobian
        determinant vanishes. That is physical rather than numerical -- pressure
        and temperature stop being independent coordinates on the saturation
        line -- and lies outside any sensible fit box, but it will surface as a
        convergence failure rather than a wrong answer.

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
        return self._solve_2d("P", P, "T", T, "set_P_T")

    def set_rho_s(self, rho, s):
        r"""Density and internal energy from density and specific entropy.

        A scalar Newton solve for the internal energy satisfying
        :math:`s(\rho, u) = s`. The residual is monotone in :math:`u` because
        :math:`(\partial s/\partial u)_\rho = 1/T > 0`, so convergence is
        assured from any starting point in the box.

        This is on the per-step nonreflecting boundary path, so it is solved for
        the whole array at once rather than node by node.

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
        return self._solve_u(rho, s, "s", "set_rho_s", rho, s)

    def set_T_rho(self, T, rho):
        r"""Density and internal energy from temperature and density.

        A scalar Newton solve for the internal energy satisfying
        :math:`T(\rho, u) = T`. The residual is monotone in :math:`u` because
        the specific heat at constant volume is positive.

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
        return self._solve_u(rho, T, "T", "set_T_rho", T, rho)

    def set_T_s(self, T, s):
        """Density and internal energy from temperature and specific entropy.

        Solved by two-dimensional Newton iteration; see :meth:`set_P_T`.

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
        return self._solve_2d("T", T, "s", s, "set_T_s")

    def get_a(self, rho, u, out=None):
        r"""Speed of sound from density and internal energy.

        .. math:: a = \sqrt{k\, p / \rho}

        where :math:`k` is the isentropic exponent of :meth:`get_gamma`. Note
        this is the isentropic exponent and not the ratio of specific heats,
        which for a real gas is a different number.

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
        st = self._state(rho, u)
        gamma = self._isentropic_exponent(st, rho)
        return self._write(np.sqrt(gamma * st["P"] / rho), out)

    def get_cp(self, rho, u, out=None):
        r"""Specific heat at constant pressure from density and internal energy.

        Obtained from the specific heat at constant volume and the ratio of the
        isentropic and isothermal density derivatives of pressure,

        .. math::

            \frac{c_p}{c_v}
                = \frac{(\partial p/\partial\rho)_s}{(\partial p/\partial\rho)_T}

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
        st = self._state(rho, u)
        cv = 1.0 / st["T_u"]
        dPdrho_s = st["P_r"] - st["P_u"] * st["s_r"] / st["s_u"]
        dPdrho_T = st["P_r"] - st["P_u"] * st["T_r"] / st["T_u"]
        return self._write(cv * dPdrho_s / dPdrho_T, out)

    def get_cv(self, rho, u, out=None):
        r"""Specific heat at constant volume from density and internal energy.

        .. math:: c_v = \left(\frac{\partial u}{\partial T}\right)_\rho

        evaluated by inverting the temperature derivative, which comes from the
        second derivative of the entropy surface.

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
        st = self._state(rho, u)
        return self._write(1.0 / st["T_u"], out)

    def get_dhdP_rho(self, rho, u, out=None):
        r"""Derivative of specific enthalpy with respect to pressure at constant density.

        .. math::

            \left.\frac{\partial h}{\partial p}\right|_\rho
                = \left.\frac{\partial u}{\partial p}\right|_\rho + \frac{1}{\rho}

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
        st = self._state(rho, u)
        return self._write(1.0 / st["P_u"] + 1.0 / rho, out)

    def get_dhdrho_P(self, rho, u, out=None):
        r"""Derivative of specific enthalpy with respect to density at constant pressure.

        .. math::

            \left.\frac{\partial h}{\partial \rho}\right|_p
                = \left.\frac{\partial u}{\partial \rho}\right|_p
                - \frac{p}{\rho^2}

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
        st = self._state(rho, u)
        dudrho_P = -st["P_r"] / st["P_u"]
        return self._write(dudrho_P - st["P"] / rho**2, out)

    def get_dsdP_rho(self, rho, u, out=None):
        r"""Derivative of specific entropy with respect to pressure at constant density.

        Obtained by chain rule through internal energy at fixed density,

        .. math::

            \left.\frac{\partial s}{\partial p}\right|_\rho
                = \frac{(\partial s/\partial u)_\rho}{(\partial p/\partial u)_\rho}

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
        st = self._state(rho, u)
        return self._write(st["s_u"] / st["P_u"], out)

    def get_dsdrho_P(self, rho, u, out=None):
        r"""Derivative of specific entropy with respect to density at constant pressure.

        .. math::

            \left.\frac{\partial s}{\partial \rho}\right|_p
                = \left.\frac{\partial s}{\partial \rho}\right|_u
                + \left.\frac{\partial s}{\partial u}\right|_\rho
                  \left.\frac{\partial u}{\partial \rho}\right|_p

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
        st = self._state(rho, u)
        dudrho_P = -st["P_r"] / st["P_u"]
        return self._write(st["s_r"] + st["s_u"] * dudrho_P, out)

    def get_dudP_rho(self, rho, u, out=None):
        r"""Derivative of specific internal energy with respect to pressure at constant density.

        The reciprocal of the pressure derivative taken directly from the
        surface,

        .. math::

            \left.\frac{\partial u}{\partial p}\right|_\rho
                = \left[\left.\frac{\partial p}{\partial u}\right|_\rho\right]^{-1}

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
        st = self._state(rho, u)
        return self._write(1.0 / st["P_u"], out)

    def get_dudrho_P(self, rho, u, out=None):
        r"""Derivative of specific internal energy with respect to density at constant pressure.

        .. math::

            \left.\frac{\partial u}{\partial \rho}\right|_p
                = -\frac{(\partial p/\partial \rho)_u}{(\partial p/\partial u)_\rho}

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
        st = self._state(rho, u)
        return self._write(-st["P_r"] / st["P_u"], out)

    def get_gamma(self, rho, u, out=None):
        r"""Isentropic exponent from density and internal energy.

        .. math::

            k = \frac{\rho}{p}\left.\frac{\partial p}{\partial \rho}\right|_s

        This is the exponent that governs acoustics and the characteristic
        decomposition, and it is what :meth:`get_a` uses. For a real gas it is
        **not** the ratio of specific heats :math:`c_p/c_v`, which is available
        separately from :meth:`get_cp` and :meth:`get_cv`; the two coincide only
        when the specific heats are constant.

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
            Isentropic exponent [--].
        """
        st = self._state(rho, u)
        return self._write(self._isentropic_exponent(st, rho), out)

    def get_h(self, rho, u, out=None):
        r"""Specific enthalpy from density and internal energy.

        .. math:: h = u + p/\rho

        Enthalpy carries an offset dependent on the arbitrary datum state where
        :math:`u = s = 0`; only changes in :math:`h` are physically meaningful.
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
        return self._write(u + self.get_P(rho, u) / rho, out)

    def get_mu(self, rho, u, out=None):
        r"""Dynamic viscosity (constant; the transport surfaces are not fitted).

        If reference scales are set, then this method returns a
        quasi-dimensional viscosity in units of [m] --- see
        :ref:`reference-scales` for details.

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

        From the combined first and second law, at constant internal energy,

        .. math:: p = -\rho^2 T \left(\frac{\partial s}{\partial \rho}\right)_u

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
        got = self._kernel_P_h_T(rho, u, (out, None, None))
        if got is not None:
            return got[0]

        s_r, s_u = self._partials1(rho, u)
        return self._write(-(rho**2) * s_r / s_u, out)

    def get_P_h_T(self, rho, u, out_P=None, out_h=None, out_T=None):
        """Batched evaluation of pressure, enthalpy and temperature.

        All three come from the same entropy partials, so evaluating them
        together costs barely more than any one of them, where the base class
        would walk the polynomial surface three times. The solver calls this
        once per Runge-Kutta stage.

        A float32 call with all three outputs supplied -- which is what the
        solver makes -- goes to a Fortran kernel instead. Anything else falls
        back to the numpy body below rather than to the base class: that body
        is already fused, and three separate calls would be three times the
        work.

        Parameters
        ----------
        rho : array_like
            Density [kg/m³].
        u : array_like
            Specific internal energy [J/kg].
        out_P, out_h, out_T : ndarray, optional
            Pre-allocated output arrays.

        Returns
        -------
        tuple of ndarray
            ``(P, h, T)``.
        """
        got = self._kernel_P_h_T(rho, u, (out_P, out_h, out_T))
        if got is not None:
            return got

        s_r, s_u = self._partials1(rho, u)
        T = 1.0 / s_u
        P = -(rho**2) * T * s_r
        return (
            self._write(P, out_P),
            self._write(u + P / rho, out_h),
            self._write(T, out_T),
        )

    def get_Pr(self, rho, u, out=None):
        r"""Prandtl number (constant; the transport surfaces are not fitted).

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
        r"""Specific gas constant (a constant property of the species).

        Note that for a real gas this is *not* :math:`p/(\rho T)`, which is
        larger by the compressibility factor.

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

        Evaluated directly from the fitted surface: an isochor polynomial in
        internal energy, less the analytic density integral of the
        compressibility factor.

        Entropy is defined relative to the arbitrary datum state where
        :math:`u = s = 0`; only changes in :math:`s` are physically meaningful.
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
        s : ndarray
            Specific entropy [J/kg/K].
        """
        return self._write(self._entropy(rho, u), out)

    def get_T(self, rho, u, out=None):
        r"""Temperature from density and internal energy.

        From the combined first and second law, at constant density,

        .. math::

            T = \left[\left(\frac{\partial s}{\partial u}\right)_\rho\right]^{-1}

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
        got = self._kernel_P_h_T(rho, u, (None, None, out))
        if got is not None:
            return got[2]

        _, s_u = self._partials1(rho, u)
        return self._write(1.0 / s_u, out)

    def change_datum(self, P_dtm, T_dtm):
        """Get a new :class:`RealFluid` with shifted datum.

        The coefficient arrays are untouched: they describe a dimensionless
        surface in normalised coordinates, and the datum only enters through the
        affine constants composed at construction. See :ref:`datum-state`.

        Parameters
        ----------
        P_dtm : float
            New datum pressure [Pa].
        T_dtm : float
            New datum temperature [K].

        Returns
        -------
        fluid_new : RealFluid
            New fluid instance with shifted energy and entropy datum.

        """
        return self._rebuild(P_dtm=P_dtm, T_dtm=T_dtm)

    def change_ref(self, rho_ref=None, V_ref=None, Rgas_ref=None):
        """Make a new :class:`RealFluid` with different reference scales.

        Omitted reference scales default to the current instance's, so only the
        scales that need to be changed must be specified. As with
        :meth:`change_datum`, no refit is involved.

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
        fluid_new : RealFluid
            New fluid instance with the same properties but different reference
            scales.

        """
        return self._rebuild(
            rho_ref=rho_ref if rho_ref is not None else self.rho_ref,
            V_ref=V_ref if V_ref is not None else self.V_ref,
            Rgas_ref=Rgas_ref if Rgas_ref is not None else self.Rgas_ref,
        )

    @property
    def rho_lim_nd(self):
        r"""Density bounds of the fit box :math:`\rho/\rho_\mathrm{ref}` [--].

        The constructor takes the box in SI, but ``get_*`` and ``set_*`` work in
        non-dimensional units, so this is the form a caller needs to check that
        a state is in range. Outside these bounds the fitted polynomials are
        meaningless and the ``set_*`` solves will refuse to converge.
        """
        return (float(self._rho_box_nd[0]), float(self._rho_box_nd[1]))

    @property
    def u_lim_nd(self):
        r"""Internal energy bounds of the fit box :math:`u/u_\mathrm{ref}` [--].

        As :attr:`rho_lim_nd`, and additionally shifted onto this fluid's datum:
        the constructor's ``u_lim`` is on whatever datum the coefficients were
        fitted against, whereas ``get_*`` and ``set_*`` take internal energy
        measured from :attr:`P_dtm`, :attr:`T_dtm`. Changing the datum moves
        these bounds even though the underlying box is unchanged.
        """
        return (float(self._u_box_nd[0]), float(self._u_box_nd[1]))


_FLUID_TYPES = {cls.__name__: cls for cls in (PerfectFluid, RealFluid)}
"""Equations of state :meth:`_Fluid.from_dict` will build, by class name.

An explicit table rather than a lookup on the module, so that a ``type`` out of
a file can only ever name one of these two classes.
"""
