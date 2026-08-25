"""Mixing plane boundary condition communication.

:class:`MixingCommunicator` pairs the two sides of the mixing plane of
:class:`~ember.mixing.MixingPatch`, holds the relaxation factor and the per-pair diagnostics,
and carries out the exchange itself: it takes the cross-plane *flux* mismatch,
splits it by direction of propagation after :cite:t:`Saxer1993`, and writes the
result in the mix variables :math:`[h_0, s, V_r, V_\\theta, p]` its patches
take their pitchwise-mean residuals against.

A plane whose two sides are
:attr:`~ember.mixing.MixingPatch.reflective` bypasses all of that for the
average of the two circumferential means; see
:meth:`MixingCommunicator._mix_uniform`.
"""

from ember import perturbation, util
from ember.util import profile
import numpy as np


class MixingCommunicator:
    r"""Cross-plane exchange for the mixing plane.

    Follows :cite:t:`Saxer1993` (his Section 5.5): it flux-averages each side,
    takes the jump in that state across the plane, converts the jump to
    characteristic variables and splits it by direction of propagation -- the
    side the mean flow reaches first at each span station owning the
    upstream-running pressure characteristic and the other the remaining four
    (:cite:t:`Holmes2008` Eq. 10-11, per station rather than fixed by
    geometry) -- and integrates the result onto the target itself,

    .. math::

        \mathrm{target}_n = \mathrm{target}_{n-1}
            + \mathrm{rf\_exchange}\,\varepsilon_n,

    following Holmes' Eq. 15 rather than re-anchoring to the live interface
    state every step: the fixed point of the integrating form is exact flux
    balance, where the proportional form leaves a standing offset the size of
    the residual mismatch. :attr:`leak` and a non-physical-state guard are the
    anti-windup this needs in return.

    The relaxation factor is read from the patches at every exchange, so it is
    per plane rather than per grid, and a solver run can retune it on a
    communicator that already exists. Both sides of a plane must agree on it.
    It is the same on every multigrid level.

    The target is written in the mix variables :math:`[h_0, s, V_r, V_\theta,
    p]`, which are exactly the quantities the two patches take their
    pitchwise-mean residuals against; they drive only the mean mode of each
    side's boundary condition and leave the harmonics to the non-reflecting
    relations of the patches themselves.

    On top of that sits :cite:t:`Holmes2008`'s **mass flow control**, his
    Eq. 20-22, which holds the interface mass flux near the flow the machine is
    actually passing while the two are still far apart. It is not a second
    controller: it is an offset added to the flux difference the exchange
    already drives to zero, so it passes through the same transformation chain,
    the same characteristic selection and the same ``rf_exchange`` as everything
    else. It is also ramped to zero as the mass flow error closes, so the
    converged fixed point is exact flux balance whether it is switched on or
    not. See :meth:`_apply_mdot_offset`, and :meth:`exchange` for how the
    target reaches it.

    See Also
    --------
    ember.mixing : The patch class this pairs
    """

    # Jacobian mapping characteristic variables to the space the exchanged
    # target is written in. Its last row must be the static pressure and its
    # first four the quantities an inflow prescribes, because _write_targets
    # expresses Saxer's split of the interface jump by direction of propagation
    # as a pair of row masks on the target vector.
    _chic_to_target = staticmethod(perturbation.chic_to_mix)

    leak = 0.0
    r"""Anti-windup leak on the integrating relaxation, as a fraction of
    :attr:`rf_exchange` per exchange.

    The exchange accumulates its correction onto the previous target
    rather than re-anchoring to the live interface baseline every step, so
    that the fixed point is exact flux balance rather than the proportional
    form's standing offset (:cite:t:`Holmes2008` Eq. 15, applied to the
    auxiliary cells rather than re-derived each step). A pure integrator can
    wind up while the mismatch has not yet resolved -- most a reversed station
    whose own boundary condition has not caught up with a target that has
    already moved past what the flow was ever in. A positive leak bleeds the
    target back toward the live baseline each step, trading a
    ``leak/rf_exchange``-scaled residual flux mismatch for a bound on how far
    the target can wander. Zero is exact Holmes; engage only if a station is
    seen to wind up.
    """

    Ma_clip = 0.05
    r"""Floor on :math:`\lvert \mathit{Ma}_x \rvert` of the symmetrised mean
    state the Jacobians are evaluated on.

    :cite:t:`Holmes2008` Eq. 16: as the mean normal velocity tends to zero the
    eigenvalues of the transformation matrices grow, so the interface
    over-controls to make up for the slow advection the small velocity implies.
    Bounding the magnitude -- not the direction, which a reversed station keeps
    -- is his remedy and this is ember's form of it. It also keeps
    :func:`~ember.perturbation.flux_to_primitive`, which divides by the axial
    velocity, away from zero.

    See :attr:`Ma_clip_max` for the other end of the same band, and
    :attr:`Ma_mdot_max` for the separate gate on the *absolute* Mach number that
    the mass flow forcing needs.
    """

    Ma_clip_max = 0.9
    r"""Ceiling on :math:`\lvert \mathit{Ma}_x \rvert` of that same state, the
    other end of :attr:`Ma_clip`'s band.

    Holmes bounds the normal velocity only from below, because that is where his
    eigenvalues blow up. The upper end matters here for a different reason: the
    plane's whole characteristic treatment is derived for a mean state subsonic
    normal to it (see :class:`~ember.mixing.MixingPatch`), and a station that
    briefly runs past sonic during a transient would otherwise be linearised
    with the wrong number of incoming characteristics rather than merely
    inaccurately. Clipping keeps such a station inside the theory until the flow
    brings it back.
    """

    Ma_mdot_max = 0.9
    r"""Absolute Mach number above which a span station takes no mass flow
    forcing.

    Not the same quantity as :attr:`Ma_clip` and :attr:`Ma_clip_max`, which
    bound the Mach number *normal* to the plane. The mass flux sensitivity
    :func:`~ember.perturbation.mdot_to_flux` divides by
    :math:`\rho - V^2/\Theta`, which for a perfect gas is
    :math:`\rho\,(1 - \mathit{Ma}^2)` in the **absolute** Mach number -- so a
    strongly swirling station can approach the singularity with a perfectly
    modest normal component, and clipping :math:`\mathit{Ma}_x` would not catch
    it. Above this the forcing is simply dropped at that station, which costs
    nothing: it is a convergence accelerator, not part of the condition.
    """

    eps_deadband = 1e-2
    r"""Fractional mass flow error below which the forcing is off entirely.

    The forcing exists to remove a transient, not to change an answer, so it is
    ramped out before the answer is reached. Below this the correction is
    identically zero, which makes the converged fixed point of the exchange
    exactly what it is without the forcing -- exact flux balance -- rather than
    something that has to be argued to be close to it.

    **This has to sit above the mass flow error the exchange settles at on its
    own, and that is what sets it.** The correction feeds an integrator, so a
    steady target needs the flux mismatch and the correction to cancel rather
    than each to vanish; a forcing still acting at the fixed point therefore
    moves that fixed point instead of merely arriving at it sooner. The
    deadband is what prevents this, and it can only do so if the forcing has
    switched off by the time the march gets there. One percent is chosen
    against a two-row duct that settles at about 0.85%, which is not a wide
    margin: a case whose planes settle at a larger standing imbalance wants
    this raised, and a run whose two ends still disagree by more than this at
    convergence is being forced the whole way, which the mass flow trace will
    show.

    It also disposes of a measurement question. The target is measured at the
    domain inlet and outlet by :meth:`~ember.grid.Grid._calc_mdot_target`, using
    :func:`~ember.average.flow_mass` over the whole face, while a plane's own
    total is the pitch mean of its flux average against its span areas. Those
    two quadratures agree closely on a surface of revolution but not exactly,
    and the difference does not shrink as the solution converges. The deadband
    is far above that bias, so it can never be what is driving the forcing.
    """

    eps_full = 5e-2
    r"""Fractional mass flow error at and above which the forcing acts in full.

    Between here and :attr:`eps_deadband` the gain ramps linearly. A hard switch
    at the deadband would do, except that the correction feeds an integrator
    (see the class docstring), and a step change in what is being integrated
    shows up as a kink in the target that the patches' own relaxation then has
    to chase. A station hovering at the threshold would chatter across it.

    Five percent puts the whole of a realistic starting transient at full gain
    -- two rows started 25% apart in mass flow open at an error five times this
    -- and confines the ramp to the final approach, which is the only place the
    smoothing is needed.
    """

    def __init__(
        self,
        grid,
        mixing_pairs,
    ):
        """Initialize with grid and mixing patch pairs.

        Parameters
        ----------
        grid : Grid
            The grid instance.
        mixing_pairs : dict
            Mapping of mixing patch pair information.

        Raises
        ------
        ValueError
            If the two sides of a plane disagree on ``rf_exchange``.
        """
        self._grid = grid
        self.pairs = {}
        self._prune_pairs(mixing_pairs)
        self._check_rf_exchange()

        # Per-pair diagnostic snapshots, lazily allocated on first exchange.
        # Keys: (bid, pid). Values: dict with 'du' (the relaxation increment in
        # the exchanged target's own variables, shape (nspan, 5)).
        self._pair_state = {}

        # Scratch buffers, lazily allocated on first exchange
        self._vec1 = None
        self._vec2 = None
        self._jac_buf = None
        self._mdot_buf = None

        # Previous entering flag per (bid, pid, side), the hysteresis state for
        # _calc_shared_entering. Keyed per side because the two patches of a
        # pair have opposite _sign_interior and so, in general, opposite flags.
        self._entering_state = {}

    def _check_rf_exchange(self):
        """Raise if either side of a plane would relax the exchange differently.

        The exchange writes one shared target, so a pair holds one relaxation
        factor; the exchange reads it from the first side. Checked once here
        rather than per exchange, so a value changed on one side alone
        afterwards -- which the solver's push cannot do, since it writes the
        same value to every patch -- goes unnoticed.
        """
        for bid, pid in self.pairs:
            patch1, patch2 = self._get_pair(bid, pid)
            if patch1.reflective:
                # No exchange to relax; see MixingCommunicator._mix_uniform.
                continue
            if patch1.rf_exchange != patch2.rf_exchange:
                raise ValueError(
                    f"Mixing plane sides disagree on rf_exchange: "
                    f"{patch1.label!r} has {patch1.rf_exchange}, "
                    f"{patch2.label!r} has {patch2.rf_exchange}"
                )

    def _prune_pairs(self, mixing_pairs):
        """Prune bidirectional pairs to unidirectional mapping."""
        seen_pairs = set()

        for (bid, pid), ((nxbid, nxpid), transform) in mixing_pairs.items():
            pair_key = tuple(sorted([(bid, pid), (nxbid, nxpid)]))

            if pair_key not in seen_pairs:
                if (bid, pid) < (nxbid, nxpid):
                    self.pairs[(bid, pid)] = ((nxbid, nxpid), transform)
                else:
                    reverse_transform = mixing_pairs.get((nxbid, nxpid))
                    if reverse_transform is not None:
                        self.pairs[(nxbid, nxpid)] = reverse_transform

                seen_pairs.add(pair_key)

    def _ensure_pair_state(self, key, nspan):
        """Allocate or resize the per-pair state for the given pair."""
        state = self._pair_state.get(key)
        if state is None or state["du"].shape[0] != nspan:
            self._pair_state[key] = {
                # Relaxation increment in target space, kept for get_stats.
                "du": np.zeros((nspan, 5), dtype=np.float32),
                # Previous exchange's target-space mismatch, for the phase lead
                # of MixingPatch.lead_exchange. Not a diagnostic: the lead is
                # part of the relaxation. False until one exchange has run, so
                # the first takes no lead rather than differencing against zero.
                "e_prev": np.zeros((nspan, 5), dtype=np.float32),
                "has_prev": False,
            }
        return self._pair_state[key]

    def _get_pair(self, bid, pid):
        """Return the two patch objects either side of one mixing plane."""
        patch1 = self._grid[bid].patches[pid]
        (nxbid, nxpid), _ = self.pairs[(bid, pid)]
        patch2 = self._grid[nxbid].patches[nxpid]
        return patch1, patch2

    def _ensure_buffers(self, nspan):
        """Allocate or resize scratch buffers for the given spanwise size."""
        if self._vec1 is None or self._vec1.shape[0] < nspan:
            self._vec1 = util.empty((nspan, 5))
            self._vec2 = util.empty((nspan, 5))
            self._jac_buf = util.empty((nspan, 5, 5))
            self._mdot_buf = util.empty((nspan, 5))

    @profile
    def _exchange_pair(self, bid, pid, flip, mdot_target, gain):
        """Compute cross-plane targets and write absolute values into each patch.

        Performs inter-patch communication only; does not apply the targets to
        block_view.conserved.  Call the patches' apply() on each side afterwards.
        """
        patch1, patch2 = self._get_pair(bid, pid)

        if patch1.reflective:
            self._mix_uniform(patch1, patch2, flip)
            return

        b_avg, nspan = self._prepare_pair(patch1, patch2, flip, mdot_target, gain)
        self._write_targets(patch1, patch2, flip, b_avg, nspan, (bid, pid))

    def _mix_uniform(self, patch1, patch2, flip):
        r"""Hand both sides of a reflective plane their common mixed-out state.

        The whole of the exchange for a plane whose two sides carry
        :attr:`~ember.mixing.MixingPatch.reflective`: circumferentially average
        each face, average the two, and give the result to both. No
        characteristic split, no Jacobians, no relaxation and no mass flow
        forcing -- see the flag's own documentation for what that costs and
        what it does not.

        Worked in ``(x, r)`` components rather than in the interface frame, and
        on the conserved variables rather than on the fluxes, which is what
        makes it this short: the two sides share a meridional geometry and a
        radius, so their absolute-frame
        :math:`[\rho, \rho V_x, \rho V_r, \rho r V_\theta, \rho e]` are
        directly comparable with nothing resolved first.
        :meth:`~ember.mixing.MixingPatch.set_block_avg` holds no rotation in
        this mode for that reason.

        The mean is written to both sides on the spot rather than accumulated
        onto anything, so it is the *current* mixed-out state that each face
        will impose until the next exchange, one outer timestep later.
        """
        patch1.set_block_avg()
        patch2.set_block_avg()

        cons1 = patch1.block_avg.conserved_nd
        cons2 = patch2.block_avg.conserved_nd
        if flip:
            cons2 = cons2[::-1]

        nspan = cons1.shape[0]
        self._ensure_buffers(nspan)
        mean = self._vec1[:nspan]
        mean[:] = cons1
        mean += cons2
        mean *= 0.5

        patch1.set_uniform(mean)
        patch2.set_uniform(mean[::-1] if flip else mean)

    def _prepare_pair(self, patch1, patch2, flip, mdot_target=None, gain=0.0):
        """Symmetrise the cross-plane average and reduce the flux mismatch to chic space.

        Leaves the characteristic mismatch ``dchic`` in the shared scratch
        buffer ``self._vec1[:nspan]``, which :meth:`_write_targets` consumes.

        Returns
        -------
        b_avg : Block
            The symmetrised pitch-averaged state both sides now share, with its
            axial Mach number clipped away from zero. Every Jacobian downstream
            is evaluated on it, so both sides see the same linearisation.
        nspan : int
            Number of span stations; the length the scratch buffers are sliced to.
        """
        # The patches have to agree on a common pitch-avg state
        # before exchanging, before resolving to interface coordinates

        # First area average the flow field into respective block_avg
        patch1.set_block_avg()
        patch2.set_block_avg()

        # Extract conserved variables
        cons1 = patch1.block_avg.conserved_nd
        cons2 = patch2.block_avg.conserved_nd

        # Compute pitch-averages
        patch1.set_flux_avg()
        patch2.set_flux_avg()
        flux1 = patch1.flux_avg_nd
        flux2 = patch2.flux_avg_nd

        if flip:
            cons2 = cons2[::-1]
            flux2 = flux2[::-1]

        # Ensure scratch buffers are allocated
        # One shared for all patches which may have different size
        # so we have to slice to nspan
        nspan = cons1.shape[0]
        self._ensure_buffers(nspan)
        v1 = self._vec1[:nspan]
        v2 = self._vec2[:nspan]
        J = self._jac_buf[:nspan]

        # Take arithmetic average of conserved variables
        v2[:] = cons1
        v2 += cons2
        v2 *= 0.5  # v2 = cons_avg

        # Put back into block_avg for use in perturbation Jacobians
        patch1.block_avg.conserved_nd[...] = v2
        patch2.block_avg.conserved_nd[...] = v2[::-1] if flip else v2

        patch1.block_avg.update_cached_conserved()
        patch2.block_avg.update_cached_conserved()

        # The symmetrised interface state in target space, which _write_targets
        # relaxes the mismatch onto. Taken here because the clip below is about
        # to move the axial momentum, and a baseline is wanted for the state the
        # two sides actually share rather than for the one the Jacobians are
        # evaluated on.
        b = patch1.block_avg
        self._baseline = np.stack(
            [b.ho_nd, b.s_nd, b.Vr_nd, b.Vt_nd, b.P_nd], axis=-1
        ).reshape(nspan, 5)

        # Store the flux difference in v1
        v1[:] = flux2
        v1 -= flux1

        # Clip b_avg axial Mach into [Ma_clip, Ma_clip_max] before evaluating
        # Jacobians. np.sign is not usable for the direction: it returns 0 at
        # Max == 0, so a stalled station would be clipped to exactly zero axial
        # momentum -- the one value the clip exists to keep out of
        # flux_to_primitive, which divides by it two lines below. A station with
        # no direction of its own takes the downstream one.
        b_avg = patch1.block_avg
        Max = b_avg.Max
        Max_abs = np.abs(Max)
        outside = (Max_abs < self.Ma_clip) | (Max_abs > self.Ma_clip_max)
        if outside.any():
            sign = np.where(Max >= 0.0, 1.0, -1.0)
            Ma_lim = np.clip(Max_abs, self.Ma_clip, self.Ma_clip_max)
            rhoVx_clip = sign * Ma_lim * b_avg.rho_nd * b_avg.a_nd
            b_avg.conserved_nd[..., 1] = np.where(
                outside, rhoVx_clip, b_avg.conserved_nd[..., 1]
            )
            b_avg.update_cached_conserved()

        # Holmes' mass flow control: an offset to the flux difference, applied
        # here so that it goes through the transformation chain and the
        # characteristic selection below along with everything else, which is
        # where his Eq. 20 puts it.
        self._apply_mdot_offset(
            patch1, patch2, b_avg, v1, flux1, flux2, flip, mdot_target, gain
        )

        # Convert flux difference to chic difference using sequential Jacobians
        perturbation.flux_to_primitive(b_avg, out=J)
        util.matvec(J, v1, out=v1)  # v1 = dprim
        perturbation.primitive_to_chic(b_avg, out=J)
        util.matvec(J, v1, out=v1)  # v1 = dchic

        # Stamp one shared entering direction per span station on both
        # patches, from the state they now both hold. Each patch's own
        # _calc_entering would otherwise read its own local interior, and a
        # mixed-sign station could then have the two sides disagree on which
        # characteristic split they are on -- the split _write_targets is
        # about to build assumes one direction shared by both.
        key = (patch1.label, patch2.label)
        patch1._entering_shared = self._calc_shared_entering(patch1, (*key, 1))
        patch2._entering_shared = self._calc_shared_entering(patch2, (*key, 2))

        return b_avg, nspan

    def _apply_mdot_offset(
        self, patch1, patch2, b_avg, v1, flux1, flux2, flip, mdot_target, gain
    ):
        r"""Add Holmes' mass flow control offset to the cross-plane flux difference.

        :cite:t:`Holmes2008` Eq. 20-22. He replaces the flux difference the
        interface drives to zero,

        .. math::

            \Delta F = F^{(1)} - F^{(2)} + \overline{\Delta F},

        where side 1 is the upstream one and the offset's mass row is
        :math:`F^{(2)}_{\dot m} - \overline{F}^{(1)}_{\dot m}`, so that at the
        fixed point the upstream mass flux sits on its target rather than merely
        matching the other side. The remaining four rows of the offset are not
        free: they are the flux perturbation consistent with that change of mass
        flux at constant :math:`h_0`, entropy and flow angle, which is his
        Appendix B and here is :func:`~ember.perturbation.mdot_to_flux`.

        Applied to ``v1`` in flux space and in place, before the transformation
        chain, so the offset passes through the characteristic selection with
        the rest of the mismatch exactly as his Eq. 15 has it. An earlier
        version of this added the correction in mix space after the split
        instead; that skipped the selection, which is the part of the scheme
        that makes the interface non-reflecting.

        Two departures from the paper remain, both deliberate. His target is
        prescribed by the user; this one is the machine's own mean mass flow,
        handed down by :meth:`~ember.grid.Grid.update_bconds`, so it needs no
        input and applies to an unthrottled run. And his invariant relative flow
        angle is replaced by the absolute one; see
        :func:`~ember.perturbation.mdot_to_flux`.

        ``gain`` scales the offset, so 1 is exactly Holmes and 0 is the plain
        flux balance this had before. Note what the intermediate values mean:
        the mass row of the error runs from ``m_dn - m_up`` at zero gain to
        ``m_up*eps`` at unit gain, so the gain trades the interface's direct
        drive on the mass *balance* for a drive on the mass *flow*. At gain 1
        the balance term is gone entirely, which is his design.

        Which side is upstream is Holmes' one irreducible asymmetry and it is
        load-bearing. The correction's only real lever is the pressure, which
        the side the flow leaves imposes, so the flow whose error it can act on
        is that side's -- his "we elect to try to control the mass flow on the
        upstream (i.e. the exit) side". Driving the mean of the two sides
        instead, which would need no side at all, is degenerate: on a two-row
        grid the plane's own mean equals the mean of the two ends identically
        whenever each row carries a uniform flow, so it would read zero error
        however far apart the two sides had drifted.

        Nothing is declared, though, and nothing depends on labelling. The
        upstream side is the one whose ``_sign_interior`` the flow itself
        settled in :meth:`~ember.mixing.MixingPatch._enter_resolved`, so the
        result is the same whichever patch of the pair happened to sort first in
        :meth:`_prune_pairs`, and a plane whose frame has not settled yet simply
        goes unforced for another step.

        The apportionment is Holmes' (his closing paragraph): the target at each
        span station is that station's current mass flux scaled by one ratio for
        the whole span. Spreading ``eps`` flat across the span instead would be
        a small fractional change mid-span and an enormous one at the endwalls,
        where the mass flux is small, and would reshape the spanwise profile
        rather than scale it.

        Does nothing where there is nothing to apply -- no target, no gain, an
        unsettled frame, inside the deadband, or a plane whose net flow has
        reversed or is not measurable.

        Parameters
        ----------
        patch1, patch2 : MixingPatch
            The two sides, read for their settled frame, span areas and blade
            counts.
        b_avg : Block
            The symmetrised mean state, already Mach-clipped, that the
            sensitivity is evaluated on.
        v1 : array
            The cross-plane flux difference, shape ``(nspan, 5)``, modified in
            place.
        flux1, flux2 : array
            The two sides' pitch-averaged fluxes, ``flux2`` already in
            ``patch1``'s span order.
        flip : bool
            Whether ``patch2``'s span runs opposite to ``patch1``'s.
        mdot_target : float or None
            Target annulus mass flow, nondimensional on
            ``rhoV_ref * L_ref**2``. None disables the offset.
        gain : float
            Fraction of Holmes' offset to apply; zero disables it.
        """
        if not gain or mdot_target is None:
            return
        if not (patch1._sign_settled and patch2._sign_settled):
            return

        # The side the flow leaves through. Read from the frame the flow
        # settled, not from geometry or class. Its flux is already in patch1's
        # span order, as is b_avg, so the profile lines up with the sensitivity
        # without further flipping. ``sign`` carries ember's flux difference
        # convention, v1 = flux2 - flux1, back to Holmes' F(1) - F(2).
        if patch1._sign_interior < 0:
            sign, patch_up, m_up, dA_up = 1.0, patch1, flux1[:, 0], patch1._dA_node
        else:
            dA2 = patch2._dA_node[::-1] if flip else patch2._dA_node
            sign, patch_up, m_up, dA_up = -1.0, patch2, flux2[:, 0], dA2

        # Per unit area and per passage; only the total needs Nb, and the two
        # sides may carry different blade counts.
        blk_up = patch_up.block
        mdot_up = float(np.sum(m_up * dA_up)) * blk_up.Nb / blk_up.L_ref**2

        # A plane whose net flow has reversed has no forward mass flow to scale,
        # and the ratio below would be a large negative number rather than a
        # small error. Left alone; the characteristic split carries it.
        if not np.isfinite(mdot_up) or mdot_up <= 0.0:
            return

        eps = mdot_target / mdot_up - 1.0
        ramp = np.clip(
            (abs(eps) - self.eps_deadband) / (self.eps_full - self.eps_deadband),
            0.0,
            1.0,
        )
        if ramp <= 0.0:
            return

        # The mass row of the offset, Holmes Eq. 22, in ember's sign. His
        # F(2) - F_bar(1) is the current cross-plane mass flux difference less
        # the apportioned target increment; v1[:, 0] already holds the former
        # (up to ``sign``), so it is not measured again here.
        #
        # Gated on the ABSOLUTE Mach number, not the normal one the clip above
        # bounds: mdot_to_flux divides by rho*(1 - Ma^2) in the absolute Mach,
        # so a strongly swirling station can approach the singularity with a
        # modest normal component. Written to catch NaN as well.
        scale = v1[:, 0] - sign * m_up * eps
        Ma = np.asarray(b_avg.Ma).reshape(-1)
        scale = np.where(Ma < self.Ma_mdot_max, scale, 0.0)

        nspan = flux1.shape[0]
        col = perturbation.mdot_to_flux(b_avg, out=self._mdot_buf[:nspan])
        v1 -= (gain * ramp) * scale[:, None] * col

    def _calc_shared_entering(self, patch, state_key):
        """One patch's entering flag from the shared symmetrised interface state.

        Mirrors :meth:`~ember.patch.NonReflectingPatch._calc_entering`,
        but reads only the interface state both patches now hold in
        ``patch.block_avg`` (already in this patch's own span order), not the
        patch's own interior -- the interior can differ between the two sides
        of a mixed-sign station, which is exactly what a shared direction is
        for. Hysteresis of :attr:`~ember.patch.NonReflectingPatch._frac_rev_off`
        is kept between exchanges, keyed by ``state_key``, so a station
        hovering about zero still settles into one split instead of
        alternating.

        Parameters
        ----------
        patch : MixingPatch
            The side to compute the flag for, read for its own
            ``_sign_interior`` and its already-oriented ``block_avg``.
        state_key : tuple
            Key identifying this side's hysteresis state across exchanges.

        Returns
        -------
        array
            Boolean, shape ``(nspan,)``.
        """
        avg = patch.block_avg
        sign = patch._sign_interior
        u_face = (sign * avg.Vx_nd).reshape(-1)
        a_nd = avg.a_nd.reshape(-1)

        on = u_face >= 0.0
        off = u_face < -patch._frac_rev_off * a_nd
        prev = self._entering_state.get(state_key)
        if prev is None or prev.shape != on.shape:
            result = on
        else:
            result = np.where(prev, ~off, on)
        self._entering_state[state_key] = result
        return result

    def _write_targets(self, patch1, patch2, flip, b_avg, nspan, key):
        """Project the characteristic mismatch into target space and write both sides.

        Reads ``dchic`` from the scratch buffer :meth:`_prepare_pair` left it
        in, splits it by direction of propagation -- the upstream-running
        pressure characteristic against the four downstream-running ones, which
        is :cite:t:`Saxer1993` Eq. 5.66 -- and relaxes the resulting mismatch
        onto the symmetrised baseline of the two sides' current targets.

        The split is expressed as a pair of row masks on the target vector, so
        :attr:`_chic_to_target` has to map characteristics into a space whose
        last row is the static pressure and whose first four rows are the
        quantities an inflow prescribes. Mix space satisfies that.
        """
        v1 = self._vec1[:nspan]
        v2 = self._vec2[:nspan]
        J = self._jac_buf[:nspan]

        # Split into upstream/downstream contributions in chic space. Which
        # acoustic is incoming to the pressure-reading side depends on the
        # direction of the mean throughflow, exactly as each patch's own
        # incoming-characteristic table does (see ember.nonreflecting): row 0
        # (Vx - a) feeds the P target where the mean runs forward, row 1
        # (Vx + a) where it has reversed. The three convective characteristics
        # (rows 2-4) always feed the four inflow quantities. Splitting by fixed
        # row index instead -- as if every station ran forward -- feeds a
        # reversed station's inflow rows from the wrong acoustic, so the
        # exchange and the boundary condition disagree on which characteristic
        # is incoming and a standing pitch-mean flux mismatch is left across the
        # plane.
        #
        # The direction is read from patch1's shared entering flag
        # (_calc_shared_entering, computed above in _prepare_pair), not
        # independently from the raw sign of b_avg.Max: that flag is exactly
        # what each patch's own _calc_entering now returns (Change 2), so this
        # is the same hysteresis-damped decision the patches split their own
        # incoming/outgoing characteristics on. Reading the raw instantaneous
        # sign here instead, with no hysteresis, could disagree with the
        # patches' own lagged decision for an exchange or two near a
        # crossing -- fine for the old proportional relaxation, which
        # re-derives its correction from scratch every step and forgets a bad
        # one immediately, but not for the integrating form below, which
        # accumulates whatever it is given and has no way to tell a
        # wrong-direction contribution from a right one afterwards.
        idx = np.arange(nspan)
        p_row = np.where((patch1._sign_interior > 0) == patch1._entering_shared, 0, 1)
        v2[:] = v1  # copy dchic into v2
        acoustic = v2[idx, p_row].copy()  # the one acoustic bound for the P bucket
        v1[...] = 0.0
        v1[idx, p_row] = acoustic  # v1 = dchic_up (the pressure-side acoustic)
        v2[idx, p_row] = 0.0  # v2 = dchic_dn (other acoustic and convective)

        # Convert both to target space with the single fused Jacobian
        self._chic_to_target(b_avg, out=J)
        util.matvec(J, v1, out=v1)  # v1 = dtarget_up
        v1[..., :-1] = 0.0  # zero non-P contribution

        util.matvec(J, v2, out=v2)  # v2 = dtarget_dn
        v2[..., -1] = 0.0  # zero P contribution

        # Combine: v1 = dtarget = dtarget_up - dtarget_dn
        # Change to the first four rows comes from downstream chics
        # Change to P comes from upstream chics
        # For some reason need a -ve sign on dtarget_dn here!
        v1 -= v2

        # v1 holds the error e_n = dtarget. Before it is scaled, take the phase
        # lead of MixingPatch.lead_exchange:
        #
        #   du = rf_exchange * (e_n + lead_exchange * (e_n - e_{n-1}))
        #
        # The exchange integrates a mismatch the rows answer only after a wave
        # has crossed them, and that delay -- not the damping -- is what limits
        # rf_exchange: it costs phase, and past a critical loop gain the plane
        # and the rows' mass storage oscillate and grow. A term in the rate of
        # change of the mismatch puts the phase back, so a larger rf_exchange
        # holds. lead_exchange is in steps, and wants to be of order the
        # oscillation period over 2*pi to be worth anything, since e_n moves
        # very little between exchanges.
        #
        # The stored error is the raw e_n, not the led one, or the lead would
        # compound on itself.
        state = self._ensure_pair_state(key, nspan)
        lead = patch1.lead_exchange
        if lead and state["has_prev"]:
            e_prev = state["e_prev"]
            v2[:] = v1
            v2 -= e_prev
            e_prev[:] = v1
            v1 += lead * v2
        else:
            state["e_prev"][:] = v1
        state["has_prev"] = True

        # The increment is du = rf_exchange * e_n, recorded for get_stats before
        # it is added onto the target below.
        v1 *= patch1.rf_exchange  # v1 = du
        state["du"][:] = v1

        # Integrate the target-space mismatch onto the previous target, not the
        # live interface baseline -- Holmes Eq. 15, applied to the auxiliary
        # cells rather than re-derived each step. At the fixed point
        # target_n = target_{n-1} forces rf_exchange*e_n = 0, i.e. exact flux
        # balance; re-anchoring to the baseline every step (the proportional
        # form this replaces) instead leaves a standing offset of size e_n
        # itself. The previous target is symmetrised across the two sides,
        # since before the first exchange each side has only seeded itself
        # from its own interior.
        target1 = patch1.get_target()
        target2 = patch2.get_target()
        if flip:
            target2 = target2[::-1]
        v2[:] = target1
        v2 += target2
        v2 *= 0.5  # v2 = target_{n-1}

        if self.leak:
            # Bleed the accumulated target back toward the live baseline, so
            # windup is bounded rather than merely under-relaxed. Zero by
            # default: engage only where a station is seen to wind up.
            v2 -= self.leak * (v2 - self._baseline)

        v2 += v1  # v2 = target_n, before the physical clamp

        self._clamp_physical(v2, patch1)

        # 0th-order extrapolation of targets at hub/casing walls
        v2[0] = v2[1]
        v2[-1] = v2[-2]

        # Assign targets back to patches with correct flip
        patch1.set_target(v2)
        patch2.set_target(v2[::-1] if flip else v2)

    _clamp_Ma_max_sq = 4.0
    r"""Square of the axial Mach number :meth:`_clamp_physical` rejects a
    target above.

    Not 1: the target is an aspiration the patches' own sigma-relaxed mean-mode
    solve chases gradually, not the physical face state itself (which the
    patch's own hard guard in :meth:`~ember.patch.NonReflectingPatch._calc_reference`
    already protects), so a target whose implied axial velocity briefly
    overshoots the subsonic boundary by a little is an ordinary transient of a
    converging integration, not a runaway. Rejecting it anyway -- tried at
    :math:`\mathit{Ma}\geq 1` during development -- snaps a recovering
    trajectory back to the baseline hard enough to leave it oscillating
    indefinitely instead of settling, which is worse than leaving it alone.
    Squared, and compared against the ratio of squares below, so no square root
    is needed at every exchange.
    """

    def _clamp_physical(self, target, patch):
        """Reject a station's updated target if it implies a non-physical state.

        The anti-windup safety net the integrating form of the relaxation
        needs (see :attr:`leak`): a target that has wound up past what the
        flow was ever in is not just wrong, it can be a state with no solution
        at all -- negative density, non-positive pressure, or an implied axial
        velocity so far past sonic that the mean-mode Newton solves of
        :class:`~ember.patch.NonReflectingPatch` were never going to
        recover it. Rejected stations fall back to :attr:`_baseline`, the live
        symmetrised interface state :meth:`_prepare_pair` just measured --
        physical by construction, unlike the wound-up target that failed the
        check.

        ``target`` is modified in place. Uses ``patch``'s fluid model; both
        sides of a plane share one, so either patch does.
        """
        fluid = patch.block_view.fluid
        ho, s, Vr, Vt, P = (target[..., i] for i in range(5))

        with np.errstate(invalid="ignore", divide="ignore"):
            rho, u = fluid.set_P_s(P, s)
            a = fluid.get_a(rho, u)
            h = fluid.get_h(rho, u)
            Vx_sq = 2.0 * (ho - h) - Vr**2 - Vt**2
            Max_sq = Vx_sq / a**2

        # A negative Vx_sq (no real axial velocity solves the energy balance),
        # of any magnitude, is not rejected on its own: right at a stalled or
        # lightly reversed station this is an ordinary excursion of a
        # converging integration -- the same territory the patches' own
        # Ma_clip exists to ride through -- and rejecting it as harshly as a
        # genuine runaway snaps a recovering trajectory back to the baseline
        # hard enough to leave it oscillating rather than settling, which was
        # tried and made things worse. Only the upper Mach bound, which a
        # negative Max_sq automatically satisfies, gates this check; a
        # negative Vx_sq large enough to matter shows up as non-physical
        # elsewhere first (rho or P failing the checks above).
        bad = (
            ~np.isfinite(rho)
            | (rho <= 0.0)
            | ~np.isfinite(P)
            | (P <= 0.0)
            | ~(Max_sq < self._clamp_Ma_max_sq)
        )
        if bad.any():
            target[bad] = self._baseline[bad]

    def get_stats(self, bid, pid):
        """Return last-step relaxation increment for one pair.

        Returns
        -------
        dict or None
            Key ``du`` (last relaxation increment in the exchanged target's own
            variables, shape ``(nspan, 5)``). Returns ``None`` if the pair has
            not been exchanged yet.
        """
        state = self._pair_state.get((bid, pid))
        if state is None:
            return None
        return {"du": state["du"].copy()}

    def exchange(self, mdot_target=None, gain=0.0):
        """Compute and write targets for all pairs (no apply step).

        A pair whose sides are
        :attr:`~ember.mixing.MixingPatch.reflective` takes neither argument and
        goes through :meth:`_mix_uniform` instead.

        Parameters
        ----------
        mdot_target : float or None, optional
            Mass flow the planes are forced toward, for the whole annulus and
            nondimensional on ``rhoV_ref * L_ref**2``, as
            :meth:`~ember.grid.Grid._calc_mdot_target` returns it. Default None,
            which is no forcing at all -- the exchange is then exactly
            :cite:t:`Saxer1993`'s, as it was before the forcing existed.
        gain : float, optional
            Gain on the forcing; zero, the default, also disables it. Passed
            per call rather than held here, as
            :meth:`~ember.patch.OutletPatch.update_target` takes its ``cfl``,
            so it is always the value the march is running at and nothing
            survives on a communicator that the grid's pickle drops.

        See Also
        --------
        MixingCommunicator._apply_mdot_offset : What the forcing does
        """
        for bid, pid in self.pairs.keys():
            _, flip = self.pairs[(bid, pid)]
            self._exchange_pair(bid, pid, flip, mdot_target, gain)
