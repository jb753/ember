

! No-op coarse-residual smoother: the plain (non-IRS) kernels pass this to
! mg_restrict_levels so the smoothing step is structurally absent (no sf_irs<=0
! test, no tri_coeffs call), rather than relying on smooth_residual_tri's internal
! guard. Signature matches smooth_residual_tri so either can be handed to the
! shared engine's `smoother` dummy argument.
subroutine mg_smooth_noop(dU, sf, work, ni, nj, nk)
    implicit none
    integer, intent(in) :: ni, nj, nk
    real, intent(in)    :: sf
    real, intent(inout) :: dU(*)
    real, intent(inout) :: work(*)
    ! Reference every dummy so -Werror=unused-dummy-argument stays quiet; the
    ! guard is never true, so this is a genuine no-op at runtime.
    if (ni < 0) dU(1) = sf + work(1) + real(nj + nk)
end subroutine mg_smooth_noop


! Gather the leading (nib,njb,nkb) corner of a level-1-strided accumulator into
! a tightly packed contiguous slice (so it can be handed to smooth_residual_tri).
subroutine mg_gather_corner(dst, src, ldi, ldj, ldk, nib, njb, nkb, np)
    implicit none
    integer, intent(in) :: ldi, ldj, ldk, nib, njb, nkb, np
    real, intent(in)    :: src(ldi, ldj, ldk, np)
    real, intent(out)   :: dst(nib, njb, nkb, np)
    integer :: ib, jb, kb, ip
    do ip = 1, np
        do kb = 1, nkb
            do jb = 1, njb
                do ib = 1, nib
                    dst(ib,jb,kb,ip) = src(ib,jb,kb,ip)
                end do
            end do
        end do
    end do
end subroutine mg_gather_corner


! Scale a (possibly smoothed) contiguous coarse residual into a correction
! IN PLACE: corr = corr * coef * dtblk. dtblk is level-1-strided (leading corner
! read).
!
! In place because the residual is gathered straight into corr_all's own slot
! for that level, which is compact and exactly the size the smoother wants, so
! the separate coarse-residual buffer this used to read from was a full copy of
! the level-1 residual earning nothing. Every element is read and written at its
! own index within one statement.
subroutine mg_scale_corr(corr, dtblk, coef, ldi, ldj, ldk, nib, njb, nkb, np)
    implicit none
    integer, intent(in) :: ldi, ldj, ldk, nib, njb, nkb, np
    real, intent(in)    :: dtblk(ldi, ldj, ldk)
    real, intent(in)    :: coef
    real, intent(inout) :: corr(nib, njb, nkb, np)
    integer :: ib, jb, kb, ip
    do ip = 1, np
        do kb = 1, nkb
            do jb = 1, njb
                do ib = 1, nib
                    corr(ib,jb,kb,ip) = corr(ib,jb,kb,ip) * coef * dtblk(ib,jb,kb)
                end do
            end do
        end do
    end do
end subroutine mg_scale_corr


! Scheme-agnostic fine term (the multigrid-off increment):  tmp = scale*dt_vol*q.
! Grouping (scale*dt_vol)*q matches the fused fine term in
! mg_prolong2x_fine_scatter, so an mg-off march is byte-identical to an mg-on
! march whose coarse correction is exactly zero (fmgrid == 0), up to that
! path's contraction of the added zero.
subroutine fine_term(q, dt_vol, scale, tmp, ni, nj, nk, np)
    implicit none
    integer, intent(in) :: ni, nj, nk, np
    real,    intent(in) :: q(ni-1, nj-1, nk-1, np)
    real,    intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: scale
    real, intent(out)   :: tmp(ni-1, nj-1, nk-1, np)
    integer :: i, j, k, ip
    do ip = 1, np
        do k = 1, nk-1
            do j = 1, nj-1
                do i = 1, ni-1
                    tmp(i,j,k,ip) = scale * dt_vol(i,j,k) * q(i,j,k,ip)
                end do
            end do
        end do
    end do
end subroutine fine_term


! ============================================================================
! Negative-feedback change limiter (multall's DAMP), applied to the ASSEMBLED
! increment.
!
! POSITION IS THE WHOLE POINT. multall applies this at tblock-p-2_3_1.f:7736,
! to STORE *after* the block-sum corrections have been summed in at 7710-7713.
! ember's previous limiter (removed in 7b4fd71) sat in set_residual, on the
! fine residual UPSTREAM of the restriction, and that is what broke it: the box
! sum is exact only because the residual is extensive and interior fluxes
! telescope, and a per-cell nonlinear rescaling applied before the sum destroys
! that, so the coarse levels were agglomerating a non-conservative field. Here
! the restriction has already happened and there is nothing left to break; the
! limiter only clips the increment that is about to reach the nodes.
!
! The formula is multall's, unchanged:
!
!   dU <- dU / (1 + |dU| / (dampin * mean|dU|))
!
! carried per conserved variable, as multall carries it (SUMFLUX is called once
! per VAR, so each variable is normalised by its own block mean) and as the
! removed ember limiter did (its ravg was likewise dimension 5).
!
! THE NORMALISER IS LAGGED ONE CALL. multall materialises STORE full-volume and
! makes two passes over it: one to accumulate SUMCHG, one to scale. ember's
! multigrid scatter is a rolling two-plane kernel that never materialises the
! increment (see mg_fine_scatter), so a same-call mean would cost either a
! full-volume temporary or a second traversal. Instead each call accumulates
! sum|dU| for the NEXT call and scales with what the previous call left in
! rfac. The mean moves slowly against the field it normalises, so this is a
! small approximation, but it IS an approximation multall does not make, and
! the first call of a march runs unlimited (rfac starts at zero).
!
! THE STORED VALUE IS SCALE-INVARIANT, which is what makes the lag survive RK.
! multall has one update per step; ember's RK has n_stage of them, and the
! increment scales with the stage coefficient alpha (1/4, 1/3, 1/2, 1 for four
! stages), so a mean lagged straight from the previous stage would be wrong by
! the ratio of consecutive alphas -- up to 4x, alternately over- and
! under-damping around the cycle. rfac therefore holds
!
!   ncell * scale / (dampin * sum|dU|)
!
! with scale the march coefficient the increment was built with (cfl for scree,
! alpha*cfl for an RK stage). sum|dU| is proportional to scale, so the stored
! quantity is not; the consumer divides by ITS OWN scale to recover the
! reciprocal mean. For an unchanged field the lag is then exact rather than
! merely close, and the scree path (one update per step, constant scale) is
! unaffected either way.
!
! rfac holds the reciprocal 1/(dampin*mean) rather than the mean itself, so the
! per-cell work is a multiply and one divide.
!
! OFF IS A SKIPPED PASS, NOT A NEUTRAL ONE. dampin <= 0 makes the whole limiter
! traversal disappear -- an early return here, an `if` around the plane pass in
! mg_fine_scatter -- rather than running it with rfac == 0 and relying on
! dU/(1 + |dU|*0) being dU. That identity is exact in IEEE, and it still is not
! enough: these routines are inlined into each other (see inline_info.txt), so
! merely READING the increment back between fine_term and the scatter changes
! how the compiler contracts the multiply-add that built it, moving rk_plain's
! increment by ~1 ULP. That is invisible on its own and not invisible at all in
! test_golden_coarse_increment, which pins the difference between a multigrid
! increment and a non-multigrid one, where the two nearly cancel and 1 ULP of
! each becomes 0.2% of the difference.
!
! So the guard is what makes an undamped march byte-identical to the
! pre-limiter code, and it has to be a guard. It is per call and per k-plane,
! never per cell, so the inner loops stay branch-free; this is the one place
! where configuration is resolved by a runtime test rather than by which block
! the caller invokes, and the reason is that the alternative was measurably
! wrong rather than merely inelegant.
!
! (The removed limiter kept fdamp/dampin as a division to stay bit-exact
! against a second fused code path that computed it that way. There is only one
! path here, so dampin is folded into rfac once per call instead.)
! ============================================================================


! Fold the accumulated sum|dU| into the scale-invariant normaliser the NEXT
! call will divide by its own scale to use. dampin <= 0 (limiter off) or a flat
! component (zero sum) both leave rfac at zero, the identity soft-clip.
subroutine damp_state_update(rfac, sum_abs, dampin, scale, ncell, np)
    implicit none
    integer, intent(in) :: np, ncell
    real,    intent(in) :: sum_abs(np), dampin, scale
    real, intent(out)   :: rfac(np)
    integer :: m
    do m = 1, np
        if (dampin > 0e0 .and. sum_abs(m) > 0e0) then
            rfac(m) = real(ncell) * scale / (dampin * sum_abs(m))
        else
            rfac(m) = 0e0
        end if
    end do
end subroutine damp_state_update


! Limit a materialised full-volume increment in place: scale by the lagged
! rfac, accumulate this call's sum|dU|, and roll rfac for the next call. This
! is the multigrid-OFF tail (scree_plain, rk_plain), which already holds the
! increment in tmp; the multigrid-ON path does the same arithmetic inside
! mg_fine_scatter's rolling buffer instead, where the increment is never
! materialised full-volume.
!
! The sum is over the UNSCALED increment, as multall's is -- SUMCHG is
! accumulated in the combine loop at 7714, before the limiter runs at 7736.
subroutine damp_increment(dU, rfac, dampin, scale, ni, nj, nk, np)
    implicit none
    integer, intent(in) :: ni, nj, nk, np
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, np)
    real, intent(inout) :: rfac(np)
    real,    intent(in) :: dampin, scale
    real    :: sum_abs(np), s, r, v, a
    integer :: i, j, k, ip
    ! Limiter off: no traversal at all, so fine_term's write is the last thing
    ! to touch dU and the scatter sees exactly what it saw before this routine
    ! existed. rfac is left at zero. See the guard note above -- reading dU back
    ! here is enough to perturb the increment by 1 ULP even when the arithmetic
    ! is the identity.
    if (dampin <= 0e0) return
    do ip = 1, np
        ! rfac is stored scale-invariant; recover this call's reciprocal mean.
        r = rfac(ip) / scale
        s = 0e0
        do k = 1, nk-1
            do j = 1, nj-1
                do i = 1, ni-1
                    v = dU(i,j,k,ip)
                    a = abs(v)
                    s = s + a
                    dU(i,j,k,ip) = v / (1e0 + a*r)
                end do
            end do
        end do
        sum_abs(ip) = s
    end do
    call damp_state_update(rfac, sum_abs, dampin, scale, &
                           (ni-1)*(nj-1)*(nk-1), np)
end subroutine damp_increment


! Form the Denton fine quantity in place in the history buffer:
!   store <- 2*residual - store
! The pre-roll store (dF/dt)_{n-1} is read once here and overwritten with the
! extrapolated q = 2*residual - store, which the engine/fine-term then consume as
! the scree fine quantity. The post-march scree_roll_and_scatter overwrites store
! again with residual, so no separate q buffer is needed. RK skips this and passes
! residual directly as q.
subroutine scree_form_q(store, residual, ni, nj, nk, np)
    implicit none
    integer, intent(in) :: ni, nj, nk, np
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real, intent(inout) :: store(ni-1, nj-1, nk-1, np)
    integer :: i, j, k, ip
    do ip = 1, np
        do k = 1, nk-1
            do j = 1, nj-1
                do i = 1, ni-1
                    store(i,j,k,ip) = 2e0*residual(i,j,k,ip) - store(i,j,k,ip)
                end do
            end do
        end do
    end do
end subroutine scree_form_q


! Roll the Denton history: store = residual, discarding the q it held. Called
! only after the engine/fine-term has consumed q from store. Split out of
! scree_roll_and_scatter so the fused scree tail, whose scatter happens inside
! mg_prolong2x_fine_scatter, can roll without a second increment buffer.
subroutine scree_roll(residual, store, ni, nj, nk, np)
    implicit none
    integer, intent(in) :: ni, nj, nk, np
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real, intent(inout) :: store(ni-1, nj-1, nk-1, np)
    integer :: i, j, k, ip
    do ip = 1, np
        do k = 1, nk-1
            do j = 1, nj-1
                do i = 1, ni-1
                    store(i,j,k,ip) = residual(i,j,k,ip)
                end do
            end do
        end do
    end do
end subroutine scree_roll


! Roll the Denton history and frozen-pressure accumulate the full-volume
! increment onto cons. Tail of the multigrid-OFF scree wrapper, which still
! materialises the increment (as rk_plain does); the multigrid-on wrappers use
! the fused mg_prolong2x_fine_scatter and scree_roll instead.
subroutine scree_roll_and_scatter(cons, residual, store, tmp, ni, nj, nk, np)
    implicit none
    integer, intent(in) :: ni, nj, nk, np
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: store(ni-1, nj-1, nk-1, np)
    real, intent(inout) :: tmp(ni-1, nj-1, nk-1, np)
    call scree_roll(residual, store, ni, nj, nk, np)
    call cell_to_node(tmp, cons, ni, nj, nk, np)
end subroutine scree_roll_and_scatter


! ============================================================================
! Hierarchical-restriction + injection block-sum multigrid engine.
!
! Scheme-agnostic: operates on a single pre-formed fine quantity `q` (residual
! for RK, 2*residual-store for scree), so it carries no `denton` branch. Called
! only by the mg-*on* wrappers, so n_levels >= 1 always (no n_levels==0 path).
! IRS is the `smoother` dummy-procedure argument: smooth_residual_tri_tiled
! (Jameson IRS) or mg_smooth_noop (none) -- no `if (sf_irs)` anywhere in here.
!
! The six production kernels, all branch-free straight-line compositions:
!   scree_plain     rk_plain      fine_term + scatter          (multigrid off)
!   scree_mg_noirs  rk_mg_noirs   restrict(mg_smooth_noop) + collapse + scatter
!   scree_mg_irs    rk_mg_irs     restrict(smooth_residual_tri_tiled) + ditto
! scree wrappers form q in store (scree_form_q) and roll the history
! (scree_roll); rk wrappers pass residual as q. Both multigrid-on wrappers
! scatter through the fused mg_fine_scatter, scree frozen-in-place
! (base = cons) and rk off the sub-stage snapshot (base = snapshot); the
! multigrid-off pair still materialises a full-volume increment.
!
! Restriction is HIERARCHICAL: a level-l block-sum equals eight level-(l-1)
! block-sums (the block-sum is associative), so only level 1 reads the fine grid;
! coarser levels reduce the small running accumulators -- rawbuf for the residual
! sum, sdt/sv for the volume-weighted dt. Level 1 is loop-peeled (it alone reads
! the fine grid), so the level loop has no `if (lvl==1)`. In-place reduction is
! safe because, with blocks visited in ascending order, an output cell maps to
! source cells whose every index is >= its own and no later block reads it.
!
! Prolongation is INJECTION: every fine cell under a coarse block takes that
! block's correction unaltered, which is what a real coarse solve gives and is
! exactly the transpose of the block-sum restriction -- on any mesh, with no
! normalisation, no weights and no geometry. mg_collapse_levels sums the levels
! in place inside corr_all and mg_fine_scatter reads the result, so the whole
! prolongation is two integer divides and an add.
!
! This replaced a cascade of factor-2 trilinear interpolations whose final hop
! targeted the fine NODES through geometry-derived weights. That scheme, its
! weight cache (Block.weight_mgrid) and the geometry ladder behind it are gone;
! see docs/dev/plan_piecewise_constant_mgrid.md for what it was and why it went.
! ============================================================================
subroutine mg_restrict_levels(q, dt_vol, vol, scale, fmgrid, expon_mgrid, &
        sf_irs, n_levels, dtblk, rawbuf, sdt, sv, &
        corr_all, triw, smoother, &
        ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_tri)

    implicit none
    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k
    integer, intent(in) :: n_corr, n_tri
    ! Coarse-residual smoother, chosen by the caller (no IRS branch in here):
    ! smooth_residual_tri_tiled for the IRS kernels, mg_smooth_noop for the plain ones.
    external :: smoother
    real, intent(in)    :: q(ni-1, nj-1, nk-1, np)
    real, intent(in)    :: dt_vol(ni-1, nj-1, nk-1)
    real, intent(in)    :: vol(ni-1, nj-1, nk-1)
    real, intent(in)    :: scale, fmgrid, expon_mgrid, sf_irs
    real, intent(inout) :: dtblk(nc1i, nc1j, nc1k)
    real, intent(inout) :: rawbuf(nc1i, nc1j, nc1k, np)
    ! sdt accumulates sum(vol/dt_vol) and sv accumulates sum(vol), so that
    ! dtblk = sv/sdt is the volume-weighted harmonic mean of dt_vol.
    real, intent(inout) :: sdt(nc1i, nc1j, nc1k)
    real, intent(inout) :: sv (nc1i, nc1j, nc1k)
    real, intent(inout) :: corr_all(n_corr)
    real, intent(inout) :: triw(n_tri)

    integer :: ip, lvl, b, nib, njb, nkb, ib, jb, kb
    integer :: ii, jj, kk, slot, cnt, o
    real    :: coef, s, s_dt, s_v
    ! Level table. Sized by a PARAMETER, not by n_levels: a runtime-sized
    ! local is an alloca on every call, and GCC's opt report shows such tables
    ! costing a stack_save/alloca/stack_restore trio that also clobbers memory
    ! across the whole body. There is a compile-time bound available --
    ! ember.block.MAX_MG_LEVELS, which ember.solver._validate_mg already
    ! enforces because Block.scratch is one arena sized for it -- so the table
    ! can simply be that long. Keep the two in step.
    integer, parameter :: MG_LEVELS_MAX = 3
    integer :: offc(MG_LEVELS_MAX)

    ! The caller validates this (ember.solver._validate_mg), so reaching it
    ! means the two bounds have drifted apart; say so rather than writing
    ! past the table.
    if (n_levels > MG_LEVELS_MAX) stop 'mg_restrict_levels: n_levels > MG_LEVELS_MAX'

    ! Coarsest-first packed geometry for corr_all, the direction
    ! mg_collapse_levels then accumulates in.
    o = 0
    do lvl = 1, n_levels
        b = 2**(n_levels - lvl + 1)
        offc(lvl) = o
        o = o + ((ni-1)/b)*((nj-1)/b)*((nk-1)/b)*np
    end do

    ! ---- Phase 1, level 1 (peeled): the only level that reads the fine grid ----
    lvl = 1
    b   = 2
    nib = (ni-1)/b
    njb = (nj-1)/b
    nkb = (nk-1)/b
    coef = scale * fmgrid / real(b*b) * expon_mgrid**(-(lvl-1))
    slot = n_levels - lvl + 1
    cnt  = nib*njb*nkb*np

    ! dt restriction (volume-weighted HARMONIC mean) from the fine grid. The
    ! block needs the reciprocal of the block's spectral radius, 1/<Lambda>,
    ! and dt_vol is 1/Lambda per cell, so the mean that belongs here is
    ! sum(vol) / sum(vol/dt_vol), NOT sum(dt_vol*vol)/sum(vol). By Jensen the
    ! arithmetic mean is the larger of the two whenever Lambda varies over the
    ! block, so it overstates the coarse timestep on a stretched mesh and
    ! over-drives the smallest cells in the block -- measured at up to 2.0x
    ! their own dt_vol on the clustered duct, against 1.01x on a uniform mesh.
    ! sdt therefore accumulates vol/dt_vol; sv still accumulates vol, and the
    ! coarser levels reduce both accumulators exactly as before. ifort declines
    ! to auto-vectorize the ib loop (remark #15541: "consider using SIMD
    ! directive") because its body is a manual 8-corner scalar reduction with
    ! stride-2 reads -- not an aliasing hazard (ib writes are all disjoint), so
    ! !DIR$ SIMD is a safe hint. gfortran already vectorizes this loop
    ! unaided.
    do kb = 1, nkb
        do jb = 1, njb
            !DIR$ SIMD
            do ib = 1, nib
                s_dt = 0e0
                s_v  = 0e0
                do kk = 2*kb-1, 2*kb
                    do jj = 2*jb-1, 2*jb
                        do ii = 2*ib-1, 2*ib
                            s_dt = s_dt + vol(ii,jj,kk)/dt_vol(ii,jj,kk)
                            s_v  = s_v  + vol(ii,jj,kk)
                        end do
                    end do
                end do
                sdt(ib,jb,kb)   = s_dt
                sv (ib,jb,kb)   = s_v
                dtblk(ib,jb,kb) = s_v / s_dt
            end do
        end do
    end do

    ! residual (fine quantity q) restriction from the fine grid into rawbuf.
    ! Same ifort vectorization gap and rationale as the dt restriction above.
    do ip = 1, np
        do kb = 1, nkb
            do jb = 1, njb
                !DIR$ SIMD
                do ib = 1, nib
                    s = 0e0
                    do kk = 2*kb-1, 2*kb
                        do jj = 2*jb-1, 2*jb
                            do ii = 2*ib-1, 2*ib
                                s = s + q(ii,jj,kk,ip)
                            end do
                        end do
                    end do
                    rawbuf(ib,jb,kb,ip) = s
                end do
            end do
        end do
    end do

    ! gather -> smooth -> scale into this level's corr slot.
    call mg_gather_corner(corr_all(offc(slot)+1), rawbuf, nc1i, nc1j, nc1k, &
                          nib, njb, nkb, np)
    call smoother(corr_all(offc(slot)+1:offc(slot)+cnt), sf_irs, &
                  triw(1:2*(nib+njb+nkb)), nib+1, njb+1, nkb+1)
    call mg_scale_corr(corr_all(offc(slot)+1), dtblk, coef, &
                       nc1i, nc1j, nc1k, nib, njb, nkb, np)

    ! ---- Phase 1, levels 2..n_levels: reduce the coarse accumulators ----
    do lvl = 2, n_levels
        b = 2**lvl
        nib = (ni-1)/b
        njb = (nj-1)/b
        nkb = (nk-1)/b
        coef = scale * fmgrid / real(b*b) * expon_mgrid**(-(lvl-1))
        slot = n_levels - lvl + 1
        cnt  = nib*njb*nkb*np

        ! dt reduction (accumulator, hierarchical in place).
        do kb = 1, nkb
            do jb = 1, njb
                do ib = 1, nib
                    s_dt = 0e0
                    s_v  = 0e0
                    do kk = 2*kb-1, 2*kb
                        do jj = 2*jb-1, 2*jb
                            do ii = 2*ib-1, 2*ib
                                s_dt = s_dt + sdt(ii,jj,kk)
                                s_v  = s_v  + sv (ii,jj,kk)
                            end do
                        end do
                    end do
                    sdt(ib,jb,kb)   = s_dt
                    sv (ib,jb,kb)   = s_v
                    dtblk(ib,jb,kb) = s_v / s_dt
                end do
            end do
        end do

        ! residual reduction (accumulator, hierarchical in place).
        do ip = 1, np
            do kb = 1, nkb
                do jb = 1, njb
                    do ib = 1, nib
                        s = 0e0
                        do kk = 2*kb-1, 2*kb
                            do jj = 2*jb-1, 2*jb
                                do ii = 2*ib-1, 2*ib
                                    s = s + rawbuf(ii,jj,kk,ip)
                                end do
                            end do
                        end do
                        rawbuf(ib,jb,kb,ip) = s
                    end do
                end do
            end do
        end do

        call mg_gather_corner(corr_all(offc(slot)+1), rawbuf, nc1i, nc1j, nc1k, &
                              nib, njb, nkb, np)
        call smoother(corr_all(offc(slot)+1:offc(slot)+cnt), sf_irs, &
                      triw(1:2*(nib+njb+nkb)), nib+1, njb+1, nkb+1)
        call mg_scale_corr(corr_all(offc(slot)+1), dtblk, coef, &
                           nc1i, nc1j, nc1k, nib, njb, nkb, np)
    end do

end subroutine mg_restrict_levels


! Accumulate a coarse field onto the next finer coarse level by INJECTION:
! every one of the eight fine cells under a coarse cell takes that cell's value,
! unaltered. The whole of the piecewise-constant prolongation is this and the
! identical read in mg_fine_scatter.
!
! No bracket and no clamp, unlike mg_prolong2x_acc: (i+1)/2 is exact only
! because ember.solver._validate_mg forces every cell dimension to be a multiple
! of 2**n_levels, so nfi == 2*nci in every direction. Relax that check and this
! reads past the end of src.
subroutine mg_inject_acc(src, nci, ncj, nck, out, nfi, nfj, nfk, np)
    implicit none
    integer, intent(in) :: nci, ncj, nck, nfi, nfj, nfk, np
    real, intent(in)    :: src(nci, ncj, nck, np)
    real, intent(inout) :: out(nfi, nfj, nfk, np)
    integer :: i, j, k, ip, jc, kc
    do ip = 1, np
        do k = 1, nfk
            kc = (k+1)/2
            do j = 1, nfj
                jc = (j+1)/2
                do i = 1, nfi
                    out(i,j,k,ip) = out(i,j,k,ip) + src((i+1)/2, jc, kc, ip)
                end do
            end do
        end do
    end do
end subroutine mg_inject_acc


! Phase 2, piecewise-constant: collapse every coarse level onto the FINEST
! coarse level (level 1), in place inside corr_all.
!
! corr_all already holds each level's scaled correction in its own compact,
! disjoint slot, packed coarsest first, so no accumulator is needed at all.
! Working coarsest first, each slot gains the injected total of the one above
! it, so the last slot ends holding sum_l inject_l(corr_l) and the fine grid
! reads that.
!
! Cost is 1 + 1/8 + 1/64 = 1.14 traversals of the level-1 grid, about 0.14
! fine-cell-equivalents: nothing here touches the fine grid at all.
!
! The two slots handed to each mg_inject_acc call are non-overlapping sections
! of one array (offc is strictly increasing and slots are exactly their own
! size), and every target element is written once, so the aliasing is only
! apparent.
subroutine mg_collapse_levels(corr_all, n_levels, ni, nj, nk, np, n_corr)
    implicit none
    integer, intent(in) :: n_levels, ni, nj, nk, np, n_corr
    real, intent(inout) :: corr_all(n_corr)
    integer, parameter :: MG_LEVELS_MAX = 3
    integer :: dib(MG_LEVELS_MAX), djb(MG_LEVELS_MAX), dkb(MG_LEVELS_MAX)
    integer :: offc(MG_LEVELS_MAX)
    integer :: lvl, b, o

    if (n_levels > MG_LEVELS_MAX) stop 'mg_collapse_levels: n_levels > MG_LEVELS_MAX'

    o = 0
    do lvl = 1, n_levels
        b = 2**(n_levels - lvl + 1)
        dib(lvl) = (ni-1)/b
        djb(lvl) = (nj-1)/b
        dkb(lvl) = (nk-1)/b
        offc(lvl) = o
        o = o + dib(lvl)*djb(lvl)*dkb(lvl)*np
    end do

    do lvl = 2, n_levels
        call mg_inject_acc(corr_all(offc(lvl-1)+1), &
                           dib(lvl-1), djb(lvl-1), dkb(lvl-1), &
                           corr_all(offc(lvl)+1), &
                           dib(lvl), djb(lvl), dkb(lvl), np)
    end do
end subroutine mg_collapse_levels


! Fused fine term + injected coarse correction + cell->node scatter: the
! piecewise-constant counterpart of mg_prolong2x_fine_scatter, and the only
! place the fine grid is touched by the correction.
!
! Under injection the correction is a CELL quantity, like the fine term, so
! both are added into the increment and ride the one scatter. The scheme this
! replaced needed two routes to the nodes and a weight cache to take the second
! of them; neither has a counterpart here.
!
! Cell-averaging the correction costs nothing, which is the whole argument for
! doing it: within a coarse block the correction is constant, so the scatter --
! a partition of unity everywhere, interior 1/8 of 8 cells through to corners
! 1 of 1 -- reproduces it exactly. The two differ only at block faces, where
! the node takes the mean of the two adjoining blocks' corrections. That is a
! one-cell smoothing of the staircase applied exactly where the staircase is,
! not the first-order clustering error that made the old scheme target nodes.
!
! The rolling two-plane rbuf and the emit stencils are carried over unchanged
! (term for term and in the same summation order as cell_to_node_generic), so
! the increment is still never materialised full-volume.
!
! src((i+1)/2, ...) repeats each coarse value across two fine cells. It reads
! from an array an eighth the size, resident in cache, so even scalarised it is
! cheap against the dt_vol and q streams it is added to.
subroutine mg_fine_scatter(src, nci, ncj, nck, base, cons, &
        scale, dt_vol, q, ni, nj, nk, np, rbuf, rfac, dampin)
    implicit none
    integer, intent(in) :: nci, ncj, nck, ni, nj, nk, np
    real, intent(in)    :: src(nci, ncj, nck, np)
    real, intent(in)    :: scale
    real, intent(in)    :: dt_vol(ni-1, nj-1, nk-1)
    real, intent(in)    :: q(ni-1, nj-1, nk-1, np)
    real, intent(in)    :: base(ni, nj, nk, np)
    ! As in mg_prolong2x_fine_scatter: RK passes a snapshot distinct from cons,
    ! scree passes cons itself and the aliasing is benign (every node is read
    ! and written at its own index within one statement).
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: rbuf(ni-1, nj-1, np, 2)
    ! Change-limiter state: lagged reciprocal normaliser in, rolled on exit.
    ! dampin <= 0 skips the limiter pass entirely and leaves rfac at zero. See
    ! the damp_increment block above for why it is skipped rather than run
    ! neutrally.
    real, intent(inout) :: rfac(np)
    real,    intent(in) :: dampin
    real    :: sum_abs(np), s, r, v, a
    integer :: i, j, ip, kc, kb, jb, cur, prev, sw

    do ip = 1, np
        sum_abs(ip) = 0e0
    end do

    cur  = 1
    prev = 2
    do kc = 1, nk-1
        kb = (kc+1)/2
        do ip = 1, np
            do j = 1, nj-1
                jb = (j+1)/2
                do i = 1, ni-1
                    rbuf(i,j,ip,cur) = scale*dt_vol(i,j,kc)*q(i,j,kc,ip) &
                                     + src((i+1)/2, jb, kb, ip)
                end do
            end do
        end do
        ! The limiter is applied to rbuf HERE -- fine term plus injected coarse
        ! correction, which is exactly multall's STORE at 7710-7713. It must
        ! land before the emits below, because emit_kint reads both the plane
        ! just built (cur) and the one before it (prev), and both must already
        ! be limited when they reach the nodes.
        !
        ! Deliberately a SECOND pass over the plane rather than fused into the
        ! assembly above, which is what it looks like it should be. Carrying the
        ! assembled value through a temporary lets the compiler contract the
        ! multiply-add differently, which moves the increment by ~1 ULP even
        ! when the arithmetic is the identity. Leaving the assembly statement
        ! untouched, and skipping this pass entirely when the limiter is off,
        ! are together what keep an undamped march byte-identical to the
        ! pre-limiter code. The cost when it IS on is a read and a write over
        ! two k-planes of rbuf, resident in cache, against the dt_vol and q
        ! streams the assembly already pulls from memory.
        if (dampin > 0e0) then
            do ip = 1, np
                ! rfac is stored scale-invariant; recover this call's reciprocal
                ! mean. See the damp_increment block above.
                r = rfac(ip) / scale
                s = 0e0
                do j = 1, nj-1
                    do i = 1, ni-1
                        v = rbuf(i,j,ip,cur)
                        a = abs(v)
                        s = s + a
                        rbuf(i,j,ip,cur) = v / (1e0 + a*r)
                    end do
                end do
                sum_abs(ip) = sum_abs(ip) + s
            end do
        end if
        ! Both ends emit two node planes in one pass.
        if (kc == 1)    call emit_kbnd(cur, 1)
        if (kc >= 2)    call emit_kint(prev, cur, kc)
        if (kc == nk-1) call emit_kbnd(cur, nk)
        sw = cur; cur = prev; prev = sw
    end do

    call damp_state_update(rfac, sum_abs, dampin, scale, &
                           (ni-1)*(nj-1)*(nk-1), np)

contains

    ! Interior-k node plane kk (2..nk-1): 8 surrounding cells, planes kk-1
    ! (buffer bp) and kk (buffer bc).
    subroutine emit_kint(bp, bc, kk)
        integer, intent(in) :: bp, bc, kk
        integer :: i, j, ip
        do ip = 1, np
            do j = 2, nj-1
                do i = 2, ni-1
                    cons(i,j,kk,ip) = base(i,j,kk,ip) + ( &
                        rbuf(i-1,j-1,ip,bp) + rbuf(i,j-1,ip,bp) &
                      + rbuf(i,j,ip,bp)     + rbuf(i-1,j,ip,bp) &
                      + rbuf(i-1,j-1,ip,bc) + rbuf(i,j-1,ip,bc) &
                      + rbuf(i,j,ip,bc)     + rbuf(i-1,j,ip,bc))*0.125e0
                end do
            end do
            do j = 2, nj-1
                cons(1,j,kk,ip) = base(1,j,kk,ip) + ( &
                    rbuf(1,j-1,ip,bp) + rbuf(1,j,ip,bp) &
                  + rbuf(1,j-1,ip,bc) + rbuf(1,j,ip,bc))*0.25e0
                cons(ni,j,kk,ip) = base(ni,j,kk,ip) + ( &
                    rbuf(ni-1,j-1,ip,bp) + rbuf(ni-1,j,ip,bp) &
                  + rbuf(ni-1,j-1,ip,bc) + rbuf(ni-1,j,ip,bc))*0.25e0
            end do
            do i = 2, ni-1
                cons(i,1,kk,ip) = base(i,1,kk,ip) + ( &
                    rbuf(i-1,1,ip,bp) + rbuf(i,1,ip,bp) &
                  + rbuf(i-1,1,ip,bc) + rbuf(i,1,ip,bc))*0.25e0
                cons(i,nj,kk,ip) = base(i,nj,kk,ip) + ( &
                    rbuf(i-1,nj-1,ip,bp) + rbuf(i,nj-1,ip,bp) &
                  + rbuf(i-1,nj-1,ip,bc) + rbuf(i,nj-1,ip,bc))*0.25e0
            end do
            cons(1,1,kk,ip) = base(1,1,kk,ip) &
                + (rbuf(1,1,ip,bp) + rbuf(1,1,ip,bc))*0.5e0
            cons(1,nj,kk,ip) = base(1,nj,kk,ip) &
                + (rbuf(1,nj-1,ip,bp) + rbuf(1,nj-1,ip,bc))*0.5e0
            cons(ni,nj,kk,ip) = base(ni,nj,kk,ip) &
                + (rbuf(ni-1,nj-1,ip,bp) + rbuf(ni-1,nj-1,ip,bc))*0.5e0
            cons(ni,1,kk,ip) = base(ni,1,kk,ip) &
                + (rbuf(ni-1,1,ip,bp) + rbuf(ni-1,1,ip,bc))*0.5e0
        end do
    end subroutine emit_kint

    ! k-boundary node plane kk (1 or nk): single adjacent cell plane (buffer
    ! bc). Interior 1/4 of 4 cells, i/j faces 1/2 of 2, corners the one cell.
    subroutine emit_kbnd(bc, kk)
        integer, intent(in) :: bc, kk
        integer :: i, j, ip
        do ip = 1, np
            do j = 2, nj-1
                do i = 2, ni-1
                    cons(i,j,kk,ip) = base(i,j,kk,ip) + ( &
                        rbuf(i-1,j-1,ip,bc) + rbuf(i,j-1,ip,bc) &
                      + rbuf(i-1,j,ip,bc)   + rbuf(i,j,ip,bc))*0.25e0
                end do
            end do
            do j = 2, nj-1
                cons(1,j,kk,ip) = base(1,j,kk,ip) &
                    + (rbuf(1,j-1,ip,bc) + rbuf(1,j,ip,bc))*0.5e0
                cons(ni,j,kk,ip) = base(ni,j,kk,ip) &
                    + (rbuf(ni-1,j-1,ip,bc) + rbuf(ni-1,j,ip,bc))*0.5e0
            end do
            do i = 2, ni-1
                cons(i,1,kk,ip) = base(i,1,kk,ip) &
                    + (rbuf(i-1,1,ip,bc) + rbuf(i,1,ip,bc))*0.5e0
                cons(i,nj,kk,ip) = base(i,nj,kk,ip) &
                    + (rbuf(i-1,nj-1,ip,bc) + rbuf(i,nj-1,ip,bc))*0.5e0
            end do
            cons(1,1,kk,ip)   = base(1,1,kk,ip)   + rbuf(1,1,ip,bc)
            cons(1,nj,kk,ip)  = base(1,nj,kk,ip)  + rbuf(1,nj-1,ip,bc)
            cons(ni,nj,kk,ip) = base(ni,nj,kk,ip) + rbuf(ni-1,nj-1,ip,bc)
            cons(ni,1,kk,ip)  = base(ni,1,kk,ip)  + rbuf(ni-1,1,ip,bc)
        end do
    end subroutine emit_kbnd

end subroutine mg_fine_scatter


! ============================================================================
! The six production kernels. Each is a branch-free straight-line composition of
! the blocks above; configuration is resolved by which blocks are called and
! which smoother is passed, never by a runtime `if`.
! ============================================================================


! scree, multigrid off: form q, fine term only, roll history and frozen-scatter.
subroutine scree_plain(cons, residual, store, dt_vol, cfl, tmp, rfac, dampin, &
        ni, nj, nk, np)
    implicit none
    integer, intent(in) :: ni, nj, nk, np
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real,    intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: cfl, dampin
    real, intent(inout) :: store(ni-1, nj-1, nk-1, np)  ! in: (dF/dt)_{n-1}; out: (dF/dt)_n
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: tmp(ni-1, nj-1, nk-1, np)
    real, intent(inout) :: rfac(np)

    call scree_form_q(store, residual, ni, nj, nk, np)
    call fine_term(store, dt_vol, cfl, tmp, ni, nj, nk, np)
    call damp_increment(tmp, rfac, dampin, cfl, ni, nj, nk, np)
    call scree_roll_and_scatter(cons, residual, store, tmp, ni, nj, nk, np)
end subroutine scree_plain


! RK stage, multigrid off: fine term only (q = residual), scatter off snapshot.
subroutine rk_plain(cons, snapshot, residual, dt_vol, alpha, cfl, tmp, &
        rfac, dampin, ni, nj, nk, np)
    implicit none
    integer, intent(in) :: ni, nj, nk, np
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real,    intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: alpha, cfl, dampin
    real,    intent(in) :: snapshot(ni, nj, nk, np)
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: tmp(ni-1, nj-1, nk-1, np)
    real, intent(inout) :: rfac(np)

    call fine_term(residual, dt_vol, alpha*cfl, tmp, ni, nj, nk, np)
    call damp_increment(tmp, rfac, dampin, alpha*cfl, ni, nj, nk, np)
    ! cons = snapshot + cell_to_node(tmp). Distinct in/out (snapshot vs cons).
    call cell_to_node_generic(tmp, snapshot, cons, ni, nj, nk, np)
end subroutine rk_plain


! ============================================================================
! The four multigrid-on wrappers: scree/rk x IRS/no-IRS, each a three-call
! straight line
!
!   mg_restrict_levels  ->  mg_collapse_levels  ->  mg_fine_scatter
!
! taking seven arena buffers and no prolongation weights at all.
!
! mg_restrict_levels packs corr_all coarsest-first, so the finest coarse level
! -- the one mg_collapse_levels leaves the total in, and the one the fine grid
! reads -- is the LAST slot, at offset n_corr - nc1i*nc1j*nc1k*np.
! ============================================================================
! scree, multigrid on, coarse-level IRS.
subroutine scree_mg_irs(cons, residual, store, dt_vol, vol, cfl, &
        fmgrid, expon_mgrid, sf_irs, n_levels, rbuf, dtblk, rawbuf, sdt, sv, &
        corr_all, triw, rfac, dampin, &
        ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_tri)
    implicit none
    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k
    integer, intent(in) :: n_corr, n_tri
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real,    intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: cfl, fmgrid, expon_mgrid, sf_irs
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: store(ni-1, nj-1, nk-1, np)   ! in: (dF/dt)_{n-1}; out: rolled to residual
    real, intent(inout) :: rbuf(ni-1, nj-1, np, 2)
    real, intent(inout) :: dtblk(nc1i, nc1j, nc1k)
    real, intent(inout) :: rawbuf(nc1i, nc1j, nc1k, np)
    real, intent(inout) :: sdt(nc1i, nc1j, nc1k)
    real, intent(inout) :: sv (nc1i, nc1j, nc1k)
    real, intent(inout) :: corr_all(n_corr)
    real, intent(inout) :: triw(n_tri)
    real, intent(inout) :: rfac(np)
    real,    intent(in) :: dampin
    external :: smooth_residual_tri_tiled

    call scree_form_q(store, residual, ni, nj, nk, np)
    call mg_restrict_levels(store, dt_vol, vol, cfl, fmgrid, expon_mgrid, sf_irs, &
                       n_levels, dtblk, rawbuf, sdt, sv, corr_all, triw, &
                       smooth_residual_tri_tiled, &
                       ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_tri)
    call mg_collapse_levels(corr_all, n_levels, ni, nj, nk, np, n_corr)
    call mg_fine_scatter(corr_all(n_corr - nc1i*nc1j*nc1k*np + 1), &
                       nc1i, nc1j, nc1k, cons, cons, &
                       cfl, dt_vol, store, ni, nj, nk, np, rbuf, rfac, dampin)
    call scree_roll(residual, store, ni, nj, nk, np)
end subroutine scree_mg_irs


! scree, multigrid on, no smoothing.
subroutine scree_mg_noirs(cons, residual, store, dt_vol, vol, cfl, &
        fmgrid, expon_mgrid, sf_irs, n_levels, rbuf, dtblk, rawbuf, sdt, sv, &
        corr_all, triw, rfac, dampin, &
        ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_tri)
    implicit none
    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k
    integer, intent(in) :: n_corr, n_tri
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real,    intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: cfl, fmgrid, expon_mgrid, sf_irs
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: store(ni-1, nj-1, nk-1, np)   ! in: (dF/dt)_{n-1}; out: rolled to residual
    real, intent(inout) :: rbuf(ni-1, nj-1, np, 2)
    real, intent(inout) :: dtblk(nc1i, nc1j, nc1k)
    real, intent(inout) :: rawbuf(nc1i, nc1j, nc1k, np)
    real, intent(inout) :: sdt(nc1i, nc1j, nc1k)
    real, intent(inout) :: sv (nc1i, nc1j, nc1k)
    real, intent(inout) :: corr_all(n_corr)
    real, intent(inout) :: triw(n_tri)
    real, intent(inout) :: rfac(np)
    real,    intent(in) :: dampin
    external :: mg_smooth_noop

    call scree_form_q(store, residual, ni, nj, nk, np)
    call mg_restrict_levels(store, dt_vol, vol, cfl, fmgrid, expon_mgrid, sf_irs, &
                       n_levels, dtblk, rawbuf, sdt, sv, corr_all, triw, &
                       mg_smooth_noop, &
                       ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_tri)
    call mg_collapse_levels(corr_all, n_levels, ni, nj, nk, np, n_corr)
    call mg_fine_scatter(corr_all(n_corr - nc1i*nc1j*nc1k*np + 1), &
                       nc1i, nc1j, nc1k, cons, cons, &
                       cfl, dt_vol, store, ni, nj, nk, np, rbuf, rfac, dampin)
    call scree_roll(residual, store, ni, nj, nk, np)
end subroutine scree_mg_noirs


! RK stage, multigrid on, coarse-level IRS. q = residual (passed directly).
subroutine rk_mg_irs(cons, snapshot, residual, dt_vol, vol, &
        alpha, cfl, fmgrid, expon_mgrid, sf_irs, n_levels, rbuf, dtblk, &
        rawbuf, sdt, sv, corr_all, triw, rfac, dampin, &
        ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_tri)
    implicit none
    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k
    integer, intent(in) :: n_corr, n_tri
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real,    intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: alpha, cfl, fmgrid, expon_mgrid, sf_irs
    real,    intent(in) :: snapshot(ni, nj, nk, np)
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: rbuf(ni-1, nj-1, np, 2)
    real, intent(inout) :: dtblk(nc1i, nc1j, nc1k)
    real, intent(inout) :: rawbuf(nc1i, nc1j, nc1k, np)
    real, intent(inout) :: sdt(nc1i, nc1j, nc1k)
    real, intent(inout) :: sv (nc1i, nc1j, nc1k)
    real, intent(inout) :: corr_all(n_corr)
    real, intent(inout) :: triw(n_tri)
    real, intent(inout) :: rfac(np)
    real,    intent(in) :: dampin
    external :: smooth_residual_tri_tiled

    call mg_restrict_levels(residual, dt_vol, vol, alpha*cfl, fmgrid, expon_mgrid, &
                       sf_irs, n_levels, dtblk, rawbuf, sdt, sv, corr_all, triw, &
                       smooth_residual_tri_tiled, &
                       ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_tri)
    call mg_collapse_levels(corr_all, n_levels, ni, nj, nk, np, n_corr)
    call mg_fine_scatter(corr_all(n_corr - nc1i*nc1j*nc1k*np + 1), &
                       nc1i, nc1j, nc1k, snapshot, cons, &
                       alpha*cfl, dt_vol, residual, ni, nj, nk, np, rbuf, &
                       rfac, dampin)
end subroutine rk_mg_irs


! RK stage, multigrid on, no smoothing. q = residual (passed directly).
subroutine rk_mg_noirs(cons, snapshot, residual, dt_vol, vol, &
        alpha, cfl, fmgrid, expon_mgrid, sf_irs, n_levels, rbuf, dtblk, &
        rawbuf, sdt, sv, corr_all, triw, rfac, dampin, &
        ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_tri)
    implicit none
    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k
    integer, intent(in) :: n_corr, n_tri
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real,    intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: alpha, cfl, fmgrid, expon_mgrid, sf_irs
    real,    intent(in) :: snapshot(ni, nj, nk, np)
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: rbuf(ni-1, nj-1, np, 2)
    real, intent(inout) :: dtblk(nc1i, nc1j, nc1k)
    real, intent(inout) :: rawbuf(nc1i, nc1j, nc1k, np)
    real, intent(inout) :: sdt(nc1i, nc1j, nc1k)
    real, intent(inout) :: sv (nc1i, nc1j, nc1k)
    real, intent(inout) :: corr_all(n_corr)
    real, intent(inout) :: triw(n_tri)
    real, intent(inout) :: rfac(np)
    real,    intent(in) :: dampin
    external :: mg_smooth_noop

    call mg_restrict_levels(residual, dt_vol, vol, alpha*cfl, fmgrid, expon_mgrid, &
                       sf_irs, n_levels, dtblk, rawbuf, sdt, sv, corr_all, triw, &
                       mg_smooth_noop, &
                       ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_tri)
    call mg_collapse_levels(corr_all, n_levels, ni, nj, nk, np, n_corr)
    call mg_fine_scatter(corr_all(n_corr - nc1i*nc1j*nc1k*np + 1), &
                       nc1i, nc1j, nc1k, snapshot, cons, &
                       alpha*cfl, dt_vol, residual, ni, nj, nk, np, rbuf, &
                       rfac, dampin)
end subroutine rk_mg_noirs
