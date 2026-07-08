! Denton "scree" explicit time-march step, fused into one kernel.
!
! Builds the lagged-residual extrapolation (Denton 2017, Eq 4 with F1=2, F2=-1,
! F3=0), scales it to a per-volume conserved-variable increment, rolls the
! residual history, and scatters the cell-centred increment onto the nodes.
!
! Equivalent to the former NumPy advance():
!   tmp   = (2*residual - store) * cfl * dt_vol   ! extrapolated, scaled change
!   store = residual                              ! roll history for next step
!   cons += cell_to_node(tmp)                     ! distribute onto nodal cons
!
! store carries (dF/dt)_{n-1} in on entry and leaves with (dF/dt)_n; it is the
! caller's persistent per-step buffer (Block.store). tmp is throwaway workspace
! borrowed from Block.scratch -- nothing outside the kernel reads it, so only its
! element count matters, not its indexing.
!
! Written as explicit scalar loops (no array-section expressions) so the build's
! -Warray-temporaries -Werror flags pass with no compiler-generated temporary;
! this is also why tmp is passed in rather than declared as a local automatic.
!
subroutine scree_advance(cons, residual, store, dt_vol, cfl, tmp, ni, nj, nk, np)

    implicit none

    integer, intent(in) :: ni, nj, nk, np
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real,    intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: cfl
    real, intent(inout) :: store(ni-1, nj-1, nk-1, np)  ! in: (dF/dt)_{n-1}; out: (dF/dt)_n
    real, intent(inout) :: cons(ni, nj, nk, np)         ! nodal conserved vars, accumulated in place
    real, intent(inout) :: tmp(ni-1, nj-1, nk-1, np)    ! borrowed scratch workspace

    integer :: i, j, k, ip

    do ip = 1, np
        do k = 1, nk-1
            do j = 1, nj-1
                do i = 1, ni-1
                    ! Read the old history (store) before overwriting it.
                    tmp(i,j,k,ip) = (2e0*residual(i,j,k,ip) - store(i,j,k,ip)) &
                                    * cfl * dt_vol(i,j,k)
                    store(i,j,k,ip) = residual(i,j,k,ip)
                end do
            end do
        end do
    end do

    ! Accumulate the cell-centred increment onto the nodal conserved variables.
    ! cell_to_node aliases its node array as both input and output, so this is an
    ! in-place += (frozen pressure: bypasses the P/T cache, as before).
    call cell_to_node(tmp, cons, ni, nj, nk, np)

end subroutine scree_advance


! Bracketing coarse-cell indices (lo, hi) and upper-neighbour weight for every
! fine cell along one direction. n_fine fine cells, block size b, n_coarse
! coarse cells. Weight clamped to [0,1] with flat extrapolation past the outer
! coarse-cell centres.
!
subroutine mg_prolong_weights(n_fine, b, n_coarse, lo, hi, w)

    implicit none

    integer, intent(in)  :: n_fine, b, n_coarse
    integer, intent(out) :: lo(n_fine), hi(n_fine)
    real,    intent(out) :: w(n_fine)

    integer :: i, icl
    real    :: t

    do i = 1, n_fine
        t   = (real(i) - 0.5e0) / real(b) + 0.5e0
        icl = floor(t)
        if (icl < 1) then
            lo(i) = 1
            hi(i) = 1
            w(i)  = 0e0
        else if (icl >= n_coarse) then
            lo(i) = n_coarse
            hi(i) = n_coarse
            w(i)  = 0e0
        else
            lo(i) = icl
            hi(i) = icl + 1
            w(i)  = t - real(icl)
        end if
    end do

end subroutine mg_prolong_weights


! Optimised variant of advance_rk_stage_mg_fused (identical result, same
! signature) exercising two restructurings of the coarse path:
!
!   Opt #1  restrict fused with the zero + scale passes as a coarse-cell
!           register reduction: iterate coarse cells outer, sum the b*b*b fine
!           block into a scalar, and store corr = sum * coef * dt_vol(centre) in
!           one shot. Removes the zero pass, the per-fine-cell integer division
!           ib=(i-1)/b+1, the strided read-modify-write into corr, and the
!           separate scale pass of the reference kernel.
!
!   Opt #2  prolongation done as three separable 1-D interpolations (trilinear
!           is a tensor product) instead of the 8-way gather. Passes A (interp i)
!           and B (interp j) are fused per (ip, kb): A is a single (ni-1, njb)
!           plane, kept in cache and consumed by B before the next kb, so the
!           lone gather (corr along i) hits an in-cache plane and no full A
!           buffer is streamed. Pass C (interp k) is contiguous in i and folds in
!           the fine term at level 1 (saving a separate tmp pass). Mathematically
!           identical to the direct blend; float rounding order differs (~1e-7).
!
! aplane (ni-1,njb) and B (ni-1,nj-1,nkb,np) are per-level heap scratch here
! (experiment convenience -- a shipping version would pass them in). Everything
! else matches advance_rk_stage_mg_fused: fine term, corr as the caller-owned
! level-1-strided workspace, cell_to_node_generic scatter, frozen pressure.
!
subroutine advance_rk_stage_mg_fused_opt(cons, snapshot, residual, dt_vol, &
        alpha, cfl, fmgrid, n_levels, tmp, corr, aplane, bb, &
        ni, nj, nk, np, nc1i, nc1j, nc1k)

    implicit none

    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real,    intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: alpha, cfl, fmgrid
    real,    intent(in) :: snapshot(ni, nj, nk, np)
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: tmp(ni-1, nj-1, nk-1, np)
    real, intent(inout) :: corr(nc1i, nc1j, nc1k, np)
    ! Caller-owned prolong scratch (level-1-strided, coarser levels use a leading
    ! corner). aplane is one interp-i plane; bb holds the interp-j result.
    real, intent(inout) :: aplane(ni-1, nc1j)
    real, intent(inout) :: bb(ni-1, nj-1, nc1k, np)

    integer :: i, j, k, ip, lvl, b, nib, njb, nkb, ib, jb, kb, mi, mj, mk
    integer :: ii, jj, kk
    real    :: coef, s
    integer :: il(ni-1), ih(ni-1), jl(nj-1), jh(nj-1), kl(nk-1), kh(nk-1)
    real    :: wi(ni-1), wj(nj-1), wk(nk-1)

    ! Fine RK term: dU = alpha*cfl*dt_vol*residual. When n_levels >= 1 this is
    ! folded into level 1's Pass C (opt below) to save a full tmp pass, so the
    ! standalone version runs only for the plain-RK (n_levels == 0) case.
    if (n_levels == 0) then
        do ip = 1, np
            do k = 1, nk-1
                do j = 1, nj-1
                    do i = 1, ni-1
                        tmp(i,j,k,ip) = alpha * cfl * dt_vol(i,j,k) * residual(i,j,k,ip)
                    end do
                end do
            end do
        end do
    end if

    do lvl = 1, n_levels
        b = 2**lvl
        nib = (ni-1)/b
        njb = (nj-1)/b
        nkb = (nk-1)/b
        coef = alpha * cfl * fmgrid / real(b*b) * 2e0**(-(lvl-1))

        ! Opt #1: fused zero + restrict + scale as a coarse-cell reduction.
        do ip = 1, np
            do kb = 1, nkb
                mk = (kb-1)*b + b/2
                do jb = 1, njb
                    mj = (jb-1)*b + b/2
                    do ib = 1, nib
                        mi = (ib-1)*b + b/2
                        s = 0e0
                        do kk = (kb-1)*b+1, kb*b
                            do jj = (jb-1)*b+1, jb*b
                                do ii = (ib-1)*b+1, ib*b
                                    s = s + residual(ii,jj,kk,ip)
                                end do
                            end do
                        end do
                        corr(ib,jb,kb,ip) = s * coef * dt_vol(mi,mj,mk)
                    end do
                end do
            end do
        end do

        ! Bracketing coarse indices and upper-neighbour weights per direction.
        call mg_prolong_weights(ni-1, b, nib, il, ih, wi)
        call mg_prolong_weights(nj-1, b, njb, jl, jh, wj)
        call mg_prolong_weights(nk-1, b, nkb, kl, kh, wk)

        ! Opt #2: separable trilinear prolong onto tmp. Passes A (interp i) and
        ! B (interp j) are fused per (ip, kb): A is a single (ni-1, njb) plane
        ! (aplane), kept in cache and consumed by B before moving to the next kb
        ! -- so the only gather (corr along i) hits an in-cache plane and no full
        ! A buffer is materialised. bb is kept whole (Pass C spans kb via kl/kh).
        ! aplane/bb are caller-owned scratch declared at level-1 strides; coarser
        ! levels write only the leading corner (njb<=nc1j, nkb<=nc1k).
        do ip = 1, np
            do kb = 1, nkb
                ! Pass A (this kb, ip): interpolate corr along i into aplane.
                do jb = 1, njb
                    do i = 1, ni-1
                        aplane(i,jb) = corr(il(i),jb,kb,ip)*(1e0-wi(i)) &
                                     + corr(ih(i),jb,kb,ip)*wi(i)
                    end do
                end do
                ! Pass B (this kb, ip): interpolate aplane along j into bb.
                do j = 1, nj-1
                    do i = 1, ni-1
                        bb(i,j,kb,ip) = aplane(i,jl(j))*(1e0-wj(j)) &
                                      + aplane(i,jh(j))*wj(j)
                    end do
                end do
            end do
        end do

        ! Pass C: interpolate B along k onto tmp. Fixed k: contiguous i. Level 1
        ! folds in the fine term (opt #2); later levels accumulate.
        if (lvl == 1) then
            do ip = 1, np
                do k = 1, nk-1
                    do j = 1, nj-1
                        do i = 1, ni-1
                            tmp(i,j,k,ip) = &
                                  alpha * cfl * dt_vol(i,j,k) * residual(i,j,k,ip) &
                                + bb(i,j,kl(k),ip)*(1e0-wk(k)) &
                                + bb(i,j,kh(k),ip)*wk(k)
                        end do
                    end do
                end do
            end do
        else
            do ip = 1, np
                do k = 1, nk-1
                    do j = 1, nj-1
                        do i = 1, ni-1
                            tmp(i,j,k,ip) = tmp(i,j,k,ip) &
                                          + bb(i,j,kl(k),ip)*(1e0-wk(k)) &
                                          + bb(i,j,kh(k),ip)*wk(k)
                        end do
                    end do
                end do
            end do
        end if
    end do

    ! cons = snapshot + cell_to_node(tmp). Distinct in/out (snapshot vs cons).
    call cell_to_node_generic(tmp, snapshot, cons, ni, nj, nk, np)

end subroutine advance_rk_stage_mg_fused_opt


! Experimental variant of advance_rk_stage_mg_fused_opt that applies implicit
! residual smoothing (Jameson IRS, cf. smooth_residual_tri in residual.f90) to
! the coarse block-restricted residual at every level, before it is scaled by
! coef*dt_vol and prolongated. sf_irs=0 makes the smoothing step an exact
! no-op (smooth_residual_tri's own guard), so this kernel is then byte-for-byte
! identical to advance_rk_stage_mg_fused_opt -- the basis of the correctness
! check against the production kernel.
!
! The production kernel has no clean "coarse residual" moment to smooth: its
! restrict step fuses the block-sum directly with the coef*dt_vol scale, and
! its coarse workspace (corr) is declared at level-1 size and reused only via
! a strided leading corner for coarser levels -- not contiguous, so it cannot
! be handed to smooth_residual_tri as-is. Here each level instead restricts the
! raw block-sum residual into its own slice of the caller-owned flat buffer
! coarse_res_buf (levels packed back-to-back, largest first, so no per-level
! stride reuse), smooths that slice in place via smooth_residual_tri (passed
! by sequence association -- the same manual-offset idiom smooth_residual_tri
! itself uses internally for its Thomas coefficients, residual.f90:758-768),
! then scales the smoothed values into corr exactly as _opt did with the raw
! sum. tri_work_buf is the equivalent caller-owned packed scratch for
! smooth_residual_tri's per-level Thomas coefficients. Both buffers are sized
! once by the caller (see mg_irs_scratch_sizes on the Python side) and reused
! every call -- no allocation here.
subroutine advance_rk_stage_mg_fused_irs(cons, snapshot, residual, dt_vol, &
        alpha, cfl, fmgrid, sf_irs, n_levels, tmp, corr, aplane, bb, &
        coarse_res_buf, tri_work_buf, &
        ni, nj, nk, np, nc1i, nc1j, nc1k, n_res_max, n_tri_max)

    implicit none

    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k
    integer, intent(in) :: n_res_max, n_tri_max
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real,    intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: alpha, cfl, fmgrid, sf_irs
    real,    intent(in) :: snapshot(ni, nj, nk, np)
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: tmp(ni-1, nj-1, nk-1, np)
    real, intent(inout) :: corr(nc1i, nc1j, nc1k, np)
    real, intent(inout) :: aplane(ni-1, nc1j)
    real, intent(inout) :: bb(ni-1, nj-1, nc1k, np)
    real, intent(inout) :: coarse_res_buf(n_res_max)
    real, intent(inout) :: tri_work_buf(n_tri_max)

    integer :: i, j, k, ip, lvl, b, nib, njb, nkb, ib, jb, kb, mi, mj, mk
    integer :: ii, jj, kk, lin
    integer :: off_res, off_tri, cnt_res, cnt_tri
    real    :: coef, s
    integer :: il(ni-1), ih(ni-1), jl(nj-1), jh(nj-1), kl(nk-1), kh(nk-1)
    real    :: wi(ni-1), wj(nj-1), wk(nk-1)

    ! Fine RK term (see advance_rk_stage_mg_fused_opt for the n_levels==1 fold-in).
    if (n_levels == 0) then
        do ip = 1, np
            do k = 1, nk-1
                do j = 1, nj-1
                    do i = 1, ni-1
                        tmp(i,j,k,ip) = alpha * cfl * dt_vol(i,j,k) * residual(i,j,k,ip)
                    end do
                end do
            end do
        end do
    end if

    off_res = 0
    off_tri = 0
    do lvl = 1, n_levels
        b = 2**lvl
        nib = (ni-1)/b
        njb = (nj-1)/b
        nkb = (nk-1)/b
        coef = alpha * cfl * fmgrid / real(b*b) * 2e0**(-(lvl-1))
        cnt_res = nib*njb*nkb*np
        cnt_tri = 2*(nib+njb+nkb)

        ! Restrict: raw block-sum residual (no scale yet), written into this
        ! level's slice of coarse_res_buf via manual column-major flat indexing.
        do ip = 1, np
            do kb = 1, nkb
                do jb = 1, njb
                    do ib = 1, nib
                        s = 0e0
                        do kk = (kb-1)*b+1, kb*b
                            do jj = (jb-1)*b+1, jb*b
                                do ii = (ib-1)*b+1, ib*b
                                    s = s + residual(ii,jj,kk,ip)
                                end do
                            end do
                        end do
                        lin = off_res + ib + (jb-1)*nib + (kb-1)*nib*njb + (ip-1)*nib*njb*nkb
                        coarse_res_buf(lin) = s
                    end do
                end do
            end do
        end do

        ! Implicit residual smoothing on the raw coarse residual, in place on
        ! this level's slice (sequence association: a contiguous 1-D section
        ! passed to smooth_residual_tri's explicit-shape (nib,njb,nkb,np) dummy).
        call smooth_residual_tri(coarse_res_buf(off_res+1:off_res+cnt_res), sf_irs, &
                                  tri_work_buf(off_tri+1:off_tri+cnt_tri), &
                                  nib+1, njb+1, nkb+1)

        ! Scale the (possibly smoothed) coarse residual into corr.
        do ip = 1, np
            do kb = 1, nkb
                mk = (kb-1)*b + b/2
                do jb = 1, njb
                    mj = (jb-1)*b + b/2
                    do ib = 1, nib
                        mi = (ib-1)*b + b/2
                        lin = off_res + ib + (jb-1)*nib + (kb-1)*nib*njb + (ip-1)*nib*njb*nkb
                        corr(ib,jb,kb,ip) = coarse_res_buf(lin) * coef * dt_vol(mi,mj,mk)
                    end do
                end do
            end do
        end do

        off_res = off_res + cnt_res
        off_tri = off_tri + cnt_tri

        ! Bracketing coarse indices and upper-neighbour weights per direction.
        call mg_prolong_weights(ni-1, b, nib, il, ih, wi)
        call mg_prolong_weights(nj-1, b, njb, jl, jh, wj)
        call mg_prolong_weights(nk-1, b, nkb, kl, kh, wk)

        ! Separable trilinear prolong onto tmp (identical to _opt).
        do ip = 1, np
            do kb = 1, nkb
                do jb = 1, njb
                    do i = 1, ni-1
                        aplane(i,jb) = corr(il(i),jb,kb,ip)*(1e0-wi(i)) &
                                     + corr(ih(i),jb,kb,ip)*wi(i)
                    end do
                end do
                do j = 1, nj-1
                    do i = 1, ni-1
                        bb(i,j,kb,ip) = aplane(i,jl(j))*(1e0-wj(j)) &
                                      + aplane(i,jh(j))*wj(j)
                    end do
                end do
            end do
        end do

        if (lvl == 1) then
            do ip = 1, np
                do k = 1, nk-1
                    do j = 1, nj-1
                        do i = 1, ni-1
                            tmp(i,j,k,ip) = &
                                  alpha * cfl * dt_vol(i,j,k) * residual(i,j,k,ip) &
                                + bb(i,j,kl(k),ip)*(1e0-wk(k)) &
                                + bb(i,j,kh(k),ip)*wk(k)
                        end do
                    end do
                end do
            end do
        else
            do ip = 1, np
                do k = 1, nk-1
                    do j = 1, nj-1
                        do i = 1, ni-1
                            tmp(i,j,k,ip) = tmp(i,j,k,ip) &
                                          + bb(i,j,kl(k),ip)*(1e0-wk(k)) &
                                          + bb(i,j,kh(k),ip)*wk(k)
                        end do
                    end do
                end do
            end do
        end if
    end do

    ! cons = snapshot + cell_to_node(tmp). Distinct in/out (snapshot vs cons).
    call cell_to_node_generic(tmp, snapshot, cons, ni, nj, nk, np)

end subroutine advance_rk_stage_mg_fused_irs
