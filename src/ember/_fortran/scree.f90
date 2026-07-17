! Explicit time-march step kernels (Denton "scree" and Jameson RK), with
! optional block-sum multigrid and coarse-level implicit residual smoothing.
!
! The whole scheme x multigrid x IRS space is served by six branch-free wrappers
! over one shared, scheme-agnostic engine (mg_coarse_correction) and a set of
! branch-free building blocks. No configuration is decided by a runtime `if`
! inside a kernel: the scheme is fixed by which fine quantity `q` the wrapper
! forms and which scatter tail it calls; multigrid on/off is which wrapper the
! caller picks (mg-off wrappers never touch the coarse engine); IRS on/off is the
! `smoother` dummy-procedure argument (smooth_residual_tri_tiled vs mg_smooth_noop).
! See the banner above mg_coarse_correction for the algorithm and the wrapper
! grid.
!
! All loops are explicit scalar loops (no array-section expressions) so the
! build's -Warray-temporaries -Werror flags pass with no compiler-generated
! temporary; this is also why scratch (tmp, coarse buffers) is passed in rather
! than declared as a local automatic. cell_to_node reuses its node array as both
! input and output (frozen pressure: the increment bypasses the P/T cache), so
! the scree scatter is an in-place +=.


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


! Copy n contiguous reals (sequence-associated cascade plumbing).
subroutine mg_copy(src, dst, n)
    implicit none
    integer, intent(in) :: n
    real, intent(in)    :: src(n)
    real, intent(inout) :: dst(n)
    integer :: i
    do i = 1, n
        dst(i) = src(i)
    end do
end subroutine mg_copy


! No-op coarse-residual smoother: the plain (non-IRS) kernels pass this to
! mg_coarse_correction so the smoothing step is structurally absent (no sf_irs<=0
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


! Scale a (possibly smoothed) contiguous coarse residual into a correction:
! corr = cres * coef * dtblk. dtblk is level-1-strided (leading corner read).
subroutine mg_scale_corr(corr, cres, dtblk, coef, ldi, ldj, ldk, nib, njb, nkb, np)
    implicit none
    integer, intent(in) :: ldi, ldj, ldk, nib, njb, nkb, np
    real, intent(in)    :: cres(nib, njb, nkb, np)
    real, intent(in)    :: dtblk(ldi, ldj, ldk)
    real, intent(in)    :: coef
    real, intent(out)   :: corr(nib, njb, nkb, np)
    integer :: ib, jb, kb, ip
    do ip = 1, np
        do kb = 1, nkb
            do jb = 1, njb
                do ib = 1, nib
                    corr(ib,jb,kb,ip) = cres(ib,jb,kb,ip) * coef * dtblk(ib,jb,kb)
                end do
            end do
        end do
    end do
end subroutine mg_scale_corr


! Separable factor-2 trilinear prolongation, ACCUMULATED onto a coarse target:
! out += interp_2x(src). Used for the cascade's coarse->coarse hops (out is
! prefilled with that level's correction). aplane/bb are the shared prolong
! scratch (full-fine leading dims ni1,nj1; third dim nkpad >= nck).
subroutine mg_prolong2x_acc(src, nci, ncj, nck, out, nfi, nfj, nfk, np, &
        aplane, bb, ni1, nj1, nkpad)
    implicit none
    integer, intent(in) :: nci, ncj, nck, nfi, nfj, nfk, np, ni1, nj1, nkpad
    real, intent(in)    :: src(nci, ncj, nck, np)
    real, intent(inout) :: out(nfi, nfj, nfk, np)
    real, intent(inout) :: aplane(ni1, *)
    real, intent(inout) :: bb(ni1, nj1, nkpad, np)
    integer :: i, j, k, ip, jc, kc
    integer :: il(nfi), ih(nfi), jl(nfj), jh(nfj), kl(nfk), kh(nfk)
    real    :: wi(nfi), wj(nfj), wk(nfk)

    call mg_prolong_weights(nfi, 2, nci, il, ih, wi)
    call mg_prolong_weights(nfj, 2, ncj, jl, jh, wj)
    call mg_prolong_weights(nfk, 2, nck, kl, kh, wk)

    do ip = 1, np
        do kc = 1, nck
            do jc = 1, ncj
                do i = 1, nfi
                    aplane(i,jc) = src(il(i),jc,kc,ip)*(1e0-wi(i)) &
                                 + src(ih(i),jc,kc,ip)*wi(i)
                end do
            end do
            do j = 1, nfj
                do i = 1, nfi
                    bb(i,j,kc,ip) = aplane(i,jl(j))*(1e0-wj(j)) &
                                  + aplane(i,jh(j))*wj(j)
                end do
            end do
        end do
        do k = 1, nfk
            do j = 1, nfj
                do i = 1, nfi
                    out(i,j,k,ip) = out(i,j,k,ip) &
                                  + bb(i,j,kl(k),ip)*(1e0-wk(k)) &
                                  + bb(i,j,kh(k),ip)*wk(k)
                end do
            end do
        end do
    end do
end subroutine mg_prolong2x_acc


! Final cascade hop onto the fine grid, fused with the fine term in one write:
!   tmp = scale*dt_vol*q + interp_2x(src)
! q is the scheme's fine quantity (residual for RK, 2*residual-store for scree),
! formed by the wrapper -- this block is scheme-agnostic (no denton branch).
subroutine mg_prolong2x_fine(src, nci, ncj, nck, tmp, scale, dt_vol, q, &
        ni, nj, nk, np, aplane, bb, nc1j, nc1k)
    implicit none
    integer, intent(in) :: nci, ncj, nck, ni, nj, nk, np, nc1j, nc1k
    real, intent(in)    :: src(nci, ncj, nck, np)
    real, intent(in)    :: scale
    real, intent(in)    :: dt_vol(ni-1, nj-1, nk-1)
    real, intent(in)    :: q(ni-1, nj-1, nk-1, np)
    real, intent(out)   :: tmp(ni-1, nj-1, nk-1, np)
    real, intent(inout) :: aplane(ni-1, nc1j)
    real, intent(inout) :: bb(ni-1, nj-1, nc1k, np)
    integer :: i, j, k, ip, jc, kc
    integer :: il(ni-1), ih(ni-1), jl(nj-1), jh(nj-1), kl(nk-1), kh(nk-1)
    real    :: wi(ni-1), wj(nj-1), wk(nk-1)
    real    :: ft

    call mg_prolong_weights(ni-1, 2, nci, il, ih, wi)
    call mg_prolong_weights(nj-1, 2, ncj, jl, jh, wj)
    call mg_prolong_weights(nk-1, 2, nck, kl, kh, wk)

    do ip = 1, np
        do kc = 1, nck
            do jc = 1, ncj
                do i = 1, ni-1
                    aplane(i,jc) = src(il(i),jc,kc,ip)*(1e0-wi(i)) &
                                 + src(ih(i),jc,kc,ip)*wi(i)
                end do
            end do
            do j = 1, nj-1
                do i = 1, ni-1
                    bb(i,j,kc,ip) = aplane(i,jl(j))*(1e0-wj(j)) &
                                  + aplane(i,jh(j))*wj(j)
                end do
            end do
        end do
    end do

    do ip = 1, np
        do k = 1, nk-1
            do j = 1, nj-1
                do i = 1, ni-1
                    ft = scale * dt_vol(i,j,k) * q(i,j,k,ip)
                    tmp(i,j,k,ip) = ft + bb(i,j,kl(k),ip)*(1e0-wk(k)) &
                                       + bb(i,j,kh(k),ip)*wk(k)
                end do
            end do
        end do
    end do
end subroutine mg_prolong2x_fine


! Scheme-agnostic fine term (the multigrid-off increment):  tmp = scale*dt_vol*q.
! Grouping (scale*dt_vol)*q matches the fused fine term in mg_prolong2x_fine, so
! an mg-off march is byte-identical to an mg-on march whose coarse correction is
! exactly zero (fmgrid == 0).
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


! Roll the Denton history (store = residual) and frozen-pressure accumulate the
! increment onto cons. Shared post-march tail of the scree wrappers; called only
! after the engine/fine-term has consumed q from store.
subroutine scree_roll_and_scatter(cons, residual, store, tmp, ni, nj, nk, np)
    implicit none
    integer, intent(in) :: ni, nj, nk, np
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: store(ni-1, nj-1, nk-1, np)
    real, intent(inout) :: tmp(ni-1, nj-1, nk-1, np)
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
    call cell_to_node(tmp, cons, ni, nj, nk, np)
end subroutine scree_roll_and_scatter


! ============================================================================
! Hierarchical-restriction + cascaded-prolongation block-sum multigrid engine.
!
! Scheme-agnostic: operates on a single pre-formed fine quantity `q` (residual
! for RK, 2*residual-store for scree), so it carries no `denton` branch. Called
! only by the mg-*on* wrappers, so n_levels >= 1 always (no n_levels==0 path).
! IRS is the `smoother` dummy-procedure argument: smooth_residual_tri_tiled
! (Jameson IRS) or mg_smooth_noop (none) -- no `if (sf_irs)` anywhere in here.
!
! The six production kernels, all branch-free straight-line compositions:
!   scree_plain     rk_plain      fine_term + scatter          (multigrid off)
!   scree_mg_noirs  rk_mg_noirs   engine(mg_smooth_noop) + scatter
!   scree_mg_irs    rk_mg_irs     engine(smooth_residual_tri_tiled) + scatter
! scree wrappers form q in store (scree_form_q) and roll+frozen-scatter
! (scree_roll_and_scatter); rk wrappers pass residual as q and scatter off the
! sub-stage snapshot (cell_to_node_generic).
!
! Restriction is HIERARCHICAL: a level-l block-sum equals eight level-(l-1)
! block-sums (the block-sum is associative), so only level 1 reads the fine grid;
! coarser levels reduce the small running accumulators -- rawbuf for the residual
! sum, sdt/sv for the volume-weighted dt. Level 1 is loop-peeled (it alone reads
! the fine grid), so the level loop has no `if (lvl==1)`. In-place reduction is
! safe because, with blocks visited in ascending order, an output cell maps to
! source cells whose every index is >= its own and no later block reads it.
!
! Prolongation is CASCADED: the per-level scaled corrections (packed coarsest
! first in corr_all) accumulate coarsest -> finest through factor-2 hops, so only
! the final hop writes the fine grid (fused with the fine term). This is a
! genuine operator change from a direct factor-b prolong -- cascaded factor-2
! trilinear interpolations are not equal to it.
! ============================================================================
subroutine mg_coarse_correction(q, dt_vol, vol, scale, fmgrid, &
        sf_irs, n_levels, tmp, dtblk, aplane, bb, rawbuf, sdt, sv, &
        corr_all, acc0, acc1, cres, triw, smoother, &
        ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)

    implicit none
    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k
    integer, intent(in) :: n_corr, n_res, n_tri
    ! Coarse-residual smoother, chosen by the caller (no IRS branch in here):
    ! smooth_residual_tri_tiled for the IRS kernels, mg_smooth_noop for the plain ones.
    external :: smoother
    real, intent(in)    :: q(ni-1, nj-1, nk-1, np)
    real, intent(in)    :: dt_vol(ni-1, nj-1, nk-1)
    real, intent(in)    :: vol(ni-1, nj-1, nk-1)
    real, intent(in)    :: scale, fmgrid, sf_irs
    real, intent(inout) :: tmp(ni-1, nj-1, nk-1, np)
    real, intent(inout) :: dtblk(nc1i, nc1j, nc1k)
    real, intent(inout) :: aplane(ni-1, nc1j)
    real, intent(inout) :: bb(ni-1, nj-1, nc1k, np)
    real, intent(inout) :: rawbuf(nc1i, nc1j, nc1k, np)
    real, intent(inout) :: sdt(nc1i, nc1j, nc1k)
    real, intent(inout) :: sv (nc1i, nc1j, nc1k)
    real, intent(inout) :: corr_all(n_corr)
    real, intent(inout) :: acc0(nc1i*nc1j*nc1k*np)
    real, intent(inout) :: acc1(nc1i*nc1j*nc1k*np)
    real, intent(inout) :: cres(n_res)
    real, intent(inout) :: triw(n_tri)

    integer :: ip, lvl, b, nib, njb, nkb, ib, jb, kb
    integer :: ii, jj, kk, slot, cnt
    real    :: coef, s, s_dt, s_v
    integer :: dib(n_levels), djb(n_levels), dkb(n_levels), offc(n_levels)
    integer :: nci, ncj, nck, cur_i, cur_j, cur_k, o

    ! Coarsest-first packed geometry for corr_all (cascade seeds at slot 1).
    o = 0
    do lvl = 1, n_levels
        b = 2**(n_levels - lvl + 1)
        dib(lvl) = (ni-1)/b
        djb(lvl) = (nj-1)/b
        dkb(lvl) = (nk-1)/b
        offc(lvl) = o
        o = o + dib(lvl)*djb(lvl)*dkb(lvl)*np
    end do

    ! ---- Phase 1, level 1 (peeled): the only level that reads the fine grid ----
    lvl = 1
    b   = 2
    nib = (ni-1)/b
    njb = (nj-1)/b
    nkb = (nk-1)/b
    coef = scale * fmgrid / real(b*b) * 2e0**(-(lvl-1))
    slot = n_levels - lvl + 1
    cnt  = nib*njb*nkb*np

    ! dt restriction (volume-weighted mean) from the fine grid.
    do kb = 1, nkb
        do jb = 1, njb
            do ib = 1, nib
                s_dt = 0e0
                s_v  = 0e0
                do kk = 2*kb-1, 2*kb
                    do jj = 2*jb-1, 2*jb
                        do ii = 2*ib-1, 2*ib
                            s_dt = s_dt + dt_vol(ii,jj,kk)*vol(ii,jj,kk)
                            s_v  = s_v  + vol(ii,jj,kk)
                        end do
                    end do
                end do
                sdt(ib,jb,kb)   = s_dt
                sv (ib,jb,kb)   = s_v
                dtblk(ib,jb,kb) = s_dt / s_v
            end do
        end do
    end do

    ! residual (fine quantity q) restriction from the fine grid into rawbuf.
    do ip = 1, np
        do kb = 1, nkb
            do jb = 1, njb
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
    call mg_gather_corner(cres, rawbuf, nc1i, nc1j, nc1k, nib, njb, nkb, np)
    call smoother(cres(1:cnt), sf_irs, &
                  triw(1:2*(nib+njb+nkb)), nib+1, njb+1, nkb+1)
    call mg_scale_corr(corr_all(offc(slot)+1), cres, dtblk, coef, &
                       nc1i, nc1j, nc1k, nib, njb, nkb, np)

    ! ---- Phase 1, levels 2..n_levels: reduce the coarse accumulators ----
    do lvl = 2, n_levels
        b = 2**lvl
        nib = (ni-1)/b
        njb = (nj-1)/b
        nkb = (nk-1)/b
        coef = scale * fmgrid / real(b*b) * 2e0**(-(lvl-1))
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
                    dtblk(ib,jb,kb) = s_dt / s_v
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

        call mg_gather_corner(cres, rawbuf, nc1i, nc1j, nc1k, nib, njb, nkb, np)
        call smoother(cres(1:cnt), sf_irs, &
                      triw(1:2*(nib+njb+nkb)), nib+1, njb+1, nkb+1)
        call mg_scale_corr(corr_all(offc(slot)+1), cres, dtblk, coef, &
                           nc1i, nc1j, nc1k, nib, njb, nkb, np)
    end do

    ! Phase 2: cascaded coarsest->fine prolongation.
    call mg_copy(corr_all(offc(1)+1), acc0, dib(1)*djb(1)*dkb(1)*np)
    cur_i = dib(1)
    cur_j = djb(1)
    cur_k = dkb(1)
    do lvl = 2, n_levels
        nci = cur_i
        ncj = cur_j
        nck = cur_k
        call mg_copy(corr_all(offc(lvl)+1), acc1, dib(lvl)*djb(lvl)*dkb(lvl)*np)
        call mg_prolong2x_acc(acc0, nci, ncj, nck, acc1, &
                              dib(lvl), djb(lvl), dkb(lvl), np, &
                              aplane, bb, ni-1, nj-1, nc1k)
        call mg_copy(acc1, acc0, dib(lvl)*djb(lvl)*dkb(lvl)*np)
        cur_i = dib(lvl)
        cur_j = djb(lvl)
        cur_k = dkb(lvl)
    end do
    call mg_prolong2x_fine(acc0, cur_i, cur_j, cur_k, tmp, scale, dt_vol, &
                           q, ni, nj, nk, np, aplane, bb, nc1j, nc1k)
end subroutine mg_coarse_correction


! ============================================================================
! The six production kernels. Each is a branch-free straight-line composition of
! the blocks above; configuration is resolved by which blocks are called and
! which smoother is passed, never by a runtime `if`.
! ============================================================================


! scree, multigrid off: form q, fine term only, roll history and frozen-scatter.
subroutine scree_plain(cons, residual, store, dt_vol, cfl, tmp, ni, nj, nk, np)
    implicit none
    integer, intent(in) :: ni, nj, nk, np
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real,    intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: cfl
    real, intent(inout) :: store(ni-1, nj-1, nk-1, np)  ! in: (dF/dt)_{n-1}; out: (dF/dt)_n
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: tmp(ni-1, nj-1, nk-1, np)

    call scree_form_q(store, residual, ni, nj, nk, np)
    call fine_term(store, dt_vol, cfl, tmp, ni, nj, nk, np)
    call scree_roll_and_scatter(cons, residual, store, tmp, ni, nj, nk, np)
end subroutine scree_plain


! scree, multigrid on, coarse-level IRS.
subroutine scree_mg_irs(cons, residual, store, dt_vol, vol, cfl, &
        fmgrid, sf_irs, n_levels, tmp, dtblk, aplane, bb, rawbuf, sdt, sv, &
        corr_all, acc0, acc1, cres, triw, &
        ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    implicit none
    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k
    integer, intent(in) :: n_corr, n_res, n_tri
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real,    intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: cfl, fmgrid, sf_irs
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: store(ni-1, nj-1, nk-1, np)   ! in: (dF/dt)_{n-1}; out: rolled to residual
    real, intent(inout) :: tmp(ni-1, nj-1, nk-1, np)
    real, intent(inout) :: dtblk(nc1i, nc1j, nc1k)
    real, intent(inout) :: aplane(ni-1, nc1j)
    real, intent(inout) :: bb(ni-1, nj-1, nc1k, np)
    real, intent(inout) :: rawbuf(nc1i, nc1j, nc1k, np)
    real, intent(inout) :: sdt(nc1i, nc1j, nc1k)
    real, intent(inout) :: sv (nc1i, nc1j, nc1k)
    real, intent(inout) :: corr_all(n_corr)
    real, intent(inout) :: acc0(nc1i*nc1j*nc1k*np)
    real, intent(inout) :: acc1(nc1i*nc1j*nc1k*np)
    real, intent(inout) :: cres(n_res)
    real, intent(inout) :: triw(n_tri)
    external :: smooth_residual_tri_tiled

    call scree_form_q(store, residual, ni, nj, nk, np)
    call mg_coarse_correction(store, dt_vol, vol, cfl, fmgrid, sf_irs, n_levels, &
                       tmp, dtblk, aplane, bb, rawbuf, sdt, sv, &
                       corr_all, acc0, acc1, cres, triw, smooth_residual_tri_tiled, &
                       ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    call scree_roll_and_scatter(cons, residual, store, tmp, ni, nj, nk, np)
end subroutine scree_mg_irs


! scree, multigrid on, no smoothing.
subroutine scree_mg_noirs(cons, residual, store, dt_vol, vol, cfl, &
        fmgrid, sf_irs, n_levels, tmp, dtblk, aplane, bb, rawbuf, sdt, sv, &
        corr_all, acc0, acc1, cres, triw, &
        ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    implicit none
    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k
    integer, intent(in) :: n_corr, n_res, n_tri
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real,    intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: cfl, fmgrid, sf_irs
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: store(ni-1, nj-1, nk-1, np)   ! in: (dF/dt)_{n-1}; out: rolled to residual
    real, intent(inout) :: tmp(ni-1, nj-1, nk-1, np)
    real, intent(inout) :: dtblk(nc1i, nc1j, nc1k)
    real, intent(inout) :: aplane(ni-1, nc1j)
    real, intent(inout) :: bb(ni-1, nj-1, nc1k, np)
    real, intent(inout) :: rawbuf(nc1i, nc1j, nc1k, np)
    real, intent(inout) :: sdt(nc1i, nc1j, nc1k)
    real, intent(inout) :: sv (nc1i, nc1j, nc1k)
    real, intent(inout) :: corr_all(n_corr)
    real, intent(inout) :: acc0(nc1i*nc1j*nc1k*np)
    real, intent(inout) :: acc1(nc1i*nc1j*nc1k*np)
    real, intent(inout) :: cres(n_res)
    real, intent(inout) :: triw(n_tri)
    external :: mg_smooth_noop

    call scree_form_q(store, residual, ni, nj, nk, np)
    call mg_coarse_correction(store, dt_vol, vol, cfl, fmgrid, sf_irs, n_levels, &
                       tmp, dtblk, aplane, bb, rawbuf, sdt, sv, &
                       corr_all, acc0, acc1, cres, triw, mg_smooth_noop, &
                       ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    call scree_roll_and_scatter(cons, residual, store, tmp, ni, nj, nk, np)
end subroutine scree_mg_noirs


! RK stage, multigrid off: fine term only (q = residual), scatter off snapshot.
subroutine rk_plain(cons, snapshot, residual, dt_vol, alpha, cfl, tmp, &
        ni, nj, nk, np)
    implicit none
    integer, intent(in) :: ni, nj, nk, np
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real,    intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: alpha, cfl
    real,    intent(in) :: snapshot(ni, nj, nk, np)
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: tmp(ni-1, nj-1, nk-1, np)

    call fine_term(residual, dt_vol, alpha*cfl, tmp, ni, nj, nk, np)
    ! cons = snapshot + cell_to_node(tmp). Distinct in/out (snapshot vs cons).
    call cell_to_node_generic(tmp, snapshot, cons, ni, nj, nk, np)
end subroutine rk_plain


! RK stage, multigrid on, coarse-level IRS. q = residual (passed directly).
subroutine rk_mg_irs(cons, snapshot, residual, dt_vol, vol, &
        alpha, cfl, fmgrid, sf_irs, n_levels, tmp, dtblk, aplane, bb, &
        rawbuf, sdt, sv, corr_all, acc0, acc1, cres, triw, &
        ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    implicit none
    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k
    integer, intent(in) :: n_corr, n_res, n_tri
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real,    intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: alpha, cfl, fmgrid, sf_irs
    real,    intent(in) :: snapshot(ni, nj, nk, np)
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: tmp(ni-1, nj-1, nk-1, np)
    real, intent(inout) :: dtblk(nc1i, nc1j, nc1k)
    real, intent(inout) :: aplane(ni-1, nc1j)
    real, intent(inout) :: bb(ni-1, nj-1, nc1k, np)
    real, intent(inout) :: rawbuf(nc1i, nc1j, nc1k, np)
    real, intent(inout) :: sdt(nc1i, nc1j, nc1k)
    real, intent(inout) :: sv (nc1i, nc1j, nc1k)
    real, intent(inout) :: corr_all(n_corr)
    real, intent(inout) :: acc0(nc1i*nc1j*nc1k*np)
    real, intent(inout) :: acc1(nc1i*nc1j*nc1k*np)
    real, intent(inout) :: cres(n_res)
    real, intent(inout) :: triw(n_tri)
    external :: smooth_residual_tri_tiled

    call mg_coarse_correction(residual, dt_vol, vol, alpha*cfl, fmgrid, sf_irs, &
                       n_levels, tmp, dtblk, aplane, bb, rawbuf, sdt, sv, &
                       corr_all, acc0, acc1, cres, triw, smooth_residual_tri_tiled, &
                       ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    call cell_to_node_generic(tmp, snapshot, cons, ni, nj, nk, np)
end subroutine rk_mg_irs


! RK stage, multigrid on, no smoothing. q = residual (passed directly).
subroutine rk_mg_noirs(cons, snapshot, residual, dt_vol, vol, &
        alpha, cfl, fmgrid, sf_irs, n_levels, tmp, dtblk, aplane, bb, &
        rawbuf, sdt, sv, corr_all, acc0, acc1, cres, triw, &
        ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    implicit none
    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k
    integer, intent(in) :: n_corr, n_res, n_tri
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real,    intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: alpha, cfl, fmgrid, sf_irs
    real,    intent(in) :: snapshot(ni, nj, nk, np)
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: tmp(ni-1, nj-1, nk-1, np)
    real, intent(inout) :: dtblk(nc1i, nc1j, nc1k)
    real, intent(inout) :: aplane(ni-1, nc1j)
    real, intent(inout) :: bb(ni-1, nj-1, nc1k, np)
    real, intent(inout) :: rawbuf(nc1i, nc1j, nc1k, np)
    real, intent(inout) :: sdt(nc1i, nc1j, nc1k)
    real, intent(inout) :: sv (nc1i, nc1j, nc1k)
    real, intent(inout) :: corr_all(n_corr)
    real, intent(inout) :: acc0(nc1i*nc1j*nc1k*np)
    real, intent(inout) :: acc1(nc1i*nc1j*nc1k*np)
    real, intent(inout) :: cres(n_res)
    real, intent(inout) :: triw(n_tri)
    external :: mg_smooth_noop

    call mg_coarse_correction(residual, dt_vol, vol, alpha*cfl, fmgrid, sf_irs, &
                       n_levels, tmp, dtblk, aplane, bb, rawbuf, sdt, sv, &
                       corr_all, acc0, acc1, cres, triw, mg_smooth_noop, &
                       ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    call cell_to_node_generic(tmp, snapshot, cons, ni, nj, nk, np)
end subroutine rk_mg_noirs
