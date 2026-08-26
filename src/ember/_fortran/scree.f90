! Explicit time-march step kernels (Denton "scree" and Jameson RK), with
! optional block-sum multigrid and coarse-level implicit residual smoothing.
!
! The whole scheme x multigrid x IRS space is served by six branch-free wrappers
! over one shared, scheme-agnostic engine (mg_coarse_correction) and a set of
! branch-free building blocks. No configuration is decided by a runtime `if`
! inside a kernel: the scheme is fixed by which fine quantity `q` the wrapper
! forms and what it passes as the scatter's `base`; multigrid on/off is which wrapper the
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


! Bracketing coarse-cell index pair and upper weight for ONE fine cell of a
! factor-2 prolongation. n_coarse coarse cells; weight clamped to [0,1] with
! flat extrapolation past the outer coarse-cell centres.
!
! This used to be a loop that filled lo/hi/w arrays the length of the fine
! direction, one set per direction per call. Those were automatic arrays --
! an alloca per call, ~4.7 KB of stack at a 1M-cell block -- recomputed
! every call for what is pure geometry, and GCC's opt report flagged the
! whole trio as memory clobbers across the routine body. The bracket is a
! closed form in the fine index, so the j and k directions now evaluate it
! per outer-loop iteration (this routine), and the contiguous i direction
! bypasses it entirely (mg_interp_i2x below).
pure subroutine mg_bracket2x(i, n_coarse, lo, hi, w)

    implicit none

    integer, intent(in)  :: i, n_coarse
    integer, intent(out) :: lo, hi
    real,    intent(out) :: w

    integer :: icl
    real    :: t

    t   = (real(i) - 0.5e0) / 2e0 + 0.5e0
    icl = floor(t)
    if (icl < 1) then
        lo = 1
        hi = 1
        w  = 0e0
    else if (icl >= n_coarse) then
        lo = n_coarse
        hi = n_coarse
        w  = 0e0
    else
        lo = icl
        hi = icl + 1
        w  = t - real(icl)
    end if

end subroutine mg_bracket2x


! One factor-2 linear interpolation along the CONTIGUOUS direction, coarse
! column -> fine column.
!
! Driven by the coarse index rather than the fine one, which is what removes
! the index arrays here rather than merely shrinking them. Every interior
! fine cell brackets one coarse pair, and the weight alternates 1/4, 3/4 by
! parity, so one coarse pair emits both of its fine cells: contiguous reads,
! no gather, no per-call geometry. Bitwise identical to the general form --
! (1 - w) is exactly 0.75 or 0.25 in binary, and the two products are summed
! in the same order.
!
! The first and last fine cells clamp to the end coarse values, exactly as
! mg_bracket2x does. nfi == 2*nci for every hop the multigrid takes
! (ember.solver._validate_mg requires exact division at every level); the
! trailing loop is what a longer fine direction would need and normally runs
! once, for the clamped last cell.
pure subroutine mg_interp_i2x(cin, nci, cout, nfi)

    implicit none

    integer, intent(in)    :: nci, nfi
    real,    intent(in)    :: cin(nci)
    real,    intent(inout) :: cout(nfi)

    integer :: m, i

    cout(1) = cin(1)
    do m = 1, nci-1
        cout(2*m)   = cin(m)*0.75e0 + cin(m+1)*0.25e0
        cout(2*m+1) = cin(m)*0.25e0 + cin(m+1)*0.75e0
    end do
    do i = 2*nci, nfi
        cout(i) = cin(nci)
    end do

end subroutine mg_interp_i2x


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
!
! `fbnd` is fac_mgrid_bnd/fac_mgrid: the extra weight carried by the outer
! SHELL of this level's blocks -- ib = 1/nib, jb = 1/njb, kb = 1/nkb, the
! blocks with a face on the block boundary. Theirs is the block sum that
! straddles the boundary: it reaches 2**lvl fine cells deep, sees nothing
! across the face, and pushes that sum back over its whole footprint, so it is
! the least trustworthy of the level. Every level's coefficient is linear in
! fmgrid, so scaling the shell here is exactly the same as having run that
! level at fac_mgrid_bnd for those blocks, and it is applied AFTER the
! smoother, leaving IRS untouched. fbnd = 1 gives coef*1 == coef bitwise, so
! the uniform correction is reproduced exactly.
!
! The shell weight is hoisted to a scalar per (jb,kb) and the two i ends are
! peeled, so the inner loop stays branch-free. The end writes are assignments,
! not increments, so a degenerate nib == 1 (both landing on the same block)
! stores the same value twice and is harmless.
subroutine mg_scale_corr(corr, cres, dtblk, coef, fbnd, ldi, ldj, ldk, nib, njb, nkb, np)
    implicit none
    integer, intent(in) :: ldi, ldj, ldk, nib, njb, nkb, np
    real, intent(in)    :: cres(nib, njb, nkb, np)
    real, intent(in)    :: dtblk(ldi, ldj, ldk)
    real, intent(in)    :: coef, fbnd
    real, intent(out)   :: corr(nib, njb, nkb, np)
    integer :: ib, jb, kb, ip
    real    :: cbnd, cin
    cbnd = coef * fbnd
    do ip = 1, np
        do kb = 1, nkb
            do jb = 1, njb
                cin = coef
                if (kb == 1 .or. kb == nkb .or. jb == 1 .or. jb == njb) cin = cbnd
                do ib = 2, nib-1
                    corr(ib,jb,kb,ip) = cres(ib,jb,kb,ip) * cin * dtblk(ib,jb,kb)
                end do
                corr(1,jb,kb,ip) = cres(1,jb,kb,ip) * cbnd * dtblk(1,jb,kb)
                corr(nib,jb,kb,ip) = cres(nib,jb,kb,ip) * cbnd * dtblk(nib,jb,kb)
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
    integer :: jlo, jhi, klo, khi
    real    :: wj, wk

    do ip = 1, np
        do kc = 1, nck
            do jc = 1, ncj
                call mg_interp_i2x(src(1,jc,kc,ip), nci, aplane(1,jc), nfi)
            end do
            do j = 1, nfj
                call mg_bracket2x(j, ncj, jlo, jhi, wj)
                do i = 1, nfi
                    bb(i,j,kc,ip) = aplane(i,jlo)*(1e0-wj) &
                                  + aplane(i,jhi)*wj
                end do
            end do
        end do
        do k = 1, nfk
            call mg_bracket2x(k, nck, klo, khi, wk)
            do j = 1, nfj
                do i = 1, nfi
                    out(i,j,k,ip) = out(i,j,k,ip) &
                                  + bb(i,j,klo,ip)*(1e0-wk) &
                                  + bb(i,j,khi,ip)*wk
                end do
            end do
        end do
    end do
end subroutine mg_prolong2x_acc


! Fused final cascade hop + cell->node scatter, shared by the RK and scree
! multigrid wrappers. Instead of writing a full-volume cell increment and
! re-reading it in cell_to_node_generic, this produces the increment one fine
! k-plane at a time into a rolling two-plane buffer `rbuf` and scatters each
! finished node plane straight into `cons`, so the 5-component increment is
! never materialised full-volume -- removing that write+read round-trip and,
! with it, the (ni-1)*(nj-1)*(nk-1)*np arena slot it used to need.
!
! The increment is `scale*dt_vol*q + k-interp(bb)` and each node value is the
! average of its surrounding cell increments (interior 1/8 of 8 cells, i/j
! faces 1/4 of 4, edges 1/2 of 2, corners 1 of 1) -- term-for-term and in the
! same summation order as the split form it replaced. Cell plane kc feeds node
! plane kc (as the k-upper plane) and, at the ends, the two k-boundary node
! planes.
!
! `base` is what the scatter adds onto. RK passes its sub-stage snapshot, an
! array distinct from `cons`. scree scatters IN PLACE and passes `cons` itself
! for both, the same aliased call cell_to_node makes into cell_to_node_generic:
! every node is read and written at its own index within one statement, so the
! aliasing is benign. scree then rolls its Denton history separately
! (scree_roll), which needs no increment buffer.
subroutine mg_prolong2x_fine_scatter(src, nci, ncj, nck, base, cons, &
        scale, dt_vol, q, ni, nj, nk, np, aplane, bb, rbuf, nc1j, nc1k)
    implicit none
    integer, intent(in) :: nci, ncj, nck, ni, nj, nk, np, nc1j, nc1k
    real, intent(in)    :: src(nci, ncj, nck, np)
    real, intent(in)    :: scale
    real, intent(in)    :: dt_vol(ni-1, nj-1, nk-1)
    real, intent(in)    :: q(ni-1, nj-1, nk-1, np)
    real, intent(in)    :: base(ni, nj, nk, np)
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: aplane(ni-1, nc1j)
    real, intent(inout) :: bb(ni-1, nj-1, nc1k, np)
    real, intent(inout) :: rbuf(ni-1, nj-1, np, 2)
    integer :: i, j, ip, jc, kc, cur, prev, sw
    integer :: jlo, jhi, klo, khi
    real    :: wj, wk

    ! Phase A: build bb, the k-interpolation source planes (i then j interp).
    do ip = 1, np
        do kc = 1, nck
            do jc = 1, ncj
                call mg_interp_i2x(src(1,jc,kc,ip), nci, aplane(1,jc), ni-1)
            end do
            do j = 1, nj-1
                call mg_bracket2x(j, ncj, jlo, jhi, wj)
                do i = 1, ni-1
                    bb(i,j,kc,ip) = aplane(i,jlo)*(1e0-wj) &
                                  + aplane(i,jhi)*wj
                end do
            end do
        end do
    end do

    ! Phase B: rolling k-plane increment + immediate node-plane scatter.
    cur  = 1
    prev = 2
    do kc = 1, nk-1
        call mg_bracket2x(kc, nck, klo, khi, wk)
        do ip = 1, np
            do j = 1, nj-1
                do i = 1, ni-1
                    rbuf(i,j,ip,cur) = scale*dt_vol(i,j,kc)*q(i,j,kc,ip) &
                                     + bb(i,j,klo,ip)*(1e0-wk) &
                                     + bb(i,j,khi,ip)*wk
                end do
            end do
        end do
        if (kc == 1)    call emit_kbnd(cur, 1)
        if (kc >= 2)    call emit_kint(prev, cur, kc)
        if (kc == nk-1) call emit_kbnd(cur, nk)
        sw = cur; cur = prev; prev = sw
    end do

contains

    ! Interior-k node plane kk (2..nk-1): 8 surrounding cells, planes kk-1
    ! (buffer bp) and kk (buffer bc). Term order matches cell_to_node_generic's
    ! interior stencil so the scatter is arithmetically identical.
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
    ! bc). interior 1/4 of 4 cells, i/j faces 1/2 of 2, corners the one cell.
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

end subroutine mg_prolong2x_fine_scatter


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
! scree wrappers form q in store (scree_form_q) and roll the history
! (scree_roll); rk wrappers pass residual as q. Both multigrid-on wrappers
! scatter through the fused mg_prolong2x_fine_scatter, scree frozen-in-place
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
! Prolongation is CASCADED: the per-level scaled corrections (packed coarsest
! first in corr_all) accumulate coarsest -> finest through factor-2 hops, so only
! the final hop writes the fine grid (fused with the fine term). This is a
! genuine operator change from a direct factor-b prolong -- cascaded factor-2
! trilinear interpolations are not equal to it.
!
! `fbnd` (fac_mgrid_bnd/fac_mgrid) weakens the correction on the boundary shell
! of blocks at EVERY level, where the block sum straddles the block face. It is
! carried through to mg_scale_corr, the one place a level's correction is
! formed; restriction, smoothing and prolongation know nothing about it.
! ============================================================================
subroutine mg_coarse_correction(q, dt_vol, vol, scale, fmgrid, fbnd, expon_mgrid, &
        sf_irs, n_levels, dtblk, aplane, bb, rawbuf, sdt, sv, &
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
    real, intent(in)    :: scale, fmgrid, expon_mgrid, sf_irs
    ! Ratio fac_mgrid_bnd/fac_mgrid, the weight this level's boundary shell of
    ! blocks carries; 1 leaves the uniform correction untouched. See
    ! mg_scale_corr, which is where it lands.
    real, intent(in)    :: fbnd
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
    ! Level tables. Sized by a PARAMETER, not by n_levels: a runtime-sized
    ! local is an alloca on every call, and GCC's opt report shows the four
    ! of them costing a stack_save/alloca/stack_restore trio that also
    ! clobbers memory across the whole body. There is a compile-time bound
    ! available -- ember.block.MAX_MG_LEVELS, which ember.solver._validate_mg
    ! already enforces because Block.scratch is one arena sized for it -- so
    ! the tables can simply be that long. Keep the two in step.
    integer, parameter :: MG_LEVELS_MAX = 3
    integer :: dib(MG_LEVELS_MAX), djb(MG_LEVELS_MAX), dkb(MG_LEVELS_MAX)
    integer :: offc(MG_LEVELS_MAX)
    integer :: nci, ncj, nck, cur_i, cur_j, cur_k, o
    logical :: in0

    ! The caller validates this (ember.solver._validate_mg), so reaching it
    ! means the two bounds have drifted apart; say so rather than writing
    ! past the tables.
    if (n_levels > MG_LEVELS_MAX) stop 'mg_coarse_correction: n_levels > MG_LEVELS_MAX'

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
    coef = scale * fmgrid / real(b*b) * expon_mgrid**(-(lvl-1))
    slot = n_levels - lvl + 1
    cnt  = nib*njb*nkb*np

    ! dt restriction (volume-weighted mean) from the fine grid. ifort declines
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
    call mg_gather_corner(cres, rawbuf, nc1i, nc1j, nc1k, nib, njb, nkb, np)
    call smoother(cres(1:cnt), sf_irs, &
                  triw(1:2*(nib+njb+nkb)), nib+1, njb+1, nkb+1)
    call mg_scale_corr(corr_all(offc(slot)+1), cres, dtblk, coef, fbnd, &
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
        call mg_scale_corr(corr_all(offc(slot)+1), cres, dtblk, coef, fbnd, &
                           nc1i, nc1j, nc1k, nib, njb, nkb, np)
    end do

    ! Phase 2: cascaded coarsest->fine prolongation.
    !
    ! Each hop seeds the NEXT buffer with that level's own correction and
    ! accumulates the interpolated coarser one into it, so the running
    ! cascade alternates between acc0 and acc1. It used to be copied back to
    ! acc0 after every hop, purely so the caller could find it there; at
    ! n_levels=3 that copy-back was most of the memmove traffic this kernel
    ! spent (about 7% of stage time in a perf profile). The parity of the hop
    ! count says where it ends up, so choosing the STARTING buffer by that
    ! parity lands it in acc0 with no copy at all: n_levels-1 hops, so start
    ! in acc0 when that is even. n_levels=1 makes no hop and starts (and
    ! ends) in acc0, as before.
    in0 = mod(n_levels - 1, 2) == 0
    if (in0) then
        call mg_copy(corr_all(offc(1)+1), acc0, dib(1)*djb(1)*dkb(1)*np)
    else
        call mg_copy(corr_all(offc(1)+1), acc1, dib(1)*djb(1)*dkb(1)*np)
    end if
    cur_i = dib(1)
    cur_j = djb(1)
    cur_k = dkb(1)
    do lvl = 2, n_levels
        nci = cur_i
        ncj = cur_j
        nck = cur_k
        ! The two branches are the same hop with the buffers exchanged --
        ! Fortran dummies cannot be swapped by name, so the alternation is
        ! spelled out rather than hidden behind a pointer.
        if (in0) then
            call mg_copy(corr_all(offc(lvl)+1), acc1, dib(lvl)*djb(lvl)*dkb(lvl)*np)
            call mg_prolong2x_acc(acc0, nci, ncj, nck, acc1, &
                                  dib(lvl), djb(lvl), dkb(lvl), np, &
                                  aplane, bb, ni-1, nj-1, nc1k)
        else
            call mg_copy(corr_all(offc(lvl)+1), acc0, dib(lvl)*djb(lvl)*dkb(lvl)*np)
            call mg_prolong2x_acc(acc1, nci, ncj, nck, acc0, &
                                  dib(lvl), djb(lvl), dkb(lvl), np, &
                                  aplane, bb, ni-1, nj-1, nc1k)
        end if
        in0 = .not. in0
        cur_i = dib(lvl)
        cur_j = djb(lvl)
        cur_k = dkb(lvl)
    end do
    ! Leaves the finest-coarse correction in acc0 at (nc1i,nc1j,nc1k). The final
    ! factor-2 hop onto the fine grid is done by the caller so that both schemes
    ! can fuse it with the cell->node scatter (mg_prolong2x_fine_scatter).
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
        fmgrid, fbnd, expon_mgrid, sf_irs, n_levels, rbuf, dtblk, aplane, bb, rawbuf, sdt, sv, &
        corr_all, acc0, acc1, cres, triw, &
        ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    implicit none
    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k
    integer, intent(in) :: n_corr, n_res, n_tri
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real,    intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: cfl, fmgrid, fbnd, expon_mgrid, sf_irs
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: store(ni-1, nj-1, nk-1, np)   ! in: (dF/dt)_{n-1}; out: rolled to residual
    real, intent(inout) :: rbuf(ni-1, nj-1, np, 2)
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
    call mg_coarse_correction(store, dt_vol, vol, cfl, fmgrid, fbnd, expon_mgrid, sf_irs, n_levels, &
                       dtblk, aplane, bb, rawbuf, sdt, sv, &
                       corr_all, acc0, acc1, cres, triw, smooth_residual_tri_tiled, &
                       ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    call mg_prolong2x_fine_scatter(acc0, nc1i, nc1j, nc1k, cons, cons, &
                           cfl, dt_vol, store, ni, nj, nk, np, &
                           aplane, bb, rbuf, nc1j, nc1k)
    call scree_roll(residual, store, ni, nj, nk, np)
end subroutine scree_mg_irs


! scree, multigrid on, no smoothing.
subroutine scree_mg_noirs(cons, residual, store, dt_vol, vol, cfl, &
        fmgrid, fbnd, expon_mgrid, sf_irs, n_levels, rbuf, dtblk, aplane, bb, rawbuf, sdt, sv, &
        corr_all, acc0, acc1, cres, triw, &
        ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    implicit none
    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k
    integer, intent(in) :: n_corr, n_res, n_tri
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real,    intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: cfl, fmgrid, fbnd, expon_mgrid, sf_irs
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: store(ni-1, nj-1, nk-1, np)   ! in: (dF/dt)_{n-1}; out: rolled to residual
    real, intent(inout) :: rbuf(ni-1, nj-1, np, 2)
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
    call mg_coarse_correction(store, dt_vol, vol, cfl, fmgrid, fbnd, expon_mgrid, sf_irs, n_levels, &
                       dtblk, aplane, bb, rawbuf, sdt, sv, &
                       corr_all, acc0, acc1, cres, triw, mg_smooth_noop, &
                       ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    call mg_prolong2x_fine_scatter(acc0, nc1i, nc1j, nc1k, cons, cons, &
                           cfl, dt_vol, store, ni, nj, nk, np, &
                           aplane, bb, rbuf, nc1j, nc1k)
    call scree_roll(residual, store, ni, nj, nk, np)
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
        alpha, cfl, fmgrid, fbnd, expon_mgrid, sf_irs, n_levels, rbuf, dtblk, aplane, bb, &
        rawbuf, sdt, sv, corr_all, acc0, acc1, cres, triw, &
        ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    implicit none
    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k
    integer, intent(in) :: n_corr, n_res, n_tri
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real,    intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: alpha, cfl, fmgrid, fbnd, expon_mgrid, sf_irs
    real,    intent(in) :: snapshot(ni, nj, nk, np)
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: rbuf(ni-1, nj-1, np, 2)
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

    call mg_coarse_correction(residual, dt_vol, vol, alpha*cfl, fmgrid, fbnd, expon_mgrid, sf_irs, &
                       n_levels, dtblk, aplane, bb, rawbuf, sdt, sv, &
                       corr_all, acc0, acc1, cres, triw, smooth_residual_tri_tiled, &
                       ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    call mg_prolong2x_fine_scatter(acc0, nc1i, nc1j, nc1k, snapshot, cons, &
                       alpha*cfl, dt_vol, residual, ni, nj, nk, np, &
                       aplane, bb, rbuf, nc1j, nc1k)
end subroutine rk_mg_irs


! RK stage, multigrid on, no smoothing. q = residual (passed directly).
subroutine rk_mg_noirs(cons, snapshot, residual, dt_vol, vol, &
        alpha, cfl, fmgrid, fbnd, expon_mgrid, sf_irs, n_levels, rbuf, dtblk, aplane, bb, &
        rawbuf, sdt, sv, corr_all, acc0, acc1, cres, triw, &
        ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    implicit none
    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k
    integer, intent(in) :: n_corr, n_res, n_tri
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real,    intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: alpha, cfl, fmgrid, fbnd, expon_mgrid, sf_irs
    real,    intent(in) :: snapshot(ni, nj, nk, np)
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: rbuf(ni-1, nj-1, np, 2)
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

    call mg_coarse_correction(residual, dt_vol, vol, alpha*cfl, fmgrid, fbnd, expon_mgrid, sf_irs, &
                       n_levels, dtblk, aplane, bb, rawbuf, sdt, sv, &
                       corr_all, acc0, acc1, cres, triw, mg_smooth_noop, &
                       ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    call mg_prolong2x_fine_scatter(acc0, nc1i, nc1j, nc1k, snapshot, cons, &
                       alpha*cfl, dt_vol, residual, ni, nj, nk, np, &
                       aplane, bb, rbuf, nc1j, nc1k)
end subroutine rk_mg_noirs
