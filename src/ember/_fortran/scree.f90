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


! Bracketing coarse-cell index pair for ONE fine cell of a factor-2
! prolongation, over n_coarse coarse cells. The two ends clamp onto a single
! coarse cell (lo == hi), which is the flat extrapolation past the outer
! coarse-cell centres.
!
! The bracket is index arithmetic and stays so: every fine cell does lie
! between the two coarse centroids its index picks, because the clamp never
! engages in the interior. The interpolation WEIGHT is not computed here --
! it is geometry, read from the fields the caller supplies (see
! mg_weight_offsets below and ember.block.Block.weight_mgrid).
!
! This used to be a loop that filled lo/hi/w arrays the length of the fine
! direction, one set per direction per call. Those were automatic arrays --
! an alloca per call, ~4.7 KB of stack at a 1M-cell block -- recomputed
! every call for what is pure geometry, and GCC's opt report flagged the
! whole trio as memory clobbers across the routine body. The bracket is a
! closed form in the fine index, so the j and k directions now evaluate it
! per outer-loop iteration (this routine), and the contiguous i direction
! bypasses it entirely (mg_interp_i2x below).
pure subroutine mg_bracket2x(i, n_coarse, lo, hi)

    implicit none

    integer, intent(in)  :: i, n_coarse
    integer, intent(out) :: lo, hi

    integer :: icl
    real    :: t

    t   = (real(i) - 0.5e0) / 2e0 + 0.5e0
    icl = floor(t)
    if (icl < 1) then
        lo = 1
        hi = 1
    else if (icl >= n_coarse) then
        lo = n_coarse
        hi = n_coarse
    else
        lo = icl
        hi = icl + 1
    end if

end subroutine mg_bracket2x


! One factor-2 linear interpolation along the CONTIGUOUS direction, coarse
! column -> fine column.
!
! Driven by the coarse index rather than the fine one, which is what removes
! the index arrays here rather than merely shrinking them. Every interior
! fine cell brackets one coarse pair, so one coarse pair emits both of its
! fine cells: contiguous reads, no gather, no index arithmetic per cell.
!
! The weight pair used to be the literals 1/4, 3/4, which is where the fine
! cell sits between its coarse neighbours only on a mesh uniform in physical
! space. It now arrives in `w`, one entry per fine cell, taken from the real
! cell centroids. Fed those same literals this reproduces the old output bit
! for bit: (1 - w) is exact for 0.75 and 0.25 in binary and the two products
! are summed in the same order.
!
! The first and last fine cells clamp to the end coarse values, exactly as
! mg_bracket2x does. nfi == 2*nci for every hop the multigrid takes
! (ember.solver._validate_mg requires exact division at every level); the
! trailing loop is what a longer fine direction would need and normally runs
! once, for the clamped last cell.
pure subroutine mg_interp_i2x(cin, nci, cout, nfi, w)

    implicit none

    integer, intent(in)    :: nci, nfi
    real,    intent(in)    :: cin(nci)
    real,    intent(inout) :: cout(nfi)
    real,    intent(in)    :: w(nfi)

    integer :: m, i

    ! cout(1) clamps onto cin(1) (lo == hi there), so its weight cannot matter.
    cout(1) = cin(1)
    do m = 1, nci-1
        cout(2*m)   = cin(m)*(1e0-w(2*m))   + cin(m+1)*w(2*m)
        cout(2*m+1) = cin(m)*(1e0-w(2*m+1)) + cin(m+1)*w(2*m+1)
    end do
    do i = 2*nci, nfi
        cout(i) = cin(nci)
    end do

end subroutine mg_interp_i2x


! Bracketing coarse-cell index pair for ONE fine NODE of the final prolongation
! hop, over n_coarse coarse cells. The node counterpart of mg_bracket2x, used by
! the fused final hop, which targets the nodes directly rather than the cell
! centres (see mg_prolong2x_fine_scatter).
!
! Node 2m sits at the interface between the two children of coarse cell m, so
! the pair (m, m+1) serves nodes 2m and 2m+1 and the whole bracket is i/2,
! clamped so the pair stays inside the grid. Integer division only.
!
! The cell routine's justification -- that the clamp never engages in the
! interior, because a fine cell centroid always lies between the coarse
! centroids its index picks -- does NOT carry over here. The coarse centroid is
! the VOLUME-weighted mean of its two children, so it sits off the node between
! them, on whichever side the local stretching puts it; node 2m can therefore
! fall just outside the pair (m, m+1). That is why the weights this pair is fed
! are allowed slightly outside [0, 1] in the interior (ember.block._mg_project):
! the bracket stays fixed and the blend extrapolates by the few percent of a
! coarse cell the clustering asks for. Clamping to [0, 1] instead would flatten
! the correction at every even node, which is the bug this pair exists to avoid.
!
! Slightly outside, and no further: Python bounds them at
! ember.block.MG_W_LO/MG_W_HI, so whatever the mesh does this stays a blend and
! the three passes cannot amplify the coarse correction they carry to the node.
! Nothing here checks that -- the kernel takes the weights as given.
pure subroutine mg_bracket2x_node(i, n_coarse, lo, hi)

    implicit none

    integer, intent(in)  :: i, n_coarse
    integer, intent(out) :: lo, hi

    lo = min(max(i/2, 1), max(n_coarse-1, 1))
    hi = min(lo+1, n_coarse)

end subroutine mg_bracket2x_node


! One factor-2 linear interpolation along the CONTIGUOUS direction, coarse
! column -> fine NODE column. The node counterpart of mg_interp_i2x, and driven
! by the coarse index for the same reason: pair (m, m+1) emits nodes 2m and
! 2m+1, so the reads stay contiguous and no index arithmetic runs per node.
!
! nfi == 2*nci + 1 for the final hop (one more node than there are fine cells).
! The two extreme nodes always lie outside the outer coarse centroids -- node 1
! below the first, node nfi above the last -- so they take the flat end value,
! exactly as the cell routine's ends do. The trailing loop covers nodes 2*nci
! and 2*nci+1, which the coarse-driven loop cannot reach because their pair is
! (nci-1, nci) rather than (nci, nci+1).
pure subroutine mg_interp_i2x_node(cin, nci, cout, nfi, w)

    implicit none

    integer, intent(in)    :: nci, nfi
    real,    intent(in)    :: cin(nci)
    real,    intent(inout) :: cout(nfi)
    real,    intent(in)    :: w(nfi)

    integer :: m, i, mlast

    ! Node 1 is always below the first coarse centroid, so its weight cannot
    ! matter; the flat extrapolation there is cin(1) whatever w(1) holds.
    cout(1) = cin(1)
    do m = 1, nci-1
        cout(2*m)   = cin(m)*(1e0-w(2*m))   + cin(m+1)*w(2*m)
        cout(2*m+1) = cin(m)*(1e0-w(2*m+1)) + cin(m+1)*w(2*m+1)
    end do
    ! nci == 1 leaves no pair at all (a two-cell direction); everything is then
    ! the single coarse value, which max(nci-1,1) == nci == 1 delivers.
    mlast = max(nci-1, 1)
    do i = 2*nci, nfi
        cout(i) = cin(mlast)*(1e0-w(i)) + cin(nci)*w(i)
    end do

end subroutine mg_interp_i2x_node


! Packed-array offsets for the per-hop prolongation weights, and the single
! source of truth for a layout Python has to reproduce exactly (see
! ember.block._mg_weight_lengths). Hop m prolongs onto a target of
! ((ni-1),(nj-1),(nk-1))/2**(m-1), so m = 1 is the final hop onto the fine grid
! (the one the caller fuses with the scatter) and m grows towards the coarse
! end. Indexing from the fine grid DOWN is what makes the layout independent of
! n_levels, so one cached array on the block serves every level count.
!
! The three directions pack separately because the three passes resolve i, then
! j, then k, so each carries a different mix of fine and coarse extents:
!   wi(tfi, tfj/2, tfk/2)   wj(tfi, tfj, tfk/2)   wk(tfi, tfj, tfk)
!
! Hop 1 is the exception: it targets the fine NODES, not the fine cell centres,
! so its three extents are (ni, nj, nk) rather than (ni-1, nj-1, nk-1) while the
! coarse extents it interpolates from are unchanged. Hops 2 and up are
! coarse->coarse and stay cell-shaped.
subroutine mg_weight_offsets(ni, nj, nk, n_hops, offwi, offwj, offwk)

    implicit none

    integer, intent(in)  :: ni, nj, nk, n_hops
    integer, intent(out) :: offwi(n_hops), offwj(n_hops), offwk(n_hops)

    integer :: m, d, tfi, tfj, tfk, oi, oj, ok

    oi = 0
    oj = 0
    ok = 0
    do m = 1, n_hops
        d   = 2**(m - 1)
        tfi = (ni-1)/d
        tfj = (nj-1)/d
        tfk = (nk-1)/d
        offwi(m) = oi
        offwj(m) = oj
        offwk(m) = ok
        if (m == 1) then
            ! Node-targeted final hop: (ni,nj,nk) targets over the same
            ! ((nj-1)/2, (nk-1)/2) coarse extents the cell form used.
            oi = oi + ni*((nj-1)/2)*((nk-1)/2)
            oj = oj + ni*nj*((nk-1)/2)
            ok = ok + ni*nj*nk
        else
            oi = oi + tfi*(tfj/2)*(tfk/2)
            oj = oj + tfi*tfj*(tfk/2)
            ok = ok + tfi*tfj*tfk
        end if
    end do

end subroutine mg_weight_offsets


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
! prefilled with that level's correction). These hops stay CELL-targeted; only
! the final hop reaches the nodes (mg_prolong2x_fine_scatter).
!
! aplane/bb are the shared prolong scratch (leading dims ni1,nj1; third dim
! nkpad >= nck). The caller sizes that scratch for the node-targeted final hop
! and hands it here at the smaller cell leading dims ni-1, nj-1 -- a strict
! subset of the storage, so the shapes no longer match by inspection but every
! access stays in bounds.
subroutine mg_prolong2x_acc(src, nci, ncj, nck, out, nfi, nfj, nfk, np, &
        aplane, bb, ni1, nj1, nkpad, wi, wj, wk)
    implicit none
    integer, intent(in) :: nci, ncj, nck, nfi, nfj, nfk, np, ni1, nj1, nkpad
    real, intent(in)    :: src(nci, ncj, nck, np)
    real, intent(inout) :: out(nfi, nfj, nfk, np)
    real, intent(inout) :: aplane(ni1, *)
    real, intent(inout) :: bb(ni1, nj1, nkpad, np)
    ! This hop's slice of the block's packed weights. Shaped to the three
    ! passes: i is resolved while j,k are still coarse, then j while k is
    ! coarse, then k.
    real, intent(in)    :: wi(nfi, ncj, nck)
    real, intent(in)    :: wj(nfi, nfj, nck)
    real, intent(in)    :: wk(nfi, nfj, nfk)
    integer :: i, j, k, ip, jc, kc
    integer :: jlo, jhi, klo, khi

    do ip = 1, np
        do kc = 1, nck
            do jc = 1, ncj
                call mg_interp_i2x(src(1,jc,kc,ip), nci, aplane(1,jc), nfi, &
                                   wi(1,jc,kc))
            end do
            do j = 1, nfj
                call mg_bracket2x(j, ncj, jlo, jhi)
                do i = 1, nfi
                    bb(i,j,kc,ip) = aplane(i,jlo)*(1e0-wj(i,j,kc)) &
                                  + aplane(i,jhi)*wj(i,j,kc)
                end do
            end do
        end do
        do k = 1, nfk
            call mg_bracket2x(k, nck, klo, khi)
            do j = 1, nfj
                do i = 1, nfi
                    out(i,j,k,ip) = out(i,j,k,ip) &
                                  + bb(i,j,klo,ip)*(1e0-wk(i,j,k)) &
                                  + bb(i,j,khi,ip)*wk(i,j,k)
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
! The two halves of the increment land at the nodes by different routes,
! because they are different kinds of quantity:
!
!   fine term      scale*dt_vol*q is a CELL quantity, and its distribution onto
!                  the nodes is the scheme's own (interior 1/8 of 8 cells, i/j
!                  faces 1/4 of 4, edges 1/2 of 2, corners 1 of 1) -- kept
!                  term-for-term and in the same summation order as
!                  cell_to_node_generic. Cell plane kc feeds node plane kc (as
!                  the k-upper plane) and, at the ends, the two k-boundary node
!                  planes.
!
!   correction     the final prolongation hop targets the NODES directly, so
!                  the coarse correction never visits a cell centre and is
!                  never averaged. This used to interpolate onto the fine cell
!                  centres and then ride the same 1/8 scatter -- which, on a
!                  stretched mesh, evaluates the (correctly weighted)
!                  interpolant at the mean of the eight surrounding fine
!                  centroids rather than at the node. That mean is offset from
!                  the node by (h_{j+1} - h_j)/4 in each direction, a
!                  first-order error in the clustering ratio that undid most of
!                  what the geometry weights bought. One weighted operator
!                  instead of a weighted one composed with an unweighted one.
!
! Note the composition dropped with it: on a uniform mesh the old pair was
! exactly [node prolongation] + 1/8 of the coarse correction's second
! difference, i.e. a smoothed prolongation. The correction now lands unsmoothed;
! coarse-level IRS (sf_irs) is the knob if it ever needs damping.
!
! `base` is what the scatter adds onto. RK passes its sub-stage snapshot, an
! array distinct from `cons`. scree scatters IN PLACE and passes `cons` itself
! for both, the same aliased call cell_to_node makes into cell_to_node_generic:
! every node is read and written at its own index within one statement, so the
! aliasing is benign. scree then rolls its Denton history separately
! (scree_roll), which needs no increment buffer.
subroutine mg_prolong2x_fine_scatter(src, nci, ncj, nck, base, cons, &
        scale, dt_vol, q, ni, nj, nk, np, aplane, bb, cbuf, rbuf, nc1j, nc1k, &
        wi, wj, wk)
    implicit none
    integer, intent(in) :: nci, ncj, nck, ni, nj, nk, np, nc1j, nc1k
    real, intent(in)    :: src(nci, ncj, nck, np)
    real, intent(in)    :: scale
    real, intent(in)    :: dt_vol(ni-1, nj-1, nk-1)
    real, intent(in)    :: q(ni-1, nj-1, nk-1, np)
    real, intent(in)    :: base(ni, nj, nk, np)
    real, intent(inout) :: cons(ni, nj, nk, np)
    ! i and j resolved onto NODES here, so these carry node extents; the
    ! coarse->coarse hops reuse the same storage at the smaller cell extents.
    real, intent(inout) :: aplane(ni, nc1j)
    real, intent(inout) :: bb(ni, nj, nc1k, np)
    ! One node plane of prolonged correction, refilled immediately before each
    ! emit. A single plane is enough because each emit consumes it at once.
    real, intent(inout) :: cbuf(ni, nj, np)
    real, intent(inout) :: rbuf(ni-1, nj-1, np, 2)
    ! The final hop's slice of the block's packed weights. Node-targeted, so
    ! each pass carries one more entry per resolved direction than the
    ! coarse->coarse shapes in mg_prolong2x_acc.
    real, intent(in)    :: wi(ni, ncj, nck)
    real, intent(in)    :: wj(ni, nj, nck)
    real, intent(in)    :: wk(ni, nj, nk)
    integer :: i, j, ip, jc, kc, cur, prev, sw
    integer :: jlo, jhi

    ! Phase A: build bb, the k-interpolation source planes (i then j interp,
    ! both already onto node positions).
    do ip = 1, np
        do kc = 1, nck
            do jc = 1, ncj
                call mg_interp_i2x_node(src(1,jc,kc,ip), nci, aplane(1,jc), ni, &
                                        wi(1,jc,kc))
            end do
            do j = 1, nj
                call mg_bracket2x_node(j, ncj, jlo, jhi)
                do i = 1, ni
                    bb(i,j,kc,ip) = aplane(i,jlo)*(1e0-wj(i,j,kc)) &
                                  + aplane(i,jhi)*wj(i,j,kc)
                end do
            end do
        end do
    end do

    ! Phase B: rolling k-plane fine term + immediate node-plane scatter, with
    ! the correction's k interpolation done straight onto each node plane.
    cur  = 1
    prev = 2
    do kc = 1, nk-1
        do ip = 1, np
            do j = 1, nj-1
                do i = 1, ni-1
                    rbuf(i,j,ip,cur) = scale*dt_vol(i,j,kc)*q(i,j,kc,ip)
                end do
            end do
        end do
        ! Both ends emit two node planes in one pass, so cbuf is refilled
        ! between them rather than once per iteration.
        if (kc == 1) then
            call fill_cbuf(1)
            call emit_kbnd(cur, 1)
        end if
        if (kc >= 2) then
            call fill_cbuf(kc)
            call emit_kint(prev, cur, kc)
        end if
        if (kc == nk-1) then
            call fill_cbuf(nk)
            call emit_kbnd(cur, nk)
        end if
        sw = cur; cur = prev; prev = sw
    end do

contains

    ! k-interpolate bb onto node plane kk, the last of the three prolongation
    ! passes. Leaves the prolonged correction in cbuf for the emit that follows.
    subroutine fill_cbuf(kk)
        integer, intent(in) :: kk
        integer :: i, j, ip, klo, khi
        call mg_bracket2x_node(kk, nck, klo, khi)
        do ip = 1, np
            do j = 1, nj
                do i = 1, ni
                    cbuf(i,j,ip) = bb(i,j,klo,ip)*(1e0-wk(i,j,kk)) &
                                 + bb(i,j,khi,ip)*wk(i,j,kk)
                end do
            end do
        end do
    end subroutine fill_cbuf

    ! Interior-k node plane kk (2..nk-1): 8 surrounding cells, planes kk-1
    ! (buffer bp) and kk (buffer bc). Term order matches cell_to_node_generic's
    ! interior stencil so the fine term's scatter is arithmetically identical.
    ! cbuf is the already-prolonged correction at this very node, added rather
    ! than averaged in -- it never passed through a cell.
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
                      + rbuf(i,j,ip,bc)     + rbuf(i-1,j,ip,bc))*0.125e0 &
                      + cbuf(i,j,ip)
                end do
            end do
            do j = 2, nj-1
                cons(1,j,kk,ip) = base(1,j,kk,ip) + ( &
                    rbuf(1,j-1,ip,bp) + rbuf(1,j,ip,bp) &
                  + rbuf(1,j-1,ip,bc) + rbuf(1,j,ip,bc))*0.25e0 &
                  + cbuf(1,j,ip)
                cons(ni,j,kk,ip) = base(ni,j,kk,ip) + ( &
                    rbuf(ni-1,j-1,ip,bp) + rbuf(ni-1,j,ip,bp) &
                  + rbuf(ni-1,j-1,ip,bc) + rbuf(ni-1,j,ip,bc))*0.25e0 &
                  + cbuf(ni,j,ip)
            end do
            do i = 2, ni-1
                cons(i,1,kk,ip) = base(i,1,kk,ip) + ( &
                    rbuf(i-1,1,ip,bp) + rbuf(i,1,ip,bp) &
                  + rbuf(i-1,1,ip,bc) + rbuf(i,1,ip,bc))*0.25e0 &
                  + cbuf(i,1,ip)
                cons(i,nj,kk,ip) = base(i,nj,kk,ip) + ( &
                    rbuf(i-1,nj-1,ip,bp) + rbuf(i,nj-1,ip,bp) &
                  + rbuf(i-1,nj-1,ip,bc) + rbuf(i,nj-1,ip,bc))*0.25e0 &
                  + cbuf(i,nj,ip)
            end do
            cons(1,1,kk,ip) = base(1,1,kk,ip) &
                + (rbuf(1,1,ip,bp) + rbuf(1,1,ip,bc))*0.5e0 + cbuf(1,1,ip)
            cons(1,nj,kk,ip) = base(1,nj,kk,ip) &
                + (rbuf(1,nj-1,ip,bp) + rbuf(1,nj-1,ip,bc))*0.5e0 + cbuf(1,nj,ip)
            cons(ni,nj,kk,ip) = base(ni,nj,kk,ip) &
                + (rbuf(ni-1,nj-1,ip,bp) + rbuf(ni-1,nj-1,ip,bc))*0.5e0 + cbuf(ni,nj,ip)
            cons(ni,1,kk,ip) = base(ni,1,kk,ip) &
                + (rbuf(ni-1,1,ip,bp) + rbuf(ni-1,1,ip,bc))*0.5e0 + cbuf(ni,1,ip)
        end do
    end subroutine emit_kint

    ! k-boundary node plane kk (1 or nk): single adjacent cell plane (buffer
    ! bc). interior 1/4 of 4 cells, i/j faces 1/2 of 2, corners the one cell.
    ! The correction is added at full weight as in emit_kint: the k-boundary
    ! node planes are where the old cell-centred route was worst, taking the
    ! interpolant a whole half cell inside the boundary.
    subroutine emit_kbnd(bc, kk)
        integer, intent(in) :: bc, kk
        integer :: i, j, ip
        do ip = 1, np
            do j = 2, nj-1
                do i = 2, ni-1
                    cons(i,j,kk,ip) = base(i,j,kk,ip) + ( &
                        rbuf(i-1,j-1,ip,bc) + rbuf(i,j-1,ip,bc) &
                      + rbuf(i-1,j,ip,bc)   + rbuf(i,j,ip,bc))*0.25e0 &
                      + cbuf(i,j,ip)
                end do
            end do
            do j = 2, nj-1
                cons(1,j,kk,ip) = base(1,j,kk,ip) &
                    + (rbuf(1,j-1,ip,bc) + rbuf(1,j,ip,bc))*0.5e0 + cbuf(1,j,ip)
                cons(ni,j,kk,ip) = base(ni,j,kk,ip) &
                    + (rbuf(ni-1,j-1,ip,bc) + rbuf(ni-1,j,ip,bc))*0.5e0 + cbuf(ni,j,ip)
            end do
            do i = 2, ni-1
                cons(i,1,kk,ip) = base(i,1,kk,ip) &
                    + (rbuf(i-1,1,ip,bc) + rbuf(i,1,ip,bc))*0.5e0 + cbuf(i,1,ip)
                cons(i,nj,kk,ip) = base(i,nj,kk,ip) &
                    + (rbuf(i-1,nj-1,ip,bc) + rbuf(i,nj-1,ip,bc))*0.5e0 + cbuf(i,nj,ip)
            end do
            cons(1,1,kk,ip)   = base(1,1,kk,ip)   + rbuf(1,1,ip,bc)         + cbuf(1,1,ip)
            cons(1,nj,kk,ip)  = base(1,nj,kk,ip)  + rbuf(1,nj-1,ip,bc)      + cbuf(1,nj,ip)
            cons(ni,nj,kk,ip) = base(ni,nj,kk,ip) + rbuf(ni-1,nj-1,ip,bc)   + cbuf(ni,nj,ip)
            cons(ni,1,kk,ip)  = base(ni,1,kk,ip)  + rbuf(ni-1,1,ip,bc)      + cbuf(ni,1,ip)
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
! The ten production kernels, all branch-free straight-line compositions:
!   scree_plain      rk_plain       fine_term + scatter         (multigrid off)
!   scree_mg_noirs   rk_mg_noirs    engine(mg_smooth_noop) + cascade scatter
!   scree_mg_irs     rk_mg_irs      engine(smooth_residual_tri_tiled) + ditto
!   scree_mgpwc_noirs rk_mgpwc_noirs restrict(noop) + collapse + pwc scatter
!   scree_mgpwc_irs  rk_mgpwc_irs   restrict(tri_tiled) + collapse + ditto
! scree wrappers form q in store (scree_form_q) and roll the history
! (scree_roll); rk wrappers pass residual as q. The cascade wrappers scatter
! through the fused mg_prolong2x_fine_scatter and the piecewise-constant ones
! through mg_pwc_fine_scatter, scree frozen-in-place (base = cons) and rk off
! the sub-stage snapshot (base = snapshot); the multigrid-off pair still
! materialises a full-volume increment.
!
! Phase 1 -- the hierarchical restriction and the coarse timestep -- is shared
! by both prolongations and lives in mg_restrict_levels. The two differ only in
! what they then do with corr_all: mg_coarse_correction cascades it through
! factor-2 trilinear hops into acc0, and mg_collapse_pwc sums it in place onto
! its own finest slot for a plain injection read. See
! docs/dev/plan_piecewise_constant_mgrid.md.
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
! the final hop reaches the fine grid (fused with the fine term). This is a
! genuine operator change from a direct factor-b prolong -- cascaded factor-2
! trilinear interpolations are not equal to it.
!
! The coarse->coarse hops target cell centres; the final hop targets the fine
! NODES directly, which is where the correction is actually applied. That last
! hop therefore carries node-shaped weights, which is the one place
! mg_weight_offsets departs from the plain cell formula.
! ============================================================================
subroutine mg_restrict_levels(q, dt_vol, vol, scale, fmgrid, expon_mgrid, &
        sf_irs, n_levels, dtblk, rawbuf, sdt, sv, &
        corr_all, cres, triw, smoother, &
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
    real, intent(inout) :: dtblk(nc1i, nc1j, nc1k)
    real, intent(inout) :: rawbuf(nc1i, nc1j, nc1k, np)
    ! sdt accumulates sum(vol/dt_vol) and sv accumulates sum(vol), so that
    ! dtblk = sv/sdt is the volume-weighted harmonic mean of dt_vol.
    real, intent(inout) :: sdt(nc1i, nc1j, nc1k)
    real, intent(inout) :: sv (nc1i, nc1j, nc1k)
    real, intent(inout) :: corr_all(n_corr)
    real, intent(inout) :: cres(n_res)
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

    ! Coarsest-first packed geometry for corr_all (the cascade seeds at slot 1,
    ! and mg_collapse_pwc accumulates in the same direction).
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

        call mg_gather_corner(cres, rawbuf, nc1i, nc1j, nc1k, nib, njb, nkb, np)
        call smoother(cres(1:cnt), sf_irs, &
                      triw(1:2*(nib+njb+nkb)), nib+1, njb+1, nkb+1)
        call mg_scale_corr(corr_all(offc(slot)+1), cres, dtblk, coef, &
                           nc1i, nc1j, nc1k, nib, njb, nkb, np)
    end do

end subroutine mg_restrict_levels


! ============================================================================
! Phase 2, cascaded: the trilinear prolongation, unchanged in arithmetic from
! when it shared a body with Phase 1. It is a separate block for two reasons.
! It is the half the piecewise-constant path replaces, so the split is the
! seam the two schemes meet at; and f2py cannot forward a dummy PROCEDURE
! through a second call level -- it infers a callback's signature from the
! `call smoother(...)` it can see, so an intermediate that only passes the
! smoother on generates a broken wrapper. The smoother therefore reaches
! mg_restrict_levels from the production kernel directly.
! ============================================================================
subroutine mg_cascade_prolong(corr_all, acc0, acc1, aplane, bb, n_levels, &
        ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, &
        pwi, pwj, pwk, n_wi, n_wj, n_wk)

    implicit none
    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k, n_corr
    real, intent(inout) :: corr_all(n_corr)
    real, intent(inout) :: acc0(nc1i*nc1j*nc1k*np)
    real, intent(inout) :: acc1(nc1i*nc1j*nc1k*np)
    real, intent(inout) :: aplane(ni, nc1j)
    real, intent(inout) :: bb(ni, nj, nc1k, np)
    ! The block's packed prolongation weights, every hop of them. Static
    ! geometry, so they are NOT arena scratch: ember.block.Block.weight_mgrid
    ! caches them keyed on x/r/t, and this reads the hops it needs.
    integer, intent(in) :: n_wi, n_wj, n_wk
    real,    intent(in) :: pwi(n_wi), pwj(n_wj), pwk(n_wk)

    integer :: lvl, b, o
    integer, parameter :: MG_LEVELS_MAX = 3
    integer :: dib(MG_LEVELS_MAX), djb(MG_LEVELS_MAX), dkb(MG_LEVELS_MAX)
    integer :: offc(MG_LEVELS_MAX)
    integer :: offwi(MG_LEVELS_MAX), offwj(MG_LEVELS_MAX), offwk(MG_LEVELS_MAX)
    integer :: nci, ncj, nck, cur_i, cur_j, cur_k
    logical :: in0

    if (n_levels > MG_LEVELS_MAX) stop 'mg_cascade_prolong: n_levels > MG_LEVELS_MAX'

    ! Weight hops are numbered from the fine grid down (hop 1 targets the fine
    ! grid's nodes), so the cascade hop onto level lvl -- whose target is
    ! (ni-1)/2**(n_levels-lvl+1) -- is hop n_levels-lvl+2. Hop 1 is the final
    ! hop, and the caller makes that one.
    call mg_weight_offsets(ni, nj, nk, n_levels, offwi, offwj, offwk)

    ! Coarsest-first packed geometry for corr_all, as mg_restrict_levels built it.
    o = 0
    do lvl = 1, n_levels
        b = 2**(n_levels - lvl + 1)
        dib(lvl) = (ni-1)/b
        djb(lvl) = (nj-1)/b
        dkb(lvl) = (nk-1)/b
        offc(lvl) = o
        o = o + dib(lvl)*djb(lvl)*dkb(lvl)*np
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
                                  aplane, bb, ni-1, nj-1, nc1k, &
                                  pwi(offwi(n_levels-lvl+2)+1), &
                                  pwj(offwj(n_levels-lvl+2)+1), &
                                  pwk(offwk(n_levels-lvl+2)+1))
        else
            call mg_copy(corr_all(offc(lvl)+1), acc0, dib(lvl)*djb(lvl)*dkb(lvl)*np)
            call mg_prolong2x_acc(acc1, nci, ncj, nck, acc0, &
                                  dib(lvl), djb(lvl), dkb(lvl), np, &
                                  aplane, bb, ni-1, nj-1, nc1k, &
                                  pwi(offwi(n_levels-lvl+2)+1), &
                                  pwj(offwj(n_levels-lvl+2)+1), &
                                  pwk(offwk(n_levels-lvl+2)+1))
        end if
        in0 = .not. in0
        cur_i = dib(lvl)
        cur_j = djb(lvl)
        cur_k = dkb(lvl)
    end do
    ! Leaves the finest-coarse correction in acc0 at (nc1i,nc1j,nc1k). The final
    ! factor-2 hop is done by the caller so that both schemes can fuse it with
    ! the fine term's cell->node scatter (mg_prolong2x_fine_scatter); that hop
    ! lands straight on the nodes, so it never visits a fine cell centre.
end subroutine mg_cascade_prolong


! Accumulate a coarse field onto the next finer coarse level by INJECTION:
! every one of the eight fine cells under a coarse cell takes that cell's value,
! unaltered. The whole of the piecewise-constant prolongation is this and the
! identical read in mg_pwc_fine_scatter.
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
! disjoint slot, packed coarsest first, so no accumulator is needed at all --
! the cascade's acc0/acc1 ping-pong has no counterpart here. Working coarsest
! first, each slot gains the injected total of the one above it, so the last
! slot ends holding sum_l inject_l(corr_l) and the fine grid reads that.
!
! Cost is 1 + 1/8 + 1/64 = 1.14 traversals of the level-1 grid, about 0.14
! fine-cell-equivalents, against the cascade's full-size hops.
!
! The two slots handed to each mg_inject_acc call are non-overlapping sections
! of one array (offc is strictly increasing and slots are exactly their own
! size), and every target element is written once, so the aliasing is only
! apparent.
subroutine mg_collapse_pwc(corr_all, n_levels, ni, nj, nk, np, n_corr)
    implicit none
    integer, intent(in) :: n_levels, ni, nj, nk, np, n_corr
    real, intent(inout) :: corr_all(n_corr)
    integer, parameter :: MG_LEVELS_MAX = 3
    integer :: dib(MG_LEVELS_MAX), djb(MG_LEVELS_MAX), dkb(MG_LEVELS_MAX)
    integer :: offc(MG_LEVELS_MAX)
    integer :: lvl, b, o

    if (n_levels > MG_LEVELS_MAX) stop 'mg_collapse_pwc: n_levels > MG_LEVELS_MAX'

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
end subroutine mg_collapse_pwc


! Fused fine term + injected coarse correction + cell->node scatter: the
! piecewise-constant counterpart of mg_prolong2x_fine_scatter, and the only
! place the fine grid is touched by the correction.
!
! Under injection the correction is a CELL quantity, so unlike the cascade
! there are not two halves landing at the nodes by different routes. The
! correction is added into the fine cell increment alongside the fine term and
! both ride the one scatter, which is why aplane/bb/cbuf and the node-targeted
! weights have no counterpart here.
!
! Cell-averaging the correction costs nothing, which is the whole argument for
! doing it: within a coarse block the correction is constant, so the scatter --
! a partition of unity everywhere, interior 1/8 of 8 cells through to corners
! 1 of 1 -- reproduces it exactly. The two differ only at block faces, where
! the node takes the mean of the two adjoining blocks' corrections. That is a
! one-cell smoothing of the staircase applied exactly where the staircase is,
! not the first-order clustering error that made the cascade target nodes.
!
! The rolling two-plane rbuf and the emit stencils are carried over unchanged
! (term for term and in the same summation order as cell_to_node_generic), so
! the increment is still never materialised full-volume.
!
! src((i+1)/2, ...) repeats each coarse value across two fine cells. It reads
! from an array an eighth the size, resident in cache, so even scalarised it is
! cheap against the dt_vol and q streams it is added to.
subroutine mg_pwc_fine_scatter(src, nci, ncj, nck, base, cons, &
        scale, dt_vol, q, ni, nj, nk, np, rbuf)
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
    integer :: i, j, ip, kc, kb, jb, cur, prev, sw

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
        ! Both ends emit two node planes in one pass.
        if (kc == 1)    call emit_kbnd(cur, 1)
        if (kc >= 2)    call emit_kint(prev, cur, kc)
        if (kc == nk-1) call emit_kbnd(cur, nk)
        sw = cur; cur = prev; prev = sw
    end do

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

end subroutine mg_pwc_fine_scatter


! ============================================================================
! The ten production kernels. Each is a branch-free straight-line composition of
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
        fmgrid, expon_mgrid, sf_irs, n_levels, rbuf, cbuf, dtblk, aplane, bb, rawbuf, sdt, sv, &
        corr_all, acc0, acc1, cres, triw, &
        ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri, &
        pwi, pwj, pwk, n_wi, n_wj, n_wk)
    implicit none
    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k
    integer, intent(in) :: n_corr, n_res, n_tri
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real,    intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: cfl, fmgrid, expon_mgrid, sf_irs
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: store(ni-1, nj-1, nk-1, np)   ! in: (dF/dt)_{n-1}; out: rolled to residual
    real, intent(inout) :: rbuf(ni-1, nj-1, np, 2)
    real, intent(inout) :: cbuf(ni, nj, np)
    real, intent(inout) :: dtblk(nc1i, nc1j, nc1k)
    real, intent(inout) :: aplane(ni, nc1j)
    real, intent(inout) :: bb(ni, nj, nc1k, np)
    real, intent(inout) :: rawbuf(nc1i, nc1j, nc1k, np)
    real, intent(inout) :: sdt(nc1i, nc1j, nc1k)
    real, intent(inout) :: sv (nc1i, nc1j, nc1k)
    real, intent(inout) :: corr_all(n_corr)
    real, intent(inout) :: acc0(nc1i*nc1j*nc1k*np)
    real, intent(inout) :: acc1(nc1i*nc1j*nc1k*np)
    real, intent(inout) :: cres(n_res)
    real, intent(inout) :: triw(n_tri)
    integer, intent(in) :: n_wi, n_wj, n_wk
    real,    intent(in) :: pwi(n_wi), pwj(n_wj), pwk(n_wk)
    external :: smooth_residual_tri_tiled

    call scree_form_q(store, residual, ni, nj, nk, np)
    call mg_restrict_levels(store, dt_vol, vol, cfl, fmgrid, expon_mgrid, sf_irs, &
                       n_levels, dtblk, rawbuf, sdt, sv, corr_all, cres, triw, smooth_residual_tri_tiled, &
                       ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    call mg_cascade_prolong(corr_all, acc0, acc1, aplane, bb, n_levels, &
                       ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, &
                       pwi, pwj, pwk, n_wi, n_wj, n_wk)
    call mg_prolong2x_fine_scatter(acc0, nc1i, nc1j, nc1k, cons, cons, &
                           cfl, dt_vol, store, ni, nj, nk, np, &
                           aplane, bb, cbuf, rbuf, nc1j, nc1k, &
                           pwi, pwj, pwk)
    call scree_roll(residual, store, ni, nj, nk, np)
end subroutine scree_mg_irs


! scree, multigrid on, no smoothing.
subroutine scree_mg_noirs(cons, residual, store, dt_vol, vol, cfl, &
        fmgrid, expon_mgrid, sf_irs, n_levels, rbuf, cbuf, dtblk, aplane, bb, rawbuf, sdt, sv, &
        corr_all, acc0, acc1, cres, triw, &
        ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri, &
        pwi, pwj, pwk, n_wi, n_wj, n_wk)
    implicit none
    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k
    integer, intent(in) :: n_corr, n_res, n_tri
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real,    intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: cfl, fmgrid, expon_mgrid, sf_irs
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: store(ni-1, nj-1, nk-1, np)   ! in: (dF/dt)_{n-1}; out: rolled to residual
    real, intent(inout) :: rbuf(ni-1, nj-1, np, 2)
    real, intent(inout) :: cbuf(ni, nj, np)
    real, intent(inout) :: dtblk(nc1i, nc1j, nc1k)
    real, intent(inout) :: aplane(ni, nc1j)
    real, intent(inout) :: bb(ni, nj, nc1k, np)
    real, intent(inout) :: rawbuf(nc1i, nc1j, nc1k, np)
    real, intent(inout) :: sdt(nc1i, nc1j, nc1k)
    real, intent(inout) :: sv (nc1i, nc1j, nc1k)
    real, intent(inout) :: corr_all(n_corr)
    real, intent(inout) :: acc0(nc1i*nc1j*nc1k*np)
    real, intent(inout) :: acc1(nc1i*nc1j*nc1k*np)
    real, intent(inout) :: cres(n_res)
    real, intent(inout) :: triw(n_tri)
    integer, intent(in) :: n_wi, n_wj, n_wk
    real,    intent(in) :: pwi(n_wi), pwj(n_wj), pwk(n_wk)
    external :: mg_smooth_noop

    call scree_form_q(store, residual, ni, nj, nk, np)
    call mg_restrict_levels(store, dt_vol, vol, cfl, fmgrid, expon_mgrid, sf_irs, &
                       n_levels, dtblk, rawbuf, sdt, sv, corr_all, cres, triw, mg_smooth_noop, &
                       ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    call mg_cascade_prolong(corr_all, acc0, acc1, aplane, bb, n_levels, &
                       ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, &
                       pwi, pwj, pwk, n_wi, n_wj, n_wk)
    call mg_prolong2x_fine_scatter(acc0, nc1i, nc1j, nc1k, cons, cons, &
                           cfl, dt_vol, store, ni, nj, nk, np, &
                           aplane, bb, cbuf, rbuf, nc1j, nc1k, &
                           pwi, pwj, pwk)
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
        alpha, cfl, fmgrid, expon_mgrid, sf_irs, n_levels, rbuf, cbuf, dtblk, aplane, bb, &
        rawbuf, sdt, sv, corr_all, acc0, acc1, cres, triw, &
        ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri, &
        pwi, pwj, pwk, n_wi, n_wj, n_wk)
    implicit none
    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k
    integer, intent(in) :: n_corr, n_res, n_tri
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real,    intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: alpha, cfl, fmgrid, expon_mgrid, sf_irs
    real,    intent(in) :: snapshot(ni, nj, nk, np)
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: rbuf(ni-1, nj-1, np, 2)
    real, intent(inout) :: cbuf(ni, nj, np)
    real, intent(inout) :: dtblk(nc1i, nc1j, nc1k)
    real, intent(inout) :: aplane(ni, nc1j)
    real, intent(inout) :: bb(ni, nj, nc1k, np)
    real, intent(inout) :: rawbuf(nc1i, nc1j, nc1k, np)
    real, intent(inout) :: sdt(nc1i, nc1j, nc1k)
    real, intent(inout) :: sv (nc1i, nc1j, nc1k)
    real, intent(inout) :: corr_all(n_corr)
    real, intent(inout) :: acc0(nc1i*nc1j*nc1k*np)
    real, intent(inout) :: acc1(nc1i*nc1j*nc1k*np)
    real, intent(inout) :: cres(n_res)
    real, intent(inout) :: triw(n_tri)
    integer, intent(in) :: n_wi, n_wj, n_wk
    real,    intent(in) :: pwi(n_wi), pwj(n_wj), pwk(n_wk)
    external :: smooth_residual_tri_tiled

    call mg_restrict_levels(residual, dt_vol, vol, alpha*cfl, fmgrid, expon_mgrid, &
                       sf_irs, n_levels, dtblk, rawbuf, sdt, sv, corr_all, cres, triw, &
                       smooth_residual_tri_tiled, &
                       ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    call mg_cascade_prolong(corr_all, acc0, acc1, aplane, bb, n_levels, &
                       ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, &
                       pwi, pwj, pwk, n_wi, n_wj, n_wk)
    call mg_prolong2x_fine_scatter(acc0, nc1i, nc1j, nc1k, snapshot, cons, &
                       alpha*cfl, dt_vol, residual, ni, nj, nk, np, &
                       aplane, bb, cbuf, rbuf, nc1j, nc1k, &
                       pwi, pwj, pwk)
end subroutine rk_mg_irs


! RK stage, multigrid on, no smoothing. q = residual (passed directly).
subroutine rk_mg_noirs(cons, snapshot, residual, dt_vol, vol, &
        alpha, cfl, fmgrid, expon_mgrid, sf_irs, n_levels, rbuf, cbuf, dtblk, aplane, bb, &
        rawbuf, sdt, sv, corr_all, acc0, acc1, cres, triw, &
        ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri, &
        pwi, pwj, pwk, n_wi, n_wj, n_wk)
    implicit none
    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k
    integer, intent(in) :: n_corr, n_res, n_tri
    real,    intent(in) :: residual(ni-1, nj-1, nk-1, np)
    real,    intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: vol(ni-1, nj-1, nk-1)
    real,    intent(in) :: alpha, cfl, fmgrid, expon_mgrid, sf_irs
    real,    intent(in) :: snapshot(ni, nj, nk, np)
    real, intent(inout) :: cons(ni, nj, nk, np)
    real, intent(inout) :: rbuf(ni-1, nj-1, np, 2)
    real, intent(inout) :: cbuf(ni, nj, np)
    real, intent(inout) :: dtblk(nc1i, nc1j, nc1k)
    real, intent(inout) :: aplane(ni, nc1j)
    real, intent(inout) :: bb(ni, nj, nc1k, np)
    real, intent(inout) :: rawbuf(nc1i, nc1j, nc1k, np)
    real, intent(inout) :: sdt(nc1i, nc1j, nc1k)
    real, intent(inout) :: sv (nc1i, nc1j, nc1k)
    real, intent(inout) :: corr_all(n_corr)
    real, intent(inout) :: acc0(nc1i*nc1j*nc1k*np)
    real, intent(inout) :: acc1(nc1i*nc1j*nc1k*np)
    real, intent(inout) :: cres(n_res)
    real, intent(inout) :: triw(n_tri)
    integer, intent(in) :: n_wi, n_wj, n_wk
    real,    intent(in) :: pwi(n_wi), pwj(n_wj), pwk(n_wk)
    external :: mg_smooth_noop

    call mg_restrict_levels(residual, dt_vol, vol, alpha*cfl, fmgrid, expon_mgrid, &
                       sf_irs, n_levels, dtblk, rawbuf, sdt, sv, corr_all, cres, triw, &
                       mg_smooth_noop, &
                       ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    call mg_cascade_prolong(corr_all, acc0, acc1, aplane, bb, n_levels, &
                       ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, &
                       pwi, pwj, pwk, n_wi, n_wj, n_wk)
    call mg_prolong2x_fine_scatter(acc0, nc1i, nc1j, nc1k, snapshot, cons, &
                       alpha*cfl, dt_vol, residual, ni, nj, nk, np, &
                       aplane, bb, cbuf, rbuf, nc1j, nc1k, &
                       pwi, pwj, pwk)
end subroutine rk_mg_noirs


! ============================================================================
! The four piecewise-constant wrappers. Same four configurations as the cascade
! pair above (scree/rk x IRS/no-IRS), selected by ember.solver from
! Solver.mgrid_pwc, and each a three-call straight line:
!
!   mg_restrict_levels  ->  mg_collapse_pwc  ->  mg_pwc_fine_scatter
!
! Phase 1 is shared with the cascade verbatim. What differs is everything
! after it: no aplane/bb/cbuf, no acc0/acc1 ping-pong, and no prolongation
! weights at all, so these take seven arena buffers where the cascade takes
! twelve and never touch ember.block.Block.weight_mgrid.
!
! mg_restrict_levels packs corr_all coarsest-first, so the finest coarse level
! -- the one mg_collapse_pwc leaves the total in, and the one the fine grid
! reads -- is the LAST slot, at offset n_corr - nc1i*nc1j*nc1k*np.
! ============================================================================


! scree, piecewise-constant multigrid, coarse-level IRS.
subroutine scree_mgpwc_irs(cons, residual, store, dt_vol, vol, cfl, &
        fmgrid, expon_mgrid, sf_irs, n_levels, rbuf, dtblk, rawbuf, sdt, sv, &
        corr_all, cres, triw, &
        ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    implicit none
    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k
    integer, intent(in) :: n_corr, n_res, n_tri
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
    real, intent(inout) :: cres(n_res)
    real, intent(inout) :: triw(n_tri)
    external :: smooth_residual_tri_tiled

    call scree_form_q(store, residual, ni, nj, nk, np)
    call mg_restrict_levels(store, dt_vol, vol, cfl, fmgrid, expon_mgrid, sf_irs, &
                       n_levels, dtblk, rawbuf, sdt, sv, corr_all, cres, triw, &
                       smooth_residual_tri_tiled, &
                       ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    call mg_collapse_pwc(corr_all, n_levels, ni, nj, nk, np, n_corr)
    call mg_pwc_fine_scatter(corr_all(n_corr - nc1i*nc1j*nc1k*np + 1), &
                       nc1i, nc1j, nc1k, cons, cons, &
                       cfl, dt_vol, store, ni, nj, nk, np, rbuf)
    call scree_roll(residual, store, ni, nj, nk, np)
end subroutine scree_mgpwc_irs


! scree, piecewise-constant multigrid, no smoothing.
subroutine scree_mgpwc_noirs(cons, residual, store, dt_vol, vol, cfl, &
        fmgrid, expon_mgrid, sf_irs, n_levels, rbuf, dtblk, rawbuf, sdt, sv, &
        corr_all, cres, triw, &
        ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    implicit none
    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k
    integer, intent(in) :: n_corr, n_res, n_tri
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
    real, intent(inout) :: cres(n_res)
    real, intent(inout) :: triw(n_tri)
    external :: mg_smooth_noop

    call scree_form_q(store, residual, ni, nj, nk, np)
    call mg_restrict_levels(store, dt_vol, vol, cfl, fmgrid, expon_mgrid, sf_irs, &
                       n_levels, dtblk, rawbuf, sdt, sv, corr_all, cres, triw, &
                       mg_smooth_noop, &
                       ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    call mg_collapse_pwc(corr_all, n_levels, ni, nj, nk, np, n_corr)
    call mg_pwc_fine_scatter(corr_all(n_corr - nc1i*nc1j*nc1k*np + 1), &
                       nc1i, nc1j, nc1k, cons, cons, &
                       cfl, dt_vol, store, ni, nj, nk, np, rbuf)
    call scree_roll(residual, store, ni, nj, nk, np)
end subroutine scree_mgpwc_noirs


! RK stage, piecewise-constant multigrid, coarse-level IRS. q = residual.
subroutine rk_mgpwc_irs(cons, snapshot, residual, dt_vol, vol, &
        alpha, cfl, fmgrid, expon_mgrid, sf_irs, n_levels, rbuf, dtblk, &
        rawbuf, sdt, sv, corr_all, cres, triw, &
        ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    implicit none
    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k
    integer, intent(in) :: n_corr, n_res, n_tri
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
    real, intent(inout) :: cres(n_res)
    real, intent(inout) :: triw(n_tri)
    external :: smooth_residual_tri_tiled

    call mg_restrict_levels(residual, dt_vol, vol, alpha*cfl, fmgrid, expon_mgrid, &
                       sf_irs, n_levels, dtblk, rawbuf, sdt, sv, corr_all, cres, triw, &
                       smooth_residual_tri_tiled, &
                       ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    call mg_collapse_pwc(corr_all, n_levels, ni, nj, nk, np, n_corr)
    call mg_pwc_fine_scatter(corr_all(n_corr - nc1i*nc1j*nc1k*np + 1), &
                       nc1i, nc1j, nc1k, snapshot, cons, &
                       alpha*cfl, dt_vol, residual, ni, nj, nk, np, rbuf)
end subroutine rk_mgpwc_irs


! RK stage, piecewise-constant multigrid, no smoothing. q = residual.
subroutine rk_mgpwc_noirs(cons, snapshot, residual, dt_vol, vol, &
        alpha, cfl, fmgrid, expon_mgrid, sf_irs, n_levels, rbuf, dtblk, &
        rawbuf, sdt, sv, corr_all, cres, triw, &
        ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    implicit none
    integer, intent(in) :: ni, nj, nk, np, n_levels, nc1i, nc1j, nc1k
    integer, intent(in) :: n_corr, n_res, n_tri
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
    real, intent(inout) :: cres(n_res)
    real, intent(inout) :: triw(n_tri)
    external :: mg_smooth_noop

    call mg_restrict_levels(residual, dt_vol, vol, alpha*cfl, fmgrid, expon_mgrid, &
                       sf_irs, n_levels, dtblk, rawbuf, sdt, sv, corr_all, cres, triw, &
                       mg_smooth_noop, &
                       ni, nj, nk, np, nc1i, nc1j, nc1k, n_corr, n_res, n_tri)
    call mg_collapse_pwc(corr_all, n_levels, ni, nj, nk, np, n_corr)
    call mg_pwc_fine_scatter(corr_all(n_corr - nc1i*nc1j*nc1k*np + 1), &
                       nc1i, nc1j, nc1k, snapshot, cons, &
                       alpha*cfl, dt_vol, residual, ni, nj, nk, np, rbuf)
end subroutine rk_mgpwc_noirs
