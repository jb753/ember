! =====================================================================
! set_residual_tiled -- a from-scratch rewrite of the inviscid residual,
! for benchmarking against production set_residual (residual.f90).
! Nothing calls this in production; it is exposed to f2py for the
! same-.so A/B in tools/bench_residual_variants.py.
!
! Written to be plain Fortran: fixed-shape tile arrays, simple nested
! loops, no hand-unrolled face corners, no scalarized pm1..pm6/mf1..mf3,
! no rolling buffers with alternating slot indices, no slab carry state.
!
! The structural idea (docs/dev/viscous_kernels.md section 18):
!
!   1. Every face flow is built from the same NINE per-node quantities
!      averaged over the face's 4 corners:
!        qn = (Vx, Vr, r*Vt, ho, dP, r*dP, rho*Vx, rho*Vr, rho*Vt_rel)
!      Production recomputes these from the raw nodal fields at every
!      corner of every face -- ~12.1 corner-evaluations per cell across
!      the three direction sweeps. Here they are computed ONCE per node
!      into qn, then shared by all three directions.
!
!   2. A 4-corner average is a 2D box filter, and box filters separate:
!        i-face = avg_k(avg_j(q))
!        j-face = avg_k(avg_i(q))
!        k-face = avg_j(avg_i(q))
!      so avg_i, computed once, serves BOTH the j-faces and the k-faces.
!      Each averaging pass is a 2-point stencil along a single axis, so
!      every operand is unit-stride in i -- there is no "corner axis" to
!      vectorize over, hence nothing for ifort to gather (contrast
!      section 16, where accum_corners is ~45% vgatherdps).
!
!   3. Working set is held in L2 by tiling all three axes. A k-slab is
!      not enough: at ni=273, nj=65 one node plane is ~71 KB per
!      quantity, so nine of them plus partial averages exceed a 2 MB L2
!      for any kb. A compact 3D tile (e.g. 96x16x8 cells, ~1.5 MB) fits,
!      and re-reads each node only ~1.21x versus ~2.04x for production's
!      rolling planes -- better locality, not worse.
!
! Vectorization notes (the two measured failure modes this shape avoids):
!   - qn is written with the component index a LITERAL per statement, so
!     there is no indexed pm(:)/mf(:) accumulator for ifort to assume a
!     cross-iteration flow dependence on. That assumption is what left
!     the section 17 idiomatic rewrite with no SIMD at all.
!   - every averaging and contraction loop is i-innermost over
!     contiguous slices.
!
! Numerics: face values are mathematically identical to production but
! reassociated -- two 2-point averages instead of one 4-term sum, and the
! wall mask applied to the averaged mass flux rather than per corner
! (equivalent: the mask is constant over a face). So results differ by
! float32 rounding only, the same bounded class as sections 9 and 13.
! =====================================================================
module residual_tiled_helpers
    implicit none
    private
    public :: node_quantities, avg_along_i, avg_along_j, avg_along_k
    public :: face_flux, set_wall_row, diff_into_du

contains

    ! Nine per-node quantities over a node tile of (mi, mj, mk).
    ! The wall mask is deliberately NOT applied here: it weights the
    ! mass-flux components (7:9) only, and only on boundary faces, so it
    ! is folded into the face contraction. That keeps this loop
    ! branch-free with nine unit-stride output streams.
    subroutine node_quantities(vx, vr, vt, ho, P, P_offset, r, cons, Omega, &
                               qn, i0, j0, k0, mi, mj, mk, ni, nj, nk)
        implicit none
        integer, intent(in) :: i0, j0, k0, mi, mj, mk, ni, nj, nk
        real, intent(in) :: vx(ni,nj,nk), vr(ni,nj,nk), vt(ni,nj,nk)
        real, intent(in) :: ho(ni,nj,nk), P(ni,nj,nk), r(ni,nj,nk)
        real, intent(in) :: cons(ni,nj,nk,5)
        real, intent(in) :: P_offset, Omega
        real, intent(inout) :: qn(mi, mj, mk, 9)

        integer :: i, j, k, gj, gk
        real :: dp, rr

        do k = 1, mk
            gk = k0 + k - 1
            do j = 1, mj
                gj = j0 + j - 1
                do i = 1, mi
                    dp = P(i0+i-1,gj,gk) - P_offset
                    rr = r(i0+i-1,gj,gk)
                    qn(i,j,k,1) = vx(i0+i-1,gj,gk)
                    qn(i,j,k,2) = vr(i0+i-1,gj,gk)
                    qn(i,j,k,3) = rr*vt(i0+i-1,gj,gk)
                    qn(i,j,k,4) = ho(i0+i-1,gj,gk)
                    qn(i,j,k,5) = dp
                    qn(i,j,k,6) = rr*dp
                    qn(i,j,k,7) = cons(i0+i-1,gj,gk,2)
                    qn(i,j,k,8) = cons(i0+i-1,gj,gk,3)
                    qn(i,j,k,9) = cons(i0+i-1,gj,gk,1) &
                                * (vt(i0+i-1,gj,gk) - Omega*rr)
                end do
            end do
        end do
    end subroutine node_quantities


    ! 2-point average along i: out(i,:,:,:) = 0.5*(in(i)+in(i+1)).
    ! Output has one fewer i entry. Unit-stride in i on both operands.
    subroutine avg_along_i(a, b, mi, mj, mk)
        implicit none
        integer, intent(in) :: mi, mj, mk
        real, intent(in) :: a(mi, mj, mk, 9)
        real, intent(inout) :: b(mi-1, mj, mk, 9)
        integer :: i, j, k, m
        do m = 1, 9
            do k = 1, mk
                do j = 1, mj
                    do i = 1, mi-1
                        b(i,j,k,m) = 0.5e0*(a(i,j,k,m) + a(i+1,j,k,m))
                    end do
                end do
            end do
        end do
    end subroutine avg_along_i


    ! 2-point average along j. Both operands are contiguous rows in i.
    subroutine avg_along_j(a, b, mi, mj, mk)
        implicit none
        integer, intent(in) :: mi, mj, mk
        real, intent(in) :: a(mi, mj, mk, 9)
        real, intent(inout) :: b(mi, mj-1, mk, 9)
        integer :: i, j, k, m
        do m = 1, 9
            do k = 1, mk
                do j = 1, mj-1
                    do i = 1, mi
                        b(i,j,k,m) = 0.5e0*(a(i,j,k,m) + a(i,j+1,k,m))
                    end do
                end do
            end do
        end do
    end subroutine avg_along_j


    ! 2-point average along k. Both operands are contiguous planes.
    subroutine avg_along_k(a, b, mi, mj, mk)
        implicit none
        integer, intent(in) :: mi, mj, mk
        real, intent(in) :: a(mi, mj, mk, 9)
        real, intent(inout) :: b(mi, mj, mk-1, 9)
        integer :: i, j, k, m
        do m = 1, 9
            do k = 1, mk-1
                do j = 1, mj
                    do i = 1, mi
                        b(i,j,k,m) = 0.5e0*(a(i,j,k,m) + a(i,j,k+1,m))
                    end do
                end do
            end do
        end do
    end subroutine avg_along_k


    ! Contract face-averaged quantities with face areas into the 5
    ! inviscid flows, for a whole (fi x fj x fk) block of faces:
    !   fl(1) = mdot
    !   fl(2) = pm1*mdot + pm5*dA1        (axial momentum)
    !   fl(3) = pm2*mdot + pm5*dA2        (radial momentum)
    !   fl(4) = pm3*mdot + pm6*dA3        (angular momentum)
    !   fl(5) = pm4*mdot + Omega*pm6*dA3  (energy)
    ! mdot = wfac*(mf1*dA1 + mf2*dA2 + mf3*dA3); scaling mdot by the
    ! (face-constant) wall factor is equivalent to masking each mf corner
    ! before averaging, and keeps the averaging passes branch-free.
    !
    ! dA is passed as the WHOLE face-area array with its true extents
    ! (na1,na2,na3) plus this tile's global origin (o1,o2,o3). Passing
    ! dA(1,i0,j0,k0) instead would be sequence association: the dummy's
    ! implied strides would be the tile's, not the array's, so every tile
    ! after the first would read the wrong elements. wfac is tile-shaped
    ! (it is built per tile) and is all ones off the domain boundary.
    subroutine face_flux(fa, dA, wfac, Omega, fl, fi, fj, fk, &
                         na1, na2, na3, o1, o2, o3)
        implicit none
        integer, intent(in) :: fi, fj, fk, na1, na2, na3, o1, o2, o3
        real, intent(in) :: fa(fi, fj, fk, 9)
        real, intent(in) :: dA(3, na1, na2, na3)
        real, intent(in) :: wfac(fi, fj, fk)
        real, intent(in) :: Omega
        real, intent(inout) :: fl(fi, fj, fk, 5)
        integer :: i, j, k
        real :: mdot, a1, a2, a3
        do k = 1, fk
            do j = 1, fj
                do i = 1, fi
                    a1 = dA(1, o1+i-1, o2+j-1, o3+k-1)
                    a2 = dA(2, o1+i-1, o2+j-1, o3+k-1)
                    a3 = dA(3, o1+i-1, o2+j-1, o3+k-1)
                    mdot = wfac(i,j,k)*(fa(i,j,k,7)*a1 + fa(i,j,k,8)*a2 &
                                      + fa(i,j,k,9)*a3)
                    fl(i,j,k,1) = mdot
                    fl(i,j,k,2) = fa(i,j,k,1)*mdot + fa(i,j,k,5)*a1
                    fl(i,j,k,3) = fa(i,j,k,2)*mdot + fa(i,j,k,5)*a2
                    fl(i,j,k,4) = fa(i,j,k,3)*mdot + fa(i,j,k,6)*a3
                    fl(i,j,k,5) = fa(i,j,k,4)*mdot + Omega*fa(i,j,k,6)*a3
                end do
            end do
        end do
    end subroutine face_flux

    ! Fill a face block's wall factors: 1.0 everywhere, then overwrite the
    ! single boundary layer when this tile touches the domain edge.
    ! `axis` selects which face-block axis carries the boundary (1=i, 2=j,
    ! 3=k). wlo/whi are the full-size wall-mask planes for that direction
    ! (their two axes are the *other* two directions), read at the tile's
    ! global offsets p0/q0 -- so no gathering or repacking is needed.
    subroutine set_wall_row(wf, fi, fj, fk, axis, lo, hi, &
                            wlo, whi, np, nq, p0, q0)
        implicit none
        integer, intent(in) :: fi, fj, fk, axis, np, nq, p0, q0
        logical, intent(in) :: lo, hi
        real, intent(in) :: wlo(np, nq), whi(np, nq)
        real, intent(inout) :: wf(fi, fj, fk)
        integer :: i, j, k
        do k = 1, fk
            do j = 1, fj
                do i = 1, fi
                    wf(i,j,k) = 1.0e0
                end do
            end do
        end do
        if (axis == 1) then
            ! wall planes are (j,k)
            if (lo) then
                do k = 1, fk
                    do j = 1, fj
                        wf(1,j,k) = wlo(p0+j-1, q0+k-1)
                    end do
                end do
            end if
            if (hi) then
                do k = 1, fk
                    do j = 1, fj
                        wf(fi,j,k) = whi(p0+j-1, q0+k-1)
                    end do
                end do
            end if
        else if (axis == 2) then
            ! wall planes are (i,k)
            if (lo) then
                do k = 1, fk
                    do i = 1, fi
                        wf(i,1,k) = wlo(p0+i-1, q0+k-1)
                    end do
                end do
            end if
            if (hi) then
                do k = 1, fk
                    do i = 1, fi
                        wf(i,fj,k) = whi(p0+i-1, q0+k-1)
                    end do
                end do
            end if
        else
            ! wall planes are (i,j)
            if (lo) then
                do j = 1, fj
                    do i = 1, fi
                        wf(i,j,1) = wlo(p0+i-1, q0+j-1)
                    end do
                end do
            end if
            if (hi) then
                do j = 1, fj
                    do i = 1, fi
                        wf(i,j,fk) = whi(p0+i-1, q0+j-1)
                    end do
                end do
            end if
        end if
    end subroutine set_wall_row


    ! Difference a face-flow block along `axis` into the tile's cells of
    ! dU: dU(cell) += fl(face) - fl(face+1). `first` selects assignment
    ! (with f_body folded in) rather than accumulation, so dU is written
    ! once and then read-modify-written twice, as production does.
    subroutine diff_into_du(fl, fi, fj, fk, axis, dU, f_body, first, &
                            i0, j0, k0, ci, cj, ck, ni, nj, nk)
        implicit none
        integer, intent(in) :: fi, fj, fk, axis, i0, j0, k0, ci, cj, ck
        integer, intent(in) :: ni, nj, nk
        logical, intent(in) :: first
        real, intent(in) :: fl(fi, fj, fk, 5)
        real, intent(in) :: f_body(ni-1, nj-1, nk-1, 5)
        real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
        integer :: i, j, k, m, di, dj, dk

        di = 0
        dj = 0
        dk = 0
        if (axis == 1) then
            di = 1
        else if (axis == 2) then
            dj = 1
        else
            dk = 1
        end if

        if (first) then
            do m = 1, 5
                do k = 1, ck
                    do j = 1, cj
                        do i = 1, ci
                            dU(i0+i-1, j0+j-1, k0+k-1, m) = &
                                fl(i,j,k,m) - fl(i+di,j+dj,k+dk,m) &
                                + f_body(i0+i-1, j0+j-1, k0+k-1, m)
                        end do
                    end do
                end do
            end do
        else
            do m = 1, 5
                do k = 1, ck
                    do j = 1, cj
                        do i = 1, ci
                            dU(i0+i-1, j0+j-1, k0+k-1, m) = &
                                dU(i0+i-1, j0+j-1, k0+k-1, m) &
                                + fl(i,j,k,m) - fl(i+di,j+dj,k+dk,m)
                        end do
                    end do
                end do
            end do
        end if
    end subroutine diff_into_du

end module residual_tiled_helpers


! =====================================================================
! Tiled inviscid residual. Sweeps 3D tiles of IB x JB x KB cells; per
! tile, precomputes the nine node quantities once, forms the three face
! averages by separable 2-point passes, contracts each with its face
! area, and differences them into dU.
!
! Every scratch array is a fixed-shape dummy carved by the caller from
! block.tau_q_halo. Sized for the largest tile:
!   qn   (IB+1, JB+1, KB+1, 9)   node quantities
!   ai   (IB,   JB+1, KB+1, 9)   avg along i   (serves j- and k-faces)
!   t1   (IB+1, JB,   KB+1, 9)   avg along j   (-> i-faces after avg_k)
!   fa   (IB+1, JB+1, KB+1, 9)   face averages (largest face block)
!   fl   (IB+1, JB+1, KB+1, 5)   face flows
!   wf   (IB+1, JB+1, KB+1)      per-face wall factors
! =====================================================================
subroutine set_residual_tiled( &
    cons, P, P_offset, &
    r, Omega, dAi, dAj, dAk, &
    f_body, &
    dU, &
    vx, vr, vt, ho, &
    qn, ai, t1, fa, fl, wf, &
    walli1, wallj1, wallk1, &
    wallni, wallnj, wallnk, &
    i_cusp_start, i_cusp_end, &
    ib, jb, kbt, ni, nj, nk &
    )

    use residual_tiled_helpers
    use residual_helpers, only: correct_cusp_kface_du

    implicit none

    integer, intent(in) :: ib, jb, kbt, ni, nj, nk
    real, intent(in) :: cons(ni, nj, nk, 5)
    real, intent(in) :: P(ni, nj, nk)
    real, intent(in) :: P_offset
    real, intent(in) :: r(ni, nj, nk)
    real, intent(in) :: Omega
    real, intent(in) :: dAi(3, ni, nj-1, nk-1)
    real, intent(in) :: dAj(3, ni-1, nj, nk-1)
    real, intent(in) :: dAk(3, ni-1, nj-1, nk)
    real, intent(in) :: f_body(ni-1, nj-1, nk-1, 5)
    real, intent(in) :: vx(ni, nj, nk), vr(ni, nj, nk)
    real, intent(in) :: vt(ni, nj, nk), ho(ni, nj, nk)
    real, intent(in) :: walli1(nj-1, nk-1), wallni(nj-1, nk-1)
    real, intent(in) :: wallj1(ni-1, nk-1), wallnj(ni-1, nk-1)
    real, intent(in) :: wallk1(ni-1, nj-1), wallnk(ni-1, nj-1)
    integer, intent(in) :: i_cusp_start, i_cusp_end
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    ! Scratch is declared FLAT and reshaped by each helper to the current
    ! tile's runtime extents. Edge tiles are short (nj-1=64 and nk-1=56 do
    ! not divide evenly by every tile size), so the helpers must see the
    ! actual tile shape, not the maximum one -- otherwise the implied
    ! strides disagree with the data. Each buffer is sized by the caller
    ! for the largest tile.
    real, intent(inout) :: qn((ib+1)*(jb+1)*(kbt+1)*9)
    real, intent(inout) :: ai((ib+1)*(jb+1)*(kbt+1)*9)
    real, intent(inout) :: t1((ib+1)*(jb+1)*(kbt+1)*9)
    real, intent(inout) :: fa((ib+1)*(jb+1)*(kbt+1)*9)
    real, intent(inout) :: fl((ib+1)*(jb+1)*(kbt+1)*5)
    real, intent(inout) :: wf((ib+1)*(jb+1)*(kbt+1))

    integer :: i0, j0, k0, ci, cj, ck, mi, mj, mk

    ! Tile over cells. ci/cj/ck are this tile's cell counts; the node tile
    ! is one larger in each direction (mi/mj/mk).
    do k0 = 1, nk-1, kbt
    ck = min(kbt, nk-1 - k0 + 1)
    mk = ck + 1
    do j0 = 1, nj-1, jb
    cj = min(jb, nj-1 - j0 + 1)
    mj = cj + 1
    do i0 = 1, ni-1, ib
    ci = min(ib, ni-1 - i0 + 1)
    mi = ci + 1

        ! --- nine node quantities, once per node in this tile ---
        call node_quantities(vx, vr, vt, ho, P, P_offset, r, cons, Omega, &
                             qn, i0, j0, k0, mi, mj, mk, ni, nj, nk)

        ! --- avg along i: shared by the j-face and k-face averages ---
        call avg_along_i(qn, ai, mi, mj, mk)

        ! ================= i-faces: avg_k(avg_j(qn)) =================
        ! mi faces spanning i0..i0+ci, over cj x ck cells.
        call avg_along_j(qn, t1, mi, mj, mk)
        call avg_along_k(t1, fa, mi, cj, mk)
        call set_wall_row(wf, mi, cj, ck, 1, i0 == 1, i0+ci-1 == ni-1, &
                          walli1, wallni, nj-1, nk-1, j0, k0)
        call face_flux(fa, dAi, wf, Omega, fl, mi, cj, ck, &
                       ni, nj-1, nk-1, i0, j0, k0)
        call diff_into_du(fl, mi, cj, ck, 1, dU, f_body, .true., &
                          i0, j0, k0, ci, cj, ck, ni, nj, nk)

        ! ================= j-faces: avg_k(avg_i(qn)) =================
        ! mj faces spanning j0..j0+cj, over ci x ck cells.
        call avg_along_k(ai, fa, ci, mj, mk)
        call set_wall_row(wf, ci, mj, ck, 2, j0 == 1, j0+cj-1 == nj-1, &
                          wallj1, wallnj, ni-1, nk-1, i0, k0)
        call face_flux(fa, dAj, wf, Omega, fl, ci, mj, ck, &
                       ni-1, nj, nk-1, i0, j0, k0)
        call diff_into_du(fl, ci, mj, ck, 2, dU, f_body, .false., &
                          i0, j0, k0, ci, cj, ck, ni, nj, nk)

        ! ================= k-faces: avg_j(avg_i(qn)) =================
        ! mk faces spanning k0..k0+ck, over ci x cj cells.
        call avg_along_j(ai, fa, ci, mj, mk)
        call set_wall_row(wf, ci, cj, mk, 3, k0 == 1, k0+ck-1 == nk-1, &
                          wallk1, wallnk, ni-1, nj-1, i0, j0)
        call face_flux(fa, dAk, wf, Omega, fl, ci, cj, mk, &
                       ni-1, nj-1, nk, i0, j0, k0)
        call diff_into_du(fl, ci, cj, mk, 3, dU, f_body, .false., &
                          i0, j0, k0, ci, cj, ck, ni, nj, nk)

    end do
    end do
    end do

    ! Cusp seam: non-local in k, deferred O(surface) correction, shared
    ! verbatim with production. nk=2 unsupported (seam cells coincide).
    if (i_cusp_start > 0 .and. nk > 2) then
        call correct_cusp_kface_du(vx, vr, vt, ho, P, P_offset, r, cons, &
                                   Omega, dAk, wallk1, wallnk, dU, &
                                   i_cusp_start, i_cusp_end, ni, nj, nk)
    end if

end subroutine set_residual_tiled
