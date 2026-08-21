! TIMING CONTROLS ONLY -- WRONG ANSWERS BY CONSTRUCTION, NEVER GATE THEM.
!
! set_tau_q_soa split by CONSUMER, to price the tau/q split. The streams are
! in stage 1 (the corner averages and velocity gradients), not stage 2, so a
! split that only separates the tau and q stores buys nothing; these partition
! stage 1 as well:
!
!   tau pass  vol r cons1 mu Vx Vr Vt xlength dAi dAj dAk   11 streams
!   q pass    vol r T cp kappa dAi dAj dAk (+ mut)           9 streams
!   fused                                                    14 streams
!
! vol, r and the three dA arrays are read by both, so the sum of these two
! includes the double read a real split would pay. What it omits is passing
! mut from one to the other, which in a real split is a rolling plane.
!
subroutine set_tau_q_tau_only( &
    cons, T, mu, cp, kappa, Pr_turb, xlength, vol, dAi, dAj, dAk, &
    r, &
    Vx, Vr, Vt, &
    tau_cell, &
    q_cell, &
    mu_turb, &
    ni, nj, nk)

    use viscous_helpers
    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in) :: cons(ni, nj, nk, 5)
    real, intent(in) :: T(ni, nj, nk)
    real, intent(in) :: Pr_turb
    ! All three nodal: a real gas's viscosity, conductivity and specific heat
    ! are surfaces over the field, and freezing any of them at one state was
    ! worth as much as its whole spread over a fit box. Averaged to the cell
    ! below, like rho. The laminar Prandtl number is not among them -- it is
    ! the ratio of two of these, so passing it as well would be a second
    ! definition of the same thing.
    real, intent(in) :: mu(ni, nj, nk)
    real, intent(in) :: cp(ni, nj, nk)
    real, intent(in) :: kappa(ni, nj, nk)
    real, intent(in) :: xlength(ni-1, nj-1, nk-1)
    real, intent(in) :: vol(ni-1, nj-1, nk-1)
    real, intent(in) :: dAi(3, ni, nj-1, nk-1)
    real, intent(in) :: dAj(3, ni-1, nj, nk-1)
    real, intent(in) :: dAk(3, ni-1, nj-1, nk)
    real, intent(in) :: r(ni, nj, nk)
    real, intent(in) :: Vx(ni, nj, nk)
    real, intent(in) :: Vr(ni, nj, nk)
    real, intent(in) :: Vt(ni, nj, nk)
    real, intent(inout) :: tau_cell(ni+1, nj+1, nk+1, 6)
    real, intent(inout) :: q_cell(ni+1, nj+1, nk+1, 3)
    ! Cell-centred mixing-length turbulent viscosity, written at the cell's
    ! low-corner node (i,j,k). The final node in each axis is padding that is
    ! not written here and must not be read. intent(inout) so that padding is
    ! left untouched rather than becoming undefined.
    real, intent(inout) :: mu_turb(ni, nj, nk)

    integer :: i, j, k
    real :: visc_lim
    ! Row temps -- i is the contiguous (dim-1) axis, the SIMD lane index.
    real :: gVx(ni-1, 3), gVr(ni-1, 3), gVt(ni-1, 3)
    real :: vct(ni-1), rcr(ni-1), ivr(ni-1), rhoc(ni-1), muc(ni-1)
    real :: f1, f2, f3, f4, f5, f6, g1, g2, g3
    real :: t1, t2, t3, t4, t5, t6, w1, w2, w3, vm, mut, fac

    ! Unused here BY DESIGN -- this control is the tau half, so cp, kappa, T
    ! and Pr_turb are exactly the streams it exists to not read. One element
    ! each in a test that is never true keeps -Werror quiet without adding a
    ! stream: a couple of cache lines per call.
    if (cp(1,1,1) /= cp(1,1,1) .or. kappa(1,1,1) /= kappa(1,1,1) &
        .or. T(1,1,1) /= T(1,1,1) .or. Pr_turb /= Pr_turb) return

    do k = 1, nk-1
    do j = 1, nj-1
        ! Stage 1: velocity gradients + cell metrics, vectorizable over i.
        do i = 1, ni-1
            ivr(i) = 0.25e0 / vol(i,j,k)
            rcr(i) = 0.125e0 * (r(i,j,k)   + r(i+1,j,k)   + r(i,j+1,k)   + r(i+1,j+1,k) &
                              + r(i,j,k+1) + r(i+1,j,k+1) + r(i,j+1,k+1) + r(i+1,j+1,k+1))
            rhoc(i) = 0.125e0 * (cons(i,j,k,1)   + cons(i+1,j,k,1)   + cons(i,j+1,k,1)   + cons(i+1,j+1,k,1) &
                               + cons(i,j,k+1,1) + cons(i+1,j,k+1,1) + cons(i,j+1,k+1,1) + cons(i+1,j+1,k+1,1))
            muc(i) = 0.125e0 * (mu(i,j,k)   + mu(i+1,j,k)   + mu(i,j+1,k)   + mu(i+1,j+1,k) &
                              + mu(i,j,k+1) + mu(i+1,j,k+1) + mu(i,j+1,k+1) + mu(i+1,j+1,k+1))
            ! --- Vx ---
            f1 = Vx(i,j,k)+Vx(i,j+1,k)+Vx(i,j,k+1)+Vx(i,j+1,k+1)
            f2 = Vx(i+1,j,k)+Vx(i+1,j+1,k)+Vx(i+1,j,k+1)+Vx(i+1,j+1,k+1)
            f3 = Vx(i,j,k)+Vx(i+1,j,k)+Vx(i,j,k+1)+Vx(i+1,j,k+1)
            f4 = Vx(i,j+1,k)+Vx(i+1,j+1,k)+Vx(i,j+1,k+1)+Vx(i+1,j+1,k+1)
            f5 = Vx(i,j,k)+Vx(i+1,j,k)+Vx(i,j+1,k)+Vx(i+1,j+1,k)
            f6 = Vx(i,j,k+1)+Vx(i+1,j,k+1)+Vx(i,j+1,k+1)+Vx(i+1,j+1,k+1)
            g1 = -(f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k)-f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1))
            g2 = -(f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k)-f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))
            g3 = -(f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k)-f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1))
            gVx(i,1) = g1*ivr(i)
            gVx(i,3) = g3*ivr(i)
            gVx(i,2) = g2*ivr(i) - 0.125e0*(f1+f2)/rcr(i)
            ! --- Vr ---
            f1 = Vr(i,j,k)+Vr(i,j+1,k)+Vr(i,j,k+1)+Vr(i,j+1,k+1)
            f2 = Vr(i+1,j,k)+Vr(i+1,j+1,k)+Vr(i+1,j,k+1)+Vr(i+1,j+1,k+1)
            f3 = Vr(i,j,k)+Vr(i+1,j,k)+Vr(i,j,k+1)+Vr(i+1,j,k+1)
            f4 = Vr(i,j+1,k)+Vr(i+1,j+1,k)+Vr(i,j+1,k+1)+Vr(i+1,j+1,k+1)
            f5 = Vr(i,j,k)+Vr(i+1,j,k)+Vr(i,j+1,k)+Vr(i+1,j+1,k)
            f6 = Vr(i,j,k+1)+Vr(i+1,j,k+1)+Vr(i,j+1,k+1)+Vr(i+1,j+1,k+1)
            g1 = -(f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k)-f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1))
            g2 = -(f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k)-f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))
            g3 = -(f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k)-f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1))
            gVr(i,1) = g1*ivr(i)
            gVr(i,3) = g3*ivr(i)
            gVr(i,2) = g2*ivr(i) - 0.125e0*(f1+f2)/rcr(i)
            ! --- Vt ---
            f1 = Vt(i,j,k)+Vt(i,j+1,k)+Vt(i,j,k+1)+Vt(i,j+1,k+1)
            f2 = Vt(i+1,j,k)+Vt(i+1,j+1,k)+Vt(i+1,j,k+1)+Vt(i+1,j+1,k+1)
            f3 = Vt(i,j,k)+Vt(i+1,j,k)+Vt(i,j,k+1)+Vt(i+1,j,k+1)
            f4 = Vt(i,j+1,k)+Vt(i+1,j+1,k)+Vt(i,j+1,k+1)+Vt(i+1,j+1,k+1)
            f5 = Vt(i,j,k)+Vt(i+1,j,k)+Vt(i,j+1,k)+Vt(i+1,j+1,k)
            f6 = Vt(i,j,k+1)+Vt(i+1,j,k+1)+Vt(i,j+1,k+1)+Vt(i+1,j+1,k+1)
            vct(i) = (f1+f2)*0.125e0
            g1 = -(f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k)-f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1))
            g2 = -(f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k)-f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))
            g3 = -(f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k)-f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1))
            gVt(i,1) = g1*ivr(i)
            gVt(i,3) = g3*ivr(i)
            gVt(i,2) = g2*ivr(i) - 0.125e0*(f1+f2)/rcr(i)
        end do
        ! Stage 2: tau, mixing-length mu_turb, and q -- store with stride-1
        ! per-component writes; vectorizable over i.
        do i = 1, ni-1
            t1 = gVx(i,1)
            t2 = gVr(i,2)
            t3 = gVt(i,3)
            t4 = gVx(i,2) + gVr(i,1)
            t5 = gVx(i,3) + gVt(i,1)
            t6 = gVr(i,3) + gVt(i,2) - vct(i)/rcr(i)
            w1 = gVt(i,2) - gVr(i,3) + vct(i)/rcr(i)
            w2 = gVx(i,3) - gVt(i,1)
            w3 = gVr(i,1) - gVx(i,2)
            vm = sqrt(w1*w1 + w2*w2 + w3*w3)
            ! The max(0) is not physics -- mut is analytically confined to
            ! [0, visc_lim] already, since rhoc, xlength and vm are all
            ! non-negative and nothing here divides. It contains a gfortran 13
            ! codegen fault: built with the setup.py production flags (the
            ! raised --param=vect-max-version-for-alias-checks is what forces
            ! this loop to vectorize at all), gfortran 13.3 returns -inf from
            ! this line for about two thirds of cells whenever the vorticity
            ! sits at the float32 cancellation noise floor, i.e. a uniform flow
            ! with no real shear -- the chi = 90/270 duct cases in
            ! tests/test_nonreflecting_integration.py. The -inf then travels
            ! fac -> tau_cell -> F_body -> residual and NaNs the whole field
            ! inside one step. gfortran 14.2 is unaffected, as is any build at
            ! -O2, without fast-math, or at the default alias-check budget.
            ! Ubuntu 24.04 (the CI runner) ships gfortran 13.3, so this is a
            ! live target, not a historical one. The clamp costs one vmaxps
            ! and leaves the loop vectorized.
            visc_lim = 3000e0 * muc(i)
            mut = max(0.0e0, min(rhoc(i) * xlength(i,j,k) * vm, visc_lim))
            mu_turb(i,j,k) = mut
            fac = (muc(i) + mut) * 0.5e0
            tau_cell(i+1,j+1,k+1,1) = t1*fac
            tau_cell(i+1,j+1,k+1,2) = t2*fac
            tau_cell(i+1,j+1,k+1,3) = t3*fac
            tau_cell(i+1,j+1,k+1,4) = t4*fac
            tau_cell(i+1,j+1,k+1,5) = t5*fac
            tau_cell(i+1,j+1,k+1,6) = t6*fac
        end do
    end do
    end do

    ! Fill boundary halo slots with +edge (identical to set_tau_q).
    do k = 1, nk-1
    do j = 1, nj-1
        tau_cell(1, j+1, k+1, :) = tau_cell(2, j+1, k+1, :)
        q_cell(1, j+1, k+1, :) = q_cell(2, j+1, k+1, :)
        tau_cell(ni+1, j+1, k+1, :) = tau_cell(ni, j+1, k+1, :)
        q_cell(ni+1, j+1, k+1, :) = q_cell(ni, j+1, k+1, :)
    end do
    end do
    do k = 1, nk-1
    do i = 1, ni-1
        tau_cell(i+1, 1, k+1, :) = tau_cell(i+1, 2, k+1, :)
        q_cell(i+1, 1, k+1, :) = q_cell(i+1, 2, k+1, :)
        tau_cell(i+1, nj+1, k+1, :) = tau_cell(i+1, nj, k+1, :)
        q_cell(i+1, nj+1, k+1, :) = q_cell(i+1, nj, k+1, :)
    end do
    end do
    do j = 1, nj-1
    do i = 1, ni-1
        tau_cell(i+1, j+1, 1, :) = tau_cell(i+1, j+1, 2, :)
        q_cell(i+1, j+1, 1, :) = q_cell(i+1, j+1, 2, :)
        tau_cell(i+1, j+1, nk+1, :) = tau_cell(i+1, j+1, nk, :)
        q_cell(i+1, j+1, nk+1, :) = q_cell(i+1, j+1, nk, :)
    end do
    end do

end subroutine set_tau_q_tau_only

subroutine set_tau_q_q_only( &
    cons, T, mu, cp, kappa, Pr_turb, xlength, vol, dAi, dAj, dAk, &
    r, &
    Vx, Vr, Vt, &
    tau_cell, &
    q_cell, &
    mu_turb, &
    ni, nj, nk)

    use viscous_helpers
    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in) :: cons(ni, nj, nk, 5)
    real, intent(in) :: T(ni, nj, nk)
    real, intent(in) :: Pr_turb
    ! All three nodal: a real gas's viscosity, conductivity and specific heat
    ! are surfaces over the field, and freezing any of them at one state was
    ! worth as much as its whole spread over a fit box. Averaged to the cell
    ! below, like rho. The laminar Prandtl number is not among them -- it is
    ! the ratio of two of these, so passing it as well would be a second
    ! definition of the same thing.
    real, intent(in) :: mu(ni, nj, nk)
    real, intent(in) :: cp(ni, nj, nk)
    real, intent(in) :: kappa(ni, nj, nk)
    real, intent(in) :: xlength(ni-1, nj-1, nk-1)
    real, intent(in) :: vol(ni-1, nj-1, nk-1)
    real, intent(in) :: dAi(3, ni, nj-1, nk-1)
    real, intent(in) :: dAj(3, ni-1, nj, nk-1)
    real, intent(in) :: dAk(3, ni-1, nj-1, nk)
    real, intent(in) :: r(ni, nj, nk)
    real, intent(in) :: Vx(ni, nj, nk)
    real, intent(in) :: Vr(ni, nj, nk)
    real, intent(in) :: Vt(ni, nj, nk)
    real, intent(inout) :: tau_cell(ni+1, nj+1, nk+1, 6)
    real, intent(inout) :: q_cell(ni+1, nj+1, nk+1, 3)
    ! Cell-centred mixing-length turbulent viscosity, written at the cell's
    ! low-corner node (i,j,k). The final node in each axis is padding that is
    ! not written here and must not be read. intent(inout) so that padding is
    ! left untouched rather than becoming undefined.
    real, intent(inout) :: mu_turb(ni, nj, nk)

    integer :: i, j, k
    ! Row temps -- i is the contiguous (dim-1) axis, the SIMD lane index.
    real :: rcr(ni-1), ivr(ni-1), cpc(ni-1), kac(ni-1)
    real :: f1, f2, f3, f4, f5, f6, lambda

    ! Unused here BY DESIGN -- this control is the q half, so the momentum,
    ! viscosity, velocity and mixing-length streams are exactly what it exists
    ! to not read. See the note in the tau half.
    if (cons(1,1,1,1) /= cons(1,1,1,1) .or. mu(1,1,1) /= mu(1,1,1) &
        .or. Vx(1,1,1) /= Vx(1,1,1) .or. Vr(1,1,1) /= Vr(1,1,1) &
        .or. mu_turb(1,1,1) /= mu_turb(1,1,1) &
        .or. Vt(1,1,1) /= Vt(1,1,1) .or. xlength(1,1,1) /= xlength(1,1,1)) return

    do k = 1, nk-1
    do j = 1, nj-1
        ! Stage 1: velocity gradients + cell metrics, vectorizable over i.
        do i = 1, ni-1
            ivr(i) = 0.25e0 / vol(i,j,k)
            rcr(i) = 0.125e0 * (r(i,j,k)   + r(i+1,j,k)   + r(i,j+1,k)   + r(i+1,j+1,k) &
                              + r(i,j,k+1) + r(i+1,j,k+1) + r(i,j+1,k+1) + r(i+1,j+1,k+1))
            cpc(i) = 0.125e0 * (cp(i,j,k)   + cp(i+1,j,k)   + cp(i,j+1,k)   + cp(i+1,j+1,k) &
                              + cp(i,j,k+1) + cp(i+1,j,k+1) + cp(i,j+1,k+1) + cp(i+1,j+1,k+1))
            kac(i) = 0.125e0 * (kappa(i,j,k)   + kappa(i+1,j,k)   + kappa(i,j+1,k)   + kappa(i+1,j+1,k) &
                              + kappa(i,j,k+1) + kappa(i+1,j,k+1) + kappa(i,j+1,k+1) + kappa(i+1,j+1,k+1))
        end do
        ! Stage 2: tau, mixing-length mu_turb, and q -- store with stride-1
        ! per-component writes; vectorizable over i.
        do i = 1, ni-1
            lambda = kac(i) + cpc(i) / Pr_turb
            f1 = T(i,j,k)+T(i,j+1,k)+T(i,j,k+1)+T(i,j+1,k+1)
            f2 = T(i+1,j,k)+T(i+1,j+1,k)+T(i+1,j,k+1)+T(i+1,j+1,k+1)
            f3 = T(i,j,k)+T(i+1,j,k)+T(i,j,k+1)+T(i+1,j,k+1)
            f4 = T(i,j+1,k)+T(i+1,j+1,k)+T(i,j+1,k+1)+T(i+1,j+1,k+1)
            f5 = T(i,j,k)+T(i+1,j,k)+T(i,j+1,k)+T(i+1,j+1,k)
            f6 = T(i,j,k+1)+T(i+1,j,k+1)+T(i,j+1,k+1)+T(i+1,j+1,k+1)
            q_cell(i+1,j+1,k+1,1) = (f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k) &
                  -f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1)) * (ivr(i)*lambda*0.5e0)
            q_cell(i+1,j+1,k+1,3) = (f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k) &
                  -f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1)) * (ivr(i)*lambda*0.5e0)
            q_cell(i+1,j+1,k+1,2) = ((f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k) &
                  -f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))*ivr(i) &
                  + 0.125e0*(f1+f2)/rcr(i)) * (lambda*0.5e0)
        end do
    end do
    end do

    ! Fill boundary halo slots with +edge (identical to set_tau_q).
    do k = 1, nk-1
    do j = 1, nj-1
        tau_cell(1, j+1, k+1, :) = tau_cell(2, j+1, k+1, :)
        q_cell(1, j+1, k+1, :) = q_cell(2, j+1, k+1, :)
        tau_cell(ni+1, j+1, k+1, :) = tau_cell(ni, j+1, k+1, :)
        q_cell(ni+1, j+1, k+1, :) = q_cell(ni, j+1, k+1, :)
    end do
    end do
    do k = 1, nk-1
    do i = 1, ni-1
        tau_cell(i+1, 1, k+1, :) = tau_cell(i+1, 2, k+1, :)
        q_cell(i+1, 1, k+1, :) = q_cell(i+1, 2, k+1, :)
        tau_cell(i+1, nj+1, k+1, :) = tau_cell(i+1, nj, k+1, :)
        q_cell(i+1, nj+1, k+1, :) = q_cell(i+1, nj, k+1, :)
    end do
    end do
    do j = 1, nj-1
    do i = 1, ni-1
        tau_cell(i+1, j+1, 1, :) = tau_cell(i+1, j+1, 2, :)
        q_cell(i+1, j+1, 1, :) = q_cell(i+1, j+1, 2, :)
        tau_cell(i+1, j+1, nk+1, :) = tau_cell(i+1, j+1, nk, :)
        q_cell(i+1, j+1, nk+1, :) = q_cell(i+1, j+1, nk, :)
    end do
    end do

end subroutine set_tau_q_q_only
