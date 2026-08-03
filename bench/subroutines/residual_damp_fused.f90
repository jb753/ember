! =====================================================================
! damp_residual_merged -- the change limiter with its full-volume sweeps
! merged. Benchmark-only companion to damp_residual (residual.f90).
!
! The production routine loops the component index m OUTSIDE the (i,j,k)
! nest, so for 5 components it makes TEN full-volume passes over dU:
! five reduction sweeps (block mean of |dU*dt_vol|) followed by five
! scaling sweeps. dU is 20 B/cell, so that is ~200 B/cell of traffic on
! an array that never leaves the pipeline -- more than set_residual's own
! ~101 B/cell of compulsory traffic (see section 21).
!
! The five reductions are mutually independent, as are the five scalings,
! so both groups collapse into a single sweep carrying five accumulators:
! 10 passes -> 2. dt_vol is also read once per sweep instead of five
! times.
!
! Bitwise argument: for each fixed m the sequence of addends and their
! order over (i,j,k) is unchanged -- only the interleaving between
! independent accumulator chains differs, and each chain's own order is
! untouched. Under -fp-model fast=2 ifort may still vectorize the merged
! reduction differently (five accumulators live at once rather than one),
! so bitwise equality is argued but not assumed: it is measured.
! =====================================================================
subroutine damp_residual_merged(dU, dt_vol, dampin, ni, nj, nk)

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    real, intent(in)    :: dt_vol(ni-1, nj-1, nk-1)
    real, intent(in)    :: dampin

    integer :: i, j, k, m, ncell
    real :: avg(5), ravg(5), chg, fdamp

    ncell = (ni-1)*(nj-1)*(nk-1)

    ! ---- Sweep 1: all five block means in one pass over dU/dt_vol ----
    do m = 1, 5
        avg(m) = 0.0e0
    end do
    ! m stays OUTSIDE the i loop: dU is component-LAST, so dU(:,j,k,m) is
    ! contiguous in i only for fixed m. Putting m innermost would make each
    ! (i,j,k) touch five locations ~(ni-1)*(nj-1)*(nk-1) elements apart and
    ! destroy the i-vectorization -- measured at +208% before this was fixed
    ! (section 2's "any layout change that strides the i reads is suspect").
    ! The saving here is therefore NOT fewer dU sweeps but fewer dt_vol
    ! reads: the k/j planes of dt_vol are hoisted and shared across m.
    do k = 1, nk-1
    do j = 1, nj-1
    do m = 1, 5
    do i = 1, ni-1
        avg(m) = avg(m) + abs(dU(i,j,k,m) * dt_vol(i,j,k))
    end do
    end do
    end do
    end do

    ! A flat field (avg = 0) would divide by zero. Production guards this
    ! with `cycle`; here the reciprocal is folded into a factor that is
    ! simply 0 for such a component, which makes fdamp 0 and the soft-clip
    ! the identity -- same outcome, no branch inside the sweep. Branch-free
    ! matters because the caller already skips this routine entirely when
    ! damping is off, so every execution here is one that does real work.
    do m = 1, 5
        avg(m) = avg(m) / ncell
        if (avg(m) > 0.0e0) then
            ravg(m) = 1.0e0 / avg(m)
        else
            ravg(m) = 0.0e0
        end if
    end do

    ! ---- Sweep 2: soft-clip every component in one pass ----
    do k = 1, nk-1
    do j = 1, nj-1
    do m = 1, 5
        do i = 1, ni-1
            chg   = abs(dU(i,j,k,m) * dt_vol(i,j,k))
            fdamp = chg * ravg(m)
            ! fdamp/dampin kept as a division, not a hoisted reciprocal
            ! multiply, so the arithmetic matches production exactly.
            dU(i,j,k,m) = dU(i,j,k,m) / (1.0e0 + fdamp/dampin)
        end do
    end do
    end do
    end do

end subroutine damp_residual_merged
