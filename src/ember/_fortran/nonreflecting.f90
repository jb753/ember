! Fused characteristic recombination for NonReflectingPatch._recombine.
!
! Replaces the chain of np.stack + 3x matvec + np.where in
! ember.nonreflecting.NonReflectingPatch._recombine with a single pass per
! node: subtract the reference primitive from both the previous and the
! freshly marched state, transform each into characteristics via p2c, blend
! per characteristic component by mask_out (1.0 keeps the marched value,
! 0.0 keeps the previous one), then transform back to primitives via c2p and
! add the reference primitive back.
!
! Two variants, mirroring matvec.f90's matvec5_bcast_{j,k}: the reference
! primitive, p2c, c2p and mask_out vary along the patch's span_dim only and
! are constant (size 1) along the other two axes, matching
! Patch._span_bcast's convention. Which variant to call is selected once per
! patch, from span_dim, not per call. No _bcast_i variant: NonReflectingPatch
! is restricted to constant-x planes (NonReflectingPatch._check_plane), so
! const_dim is always the i axis and span_dim is always j or k, never i.

subroutine nonreflecting_recombine_bcast_j(rho, vx, vr, vt, p, prim_prev, &
                                            ref_prim, p2c, c2p, mask_out, &
                                            dchic_out, prim_out, ni, nj, nk)
    ! rho,vx,vr,vt,p : (ni,nj,nk)      marched primitive components
    ! prim_prev      : (ni,nj,nk,5)    patch's held incoming-characteristic state
    ! ref_prim       : (1,nj,1,5)      reference primitive, broadcast over i,k
    ! p2c, c2p       : (1,nj,1,5,5)    characteristic transforms, broadcast over i,k
    ! mask_out       : (1,nj,1,5)      1.0 keeps the marched characteristic, 0.0 the previous one
    ! dchic_out      : (ni,nj,nk,5)    out: blended characteristic deviation
    ! prim_out       : (ni,nj,nk,5)    out: recombined primitive

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in) :: rho(ni,nj,nk), vx(ni,nj,nk), vr(ni,nj,nk), vt(ni,nj,nk), p(ni,nj,nk)
    real, intent(in) :: prim_prev(ni,nj,nk,5)
    real, intent(in) :: ref_prim(1,nj,1,5)
    real, intent(in) :: p2c(1,nj,1,5,5)
    real, intent(in) :: c2p(1,nj,1,5,5)
    real, intent(in) :: mask_out(1,nj,1,5)
    real, intent(inout) :: dchic_out(ni,nj,nk,5)
    real, intent(inout) :: prim_out(ni,nj,nk,5)

    integer :: i, j, k
    real :: dp1, dp2, dp3, dp4, dp5
    real :: dm1, dm2, dm3, dm4, dm5
    real :: cp1, cp2, cp3, cp4, cp5
    real :: cm1, cm2, cm3, cm4, cm5
    real :: d1, d2, d3, d4, d5

    do k = 1, nk
        do j = 1, nj
            do i = 1, ni
                dp1 = prim_prev(i,j,k,1) - ref_prim(1,j,1,1)
                dp2 = prim_prev(i,j,k,2) - ref_prim(1,j,1,2)
                dp3 = prim_prev(i,j,k,3) - ref_prim(1,j,1,3)
                dp4 = prim_prev(i,j,k,4) - ref_prim(1,j,1,4)
                dp5 = prim_prev(i,j,k,5) - ref_prim(1,j,1,5)

                dm1 = rho(i,j,k) - ref_prim(1,j,1,1)
                dm2 = vx(i,j,k)  - ref_prim(1,j,1,2)
                dm3 = vr(i,j,k)  - ref_prim(1,j,1,3)
                dm4 = vt(i,j,k)  - ref_prim(1,j,1,4)
                dm5 = p(i,j,k)   - ref_prim(1,j,1,5)

                cp1 = p2c(1,j,1,1,1)*dp1 + p2c(1,j,1,1,2)*dp2 + p2c(1,j,1,1,3)*dp3 + p2c(1,j,1,1,4)*dp4 + p2c(1,j,1,1,5)*dp5
                cp2 = p2c(1,j,1,2,1)*dp1 + p2c(1,j,1,2,2)*dp2 + p2c(1,j,1,2,3)*dp3 + p2c(1,j,1,2,4)*dp4 + p2c(1,j,1,2,5)*dp5
                cp3 = p2c(1,j,1,3,1)*dp1 + p2c(1,j,1,3,2)*dp2 + p2c(1,j,1,3,3)*dp3 + p2c(1,j,1,3,4)*dp4 + p2c(1,j,1,3,5)*dp5
                cp4 = p2c(1,j,1,4,1)*dp1 + p2c(1,j,1,4,2)*dp2 + p2c(1,j,1,4,3)*dp3 + p2c(1,j,1,4,4)*dp4 + p2c(1,j,1,4,5)*dp5
                cp5 = p2c(1,j,1,5,1)*dp1 + p2c(1,j,1,5,2)*dp2 + p2c(1,j,1,5,3)*dp3 + p2c(1,j,1,5,4)*dp4 + p2c(1,j,1,5,5)*dp5

                cm1 = p2c(1,j,1,1,1)*dm1 + p2c(1,j,1,1,2)*dm2 + p2c(1,j,1,1,3)*dm3 + p2c(1,j,1,1,4)*dm4 + p2c(1,j,1,1,5)*dm5
                cm2 = p2c(1,j,1,2,1)*dm1 + p2c(1,j,1,2,2)*dm2 + p2c(1,j,1,2,3)*dm3 + p2c(1,j,1,2,4)*dm4 + p2c(1,j,1,2,5)*dm5
                cm3 = p2c(1,j,1,3,1)*dm1 + p2c(1,j,1,3,2)*dm2 + p2c(1,j,1,3,3)*dm3 + p2c(1,j,1,3,4)*dm4 + p2c(1,j,1,3,5)*dm5
                cm4 = p2c(1,j,1,4,1)*dm1 + p2c(1,j,1,4,2)*dm2 + p2c(1,j,1,4,3)*dm3 + p2c(1,j,1,4,4)*dm4 + p2c(1,j,1,4,5)*dm5
                cm5 = p2c(1,j,1,5,1)*dm1 + p2c(1,j,1,5,2)*dm2 + p2c(1,j,1,5,3)*dm3 + p2c(1,j,1,5,4)*dm4 + p2c(1,j,1,5,5)*dm5

                d1 = mask_out(1,j,1,1)*cm1 + (1.0 - mask_out(1,j,1,1))*cp1
                d2 = mask_out(1,j,1,2)*cm2 + (1.0 - mask_out(1,j,1,2))*cp2
                d3 = mask_out(1,j,1,3)*cm3 + (1.0 - mask_out(1,j,1,3))*cp3
                d4 = mask_out(1,j,1,4)*cm4 + (1.0 - mask_out(1,j,1,4))*cp4
                d5 = mask_out(1,j,1,5)*cm5 + (1.0 - mask_out(1,j,1,5))*cp5

                dchic_out(i,j,k,1) = d1
                dchic_out(i,j,k,2) = d2
                dchic_out(i,j,k,3) = d3
                dchic_out(i,j,k,4) = d4
                dchic_out(i,j,k,5) = d5

                prim_out(i,j,k,1) = ref_prim(1,j,1,1) + c2p(1,j,1,1,1)*d1 + c2p(1,j,1,1,2)*d2 &
                    + c2p(1,j,1,1,3)*d3 + c2p(1,j,1,1,4)*d4 + c2p(1,j,1,1,5)*d5
                prim_out(i,j,k,2) = ref_prim(1,j,1,2) + c2p(1,j,1,2,1)*d1 + c2p(1,j,1,2,2)*d2 &
                    + c2p(1,j,1,2,3)*d3 + c2p(1,j,1,2,4)*d4 + c2p(1,j,1,2,5)*d5
                prim_out(i,j,k,3) = ref_prim(1,j,1,3) + c2p(1,j,1,3,1)*d1 + c2p(1,j,1,3,2)*d2 &
                    + c2p(1,j,1,3,3)*d3 + c2p(1,j,1,3,4)*d4 + c2p(1,j,1,3,5)*d5
                prim_out(i,j,k,4) = ref_prim(1,j,1,4) + c2p(1,j,1,4,1)*d1 + c2p(1,j,1,4,2)*d2 &
                    + c2p(1,j,1,4,3)*d3 + c2p(1,j,1,4,4)*d4 + c2p(1,j,1,4,5)*d5
                prim_out(i,j,k,5) = ref_prim(1,j,1,5) + c2p(1,j,1,5,1)*d1 + c2p(1,j,1,5,2)*d2 &
                    + c2p(1,j,1,5,3)*d3 + c2p(1,j,1,5,4)*d4 + c2p(1,j,1,5,5)*d5
            end do
        end do
    end do

end subroutine nonreflecting_recombine_bcast_j


subroutine nonreflecting_recombine_bcast_k(rho, vx, vr, vt, p, prim_prev, &
                                            ref_prim, p2c, c2p, mask_out, &
                                            dchic_out, prim_out, ni, nj, nk)
    ! As nonreflecting_recombine_bcast_j, but the reference primitive, p2c,
    ! c2p and mask_out vary along k and broadcast over i,j:
    ! ref_prim/mask_out (1,1,nk,5), p2c/c2p (1,1,nk,5,5).

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in) :: rho(ni,nj,nk), vx(ni,nj,nk), vr(ni,nj,nk), vt(ni,nj,nk), p(ni,nj,nk)
    real, intent(in) :: prim_prev(ni,nj,nk,5)
    real, intent(in) :: ref_prim(1,1,nk,5)
    real, intent(in) :: p2c(1,1,nk,5,5)
    real, intent(in) :: c2p(1,1,nk,5,5)
    real, intent(in) :: mask_out(1,1,nk,5)
    real, intent(inout) :: dchic_out(ni,nj,nk,5)
    real, intent(inout) :: prim_out(ni,nj,nk,5)

    integer :: i, j, k
    real :: dp1, dp2, dp3, dp4, dp5
    real :: dm1, dm2, dm3, dm4, dm5
    real :: cp1, cp2, cp3, cp4, cp5
    real :: cm1, cm2, cm3, cm4, cm5
    real :: d1, d2, d3, d4, d5

    do k = 1, nk
        do j = 1, nj
            do i = 1, ni
                dp1 = prim_prev(i,j,k,1) - ref_prim(1,1,k,1)
                dp2 = prim_prev(i,j,k,2) - ref_prim(1,1,k,2)
                dp3 = prim_prev(i,j,k,3) - ref_prim(1,1,k,3)
                dp4 = prim_prev(i,j,k,4) - ref_prim(1,1,k,4)
                dp5 = prim_prev(i,j,k,5) - ref_prim(1,1,k,5)

                dm1 = rho(i,j,k) - ref_prim(1,1,k,1)
                dm2 = vx(i,j,k)  - ref_prim(1,1,k,2)
                dm3 = vr(i,j,k)  - ref_prim(1,1,k,3)
                dm4 = vt(i,j,k)  - ref_prim(1,1,k,4)
                dm5 = p(i,j,k)   - ref_prim(1,1,k,5)

                cp1 = p2c(1,1,k,1,1)*dp1 + p2c(1,1,k,1,2)*dp2 + p2c(1,1,k,1,3)*dp3 + p2c(1,1,k,1,4)*dp4 + p2c(1,1,k,1,5)*dp5
                cp2 = p2c(1,1,k,2,1)*dp1 + p2c(1,1,k,2,2)*dp2 + p2c(1,1,k,2,3)*dp3 + p2c(1,1,k,2,4)*dp4 + p2c(1,1,k,2,5)*dp5
                cp3 = p2c(1,1,k,3,1)*dp1 + p2c(1,1,k,3,2)*dp2 + p2c(1,1,k,3,3)*dp3 + p2c(1,1,k,3,4)*dp4 + p2c(1,1,k,3,5)*dp5
                cp4 = p2c(1,1,k,4,1)*dp1 + p2c(1,1,k,4,2)*dp2 + p2c(1,1,k,4,3)*dp3 + p2c(1,1,k,4,4)*dp4 + p2c(1,1,k,4,5)*dp5
                cp5 = p2c(1,1,k,5,1)*dp1 + p2c(1,1,k,5,2)*dp2 + p2c(1,1,k,5,3)*dp3 + p2c(1,1,k,5,4)*dp4 + p2c(1,1,k,5,5)*dp5

                cm1 = p2c(1,1,k,1,1)*dm1 + p2c(1,1,k,1,2)*dm2 + p2c(1,1,k,1,3)*dm3 + p2c(1,1,k,1,4)*dm4 + p2c(1,1,k,1,5)*dm5
                cm2 = p2c(1,1,k,2,1)*dm1 + p2c(1,1,k,2,2)*dm2 + p2c(1,1,k,2,3)*dm3 + p2c(1,1,k,2,4)*dm4 + p2c(1,1,k,2,5)*dm5
                cm3 = p2c(1,1,k,3,1)*dm1 + p2c(1,1,k,3,2)*dm2 + p2c(1,1,k,3,3)*dm3 + p2c(1,1,k,3,4)*dm4 + p2c(1,1,k,3,5)*dm5
                cm4 = p2c(1,1,k,4,1)*dm1 + p2c(1,1,k,4,2)*dm2 + p2c(1,1,k,4,3)*dm3 + p2c(1,1,k,4,4)*dm4 + p2c(1,1,k,4,5)*dm5
                cm5 = p2c(1,1,k,5,1)*dm1 + p2c(1,1,k,5,2)*dm2 + p2c(1,1,k,5,3)*dm3 + p2c(1,1,k,5,4)*dm4 + p2c(1,1,k,5,5)*dm5

                d1 = mask_out(1,1,k,1)*cm1 + (1.0 - mask_out(1,1,k,1))*cp1
                d2 = mask_out(1,1,k,2)*cm2 + (1.0 - mask_out(1,1,k,2))*cp2
                d3 = mask_out(1,1,k,3)*cm3 + (1.0 - mask_out(1,1,k,3))*cp3
                d4 = mask_out(1,1,k,4)*cm4 + (1.0 - mask_out(1,1,k,4))*cp4
                d5 = mask_out(1,1,k,5)*cm5 + (1.0 - mask_out(1,1,k,5))*cp5

                dchic_out(i,j,k,1) = d1
                dchic_out(i,j,k,2) = d2
                dchic_out(i,j,k,3) = d3
                dchic_out(i,j,k,4) = d4
                dchic_out(i,j,k,5) = d5

                prim_out(i,j,k,1) = ref_prim(1,1,k,1) + c2p(1,1,k,1,1)*d1 + c2p(1,1,k,1,2)*d2 &
                    + c2p(1,1,k,1,3)*d3 + c2p(1,1,k,1,4)*d4 + c2p(1,1,k,1,5)*d5
                prim_out(i,j,k,2) = ref_prim(1,1,k,2) + c2p(1,1,k,2,1)*d1 + c2p(1,1,k,2,2)*d2 &
                    + c2p(1,1,k,2,3)*d3 + c2p(1,1,k,2,4)*d4 + c2p(1,1,k,2,5)*d5
                prim_out(i,j,k,3) = ref_prim(1,1,k,3) + c2p(1,1,k,3,1)*d1 + c2p(1,1,k,3,2)*d2 &
                    + c2p(1,1,k,3,3)*d3 + c2p(1,1,k,3,4)*d4 + c2p(1,1,k,3,5)*d5
                prim_out(i,j,k,4) = ref_prim(1,1,k,4) + c2p(1,1,k,4,1)*d1 + c2p(1,1,k,4,2)*d2 &
                    + c2p(1,1,k,4,3)*d3 + c2p(1,1,k,4,4)*d4 + c2p(1,1,k,4,5)*d5
                prim_out(i,j,k,5) = ref_prim(1,1,k,5) + c2p(1,1,k,5,1)*d1 + c2p(1,1,k,5,2)*d2 &
                    + c2p(1,1,k,5,3)*d3 + c2p(1,1,k,5,4)*d4 + c2p(1,1,k,5,5)*d5
            end do
        end do
    end do

end subroutine nonreflecting_recombine_bcast_k
