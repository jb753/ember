subroutine flux_to_primitive(conserved, r, ho, dhdP_rho, dhdrho_P, out, n)
    ! EOS-agnostic flux->primitive Jacobian for 1D blocks
    ! conserved(n,5) = [rho, rhoVx, rhoVr, rhorVt, rhoe]
    ! r(n), ho(n), dhdP_rho(n), dhdrho_P(n) : precomputed EOS arrays
    ! out(n,5,5) : preallocated output Jacobian
    implicit none
    integer, intent(in) :: n
    real, intent(in)    :: conserved(n, 5), r(n)
    real, intent(in)    :: ho(n), dhdP_rho(n), dhdrho_P(n)
    real, intent(inout) :: out(n, 5, 5)
    integer :: i
    real :: rho, Vx, Vr, Vt, r_i, ho_i, dhdP, dhdrho
    real :: VxVx, Vsq, dhdP_rho_rho, dhdrho_rho
    real :: D_inv, Vx_inv, rho_inv, rhoVx_inv, r_inv
    real :: D_inv_Vx, D_inv_rho

    do i = 1, n
        rho     = conserved(i, 1)
        Vx      = conserved(i, 2) / rho
        Vr      = conserved(i, 3) / rho
        r_i     = r(i)
        Vt      = conserved(i, 4) / (rho * r_i)
        ho_i    = ho(i)
        dhdP    = dhdP_rho(i)
        dhdrho  = dhdrho_P(i)

        VxVx         = Vx * Vx
        Vsq          = Vr * Vr + Vt * Vt
        dhdP_rho_rho = dhdP * rho
        dhdrho_rho   = dhdrho * rho
        D_inv        = 1.0 / (VxVx * (dhdP_rho_rho - 1.0) + dhdrho_rho)
        Vx_inv       = 1.0 / Vx
        rho_inv      = 1.0 / rho
        rhoVx_inv    = rho_inv * Vx_inv
        r_inv        = 1.0 / r_i
        D_inv_Vx     = D_inv * Vx_inv
        D_inv_rho    = D_inv * rho_inv

        out(i, :, :) = 0.0

        ! Row 1: d(rho)/d(flux)
        out(i,1,1) = (Vsq - ho_i + VxVx*(2.0*dhdP_rho_rho - 1.0)) * D_inv_Vx
        out(i,1,2) = -dhdP_rho_rho * D_inv
        out(i,1,3) = -Vr * D_inv_Vx
        out(i,1,4) = -Vt * r_inv * D_inv_Vx
        out(i,1,5) = D_inv_Vx

        ! Row 2: d(Vx)/d(flux)
        out(i,2,1) = (-Vsq + ho_i + dhdrho_rho - VxVx*dhdP_rho_rho) * D_inv_rho
        out(i,2,2) = Vx * dhdP * D_inv
        out(i,2,3) = Vr * D_inv_rho
        out(i,2,4) = Vt * r_inv * D_inv_rho
        out(i,2,5) = -D_inv_rho

        ! Row 3: d(Vr)/d(flux)
        out(i,3,1) = -Vr * rhoVx_inv
        out(i,3,3) = rhoVx_inv

        ! Row 4: d(Vt)/d(flux)
        out(i,4,1) = -Vt * rhoVx_inv
        out(i,4,4) = r_inv * rhoVx_inv

        ! Row 5: d(P)/d(flux)
        out(i,5,1) = Vx*(Vsq + VxVx - 2.0*dhdrho_rho - ho_i)*D_inv
        out(i,5,2) = (dhdrho_rho - VxVx)*D_inv
        out(i,5,3) = -Vx*Vr*D_inv
        out(i,5,4) = -Vx*Vt*r_inv*D_inv
        out(i,5,5) = Vx*D_inv
    end do
end subroutine flux_to_primitive

subroutine primitive_to_chic(rho, a, out, n)
    ! Primitive->characteristic Jacobian for 1D blocks
    ! rho(n), a(n) : density and speed of sound
    ! out(n,5,5) : preallocated output Jacobian
    implicit none
    integer, intent(in) :: n
    real, intent(in)    :: rho(n), a(n)
    real, intent(inout) :: out(n, 5, 5)
    integer :: i
    real :: rhoa_i, asq_i

    do i = 1, n
        rhoa_i = rho(i) * a(i)
        asq_i  = a(i) * a(i)

        out(i, :, :) = 0.0

        ! Row 1: d(chic1)/d(prim) = [0, -rhoa, 0, 0, 1]
        out(i,1,2) = -rhoa_i
        out(i,1,5) =  1.0

        ! Row 2: d(chic2)/d(prim) = [0, rhoa, 0, 0, 1]
        out(i,2,2) = rhoa_i
        out(i,2,5) = 1.0

        ! Row 3: d(chic3)/d(prim) = [0, 0, rhoa, 0, 0]
        out(i,3,3) = rhoa_i

        ! Row 4: d(chic4)/d(prim) = [0, 0, 0, rhoa, 0]
        out(i,4,4) = rhoa_i

        ! Row 5: d(chic5)/d(prim) = [-a^2, 0, 0, 0, 1]
        out(i,5,1) = -asq_i
        out(i,5,5) =  1.0
    end do
end subroutine primitive_to_chic

subroutine chic_to_primitive(rho, a, out, n)
    ! Characteristic->primitive Jacobian for 1D blocks (inverse of primitive_to_chic)
    ! rho(n), a(n) : density and speed of sound
    ! out(n,5,5) : preallocated output Jacobian
    implicit none
    integer, intent(in) :: n
    real, intent(in)    :: rho(n), a(n)
    real, intent(inout) :: out(n, 5, 5)
    integer :: i
    real :: asq_recip, rhoa_recip, half_asq_recip, half_rhoa_recip

    do i = 1, n
        asq_recip      = 1.0 / (a(i) * a(i))
        rhoa_recip     = 1.0 / (rho(i) * a(i))
        half_asq_recip = 0.5 * asq_recip
        half_rhoa_recip = 0.5 * rhoa_recip

        out(i, :, :) = 0.0

        ! Row 1: d(rho)/d(chic) = [asq/2, asq/2, 0, 0, -asq]
        out(i,1,1) =  half_asq_recip
        out(i,1,2) =  half_asq_recip
        out(i,1,5) = -asq_recip

        ! Row 2: d(Vx)/d(chic) = [-1/(2*rhoa), 1/(2*rhoa), 0, 0, 0]
        out(i,2,1) = -half_rhoa_recip
        out(i,2,2) =  half_rhoa_recip

        ! Row 3: d(Vr)/d(chic) = [0, 0, 1/(rhoa), 0, 0]
        out(i,3,3) = rhoa_recip

        ! Row 4: d(Vt)/d(chic) = [0, 0, 0, 1/(rhoa), 0]
        out(i,4,4) = rhoa_recip

        ! Row 5: d(P)/d(chic) = [0.5, 0.5, 0, 0, 0]
        out(i,5,1) = 0.5
        out(i,5,2) = 0.5
    end do
end subroutine chic_to_primitive

subroutine primitive_to_mix(conserved, r, dhdrho_P, dhdP_rho, dsdrho_P, dsdP_rho, out, n)
    ! Primitive->mix Jacobian for 1D blocks
    ! conserved(n,5) = [rho, rhoVx, rhoVr, rhorVt, rhoe]
    ! mix = [ho, s, Vr, Vt, P]
    ! out(n,5,5) : preallocated output Jacobian
    implicit none
    integer, intent(in) :: n
    real, intent(in)    :: conserved(n, 5), r(n)
    real, intent(in)    :: dhdrho_P(n), dhdP_rho(n), dsdrho_P(n), dsdP_rho(n)
    real, intent(inout) :: out(n, 5, 5)
    integer :: i
    real :: rho, Vx, Vr, Vt

    do i = 1, n
        rho = conserved(i, 1)
        Vx  = conserved(i, 2) / rho
        Vr  = conserved(i, 3) / rho
        Vt  = conserved(i, 4) / (rho * r(i))

        out(i, :, :) = 0.0

        ! Row 1: d(ho)/d(prim) = [dhdrho_P, Vx, Vr, Vt, dhdP_rho]
        out(i,1,1) = dhdrho_P(i)
        out(i,1,2) = Vx
        out(i,1,3) = Vr
        out(i,1,4) = Vt
        out(i,1,5) = dhdP_rho(i)

        ! Row 2: d(s)/d(prim) = [dsdrho_P, 0, 0, 0, dsdP_rho]
        out(i,2,1) = dsdrho_P(i)
        out(i,2,5) = dsdP_rho(i)

        ! Row 3: d(Vr)/d(prim) = [0, 0, 1, 0, 0]
        out(i,3,3) = 1.0

        ! Row 4: d(Vt)/d(prim) = [0, 0, 0, 1, 0]
        out(i,4,4) = 1.0

        ! Row 5: d(P)/d(prim) = [0, 0, 0, 0, 1]
        out(i,5,5) = 1.0
    end do
end subroutine primitive_to_mix

subroutine mix_to_primitive(conserved, r, dhdrho_P, dhdP_rho, dsdrho_P, dsdP_rho, out, n)
    ! Mix->primitive Jacobian for 1D blocks (inverse of primitive_to_mix)
    ! conserved(n,5) = [rho, rhoVx, rhoVr, rhorVt, rhoe]
    ! mix = [ho, s, Vr, Vt, P]
    ! out(n,5,5) : preallocated output Jacobian
    implicit none
    integer, intent(in) :: n
    real, intent(in)    :: conserved(n, 5), r(n)
    real, intent(in)    :: dhdrho_P(n), dhdP_rho(n), dsdrho_P(n), dsdP_rho(n)
    real, intent(inout) :: out(n, 5, 5)
    integer :: i
    real :: rho, Vx, Vr, Vt
    real :: Vx_inv, dsdrho_inv, dsdP, cross

    do i = 1, n
        rho  = conserved(i, 1)
        Vx   = conserved(i, 2) / rho
        Vr   = conserved(i, 3) / rho
        Vt   = conserved(i, 4) / (rho * r(i))

        Vx_inv     = 1.0 / Vx
        dsdrho_inv = 1.0 / dsdrho_P(i)
        dsdP       = dsdP_rho(i)
        cross      = (dhdrho_P(i) * dsdP - dhdP_rho(i) * dsdrho_P(i)) * dsdrho_inv

        out(i, :, :) = 0.0

        ! Row 1: d(rho)/d(mix) = [0, dsdrho_inv, 0, 0, -dsdP*dsdrho_inv]
        out(i,1,2) =  dsdrho_inv
        out(i,1,5) = -dsdP * dsdrho_inv

        ! Row 2: d(Vx)/d(mix) = [Vx_inv, -dhdrho_P*Vx_inv*dsdrho_inv, -Vr*Vx_inv, -Vt*Vx_inv, cross*Vx_inv]
        out(i,2,1) =  Vx_inv
        out(i,2,2) = -dhdrho_P(i) * Vx_inv * dsdrho_inv
        out(i,2,3) = -Vr * Vx_inv
        out(i,2,4) = -Vt * Vx_inv
        out(i,2,5) =  cross * Vx_inv

        ! Row 3: d(Vr)/d(mix) = [0, 0, 1, 0, 0]
        out(i,3,3) = 1.0

        ! Row 4: d(Vt)/d(mix) = [0, 0, 0, 1, 0]
        out(i,4,4) = 1.0

        ! Row 5: d(P)/d(mix) = [0, 0, 0, 0, 1]
        out(i,5,5) = 1.0
    end do
end subroutine mix_to_primitive

subroutine primitive_to_conserved(conserved, r, dudrho_P, dudP_rho, out, n)
    ! Primitive->conserved Jacobian for 1D blocks
    ! conserved(n,5) = [rho, rhoVx, rhoVr, rhorVt, rhoe]
    ! dudrho_P(n), dudP_rho(n) : EOS derivatives of specific internal energy
    ! out(n,5,5) : preallocated output Jacobian
    implicit none
    integer, intent(in) :: n
    real, intent(in)    :: conserved(n, 5), r(n)
    real, intent(in)    :: dudrho_P(n), dudP_rho(n)
    real, intent(inout) :: out(n, 5, 5)
    integer :: i
    real :: rho, Vx, Vr, Vt, r_i, e_i
    real :: drhoe_drho_P, drhoe_dP_rho

    do i = 1, n
        rho  = conserved(i, 1)
        Vx   = conserved(i, 2) / rho
        Vr   = conserved(i, 3) / rho
        r_i  = r(i)
        Vt   = conserved(i, 4) / (rho * r_i)
        e_i  = conserved(i, 5) / rho

        drhoe_drho_P = e_i + rho * dudrho_P(i)
        drhoe_dP_rho = rho * dudP_rho(i)

        out(i, :, :) = 0.0

        ! Row 1: d(rho)/d(prim) = [1, 0, 0, 0, 0]
        out(i,1,1) = 1.0

        ! Row 2: d(rhoVx)/d(prim) = [Vx, rho, 0, 0, 0]
        out(i,2,1) = Vx
        out(i,2,2) = rho

        ! Row 3: d(rhoVr)/d(prim) = [Vr, 0, rho, 0, 0]
        out(i,3,1) = Vr
        out(i,3,3) = rho

        ! Row 4: d(rhorVt)/d(prim) = [r*Vt, 0, 0, r*rho, 0]
        out(i,4,1) = r_i * Vt
        out(i,4,4) = r_i * rho

        ! Row 5: d(rhoe)/d(prim) = [drhoe_drho_P, rhoVx, rhoVr, rho*Vt, drhoe_dP_rho]
        out(i,5,1) = drhoe_drho_P
        out(i,5,2) = conserved(i, 2)
        out(i,5,3) = conserved(i, 3)
        out(i,5,4) = rho * Vt
        out(i,5,5) = drhoe_dP_rho
    end do
end subroutine primitive_to_conserved

subroutine bcond_to_primitive(conserved, r, dhdrho_P, dhdP_rho, dsdrho_P, dsdP_rho, out, n)
    ! Bcond->primitive Jacobian for 1D blocks (inverse of primitive_to_bcond)
    ! conserved(n,5) = [rho, rhoVx, rhoVr, rhorVt, rhoe]
    ! bcond = [ho, s, tanAlpha, sinBeta, P]
    ! out(n,5,5) : preallocated output Jacobian
    implicit none
    integer, intent(in) :: n
    real, intent(in)    :: conserved(n, 5), r(n)
    real, intent(in)    :: dhdrho_P(n), dhdP_rho(n), dsdrho_P(n), dsdP_rho(n)
    real, intent(inout) :: out(n, 5, 5)
    integer :: i
    real :: rho, Vx, Vr, Vt, r_i
    real :: Vm_sq, Vm, Vm_mer_sq, Vm_mer, Vm_mer_cu
    real :: tanAlpha, E_inv, F_inv, Vm_F_inv, Vm_sq_F_inv
    real :: dsdrho_inv, dsdP, cross

    do i = 1, n
        rho  = conserved(i, 1)
        Vx   = conserved(i, 2) / rho
        Vr   = conserved(i, 3) / rho
        r_i  = r(i)
        Vt   = conserved(i, 4) / (rho * r_i)

        Vm_mer_sq = Vx*Vx + Vr*Vr
        Vm_mer    = sqrt(Vm_mer_sq)
        Vm_mer_cu = Vm_mer * Vm_mer_sq
        ! Vm is meridional speed = sqrt(Vx^2+Vr^2), Vm_sq = Vm_mer_sq
        Vm        = Vm_mer
        Vm_sq     = Vm_mer_sq
        tanAlpha  = Vt / Vm
        ! E_inv = 1 / (Vm + Vt*tanAlpha)
        E_inv     = 1.0 / (Vm + Vt * tanAlpha)
        ! F_inv = E_inv / Vm_mer^2
        F_inv     = E_inv / Vm_mer_sq
        Vm_F_inv  = Vm * F_inv
        Vm_sq_F_inv = Vm_sq * F_inv

        dsdrho_inv = 1.0 / dsdrho_P(i)
        dsdP       = dsdP_rho(i)
        cross      = (dhdrho_P(i) * dsdP - dhdP_rho(i) * dsdrho_P(i)) * dsdrho_inv

        out(i, :, :) = 0.0

        ! Row 1: d(rho)/d(bcond) = [0, dsdrho_inv, 0, 0, -dsdP*dsdrho_inv]
        out(i,1,2) =  dsdrho_inv
        out(i,1,5) = -dsdP * dsdrho_inv

        ! Row 2: d(Vx)/d(bcond)
        out(i,2,1) =  Vm_F_inv * Vx
        out(i,2,2) = -dhdrho_P(i) * Vm_F_inv * Vx * dsdrho_inv
        out(i,2,3) = -Vm_sq * Vt * Vx * F_inv
        out(i,2,4) = -Vm_mer_cu * Vr / (Vx * Vm_mer_sq)
        out(i,2,5) =  cross * Vm_F_inv * Vx

        ! Row 3: d(Vr)/d(bcond)
        out(i,3,1) =  Vm_F_inv * Vr
        out(i,3,2) = -dhdrho_P(i) * Vm_F_inv * Vr * dsdrho_inv
        out(i,3,3) = -Vm_sq * Vr * Vt * F_inv
        out(i,3,4) =  Vm_mer_cu / Vm_mer_sq
        out(i,3,5) =  cross * Vm_F_inv * Vr

        ! Row 4: d(Vt)/d(bcond)
        out(i,4,1) =  tanAlpha * E_inv
        out(i,4,2) = -dhdrho_P(i) * tanAlpha * E_inv * dsdrho_inv
        out(i,4,3) =  Vm_sq * E_inv
        out(i,4,5) =  cross * tanAlpha * E_inv

        ! Row 5: d(P)/d(bcond) = [0, 0, 0, 0, 1]
        out(i,5,5) = 1.0
    end do
end subroutine bcond_to_primitive
