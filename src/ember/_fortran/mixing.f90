! Pitch-weighted averaging subroutines for mixing plane calculations

subroutine pitch_avg_i(cons, w, dest, ni, nj, nk)
    ! Pitch-weighted average of cons over the i dimension (pitch_dim=0).
    !
    ! Parameters
    ! ----------
    ! cons : real(ni, nj, nk, 5)
    !     Conserved variables
    ! w : real(ni, nj, nk)
    !     Pitch weights at each face node
    ! dest : real(nj, nk, 5)
    !     Output pitch-averaged conserved variables
    ! ni, nj, nk : integer
    !     Grid dimensions

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in)    :: cons(ni, nj, nk, 5)
    real, intent(in)    :: w(ni, nj, nk)
    real, intent(inout) :: dest(nj, nk, 5)

    integer :: i, j, k, v

    dest = 0.0
    do v = 1, 5
        do i = 1, ni
            do k = 1, nk
                do j = 1, nj
                    dest(j, k, v) = dest(j, k, v) + w(i, j, k) * cons(i, j, k, v)
                end do
            end do
        end do
    end do

end subroutine pitch_avg_i


subroutine pitch_avg_j(cons, w, dest, ni, nj, nk)
    ! Pitch-weighted average of cons over the j dimension (pitch_dim=1).
    !
    ! Parameters
    ! ----------
    ! cons : real(ni, nj, nk, 5)
    !     Conserved variables
    ! w : real(ni, nj, nk)
    !     Pitch weights at each face node
    ! dest : real(ni, nk, 5)
    !     Output pitch-averaged conserved variables
    ! ni, nj, nk : integer
    !     Grid dimensions

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in)    :: cons(ni, nj, nk, 5)
    real, intent(in)    :: w(ni, nj, nk)
    real, intent(inout) :: dest(ni, nk, 5)

    integer :: i, j, k, v

    dest = 0.0
    do v = 1, 5
        do j = 1, nj
            do k = 1, nk
                do i = 1, ni
                    dest(i, k, v) = dest(i, k, v) + w(i, j, k) * cons(i, j, k, v)
                end do
            end do
        end do
    end do

end subroutine pitch_avg_j


subroutine pitch_avg_k(cons, w, dest, ni, nj, nk)
    ! Pitch-weighted average of cons over the k dimension (pitch_dim=2).
    !
    ! Parameters
    ! ----------
    ! cons : real(ni, nj, nk, 5)
    !     Conserved variables
    ! w : real(ni, nj, nk)
    !     Pitch weights at each face node
    ! dest : real(ni, nj, 5)
    !     Output pitch-averaged conserved variables
    ! ni, nj, nk : integer
    !     Grid dimensions

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in)    :: cons(ni, nj, nk, 5)
    real, intent(in)    :: w(ni, nj, nk)
    real, intent(inout) :: dest(ni, nj, 5)

    integer :: i, j, k, v

    dest = 0.0
    do v = 1, 5
        do k = 1, nk
            do j = 1, nj
                do i = 1, ni
                    dest(i, j, v) = dest(i, j, v) + w(i, j, k) * cons(i, j, k, v)
                end do
            end do
        end do
    end do

end subroutine pitch_avg_k


subroutine flux_avg_i(cons, P, ho, w, dest, ni, nj, nk)
    ! Pitch-weighted flux average over the i dimension (pitch_dim=0).
    !
    ! Parameters
    ! ----------
    ! cons : real(ni, nj, nk, 5)   conserved variables (rho,rhoVx,rhoVr,rhorVt,rhoe)
    ! P    : real(ni, nj, nk)      pressure
    ! ho   : real(ni, nj, nk)      stagnation enthalpy
    ! w    : real(ni, nj, nk)      pitch weights at each face node
    ! dest : real(nj, nk, 5)       output pitch-averaged flux

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in)    :: cons(ni, nj, nk, 5)
    real, intent(in)    :: P(ni, nj, nk)
    real, intent(in)    :: ho(ni, nj, nk)
    real, intent(in)    :: w(ni, nj, nk)
    real, intent(inout) :: dest(nj, nk, 5)

    integer :: i, j, k
    real    :: rhoV, wn

    dest = 0.0
    do i = 1, ni
        do k = 1, nk
            do j = 1, nj
                wn = w(i, j, k)
                rhoV = cons(i, j, k, 2)
                dest(j, k, 1) = dest(j, k, 1) + wn * rhoV
                dest(j, k, 2) = dest(j, k, 2) + wn * (rhoV * cons(i,j,k,2)/cons(i,j,k,1) + P(i,j,k))
                dest(j, k, 3) = dest(j, k, 3) + wn * rhoV * cons(i,j,k,3)/cons(i,j,k,1)
                dest(j, k, 4) = dest(j, k, 4) + wn * rhoV * cons(i,j,k,4)/cons(i,j,k,1)
                dest(j, k, 5) = dest(j, k, 5) + wn * rhoV * ho(i,j,k)
            end do
        end do
    end do

end subroutine flux_avg_i


subroutine flux_avg_j(cons, P, ho, w, dest, ni, nj, nk)
    ! Pitch-weighted flux average over the j dimension (pitch_dim=1).
    !
    ! Parameters
    ! ----------
    ! cons : real(ni, nj, nk, 5)   conserved variables
    ! P    : real(ni, nj, nk)      pressure
    ! ho   : real(ni, nj, nk)      stagnation enthalpy
    ! w    : real(ni, nj, nk)      pitch weights at each face node
    ! dest : real(ni, nk, 5)       output pitch-averaged flux

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in)    :: cons(ni, nj, nk, 5)
    real, intent(in)    :: P(ni, nj, nk)
    real, intent(in)    :: ho(ni, nj, nk)
    real, intent(in)    :: w(ni, nj, nk)
    real, intent(inout) :: dest(ni, nk, 5)

    integer :: i, j, k
    real    :: rhoV, wn

    dest = 0.0
    do j = 1, nj
        do k = 1, nk
            do i = 1, ni
                wn = w(i, j, k)
                rhoV = cons(i, j, k, 2)
                dest(i, k, 1) = dest(i, k, 1) + wn * rhoV
                dest(i, k, 2) = dest(i, k, 2) + wn * (rhoV * cons(i,j,k,2)/cons(i,j,k,1) + P(i,j,k))
                dest(i, k, 3) = dest(i, k, 3) + wn * rhoV * cons(i,j,k,3)/cons(i,j,k,1)
                dest(i, k, 4) = dest(i, k, 4) + wn * rhoV * cons(i,j,k,4)/cons(i,j,k,1)
                dest(i, k, 5) = dest(i, k, 5) + wn * rhoV * ho(i,j,k)
            end do
        end do
    end do

end subroutine flux_avg_j


subroutine flux_avg_k(cons, P, ho, w, dest, ni, nj, nk)
    ! Pitch-weighted flux average over the k dimension (pitch_dim=2).
    !
    ! Parameters
    ! ----------
    ! cons : real(ni, nj, nk, 5)   conserved variables
    ! P    : real(ni, nj, nk)      pressure
    ! ho   : real(ni, nj, nk)      stagnation enthalpy
    ! w    : real(ni, nj, nk)      pitch weights at each face node
    ! dest : real(ni, nj, 5)       output pitch-averaged flux

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in)    :: cons(ni, nj, nk, 5)
    real, intent(in)    :: P(ni, nj, nk)
    real, intent(in)    :: ho(ni, nj, nk)
    real, intent(in)    :: w(ni, nj, nk)
    real, intent(inout) :: dest(ni, nj, 5)

    integer :: i, j, k
    real    :: rhoV, wn

    dest = 0.0
    do k = 1, nk
        do j = 1, nj
            do i = 1, ni
                wn = w(i, j, k)
                rhoV = cons(i, j, k, 2)
                dest(i, j, 1) = dest(i, j, 1) + wn * rhoV
                dest(i, j, 2) = dest(i, j, 2) + wn * (rhoV * cons(i,j,k,2)/cons(i,j,k,1) + P(i,j,k))
                dest(i, j, 3) = dest(i, j, 3) + wn * rhoV * cons(i,j,k,3)/cons(i,j,k,1)
                dest(i, j, 4) = dest(i, j, 4) + wn * rhoV * cons(i,j,k,4)/cons(i,j,k,1)
                dest(i, j, 5) = dest(i, j, 5) + wn * rhoV * ho(i,j,k)
            end do
        end do
    end do

end subroutine flux_avg_k
