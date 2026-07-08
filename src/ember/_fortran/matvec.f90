! Batch matrix-vector multiplication for stacks of 5x5 matrices

subroutine matvec5(A, b, out, n)
    ! Multiply n stacked 5x5 matrices by n stacked 5-vectors
    !
    ! Parameters
    ! ----------
    ! A : real(n, 5, 5)
    !     Input matrices, batch dimension first
    ! b : real(n, 5)
    !     Input vectors
    ! out : real(n, 5)
    !     Output vectors, may alias b
    ! n : integer
    !     Batch size

    implicit none

    integer, intent(in) :: n
    real, intent(in) :: A(n, 5, 5)
    real, intent(in) :: b(n, 5)
    real, intent(inout) :: out(n, 5)

    integer :: i
    real :: b1, b2, b3, b4, b5

    do i = 1, n
        b1 = b(i, 1)
        b2 = b(i, 2)
        b3 = b(i, 3)
        b4 = b(i, 4)
        b5 = b(i, 5)
        out(i, 1) = A(i,1,1)*b1 + A(i,1,2)*b2 + A(i,1,3)*b3 + A(i,1,4)*b4 + A(i,1,5)*b5
        out(i, 2) = A(i,2,1)*b1 + A(i,2,2)*b2 + A(i,2,3)*b3 + A(i,2,4)*b4 + A(i,2,5)*b5
        out(i, 3) = A(i,3,1)*b1 + A(i,3,2)*b2 + A(i,3,3)*b3 + A(i,3,4)*b4 + A(i,3,5)*b5
        out(i, 4) = A(i,4,1)*b1 + A(i,4,2)*b2 + A(i,4,3)*b3 + A(i,4,4)*b4 + A(i,4,5)*b5
        out(i, 5) = A(i,5,1)*b1 + A(i,5,2)*b2 + A(i,5,3)*b3 + A(i,5,4)*b4 + A(i,5,5)*b5
    end do

end subroutine matvec5


subroutine matvec5_bcast_j(A, b, out, ni, nj, nk)
    ! Multiply (ni,nj,1) broadcast 5x5 matrices by (ni,nj,nk) 5-vectors
    !
    ! A has shape (ni, nj, 1, 5, 5) -- span along j, broadcast over k
    ! b has shape (ni, nj, nk, 5)
    ! out has shape (ni, nj, nk, 5), may alias b
    !
    ! Parameters
    ! ----------
    ! A : real(ni, nj, 1, 5, 5)
    !     Input matrices, broadcast over k
    ! b : real(ni, nj, nk, 5)
    !     Input vectors
    ! out : real(ni, nj, nk, 5)
    !     Output vectors, may alias b
    ! ni, nj, nk : integer
    !     Grid dimensions

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in) :: A(ni, nj, 1, 5, 5)
    real, intent(in) :: b(ni, nj, nk, 5)
    real, intent(inout) :: out(ni, nj, nk, 5)

    integer :: i, j, k
    real :: b1, b2, b3, b4, b5

    do k = 1, nk
        do j = 1, nj
            do i = 1, ni
                b1 = b(i, j, k, 1)
                b2 = b(i, j, k, 2)
                b3 = b(i, j, k, 3)
                b4 = b(i, j, k, 4)
                b5 = b(i, j, k, 5)
                out(i,j,k,1) = A(i,j,1,1,1)*b1 + A(i,j,1,1,2)*b2 + A(i,j,1,1,3)*b3 + A(i,j,1,1,4)*b4 + A(i,j,1,1,5)*b5
                out(i,j,k,2) = A(i,j,1,2,1)*b1 + A(i,j,1,2,2)*b2 + A(i,j,1,2,3)*b3 + A(i,j,1,2,4)*b4 + A(i,j,1,2,5)*b5
                out(i,j,k,3) = A(i,j,1,3,1)*b1 + A(i,j,1,3,2)*b2 + A(i,j,1,3,3)*b3 + A(i,j,1,3,4)*b4 + A(i,j,1,3,5)*b5
                out(i,j,k,4) = A(i,j,1,4,1)*b1 + A(i,j,1,4,2)*b2 + A(i,j,1,4,3)*b3 + A(i,j,1,4,4)*b4 + A(i,j,1,4,5)*b5
                out(i,j,k,5) = A(i,j,1,5,1)*b1 + A(i,j,1,5,2)*b2 + A(i,j,1,5,3)*b3 + A(i,j,1,5,4)*b4 + A(i,j,1,5,5)*b5
            end do
        end do
    end do

end subroutine matvec5_bcast_j


subroutine matvec5_bcast_k(A, b, out, ni, nj, nk)
    ! Multiply (ni,nj,nk) broadcast 5x5 matrices by (ni,nj,nk) 5-vectors
    ! where A has shape (1, 1, nk, 5, 5) -- constant along i and j dimensions
    !
    ! Parameters
    ! ----------
    ! A : real(1, 1, nk, 5, 5)
    !     Input matrices, broadcast over i and j
    ! b : real(ni, nj, nk, 5)
    !     Input vectors
    ! out : real(ni, nj, nk, 5)
    !     Output vectors, may alias b
    ! ni, nj, nk : integer
    !     Grid dimensions

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in) :: A(1, 1, nk, 5, 5)
    real, intent(in) :: b(ni, nj, nk, 5)
    real, intent(inout) :: out(ni, nj, nk, 5)

    integer :: i, j, k
    real :: b1, b2, b3, b4, b5

    do k = 1, nk
        do j = 1, nj
            do i = 1, ni
                b1 = b(i, j, k, 1)
                b2 = b(i, j, k, 2)
                b3 = b(i, j, k, 3)
                b4 = b(i, j, k, 4)
                b5 = b(i, j, k, 5)
                out(i,j,k,1) = A(1,1,k,1,1)*b1 + A(1,1,k,1,2)*b2 + A(1,1,k,1,3)*b3 + A(1,1,k,1,4)*b4 + A(1,1,k,1,5)*b5
                out(i,j,k,2) = A(1,1,k,2,1)*b1 + A(1,1,k,2,2)*b2 + A(1,1,k,2,3)*b3 + A(1,1,k,2,4)*b4 + A(1,1,k,2,5)*b5
                out(i,j,k,3) = A(1,1,k,3,1)*b1 + A(1,1,k,3,2)*b2 + A(1,1,k,3,3)*b3 + A(1,1,k,3,4)*b4 + A(1,1,k,3,5)*b5
                out(i,j,k,4) = A(1,1,k,4,1)*b1 + A(1,1,k,4,2)*b2 + A(1,1,k,4,3)*b3 + A(1,1,k,4,4)*b4 + A(1,1,k,4,5)*b5
                out(i,j,k,5) = A(1,1,k,5,1)*b1 + A(1,1,k,5,2)*b2 + A(1,1,k,5,3)*b3 + A(1,1,k,5,4)*b4 + A(1,1,k,5,5)*b5
            end do
        end do
    end do

end subroutine matvec5_bcast_k


subroutine matvec5_bcast_i(A, b, out, ni, nj, nk)
    ! Multiply broadcast 5x5 matrices by (ni,nj,nk) 5-vectors
    ! where A has shape (ni, 1, 1, 5, 5) -- constant along j and k dimensions
    !
    ! Parameters
    ! ----------
    ! A : real(ni, 1, 1, 5, 5)
    !     Input matrices, broadcast over j and k
    ! b : real(ni, nj, nk, 5)
    !     Input vectors
    ! out : real(ni, nj, nk, 5)
    !     Output vectors, may alias b
    ! ni, nj, nk : integer
    !     Grid dimensions

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in) :: A(ni, 1, 1, 5, 5)
    real, intent(in) :: b(ni, nj, nk, 5)
    real, intent(inout) :: out(ni, nj, nk, 5)

    integer :: i, j, k
    real :: b1, b2, b3, b4, b5

    do k = 1, nk
        do j = 1, nj
            do i = 1, ni
                b1 = b(i, j, k, 1)
                b2 = b(i, j, k, 2)
                b3 = b(i, j, k, 3)
                b4 = b(i, j, k, 4)
                b5 = b(i, j, k, 5)
                out(i,j,k,1) = A(i,1,1,1,1)*b1 + A(i,1,1,1,2)*b2 + A(i,1,1,1,3)*b3 + A(i,1,1,1,4)*b4 + A(i,1,1,1,5)*b5
                out(i,j,k,2) = A(i,1,1,2,1)*b1 + A(i,1,1,2,2)*b2 + A(i,1,1,2,3)*b3 + A(i,1,1,2,4)*b4 + A(i,1,1,2,5)*b5
                out(i,j,k,3) = A(i,1,1,3,1)*b1 + A(i,1,1,3,2)*b2 + A(i,1,1,3,3)*b3 + A(i,1,1,3,4)*b4 + A(i,1,1,3,5)*b5
                out(i,j,k,4) = A(i,1,1,4,1)*b1 + A(i,1,1,4,2)*b2 + A(i,1,1,4,3)*b3 + A(i,1,1,4,4)*b4 + A(i,1,1,4,5)*b5
                out(i,j,k,5) = A(i,1,1,5,1)*b1 + A(i,1,1,5,2)*b2 + A(i,1,1,5,3)*b3 + A(i,1,1,5,4)*b4 + A(i,1,1,5,5)*b5
            end do
        end do
    end do

end subroutine matvec5_bcast_i


subroutine matvec2_bcast_j(A, b, out, ni, nj, nk)
    ! Multiply (ni,nj,1) broadcast 2x2 matrices by (ni,nj,nk) 2-vectors
    ! span_dim=1: A(ni, nj, 1, 2, 2), b(ni, nj, nk, 2), out(ni, nj, nk, 2)

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in) :: A(ni, nj, 1, 2, 2)
    real, intent(in) :: b(ni, nj, nk, 2)
    real, intent(inout) :: out(ni, nj, nk, 2)

    integer :: i, j, k
    real :: b1, b2

    do k = 1, nk
        do j = 1, nj
            do i = 1, ni
                b1 = b(i, j, k, 1)
                b2 = b(i, j, k, 2)
                out(i,j,k,1) = A(i,j,1,1,1)*b1 + A(i,j,1,1,2)*b2
                out(i,j,k,2) = A(i,j,1,2,1)*b1 + A(i,j,1,2,2)*b2
            end do
        end do
    end do

end subroutine matvec2_bcast_j


subroutine matvec2_bcast_k(A, b, out, ni, nj, nk)
    ! Multiply (1,1,nk) broadcast 2x2 matrices by (ni,nj,nk) 2-vectors
    ! span_dim=2: A(1, 1, nk, 2, 2), b(ni, nj, nk, 2), out(ni, nj, nk, 2)

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in) :: A(1, 1, nk, 2, 2)
    real, intent(in) :: b(ni, nj, nk, 2)
    real, intent(inout) :: out(ni, nj, nk, 2)

    integer :: i, j, k
    real :: b1, b2

    do k = 1, nk
        do j = 1, nj
            do i = 1, ni
                b1 = b(i, j, k, 1)
                b2 = b(i, j, k, 2)
                out(i,j,k,1) = A(1,1,k,1,1)*b1 + A(1,1,k,1,2)*b2
                out(i,j,k,2) = A(1,1,k,2,1)*b1 + A(1,1,k,2,2)*b2
            end do
        end do
    end do

end subroutine matvec2_bcast_k


subroutine matvec2_bcast_i(A, b, out, ni, nj, nk)
    ! Multiply (ni,1,1) broadcast 2x2 matrices by (ni,nj,nk) 2-vectors
    ! span_dim=0: A(ni, 1, 1, 2, 2), b(ni, nj, nk, 2), out(ni, nj, nk, 2)

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in) :: A(ni, 1, 1, 2, 2)
    real, intent(in) :: b(ni, nj, nk, 2)
    real, intent(inout) :: out(ni, nj, nk, 2)

    integer :: i, j, k
    real :: b1, b2

    do k = 1, nk
        do j = 1, nj
            do i = 1, ni
                b1 = b(i, j, k, 1)
                b2 = b(i, j, k, 2)
                out(i,j,k,1) = A(i,1,1,1,1)*b1 + A(i,1,1,1,2)*b2
                out(i,j,k,2) = A(i,1,1,2,1)*b1 + A(i,1,1,2,2)*b2
            end do
        end do
    end do

end subroutine matvec2_bcast_i
