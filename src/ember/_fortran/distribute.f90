! Routines for distributing node/face/cell values around

subroutine node_to_cell(xn, xc, ni, nj, nk, np)

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: np

    real, intent (in)  :: xn(ni, nj, nk, np)

    ! Note: seems to go faster if the outputs are 'inout'
    real, intent (inout)  :: xc(ni-1, nj-1, nk-1, np)

    ! Cell values are the average of all eight hex vertices
    xc = (&
        xn(1:ni-1, 1:nj-1, 1:nk-1, :) & ! i,j,k
        + xn(2:ni,   1:nj-1, 1:nk-1, :) & ! i+1,j,k
        + xn(2:ni,   2:nj,   1:nk-1, :) & ! i+1,j+1,k
        + xn(1:ni-1, 2:nj,   1:nk-1, :) & ! i,j+1,k
        + xn(1:ni-1, 1:nj-1, 2:nk,   :) & ! i,j,k+1
        + xn(2:ni,   1:nj-1, 2:nk,   :) & ! i+1,j,k+1
        + xn(2:ni,   2:nj,   2:nk,   :) & ! i+1,j+1,k+1
        + xn(1:ni-1, 2:nj,   2:nk,   :) & ! i,j+1,k+1
    )*0.125e0


end subroutine

subroutine cell_to_node_generic(xc, xn_in, xn_out, ni, nj, nk, np)

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: np

    real, intent (in)  :: xc(ni-1, nj-1, nk-1, np)

    ! Note: seems to go faster if the outputs are 'inout'
    real, intent (in)  :: xn_in(ni, nj, nk, np)
    real, intent (inout)  :: xn_out(ni, nj, nk, np)


    ! Interior nodes take 1/8 from each adjacent cell
    xn_out(2:ni-1, 2:nj-1, 2:nk-1, :) = xn_in(2:ni-1, 2:nj-1, 2:nk-1, :) + (&
        xc(1:ni-2, 1:nj-2, 1:nk-2, :) & ! i,j,k
        + xc(2:ni-1, 1:nj-2, 1:nk-2, :) & ! i+1,j,k
        + xc(2:ni-1, 2:nj-1, 1:nk-2, :) & ! i+1,j+1,k
        + xc(1:ni-2, 2:nj-1, 1:nk-2, :) & ! i,j+1,k
        + xc(1:ni-2, 1:nj-2, 2:nk-1, :) & ! i,j,k+1
        + xc(2:ni-1, 1:nj-2, 2:nk-1, :) & ! i+1,j,k+1
        + xc(2:ni-1, 2:nj-1, 2:nk-1, :) & ! i+1,j+1,k+1
        + xc(1:ni-2, 2:nj-1, 2:nk-1, :) & ! i,j+1,k+1
    )*0.125e0

    ! Face nodes take 1/4 from each adjacent cell

    ! i=1
    xn_out(1, 2:nj-1, 2:nk-1, :) = xn_in(1, 2:nj-1, 2:nk-1, :) + (&
        xc(1, 1:nj-2, 1:nk-2, :) & ! 1,j,k
        + xc(1, 2:nj-1, 1:nk-2, :) & ! 1,j+1,k
        + xc(1, 1:nj-2, 2:nk-1, :) & ! 1,j,k+1
        + xc(1, 2:nj-1, 2:nk-1, :) & ! 1,j+1,k+1
    )*0.25e0

    ! i=ni
    xn_out(ni, 2:nj-1, 2:nk-1, :) = xn_in(ni, 2:nj-1, 2:nk-1, :) + (&
        xc(ni-1, 1:nj-2, 1:nk-2, :) & ! ni-1,j,k
        + xc(ni-1, 2:nj-1, 1:nk-2, :) & ! ni-1,j+1,k
        + xc(ni-1, 1:nj-2, 2:nk-1, :) & ! ni-1,j,k+1
        + xc(ni-1, 2:nj-1, 2:nk-1, :) & ! ni-1,j+1,k+1
    )*0.25e0

    ! j=1
    xn_out(2:ni-1, 1, 2:nk-1, :) = xn_in(2:ni-1, 1, 2:nk-1, :) + (&
        xc(1:ni-2, 1, 1:nk-2, :) & ! i,1,k
        + xc(2:ni-1, 1, 1:nk-2, :) & ! i+1,1,k
        + xc(1:ni-2, 1, 2:nk-1, :) & ! i,1,k+1
        + xc(2:ni-1, 1, 2:nk-1, :) & ! i+1,1,k+1
    )*0.25e0

    ! j=nj
    xn_out(2:ni-1, nj, 2:nk-1, :) = xn_in(2:ni-1, nj, 2:nk-1, :) + (&
        xc(1:ni-2, nj-1, 1:nk-2, :) & ! i,nj-1,k
        + xc(2:ni-1, nj-1, 1:nk-2, :) & ! i+1,nj-1,k
        + xc(1:ni-2, nj-1, 2:nk-1, :) & ! i,nj-1,k+1
        + xc(2:ni-1, nj-1, 2:nk-1, :) & ! i+1,nj-1,k+1
    )*0.25e0

    ! k=1
    xn_out(2:ni-1, 2:nj-1, 1, :) = xn_in(2:ni-1, 2:nj-1, 1, :) + (&
        xc(1:ni-2, 1:nj-2, 1, :) &
        + xc(2:ni-1, 1:nj-2, 1, :) &
        + xc(1:ni-2, 2:nj-1, 1, :) &
        + xc(2:ni-1, 2:nj-1, 1, :) &
    )*0.25e0

    ! k=nk
    xn_out(2:ni-1, 2:nj-1, nk, :) = xn_in(2:ni-1, 2:nj-1, nk, :) + (&
        xc(1:ni-2, 1:nj-2, nk-1, :) &
        + xc(2:ni-1, 1:nj-2, nk-1, :) &
        + xc(1:ni-2, 2:nj-1, nk-1, :) &
        + xc(2:ni-1, 2:nj-1, nk-1, :) &
    )*0.25e0

    ! Edges take 1/2 from each adjacent cell

    ! i=1, j=1
    xn_out(1, 1, 2:nk-1, :) = xn_in(1, 1, 2:nk-1, :) + (&
        xc(1, 1, 1:nk-2, :) &
        + xc(1, 1, 2:nk-1, :) &
    )*0.5e0

    ! i=1, j=nj
    xn_out(1, nj, 2:nk-1, :) = xn_in(1, nj, 2:nk-1, :) + (&
        xc(1, nj-1, 1:nk-2, :) &
        + xc(1, nj-1, 2:nk-1, :) &
    )*0.5e0

    ! i=ni, j=1
    xn_out(ni, 1, 2:nk-1, :) = xn_in(ni, 1, 2:nk-1, :) + (&
        xc(ni-1, 1, 1:nk-2, :) &
        + xc(ni-1, 1, 2:nk-1, :) &
    )*0.5e0

    ! i=ni, j=nj
    xn_out(ni, nj, 2:nk-1, :) = xn_in(ni, nj, 2:nk-1, :) + (&
        xc(ni-1, nj-1, 1:nk-2, :) &
        + xc(ni-1, nj-1, 2:nk-1, :) &
    )*0.5e0

    ! i=1, k=1
    xn_out(1, 2:nj-1, 1, :) = xn_in(1, 2:nj-1, 1, :) + (&
        xc(1, 1:nj-2, 1, :) &
        + xc(1, 2:nj-1, 1, :) &
    )*0.5e0

    ! i=1, k=nk
    xn_out(1, 2:nj-1, nk, :) = xn_in(1, 2:nj-1, nk, :) + (&
        xc(1, 1:nj-2, nk-1, :) &
        + xc(1, 2:nj-1, nk-1, :) &
    )*0.5e0

    ! i=ni, k=1
    xn_out(ni, 2:nj-1, 1, :) = xn_in(ni, 2:nj-1, 1, :) + (&
        xc(ni-1, 1:nj-2, 1, :) &
        + xc(ni-1, 2:nj-1, 1, :) &
    )*0.5e0

    ! i=ni, k=nk
    xn_out(ni, 2:nj-1, nk, :) = xn_in(ni, 2:nj-1, nk, :) + (&
        xc(ni-1, 1:nj-2, nk-1, :) &
        + xc(ni-1, 2:nj-1, nk-1, :) &
    )*0.5e0

    ! j=1, k=1
    xn_out(2:ni-1, 1, 1, :) = xn_in(2:ni-1, 1, 1, :) + (&
        xc(1:ni-2, 1, 1, :) &
        + xc(2:ni-1, 1, 1, :) &
    )*0.5e0

    ! j=1, k=nk
    xn_out(2:ni-1, 1, nk, :) = xn_in(2:ni-1, 1, nk, :) + (&
        xc(1:ni-2, 1, nk-1, :) &
        + xc(2:ni-1, 1, nk-1, :) &
    )*0.5e0

    ! j=nj, k=1
    xn_out(2:ni-1, nj, 1, :) = xn_in(2:ni-1, nj, 1, :) + (&
        xc(1:ni-2, nj-1, 1, :) &
        + xc(2:ni-1, nj-1, 1, :) &
    )*0.5e0

    ! j=nj, k=nk
    xn_out(2:ni-1, nj, nk, :) = xn_in(2:ni-1, nj, nk, :) + (&
        xc(1:ni-2, nj-1, nk-1, :) &
        + xc(2:ni-1, nj-1, nk-1, :) &
    )*0.5e0

    ! Corners take entirety from nearest cell
    xn_out(1,  1,  1, :) = xn_in(1,  1,  1, :) + xc(1,    1,    1, :)
    xn_out(1,  nj, 1, :) = xn_in(1,  nj, 1, :) + xc(1,    nj-1, 1, :)
    xn_out(ni, nj, 1, :) = xn_in(ni, nj, 1, :) + xc(ni-1, nj-1, 1, :)
    xn_out(ni, 1,  1, :) = xn_in(ni, 1,  1, :) + xc(ni-1, 1,    1, :)
    xn_out(1,  1,  nk, :) = xn_in(1,  1,  nk, :) + xc(1,    1,    nk-1, :)
    xn_out(1,  nj, nk, :) = xn_in(1,  nj, nk, :) + xc(1,    nj-1, nk-1, :)
    xn_out(ni, nj, nk, :) = xn_in(ni, nj, nk, :) + xc(ni-1, nj-1, nk-1, :)
    xn_out(ni, 1,  nk, :) = xn_in(ni, 1,  nk, :) + xc(ni-1, 1,    nk-1, :)


end subroutine cell_to_node_generic


subroutine cell_to_node(xc, xn, ni, nj, nk, np)
    ! Wrapper for cell_to_node_generic where input and output are the same array
    ! Maintains backward compatibility with existing code
    implicit none

    integer, intent (in)  :: ni, nj, nk, np
    real, intent (in)  :: xc(ni-1, nj-1, nk-1, np)
    real, intent (inout)  :: xn(ni, nj, nk, np)

    call cell_to_node_generic(xc, xn, xn, ni, nj, nk, np)

end subroutine cell_to_node
