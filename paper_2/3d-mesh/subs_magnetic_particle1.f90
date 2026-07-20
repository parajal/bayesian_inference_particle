module subs_magnetic_particle1_m

  use tfem_m
  use math_defs_m
  use stokes_elements_m
  use hsl_ma57_m
  use laplace_elements_ale_m
  
  implicit none

  real(dp), dimension(:), allocatable :: xp
  real(dp), dimension(:), allocatable :: rp


contains
 
! computes the magnetic force acting on the particle due to external magnetic 
! field 

  subroutine rhs_max ( mesh, problem, elgrp, elem, matrix, vector, &
    first, last, coefficients, oldvectors, elemmat, elemvec )

    use stokes_globals_m

    type(mesh_t), intent(in) :: mesh
    type(problem_t), intent(in) :: problem
    integer, intent(in) :: elgrp, elem
    logical, intent(in) :: matrix, vector, first, last
    type(coefficients_t), intent(in) :: coefficients

    type(oldvectors_t), intent(in) :: oldvectors

    real(dp), intent(out), dimension(:,:) :: elemmat
    real(dp), intent(out), dimension(:) :: elemvec

    integer ::  ip, i, j

    real(dp)::alpha
    real(dp), dimension(:), allocatable, save :: B_sq 
    real(dp), dimension(:,:), allocatable, save  ::  B, tmp1, tmp2    
    real(dp), dimension(:,:,:), allocatable, save  :: tau

!   set globals, gauss, shapefunctions, ...

    call set_stokes_elem ( mesh, problem, elgrp, elem, first, last, &
      coefficients, oldvectors )

      ! allocate arrays

    if (first) then
      allocate ( u(ndf) )
      allocate ( B_sq(ninti) )
      allocate ( B(ninti, ndim) )
      allocate ( tmp1(ninti, ndf) )
      allocate ( tmp2(ndf, ndim) )
      allocate ( tau(ninti, ndim, ndim) )
    end if  

    call set_Gauss_integration ( gauss, xig, wg )

    call set_shape_function ( globalshape, xig, phi, dphi )

    call get_coordinates ( mesh, elgrp, elem, x )

    call isoparametric_deformation ( x(1:ndf,:), dphi, F, Finv, detF )

    call isoparametric_coordinates ( x(1:ndf,:), phi, xg )

    if ( coorsys == 1 ) then
      detF = 2 * pi * xg(:,2) * detF
    end if

    call shape_derivative ( dphi, Finv, dphidx )

    if ( vector ) then

      elemvec = 0 

    ! get u variable (magnetic potential) 

      call get_sysvector ( mesh, oldvectors%p(1)%p, oldvectors%s(1)%p, &
        elgrp, elem, u)

    !  magnetic permeability 
       alpha = coefficients%r(1)

    !  curl of the magnetic potential (B = magnetic flux density)
       B(:,1) =  matmul (dphidx(:,:,2), u )
       B(:,2) = -matmul (dphidx(:,:,1), u)

    ! compute the magnetic field squared 

       do ip = 1, ninti
         B_sq(ip) = sum ( B(ip,:) * B(ip,:) )       
       end do

    !  Maxwell stress tensor

       do i = 1, ndim
          do j = i, ndim        
            tau(:,i,j) =  B(:,i) * B(:,j) 
            if (i == j) then       
              tau(:,i,j) = 1/alpha * ( tau(:,i,j) - 0.5_dp * B_sq )
            else
              tau(:,i,j) = 1/alpha * tau(:,i,j)
              tau(:,j,i) = tau(:,i,j)
            end if  
          end do
       end do

    !  Magnetic force acting on the particle 

        do i = 1, ndim
          do ip = 1, ninti
            tmp1(ip,:) = matmul(dphidx(ip,:,:), tau(ip,:,i)) 
          end do   
        do j = 1, ndf
            tmp2(j,i) = sum ( tmp1(:,j) * detF * wg )
          end do
        end do

        elemvec = - reshape ( tmp2, (/ ndim*ndf /) )

    end if 

      !   matrix

    if (matrix) then
      
      elemmat = 0

    end if
    
    if ( last ) then 

      !Unset globals, gauss, shapefunctions, ...
      call unset_stokes_elem ( last, coefficients )

      ! Deallocate arrays

      deallocate( u )
      deallocate ( B )
      deallocate( B_sq )
      deallocate( tau )
      deallocate( tmp1 )
      deallocate( tmp2 )
   
    end if

  end subroutine rhs_max

! element subroutine to impose rigid body constraint on the whole particle domain  

  subroutine elementc ( mesh, problem, constr, elem, node, &
    matrix, vector, first, last, coefficients, oldvectors, elemmat, elemmat2, &
      elemmatadd, elemvec, elemvecadd )

    type(mesh_t), intent(in) :: mesh
    type(problem_t), intent(in) :: problem
    integer, intent(in) :: constr, elem, node
    logical, intent(in) :: matrix, vector, first, last
    type(coefficients_t), intent(in) :: coefficients
    type(oldvectors_t), intent(in) :: oldvectors
    real(dp), intent(out), dimension(:,:) :: elemmat, elemmat2, elemmatadd
    real(dp), intent(out), dimension(:) :: elemvec, elemvecadd
                                            
    real(dp) :: xr(1,2)
                                                
    xr(1,:) = mesh%coor(node,:)
 
    elemvec = 0
    elemvecadd = 0
                                                               
!   connection through collocation

    elemmat(1,1) = 1
    elemmat(1,2) = 0

    elemmat(2,1) = 0
    elemmat(2,2) = 1

    elemmatadd(1,:) = (/ -1._dp, 0._dp, -xr(1,2) + xp(2) /)
    elemmatadd(2,:) = (/ 0._dp, -1._dp, xr(1,1) - xp(1) /)                                              
                                                
  end subroutine elementc

!  element subroutine to impose rigid body constraint 
!  on the whole particle boundary
  subroutine elementc_b ( mesh, problem, constr, elem, node, &
    matrix, vector, first, last, coefficients, oldvectors, elemmat, elemmat2, &
    elemmatadd, elemvec, elemvecadd )

    type(mesh_t), intent(in) :: mesh
    type(problem_t), intent(in) :: problem
    integer, intent(in) :: constr, elem, node
    logical, intent(in) :: matrix, vector, first, last
    type(coefficients_t), intent(in) :: coefficients
    type(oldvectors_t), intent(in) :: oldvectors
    real(dp), intent(out), dimension(:,:) :: elemmat, elemmat2, elemmatadd
    real(dp), intent(out), dimension(:) :: elemvec, elemvecadd

    integer :: ncurveconstr
    real(dp) :: xr(1,2)

    ncurveconstr = problem%constraints(constr)%geometry1

    xr(1,:) = mesh%coor(mesh%curves(ncurveconstr)%nodes(node),:)

!   set shape function in the point
                                                                                
    if ( vector ) then
                                                                                
      elemvec = 0
      elemvecadd = 0
                                                                                
    end if
                                                                                
!   connection through collocation

    elemmat(1,1) = 1
    elemmat(1,2) = 0

    elemmat(2,1) = 0
    elemmat(2,2) = 1

    elemmatadd(1,:) = (/ -1._dp, 0._dp, -xr(1,2) + xp(2) /)
    elemmatadd(2,:) = (/ 0._dp, -1._dp, xr(1,1) - xp(1) /)

  end subroutine elementc_b

  ! use where no remeshing is required . 
subroutine update_mesh_nodes ( mesh, problem, velp, dt, sol, method )

  !   input/output
  
  !   coordinates of the mesh are updated after calling this routine
      type(mesh_t), intent(inout) :: mesh
  
  !   Is created if empty on call and kept at output
      type(problem_t), intent(inout) :: problem
  
  !   velocity of the particle and the time step
      real(dp), intent(in) :: velp(1,3), dt
  
  !   mesh velocity
      type(sysvector_t), dimension(:), intent(inout) :: sol
  
  !   subscript
  
  !   Time integration:
  !     method=1  Euler explicit
  !     method=2  Second-order Adams-Bashforth 
      integer, intent(in) :: method
  
  
  !   local definitions
  
      type(input_probdef_t) :: input_probdef
      type(sysmatrix_t) :: sysmatrix
      type(sysvector_t), dimension(size(sol)) :: rhsd, sol_old
      type(coefficients_t) :: coefficients
      type(lu_ma57_t) :: lu_ma57
  
  !   constants
  
      integer, parameter :: &
        uintpl = 6,         & ! scalar interpolation
        gauss =  8,          & ! 6-point Gauss integration of triangles
        gaussb = 3            ! 3 point integration of boundary elements
  
      integer :: i
  
      real(dp), parameter :: &
        alpha = 1._dp     ! diffusion coefficient. Not used in laplace_elem_ale
                          ! Only for compatibility with poisson_elem.
  
  
      call create_coefficients ( coefficients, ncoefi=100, ncoefr=50 )
  
      coefficients%i = 0
      coefficients%i(1) = uintpl
      coefficients%i(10:11) = (/ gauss, gaussb /)
  
      coefficients%r(1) = alpha
      coefficients%r(2:) = 0
  
      if ( .not. problem%created ) then
  
  !     problem definition
  
        call create_input_probdef ( mesh, input_probdef )
  
        do i = 1, mesh%nelgrp
          input_probdef%elementdof(i)%a = 1
        end do   
        call define_essential ( mesh, input_probdef, surface1=1, surface2 = 5)
        call define_essential ( mesh, input_probdef, surface1 = 7)  
        call problem_definition ( input_probdef, mesh, problem )
  
      end if
  
  !   create old solution vectors and right-hand side
  
      call create ( problem, sol_old, rhsd )
  
      if ( .not. sol(1)%created ) then
        call create ( problem, sol )
      else
  !     copy mesh velocity to sol_old
        call copy ( sol, sol_old )
      end if
  
  
  !   create system matrix
  
      call create_sysmatrix_structure ( sysmatrix, mesh, problem, &
        symmetric=.true. )
      call create_sysmatrix_data ( sysmatrix )
  
  !   build (assemble) matrix and vector from elements 
  
      call build_system ( mesh, problem, sysmatrix, msysvector=rhsd, &
        elemsub=laplace_elem_ale, coefficients=coefficients )
  
  !   solve
  
      do i = 1, size(sol)
    
      !     fill solution vector with zero essential boundary conditions
      
        call fill_sysvector ( mesh, problem, sol(i), surface1=1, surface2=5, &
        value=0._dp )
  !     fill solution vector with essential boundary conditions
  
        call fill_sysvector ( mesh, problem, sol(i), surface1 = 7, value=velp(1,3) )
    
        call add_effect_of_essential_to_rhs ( problem, sysmatrix, sol(i),&
           rhsd(i) )
  
  !     solve the system
        call solve_system_ma57 ( sysmatrix, rhsd(i), sol(i), lu=lu_ma57 )
  
      end do
  !   update mesh
  
      do i = 1, size(sol)
  
        if ( method==1 ) then
  !       update mesh nodes with forward Euler
          mesh%coor(:,i) = mesh%coor(:,i) + sol(i)%u(problem%degfdperm(:,2))*dt
        else if ( method==2 ) then
  !       update mesh nodes with 2nd order Adams-Bashforth
          mesh%coor(:,i) = mesh%coor(:,i) + &
                dt*(3.0_dp*sol(i)%u(problem%degfdperm(:,2))/2.0_dp - &
                sol_old(i)%u(problem%degfdperm(:,2))/2.0_dp)
        end if
  
      end do
  
  !   delete all data including all allocated memory
    
      call delete ( coefficients )
      if (input_probdef%created) call delete ( input_probdef )
      call delete ( rhsd )
      call delete ( sysmatrix )
      call delete ( lu_ma57 )
  
    end subroutine update_mesh_nodes
    
  subroutine update_mesh_nodes_2D ( mesh, problem, disp )

!   coordinates of the mesh are updated after calling this routine
    type(mesh_t), intent(inout) :: mesh

!   Is created if empty on call and kept at output
    type(problem_t), intent(inout) :: problem

!   displacement of the particle
    real(dp), dimension(:), intent(in) :: disp


!   local definitions

    type(input_probdef_t) :: input_probdef
    type(sysmatrix_t) :: sysmatrix
    type(sysvector_t), dimension(size(disp)) :: rhsd, sol
    type(coefficients_t) :: coefficients
    type(lu_ma57_t) :: lu

!   constants

    integer, parameter :: &
      uintpl = 6,         & ! scalar interpolation
      gauss = 6,          & ! 8-point Gauss integration of triangles
      gaussb = 3            ! 3 point integration of boundary elements

    real(dp), parameter :: &
      alpha = 1._dp     ! diffusion coefficient. Not used in laplace_elem_ale 
                        ! Only for compatibility with poisson_elem.

    integer :: i


    call create_coefficients ( coefficients, ncoefi=100, ncoefr=50 )

    coefficients%i = 0
    coefficients%i(1) = uintpl
    coefficients%i(10:11) = (/ gauss, gaussb /)

    coefficients%r(1) = alpha
    coefficients%r(2:) = 0

    if ( .not. problem%created ) then

!     problem definition
      call create_input_probdef ( mesh, input_probdef )
      input_probdef%elementdof(1)%a = 1
      input_probdef%elementdof(2)%a = 1
      
      call define_essential ( mesh, input_probdef, curve1=1, curve2=4 )
      call define_essential ( mesh, input_probdef, curve1=5 )

      call problem_definition ( input_probdef, mesh, problem )

      call delete ( input_probdef )

    end if

!   create solution and right-hand side

    call create ( problem, sol, rhsd )

!   create system matrix

    call create_sysmatrix_structure ( sysmatrix, mesh, problem, &
      symmetric=.true. )
    call create_sysmatrix_data ( sysmatrix )

!   build (assemble) matrix and vector from elements 

    call build_system ( mesh, problem, sysmatrix, msysvector=rhsd, &
      elemsub=laplace_elem_ale, coefficients=coefficients )

!   solve

    do i = 1, size(sol)

!     fill solution vector with zero essential boundary conditions

      call fill_sysvector ( mesh, problem, sol(i), curve1=1, curve2=4, &
        value=0._dp )

!     fill solution vector with essential boundary conditions

      call fill_sysvector ( mesh, problem, sol(i), curve1=5, value=disp(i) )

      call add_effect_of_essential_to_rhs ( problem, sysmatrix, sol(i),&
         rhsd(i) )

!     solve the system
      call solve_system_ma57 ( sysmatrix, rhsd(i), sol(i), lu=lu )

    end do

!   update mesh
    do i = 1, size(sol)
      mesh%coor(:,i) = mesh%coor(:,i) + sol(i)%u(problem%degfdperm(:,2))
    end do

!   delete all data including all allocated memory
  
    call delete ( coefficients )
    call delete ( rhsd )
    call delete ( sysmatrix )
    call delete ( lu )

  end subroutine update_mesh_nodes_2D

end  module subs_magnetic_particle1_m
