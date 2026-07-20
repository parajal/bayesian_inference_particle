module subs_magnetic_particle3d_m

  use tfem_m
  use math_defs_m
  use stokes_elements_m
  use hsl_ma57_m
  use laplace_elements_ale_m
  
  implicit none

  real(dp), dimension(:,:), allocatable :: xp
  real(dp), dimension(:), allocatable :: rp
  real(dp) :: delp   
  integer :: ipart
  real(dp) :: force(3)
  real(dp),  dimension(:,:,:), allocatable, save  :: tau

  real(dp), save :: mmm1, mmm2

contains
 
! computes the magnetic force acting on the particle due to external magnetic field 

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
  integer :: N,  n1, n2,k

  real(dp)::alpha
  real(dp), dimension(:), allocatable, save :: H_sq 
  real(dp),  dimension(:,:), allocatable, save  ::  H, tmp1, tmp2 


!   set globals, gauss, shapefunctions, ...

  call set_stokes_elem ( mesh, problem, elgrp, elem, first, last, &
    coefficients, oldvectors )

!   allocate arrays

    if (first) then
      allocate ( u(ndf) )
      allocate ( H_sq(ninti) )
      allocate ( H(ninti, ndim) )
      allocate ( tmp1(ninti, ndf) )
      allocate ( tmp2(ndf, ndim) )
      allocate ( tau(ninti, ndim, ndim) )
      allocate ( work4(ninti,ndf) )

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

  !  gradient of the magnetic potential

     do i = 1, ndim
      H(:,i) = - matmul ( dphidx(:,:,i), u )
    end do

     do ip = 1, ninti
       H_sq(ip) = sum ( H(ip,:) * H(ip,:) )       
     end do

  !  Maxwell stress tensor

     do i = 1, ndim
        do j = i, ndim        
          tau(:,i,j) =  H(:,i) * H(:,j) 
          if (i == j) then       
            tau(:,i,j) = alpha * ( tau(:,i,j) - 0.5_dp * H_sq )
          else
            tau(:,i,j) = alpha * tau(:,i,j)
            tau(:,j,i) = tau(:,i,j)
          end if  
        end do
     end do

!     force vector

     do i = 1, ndim

      do ip = 1, ninti
        work4(ip,:) = matmul(dphidx(ip,:,:), tau(ip,:,i))
      end do

      n1 = (i-1)*ndf + 1
      n2 =  i   *ndf

      do N = n1, n2
        k = N - n1 + 1
        elemvec(N) = - sum ( work4(:,k) * detF * wg )
      end do

    end do

  end if


!   Matrix

      if (matrix) then
          
        elemmat = 0
  
      end if
  
    if (last ) then 

!       Unset globals, gauss, shapefunctions, ...
  
      call unset_stokes_elem ( last, coefficients )

!       Deallocate arrays

      deallocate( u )
      deallocate ( H )
      deallocate( H_sq )
      deallocate( tau )
      deallocate( tmp1 )
      deallocate( tmp2 )
      deallocate(work4)

    end if

end subroutine rhs_max

subroutine rhs_max_B ( mesh, problem, elgrp, elem, matrix, vector, &
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

end subroutine rhs_max_B

  ! routine for updating the mesh nodes in the ALE scheme (displacement based)

  subroutine update_mesh_nodes_full( mesh, problem, disp )

    !   input/output
    
    !   coordinates of the mesh are updated after calling this routine
        type(mesh_t), intent(inout) :: mesh
    
    !   Is created if empty on call and kept at output
        type(problem_t), intent(inout) :: problem
    
    !   displacement of the particle
        real(dp), dimension(:,:), intent(in) :: disp
    
    
    !   local definitions
    
        type(input_probdef_t) :: input_probdef
        type(sysmatrix_t) :: sysmatrix
        type(sysvector_t), dimension(size(disp,2)) :: rhsd, sol
        type(coefficients_t) :: coefficients
        type(lu_ma57_t) :: lu_ma57
    
    !   constants
    
        integer, parameter :: &
          uintpl = 6,         & ! scalar interpolation
          gauss = 8,          & ! 8-point Gauss integration of tets
          gaussb = 3            ! 3 point integration of boundary elements
    
        real(dp), parameter :: &
          alpha = 1._dp     ! diffusion coefficient. Not used in laplace_elem_ale 
                            ! Only for compatibility with poisson_elem.
    
        integer :: i,j, npart
        npart = size(disp,1)

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

          call define_essential ( mesh, input_probdef, surface1=1, surface2 = 6)
          do i = 1, npart
            call define_essential ( mesh, input_probdef, surface1=6+i )
          end do 
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
    
        do i = 1,size(sol)
    
    !     fill solution vector with zero essential boundary conditions
    
          call fill_sysvector ( mesh, problem, sol(i), surface1=1, surface2=6, &
            value=0._dp )
    
    !     fill solution vector with essential boundary conditions
            do j = 1, npart
              call fill_sysvector ( mesh, problem, sol(i), surface1=6+j, &
              value=disp(j,i) )
            end do
    
          call add_effect_of_essential_to_rhs ( problem, sysmatrix, sol(i),&
             rhsd(i) )
    
    !     solve the system
          call solve_system_ma57 ( sysmatrix, rhsd(i), sol(i), lu=lu_ma57 )
    
        end do

!   update mesh
        do i = 1, size(sol)
          mesh%coor(:,i) = mesh%coor(:,i) + sol(i)%u(problem%degfdperm(:,2))
        end do
    
    !   delete all data including all allocated memory
      
        call delete ( coefficients )
        call delete ( rhsd )
        call delete ( sysmatrix )
        call delete ( lu_ma57 )
    
    end subroutine update_mesh_nodes_full
    	
    ! for one particle
    subroutine update_mesh_nodes_sym_1p( mesh, problem, disp )

      !   input/output
      
      !   coordinates of the mesh are updated after calling this routine
          type(mesh_t), intent(inout) :: mesh
      
      !   Is created if empty on call and kept at output
          type(problem_t), intent(inout) :: problem
      
      !   displacement of the particle
          real(dp), dimension(:,:), intent(in) :: disp
      
      
      !   local definitions
      
          type(input_probdef_t) :: input_probdef
          type(sysmatrix_t) :: sysmatrix
          type(sysvector_t), dimension(size(disp,2)) :: rhsd, sol
          type(coefficients_t) :: coefficients
          type(lu_ma57_t) :: lu_ma57
      
      !   constants
      
          integer, parameter :: &
            uintpl = 6,         & ! scalar interpolation
            gauss = 8,          & ! 8-point Gauss integration of tets
            gaussb = 3            ! 3 point integration of boundary elements
      
          real(dp), parameter :: &
            alpha = 1._dp     ! diffusion coefficient. Not used in laplace_elem_ale 
                              ! Only for compatibility with poisson_elem.
      
          integer :: i,j
      
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
      
          do i = 1,2
      
      !     fill solution vector with zero essential boundary conditions
      
          call fill_sysvector ( mesh, problem, sol(i), surface1=1, surface2=5, &
            value=0._dp )
      !     fill solution vector with essential boundary conditions
      
            call fill_sysvector ( mesh, problem, sol(i), surface1 = 7, value=disp(1,i) )
        
            call add_effect_of_essential_to_rhs ( problem, sysmatrix, sol(i),&
               rhsd(i) )
      
      !     solve the system
            call solve_system_ma57 ( sysmatrix, rhsd(i), sol(i), lu=lu_ma57 )
      
          end do
      
      !   update mesh
          mesh%coor(:,1) = mesh%coor(:,1) + sol(1)%u(problem%degfdperm(:,2))
          mesh%coor(:,2) = mesh%coor(:,2) + sol(2)%u(problem%degfdperm(:,2))
  
      !   delete all data including all allocated memory
        
          call delete ( coefficients )
          call delete ( rhsd )
          call delete ( sysmatrix )
          call delete ( lu_ma57 )
      
      end subroutine update_mesh_nodes_sym_1p

      subroutine update_mesh_nodes_sym_2p( mesh, problem, disp )

        !   input/output
        
        !   coordinates of the mesh are updated after calling this routine
            type(mesh_t), intent(inout) :: mesh
        
        !   Is created if empty on call and kept at output
            type(problem_t), intent(inout) :: problem
        
        !   displacement of the particle
            real(dp), dimension(:,:), intent(in) :: disp
        
        
        !   local definitions
        
            type(input_probdef_t) :: input_probdef
            type(sysmatrix_t) :: sysmatrix
            type(sysvector_t), dimension(size(disp,2)) :: rhsd, sol
            type(coefficients_t) :: coefficients
            type(lu_ma57_t) :: lu_ma57
        
        !   constants
        
            integer, parameter :: &
              uintpl = 6,         & ! scalar interpolation
              gauss = 8,          & ! 8-point Gauss integration of tets
              gaussb = 3            ! 3 point integration of boundary elements
        
            real(dp), parameter :: &
              alpha = 1._dp     ! diffusion coefficient. Not used in laplace_elem_ale 
                                ! Only for compatibility with poisson_elem.
        
            integer :: i,j
        
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
              call define_essential ( mesh, input_probdef, surfaces = [7,8])
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
        
            do i = 1,2
        
        !     fill solution vector with zero essential boundary conditions
        
            call fill_sysvector ( mesh, problem, sol(i), surface1=1, surface2=5, &
              value=0._dp )
        !     fill solution vector with essential boundary conditions
        
              call fill_sysvector ( mesh, problem, sol(i), surface1 = 7, value=disp(1,i) )
              call fill_sysvector ( mesh, problem, sol(i), surface1 = 8, value=disp(2,i) )
              call add_effect_of_essential_to_rhs ( problem, sysmatrix, sol(i),&
                 rhsd(i) )
        
        !     solve the system
              call solve_system_ma57 ( sysmatrix, rhsd(i), sol(i), lu=lu_ma57 )
        
            end do
        
        !   update mesh
            mesh%coor(:,1) = mesh%coor(:,1) + sol(1)%u(problem%degfdperm(:,2))
            mesh%coor(:,2) = mesh%coor(:,2) + sol(2)%u(problem%degfdperm(:,2))
    
        !   delete all data including all allocated memory
          
            call delete ( coefficients )
            call delete ( rhsd )
            call delete ( sysmatrix )
            call delete ( lu_ma57 )
        
        end subroutine update_mesh_nodes_sym_2p

   ! rigid body constraint on whole particle domain 
   subroutine elementc_3D ( mesh, problem, constr, elem, node, &
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

    real(dp) :: xr(1,3), r(3)

    xr(1,:) = mesh%coor(node,:)

!   set shape function in the point
                                                                                
    if ( vector ) then
                                                                                
      elemvec = 0
      elemvecadd = 0
                                                                                
    end if

    if ( matrix ) then

      elemmat(:,:) = 0._dp
      elemmat(1,1) = 1._dp
      elemmat(2,2) = 1._dp
      elemmat(3,3) = 1._dp

      r = xr(1,:) - xp(1,:)

      elemmatadd(1,:) = (/ -1._dp,   0._dp,   0._dp,  0._dp,  -r(3),   r(2) /)
      elemmatadd(2,:) = (/  0._dp,  -1._dp,   0._dp,   r(3),  0._dp,  -r(1) /)
      elemmatadd(3,:) = (/  0._dp,   0._dp,  -1._dp,  -r(2),   r(1),  0._dp /)

    end if

  end subroutine elementc_3D

  ! subroutine for freely floating particles for 1st particle (3D)

  subroutine elementc_full ( mesh, problem, constr, elem, node, &
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

    integer :: nsurfaceconstr
    real(dp) :: xr(1,3), r(3)

    nsurfaceconstr = problem%constraints(constr)%geometry1

    xr(1,:) = mesh%coor(mesh%surfaces(nsurfaceconstr)%nodes(node),:)
 
!   set shape function in the point
                                                                                
    if ( vector ) then
                                                                                
      elemvec = 0
      elemvecadd = 0 
      
      if ( first ) then
        if (ipart == 1) then 
        elemvecadd(1:3) = force
        else 
          elemvecadd(1:3) = -force
        end if 
      end if    
      
    end if

    if ( matrix ) then

      elemmat(:,:) = 0._dp
      elemmat(1,1) = 1._dp
      elemmat(2,2) = 1._dp
      elemmat(3,3) = 1._dp

      r = xr(1,:) - xp(ipart,:)

      elemmatadd(1,:) = (/ -1._dp,   0._dp,   0._dp,  0._dp,  -r(3),   r(2) /)
      elemmatadd(2,:) = (/  0._dp,  -1._dp,   0._dp,   r(3),  0._dp,  -r(1) /)
      elemmatadd(3,:) = (/  0._dp,   0._dp,  -1._dp,  -r(2),   r(1),  0._dp /)

    end if

  end subroutine elementc_full

  subroutine elementc_full_force ( mesh, problem, constr, elem, node, &
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

    integer :: nsurfaceconstr
    real(dp) :: xr(1,3), r(3)

    nsurfaceconstr = problem%constraints(constr)%geometry1

    xr(1,:) = mesh%coor(mesh%surfaces(nsurfaceconstr)%nodes(node),:)
 
!   set shape function in the point
                                                                                
    if ( vector ) then
                                                                                
      elemvec = 0
      elemvecadd = 0
      if ( first ) then
        if (ipart == 1) then 
        elemvecadd(1:3) = force
        else 
          elemvecadd(1:3) = -force
        end if 
      end if                                                                          
    end if

    if ( matrix ) then

      elemmat(:,:) = 0._dp
      elemmat(1,1) = 1._dp
      elemmat(2,2) = 1._dp
      elemmat(3,3) = 1._dp

      r = xr(1,:) - xp(ipart,:)

      elemmatadd(1,:) = (/ -1._dp,   0._dp,   0._dp,  0._dp,  -r(3),   r(2) /)
      elemmatadd(2,:) = (/  0._dp,  -1._dp,   0._dp,   r(3),  0._dp,  -r(1) /)
      elemmatadd(3,:) = (/  0._dp,   0._dp,  -1._dp,  -r(2),   r(1),  0._dp /)

    end if

  end subroutine elementc_full_force

  subroutine elementc_sym_1p(mesh, problem, constr, elem, node, &
    matrix, vector, first, last, coefficients, oldvectors, elemmat, elemmat2, &
    elemmatadd, elemvec, elemvecadd)

    type(mesh_t), intent(in) ::mesh
    type(problem_t), intent(in) :: problem
    integer, intent(in) :: constr, elem, node
    logical, intent(in) :: matrix, vector, first, last
    type(coefficients_t), intent(in) :: coefficients
    type(oldvectors_t), intent(in) :: oldvectors 
    real(dp), intent(out), dimension(:,:) :: elemmat, elemmat2, elemmatadd
    real(dp), intent(out), dimension(:) :: elemvec, elemvecadd

    logical :: rotate = .false.
    integer :: nsurfaceconstr
    real(dp) :: xr(1,3), r(3)
    nsurfaceconstr = problem%constraints(constr)%geometry1

    xr(1,:) = mesh%coor(mesh%surfaces(nsurfaceconstr)%nodes(node),:)

    if (vector) then
      elemvec = 0
      elemvecadd = 0
    end if

   if ( matrix ) then
    elemmat(:,:) = 0._dp
    elemmat(1,1) = 1._dp
    elemmat(2,2) = 1._dp
    r = xr(1,:) - xp(1,:) 
    if ( rotate ) then
      elemmatadd(1,:) = (/r(2) /)
      elemmatadd(2,:) = (/ -r(1) /)
    else
      elemmatadd(1,:) = (/ -1._dp,0._dp, r(2) /)
      elemmatadd(2,:) = (/0._dp,-1._dp, -r(1) /)
    end if 
  end if
end subroutine elementc_sym_1p

subroutine elementc_sym_1p_force(mesh, problem, constr, elem, node, &
  matrix, vector, first, last, coefficients, oldvectors, elemmat, elemmat2, &
  elemmatadd, elemvec, elemvecadd)

  type(mesh_t), intent(in) ::mesh
  type(problem_t), intent(in) :: problem
  integer, intent(in) :: constr, elem, node
  logical, intent(in) :: matrix, vector, first, last
  type(coefficients_t), intent(in) :: coefficients
  type(oldvectors_t), intent(in) :: oldvectors 
  real(dp), intent(out), dimension(:,:) :: elemmat, elemmat2, elemmatadd
  real(dp), intent(out), dimension(:) :: elemvec, elemvecadd

  logical :: rotate = .false.
  integer :: nsurfaceconstr
  real(dp) :: xr(1,3), r(3)
  nsurfaceconstr = problem%constraints(constr)%geometry1

  xr(1,:) = mesh%coor(mesh%surfaces(nsurfaceconstr)%nodes(node),:)

  if (vector) then
    elemvec = 0
    elemvecadd = 0
    if ( first ) elemvecadd(1:3) = force
  end if

 if ( matrix ) then
  elemmat(:,:) = 0._dp
  elemmat(1,1) = 1._dp
  elemmat(2,2) = 1._dp
  r = xr(1,:) - xp(1,:) 
  if ( rotate ) then
    elemmatadd(1,:) = (/r(2) /)
    elemmatadd(2,:) = (/ -r(1) /)
  else
    elemmatadd(1,:) = (/ -1._dp,0._dp, r(2) /)
    elemmatadd(2,:) = (/0._dp,-1._dp, -r(1) /)
  end if 
end if
end subroutine elementc_sym_1p_force

subroutine elementc_sym_2p(mesh, problem, constr, elem, node, &
  matrix, vector, first, last, coefficients, oldvectors, elemmat, elemmat2, &
  elemmatadd, elemvec, elemvecadd)

  type(mesh_t), intent(in) ::mesh
  type(problem_t), intent(in) :: problem
  integer, intent(in) :: constr, elem, node
  logical, intent(in) :: matrix, vector, first, last
  type(coefficients_t), intent(in) :: coefficients
  type(oldvectors_t), intent(in) :: oldvectors 
  real(dp), intent(out), dimension(:,:) :: elemmat, elemmat2, elemmatadd
  real(dp), intent(out), dimension(:) :: elemvec, elemvecadd

  logical :: rotate = .false.
  integer :: nsurfaceconstr
  real(dp) :: xr(1,3), r(3)
  nsurfaceconstr = problem%constraints(constr)%geometry1

  xr(1,:) = mesh%coor(mesh%surfaces(nsurfaceconstr)%nodes(node),:)

  if (vector) then
    elemvec = 0
    elemvecadd = 0 
  end if

 if ( matrix ) then
  elemmat(:,:) = 0._dp
  elemmat(1,1) = 1._dp
  elemmat(2,2) = 1._dp
  r = xr(1,:) - xp(ipart,:) 
  if ( rotate ) then
    elemmatadd(1,:) = (/r(2) /)
    elemmatadd(2,:) = (/ -r(1) /)
  else
    elemmatadd(1,:) = (/ -1._dp,0._dp, r(2) /)
    elemmatadd(2,:) = (/0._dp,-1._dp, -r(1) /)
  end if 
end if
end subroutine elementc_sym_2p

subroutine elementc_sym_2p_force(mesh, problem, constr, elem, node, &
  matrix, vector, first, last, coefficients, oldvectors, elemmat, elemmat2, &
  elemmatadd, elemvec, elemvecadd)

  type(mesh_t), intent(in) ::mesh
  type(problem_t), intent(in) :: problem
  integer, intent(in) :: constr, elem, node
  logical, intent(in) :: matrix, vector, first, last
  type(coefficients_t), intent(in) :: coefficients
  type(oldvectors_t), intent(in) :: oldvectors 
  real(dp), intent(out), dimension(:,:) :: elemmat, elemmat2, elemmatadd
  real(dp), intent(out), dimension(:) :: elemvec, elemvecadd

  logical :: rotate = .false.
  integer :: nsurfaceconstr
  real(dp) :: xr(1,3), r(3)
  nsurfaceconstr = problem%constraints(constr)%geometry1

  xr(1,:) = mesh%coor(mesh%surfaces(nsurfaceconstr)%nodes(node),:)

  if (vector) then
    elemvec = 0
    elemvecadd = 0
    if (first) then 
    if (ipart == 1) then 
      elemvecadd(1:3) = force
      else 
        elemvecadd(1:3) = -force
      end if 
    end if   
      end if

 if ( matrix ) then
  elemmat(:,:) = 0._dp
  elemmat(1,1) = 1._dp
  elemmat(2,2) = 1._dp
  r = xr(1,:) - xp(ipart,:) 
  if ( rotate ) then
    elemmatadd(1,:) = (/r(2) /)
    elemmatadd(2,:) = (/ -r(1) /)
  else
    elemmatadd(1,:) = (/ -1._dp,0._dp, r(2) /)
    elemmatadd(2,:) = (/0._dp,-1._dp, -r(1) /)
  end if 
end if
end subroutine elementc_sym_2p_force

  ! Element for the constraints (connection through collocation) to specify a
  ! potential difference between two boundaries.

  subroutine magnetic_node_conn ( mesh, problem, constr, elem, node, &
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

    integer :: i

!   connection through collocation
    elemmat  = 0
    do i = 1, size(elemmat,1)
      elemmat(i,i) = 1
    end do
    elemmat2 = - elemmat

    elemvec  = 0

    do i = 1, size(elemvec)
      elemvec(i) = -delp
    end do
end subroutine magnetic_node_conn

! compute H :grad(u)


subroutine integrate_maxwell_stress ( mesh, problem, geometry, &
  elem, first, last, coefficients, oldvectors, elemvec )

  use stokes_globals_m

  type(mesh_t), intent(in) :: mesh
  type(problem_t), intent(in) :: problem
  integer, intent(in) :: geometry, elem
  logical, intent(in) :: first, last
  type(coefficients_t), intent(in) :: coefficients
  type(oldvectors_t), intent(in) :: oldvectors
  real(dp), intent(out), dimension(:) :: elemvec

  integer :: i, j, k, ip
  integer, save :: ns
  real(dp), dimension(:), allocatable, save :: H_sq
  real(dp):: mu(2)
  real(dp),  dimension(:,:), allocatable, save  ::  H

  mu(1) = mmm1
  mu(2) = mmm2  

  if ( first ) then

!     first element on this geometry

    call check ( coefficients, 'stokes_integrate_stress_geometry', &
      indexarray=[36], minimum=[-1], maximum=[1] )

!     set globals

    call set_globals_stokes_vp_boun ( mesh, coefficients, &
      ndimr=mesh%ndim-1, geometry=geometry )

    !ns = ndim*(ndim+1)/2
    ns = ndim
!     allocate arrays

    allocate ( wg(ninti), surfl(ninti), normal(ninti,ndim) )
    allocate ( xig(ninti,ndim-1), x(nodalp,ndim), xg(ninti,ndim))
    allocate ( phi(ninti,ndf), dphi(ninti,ndf,ndim-1) )
    allocate ( dxdxis(ninti,ndim,ndim-1) )
    allocate ( st(ndf*ns), work2(ndf,ns), work4(ninti,ndim), tmp(ndim,ndim) )
    allocate ( tauten(ninti,ndim,ndim) )
    allocate(H(ninti,ndim))
    allocate ( H_sq(ninti) )

!     set Gauss integration and shape function

    call set_Gauss_integration ( gauss, xig, wg )

    call set_shape_function ( shapefunc, xig, phi, dphi )

  end if

  call get_coordinates_geometry ( mesh, elem, x, ndimr=ndim-1, &
    geometry=geometry )

  call isoparametric_deformation_curved ( x, dphi, dxdxis, surfl, normal )

  xg = matmul ( phi, x ) 

  if ( coorsys == 1 ) then
    write(*,'(/a/a/)') 'stokes_integrate_stress_geometry:', &
      ' Axisymmetric coordinates not yet implemented'
    stop
  end if

  do k = 1, 2

!   stokes stress tensor 

    call get_vector_geometry ( mesh, problem, oldvectors%v(k)%p, elem, st, &
     ndimr=ndim-1, geometry=geometry, layer=layer )

    work2 = reshape ( st, (/ndf,ns/) )

    do i = 1, ndim
      H(:,i) = -matmul ( phi, work2(:,i) )
    end do

    do ip = 1, ninti
      H_sq(ip) = sum ( H(ip,:) * H(ip,:) )       
    end do

!    Maxwell stress tensor

    do i = 1, ndim
       do j = i, ndim        
         tauten(:,i,j) =  H(:,i) * H(:,j) 
         if (i == j) then       
           tauten(:,i,j) =  mu(k) * ( tauten(:,i,j) - 0.5_dp * H_sq )
         else
           tauten(:,i,j) =  mu(k) * tauten(:,i,j)
           tauten(:,j,i) = tauten(:,i,j)
         end if  
       end do
    end do

!     evaluate in each integration point: 2*eta*D.n

    do ip = 1, ninti
      work4(ip,:) = matmul( tauten(ip,:,:), normal(ip,:) )
    end do

!     evaluate in each integration point: (2*eta*D.n)x and integrate
    if ( k == 1 ) then
       do j = 1, ndim
         elemvec(j) = sum ( work4(:,j)  * surfl * wg )
       end do
    else
       do j = 1, ndim
         elemvec(j) = elemvec(j) - sum ( work4(:,j)  * surfl * wg )
       end do
    end if
   end do
 

!   element vector

  ! elemvec = reshape ( transpose(tmp), [ndim] )

  if ( last ) then

!     last element on this surface

    deallocate ( wg, surfl, normal )
    deallocate ( xig, x, xg )
    deallocate ( phi, dphi, dxdxis)
    deallocate ( st, work2, work4, tmp )
    deallocate ( tauten )
    deallocate(H)
    deallocate(H_sq)

  end if

end subroutine integrate_maxwell_stress

subroutine add_boundary ( mesh, problem, geometry, elem, &
  matrix, vector, first, last, coefficients, oldvectors, elemmat, elemvec )

  use stokes_globals_m

  type(mesh_t), intent(in) :: mesh
  type(problem_t), intent(in) :: problem
  integer, intent(in) :: geometry, elem
  logical, intent(in) :: matrix, vector, first, last
  type(coefficients_t), intent(in) :: coefficients
  type(oldvectors_t), intent(in) :: oldvectors
  real(dp), intent(out), dimension(:,:) :: elemmat
  real(dp), intent(out), dimension(:) :: elemvec

  integer :: i, j, k, ip, N
  real(dp), dimension(:), allocatable, save :: H_sq
  real(dp),  dimension(:,:), allocatable, save  ::  H

  if ( first ) then

!     set globals

    call set_globals_stokes_vp_boun ( mesh, coefficients, &
    ndimr=mesh%ndim-1, geometry=geometry )

!     allocate arrays

    allocate ( wg(ninti), fg(ninti,ndim), surfl(ninti) )
    allocate ( tmp(ndf,ndim) )
    allocate ( normal(ninti,ndim) )
    allocate ( xig(ninti,ndim-1), phi(ninti,ndf), x(nodalp,ndim) )
    allocate ( xg(ninti,ndim) )
    allocate ( dphi(ninti,ndf,ndim-1), dxdxis(ninti,ndim,ndim-1) )
    allocate ( tauten(ninti,ndim,ndim) )
    allocate ( work4(ninti,ndim) )
    allocate(H(ninti,ndim))
    allocate ( H_sq(ninti) )
    allocate ( st(ndf*ndim), work2(ndf,ndim))

!     set Gauss integration and shape function

    call set_Gauss_integration ( gauss, xig, wg )

    call set_shape_function ( shapefunc, xig, phi, dphi )

  end if

  call get_coordinates_geometry ( mesh, elem, x, ndimr=ndim-1, &
  geometry=geometry )
    call isoparametric_deformation_curved ( x, dphi, dxdxis, surfl, normal )


    xg = matmul ( phi, x ) 


  if ( vector ) then

    elemvec = 0


!   stokes stress tensor 
    call get_vector_geometry ( mesh, oldvectors%p(1)%p, oldvectors%v(1)%p, elem, st, &
    ndimr=ndim-1, geometry=geometry, layer=layer )

    work2 = reshape ( st, (/ndf,ndim/) )

      do i = 1, ndim
        H(:,i) = -matmul ( phi, work2(:,i) )
      end do
  
      do ip = 1, ninti
        H_sq(ip) = sum ( H(ip,:) * H(ip,:) )       
      end do

  !    Maxwell stress tensor
  
      do i = 1, ndim
         do j = i, ndim        
           tauten(:,i,j) =  H(:,i) * H(:,j) 
           if (i == j) then       
             tauten(:,i,j) =  mmm1 * ( tauten(:,i,j) - 0.5_dp * H_sq )
           else
             tauten(:,i,j) =  mmm1 * tauten(:,i,j)
             tauten(:,j,i) = tauten(:,i,j)
           end if  
         end do
      end do

  !     evaluate in each integration point: T.n
  
      do ip = 1, ninti
        work4(ip,:) = matmul( tauten(ip,:,:), normal(ip,:) )
      end do

    do j = 1, ndim
    do N = 1, ndf
      tmp(N,j) =  sum ( work4(:,j) * phi(:,N) * surfl * wg )
    end do
    end do

      elemvec =  reshape ( tmp, (/ndim*ndf/) )
    end if 


      if (matrix) then
          
        elemmat = 0
  
      end if
  
  if ( last ) then

!     last element on this geometry
    deallocate ( wg, fg, surfl, normal )
    deallocate ( tmp )
    deallocate ( xig, phi, x )
    deallocate ( xg )
    deallocate ( dphi, dxdxis )
    deallocate ( tauten )
    deallocate ( work4 )
    deallocate(H)
    deallocate(H_sq)
    deallocate(work2, st)

  end if

end subroutine add_boundary

end  module subs_magnetic_particle3d_m
