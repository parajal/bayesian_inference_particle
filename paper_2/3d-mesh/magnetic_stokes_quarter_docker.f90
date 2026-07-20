! Quarter mesh
! magnetic = .true. to impose magnetic field 
! Otherwise impose body force 
! input : (H, init_dist, force), output: velocity of the particle

program magnetic_particle1

  use tfem_m
  use math_defs_m
  use hsl_ma41_m
  use stokes_elements_m
  use poisson_functions_m
  use poisson_elements_m
  use io_utils_m
  use subs_magnetic_particle4_m
  use timer_m
  use metis5_m
  use limits_m

  implicit none

! constants

  integer, parameter ::     &
    uintpl = 6,             & ! P2 velocities
    pintpl = 2,             & ! P1 pressures
    physqvel = 1,           & ! physical quantity nr of the velocities
    physqpress = 2,         & ! physical quantity nr of the pressures
    gauss = 8,              & ! 6-point Gauss integration of triangles
    gaussb = 3,             & ! 3 point integration of boundary elements
    funcnr = 1,             & ! function number for the right-hand side
    ndim = 3 


! definitions for magnetic and flow problems
  
  type(mesh_t) :: mesh
  type(input_probdef_t) :: input_probdef, input_probdefp
  type(problem_t), target :: problem, problemp
  type(sysmatrix_t) :: sysmatrix, sysmatrixp
  type(sysvector_t), target :: sol, solp
  type(sysvector_t) :: rhsd, rhsdp
  type(vector_t):: velocity, pressure
  type(oldvectors_t) ::  oldvectors_max, oldvectors
  type(vector_t),target :: grad
  type(coefficients_t) :: coefficients, coefficientsp(2), coefficients_sp(2)
  type(problem_t) :: problem_lapl
  type(solver_options_ma41_t) :: solver_options_p  
  type(solver_options_ma41_t) :: solver_options_up

! material variables 
  integer :: nobj, numtimesteps, vtkevery
  integer :: i, step, ipost=0

  real(dp) ::                &
  eta = 1._dp,             & ! fluid viscosity
  alpha1 = 1._dp ,  &          ! magnetic permeability of fluid
  alpha2 = 2._dp,   &     !magnetic permeability of particle
  time = 0._dp,             & ! initial time step
  deltat = 0.4_dp,          & ! time step
  rs_p = 2._dp,             &
  is_p = 2._dp,             &
  rs_up = 2._dp,            & ! real_storage velocity-pressure LU (HSL)
  is_up = 2._dp,            & ! integer_storage velocity-pressure LU (HSL)  
  rc = 1._dp,                 &          ! radius of the two particles (mu m)
  n = 10._dp,                 & ! number of elements in  and y direction
  nc = 20._dp,              & ! number of elements on the circle 
  volume_threshold = 1.39_dp,& ! remeshing volume threshold
  aspect_ratio_threshold = 1.39_dp ! remeshing aspect ratio threshold

! variables 

  real(dp) :: nelem_min
  real(dp) ::   delta, init_dist, H, force1, angle
  real(dp) ::  lx, ly, lz, dx_part, dx_box
  real(dp) :: new_dist, beta1, char_velocity
  real(dp) :: up(1,3), unm1(1,3), un(1,3)
  real(dp) :: xpn(1,ndim), pp(1,3), gap_par

  real(dp), dimension(:), allocatable :: init_volume, volumev
  real(dp), dimension(:), allocatable :: init_aspect_ratio, aspect_ratio
  real(dp) :: norm_volume, norm_aspect_ratio
  real(dp), allocatable :: coor(:,:)
  real(dp), dimension(:,:), allocatable :: refinement_coor,refinement_coor1,refinement_coor2,refinement_coor3  
  integer :: nrefine

! mesh variables  

  character(len=30) :: filename
  logical :: remeshing=.false., initial_mesh
  logical :: magnetic = .true.
  logical :: periodic = .true.

  namelist /comppar/ eta, alpha1, alpha2, force1,deltat, &
  delta, angle, init_dist,nobj,rc, dx_box,dx_part, lx, ly, lz, &
  nelem_min, H, magnetic, periodic, &
  numtimesteps, vtkevery, volume_threshold, &
  aspect_ratio_threshold

  read ( unit=*, nml=comppar )

  ! Let the OMP_NUM_THREADS environment variable govern threading
  ! (solver libraries and any OpenMP regions). Hardcoding 4 threads here
  ! prevents users from scaling up on bigger nodes.

  call tic_w
  call execute_command_line ('rm *.vtk' )

  ! open particle trajectory file once; close after the time loop.
  open(unit=12, file='particle.txt', status='unknown', position='append')

! fill coefficients ( flow problem )
  call create_coefficients ( coefficients, ncoefi=100, ncoefr=50 )

  coefficients%i(1:11) = &
    (/ uintpl,   pintpl,     0,     0,         0,  &
       physqvel, physqpress, 0,     0,     gauss,  &
       gaussb /)
       
  coefficients%i(12:) = 0

  coefficients%r(1) = eta
  coefficients%r(2:) = 0

if (magnetic) then 
  call create ( coefficientsp, ncoefi=100, ncoefr=50 )
! fluid domain ( elgroup = 1 )

  coefficientsp(1)%i = 0
  coefficientsp(1)%i(1) = uintpl
  coefficientsp(1)%i(10:12) = (/ gauss, gaussb, funcnr /)
  
  coefficientsp(1)%r(1) = alpha1
  coefficientsp(1)%func => func

! particle domain ( elgroup = 2 )

  coefficientsp(2) = coefficientsp(1)
  coefficientsp(2)%r(1) = alpha2

! fill coefficients ( magnetic - flow coupling problem ) 
  coefficients_sp = coefficients
  coefficients_sp(1)%r(1) = alpha1
  coefficients_sp(2)%r(1) = alpha2

end if 

print *, 'cos(angle)', cos(angle * pi / 180.0_dp)
if (.not. magnetic) then
  force = 0.5_dp * (/ force1 *cos(angle * pi / 180.0_dp), &
                      force1 *sin(angle * pi / 180.0_dp), &
                      0._dp /)
end if

  allocate(xp(1,3),rp(nobj))

  new_dist = init_dist

  if (nobj == 2) then 
    xp(1,1) = (lx/2._dp - init_dist/2._dp)
  else 
    xp(1,1) = 0._dp
  end if   
  if (delta == 0) then 
    xp(1,2) = 0._dp
  else 

    xp(1,2) =  -ly/2._dp + delta + rc
  end if  
  
  xp(1,3) = 0._dp 

  pp = xp
  rp(:) = rc                ! particle radius

  print *, 'dx_part', dx_part
  print *, 'dx_box', dx_box
  MAXNUMWARNINGS = 0
  WARN_ON_NO_DATA_IN_NODES = .false.
  
    call generate_read_mesh  
  print *, 'number of nodes', mesh%nnodes
  print *, 'number of elements ', mesh%nelem
  call toc_w('Generating mesh')
  
if (magnetic) then 
    delp = H * 2._dp * lx 
    print *, 'H-field', H

    ! define magnetic problem
  call define_magnetic_problem
  call toc_w('define magnetic problem')

  ! define oldvectors
  call create ( oldvectors_max, nsysvec=1, nprob=1)
  oldvectors_max%s(1)%p => solp
  oldvectors_max%p(1)%p => problemp

end if 

! define flow problem

  call define_flow_problem
  call toc_w('define flow problem')

  time = 0

  do step = 1, numtimesteps  
  
    if ( step >= 2 ) then
      
      time = time+deltat
      
      if ( step == 2 ) then

      ! advance particles with forward Euler
          un = up
          xpn = xp
          xp(1,1:2) = xp(1,1:2) + deltat * un(1,1:2)
      
      else
      ! advance particle positions with 2nd order Adams-Bashforth
          unm1 = un
          un = up
          xpn = xp
          xp(1,1:2) = xp(1,1:2) + deltat*(3*un(1,1:2)/2 - unm1(1,1:2)/2)
  
      end if

    ! update mesh nodes: solve a Laplace's equation
    call update_mesh_nodes_sym_1p ( mesh, problem_lapl, xp(:,1:2)-xpn(:,1:2)) 
    end if 

    call find_bounds_blocks ( mesh )

!   compute current aspect ratio

    call compute_element_aspect_ratio ( mesh, volumev, aspect_ratio )

!   compute maximum normalized aspect ratio and volume

    norm_volume = maxval( abs(log(volumev/init_volume)) )
    norm_aspect_ratio = maxval( abs(log(aspect_ratio/init_aspect_ratio)) )

    print *, '--------------------------------------------'
    print *, 'Mesh quality metrics:'
    print *, '  norm_volume         = ', norm_volume, ' (threshold = ', volume_threshold, ')'
    print *, '  norm_aspect_ratio   = ', norm_aspect_ratio, ' (threshold = ', aspect_ratio_threshold, ')'
    print *, '--------------------------------------------'

  ! remeshing criterion

    remeshing = .false.
    if ( ( norm_volume >= volume_threshold .or. &
      norm_aspect_ratio >= aspect_ratio_threshold ) ) then
      remeshing = .true.
    end if

    if ( remeshing ) then
      print *,'Remeshing and projection...'
      
      call delete_old_problems 

      ! generate new mesh
      call generate_read_mesh
    
      ! Reset normalized metrics after remeshing to prevent continuous remeshing
      norm_volume = 0._dp
      norm_aspect_ratio = 0._dp
    
      if (magnetic) then 
        call define_magnetic_problem
      end if 

      call define_flow_problem
      call toc_w('remeshing')

    end if 

    if (magnetic) then 
      call solve_magnetic_problem
      call toc_w('solving magnetic problem')
    end if 
      call solve_flow_problem
      call toc_w('solving flow problem')

       call get_sysvector_constraint ( mesh, problem, sol, &
       constraint= 1, addunknowns=.true., u=up(1,:) )

       new_dist = lx - 2*xp(1,1)
       beta1 = 3 * (alpha2 - alpha1)/(alpha2 + 2 * alpha1)
       char_velocity =  4*beta1**2*H**2/(9*eta*new_dist**4)
       print *, 'step = ', step
       print *, 'new_dist = ', new_dist
       print *, 'pp = ', xp(1,1:3)
       print *, 'up = ', up(1,1:3)
       print *, 'characteristic velocity =', char_velocity
       print *,  'normalized volume = ', norm_volume
       print *, 'normalized aspect ratio = ', norm_aspect_ratio       


       write(11,'(9es16.8)') time, xp(1,1:3), up(1,1:3), new_dist, char_velocity

       if (new_dist <= 2.1_dp) then 
        stop 
       end if 

       write(12,'(9es16.8)') time, xp(1,1:3), up(1,1:3)
	   
       if ( vtkevery > 0 ) then 
        if ( mod(step,vtkevery) == 0 ) then
          ipost = ipost + 1
          call create ( oldvectors, nsysvec=1)     
    
          call create_vector ( problem, velocity, physq=1)
          call create_vector ( problem, pressure, vec = 1 )
    
          call extract_physvector ( mesh, problem, sol, velocity)
          ! derive the pressure in all nodes
          oldvectors%s(1)%p => sol
          call derive_vector ( mesh, problem, pressure, elemsub=stokes_pressure,coefficients=coefficients, &  
          oldvectors=oldvectors,elgroup1 = 1 )
          write(filename,'(a,i4.4,a)') 'velocity', ipost, '.vtk'
          call write_vector_vtk ( mesh, problem, filename=filename, &
          dataname='velocity',vector=velocity, groups=[1] )
          write(filename,'(a,i4.4,a)') 'pressure', ipost, '.vtk'  
          call write_scalar_vtk ( mesh, problem, vector=pressure, &
          dataname='pressure',  filename=filename, groups=[1] )

          call delete(pressure)
          call delete ( velocity )
          call delete(oldvectors)
        end if
      end if

end do

close(unit = 11)
close(unit = 12)

call delete_old_problems

contains

! generate and read mesh

subroutine generate_read_mesh

  call write_gmsh_parameters (  lx, ly, lz, xp, rp, dx_box, &
  dx_part )

  ! HXT is a parallel 3D Delaunay algorithm; honors OMP_NUM_THREADS.
  call execute_command_line ( 'gmsh -3 -order 2 -algo hxt -o mesh.msh &
  &mesh.geo > outputmesh.out' )

  ! read mesh generated by gmsh 
  call read_mesh_gmsh ( mesh, filename='mesh.msh', ndim=3, &
  physgeom=.true. )

  if(periodic) then 
    call add_to_mesh ( mesh, matchingsurface=[1,3], replace=1, &
    displacement=[0._dp,-ly,0._dp] )
  end if 
  
  call fill_mesh_parts ( mesh )

  ! allocate aspect ratio arrays
  allocate ( init_volume(mesh%nelem), volumev(mesh%nelem) )
  allocate ( init_aspect_ratio(mesh%nelem), aspect_ratio(mesh%nelem) )

  ! compute initial element aspect ratio
  call compute_element_aspect_ratio ( mesh, init_volume, &
  init_aspect_ratio )

  call write_mesh_vtk ( mesh, filename='mesh.vtk' )

end subroutine generate_read_mesh

subroutine write_gmsh_parameters (  lx, ly, lz, xp, rp, dx_box, &
  dx_part )

  integer :: i, nobj
  real(dp) ::  lx, ly, lz, xp(:,:), rp(:), dx_box, dx_part
  
  type(refinement_fields_t) :: refinement_fields
  nrefine = 50 
  nobj = size(rp)

  allocate ( refinement_coor(nrefine,3) )

  call add_refinement_field ( refinement_fields, coor=xp, &
    distmin=1.1_dp , distmax=lx/2._dp, dx_fine = dx_part, &
    dx_coarse= dx_box )

  if ( delta <= 1._dp ) then 
     refinement_coor(:,1) = xp(1,1) 
     refinement_coor(:,2) = xp(1,2) - rp(1) - delta/2._dp
     refinement_coor(:,3) = 0
     ! Refinement in gap region: ensure at least nelem_min elements between particle and wall
     call add_refinement_field ( refinement_fields, &
     coor=refinement_coor, distmin=4._dp*delta, distmax=lx/3._dp, &
     dx_fine=delta/nelem_min, dx_coarse=dx_box )

  end if 

  deallocate ( refinement_coor)


  open ( unit=25, file='mesh.geo' )

  write ( 25, '(1X,A,F21.14,A)' ) 'lx = ', lx, ';'
  write ( 25, '(1X,A,F21.14,A)' ) 'ly = ', ly, ';'
  write ( 25, '(1X,A,F21.14,A)' ) 'lz = ', lz, ';'
  write ( 25, '(1X,A,F21.14,A)' ) 'dx_box = ', dx_box, ';'
  write ( 25, '(1X,A,I0,A)' ) 'nobj = ', nobj, ';'

  write ( 25, '(1X,A,F21.14,A)' ) 'xp = ', xp(1,1), ';'
  write ( 25, '(1X,A,F21.14,A)' ) 'yp = ', xp(1,2), ';'
  write ( 25, '(1X,A,F21.14,A)' ) 'zp = ', xp(1,3), ';'
  write ( 25, '(1X,A,F21.14,A)' ) 'rp = ', rp(1), ';'

  write ( 25, '(1X,A,F21.14,A)' ) 'dx_part = ', dx_part, ';'

  if (periodic .and. magnetic) then 
      write ( 25, '(/1x,a)' ) 'Include "half_particles_in_a_box_3D_2_periodic.igo";'
  else if (magnetic .and. .not. periodic) then 
      write ( 25, '(/1x,a)' ) 'Include "half_particles_in_a_box_3D_2.igo";'
  else if ( periodic .and. .not. magnetic) then 
      write ( 25, '(/1x,a)' ) 'Include "half_particles_in_a_box_3D_1_periodic.igo";'
  else   
      write ( 25, '(/1x,a)' ) 'Include "half_particles_in_a_box_3D_1.igo";'
  end if

  call write_refinement_fields ( refinement_fields, 'mesh.geo' )

  write ( 25, '(/1x,a)' ) 'Include "refinement.igo";'  
  close ( 25 )

end subroutine write_gmsh_parameters

subroutine compute_element_aspect_ratio ( mesh, volume, asp_ratio )

  type(mesh_t), intent(in) :: mesh
  real(dp), dimension(:), intent(out) :: asp_ratio, volume

  integer :: elem, node(4), grp, totelem
  real(dp) :: la, lb, lc, ld, le, lf
  real(dp) :: vert(4,3)

  node = [1,3,5,10]

  totelem = 0

  do grp = 1, mesh%nelgrp
    do elem = 1,mesh%grpnumel(grp)

    totelem = totelem + 1

  ! get coordinates of tetrahedron vertices
    vert = mesh%coor(mesh%topology(grp)%a(node,elem),:)

    ! compute side lengths
    la = lv ( vert(1,:) - vert(2,:) )
    lb = lv ( vert(2,:) - vert(3,:) )
    lc = lv ( vert(3,:) - vert(1,:) )
    ld = lv ( vert(1,:) - vert(4,:) )
    le = lv ( vert(2,:) - vert(4,:) )
    lf = lv ( vert(3,:) - vert(4,:) )

    ! compute volume of tetrahedron
    volume(totelem) = abs ( dot_product ( vert(1,:) - vert(4,:), &
    cross_product ( vert(2,:) - vert(4,:), vert(3,:) - vert(4,:) ) ) ) / 6

    ! compute aspect ratio
    asp_ratio(totelem) = max(la,lb,lc,ld,le,lf)**3/volume(totelem)

    end do
  end do

end subroutine compute_element_aspect_ratio


! length function

function lv ( a )

  real(dp), intent(in), dimension(3) :: a
  real(dp) :: lv

  lv = sqrt ( a(1)**2 + a(2)**2 + a(3)**2 )

end function lv


subroutine define_magnetic_problem
  integer :: j

! problem definition
  call create_input_probdef ( mesh, input_probdefp, nvec = 1)
  
  do j = 1, mesh%nelgrp
    input_probdefp%elementdof(j)%a = 1
    input_probdefp%vec_elementdof(j)%a(:,1) = 3
  end do        

  call define_essential ( mesh, input_probdefp, surfaces = [2,4] )
  if (periodic) then 
    call define_essential ( mesh, input_probdefp, point=1 )
    call define_constraint ( mesh, input_probdefp, &
    surface1=1, surface2=3, discretization='collocation',excludesurfaces=[2,4])
  end if 
  call problem_definition ( input_probdefp, mesh, problemp )

! create system vectors (solution and right-hand side)

  call create_sysvector ( problemp, solp, rhsdp )
  if (periodic) then 
    call fill_sysvector ( mesh, problemp, solp, point=1, value=0._dp )
  end if 
  call fill_sysvector ( mesh, problemp, solp, &
  surface1 = 2, value = delp)
  call fill_sysvector ( mesh, problemp, solp, &
  surface1 = 4, value = delp/2.0_dp )

    ! create system matrix  
  call create_sysmatrix_structure_base ( sysmatrixp, mesh, problemp, &
  symmetric=.false. )
  call create_sysmatrix_structure_constraint ( sysmatrixp, mesh, problemp )
  call finalize_sysmatrix_structure ( sysmatrixp )
  call create_sysmatrix_data ( sysmatrixp )

end subroutine define_magnetic_problem

subroutine solve_magnetic_problem

! build (assemble) matrix and vector from elements 

  call build_system ( mesh, problemp, sysmatrixp, rhsdp, &
    elemsub=poisson_elem, mcoefficients=coefficientsp)
  if (periodic) then 
    call build_system_constraint ( mesh, problemp, sysmatrixp, rhsdp, &
    constraint1=1, elemsub=poisson_node_conn, addmatvec=.true.)
  end if 
  call check ( sysmatrixp )

  call add_effect_of_essential_to_rhs ( problemp, sysmatrixp, solp, rhsdp )

  solver_options_p%real_storage = rs_p
  solver_options_p%integer_storage = is_p
  call solve_system_ma41 ( sysmatrixp, rhsdp, solp, &
    solver_options=solver_options_p )

end subroutine solve_magnetic_problem
  
subroutine define_flow_problem

  integer :: j
  !   problem definition

  call create_input_probdef ( mesh, input_probdef, nvec=3,&
  nphysq=2 )

  input_probdef%vec_elementdof(1)%a =  &
  reshape ( (/ 3,3,3,3,3,3,3,3,3,3,    &  ! velocity
  1,0,1,0,1,0,0,0,0,1,    &  ! pressure
  1,1,1,1,1,1,1,1,1,1 /),   &  ! scalar, such as vorticity
  (/10,3/) )

  input_probdef%physq = (/1,2/)
  input_probdef%probnr = 2

  if (nobj == 2) then 
    call define_essential ( mesh, input_probdef, surfaces = [1,3,4,5], &
    physq=physqvel)
    call define_essential ( mesh, input_probdef, surfaces=[2], physq=physqvel, degfd=(/1,0,0/) )
  else 
    call define_essential ( mesh, input_probdef, surfaces = [1,2,3,4,5], &
    physq=physqvel)
  end if 
  call define_essential ( mesh, input_probdef, surfaces=[6,7], physq=1, &
  degfd=(/0,0,1/) )
  call define_essential ( mesh, input_probdef, point=1, physq=2 )

  !   constraint on particle surface only
  call define_constraint ( mesh, input_probdef, surface1 = 7, &
  physq=1, discretization='collocation', &
  nodedof=2, naddunknowns=3 )

  allow_different_numdegfd_in_nodes = .true.

  call problem_definition ( input_probdef, mesh, problem )

  !   create system vectors (solution and right-hand side)
  call create_sysvector( problem, sol, rhsd )

  !   create system matrix

  call create_sysmatrix_structure_base ( sysmatrix, mesh, problem, &
  symmetric = .false. )
  call create_sysmatrix_structure_constraint ( sysmatrix, mesh, problem )
  call finalize_sysmatrix_structure ( sysmatrix )

  call create_sysmatrix_data ( sysmatrix )

end subroutine define_flow_problem
    
subroutine solve_flow_problem

  sol%u = 0._dp 

  call build_system ( mesh, problem, sysmatrix, rhsd, elemsub=stokes_elem, &
  coefficients=coefficients, elgroup1 = 1 )

  if (magnetic) then 
! implementing Maxwell Stress tensor
    call build_system ( mesh, problem, sysmatrix, rhsd, elemsub=rhs_max, &
    oldvectors=oldvectors_max, buildmatrix = .false., &
    mcoefficients=coefficients_sp, addmatvec=.true., &
    physqrow=[physqvel],physqcol=[physqvel], elgroup1 = 1) 
  end if 

  if (magnetic) then 
    call build_system_constraint ( mesh, problem, sysmatrix, rhsd, &
    constraint1=1, elemsub=elementc_sym_1p, &
    addmatvec=.true. ) 
  else 
    call build_system_constraint ( mesh, problem, sysmatrix, rhsd, &
    constraint1=1, elemsub=elementc_sym_1p_force, &
    addmatvec=.true. ) 
  end if 
  call check ( sysmatrix )

  call add_effect_of_essential_to_rhs ( problem, sysmatrix, sol, rhsd )

  solver_options_up%real_storage = rs_up
  solver_options_up%integer_storage = is_up
  if ( .not. sysmatrix%renumber ) call renumber_metis ( sysmatrix )
  solver_options_up%pivot_order = 1
  solver_options_up%scaling = 1
  call solve_system_ma41 ( sysmatrix, rhsd, sol, &
    solver_options=solver_options_up )
  
end subroutine solve_flow_problem

subroutine delete_old_problems

  if (magnetic ) then 
    call delete(problemp)
    call delete(sysmatrixp)
    call delete(input_probdefp)
    call delete(solp, rhsdp)
  end if  
  call delete ( problem)
  call delete ( problem_lapl)
  call delete ( sysmatrix )
  call delete ( input_probdef)
  call delete ( mesh )
  call delete ( sol)
  call delete ( rhsd )

  ! deallocate volume and aspect ratio arrays
  deallocate ( init_aspect_ratio, aspect_ratio )
  deallocate ( init_volume, volumev )

end subroutine delete_old_problems


end program magnetic_particle1
