! quarter mesh
! magnetic = .true. or .false., othewise force = ()
! nobj = 2 or 1 
! Mesh parameters: (n,nc), gap_par, gap_wall, nelem_min

program magnetic_particle9a

  use tfem_m
  use math_defs_m
  use hsl_ma41_m
  use hsl_ma57_m
  use stokes_elements_m
  use poisson_functions_m
  use poisson_elements_m
  use viscoelastic_elements_m
  use io_utils_m
  use subs_magnetic_particle4_m
  use projection_elements_m
  use timer_m
  use pardiso_m
  
  implicit none

! constants

  integer, parameter :: &
    uintpl = 6,         & ! P2 velocities
    pintpl = 2,         & ! P1 pressures
    gintpl = 2,         & ! P1 gradients
    cintpl = 2,         & ! P1 conformation
    physqgrad = 1,      & ! physical quantity nr of the gradients
    physqvel = 2,       & ! physical quantity nr of the velocities
    physqpress = 3,     & ! physical quantity nr of the pressures
    gauss = 8,          & ! 8-point Gauss integration of tets
    gaussb = 3,         & ! 3-point integration of boundary elements
    gauss_proj = 8,     & ! integration rule for the projection problem 
    inttype_proj = 0,   & ! standard Gauss-Legendre
    ncompc = 6,         & ! number of conformation tensor components
    nmodes = 1,         & ! number of modes/ parameters
    startm = 51,        & ! start of material model data
    coorsys = 0,        & ! coordinate system
    ncompv_P2 = 6,      & ! number of components for P2 projection:
                          ! x,y,z-coordinates at n and n-1
    ncompv_P1 = 12,     & ! number of components for P1 projection: 
                          ! conformation components at n and n-1
    funcnr = 1,         & ! rhs for magnetic problem
    ndim = 3   
                 
! variables

  integer :: &
    vtkevery = 1,       & ! vtk file every vtkevery steps. -1: means none
    nobj = 2,           & ! number of dimension
    timeint1 = 1,       & ! first-order, semi-implicit Euler
    timeint2 = 7,       & ! second-order, semi-implicit Gear/Karniadakis with
                          ! conformation prediction
    model = 3,           & ! 2: UCM/Oldroyd-B, 3: Giesekus,
                            ! 5: PTT linear, 6: PTT exponential
    alam_model = 2,    &      ! 0:none, 1:elastic, 2: Saramito1, 3: Saramito2
    logc = 1            ! standard scheme or log transformation

    real(dp) :: &
    eta_s = 0.1_dp,             & ! solvent viscosity
    eta_p = 0.9_dp,             & ! polymer viscosity
    alphapar = 0.1_dp,          & ! alpha parameter in the Giesekus model
    alpha1 = 1._dp,             & ! mu_p
    alpha2 = 2._dp,             & !mu_f
    rc = 1._dp,                 & ! radius of the particle
    lambda = 16._dp,             & ! relaxation time
    mobility = 0.2_dp,          & ! alpha (Giesekus)
    epspar = 0.1_dp,   & ! epsilon parameter in the PTT model
    tau_y = 2.0_dp,    & ! yield stress
    K = 1.0_dp,        & ! power-law coefficient
    time = 0._dp,               & ! initial time
    n =  10._dp,                & ! mesh size
    nc = 20._dp,                & ! particle size 
    deltat = 0.0025_dp,         & ! time step
    force1 = 1._dp,             &
    beta = 1._dp,               & ! upwinding parameter in the SUPG method
    rs_p = 2._dp,               &
    is_p = 2._dp,               &
    rs_up = 2._dp,              & ! real_storage gradient-velocity-pressure 
    is_up = 2._dp,              & ! integer_storage gradient-velocity-pressure
    rs_gup = 2._dp,             & ! real_storage gradient-velocity-pressure 
    is_gup = 2._dp,             & ! integer_storage gradient-velocity-pressure 
    rs_c  = 2._dp,              & ! real_storage for the conformation
    is_c  = 2._dp,              & ! integer_storage for conformation 
    volume_threshold = 0.2_dp, & ! remeshing volume threshold
    aspect_ratio_threshold = 0.2_dp ! remeshing aspect ratio threshold

! definitions for ve and upg problems

  type(mesh_t) :: mesh_np1
  type(input_probdef_t) :: input_probdef, input_probdefp
  type(input_probdef_t) :: input_probdefc, input_probdefc_projc
  type(problem_t), target :: problem, problemp, problemc, problemc_projc
  type(sysmatrix_t) :: sysmatrix,sysmatrixp, sysmatrixc, sysmatrixc_projc
  type(sysvector_t), target :: sol_np1, sol_n, sol_nm1, solp
  type(sysvector_t) :: rhsd, rhsdp
  type(vector_t):: velocity
  type(vector_t),target :: grad1, grad2
  type(oldvectors_t) :: oldvectors_ve, oldvectors_max
  type(oldvectors_t) :: oldvectors, oldvectors_int
  type(coefficients_t) :: coefficients, coefficientsp(2),coefficients_sp(2)
  type(sysvector_t), dimension(ncompc,nmodes), target :: solc_np1, solc_n
  type(sysvector_t), dimension(ncompc,nmodes), target :: solc_nm1, rhsc
  type(sysvector_t), dimension(ncompc,nmodes), target :: solc_projc, rhsc_projc
  type(lu_ma41_t) :: luc
  type(lu_ma57_t) :: lu_exps_projc
  type(solver_options_ma41_t) :: solver_options_ve
  type(solver_options_ma41_t) :: solver_options_p
  type(solver_options_pardiso_t) :: solver_options_pardiso
  type(vector_t), target :: meshvel_np1
  type(problem_t) :: problem_lapl
  type(subscript_t) :: velx, vely, velz, vel_all

! temp definitions used to store old definitions before projection

  type(mesh_t), target :: mesh_np1_temp
  type(sysvector_t), dimension(ncompc,nmodes) :: solc_n_temp, solc_nm1_temp
  type(sysvector_t) :: sol_n_temp, sol_nm1_temp
  type(vector_t) :: coords_n_temp, coords_nm1_temp
  type(problem_t), target :: problem_temp, problemc_temp

! variables

  integer :: i, step=0, ipost=0
  integer :: numtimesteps,  ec, nln
  real(dp) :: nelem_min
  real(dp) :: delta, init_dist, Wi
  real(dp) :: H, new_dist, beta1, char_velocity
  real(dp) :: lx, ly, lz
  real(dp) :: dx_box, dx_part
  real(dp) :: up(1,3), unm1(1,3), un(1,3), pp(1,3),ppn(1,3), xpn(1,ndim)
  real(dp) :: xp1(1,3),xp2(1,3), gap_par, gap_wall

  real(dp), dimension(:), allocatable :: init_volume, volumev
  real(dp), dimension(:), allocatable :: init_aspect_ratio, aspect_ratio
  real(dp), allocatable, dimension(:,:) :: meshcoor_n, meshcoor_nm1, &
    meshcoor_temp
  real(dp) :: norm_volume, norm_aspect_ratio
  real(dp), allocatable :: coor(:,:)
  real(dp), dimension(:,:), allocatable :: refinement_coor,refinement_coor1,refinement_coor2,refinement_coor3 
  integer :: nrefine
  real(dp) :: alpha, G, time_stop, angle
  real(dp) :: int_stress_bound(3), moment

  character(len=30) :: filename
  logical :: remeshing=.false., initial_mesh 
  logical :: periodic = .true.
  logical :: magnetic = .true.
  
  ! namelist for input of variables; read from standard input

namelist /comppar/ eta_s, eta_p,alpha1, alpha2,lambda, mobility, &
        model, alam_model, tau_y, &
        delta, angle, init_dist,dx_box,nelem_min, dx_part, H, deltat, &
       force1, time_stop, nobj, volume_threshold, lx, ly, lz,rc,  &
       aspect_ratio_threshold, periodic, magnetic, vtkevery, numtimesteps

read ( unit=*, nml=comppar )

  mmm1 = alpha1
  mmm2 = alpha2

  MAXNUMWARNINGS = 0
  WARN_ON_NO_DATA_IN_NODES = .false.
  
! set some parameters

  alpha = eta_p         ! DEVSS parameter
  G = eta_p / lambda    ! modulus

  open ( unit=11, file='output.out' )

! fill coefficients

  call create_coefficients ( coefficients, ncoefi=100, ncoefr=50+6*nmodes )

  coefficients%i = &
    [ uintpl,   pintpl,     0,     0,         gintpl, &
      physqvel, physqpress, 0,     physqgrad, gauss,  &
      gaussb,   cintpl,     0,     0,         0,      &
      0,        0,          model, nmodes,    startm, &
      logc,     timeint1,   coorsys, ( 0, i = 24, 100 )  &
    ]

  coefficients%i(48) = 1  ! use mesh velocity for ALE formulation
  coefficients%i(49) = 1  ! exp(s) projection =.true. for logc=1
  coefficients%i(80) = alam_model ! adapted lambda

  coefficients%r = 0
  coefficients%r(1:10) = &
    [ eta_s, 0._dp,   0._dp,   alpha, 0._dp, &
      0._dp, 0._dp,  deltat,   beta,  0._dp ]

  ec = 50

  coefficients%r(ec+1:ec+2) = [ G, lambda ]

  ec = 52

! set nonlinear material parameters for the model

  select case ( model )
  case(2) ! Oldroyd-B
    nln = 0
  case(3) ! Giesekus
    nln = 1
    coefficients%r(ec+1) = alphapar
  case(5,6) ! PTT
    nln = 1
    coefficients%r(ec+1) = epspar
  case default
    write(*,*) 'Error: model unknown = ', model
  end select

  ec = ec + nln

! set material parameters for the adapted lambda model

  select case ( alam_model )
  case(0,1) ! none or elastic
  case(2) ! SRM1
    coefficients%r(ec+1:ec+1) = [ tau_y ]
  case(3) ! SRM2
    coefficients%r(ec+1:ec+3) = [ tau_y, K, n ]
  case default
    write(*,*) 'Error: alam_model unknown = ', alam_model
  end select

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

 ! Let OMP_NUM_THREADS govern threading (PARDISO/MKL etc.).
 ! Hardcoding here prevents scaling on bigger nodes.

call execute_command_line ('rm *.vtk' )

!if (.not. magnetic) then 
!force =  0.5_dp *(/force1, 0.0_dp, 0._dp /)
!end if 
  
if (.not. magnetic) then
  force = 0.5_dp * (/ force1 *cos(angle * pi / 180.0_dp), &
                      force1 *sin(angle * pi / 180.0_dp), &
                      0._dp /)
end if

print *, 'force', force
! mesh parameters
  allocate(xp(1, ndim),rp(nobj))

  new_dist = init_dist

! particle position
  
  if (nobj == 2) then 
    xp(1,1) = (lx/2._dp - init_dist/2._dp  )
  else 
    xp(1,1) = 0._dp
  end if   
  if (delta == 0) then 
    xp(1,2) = 0._dp 
  else   
    xp(1,2) =  -ly/2._dp + delta + rc
  end if 
  xp(1,3) = 0._dp 

  pp(1,1:2) = xp(1,1:2)
  pp(1,3) = 0._dp ! initial rotation angle

! mesh refinement

  rp(:) = rc                ! particle radius
  print *, 'dx_box:', dx_box
  print *, 'dx_part:', dx_part

  call tic_w
  call generate_read_mesh
  
  call write_mesh_vtk(mesh_np1, filename = 'mesh.vtk')
  print *, 'number of nodes', mesh_np1%nnodes
  print *, 'number of elements ', mesh_np1%nelem
  print *, 'nelem on surface 7',mesh_np1%surfaces(7)%nelem

  call toc_w('Generating mesh')

  ! define the magnetic, flow and viscoelastic problem

  if (magnetic) then 
    delp = H *2._dp * lx
    beta1 = 3._dp * (alpha2 - alpha1)/(alpha2 + 2 * alpha1)
    print *, 'beta1',beta1
    call define_magnetic_problem
    call toc_w('Defining magnetic problem')

    call create ( oldvectors_max, nsysvec=1, nprob=1)
    oldvectors_max%s(1)%p => solp
    oldvectors_max%p(1)%p => problemp

  end if 

  call define_problems_create_vectors
  call toc_w('Defining flow problem')

! zero polymer stress as initial condition

  if ( logc == 0 ) then ! standard
    solc_np1(1,1)%u = 1 ! initial cxx
    solc_np1(2,1)%u = 0 ! initial cxy
    solc_np1(3,1)%u = 0 ! initial cxz
    solc_np1(4,1)%u = 1 ! initial cyy
    solc_np1(5,1)%u = 0 ! initial cyz
    solc_np1(6,1)%u = 1 ! initial czz
  else if ( logc == 1 ) then ! log scheme
    solc_np1(1,1)%u = 0 ! initial sxx
    solc_np1(2,1)%u = 0 ! initial sxy
    solc_np1(3,1)%u = 0 ! initial sxz
    solc_np1(4,1)%u = 0 ! initial syy
    solc_np1(5,1)%u = 0 ! initial syz
    solc_np1(6,1)%u = 0 ! initial szz
  end if

! solve upG problem without the divtau term

  if (magnetic) then 
    ! solve upG problem without the divtau term
    call solve_magnetic_problem
    call toc_w('solving magnetic problem')
    call compute_maxwell_force
  end if

  call build_solve_upG
  call toc_w('solving flow problem')

! extract info on particle
  call get_sysvector_constraint ( mesh_np1, problem, sol_np1, constraint=1, &
  addunknowns=.true., u=up(1,1:3) )

  call create_vector ( problem, velocity, physq= physqvel)
  call extract_physvector ( mesh_np1, problem, sol_np1, velocity )

  if (nobj == 2) then 

    char_velocity = 4._dp*alpha1*rc**5*beta1**2*H**2/(9._dp*(eta_s+eta_p)*new_dist**4)
    Wi = lambda*char_velocity/rc
	
	print *, 'new_dist', new_dist
 
	
    print *, 'new distance', new_dist
	
    write(11,'(9es16.8)') time, xp(1,1:2), up(1,1:3),new_dist,char_velocity,2._dp*int_stress_bound(1)
  else 
    write(11,'(9es16.8)') time, xp(1,1:3), up(1,1:3)
  end if 
    print *, 'step', step
    write ( *, '(1X,A,6F12.6)' ) 'pp1 = ', xp(1,1:3)
    write ( *, '(1X,A,6F12.6)' ) 'up1 = ', up(1,1:3)
! write VTK files
  call postprocessing ( mesh_np1, sol_np1, solc_np1, ipost=ipost )
  ipost = ipost + 1

! copy old values 

    call copy ( sol_np1, sol_n )
    call copy ( solc_np1, solc_n )
    meshcoor_n = mesh_np1%coor(mesh_np1%elementsets(1)%nodes,:)

! start time integration
  time = 0

  do step = 1, numtimesteps
	
    time = time + deltat

    if ( step > 1 ) coefficients%i(22) = timeint2

    if ( step == 1 ) then

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

      call update_mesh_nodes_sym_1p( mesh_np1, problem_lapl, &
        xp(:,1:3)-xpn(:,1:3) )    

      call find_bounds_blocks ( mesh_np1 )

    ! compute current aspect ratio

      call compute_element_aspect_ratio ( mesh_np1, volumev, aspect_ratio )

    !  compute maximum normalized aspect ratio and volume

      norm_volume = maxval( abs(log(volumev/init_volume)) )
      norm_aspect_ratio = maxval( abs(log(aspect_ratio/init_aspect_ratio)) )

    ! remeshing criterion

      remeshing = .false.

      if ( ( norm_volume >= volume_threshold .or. &
            norm_aspect_ratio >= aspect_ratio_threshold ) .and. &
            step /= numtimesteps ) then
        remeshing = .true.
      
      end if
	  
      if ( remeshing ) then
        
        print *,'Remeshing and projection...'

        ! create vectors to store the mesh coordinates, these coordinates will
        ! be used in the projection problem to find the new mesh at t_n and t_nm1

        call create ( problem, coords_n_temp, vec=2 )
        call create ( problem, coords_nm1_temp, vec=2 )

        coords_n_temp%u = reshape ( transpose ( meshcoor_n ), [size(meshcoor_n)] ) 
        coords_nm1_temp%u = reshape ( transpose ( meshcoor_nm1 ), &
                                      [size(meshcoor_nm1)] )
        ! copy current definitions to temporary definitions used for the projection

        call copy ( mesh_np1, mesh_np1_temp )
        call fill_mesh_parts ( mesh_np1_temp )
        call copy ( solc_n, solc_n_temp )
        call copy ( sol_n, sol_n_temp )
        call copy ( solc_nm1, solc_nm1_temp )
        call copy ( sol_nm1, sol_nm1_temp )
        call problem_definition ( input_probdef, mesh_np1, problem_temp )

        call problem_definition ( input_probdefc, mesh_np1, problemc_temp )

        ! delete the old problems

        call delete_old_problems 

		call generate_read_mesh
		 
      ! define the upg and ve problem
        if (magnetic) then 
			call define_magnetic_problem
        end if 
		
        call define_problems_create_vectors

      ! project the old solutions onto the new mesh and fill in the values
        call project_old_solutions_P2

        call project_old_solutions_P1

      ! delete the temporary definitions

        call delete ( coords_n_temp, coords_nm1_temp )
        call delete ( mesh_np1_temp )
        call delete ( solc_n_temp )
        call delete ( sol_n_temp )
        call delete ( solc_nm1_temp )
        call delete ( sol_nm1_temp )
        call delete ( problemc_temp )
        call delete ( problem_temp )
        call toc_w('remeshing')

      end if 

      !find the mesh velocity using a backwards differencing scheme
      if ( step == 1  ) then
        meshvel_np1%u = (reshape ( transpose(mesh_np1%coor(mesh_np1%elementsets(1)%nodes,:) - meshcoor_n )/deltat, &
        [mesh_np1%ndim*mesh_np1%elementsets(1)%nnodes] ))
      else

        meshvel_np1%u = reshape ( transpose ( &
        ( 1.5_dp*mesh_np1%coor(mesh_np1%elementsets(1)%nodes,:) - 2*meshcoor_n + 0.5_dp*meshcoor_nm1 ) / &
        deltat ), [mesh_np1%ndim*mesh_np1%elementsets(1)%nnodes]  )
      end if
    
      !  build and solve the upG system
      if (magnetic) then 
        call solve_magnetic_problem
        call toc_w('solving magnetic problem')
		call compute_maxwell_force
      end if 

      call build_solve_upG
      call toc_w('solving flow problem')
      
      !  copy old value (already here, so sol_np1 is used in the VE problem!)
      call copy ( sol_n, sol_nm1 )
      call copy ( sol_np1, sol_n )

  !  solve the conformation problem

	
      call build_solve_ve
      call toc_w('solving viscoelastic problem')

      ! extract info on particle
      call get_sysvector_constraint ( mesh_np1, problem, sol_np1, constraint=1, addunknowns=.true., u=up(1,:) )
    
      print *, 'step', step
      write ( *, '(1X,A,6F12.6)' ) 'pp1 = ', xp(1,1:3)
      write ( *, '(1X,A,6F12.6)' ) 'up1 = ', up(1,1:3)

      if (nobj == 2) then 
        new_dist = lx - 2*xp(1,1)
        char_velocity = 4._dp*alpha1*rc**5*beta1**2*H**2/(9._dp*(eta_s+eta_p)*new_dist**4)
        Wi = lambda*char_velocity/rc
        print *, 'new distance', new_dist
        print *, 'char velocity', char_velocity
        print *, 'Wi', Wi
        write(11,'(9es16.8)') time, xp(1,1:2), up(1,1:3),new_dist,char_velocity,2._dp*int_stress_bound(1)
      else 
        write(11,'(9es16.8)') time, xp(1,1:3), up(1,1:3)
      end if 

      if (new_dist <= 2.1_dp) then 
        stop
      end if  
      if ( vtkevery > 0 ) then
          if ( mod(step,vtkevery) == 0 ) then
            call postprocessing ( mesh_np1, sol_np1, solc_np1, ipost=ipost )
            ipost = ipost + 1
          end if
        end if

      call toc_w('postprocessing')
     
      call copy ( solc_n, solc_nm1 )
      call copy ( solc_np1, solc_n )
      meshcoor_nm1 = meshcoor_n
      meshcoor_n = mesh_np1%coor(mesh_np1%elementsets(1)%nodes,:)

  end do

  close(unit=11)

  ! delete all data including all allocated memory
  
    call delete_old_problems

  contains

  subroutine generate_read_mesh

    call write_gmsh_parameters (  lx, ly, lz, xp, rp, dx_box, &
    dx_part )

    ! HXT: parallel 3D Delaunay; honors OMP_NUM_THREADS.
    call execute_command_line ( 'gmsh -3 -order 2 -algo hxt -o mesh.msh &
                  &mesh.geo > outputmesh.out' )

  ! read mesh generated by gmsh

    call read_mesh_gmsh ( mesh_np1, filename='mesh.msh',ndim =3, &
      physgeom=.true. ) 

    if(periodic) then 
      call add_to_mesh ( mesh_np1, matchingsurface=[1,3], replace=1, &
      displacement=[0._dp,-ly,0._dp] )
    end if 

    call fill_mesh_parts ( mesh_np1 )

    ! allocate aspect ratio arrays
    allocate ( init_volume(mesh_np1%nelem), volumev(mesh_np1%nelem) )
    allocate ( init_aspect_ratio(mesh_np1%nelem), aspect_ratio(mesh_np1%nelem) )

     ! adding elements from group = 1 to the elementset 
    call add_to_mesh( mesh_np1, elementset='elements', group = 1, &
    elements=(/(i,i=1,mesh_np1%nelem,1)/) )

    ! numbering the elementset to 1     
    call add_to_mesh ( mesh_np1, elementset='nodes', elementsetnr=1 )

    ! adding nodes to the nodeset from the elementsetnr = 1  
    call add_to_mesh ( mesh_np1 ,nodeset = 'nodes', nodes = mesh_np1%elementsets(1)%nodes)

    ! define some arrays for the ale mesh position at old times
    allocate ( meshcoor_n(mesh_np1%elementsets(1)%nnodes,mesh_np1%ndim), &
    meshcoor_nm1(mesh_np1%elementsets(1)%nnodes,mesh_np1%ndim) )


    ! compute initial element aspect ratio
    call compute_element_aspect_ratio ( mesh_np1, init_volume, &
    init_aspect_ratio )

    norm_volume = 0.0_dp
    norm_aspect_ratio = 0.0_dp

  end subroutine generate_read_mesh


  subroutine write_gmsh_parameters (  lx, ly, lz, xp, rp, dx_box, &
  dx_part )

  integer :: i, nobj
  real(dp) ::  lx, ly, lz, xp(:,:), rp(:), dx_box, dx_part
  
  type(refinement_fields_t) :: refinement_fields
  nrefine = 50 
  nobj = size(rp)

  call add_refinement_field ( refinement_fields, coor=xp, &
    distmin= 1.1_dp, distmax=lx/2._dp, dx_fine = dx_part, &
    dx_coarse= dx_box )

  allocate ( refinement_coor(nrefine,3) )

  if ( delta <= 1._dp ) then 
     refinement_coor(:,1) = xp(1,1) 
     refinement_coor(:,2) = xp(1,2) - rp(1) - delta/2._dp
     refinement_coor(:,3) = 0
     ! Refinement in gap region: ensure at least nelem_min elements between particle and wall
     call add_refinement_field ( refinement_fields, &
     coor=refinement_coor, distmin=0.1_dp, distmax=lx/3._dp, &
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
    
   subroutine define_magnetic_problem
      integer :: j
    
      ! problem definition
      call create_input_probdef ( mesh_np1, input_probdefp, nvec = 1)
    
      do j = 1, mesh_np1%nelgrp
        input_probdefp%elementdof(j)%a = 1
        input_probdefp%vec_elementdof(j)%a(:,1) = 3
      end do        
      input_probdef%probnr = 1

      call define_essential ( mesh_np1, input_probdefp, surfaces = [2,4] )
      if (periodic) then 
	    call define_essential ( mesh_np1, input_probdefp, point=1 )
        call define_constraint ( mesh_np1, input_probdefp, &
        surface1=1, surface2=3, discretization='collocation', &
        excludesurfaces=[2,4] )
      end if 
	  
      call problem_definition ( input_probdefp, mesh_np1, problemp )
    
      ! create system vectors (solution and right-hand side)
    
      call create_sysvector ( problemp, solp, rhsdp )
	  if (periodic) then 
        call fill_sysvector ( mesh_np1, problemp, solp, &
         point = 1, value = 0._dp)
	  end if 
      call fill_sysvector ( mesh_np1, problemp, solp, &
      surface1 = 2, value = delp)
      call fill_sysvector ( mesh_np1, problemp, solp, &
      surface1 = 4, value = delp/2.0_dp)
    
      ! create system matrix  
      call create_sysmatrix_structure_base ( sysmatrixp, mesh_np1, problemp, &
      symmetric=.false. )
      call create_sysmatrix_structure_constraint ( sysmatrixp, mesh_np1, problemp )
      call finalize_sysmatrix_structure ( sysmatrixp )
      call create_sysmatrix_data ( sysmatrixp )
    
    end subroutine define_magnetic_problem
      
    subroutine solve_magnetic_problem
    
      ! build (assemble) matrix and vector from elements 
    
      call build_system ( mesh_np1, problemp, sysmatrixp, rhsdp, &
      elemsub=poisson_elem, mcoefficients=coefficientsp )
      if (periodic) then 
        call build_system_constraint ( mesh_np1, problemp, sysmatrixp, rhsdp, &
        constraint1=1, elemsub=poisson_node_conn, addmatvec=.true. )
      end if 
      call check ( sysmatrixp )
    
      call add_effect_of_essential_to_rhs ( problemp, sysmatrixp, solp, rhsdp )
    
		solver_options_pardiso%matrixtype = 11
        call solve_system_pardiso ( sysmatrixp, rhsdp, solp,&
        solver_options=solver_options_pardiso )

    end subroutine solve_magnetic_problem
    

  subroutine define_problems_create_vectors

    !   problem definition of gradient/velocity/pressure
        call create_input_probdef ( mesh_np1, input_probdef, nvec=5, nphysq=3 )
    
        input_probdef%vec_elementdof(1)%a =   &
            reshape ( (/ 9,0,9,0,9,0,0,0,0,9,    &  ! G
                         3,3,3,3,3,3,3,3,3,3,    &  ! velocity
                         1,0,1,0,1,0,0,0,0,1,    &  ! pressure
                         1,1,1,1,1,1,1,1,1,1,    &  ! scalar, such as vorticity
                         [(ncompv_P2,i=1,10)] /),&  ! projection
                         (/10,5/) )
    
        input_probdef%physq = (/1,2,3/)
        input_probdef%probnr = 1
    
    !  no slip boundary conditions

        call define_essential ( mesh_np1, input_probdef, point=1, physq=physqpress )
        ! constraint on particle surface 
        
          call define_constraint ( mesh_np1, input_probdef, surface1 = 7, &
          physq= physqvel, discretization='collocation', &
          nodedof=2, naddunknowns=3 )
        
        !  no slip boundary conditions
        if (nobj == 2) then 
          call define_essential ( mesh_np1, input_probdef, surfaces = [1,3,4,5], &
          physq=physqvel )
          call define_essential ( mesh_np1, input_probdef, surfaces=[2], physq=physqvel, &
          degfd=(/1,0,0/) )
        else 
          call define_essential ( mesh_np1, input_probdef, surfaces = [1,2,3,4,5], &
          physq=physqvel)
        end if   
        call define_essential ( mesh_np1, input_probdef, surfaces=[6, 7], physq=physqvel, &
        degfd=(/0,0,1/) )


    allow_different_numdegfd_in_nodes = .true.

    call problem_definition ( input_probdef, mesh_np1, problem )
    
    !   define viscoelastic problem, essential conditions and constraints
        call create_input_probdef ( mesh_np1, input_probdefc, nvec=3, nphysq=1 )
    
        input_probdefc%vec_elementdof(1)%a =   &
            reshape ( (/ 1,0,1,0,1,0,0,0,0,1,    &  ! c
                         1,1,1,1,1,1,1,1,1,1,    &  ! scalar for plotting
                         ncompv_P1,0,ncompv_P1,0,ncompv_P1,    & ! projection
                                 0,0,        0,0,ncompv_p1 /), &
                         (/10,3/) )
    
        input_probdefc%physq = (/1/)
        input_probdefc%probnr = 2
    

        call problem_definition ( input_probdefc, mesh_np1, problemc )
    
    !   create system vectors for the upG problem
        call create_sysvector ( problem, sol_np1, sol_n, sol_nm1, rhsd )
      

    !   create mesh velocity vectors and vector to store mesh coordinates
        call create ( problem, meshvel_np1, vec=2 )
    
    !   create subscripts for the mesh velocity
        call create_subscript ( mesh_np1, problem, velx, physqarr=[2], degfd=1 )
        call create_subscript ( mesh_np1, problem, vely, physqarr=[2], degfd=2 )
        call create_subscript ( mesh_np1, problem, velz, physqarr=[2], degfd=3 )
    
    !   create system vectors (solution and right-hand side) for conformation and
        call create ( problemc, solc_np1, solc_n, solc_nm1, rhsc )
   
    !   create system matrix for the upg problem
        call create_sysmatrix_structure_base ( sysmatrix, mesh_np1, problem, &
        symmetric = .false. )
        call create_sysmatrix_structure_constraint ( sysmatrix, mesh_np1, problem )
        call finalize_sysmatrix_structure ( sysmatrix )
        call create_sysmatrix_data ( sysmatrix )
    
    !   create system matrix for the ve problem
        call create_sysmatrix_structure_base ( sysmatrixc, mesh_np1,problemc,&  
        symmetric = .false. )
        call create_sysmatrix_structure_constraint ( sysmatrixc, mesh_np1, &
          problemc )
        call finalize_sysmatrix_structure ( sysmatrixc )
        call create_sysmatrix_data ( sysmatrixc )
    
    !   initialize all vectors to zero
        sol_nm1%u = 0.0_dp
        solc_nm1(1,1)%u = 0.0_dp
        solc_nm1(2,1)%u = 0.0_dp
        solc_nm1(3,1)%u = 0.0_dp
        sol_n%u = 0.0_dp
        sol_np1%u = 0.0_dp
        solc_n(1,1)%u = 0.0_dp
        solc_n(2,1)%u = 0.0_dp
        solc_n(3,1)%u = 0.0_dp
        meshvel_np1%u = 0.0_dp 
    
    !   problem definition for projected "c=exp(s)" of the log conformation s
        call create_input_probdef ( mesh_np1, input_probdefc_projc, nvec=1, &
          nphysq=1 )
    
          input_probdefc_projc%vec_elementdof(1)%a = &
              reshape ( [ 1,0,1,0,1,0,0,0,0,1 ], &
                           [10,1] )
    
        input_probdefc_projc%physq = [1]
        input_probdefc_projc%probnr = 3
    
        call problem_definition ( input_probdefc_projc, mesh_np1, problemc_projc )
    
        call create ( problemc_projc, solc_projc, rhsc_projc )
    
        solc_projc(1,1)%u = 1 ! initial cxx
        solc_projc(2,1)%u = 0 ! initial cxy
        solc_projc(3,1)%u = 0 ! initial cxz
        solc_projc(4,1)%u = 1 ! initial cyy
        solc_projc(5,1)%u = 0 ! initial cyz
        solc_projc(6,1)%u = 1 ! initial czz
    
    !   create system matrix for projection problem
        call create_sysmatrix_structure ( sysmatrixc_projc, mesh_np1, &
          problemc_projc, symmetric=.true. )
        call create_sysmatrix_data ( sysmatrixc_projc )
    
    !   create oldvectors
        call create_oldvectors ( oldvectors_ve, nsysvec=2, nsysvec2=3, nprob=3, &
          nvec=2 )
        oldvectors_ve%s(1)%p => sol_n
        oldvectors_ve%s(2)%p => sol_nm1
        oldvectors_ve%s2(1)%p => solc_n
        oldvectors_ve%s2(2)%p => solc_nm1
        oldvectors_ve%s2(3)%p => solc_projc
        oldvectors_ve%p(1)%p => problem
        oldvectors_ve%p(2)%p => problemc
        oldvectors_ve%p(3)%p => problemc_projc
        oldvectors_ve%v(1)%p => meshvel_np1
    
      end subroutine define_problems_create_vectors
    
      subroutine build_solve_upG

  !   stokes velocity/pressure
      call build_system ( mesh_np1, problem, sysmatrix, rhsd, &
        elemsub=stokes_elem, physqrow=(/2,3/), physqcol=(/2,3/), &
        coefficients=coefficients, elgroup1 = 1  )
        
    if ( magnetic ) then
  
	  call build_system ( mesh_np1, problem, sysmatrix, rhsd, elemsub=rhs_max, &
	  oldvectors=oldvectors_max, buildmatrix = .false., &
	  mcoefficients=coefficients_sp, addmatvec=.true., &
	  physqrow=[physqvel],physqcol=[physqvel] , elgroup1 = 1)

    end if   
  !   DEVSS-G
      call build_system ( mesh_np1, problem, sysmatrix, rhsd, &
        elemsub=devssg_elem, addmatvec=.true., &
        physqrow=(/1,2/), physqcol=(/1,2/), coefficients=coefficients, elgroup1 = 1  )
  
    !   set to zero off-diagonal blocks gradient-pressure
        call build_system ( mesh_np1, problem, sysmatrix, rhsd, addmatvec=.true., &
          buildvector=.false., physqrow=(/1/), physqcol=(/3/), zeromatvec=.true., elgroup1 = 1  )
        call build_system ( mesh_np1, problem, sysmatrix, rhsd, addmatvec=.true., &
          buildvector=.false., physqrow=(/3/), physqcol=(/1/), zeromatvec=.true., elgroup1 = 1  )
		
		print *, 'time', time
		print *, 'time_stop', time_stop

		print *, 'force', force 
    !   constraints on surfaces
        if (magnetic) then
        call build_system_constraint ( mesh_np1, problem, sysmatrix, rhsd, &
          constraint1=1, elemsub= elementc_sym_1p, addmatvec=.true. )
        else 
			! Only apply stopping logic if time_stop is not 0
			if (time_stop > 0.0_dp) then
				if (time >= time_stop) then
				print *, 'time', time
				print *, 'time_stop', time_stop

				print *, 'force', force
					force = (/0.0_dp, 0.0_dp, 0._dp /)
					force1 = 0.0_dp
				end if
			end if
          call build_system_constraint ( mesh_np1, problem, sysmatrix, rhsd, &
          constraint1=1, elemsub= elementc_sym_1p_force, addmatvec=.true. )
        end if   
    !   build (assemble) vector for gradient/velocity/pressure problem
        if ( step .ge. 1 ) then

          !     exps projection (for log conformation)
          if ( logc == 1 ) then
            call solve_exps_projection
          end if
          call build_system ( mesh_np1, problem, sysmatrix, rhsd, &
            elemsub=divtau_implicit_ce_elem_c, &
            oldvectors=oldvectors_ve, physqrow=(/2/), physqcol=(/2/), &
            addmatvec=.true., coefficients=coefficients, elgroup1 = 1  )

        end if
    
        call check ( sysmatrix )
        
        call add_effect_of_essential_to_rhs ( problem, sysmatrix, sol_np1, rhsd ) 
		
		solver_options_pardiso%matrixtype = 11
        call solve_system_pardiso ( sysmatrix, rhsd, sol_np1,&
        solver_options=solver_options_pardiso )

        end subroutine build_solve_upG
        
        
    subroutine build_solve_ve

            integer :: icomp
        !   build (assemble) matrix and vector for conformation problem   
            if ( step == 1 ) then
              call build_system ( mesh_np1, problemc, sysmatrixc, m2sysvector=rhsc, &
                elemsub=ce_supg_elem, oldvectors=oldvectors_ve, &
                coefficients=coefficients, elgroup1 = 1 )
            else
              call build_system ( mesh_np1, problemc, sysmatrixc, m2sysvector=rhsc, &
                elemsub=ce_supg_elem_implicit_2nd_order, oldvectors=oldvectors_ve, &
                coefficients=coefficients, elgroup1 = 1 )
            end if

            call check ( sysmatrixc )
        
        !   solve conformation and keep LU decomposition in the loop over components
            solver_options_ve%real_storage=rs_c
            solver_options_ve%integer_storage=is_c
        
            do icomp = 1, ncompc
              call solve_system_ma41 ( sysmatrixc, rhsc(icomp,1), solc_np1(icomp,1), &
                luc, solver_options=solver_options_ve )
            end do
        
            call delete ( luc )  ! remove LU decomposition and rebuild next time step
        
    end subroutine build_solve_ve
        
  subroutine postprocessing ( mesh, sol, solc, ipost, check_proj  )

    type(mesh_t), intent(inout) :: mesh
    type(sysvector_t), intent(in), target :: sol
    type(sysvector_t), dimension(:,:), intent(in), target :: solc
    integer, intent(in), optional :: ipost
    character(len=*), intent(in), optional :: check_proj

    type(oldvectors_t) :: oldvectors_dve
    type(vector_t) :: cxx, cxy, cxz, cyy, cyz, czz, pressure, gammadot

    if ( .not. mesh%meshparts) call fill_mesh_parts ( mesh )

    call create_oldvectors ( oldvectors_dve, nsysvec=1, nsysvec2=1 )
    oldvectors_dve%s(1)%p => sol

    call create_vector ( problem, pressure, vec=4 )
    call create_vector ( problem, gammadot, vec=4 )

!  derive the pressure in all nodes
    call derive_vector ( mesh, problem, pressure, &
      elemsub=stokes_pressure, coefficients=coefficients, &
      oldvectors=oldvectors_dve, elgroup1 = 1 )

!  derive the shear rate (gammadot) in all nodes
    coefficients%i(13) = 8
    call derive_vector ( mesh, problem, gammadot, elemsub=stokes_deriv, &
      coefficients=coefficients, oldvectors=oldvectors_dve, elgroup1 = 1 )

    if ( present(ipost) ) write(filename,'(a,i4.4,a)') 'flow', ipost, '.vtk'
    if ( present(check_proj) ) then
      write(filename,'(a,i4.4,a)') 'flow_'//check_proj//'.vtk'
    end if
    call write_scalar_vtk ( mesh, problem, vector=pressure, &
      dataname='pressure',  filename=filename, groups = [1]  )
    call write_vector_vtk ( mesh, problem, filename=filename, &
      dataname='velocity', sysvector=sol, physq=physqvel, &
      append=.true., groups = [1]  )
    call write_scalar_vtk ( mesh, problem, vector=gammadot, &
      filename=filename, dataname='gammadot', append=.true., groups = [1] )  

    oldvectors_dve%s2(1)%p => solc
    call create_vector ( problemc, cxx, vec=2 )
    call create_vector ( problemc, cxy, vec=2 )
    call create_vector ( problemc, cxz, vec=2 )
    call create_vector ( problemc, cyy, vec=2 )
    call create_vector ( problemc, cyz, vec=2 )
    call create_vector ( problemc, czz, vec=2 )

    coefficients%i(13)=1
    call derive_vector ( mesh, problemc, cxx, elemsub=deriv_conformation, &
      coefficients=coefficients, oldvectors=oldvectors_dve, elgroup1 = 1 )
    coefficients%i(13)=2
    call derive_vector ( mesh, problemc, cxy, elemsub=deriv_conformation, &
      coefficients=coefficients, oldvectors=oldvectors_dve, elgroup1 = 1 )
    coefficients%i(13)=3
    call derive_vector ( mesh, problemc, cxz, elemsub=deriv_conformation, &
      coefficients=coefficients, oldvectors=oldvectors_dve, elgroup1 = 1 )
    coefficients%i(13)=4
    call derive_vector ( mesh, problemc, cyy, elemsub=deriv_conformation, &
      coefficients=coefficients, oldvectors=oldvectors_dve, elgroup1 = 1 )
    coefficients%i(13)=5
    call derive_vector ( mesh, problemc, cyz, elemsub=deriv_conformation, &
      coefficients=coefficients, oldvectors=oldvectors_dve, elgroup1 = 1 )
    coefficients%i(13)=6
    call derive_vector ( mesh, problemc, czz, elemsub=deriv_conformation, &
      coefficients=coefficients, oldvectors=oldvectors_dve, elgroup1 = 1 )

    if ( present(ipost) ) write(filename,'(a,i4.4,a)') 'c', ipost, '.vtk'
    if ( present(check_proj) ) then
      write(filename,'(a,i4.4,a)') 'c_'//check_proj//'.vtk'
    end if
    call write_scalar_vtk ( mesh, problemc, vector=cxx, &
      filename=filename, dataname='cxx', groups = [1]  )

    call write_scalar_vtk ( mesh, problemc, vector=cxy, &
      filename=filename, dataname='cxy', append=.true., groups = [1]  )

    call write_scalar_vtk ( mesh, problemc, vector=cxz, &
      filename=filename, dataname='cxz', append=.true., groups = [1]  )

    call write_scalar_vtk ( mesh, problemc, vector=cyy, &
      filename=filename, dataname='cyy', append=.true.,groups = [1]  )

    call write_scalar_vtk ( mesh, problemc, vector=cyz, &
      filename=filename, dataname='cyz', append=.true.,groups = [1]  )

    call write_scalar_vtk ( mesh, problemc, vector=czz, &
      filename=filename, dataname='czz', append=.true.,groups = [1]  )

    call delete(cxx, cxy, cxz, cyy)
    call delete(cyz, czz, pressure, gammadot)
    call delete(oldvectors_dve)

  end subroutine postprocessing

! delete old problem

  subroutine delete_old_problems

    if (magnetic ) then 
      call delete(problemp)
      call delete(sysmatrixp)
      call delete(input_probdefp)
      call delete(solp, rhsdp)

    end if 
    call delete ( problem, problemc )
    call delete ( problem_lapl )
    call delete ( sysmatrix, sysmatrixc )
    call delete ( input_probdef, input_probdefc )
    call delete ( mesh_np1 )
    call delete ( sol_np1, sol_n, sol_nm1 )
    call delete ( solc_np1, solc_n, solc_nm1 )
    call delete ( rhsc)
    call delete ( rhsd )
    call delete ( oldvectors_ve )
    call delete ( meshvel_np1 )
    call delete ( velx, vely, velz, vel_all )
    call delete ( problemc_projc )
    call delete ( input_probdefc_projc )
    call delete ( solc_projc, rhsc_projc )
    call delete ( sysmatrixc_projc )

!   deallocate volume and aspect ratio arrays
    deallocate ( init_aspect_ratio, aspect_ratio )
    deallocate ( init_volume, volumev )
    deallocate ( meshcoor_n, meshcoor_nm1 )
  end subroutine delete_old_problems


  
  ! project c=exp(s) on discrete fem space
  subroutine solve_exps_projection
  
    type(solver_options_ma57_t) :: solver_options_ma57

    integer :: i, m

!   build system matrix and vector for projection problem
    call build_system ( mesh_np1, problemc_projc, sysmatrixc_projc, &
      m2sysvector=rhsc_projc, elemsub=exps_projection_elem, &
      oldvectors=oldvectors_ve, coefficients=coefficients, elgroup1 =1  )

    call check ( sysmatrixc_projc )

!   MA57 solver storage
    solver_options_ma57%integer_storage = 1.3
    solver_options_ma57%real_storage    = 1.3

!   LU decomposition is done in the first loop traversing
    do m = 1, nmodes
      do i = 1, ncompc

        call add_effect_of_essential_to_rhs ( problemc_projc, &
          sysmatrixc_projc, solc_projc(i,m), rhsc_projc(i,m) )

        call solve_system_ma57 ( sysmatrixc_projc, rhsc_projc(i,m), &
           solc_projc(i,m), lu_exps_projc, solver_options=solver_options_ma57 )

      end do
    end do

    call delete ( lu_exps_projc )

  end subroutine solve_exps_projection
    
  ! P2 project the velocities and coordinates on the new mesh

  subroutine project_old_solutions_P2

    type(input_probdef_t) :: input_probdef_proj
    type(problem_t) :: problem_proj
    type(sysmatrix_t) :: sysmatrix_proj
    type(sysvector_t) :: rhsd_proj(ncompv_P2)
    type(sysvector_t) :: sol_proj(ncompv_P2)
    type(oldvectors_t) :: oldvectors_proj
    type(coefficients_t) :: coefficients_proj
    type(vector_t), target :: vec_p
    type(lu_ma57_t) :: lu2
    type(subscript_t) :: val_proj

    integer :: i

  ! create vector for projection
    call create ( problem_temp, vec_p, vec=5 )

    ! transfer the coordinates at n
    call transfer_data ( mesh_np1_temp, problem_temp, &
      vector1=coords_n_temp, vector2=vec_p, degfd2=[1,2,3] )

    ! transfer the coordinates at n-1
    call transfer_data ( mesh_np1_temp, problem_temp, &
      vector1=coords_nm1_temp, vector2=vec_p, degfd2=[4,5,6] )

!   problem definition for projection
    call create_input_probdef ( mesh_np1, input_probdef_proj )

    input_probdef_proj%elementdof(1)%a = 1
    if (mesh_np1%nelgrp == 2) then 
      input_probdef_proj%elementdof(2)%a = 0
    end if 

    call problem_definition ( input_probdef_proj, mesh_np1, problem_proj )

!   create system vectors (solution and right-hand side)  
    call create ( problem_proj, sol_proj )
    call create ( problem_proj, rhsd_proj )

!   create system matrix
    call create_sysmatrix_structure ( sysmatrix_proj, mesh_np1, &
      problem_proj, symmetric=.true. )
    call create_sysmatrix_data ( sysmatrix_proj )

!   fill coefficients
    call create_coefficients ( coefficients_proj, ncoefi=100, ncoefr=50 )
    coefficients_proj%i = 0
    coefficients_proj%i(1:4) = [ uintpl, ncompv_P2, uintpl, uintpl ]
    coefficients_proj%i(10) = gauss_proj
    coefficients_proj%i(23) = coorsys
    coefficients_proj%i(40) = inttype_proj
    coefficients_proj%r = 0

!   oldvectors
    call create_oldvectors ( oldvectors_proj, nvec=1, nprob=1, nmesh=1 )
    oldvectors_proj%v(1)%p => vec_p
    oldvectors_proj%p(1)%p => problem_temp
    oldvectors_proj%m(1)%p => mesh_np1_temp

!   build (assemble) matrix and vector from elements
    call build_system ( mesh_np1, problem_proj, sysmatrix_proj, &
    msysvector=rhsd_proj, elemsub=projection_elem, &
    coefficients=coefficients_proj, oldvectors=oldvectors_proj, elgroup1 = 1 )

!   solve the projection problem
    do i = 1, ncompv_P2
      call add_effect_of_essential_to_rhs ( problem_proj, sysmatrix_proj, &
       sol_proj(i), rhsd_proj(i) )
      call solve_system_ma57 ( sysmatrix_proj, rhsd_proj(i), sol_proj(i), &
        lu=lu2 )
    end do

    call create_subscript ( mesh_np1, problem_proj, val_proj )

!   create meshes at t_n and t_nm1 and update the old coordinates
    meshcoor_n(:,1) =   sol_proj(1)%u(val_proj%s) 
    meshcoor_n(:,2) =   sol_proj(2)%u(val_proj%s)
    meshcoor_n(:,3) =   sol_proj(3)%u(val_proj%s)
    meshcoor_nm1(:,1) = sol_proj(4)%u(val_proj%s)
    meshcoor_nm1(:,2) = sol_proj(5)%u(val_proj%s)
    meshcoor_nm1(:,3) = sol_proj(6)%u(val_proj%s)

    call write_scalar_vtk ( mesh_np1, problem_proj, 'test_proj.vtk', 'x_n', sysvector=sol_proj(1))
    call write_scalar_vtk ( mesh_np1, problem_proj, 'test_proj.vtk', 'y_n', sysvector=sol_proj(2), append=.true. )
    call write_scalar_vtk ( mesh_np1, problem_proj, 'test_proj.vtk', 'z_n', sysvector=sol_proj(3), append=.true. )
    call write_scalar_vtk ( mesh_np1, problem_proj, 'test_proj.vtk', 'x_nm1', sysvector=sol_proj(1), append=.true. )
    call write_scalar_vtk ( mesh_np1, problem_proj, 'test_proj.vtk', 'y_nm1', sysvector=sol_proj(2), append=.true. )
    call write_scalar_vtk ( mesh_np1, problem_proj, 'test_proj.vtk', 'z_nm1', sysvector=sol_proj(3), append=.true. )

  ! delete definitions used for the projection problem
    call delete ( lu2 ) 
    call delete( input_probdef_proj )
    call delete( problem_proj )
    call delete( sysmatrix_proj )
    call delete( rhsd_proj )
    call delete( sol_proj )
    call delete( oldvectors_proj )
    call delete( coefficients_proj )
    call delete( vec_p )

  end subroutine project_old_solutions_P2

! P1 project the conformation components and velocity gradients on the new mesh

  subroutine project_old_solutions_P1

    type(input_probdef_t) :: input_probdef_proj
    type(problem_t) :: problem_proj
    type(sysmatrix_t) :: sysmatrix_proj
    type(sysvector_t) :: rhsd_proj(ncompv_P1)
    type(sysvector_t) :: sol_proj(ncompv_P1)
    type(oldvectors_t) :: oldvectors_proj
    type(coefficients_t) :: coefficients_proj
    type(vector_t), target :: vec_p
    type(lu_ma57_t) :: lu2

    type(subscript_t) :: cc

    integer :: i

!   create vector for projection
    call create ( problemc_temp, vec_p, vec=3 )

!   transfer the conformation tensor at n
    call transfer_data ( mesh_np1_temp, problemc_temp, &
      sysvector1=solc_n_temp(1,1), vector2=vec_p, physq1=[1], degfd2=[1] )
    call transfer_data ( mesh_np1_temp, problemc_temp, &
      sysvector1=solc_n_temp(2,1), vector2=vec_p, physq1=[1], degfd2=[2] )
    call transfer_data ( mesh_np1_temp, problemc_temp, &
      sysvector1=solc_n_temp(3,1), vector2=vec_p, physq1=[1], degfd2=[3] )
    call transfer_data ( mesh_np1_temp, problemc_temp, &
      sysvector1=solc_n_temp(4,1), vector2=vec_p, physq1=[1], degfd2=[4] )
    call transfer_data ( mesh_np1_temp, problemc_temp, &
      sysvector1=solc_n_temp(5,1), vector2=vec_p, physq1=[1], degfd2=[5] )
    call transfer_data ( mesh_np1_temp, problemc_temp, &
      sysvector1=solc_n_temp(6,1), vector2=vec_p, physq1=[1], degfd2=[6] )

!   transfer the conformation tensor at n-1
    call transfer_data ( mesh_np1_temp, problemc_temp, &
      sysvector1=solc_nm1_temp(1,1), vector2=vec_p, physq1=[1], degfd2=[7] )
    call transfer_data ( mesh_np1_temp, problemc_temp, &
      sysvector1=solc_nm1_temp(2,1), vector2=vec_p, physq1=[1], degfd2=[8] )
    call transfer_data ( mesh_np1_temp, problemc_temp, &
      sysvector1=solc_nm1_temp(3,1), vector2=vec_p, physq1=[1], degfd2=[9] )
    call transfer_data ( mesh_np1_temp, problemc_temp, &
      sysvector1=solc_nm1_temp(4,1), vector2=vec_p, physq1=[1], degfd2=[10] )
    call transfer_data ( mesh_np1_temp, problemc_temp, &
      sysvector1=solc_nm1_temp(5,1), vector2=vec_p, physq1=[1], degfd2=[11] )
    call transfer_data ( mesh_np1_temp, problemc_temp, &
      sysvector1=solc_nm1_temp(6,1), vector2=vec_p, physq1=[1], degfd2=[12] )

!   problem definition for projection
    call create_input_probdef ( mesh_np1, input_probdef_proj )

    input_probdef_proj%elementdof(1)%a = [1,0,1,0,1,0,0,0,0,1]
    if (mesh_np1%nelgrp == 2) then 
      input_probdef_proj%elementdof(2)%a = [0,0,0,0,0,0,0,0,0,0]
    end if 
    call problem_definition ( input_probdef_proj, mesh_np1, problem_proj )

!   create system vectors (solution and right-hand side)  
    call create ( problem_proj, sol_proj )
    call create ( problem_proj, rhsd_proj )

!   create system matrix
    call create_sysmatrix_structure ( sysmatrix_proj, mesh_np1, &
      problem_proj, symmetric=.true. )
    call create_sysmatrix_data ( sysmatrix_proj )

!   fill coefficients
    call create_coefficients ( coefficients_proj, ncoefi=100, ncoefr=50 )
    coefficients_proj%i = 0
    coefficients_proj%i(1:4) = [ gintpl, ncompv_P1, uintpl, gintpl ]
    coefficients_proj%i(10) = gauss_proj
    coefficients_proj%i(23) = coorsys
    coefficients_proj%i(40) = inttype_proj
    coefficients_proj%r = 0

!   oldvectors
    call create_oldvectors ( oldvectors_proj, nvec=1, nprob=1, nmesh=1 )
    oldvectors_proj%v(1)%p => vec_p
    oldvectors_proj%p(1)%p => problemc_temp
    oldvectors_proj%m(1)%p => mesh_np1_temp

!   build (assemble) matrix and vector from elements
    call build_system ( mesh_np1, problem_proj, sysmatrix_proj, &
      msysvector=rhsd_proj, elemsub=projection_elem, &
      coefficients=coefficients_proj, oldvectors=oldvectors_proj, elgroup1 = 1 )

!   solve the projection problem
    do i = 1, ncompv_P1
      call add_effect_of_essential_to_rhs ( problem_proj, sysmatrix_proj, &
       sol_proj(i), rhsd_proj(i) )
      call solve_system_ma57 ( sysmatrix_proj, rhsd_proj(i), sol_proj(i), &
        lu=lu2 )
    end do

!   update the conformation tensor
!   note: because a subscript to the conformation tensor is not needed in the
!   main part of the program, a subscript is created here 'on the fly' and
!   deleted after it is used.
    call create_subscript ( mesh_np1, problemc, cc, physqarr=[1] )
    solc_n(1,1)%u(cc%s) = sol_proj(1)%u
    solc_n(2,1)%u(cc%s) = sol_proj(2)%u
    solc_n(3,1)%u(cc%s) = sol_proj(3)%u
    solc_n(4,1)%u(cc%s) = sol_proj(4)%u
    solc_n(5,1)%u(cc%s) = sol_proj(5)%u
    solc_n(6,1)%u(cc%s) = sol_proj(6)%u
    solc_nm1(1,1)%u(cc%s) = sol_proj(7)%u
    solc_nm1(2,1)%u(cc%s) = sol_proj(8)%u
    solc_nm1(3,1)%u(cc%s) = sol_proj(9)%u
    solc_nm1(4,1)%u(cc%s) = sol_proj(10)%u
    solc_nm1(5,1)%u(cc%s) = sol_proj(11)%u
    solc_nm1(6,1)%u(cc%s) = sol_proj(12)%u
    call delete ( cc )

!   delete definitions used for the projection problem
    call delete ( lu2 ) 
    call delete( input_probdef_proj )
    call delete( problem_proj )
    call delete( sysmatrix_proj )
    call delete( rhsd_proj )
    call delete( sol_proj )
    call delete( oldvectors_proj )
    call delete( coefficients_proj )
    call delete( vec_p )

  end subroutine project_old_solutions_P1

  ! compute the aspect ratio and volume for each mesh element

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

!       get coordinates of tetrahedron vertices
        vert = mesh%coor(mesh%topology(grp)%a(node,elem),:)

!       compute side lengths
        la = lv ( vert(1,:) - vert(2,:) )
        lb = lv ( vert(2,:) - vert(3,:) )
        lc = lv ( vert(3,:) - vert(1,:) )
        ld = lv ( vert(1,:) - vert(4,:) )
        le = lv ( vert(2,:) - vert(4,:) )
        lf = lv ( vert(3,:) - vert(4,:) )

!       compute volume of tetrahedron
        volume(totelem) = abs ( dot_product ( vert(1,:) - vert(4,:), &
          cross_product ( vert(2,:) - vert(4,:), vert(3,:) - vert(4,:) ) ) ) / 6

!       compute aspect ratio
        asp_ratio(totelem) = max(la,lb,lc,ld,le,lf)**3/volume(totelem)

      end do
    end do
     
  end subroutine compute_element_aspect_ratio

! length function

 ! length function
  
  function lv ( a )
  
    real(dp), intent(in), dimension(3) :: a
    real(dp) :: lv

    lv = sqrt ( a(1)**2 + a(2)**2 + a(3)**2 )

  end function lv
  
  subroutine compute_maxwell_force 
  ! Gradient of solution 
      call create ( problemp, grad1, vec=1 ) 
      call create ( problemp, grad2, vec=1 ) 
      call create ( oldvectors, nsysvec=1)
      oldvectors%s(1)%p => solp
    
      grad1%u=0
    
      call derive_vector ( mesh_np1, problemp, grad1, elemsub=poisson_deriv, &
        coefficients=coefficientsp(1), oldvectors=oldvectors, elgroup1=1 )
    
      grad2%u=0
    
      call derive_vector ( mesh_np1, problemp, grad2, elemsub=poisson_deriv, &
        coefficients=coefficientsp(2), oldvectors=oldvectors, elgroup1=2 )
    
    ! Integration of Maxwell stress tensor
      call create ( oldvectors_int, nvec=2)
      oldvectors_int%v(1)%p => grad1
      oldvectors_int%v(2)%p => grad2
    
      call integrate_boundary_elements ( mesh_np1, problemp, int_stress_bound, &
        elemsub=integrate_maxwell_stress, surface = 7, &
        coefficients=coefficients, oldvectors=oldvectors_int )
    
      moment = pi*H
      print * ,'Maxwell force', 2.0_dp * int_stress_bound
    
      call delete(grad1,grad2)
      call delete(oldvectors_int, oldvectors)
    end subroutine compute_maxwell_force
	
end program magnetic_particle9a



