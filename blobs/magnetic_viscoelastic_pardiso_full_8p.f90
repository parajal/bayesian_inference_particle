! Multi-particle version of magnetic_viscoelastic_pardiso_full.
! n_part freely-suspended rigid spheres in a triperiodic Gaussian-blob
! viscoelastic fluid.  The particle centres are either supplied directly by
! x_center/y_center/z_center arrays or placed by Latin-hypercube sampling
! subject to a minimum surface-to-surface gap (particle_min_dist, in radii) and
! a wall clearance (particle_wall_min_dist, in radii); if no valid layout is
! found the run stops with an error.  Each particle is loaded with the same
! creep-recovery force history (force on until time_stop, then off) and its
! trajectory is written to
! particle_<k>_trajectory_full.out.

program magnetic_viscoelastic_pardiso_full_8p

  use iso_fortran_env, only: int64
  use tfem_m
  use math_defs_m
  use hsl_ma57_m
  use pardiso_m
  use stokes_elements_m
  use viscoelastic_elements_m
  use convection_diffusion_supg_elements_m
  use io_utils_m
  use subs_particle_full_8p_m
  use subs_blobs_full_m
  use timer_m

  implicit none

  integer, parameter :: &
    uintpl = 6, pintpl = 2, gintpl = 2, cintpl = 2, &
    physqgrad = 1, physqvel = 2, physqpress = 3, &
    gauss = 8, gaussb = 3, ncompc = 6, nmodes = 1, &
    startm = 51, coorsys = 0, ndim = 3

  integer :: &
    vtkevery = 1, timeint1 = 1, timeint2 = 7, &
    model = 3, alam_model = 2, logc = 1

  real(dp) :: &
    eta_s = 0.1_dp, eta_p = 0.9_dp, alphapar = 0.1_dp, &
    lambda = 0.1_dp, epspar = 0.1_dp, &
    tau_y = 2._dp, K = 1._dp, time = 0._dp, n = 10._dp, &
    deltat = 0.0025_dp, force1 = 1._dp, beta = 1._dp, &
    phi_blob = 1._dp, phi_bg = 0._dp, phi_ref = 1._dp, &
    phi_exp = 1._dp, lambda_exp = 0._dp, base_G = 9._dp, &
    G_min = 1.0e-12_dp

  integer, parameter :: max_input_particles = 128
  real(dp), parameter :: rc = 1._dp
  real(dp), parameter :: center_unset = -huge(1._dp)
  real(dp) :: possible_blob_radii(4) = [ 0.5_dp, 1._dp, 2._dp, 3._dp ]
  real(dp) :: possible_blob_amplitudes(4) = [ 1._dp, 2._dp, 3._dp, 4._dp ]

  ! ----- multi-particle controls -----
  integer  :: n_part = 8, c_flr
  real(dp) :: particle_min_dist = 1._dp        ! min surface gap, in radii
  real(dp) :: particle_wall_min_dist = 0._dp   ! optional min wall clearance, in radii
  
  type(mesh_t) :: mesh_np1
  type(input_probdef_t) :: input_probdef, input_probdefc
  type(input_probdef_t) :: input_probdefc_projc, input_probdefphi
  type(problem_t), target :: problem, problemc, problemc_projc, problemphi
  type(sysmatrix_t) :: sysmatrix, sysmatrixc, sysmatrixc_projc, sysmatrixphi
  type(sysvector_t), target :: sol_np1, sol_n, sol_nm1
  type(sysvector_t), target :: solphi_np1, solphi_n, solphi_nm1
  type(sysvector_t), target :: solhat, solphi_hat
  type(sysvector_t) :: rhsd, rhsphi
  type(oldvectors_t) :: oldvectors_ve, oldvectors_phi, oldvectors_phi_cd
  type(coefficients_t) :: coefficients, coefficients_phi
  type(elvector_t), target :: modulus_phi, lambda_phi
  type(sysvector_t), dimension(ncompc,nmodes), target :: solc_np1, solc_n
  type(sysvector_t), dimension(ncompc,nmodes), target :: solc_nm1, rhsc
  type(sysvector_t), dimension(ncompc,nmodes), target :: solc_projc, rhsc_projc
  type(lu_pardiso_t) :: luc
  type(lu_ma57_t) :: lu_exps_projc
  type(solver_options_pardiso_t) :: solver_options_ve, solver_options_up
  type(solver_options_pardiso_t) :: solver_options_phi
  type(vector_t), target :: meshvel_np1
  type(problem_t) :: problem_lapl

  integer :: i, ip, step = 0, ipost = 0
  integer :: numtimesteps = 0, diagnostics_every = 0, ec, nln
  integer :: n_blobs = 10, seed = 42, particle_seed = 314159, shift_seed = 0
  integer :: nx_center, ny_center, nz_center
  integer :: periodic_x_constraint, periodic_y_constraint, periodic_z_constraint
  integer, allocatable :: particle_constraint(:), unit_part(:)
  real(dp) :: lx = 10._dp, ly = 10._dp, lz = 10._dp
  real(dp) :: dx_box = 1._dp, dx_part = 0.1_dp
  real(dp) :: x_center(max_input_particles) = center_unset
  real(dp) :: y_center(max_input_particles) = center_unset
  real(dp) :: z_center(max_input_particles) = center_unset
  real(dp) :: particle_blob_mesh_ratio = 2._dp
  real(dp) :: particle_transition_elems = 5._dp
  real(dp) :: particle_wall_min_elems = 2._dp
  real(dp) :: blob_refine_distance = -1._dp, blob_mesh_factor = 4._dp
  real(dp) :: alpha, G, time_stop = 0._dp
  real(dp) :: fluid_volume, G_shift(3) = 0._dp
  real(dp), allocatable :: up(:,:), un(:,:), xpn(:,:)
  real(dp), allocatable, dimension(:,:) :: meshcoor_n, meshcoor_nm1
  character(len=40) :: filename, fname
  logical :: refine_blobs = .true., refine_all_blobs = .false.
  logical :: uniform_mesh = .false.
  logical :: compact_gaussian = .false., overlap_blobs = .true.
  logical :: shift_G = .false.
  logical :: do_diagnostics

  namelist /comppar/ eta_s, eta_p, lambda, alphapar, epspar, &
    phi_blob, phi_bg, phi_ref, phi_exp, lambda_exp, &
    base_G, possible_blob_radii, possible_blob_amplitudes, &
    n_blobs, seed, particle_seed, x_center, y_center, z_center, &
    compact_gaussian, overlap_blobs, G_min, &
    shift_G, shift_seed, &
    model, alam_model, logc, tau_y, K, n, &
    dx_box, dx_part, deltat, &
    blob_refine_distance, blob_mesh_factor, particle_blob_mesh_ratio, &
    particle_transition_elems, particle_wall_min_elems, &
    refine_blobs, refine_all_blobs, uniform_mesh, &
    diagnostics_every, force1, time_stop, lx, ly, lz, &
    vtkevery, numtimesteps, &
    n_part, particle_min_dist, particle_wall_min_dist

  read ( unit=*, nml=comppar )

  nx_center = count ( x_center /= center_unset )
  ny_center = count ( y_center /= center_unset )
  nz_center = count ( z_center /= center_unset )
  if ( nx_center > 0 .or. ny_center > 0 .or. nz_center > 0 ) then
    if ( nx_center /= ny_center .or. nx_center /= nz_center ) then
      write(*,'(a,i0,a,i0,a,i0)') &
        'ERROR: x_center/y_center/z_center lengths differ: x=', &
        nx_center, ' y=', ny_center, ' z=', nz_center
      stop 1
    end if
    if ( nx_center < 1 ) stop 'at least one explicit center is required'
    n_part = nx_center
  end if

  if ( n_part < 1 ) stop 'n_part must be >= 1'

  call configure_random_blob_case ( lx, ly, lz, dx_part, dx_box, &
    eta_p, lambda, phi_blob, phi_bg, phi_ref, phi_exp, lambda_exp, &
    G_min, base_G, n_blobs, seed, possible_blob_radii, &
    possible_blob_amplitudes, compact_gaussian, overlap_blobs )

  MAXNUMWARNINGS = 0
  WARN_ON_NO_DATA_IN_NODES = .false.
  alpha = eta_p
  G = base_G
	

  call omp_set_num_threads(8) ! number of threads (overrules OMP_NUM_THREADS)
  call create_coefficients ( coefficients, ncoefi=100, ncoefr=50+6*nmodes )
  coefficients%i = [ uintpl, pintpl, 0, 0, gintpl, &
    physqvel, physqpress, 0, physqgrad, gauss, &
    gaussb, cintpl, 0, 0, 0, 0, 0, model, nmodes, startm, &
    logc, timeint1, coorsys, ( 0, i = 24, 100 ) ]

  coefficients%i(48) = 1
  coefficients%i(49) = 1
  coefficients%i(68) = 3
  coefficients%i(69) = 3
  coefficients%i(73) = 1
  coefficients%i(80) = alam_model
  ! coefficients%i(14) = 1  ! triperiodic body-force callback
  coefficients%r = 0._dp
  coefficients%r(1:10) = [ eta_s, 0._dp, 0._dp, alpha, 0._dp, &
    0._dp, 0._dp, deltat, beta, 0._dp ]
  ! coefficients%vfunc => periodic_body_force
  coefficients%r(6) =0._dp ! flow rate

  ec = 50
  coefficients%r(ec+1:ec+2) = [ G, lambda ]
  ec = 52

  select case ( model )
  case ( 2 )
    nln = 0
  case ( 3 )
    nln = 1
    coefficients%r(ec+1) = alphapar
  case ( 5, 6 )
    nln = 1
    coefficients%r(ec+1) = epspar
  case default
    stop 'unknown viscoelastic model'
  end select

  ec = ec + nln
  select case ( alam_model )
  case ( 0, 1 )
  case ( 2 )
    coefficients%r(ec+1) = tau_y
  case ( 3 )
    coefficients%r(ec+1:ec+3) = [ tau_y, K, n ]
  case default
    stop 'unknown adapted-lambda model'
  end select

  call create_coefficients ( coefficients_phi, ncoefi=450, ncoefr=400 )
  coefficients_phi%i = 0
  coefficients_phi%i(1) = uintpl
  coefficients_phi%i(2) = pintpl
  coefficients_phi%i(6) = physqvel
  coefficients_phi%i(10) = gauss
  coefficients_phi%i(11) = gaussb
  coefficients_phi%i(23) = coorsys
  coefficients_phi%i(48) = 1
  coefficients_phi%i(401) = timeint1
  coefficients_phi%i(402) = cintpl
  coefficients_phi%i(403) = 0
  coefficients_phi%i(404) = 0
  coefficients_phi%i(405) = 0
  coefficients_phi%i(406) = 0
  coefficients_phi%i(411) = 0
  coefficients_phi%r = 0._dp
  coefficients_phi%r(351:353) = [ deltat, 1._dp, 0._dp ]

  call update_modulus_from_phi ( mesh_np1, problem, coefficients, &
    modulus_phi, lambda_phi, oldvectors_phi, G )
  print *, 'base G =', base_G, '  G(phi_bg) =', shear_modulus_from_phi(phi_bg)
  print *, 'compact_gaussian =', compact_gaussian, '  overlap_blobs =', overlap_blobs

  call execute_command_line ( 'rm *.vtk' )

  ! ----- allocate per-particle storage -----
  allocate ( xp(n_part,ndim), rp(n_part) )
  allocate ( force_p(n_part,3), torque_p(n_part,3) )
  allocate ( up(n_part,6), un(n_part,6), xpn(n_part,ndim) )
  allocate ( particle_constraint(n_part), unit_part(n_part) )

  rp(:) = rc
  do ip = 1, n_part
    force_p(ip,:)  = [ force1, 0._dp, 0._dp ]   ! same creep load on every particle
    torque_p(ip,:) = 0._dp
  end do
  print '(1x,a,i0)', 'number of particles n_part = ', n_part
  print '(1x,a,f6.2,a,f6.2,a)', 'min surface gap = ', particle_min_dist, &
    ' radii,  wall clearance = ', particle_wall_min_dist, ' radii'
  print '(1x,a,f6.2)', 'min particle-wall mesh elements = ', &
    particle_wall_min_elems
  print *, 'force per particle', force_p(1,:)

  ! ----- particle placement (errors out if it cannot fit) -----
  if ( nx_center > 0 ) then
    do ip = 1, n_part
      xp(ip,:) = [ x_center(ip), y_center(ip), z_center(ip) ]
    end do
    call validate_particle_centers ( n_part, lx, ly, lz, rc, &
      particle_min_dist, particle_wall_min_dist, xp )
    print '(1x,a)', 'particle centers from x_center/y_center/z_center arrays'
  else
    call place_particles_lhs ( n_part, lx, ly, lz, rc, &
      particle_min_dist, particle_wall_min_dist, particle_seed, xp )
    print '(1x,a,i0)', 'particle centers from LHS seed = ', particle_seed
  end if
  do ip = 1, n_part
    print '(1x,a,i0,a,3f12.5)', 'particle ', ip, ' center =', xp(ip,:)
  end do

  fluid_volume = lx*ly*lz - real(n_part,dp)*4._dp*pi*rc**3/3._dp
  if ( fluid_volume <= tiny(1._dp) ) then
    stop 'non-positive fluid volume in periodic force balance'
  end if
  print '(1x,a,es14.6)', 'triperiodic fluid volume = ', fluid_volume

  call initialize_random_blob_centers
  if ( shift_G ) then
    ! Randomly translate the whole G field so this run's particles sample a
    ! different region of the same G (shift_seed=0 -> new shift every run).
    call shift_blob_centers ( shift_seed, G_shift )
    print '(1x,a,3f12.5)', 'random G-field shift (dx,dy,dz) =', G_shift
    if ( shift_seed == 0 ) then
      print '(1x,a)', '  shift_seed = 0: drawn from system clock (new every run)'
    else
      print '(1x,a,i0)', '  reproducible from shift_seed = ', shift_seed
    end if
  else
    print '(1x,a,i0)', 'G-field shift disabled; blob centers fixed by seed = ', &
      seed
  end if
  call print_blob_centers ( rc )

  print *, 'dx_box:', dx_box
  print *, 'dx_part:', dx_part

  call tic_w
  call generate_read_mesh
  call write_mesh_vtk ( mesh_np1, filename='mesh.vtk' )
  print *, 'number of nodes', mesh_np1%nnodes
  print *, 'number of elements ', mesh_np1%nelem
  call write_geometries_vtk ( mesh_np1, write_normals=.true. )
  call toc_w ( 'Generating mesh' )

  call define_problems_create_vectors
  call toc_w ( 'Defining flow problem' )

  print '(1x,a)', '--- system matrix sizes ---'
  print '(1x,a,i0)', 'flow (u-p-G) matrix order      N = ', size(sol_np1%u)
  print '(1x,a,i0)', 'conformation matrix order      N = ', size(solc_np1(1,1)%u)
  print '(1x,a,i0)', 'phi-transport matrix order     N = ', size(solphi_np1%u)

  if ( logc == 0 ) then
    solc_np1(1,1)%u = 1._dp; solc_np1(2,1)%u = 0._dp; solc_np1(3,1)%u = 0._dp
    solc_np1(4,1)%u = 1._dp; solc_np1(5,1)%u = 0._dp; solc_np1(6,1)%u = 1._dp
  else if ( logc == 1 ) then
    do i = 1, ncompc
      solc_np1(i,1)%u = 0._dp
    end do
  end if

  ! ----- open one trajectory file per particle -----
  do ip = 1, n_part
    write ( fname, '(a,i0,a)' ) 'particle_', ip, '_trajectory_full.out'
    open ( newunit=unit_part(ip), file=trim(fname) )
  end do

  call update_modulus_from_phi ( mesh_np1, problem, coefficients, &
    modulus_phi, lambda_phi, oldvectors_phi, G )
  call diagnose_phi_field ( mesh_np1, problemphi, solphi_np1, xp(1,1:3), rc )
  call build_solve_upG
  call toc_w ( 'solving flow problem' )

  do ip = 1, n_part
    call get_sysvector_constraint ( mesh_np1, problem, sol_np1, &
      constraint=particle_constraint(ip), addunknowns=.true., u=up(ip,:) )
  end do
  un = up

  print *, 'step', step
  do ip = 1, n_part
    write ( *, '(1X,A,I0,A,3F12.6,A,3F12.6)' ) 'p', ip, ' x =', xp(ip,1:3), &
      '  U =', up(ip,1:3)
    write ( unit_part(ip), '(10es16.8)' ) time, xp(ip,1:3), up(ip,1:6)
  end do

  call postprocessing ( mesh_np1, sol_np1, solc_np1, ipost=ipost )
  ipost = ipost + 1

  call copy ( sol_np1, sol_n )
  call copy ( solc_np1, solc_n )
  call copy ( solphi_np1, solphi_n )
  meshcoor_n = mesh_np1%coor(mesh_np1%elementsets(1)%nodes,:)
  time = 0._dp

  do step = 1, numtimesteps
    time = time + deltat
    if ( step > 1 ) coefficients%i(22) = timeint2

    xpn = xp
    do ip = 1, n_part
      if ( step == 1 ) then
        xp(ip,:) = xp(ip,:) + deltat*up(ip,1:3)
      else
        xp(ip,:) = xp(ip,:) + deltat*( 1.5_dp*up(ip,1:3) - 0.5_dp*un(ip,1:3) )
      end if
    end do
    un = up

    call update_mesh_nodes_multi ( mesh_np1, problem_lapl, n_part, xp-xpn )
    call find_bounds_blocks ( mesh_np1 )

    if ( step == 1 ) then
      meshvel_np1%u = reshape ( transpose ( &
        mesh_np1%coor(mesh_np1%elementsets(1)%nodes,:) - meshcoor_n ) / &
        deltat, [mesh_np1%ndim*mesh_np1%elementsets(1)%nnodes] )
    else
      meshvel_np1%u = reshape ( transpose ( &
        ( 1.5_dp*mesh_np1%coor(mesh_np1%elementsets(1)%nodes,:) - &
          2._dp*meshcoor_n + 0.5_dp*meshcoor_nm1 ) / deltat ), &
        [mesh_np1%ndim*mesh_np1%elementsets(1)%nnodes] )
    end if

    if ( step == 1 ) then
      solhat%u = sol_n%u
      solphi_hat%u = solphi_n%u
      coefficients_phi%i(401) = 1
    else
      solhat%u = 2._dp*sol_n%u - sol_nm1%u
      solphi_hat%u = 2._dp*solphi_n%u - solphi_nm1%u
      coefficients_phi%i(401) = 2
    end if

    call build_system ( mesh_np1, problemphi, sysmatrixphi, rhsphi, &
      elemsub=conv_diff_supg_elem, coefficients=coefficients_phi, &
      oldvectors=oldvectors_phi_cd, elgroup1=1 )
    call build_system_constraint ( mesh_np1, problemphi, sysmatrixphi, &
      rhsphi, constraint1=1, elemsub=stokes_constr_node_conn, addmatvec=.true. )
    call build_system_constraint ( mesh_np1, problemphi, sysmatrixphi, &
      rhsphi, constraint1=2, elemsub=stokes_constr_node_conn, addmatvec=.true. )
    call build_system_constraint ( mesh_np1, problemphi, sysmatrixphi, &
      rhsphi, constraint1=3, elemsub=stokes_constr_node_conn, addmatvec=.true. )
    call add_effect_of_essential_to_rhs ( problemphi, sysmatrixphi, &
      solphi_np1, rhsphi )
    solver_options_phi%matrixtype = 11
    call solve_system_pardiso ( sysmatrixphi, rhsphi, solphi_np1, &
      solver_options=solver_options_phi )
    call enforce_phi_bounds ( solphi_np1 )
    call toc_w ( 'convecting phi' )

    if ( diagnostics_every > 0 ) then
      do_diagnostics = mod(step,diagnostics_every) == 0
    else
      do_diagnostics = .false.
    end if
    call update_modulus_from_phi ( mesh_np1, problem, coefficients, &
      modulus_phi, lambda_phi, oldvectors_phi, G, verbose=do_diagnostics )
    call build_solve_upG
    call toc_w ( 'solving flow problem' )

    call copy ( sol_n, sol_nm1 )
    call copy ( sol_np1, sol_n )

    call build_solve_ve
    call toc_w ( 'solving viscoelastic problem' )

    do ip = 1, n_part
      call get_sysvector_constraint ( mesh_np1, problem, sol_np1, &
        constraint=particle_constraint(ip), addunknowns=.true., u=up(ip,:) )
    end do

    print *, 'step', step
    do ip = 1, n_part
      write ( *, '(1X,A,I0,A,3F12.6,A,3F12.6)' ) 'p', ip, ' x =', xp(ip,1:3), &
        '  U =', up(ip,1:3)
      write ( unit_part(ip), '(10es16.8)' ) time, xp(ip,1:3), up(ip,1:6)
    end do

    if ( vtkevery > 0 ) then
      if ( mod(step,vtkevery) == 0 ) then
        call postprocessing ( mesh_np1, sol_np1, solc_np1, ipost=ipost )
        ipost = ipost + 1
      end if
    end if
    call toc_w ( 'postprocessing' )

    call copy ( solc_n, solc_nm1 )
    call copy ( solc_np1, solc_n )
    call copy ( solphi_n, solphi_nm1 )
    call copy ( solphi_np1, solphi_n )
    meshcoor_nm1 = meshcoor_n
    meshcoor_n = mesh_np1%coor(mesh_np1%elementsets(1)%nodes,:)
  end do

  do ip = 1, n_part
    close ( unit_part(ip) )
  end do
  call delete_old_problems
  call delete ( coefficients )
  call delete ( coefficients_phi )
  call cleanup_blobs
  deallocate ( xp, rp, force_p, torque_p, up, un, xpn )
  deallocate ( particle_constraint, unit_part )

  contains

  subroutine generate_read_mesh

    call write_gmsh_parameters ( lx, ly, lz, xp, rp, dx_box, dx_part, &
      refine_blobs, refine_all_blobs, blob_refine_distance, blob_mesh_factor, &
      particle_blob_mesh_ratio, particle_transition_elems, &
      particle_wall_min_elems, uniform_mesh )

    call execute_command_line ( 'gmsh -3 -order 2 -algo hxt -o mesh.msh &
                  &mesh.geo > outputmesh.out' )
    call read_mesh_gmsh ( mesh_np1, filename='mesh.msh', ndim=3, physgeom=.true. )

    call add_to_mesh ( mesh_np1, matchingsurface=[4,2], replace=4, &
      displacement=[-lx,0._dp,0._dp] )
    call add_to_mesh ( mesh_np1, matchingsurface=[1,3], replace=1, &
      displacement=[0._dp,-ly,0._dp] )
    call add_to_mesh ( mesh_np1, matchingsurface=[6,5], replace=6, &
      displacement=[0._dp,0._dp,-lz] )

    call fill_mesh_parts ( mesh_np1 )
    call add_to_mesh ( mesh_np1, elementset='elements', group=1, &
      elements=(/(i,i=1,mesh_np1%nelem,1)/) )
    call add_to_mesh ( mesh_np1, elementset='nodes', elementsetnr=1 )
    call add_to_mesh ( mesh_np1, nodeset='nodes', &
      nodes=mesh_np1%elementsets(1)%nodes )

    allocate ( meshcoor_n(mesh_np1%elementsets(1)%nnodes,mesh_np1%ndim), &
      meshcoor_nm1(mesh_np1%elementsets(1)%nnodes,mesh_np1%ndim) )

  end subroutine generate_read_mesh

  subroutine define_problems_create_vectors

    call create_input_probdef ( mesh_np1, input_probdef, nvec=4, nphysq=3 )

    input_probdef%vec_elementdof(1)%a = reshape ( (/ &
      9,0,9,0,9,0,0,0,0,9, &
      3,3,3,3,3,3,3,3,3,3, &
      1,0,1,0,1,0,0,0,0,1, &
      1,1,1,1,1,1,1,1,1,1 /), (/10,4/) )

    input_probdef%physq = (/1,2,3/)
    input_probdef%probnr = 1

    call define_essential ( mesh_np1, input_probdef, point=1, physq=physqpress )
    call define_essential ( mesh_np1, input_probdef, point=1, physq=physqvel )

    ! one rigid-body + prescribed-force constraint per particle (surfaces 7..6+n)
    do ip = 1, n_part
      call define_constraint ( mesh_np1, input_probdef, surface1=part_surface0+ip, &
        physq=physqvel, discretization='collocation', nodedof=3, &
        naddunknowns=6, num=particle_constraint(ip) )
    end do

    ! Periodic constraints for the triperiodic domain.
    call define_constraint ( mesh_np1, input_probdef, surface1=4, &
      surface2=2, physq=physqvel, discretization='collocation', &
      num=periodic_x_constraint )
    call define_constraint ( mesh_np1, input_probdef, surface1=1, &
      surface2=3, physq=physqvel, discretization='collocation', &
      excludesurfaces=[4], num=periodic_y_constraint )
    call define_constraint ( mesh_np1, input_probdef, surface1=6, &
      surface2=5, physq=physqvel, discretization='collocation', &
      excludesurfaces=[4,1], num=periodic_z_constraint )

      call define_constraint ( mesh_np1, input_probdef, physq=physqvel, &
        surface1=2, nglobalc=1, num=c_flr )

    call problem_definition ( input_probdef, mesh_np1, problem )

    call create_input_probdef ( mesh_np1, input_probdefc, nvec=2, nphysq=1 )
    input_probdefc%vec_elementdof(1)%a = reshape ( (/ &
      1,0,1,0,1,0,0,0,0,1, &
      1,1,1,1,1,1,1,1,1,1 /), (/10,2/) )
    input_probdefc%physq = (/1/)
    input_probdefc%probnr = 2

    call define_constraint ( mesh_np1, input_probdefc, surface1=4, &
      surface2=2, discretization='collocation' )
    call define_constraint ( mesh_np1, input_probdefc, surface1=1, &
      surface2=3, discretization='collocation', excludesurfaces=[4] )
    call define_constraint ( mesh_np1, input_probdefc, surface1=6, &
      surface2=5, discretization='collocation', excludesurfaces=[4,1] )

    call problem_definition ( input_probdefc, mesh_np1, problemc )

    call create_input_probdef ( mesh_np1, input_probdefphi, nvec=1, nphysq=1 )
    input_probdefphi%vec_elementdof(1)%a = &
      reshape ( [ 1,0,1,0,1,0,0,0,0,1 ], [10,1] )
    input_probdefphi%physq = [1]
    input_probdefphi%probnr = 4

    call define_constraint ( mesh_np1, input_probdefphi, surface1=4, &
      surface2=2, discretization='collocation' )
    call define_constraint ( mesh_np1, input_probdefphi, surface1=1, &
      surface2=3, discretization='collocation', excludesurfaces=[4] )
    call define_constraint ( mesh_np1, input_probdefphi, surface1=6, &
      surface2=5, discretization='collocation', excludesurfaces=[4,1] )

    call problem_definition ( input_probdefphi, mesh_np1, problemphi )

    call create_sysvector ( problem, sol_np1, sol_n, sol_nm1, rhsd )
    call create_sysvector ( problem, solhat )
    call create ( problem, meshvel_np1, vec=2 )
    call create ( problemc, solc_np1, solc_n, solc_nm1, rhsc )
    call create ( problemphi, solphi_np1, solphi_n, solphi_nm1, rhsphi )
    call create ( problemphi, solphi_hat )
    call create ( mesh_np1, modulus_phi, nreal1d=nmodes )
    call create ( mesh_np1, lambda_phi, nreal1d=nmodes )

    call create_sysmatrix_structure_base ( sysmatrix, mesh_np1, problem, &
      symmetric=.false. )
    call create_sysmatrix_structure_constraint ( sysmatrix, mesh_np1, problem )
    call finalize_sysmatrix_structure ( sysmatrix )
    call create_sysmatrix_data ( sysmatrix )

    call create_sysmatrix_structure_base ( sysmatrixc, mesh_np1, problemc, &
      symmetric=.false. )
    call create_sysmatrix_structure_constraint ( sysmatrixc, mesh_np1, problemc )
    call finalize_sysmatrix_structure ( sysmatrixc )
    call create_sysmatrix_data ( sysmatrixc )

    call create_sysmatrix_structure_base ( sysmatrixphi, mesh_np1, &
      problemphi, symmetric=.false. )
    call create_sysmatrix_structure_constraint ( sysmatrixphi, mesh_np1, problemphi )
    call finalize_sysmatrix_structure ( sysmatrixphi )
    call create_sysmatrix_data ( sysmatrixphi )

    sol_nm1%u = 0.0_dp
    sol_n%u = 0.0_dp
    sol_np1%u = 0.0_dp
    do i = 1, ncompc
      solc_nm1(i,1)%u = 0.0_dp
      solc_n(i,1)%u = 0.0_dp
    end do
    solphi_np1%u = phi_bg
    solphi_n%u = phi_bg
    solphi_nm1%u = phi_bg
    call fill_phi_blob_from_xp ( mesh_np1, problemphi, solphi_np1, &
      solphi_n, solphi_nm1 )
    meshvel_np1%u = 0.0_dp

    call create_input_probdef ( mesh_np1, input_probdefc_projc, nvec=1, nphysq=1 )
    input_probdefc_projc%vec_elementdof(1)%a = &
      reshape ( [ 1,0,1,0,1,0,0,0,0,1 ], [10,1] )
    input_probdefc_projc%physq = [1]
    input_probdefc_projc%probnr = 3

    call define_constraint ( mesh_np1, input_probdefc_projc, surface1=4, &
      surface2=2, discretization='collocation' )
    call define_constraint ( mesh_np1, input_probdefc_projc, surface1=1, &
      surface2=3, discretization='collocation', excludesurfaces=[4] )
    call define_constraint ( mesh_np1, input_probdefc_projc, surface1=6, &
      surface2=5, discretization='collocation', excludesurfaces=[4,1] )

    call problem_definition ( input_probdefc_projc, mesh_np1, problemc_projc )
    call create ( problemc_projc, solc_projc, rhsc_projc )

    solc_projc(1,1)%u = 1; solc_projc(2,1)%u = 0; solc_projc(3,1)%u = 0
    solc_projc(4,1)%u = 1; solc_projc(5,1)%u = 0; solc_projc(6,1)%u = 1

    call create_sysmatrix_structure_base ( sysmatrixc_projc, mesh_np1, &
      problemc_projc, symmetric=.true. )
    call create_sysmatrix_structure_constraint ( sysmatrixc_projc, &
      mesh_np1, problemc_projc )
    call finalize_sysmatrix_structure ( sysmatrixc_projc )
    call create_sysmatrix_data ( sysmatrixc_projc )

    call create_oldvectors ( oldvectors_ve, nsysvec=2, nsysvec2=3, nprob=3, &
      nvec=2, nelvec=2 )
    oldvectors_ve%s(1)%p => sol_n
    oldvectors_ve%s(2)%p => sol_nm1
    oldvectors_ve%s2(1)%p => solc_n
    oldvectors_ve%s2(2)%p => solc_nm1
    oldvectors_ve%s2(3)%p => solc_projc
    oldvectors_ve%p(1)%p => problem
    oldvectors_ve%p(2)%p => problemc
    oldvectors_ve%p(3)%p => problemc_projc
    oldvectors_ve%v(1)%p => meshvel_np1
    oldvectors_ve%e(1)%p => modulus_phi
    oldvectors_ve%e(2)%p => lambda_phi

    call create_oldvectors ( oldvectors_phi, nsysvec=1, nprob=1, nelvec=2 )
    oldvectors_phi%s(1)%p => solphi_np1
    oldvectors_phi%p(1)%p => problemphi
    oldvectors_phi%e(1)%p => modulus_phi
    oldvectors_phi%e(2)%p => lambda_phi

    call create_oldvectors ( oldvectors_phi_cd, nsysvec=5, nprob=2, nvec=1 )
    oldvectors_phi_cd%s(1)%p => solhat
    oldvectors_phi_cd%s(2)%p => solphi_n
    oldvectors_phi_cd%s(3)%p => solphi_nm1
    oldvectors_phi_cd%s(4)%p => solphi_hat
    oldvectors_phi_cd%s(5)%p => solphi_np1
    oldvectors_phi_cd%p(1)%p => problem
    oldvectors_phi_cd%p(2)%p => problemphi
    oldvectors_phi_cd%v(1)%p => meshvel_np1

  end subroutine define_problems_create_vectors

  subroutine build_solve_upG

    integer :: ip

    ! Creep recovery: switch every particle's force off after time_stop.
    if ( time_stop > 0._dp .and. time >= time_stop ) then
      force_p = 0._dp
    end if

    ! Force balance for the triperiodic cell: uniform body force cancels the
    ! total applied particle force.
    ! periodic_body_force_density = -sum(force_p, dim=1) / fluid_volume

    call fill_sysvector ( mesh_np1, problem, sol_np1, point=1, &
      physq=physqvel, value=0._dp )

    call build_system ( mesh_np1, problem, sysmatrix, rhsd, &
      elemsub=stokes_elem, physqrow=[2,3], physqcol=[2,3], &
      coefficients=coefficients, elgroup1=1 )
    call build_system ( mesh_np1, problem, sysmatrix, rhsd, &
      elemsub=devssg_elem, addmatvec=.true., physqrow=[1,2], &
      physqcol=[1,2], coefficients=coefficients, elgroup1=1 )
    call build_system ( mesh_np1, problem, sysmatrix, rhsd, addmatvec=.true., &
      buildvector=.false., physqrow=[1], physqcol=[3], zeromatvec=.true., &
      elgroup1=1 )
    call build_system ( mesh_np1, problem, sysmatrix, rhsd, addmatvec=.true., &
      buildvector=.false., physqrow=[3], physqcol=[1], zeromatvec=.true., &
      elgroup1=1 )

    ! print *, 'time', time, '  triperiodic background force density', &
    !   periodic_body_force_density

    call build_system_constraint ( mesh_np1, problem, sysmatrix, rhsd, &
      constraint1=periodic_x_constraint, &
      elemsub=stokes_constr_node_conn, addmatvec=.true. )
    call build_system_constraint ( mesh_np1, problem, sysmatrix, rhsd, &
      constraint1=periodic_y_constraint, &
      elemsub=stokes_constr_node_conn, addmatvec=.true. )
    call build_system_constraint ( mesh_np1, problem, sysmatrix, rhsd, &
      constraint1=periodic_z_constraint, &
      elemsub=stokes_constr_node_conn, addmatvec=.true. )

    do ip = 1, n_part
      call build_system_constraint ( mesh_np1, problem, sysmatrix, rhsd, &
        constraint1=particle_constraint(ip), elemsub=elementc_full_force, &
        addmatvec=.true. )
    end do

    call build_system_constraint ( mesh_np1, problem, sysmatrix, rhsd, &
          constraint1=c_flr, elemsub=stokes_constr_flowr, addmatvec=.true., &
          coefficients=coefficients )
    if ( step >= 1 ) then
      if ( logc == 1 ) call solve_exps_projection
      call build_system ( mesh_np1, problem, sysmatrix, rhsd, &
        elemsub=divtau_implicit_ce_elem_c, oldvectors=oldvectors_ve, &
        physqrow=[2], physqcol=[2], addmatvec=.true., &
        coefficients=coefficients, elgroup1=1 )
    end if

    call check ( sysmatrix )
    call add_effect_of_essential_to_rhs ( problem, sysmatrix, sol_np1, rhsd )

    solver_options_up%matrixtype = 11
    call solve_system_pardiso ( sysmatrix, rhsd, sol_np1, &
      solver_options=solver_options_up )

  end subroutine build_solve_upG

  subroutine build_solve_ve

    integer :: icomp

    if ( step == 1 ) then
      call build_system ( mesh_np1, problemc, sysmatrixc, m2sysvector=rhsc, &
        elemsub=ce_supg_elem, oldvectors=oldvectors_ve, &
        coefficients=coefficients, elgroup1=1 )
    else
      call build_system ( mesh_np1, problemc, sysmatrixc, m2sysvector=rhsc, &
        elemsub=ce_supg_elem_implicit_2nd_order, oldvectors=oldvectors_ve, &
        coefficients=coefficients, elgroup1=1 )
    end if

    call build_system_constraint ( mesh_np1, problemc, sysmatrixc, &
      m2sysvector=rhsc, constraint1=1, elemsub=stokes_constr_node_conn, &
      addmatvec=.true. )
    call build_system_constraint ( mesh_np1, problemc, sysmatrixc, &
      m2sysvector=rhsc, constraint1=2, elemsub=stokes_constr_node_conn, &
      addmatvec=.true. )
    call build_system_constraint ( mesh_np1, problemc, sysmatrixc, &
      m2sysvector=rhsc, constraint1=3, elemsub=stokes_constr_node_conn, &
      addmatvec=.true. )

    call check ( sysmatrixc )
    solver_options_ve%matrixtype = 11

    do icomp = 1, ncompc
      call solve_system_pardiso ( sysmatrixc, rhsc(icomp,1), &
        solc_np1(icomp,1), luc, solver_options=solver_options_ve )
    end do

    call delete ( luc )

  end subroutine build_solve_ve

  subroutine postprocessing ( mesh, sol, solc, ipost, check_proj )

    type(mesh_t), intent(inout) :: mesh
    type(sysvector_t), intent(in), target :: sol
    type(sysvector_t), dimension(:,:), intent(in), target :: solc
    integer, intent(in), optional :: ipost
    character(len=*), intent(in), optional :: check_proj

    type(oldvectors_t) :: oldvectors_dve
    type(vector_t) :: cvec(6), pressure, gammadot
    character(len=2), parameter :: cname(6) = &
      [ character(len=2) :: 'xx', 'xy', 'xz', 'yy', 'yz', 'zz' ]
    integer :: kk

    if ( .not. mesh%meshparts) call fill_mesh_parts ( mesh )

    call create_oldvectors ( oldvectors_dve, nsysvec=1, nsysvec2=1 )
    oldvectors_dve%s(1)%p => sol

    call create_vector ( problem, pressure, vec=4 )
    call create_vector ( problem, gammadot, vec=4 )

    call derive_vector ( mesh, problem, pressure, &
      elemsub=stokes_pressure, coefficients=coefficients, &
      oldvectors=oldvectors_dve, elgroup1 = 1 )

    coefficients%i(13) = 8
    call derive_vector ( mesh, problem, gammadot, elemsub=stokes_deriv, &
      coefficients=coefficients, oldvectors=oldvectors_dve, elgroup1 = 1 )

    if ( present(ipost) ) write(filename,'(a,i4.4,a)') 'flow', ipost, '.vtk'
    if ( present(check_proj) ) write(filename,'(a,i4.4,a)') 'flow_'//check_proj//'.vtk'
    call write_scalar_vtk ( mesh, problem, vector=pressure, &
      dataname='pressure',  filename=filename, groups = [1]  )
    call write_vector_vtk ( mesh, problem, filename=filename, &
      dataname='velocity', sysvector=sol, physq=physqvel, &
      append=.true., groups = [1]  )
    call write_scalar_vtk ( mesh, problem, vector=gammadot, &
      filename=filename, dataname='gammadot', append=.true., groups = [1] )

    call write_phi_vtk_fields ( mesh, problem, problemphi, solphi_np1, filename )

    oldvectors_dve%s2(1)%p => solc
    do kk = 1, 6
      call create_vector ( problemc, cvec(kk), vec=2 )
      coefficients%i(13) = kk
      call derive_vector ( mesh, problemc, cvec(kk), elemsub=deriv_conformation, &
        coefficients=coefficients, oldvectors=oldvectors_dve, elgroup1 = 1 )
    end do

    if ( present(ipost) ) write(filename,'(a,i4.4,a)') 'c', ipost, '.vtk'
    if ( present(check_proj) ) write(filename,'(a,i4.4,a)') 'c_'//check_proj//'.vtk'

    do kk = 1, 6
      call write_scalar_vtk ( mesh, problemc, vector=cvec(kk), filename=filename, &
        dataname='c'//cname(kk), append=(kk>1), groups=[1] )
    end do

    do kk = 1, 6
      call delete ( cvec(kk) )
    end do
    call delete ( pressure, gammadot )
    call delete ( oldvectors_dve )

  end subroutine postprocessing

  subroutine delete_old_problems

    call delete ( problem, problemc )
    call delete ( problemphi )
    if ( problem_lapl%created ) call delete ( problem_lapl )
    call delete ( sysmatrix, sysmatrixc )
    call delete ( sysmatrixphi )
    call delete ( input_probdef, input_probdefc )
    call delete ( input_probdefphi )
    call delete ( mesh_np1 )
    call delete ( sol_np1, sol_n, sol_nm1 )
    call delete ( solc_np1, solc_n, solc_nm1 )
    call delete ( solphi_np1, solphi_n, solphi_nm1 )
    call delete ( solhat, solphi_hat )
    call delete ( rhsc)
    call delete ( rhsd, rhsphi )
    call delete ( oldvectors_ve, oldvectors_phi, oldvectors_phi_cd )
    call delete ( modulus_phi )
    call delete ( lambda_phi )
    call delete ( meshvel_np1 )
    call delete ( problemc_projc )
    call delete ( input_probdefc_projc )
    call delete ( solc_projc, rhsc_projc )
    call delete ( sysmatrixc_projc )

    if ( allocated(meshcoor_n) ) deallocate ( meshcoor_n )
    if ( allocated(meshcoor_nm1) ) deallocate ( meshcoor_nm1 )

  end subroutine delete_old_problems

  subroutine solve_exps_projection

    type(solver_options_ma57_t) :: solver_options_ma57
    integer :: ii, m

    call build_system ( mesh_np1, problemc_projc, sysmatrixc_projc, &
      m2sysvector=rhsc_projc, elemsub=exps_projection_elem, &
      oldvectors=oldvectors_ve, coefficients=coefficients, elgroup1 =1  )

    call build_system_constraint ( mesh_np1, problemc_projc, &
      sysmatrixc_projc, m2sysvector=rhsc_projc, constraint1=1, &
      elemsub=stokes_constr_node_conn, addmatvec=.true. )
    call build_system_constraint ( mesh_np1, problemc_projc, &
      sysmatrixc_projc, m2sysvector=rhsc_projc, constraint1=2, &
      elemsub=stokes_constr_node_conn, addmatvec=.true. )
    call build_system_constraint ( mesh_np1, problemc_projc, &
      sysmatrixc_projc, m2sysvector=rhsc_projc, constraint1=3, &
      elemsub=stokes_constr_node_conn, addmatvec=.true. )

    call check ( sysmatrixc_projc )
    solver_options_ma57%integer_storage = 1.3
    solver_options_ma57%real_storage    = 1.3

    do m = 1, nmodes
      do ii = 1, ncompc
        call add_effect_of_essential_to_rhs ( problemc_projc, &
          sysmatrixc_projc, solc_projc(ii,m), rhsc_projc(ii,m) )
        call solve_system_ma57 ( sysmatrixc_projc, rhsc_projc(ii,m), &
           solc_projc(ii,m), lu_exps_projc, solver_options=solver_options_ma57 )
      end do
    end do

    call delete ( lu_exps_projc )

  end subroutine solve_exps_projection

end program magnetic_viscoelastic_pardiso_full_8p
