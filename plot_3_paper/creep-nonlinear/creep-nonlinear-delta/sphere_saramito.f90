! Axisymmetrical EVP problem: a sphere translating in a cylinder
! on the center line of a cylindrical container
! Far-field boundary conditions assumed on walls of cylinder
! Either imposed force OR imposed velocity

program sphere_saramito

  use tfem_m
  use viscoelastic_elements_m
  use pardiso_m
  use io_utils_m

  implicit none

! constants

  integer, parameter :: &
    uintpl = 6,     & ! P2 velocities
    pintpl = 2,     & ! P1 pressures
    gintpl = 2,     & ! P1 gradients
    cintpl = 2,     & ! P1 conformation
    physqgrad = 1,  & ! physical quantity nr of the gradients
    physqvel = 2,   & ! physical quantity nr of the velocities
    physqpress = 3, & ! physical quantity nr of the pressures
    gauss = 6,      & ! 6-point integration of triangles
    gaussb = 3,     & ! 3-point integration of lines
    timeint1 = 1,   & ! (first-order) time integration (first step)
    timeint2 = 7,   & ! (second-order) time integration
    ncompc = 4,     & ! number of conformation tensor components
    nmodes = 1,     & ! number of modes
    startm = 51,    & ! start of material model data
    logc = 1,       & ! standard scheme or log transformation
    coorsys = 1       ! axisymmetric coordinate system

  logical, parameter :: fixU = .false. ! if true: prescribe the velocity

! variables

  integer :: &
    vtkevery = 50,       & ! vtk file every vtkevery steps. 0: means none
    numtimesteps = 3000, & ! number of time steps
    model = 3,           & ! 2: UCM/Oldroyd-B, 3: Giesekus,
                           ! 5: PTT linear, 6: PTT exponential
    alam_model = 2         ! 0:none, 1:elastic, 2: Saramito1, 3: Saramito2

  real(dp) :: &
    eta_s = 0.1_dp,    & ! solvent viscosity
    Gmod = 1.0_dp,     & ! G modulus
    lambda = 2.0_dp,   & ! relaxation time
    alphapar = 0.1_dp, & ! alpha parameter in the Giesekus model
    epspar = 0.1_dp,   & ! epsilon parameter in the PTT model
    tau_y = 2.0_dp,    & ! yield stress
    K = 1.0_dp,        & ! power-law coefficient
    n = 0.5_dp,        & ! power-law exponent
    deltat = 0.01_dp,  & ! time step
    beta = 1.0_dp,     & ! upwinding parameter in the SUPG method
    force = 1._dp,     & ! force on sphere (relevant for fixU=F)
    t0 = 1.0e30_dp,    & ! creep test: force is applied for t<=t0, then 0
    force_applied = 1._dp, & ! current force passed to constraint element
    Upart = 1._dp        ! flowrate (relevant for fixU=T)

  character(len=30) :: meshfile = 'mesh.out'

! definitions

  type(mesh_t) :: mesh
  type(input_probdef_t) :: input_probdef, input_probdefc, input_probdefc_proj
  type(problem_t), target :: problem, problemc, problemc_proj
  type(sysmatrix_t) :: sysmatrix, sysmatrixc, sysmatrixc_proj
  type(sysvector_t), target :: sol, solm1
  type(sysvector_t) :: rhsd, reacf
  type(oldvectors_t) :: oldvectors_ve
  type(coefficients_t) :: coefficients
  type(sysvector_t), dimension(ncompc,nmodes), target :: solc, solcm1, solc_proj
  type(sysvector_t), dimension(ncompc,nmodes) :: rhsc, rhsc_proj
  type(lu_pardiso_t) :: luc
  type(lu_pardiso_t) :: lu_exps_proj
  type(solver_options_pardiso_t) :: solver_options_u, solver_options_c
  type(vector_t), target :: meshvel
  type(subscriptvec_t) :: velz
  type(subscript_t) :: velzparticle

  integer :: icomp, step, i, ipost, ec, nln
  integer :: gnodes(3) = [1,3,5], pnodes(3) = [1,3,5], cnodes(3) = [1,3,5]
  real(dp) :: alpha, Fp
  real(dp), dimension(1) :: up
  real(dp) :: upn, upnm1, zp, zpn, zpnm1

! namelist for input of variables; read from standard input

  namelist /comppar/ meshfile, eta_s, model, alam_model, Gmod, lambda, &
    alphapar, epspar, tau_y, K, n, &
    deltat, force, t0, vtkevery, numtimesteps

  read ( unit=*, nml=comppar )

  call execute_command_line ('rm *.vtk' )

! set some parameters

  alpha = Gmod * lambda  ! DEVSS parameter

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

  coefficients%r(ec+1:ec+2) = [ Gmod, lambda ]

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

! read mesh

  call read_mesh ( mesh, filename=meshfile )

  call fill_mesh_parts ( mesh )

! problem definition of gradient/velocity/pressure

  call create_input_probdef ( mesh, input_probdef, nvec=5, nphysq=3 )

  input_probdef%vec_elementdof(1)%a(:,1) = 0
  input_probdef%vec_elementdof(1)%a(gnodes,1) = 4  ! G
  input_probdef%vec_elementdof(1)%a(:,2) = 2       ! velocity
  input_probdef%vec_elementdof(1)%a(:,3) = 0
  input_probdef%vec_elementdof(1)%a(pnodes,3) = 1  ! pressure
  input_probdef%vec_elementdof(1)%a(:,4) = 1       ! scalar, such as vorticity
  input_probdef%vec_elementdof(1)%a(:,5) = 4       ! tensor

  input_probdef%physq = [1,2,3]
  input_probdef%probnr = 1

! define essential boundaries

! center line and particle
  call define_essential ( mesh, input_probdef, curve1=8, curve2=9, &
    physq=physqvel, degfd=[0,1] )

! ends and cylinder wall
  call define_essential ( mesh, input_probdef, curve1=1, curve2=3, &
    physq=physqvel )

! pressure level
  call define_essential ( mesh, input_probdef, point=1, physq=physqpress )

! boundary condition on the particle

  if ( fixU ) then ! prescribed velocity

    call define_essential ( mesh, input_probdef, curve1=8, &
      physq=physqvel, degfd=[1,0] )

  else ! prescribed force using constraint

    call define_constraint ( mesh, input_probdef, curve1=8, &
      physq=physqvel, discretization='collocation', nodedof=1, naddunknowns=1 )

  end if

  call problem_definition ( input_probdef, mesh, problem )

! problem definition conformation tensor

  call create_input_probdef ( mesh, input_probdefc, nvec=3, nphysq=1 )

  input_probdefc%vec_elementdof(1)%a(:,1) = 0
  input_probdefc%vec_elementdof(1)%a(cnodes,1) = 1 ! c
  input_probdefc%vec_elementdof(1)%a(:,2) = 1      ! scalar for plotting
  input_probdefc%vec_elementdof(1)%a(:,3) = 4      ! tensor

  input_probdefc%physq = [1]
  input_probdefc%probnr = 2

! essential bc for conformation at the right inflow boundary

  call define_essential ( mesh, input_probdefc, curve1=1 )

  call problem_definition ( input_probdefc, mesh, problemc )

! create system vectors (solution and right-hand side)

  call create_sysvector ( problem, sol, solm1, rhsd, reacf )

! create mesh velocity vectors and vector to store mesh coordinates

  call create ( problem, meshvel, vec=2 )

! create subscripts for the mesh velocity

  call create_subscript ( mesh, problem, velz, vec=2, degfd=1 )
  call create_subscript ( mesh, problem, velzparticle, physqarr=[physqvel], &
    degfd=1, curves=[8] )

! initialize mesh veloicty to zero

  meshvel%u = 0

! fill solution vector with essential boundary conditions

  sol%u = 0

! create system matrix of gradient/velocity/pressure problem

  call create_sysmatrix_structure_base ( sysmatrix, mesh, problem )
  if (.not. fixU ) then
    call create_sysmatrix_structure_constraint ( sysmatrix, mesh, problem )
  end if
  call finalize_sysmatrix_structure ( sysmatrix )

  call create_sysmatrix_data ( sysmatrix )

! create system vectors (solution and right-hand side) for conformation and
! initialize vectors with zero stress

  call create ( problemc, solc, solcm1, rhsc )

  if ( logc == 0 ) then ! standard
    solc(1,1)%u = 1   ! initial czz
    solc(2,1)%u = 0   ! initial czr
    solc(3,1)%u = 1   ! initial crr
    solc(4,1)%u = 1   ! initial ctt
  else if ( logc == 1 ) then ! log scheme
    solc(1,1)%u = 0   ! initial szz
    solc(2,1)%u = 0   ! initial szr
    solc(3,1)%u = 0   ! initial srr
    solc(4,1)%u = 0   ! initial ctt
  end if

! problem definition for projected "c=exp(s)" of the log conformation s

  call create_input_probdef ( mesh, input_probdefc_proj, nvec=1, nphysq=1 )

  input_probdefc_proj%vec_elementdof(1)%a = 0
  input_probdefc_proj%vec_elementdof(1)%a(cnodes,1) = 1

  input_probdefc_proj%physq = [1]
  input_probdefc_proj%probnr = 3

  call problem_definition ( input_probdefc_proj, mesh, problemc_proj )

  call create ( problemc_proj, solc_proj, rhsc_proj )

  solc_proj(1,1)%u = 1   ! initial czz
  solc_proj(2,1)%u = 0   ! initial czr
  solc_proj(3,1)%u = 1   ! initial crr
  solc_proj(4,1)%u = 1   ! initial ctt

! create system matrix for conformation problem

  call create_sysmatrix_structure ( sysmatrixc, mesh, problemc )

  call create_sysmatrix_data ( sysmatrixc )

! create the structure oldvectors_ve

  call create_oldvectors ( oldvectors_ve, nsysvec=2, nsysvec2=3, nprob=3, &
    nvec=1 )

! store solution vectors and problem structures

  oldvectors_ve%s(1)%p => sol
  oldvectors_ve%s(2)%p => solm1
  oldvectors_ve%s2(1)%p => solc
  oldvectors_ve%s2(2)%p => solcm1
  oldvectors_ve%s2(3)%p => solc_proj
  oldvectors_ve%p(1)%p => problem
  oldvectors_ve%p(2)%p => problemc
  oldvectors_ve%p(3)%p => problemc_proj
  oldvectors_ve%v(1)%p => meshvel

! create and build system matrix for projection problem
! NOTE matrix remains constant and needs to be build once.

  call create_sysmatrix_structure ( sysmatrixc_proj, mesh, problemc_proj, &
    symmetric=.true. )
  call create_sysmatrix_data ( sysmatrixc_proj )

  call build_system ( mesh, problemc_proj, sysmatrixc_proj, &
    m2sysvector=rhsc_proj, elemsub=exps_projection_elem, &
    oldvectors=oldvectors_ve, coefficients=coefficients, &
    buildvector=.false. )

  call check ( sysmatrixc_proj )

! fill solution vector with essential boundary conditions

  sol%u = 0

  if ( fixU ) call fill_sysvector ( mesh, problem, sol, &
    curve1=1, curve2=3, physq=2, degfd=1, value=-Upart )

! creep test: set the applied force at t = 0

  if ( 0._dp <= t0 ) then
    force_applied = force
  else
    force_applied = 0._dp
  end if

! build (assemble) matrix/vector for gradient/velocity/pressure problem

  call build_vpG

! exps projection (for log conformation)

  if ( logc == 1 ) call solve_exps_projection

! build implicit terms

  call build_system ( mesh, problem, sysmatrix, rhsd, &
    elemsub=divtau_implicit_ce_elem_c, &
    oldvectors=oldvectors_ve, physqrow=[physqvel], physqcol=[physqvel],&
    addmatvec=.true., coefficients=coefficients )

  call add_effect_of_essential_to_rhs ( problem, sysmatrix, sol, rhsd )

  call copy ( sol, solm1 )

! solve gradient/velocity/pressure problem

  solver_options_u%matrixtype = 11

  call solve_system_pardiso ( sysmatrix, rhsd, sol, &
    solver_options=solver_options_u  )

  step = 0
  print *, 'step = ', step

! extract info on particle

  zp = 0._dp ! particle position at t=0

  open ( unit=10, file='particle.out' )

  if ( fixU ) then

    call reaction_forces ( problem, sysmatrix, sol, rhsd, reacf )
    Fp = sum ( reacf%u(velzparticle%s) )

    print *, 'Fp   = ', Fp

!   write initial particle data

    write(10,'(9es16.8)') step,  Fp

  else

    call get_sysvector_constraint ( mesh, problem, sol, constraint=1, &
      addunknowns=.true., u=up )

    print *, 'up   = ', up

!   write initial particle data

    write(10,'(9es16.8)') step, zp, up(1)

  end if

  ipost = 0
  if ( vtkevery > 0 ) then
     call postprocessing
     ipost = ipost + 1
  end if

! time stepping

  do step = 1, numtimesteps

    print *, 'step = ', step

    if ( step >= 2 ) then
      coefficients%i(22) = timeint2
    end if

    if ( fixU ) then

      meshvel%u = 0

    else

!     update the particle location

      if ( step == 1 ) then

!       advance with forward Euler
        upn = up(1)
        zpn = zp
        zp = zpn + deltat * upn

      else

!       advance particle position with 2nd order Adams-Bashforth
        upnm1 = upn
        upn = up(1)
        zpnm1 = zpn
        zpn = zp
        zp = zpn + deltat*(3*upn/2 - upnm1/2)

      end if

!     find the mesh velocity using backwards differencing

      if ( step == 1  ) then
        meshvel%u(velz%s) = ( zp - zpn ) / deltat
      else
        meshvel%u(velz%s) = ( 3*zp/2 - 2*zpn + zpnm1/2 ) / deltat
      end if

    end if

!   creep test: update applied force based on current time

    if ( step*deltat <= t0 ) then
      force_applied = force
    else
      force_applied = 0._dp
    end if

!   build (assemble) matrix/vector for gradient/velocity/pressure problem

    call build_vpG

!   exps projection (for log conformation)

    if ( logc == 1 ) call solve_exps_projection

!   build implicit terms

    call build_system ( mesh, problem, sysmatrix, rhsd, &
      elemsub=divtau_implicit_ce_elem_c, &
      oldvectors=oldvectors_ve, physqrow=[physqvel], physqcol=[physqvel],&
      addmatvec=.true., coefficients=coefficients )

    call add_effect_of_essential_to_rhs ( problem, sysmatrix, sol, rhsd )

    call copy ( sol, solm1 )

!   solve gradient/velocity/pressure problem

    solver_options_u%matrixtype = 11

    call solve_system_pardiso ( sysmatrix, rhsd, sol, &
      solver_options=solver_options_u  )

!   extract info on particle

    if ( fixU ) then

      call reaction_forces ( problem, sysmatrix, sol, rhsd, reacf )
      Fp = sum ( reacf%u(velzparticle%s) )

      print *, 'Fp   = ', Fp

!     write particle data

      write(10,'(9es16.8)') step*deltat, step*deltat*Upart

    else

      call get_sysvector_constraint ( mesh, problem, sol, constraint=1, &
        addunknowns=.true., u=up )

      print *, 'up   = ', up

!     write particle data

    write(10,'(9es16.8)') step*deltat, zp, up(1)

    end if

!   build (assemble) matrix and vector for conformation problem

    if ( coefficients%i(22) == timeint1 ) then

      call build_system ( mesh, problemc, sysmatrixc, m2sysvector=rhsc, &
        elemsub=ce_supg_elem, oldvectors=oldvectors_ve, &
        coefficients=coefficients )

    else

      call build_system ( mesh, problemc, sysmatrixc, m2sysvector=rhsc, &
        elemsub=ce_supg_elem_implicit_2nd_order, oldvectors=oldvectors_ve, &
        coefficients=coefficients )

    end if

    call check ( sysmatrixc )

    call copy ( solc, solcm1 )

    do icomp = 1, ncompc
      call add_effect_of_essential_to_rhs ( problemc, sysmatrixc, &
        solc(icomp,1), rhsc(icomp,1) )
    end do

!   solve conformation and keep LU decomposition in the loop over components

    solver_options_c%matrixtype = 11

    do icomp = 1, ncompc
      call solve_system_pardiso ( sysmatrixc, rhsc(icomp,1), solc(icomp,1), &
        lu=luc, solver_options=solver_options_c  )
    end do

    call delete ( luc )  ! remove LU decomposition and rebuild next time step

!   output velocity to vtk

    if ( vtkevery > 0 ) then
      if ( mod(step,vtkevery) == 0 ) then
        call postprocessing
        ipost = ipost + 1
      end if
    end if

  end do

  close ( unit=10 )

! delete all data including all allocated memory

  call delete ( mesh )
  call delete ( problem )
  call delete ( input_probdef )
  call delete ( sol, solm1, rhsd, reacf )
  call delete ( sysmatrix )
  call delete ( oldvectors_ve )
  call delete ( problemc )
  call delete ( input_probdefc )
  call delete ( sysmatrixc )
  call delete ( solc, solcm1, rhsc )
  call delete ( coefficients )
  call delete ( sysmatrixc_proj )
  call delete ( problemc_proj )
  call delete ( input_probdefc_proj )
  call delete ( solc_proj, rhsc_proj )
  call delete ( lu_exps_proj )
  call delete ( meshvel )
  call delete ( velz )

contains

  subroutine build_vpG

!   build (assemble) matrix and vector for gradient/velocity/pressure problem

!   stokes velocity/pressure
    call build_system ( mesh, problem, sysmatrix, rhsd, &
      elemsub=stokes_elem, physqrow=[physqvel,physqpress], &
      physqcol=[physqvel,physqpress], &
      coefficients=coefficients )

!   DEVSS-G
    call build_system ( mesh, problem, sysmatrix, rhsd, &
      elemsub=devssg_elem, addmatvec=.true., &
      physqrow=[physqgrad,physqvel], physqcol=[physqgrad,physqvel], &
      coefficients=coefficients )

!   set to zero off-diagonal blocks gradient-pressure
    call build_system ( mesh, problem, sysmatrix, rhsd, addmatvec=.true., &
      buildvector=.false., physqrow=[physqgrad], physqcol=[physqpress], &
      zeromatvec=.true. )
    call build_system ( mesh, problem, sysmatrix, rhsd, addmatvec=.true., &
      buildvector=.false., physqrow=[physqpress], physqcol=[physqgrad], &
      zeromatvec=.true. )

!   constraint for the particle
    if ( .not. fixU ) then
      call build_system_constraint ( mesh, problem, sysmatrix, rhsd, &
        constraint1=1, elemsub=elementc_axi, addmatvec=.true., &
        coefficients=coefficients )
    end if

  end subroutine build_vpG

! project c=exp(s) on discrete fem space

  subroutine solve_exps_projection

    type(solver_options_pardiso_t) :: solver_options_pardiso

    integer :: i, m

!   build vector only (matrix is constant)

    call build_system ( mesh, problemc_proj, sysmatrixc_proj, &
      m2sysvector=rhsc_proj, elemsub=exps_projection_elem, &
      oldvectors=oldvectors_ve, coefficients=coefficients, &
      buildmatrix=.false. )

    ! PARDISO real symmetric indefinite (matrix built with symmetric=.true.)
    solver_options_pardiso%matrixtype = -2

!   LU decomposition is done in the first call only

    do m = 1, nmodes
      do i = 1, ncompc

        call add_effect_of_essential_to_rhs ( problemc_proj, sysmatrixc_proj, &
           solc_proj(i,m), rhsc_proj(i,m) )

        call solve_system_pardiso ( sysmatrixc_proj, rhsc_proj(i,m), &
           solc_proj(i,m), lu=lu_exps_proj, &
           solver_options=solver_options_pardiso )

      end do
    end do

  end subroutine solve_exps_projection

! axisymmetric constraint matrix for a freely floating particle (collocation)

  subroutine elementc_axi ( mesh, problem, constr, elem, node, &
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

!   set shape function in the point

    if ( vector ) then

      elemvec = 0
      elemvecadd = 0
      if ( first ) elemvecadd(1) = force_applied

    end if

!   connection through collocation

    elemmat = 0
    elemmatadd = 0
    elemmat(1,1) = 1._dp
    elemmatadd(1,1) = -1._dp

  end subroutine elementc_axi

! write the data to .vtk files

  subroutine postprocessing

    character(len=399) :: filename
    type(oldvectors_t) :: oldvectors_dve
    type(vector_t) :: pressure, ctensor, vonmises, D_tensor, gammadot, &
      viscous_stress_tensor, viscoelastic_stress_tensor

    call create_oldvectors ( oldvectors_dve, nsysvec=1, nsysvec2=1 )
    oldvectors_dve%s(1)%p => sol

    call create_vector ( problem, gammadot, vec=4 )

    coefficients%i(13)=8

    call derive_vector ( mesh, problem, gammadot, elemsub=stokes_deriv, &
      coefficients=coefficients, oldvectors=oldvectors_dve )

    if (step*deltat <= 0.5_dp) then 

    open(unit=12, file='gammadot.out', status='unknown', position='append')

    write(12, '(*(es16.8,1x))') gammadot%u

    end if 

  end subroutine postprocessing

end program sphere_saramito
