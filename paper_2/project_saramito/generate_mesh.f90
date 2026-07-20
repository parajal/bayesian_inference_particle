program generate_mesh

  use tfem_m
  use io_utils_m
  use figplot_m
  use math_defs_m

! definitions

  type(mesh_t) :: mesh
  type(plot_options_t) :: plot_options

! definitions for the refinement fields

  type(refinement_fields_t) :: refinement_fields
  real(dp), dimension(:,:), allocatable :: refinement_coor

! variables

! if sphere=.true, particle is spherical with a radius radr.
! if sphere=.false, particle is a spheroid with independent radii radz, radr.
  logical :: sphere = .true. 

  real(dp) :: &
    radz = 1._dp,      & ! radius of the particle in z-direction
    radr = 1._dp,      & ! radius of the particle in r-direction
    radc = 2._dp,      & ! radius of the enclosing cylinder
    lenc1 = 24._dp,    & ! length of the enclosing cylinder (left of particle)
    lenc2 = 6._dp,     & ! length of the enclosing cylinder (right of particle)
    nelem_min = 1._dp, & ! minimal number of elements in gap
    dx_part = 0.2_dp,  & ! element size on the particle boundary
    dx_box = 1._dp       ! element size on the box boundary

  character(len=30) :: meshfile = 'mesh.out'
  real(dp) :: gap
  integer :: nrefine

! namelist for input of variables; read from standard input

  namelist /comppar/ meshfile, sphere, radr, radz, radc, lenc1, lenc2, &
    dx_part, dx_box, nelem_min

  read ( unit=*, nml=comppar )

  if ( sphere ) radz = radr

! generate and read mesh

  nrefine = 40 ! number of refinment points on boundary of particle

  allocate ( refinement_coor(nrefine,3) )

  call coors_ellipse ( refinement_coor )

  call add_refinement_field ( refinement_fields, &
    coor=refinement_coor, distmin=(radz+radr)/2, distmax=4*(radz+radr), &
    dx_fine=dx_part, dx_coarse=dx_box )

  gap = radc-radr

  refinement_coor(:,1) = [(-radz/2+radz*(i-1)/(nrefine-1),i=1,nrefine)]
  refinement_coor(:,2) = radr + gap/2
  refinement_coor(:,3) = 0

  call add_refinement_field ( refinement_fields, &
    coor=refinement_coor, distmin=gap, distmax=16*gap, &
    dx_fine=gap/nelem_min, dx_coarse=dx_box )

  deallocate ( refinement_coor )

! write the mesh parameters

  open ( unit=25, file='mesh.geo' )
  write ( 25, '(1X,A,es16.8,A)' ) 'radz = ', radz, ';'
  write ( 25, '(1X,A,es16.8,A)' ) 'radr = ', radr, ';'
  write ( 25, '(1X,A,es16.8,A)' ) 'lenc1 = ', lenc1, ';'
  write ( 25, '(1X,A,es16.8,A)' ) 'lenc2 = ', lenc2, ';'
  write ( 25, '(1X,A,es16.8,A)' ) 'radc = ', radc, ';'
  write ( 25, '(1X,A,es16.8,A)' ) 'dx_box = ', dx_box, ';'
  write ( 25, '(1X,A,es16.8,A)' ) 'dx_part = ', dx_part, ';'

  call write_refinement_fields ( refinement_fields, "mesh.geo" )

  write ( 25, '(/1x,a)' ) 'Include "particle_axi.igo";'
  write ( 25, '(/1x,a)' ) 'Include "refinement.igo";'

  close ( 25 )

  call execute_command_line ('gmsh -2 -order 2 -o mesh.msh mesh.geo > gmsh.out')

  call read_mesh_gmsh ( mesh, filename='mesh.msh', ndim=2, &
    physgeom=.true., sortphys=.true. )

  !call add_to_mesh ( mesh, matchingcurve=[3,1], replace=3, &
  !  displacement=[-(lenc1+lenc2),0._dp] )

  call fill_mesh_parts ( mesh )

  call write_mesh_vtk ( mesh, 'mesh.vtk' )

  !call write_geometries_vtk ( mesh )
  
  call write_mesh ( mesh, filename=meshfile )

  print *, "Mesh generation completed."
  print *, "Number of elements: ", mesh%nelem
  print *, "Number of nodes: ", mesh%nnodes
contains

! subroutine to give coordinates on an ellipse

  subroutine coors_ellipse ( coor )

    real(dp), dimension(:,:), intent(inout) :: coor

    integer :: np, i
    real(dp) :: p(size(coor,1))

    np = size(coor,1)
    p = (/(pi*(i-1)/np,i=1,np)/)

    coor(:,1) = radz*cos(p)
    coor(:,2) = radr*sin(p)
    coor(:,3) = 0

  end subroutine coors_ellipse

end program generate_mesh

