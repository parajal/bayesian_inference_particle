module poisson_functions_m

  use math_defs_m

  implicit none

contains
 
  function func ( nr, x )
    integer, intent(in) :: nr
    real(dp), intent(in), dimension(:) :: x
    real(dp) :: func
    real(dp) :: rad, R = 1.0_dp
    !rad = sqrt(x(1)**2 + x(2)**2 + x(3)**2)
    rad = sqrt((x(1)-20.0_dp)**2 + x(2)**2)

    select case(nr)
      case(1)
        func = 0
      case(2)
        func = x(2)
      case(3)
        func = 1 + cos(pi*x(1))*cos(pi*x(2))
      case(4)
        func = 2*pi**2*cos(pi*x(1))*cos(pi*x(2))
      case(5)
        func = -pi*sin(pi*x(1))*cos(pi*x(2))
      case(6)
        func = -pi*cos(pi*x(1))*sin(pi*x(2))
      case(7)
        func = -100.0_dp*log(rad)/(2.0_dp*pi)
        !  print *, rad
      case(8)
        func = 2*pi**2*cos(pi*x(1))*cos(pi*x(2)) &
                    -6*(x(1)*x(2)**3+x(1)**3*x(2)) 
      case(9)
        func = cos(pi*x(1))*cos(pi*x(2))*cos(pi*x(3)) + (x(1)*x(2)*x(3))**3
      case(10)
        func = 3*pi**2*cos(pi*x(1))*cos(pi*x(2))*cos(pi*x(3)) &
                    -6*(x(1)*x(2)**3*x(3)**3+x(1)**3*x(2)*x(3)**3 + &
                        x(1)**3*x(2)**3*x(3) )
      case(11)
        func = cos(2*pi*x(1)+1._dp)*cos(pi*x(2))*cos(pi*x(3)) + (x(2)*x(3))**3
      case(12)
        func = 6*pi**2*cos(2*pi*x(1)+1._dp)*cos(pi*x(2))*cos(pi*x(3)) &
                    -6*( x(2)*x(3)**3 + x(2)**3*x(3) )
      case(13)
        func = - ( -pi*sin(pi*x(1))*cos(pi*x(2)) + 3*x(1)**2*x(2)**3  &
                    + cos(pi*x(1))*cos(pi*x(2)) + (x(1)*x(2))**3 )
      case(14)
        func = 10 - 8*2*log(3/x(2))  ! u1=10, R0=2, R1=3, h0=8
      case(15)
        func = 3.0 * cos(rad) - 2.0 * cos(R) !u1
      case(16)
        func = cos(rad)! u2  
      case(17)
        func = 2.0_dp * cos(rad) - cos(R) !f
      case(18)
        func = 3 * (cos(rad) + sin(rad)/rad)  !f2
      case(19) 
        func = pi * cos(4*pi*x(1))*cos(4*pi*x(2)) !exact pressure
      case(20)
         func = cos(x(1))*cos(x(2)) !exact phi for Poisson problem
      case(21)
         func = cos(x(1))*cos(x(2))*cos(x(3)) !exact phi for Poisson problem
      case default
        write(*,'(/a,i0/)') 'Error func: wrong function number: ', nr
        stop
    end select

  end function func 

  function vfunc ( n, nr, x )

    integer, intent(in) :: n, nr
    real(dp), intent(in), dimension(:) :: x
    real(dp), dimension(n) :: vfunc
                                                                                
    select case(nr)
      case(1)
        vfunc = - (/ -pi*sin(pi*x(1))*cos(pi*x(2)) + 3*x(1)**2*x(2)**3,  &
                     -pi*cos(pi*x(1))*sin(pi*x(2)) + 3*x(1)**3*x(2)**2 /)
      case(2)
        vfunc = - (/ -pi*sin(pi*x(1))*cos(pi*x(2)) ,  &
                     -pi*cos(pi*x(1))*sin(pi*x(2)) /)
      case(3)
        vfunc = - (/ -pi*sin(pi*x(1))*cos(pi*x(2))*cos(pi*x(3)) &
                     + 3*x(1)**2*x(2)**3*x(3)**3,  &
                     -pi*cos(pi*x(1))*sin(pi*x(2))*cos(pi*x(3)) &
                     + 3*x(1)**3*x(2)**2*x(3)**3, &
                     -pi*cos(pi*x(1))*cos(pi*x(2))*sin(pi*x(3)) &
                     + 3*x(1)**3*x(2)**3*x(3)**2 /)
      case(4)
        vfunc = (/ -pi*sin(pi*x(1))*cos(pi*x(2))*cos(pi*x(3)) &
                   + 3*x(1)**2*x(2)**3*x(3)**3,  &
                   -pi*cos(pi*x(1))*sin(pi*x(2))*cos(pi*x(3)) &
                   + 3*x(1)**3*x(2)**2*x(3)**3, &
                   -pi*cos(pi*x(1))*cos(pi*x(2))*sin(pi*x(3)) &
                   + 3*x(1)**3*x(2)**3*x(3)**2 /)
      case(5)
        vfunc = - (/ -pi*sin(pi*x(1))*cos(pi*x(2))*cos(pi*x(3)), &
                     -pi*cos(pi*x(1))*sin(pi*x(2))*cos(pi*x(3)), &
                     -pi*cos(pi*x(1))*cos(pi*x(2))*sin(pi*x(3)) /)
      case(6)
        vfunc = (/ -pi*sin(pi*x(1))*cos(pi*x(2))*cos(pi*x(3)), &
                   -pi*cos(pi*x(1))*sin(pi*x(2))*cos(pi*x(3)), &
                   -pi*cos(pi*x(1))*cos(pi*x(2))*sin(pi*x(3)) /)
      !case(7): rhs for stokes problem             
      case(7)
        vfunc = (/28*pi**2*sin(4*pi*x(1))*cos(4*pi*x(2)), -36*pi**2*cos(4*pi*x(1))*sin(4*pi*x(2))/) 
     ! Case(8): velocity for Stokes (only) problem
      case(8)
        vfunc = (/ sin(4*pi*x(1))*cos(4*pi*x(2)), -cos(4*pi*x(1))*sin(4*pi*x(2)) /) ! exact velocity for stokes problem  
      case(9)
        vfunc = (/ 28*pi**2*sin(4*pi*x(1))*cos(4*pi*x(2)) - 2*sin(x(1))*cos(x(1))*cos(x(2))**2, &
        -36*pi**2*cos(4*pi*x(1))*sin(4*pi*x(2)) - 2*sin(x(2))*cos(x(1))**2*cos(x(2)) /)         
      case(10)
        vfunc = (/ 28*pi**2*sin(4*pi*x(1))*cos(4*pi*x(2)) - 6*sin(x(1))*cos(x(1))*cos(x(2))**2, &
        -36*pi**2*cos(4*pi*x(1))*sin(4*pi*x(2)) - 6*sin(x(2))*cos(x(1))**2*cos(x(2)) /)  
      case(11)
        vfunc = (/ 28*pi**2*sin(4*pi*x(1))*cos(4*pi*x(2)) - (4*x(1)-2), &
        -36*pi**2*cos(4*pi*x(1))*sin(4*pi*x(2)) - (4*x(2)-2) /)         
      case(12)
        vfunc = (/ 28*pi**2*sin(4*pi*x(1))*cos(4*pi*x(2)) - 3*(4*x(1)-2), &
        -36*pi**2*cos(4*pi*x(1))*sin(4*pi*x(2)) - 3*(4*x(2)-2) /) 
       
      case default  
        write(*,'(/a,i0/)') 'Error vfunc: wrong function number: ', nr
        stop
    end select
                                                                                
  end function vfunc
                                                                                
end module poisson_functions_m
 


