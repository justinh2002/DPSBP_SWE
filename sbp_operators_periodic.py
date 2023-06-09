
'''''SBP Operators with periodic BC'''
'''Author: Justin Kin Jun Hew (2023)'''

import numpy
def dxd_m_SBP_periodic(ux, u, nx, dx, order):
    # summation-by-parts finite difference operators for first derivatives du/dx
    # summation-by-parts finite difference operators for first derivatives du/dx
    
    m = nx-1
    
    # second order accurate case
    if order==2:
        # calculate partial derivatives on the boundaries:[0, m] using periodicity
        ux[0, :] = (u[1, :] -  u[m, :])/dx
        ux[m, :] = (u[0, :] -  u[m-1, :])/dx
        
        #calculate partial derivatives in the interior:(1:nx-1) using periodicity
        for j in range(1, m):
            ux[j, :] = (u[(j+1)%nx, :] -  u[(j-1)%nx, :])/(2.0*dx)

        
                   
    # fourth order accurate case        
    if order==4:
                ################################################# 
            
        for i in range(m):
            ux[i,:] = 0.083333333333333*u[(i-2)%m,:] - 0.666666666666667*u[(i-1)%m,:] + 0.666666666666667*u[(i+1)%m,:] - 0.083333333333333*u[(i+2)%m,:]
        ux[:,:] = ux[:,:]/dx


    # sixth order accurate case        
    ################################################# 
       
    if order==6:

        
        for i in range(m):
            ux[i,:] = -0.016666666666667*u[(i-3)%m,:] + 0.15*u[(i-2)%m,:] - 0.75*u[(i-1)%m,:] + 0.75*u[(i+1)%m,:] - 0.15*u[(i+2)%m,:] + 0.016666666666667*u[(i+3)%m,:]
    
        ux[:,:] = ux[:,:]/dx