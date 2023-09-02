import numpy as np
from scipy.special import hyp2f1


class SWE_SBP:
    
    def __init__(self):

        self.ndim     = 1 # the dimensionality (1,2)
        self.ncells   = np.array([500]) # number of cells (2D)
        self.scheme   = "sbp" # sbp
        self.boundary = "non-periodic" # boundary condition
        self.cfl      = 0.8 # CFL safety factor
        self.nend     = 100 # number of timesteps
        self.tmax     = 1e10 # end time
        self.dx       = 1.0 # cell size x
        self.dy       = 1.0 # cell size y
        
    def mms(self,U,V, P, U_t, V_t, P_t, U_x,V_x, P_x,x, y, t, Ubar,Vbar,type_0,nx,ny,dy,dx,order):
        from sbp_operators_periodic import dxd_m_SBP_periodic,dyd_m_SBP_periodic

        import numpy as np


        if type_0 in ('Gaussian'):

            delta = (y[-1,0]-y[0,0])

            rho = 1#2.6702     # density [g/cm^3]
            K =  2.2 #rho*cs**2 
            #cs = np.sqrt(K/rho)# shear modulus [GPa]
            #Zs = rho * cs
            
            g = 9.81
            H = 10
            
            cs =np.sqrt(g*H) #Ubar np.sqrt(g*H)

            y0 = 0.5*(y[0, -1]-y[0, 0])
            x0 = 0.5*(x[-1, 0]-x[0, 0])
            

            V[:,:]=  (np.exp(-((y -y0)**2 +(x - x0 )**2-  cs*t)**2)/1)
            U[:,:] =  (np.exp(-((y -y0)**2 +(x - x0 )**2-  cs*t)**2)/1)
            #*p.cos(2*np.pi*y) #1/np.sqrt(2.0*np.pi*delta**2)*0.5*(np.exp(-(y+cs*(t)-x0)**2/(2.0*delta**2))\
            U_t[:,:]   =2* cs* V*(x -x0 -  cs*t)/1                          #   + np.exp(-(y-cs*(t)-x0)**2/(2.0*delta**2)))

            P[:,:] =   V + 10 #(np.exp(-(y - cs*(t))**2/(0.3 * x0)))#*np.cos(2*np.pi*y) #0 #1/np.sqrt(2.0*np.pi*delta**2)*0.5*Zs*(np.exp(-(y+cs*(t)-x0)**2/(2.0*delta**2))\
                                                 #   - np.exp(-(y-cs*(t)-x0)**2/(2.0*delta**2)))

            V_t[:,:] =   2* cs* V*(y -x0 -  cs*t)/1# 10/x0* (-cs/0.3) * 2 * t  * (np.exp(-( cs*(t))**2/(0.3)))*np.cos(2*np.pi*y)
            P_t[:,:] =  V_t
            
            U_x =-U*2*(x -x0 -  cs*t)/1 # 5/x0 *(-cs/0.3) * 2 * t  * (np.exp(-(cs*(t))**2/(0.3)))*np.cos(2*np.pi*y)
            V_x[:,:] =  -V*2*(y -x0 -  cs*t)/1 #10 * (-2*np.pi) * (np.exp(-( cs*(t))**2/(0.3)))*np.sin(2*np.pi*y)
            P_x[:,:] =  V_x#
            
            
        elif type_0 in ('double_Gaussian'):
            
            f = 5 
            g = 5
            H = 8
            # xc1 = 1.5 * np.pi
            # xc2 = 2 * np.pi
            # yc1 = 1 * np.pi
            # yc2 = 2 * np.pi            
            xc1 = 1 *  np.pi
            xc2 = 1 * np.pi
            yc1 = 2.6 * np.pi/3
            yc2 = 3.5 * np.pi/3
            g1 = np.exp(-2.5 * ((x - xc1) ** 2 + (y - yc1) ** 2))
            g2 = np.exp(-2.5 * ((x - xc2) ** 2 + (y - yc2) ** 2))
            stream = g1 + g2
            P[:,:] = H + stream * f / g
    # u = -np.einsum('fgcd,abcd->fgab', stream, solver.grad[..., 0])
    # v = np.einsum('fgcd,abcd->fgab', stream, solver.grad[..., 1])
            U[:,:] = 5 * (y - yc1) * g1 + 5 * (y - yc2) * g2
            V[:,:] = -5 * (x - xc1) * g1 - 5 * (x - xc2) * g2
            #P[:,:] = -g/f * np.exp(-2.5*((x-np.pi) **2 + (y-2*np.pi/3) ** 2) ) + np.exp(-2.5*((x-np.pi)**2 + (y-4*np.pi/3) ** 2))#psi            
           # psi[:,:] = np.exp(-2.5*((x-np.pi) **2 + (y-2*np.pi/3) ** 2) ) + np.exp(-2.5*((x-np.pi)**2 - (y-4*np.pi/3) ** 2))  #10*np.pi#         
            #U[:,:] = -(5*np.exp(-2.5*((x-np.pi)**2 - (y - 4*np.pi/3) **2)) * (y - 4 *np.pi/3)  - 5 * np.exp(-2.5*((x - np.pi)**2 + (y-2*np.pi/3)**2) )* (y - 2 * np.pi /3))# psi_y
            #V[:,:] =  -5*np.exp(-2.5*((x-np.pi)**2 - (y - 4*np.pi/3) **2)) * (x - np.pi)  - 5 * np.exp(-2.5*((x - np.pi)**2 + (y-2*np.pi/3)**2)) * (x - np.pi)#psi_x 

                        
        elif type_0 in ('KH_Peixoto'):
                    g = 9.81/1000    
                    H = 10         
                    mask_hl= np.logical_and(0< y, y < 5)
                    mask_hr = np.logical_and(5< y, y < 10)
                    psi = np.zeros((nx,ny))
                    psi_x = np.zeros((nx,ny))
                    psi_y = np.zeros((nx,ny))            
                    f = 2 * 7.292e-5
                    w0 = 1
                    sigma = 1
                    k = 1000
                    n = 81
                    def sech(x):
                        return 1/np.cosh(x)            
                    for i in np.arange(nx):
                        for j in np.arange(ny):
                            # if y[i,j] >= 2e7:
                            U[i,j] = 50 * np.sin(2*np.pi * y[i,j]/4e5) ** 81
                            P[i,j] =  H -f/g * 50 * -np.cos(2*np.pi*y[i,j]/4e5) * hyp2f1(1/2, (1 - n)/2, 3/2, np.cos(y[i,j]*2*np.pi/4e5)**2) #* np.sum(U[i,j]*y[i,j])# 0.1* f/(1e-6*g) * 1e-6 * sech(1e-6* (y[i,j] - 3e7))  ** 2
                            P[i,j] += 0.01 * H * np.exp(-k * ((x[i,j] - 0.85*4e5)** 2/4e5**2 + (y[i,j] - 0.75* 4e5) ** 2/4e5**2) + np.exp(-k * ((x[i,j] - 0.15*4e5)** 2/4e5**2 + (y[i,j] - 0.25* 4e5) ** 2)/4e5**2)) #H+ 0.1* f/(1e-6*g) * np.tanh(1e-6 * (y[i,j]-3e7))
                            V[i,j] = 0
                                #P[i,j] += 20 * H * np.exp(-((x[i,j]** 2 + (y[i,j] - 0.95 * 3e7) ** 2) / (5_000_000 ** 2) ))
                            # elif y[i,j] < 2e7:
                            #     P[i,j] = H  -0.1 * f/(1e-6*g) * np.tanh(1e-6 * (y[i,j]-1e7))
                            #     U[i,j] = -  0.1 * f/(1e-6*g) * 1e-6* sech(1e-6 * (y[i,j]- 1e7)) **2 
                            #     V[i,j] = 0.
                            #     #P[i,j] += w
            
        elif type_0 in ('KH'):
                    g = 9.81    
                    H = 1         
                    mask_hl= np.logical_and(0< y, y < 5)
                    mask_hr = np.logical_and(5< y, y < 10)
                    psi = np.zeros((nx,ny))
                    psi_x = np.zeros((nx,ny))
                    psi_y = np.zeros((nx,ny))            
                    f = 2 * 7.292e-5
                    w0 = 1
                    sigma = 1
                    
                    def sech(x):
                        return 1/np.cosh(x)            
                    for i in np.arange(nx):
                        for j in np.arange(ny):
                            if y[i,j] >= 2e7:
                                U[i,j] = 0.1* f/(1e-6*g) * 1e-6 * sech(1e-6* (y[i,j] - 3e7))  ** 2
                                P[i,j] = H+ 0.1* f/(1e-6*g) * np.tanh(1e-6 * (y[i,j]-3e7))
                                V[i,j] = 0.
                                P[i,j] += 20 * H * np.exp(-((x[i,j]** 2 + (y[i,j] - 0.95 * 3e7) ** 2) / (5_000_000 ** 2) ))
                            elif y[i,j] < 2e7:
                                P[i,j] = H  -0.1 * f/(1e-6*g) * np.tanh(1e-6 * (y[i,j]-1e7))
                                U[i,j] = -  0.1 * f/(1e-6*g) * 1e-6* sech(1e-6 * (y[i,j]- 1e7)) **2 
                                V[i,j] = 0.
                                #P[i,j] += w0 * np.sin(4*np.pi*x[i,j]) * np.exp(-()) ** 2) 
                                
        elif type_0 in ('KH_Euler'):
            
                    g = 9.81    
                    H = 1         
                    mask_hl= np.logical_and(0< y, y < 5)
                    mask_hr = np.logical_and(5< y, y < 10)
                    psi = np.zeros((nx,ny))
                    psi_x = np.zeros((nx,ny))
                    psi_y = np.zeros((nx,ny))            
                    f = 2 * 7.292e-5
                    sigma = 0.05/np.sqrt(2.)
                    w0 = 0.1

	# rho = 1. + (np.abs(Y-0.5) < 0.25)
	# vx = -0.5 + (np.abs(Y-0.5)<0.25)
	# vy = w0*np.sin(4*np.pi*X) * ( np.exp(-(Y-0.25)**2/(2 * sigma**2)) + np.exp(-(Y-0.75)**2/(2*sigma**2)) )
	# P = 2.5 * np.ones(X.shape)
                    
                    def sech(x):
                        return 1/np.cosh(x)            
                    for i in np.arange(nx):
                        for j in np.arange(ny):
                            if y[i,j] >= 2e7:
                                U[i,j] = 1 + np.abs(y[i,j] -1e7)#H = np.abs(y[i,j] - 3e7) # 100 *f/(1e-6*g) * 1e-6 * sech(1e-6* (y[i,j] - 3e7))  ** 2
                                P[i,j] = H +  np.abs(y[i,j] - 3e7)#H + np.abs(y[i,j] - 3e7)#  H+ 100 *f/(1e-6*g) * np.tanh(1e-6 * (y[i,j]-3e7))
                                V[i,j] = 0
                            elif y[i,j] < 2e7:
                                P[i,j] = H + np.abs(y[i,j] -1e7)#H  - 100 *f/(1e-6*g) * np.tanh(1e-6 * (y[i,j]-1e7))
                                U[i,j] = 1 + np.abs(y[i,j] -1e7) #- 100 *f/(1e-6*g) * 1e-6* sech(1e-6 * (y[i,j]- 1e7)) **2 
                                V[i,j] = 0 
                            P[i,j] += w0 * np.sin(4*np.pi*x[i,j]) * np.exp(-(y[i,j]/(2*sigma)) ** 2) 
                    # psi = -g/f * P
                    # dyd_m_SBP_periodic(psi_y, psi,ny,dy,order)
                    # dxd_m_SBP_periodic(psi_x,psi, nx,dx,order)
                    # U = -psi_y
                    # V = psi_x   #(edited)

                
            
            # g = 9.81
            # mask_hl= np.logical_and(0< y, y < 5)
            # mask_hr = np.logical_and(5< y, y < 10)
            # psi = np.zeros((nx,ny))
            # psi_x = np.zeros((nx,ny))
            # psi_y = np.zeros((nx,ny))
            
            
            

            # #gradient_h = 
            # #b[mask_b] = 0.2 - 0.05*(y[mask_b] - 10 )**2
            # f = 2 * 7.292e-5
            
            # for i in np.arange(nx):
            #     for j in np.arange(ny):
            #         if y[i,j] >= 2e7:
            #             P[i,j] = 10**4 + 50 *f/(1e-6*g) * 1/np.cosh(1e-6 * (y[i,j]-3e7))
            #         elif y[i,j] < 2e7:
            #             P[i,j] = 10**4  -50 *f/(1e-6*g) * 1/np.cosh(1e-6 * (y[i,j]- 1e7))  
            # psi = -g/f * P         
            # dyd_m_SBP_periodic(psi_y, psi,ny,dy,order)
            # dxd_m_SBP_periodic(psi_x,psi, nx,dx,order)
            # U = psi_y 
            # V = -psi_x
                
    def acoustic_rate(self,hu,hv, hp,u, v, h, rho, K, nx, ny,dx,dy, order, t, x,y, type_0, Ubar,Vbar,H, g,flux_type,vorticity):
    # we compute rates that will be used for Runge-Kutta time-stepping
    # # 
    #         from dpsbp_operators_correct import dxd_m_DP, dxd_p_DP
    #         from DRP_FD import dxd_m_DRP, dxd_p_DRP
            from DRP_FD_periodic import dxd_m_DRP_periodic, dxd_p_DRP_periodic
            from dpsbp_operators_periodic import dxd_m_DP_periodic, dxd_p_DP_periodic, dyd_m_DP_periodic, dyd_p_DP_periodic
            from sbp_operators_periodic import dxd_m_SBP_periodic,dyd_m_SBP_periodic
            from DRP_FD_periodic import dyd_m_DRP_periodic, dyd_p_DRP_periodic,  dxd_m_DRP_periodic, dxd_p_DRP_periodic       
            # from sbp_operators import dxd_m_SBP

            import numpy as np
        
            #delta = (y[-1,0]-y[0,0])
        # L = 10
            #ubar = 0
            #H = 1
            #g = 9.8
            #ubar= 0.5*np.sqrt(g*H)


            U = np.zeros((nx, ny))
            V = np.zeros((nx, ny))
            P = np.zeros((nx, ny))
            
            Ut = np.zeros((nx, ny))
            Vt =  np.zeros((nx, ny))
            Pt = np.zeros((nx, ny))
            
            Ux = np.zeros((nx, ny))
            Vx = np.zeros((nx, ny))
            Px = np.zeros((nx, ny))
            

            uyp = np.zeros((nx,ny))
            vxp = np.zeros((nx,ny))
            uym = np.zeros((nx,ny))
            vxm = np.zeros((nx,ny))            
            
            #p = np.zeros((nx,ny))
           # u_x = np.zeros((nx,ny))
            
            
            
                    
            bx = np.zeros((nx, ny))
            
            b = np.zeros((nx, ny))
            # if topography == 'non_smooth':
            #     b = 0 * y
            #     bx = 0 * y
            #     mask_b = np.logical_and(8 < y, y < 12)
            #     b[mask_b] = 0.2 - 0.05*(y[mask_b] - 10 )**2 
                
            #     bx[mask_b] = - 0.10 * (y[mask_b] - 10 )
            #     #mask_b = np.logical_and(8 < y, y < 12)
            
            #     #mask_a = np.logical_and(12 <= y, y <= 8)

            # elif topography == 'smooth':
            #     b = 0 * y
            #     bx = 0 * y
            #     b = 0.1 * np.exp( -(y-10)**2/0.3) 
            #     bx = -2*b* ( y - 10) / 0.3
                
            # else:
            #     bx = 0.0
            #     b = 0.
            
            self.mms(U, V,P, Ut,Vt,Pt, Ux,Vx, Px,x, y, t,Ubar,Vbar, type_0,nx,ny,dx,dy,order)

            if flux_type == 'linear':
                flux_hx = Ubar*(h)+ H*u
                flux_hy = Vbar*(h) +H*v  #h
                flux_u =  (Ubar*u+Vbar*v) + g*(h) ##Ubar*v + g*h

            #             flux_1 = Ubar*v + g*h
            # flux_2 = Ubar*h + H*v
            
            if flux_type == 'nonlinear':
                #b = 0
                flux_hx = u*(h)
                flux_hy = v*(h)  #h
                flux_u =  (u**2+v**2)/2 + g*(h) ##Ubar*v + g*h
                
                #flux_v = (u**2 + v**2)/2 + g*(h+b)
            # initialize arrays for computing derivatives

                
            flux_hx_x = np.zeros((nx,ny))
            
            
            flux_hy_y = np.zeros((nx,ny))
            flux_u_x = np.zeros((nx,ny))
            flux_u_y =np.zeros((nx,ny))
            flux_hx_x_p = np.zeros((nx,ny))
            
            
            flux_hy_y_p = np.zeros((nx,ny))
            flux_u_x_p = np.zeros((nx,ny))
            flux_u_y_p =np.zeros((nx,ny))            
            flux_hx_x_m = np.zeros((nx,ny))
            
            
            flux_hy_y_m = np.zeros((nx,ny))
            flux_u_x_m = np.zeros((nx,ny))
            flux_u_y_m =np.zeros((nx,ny))
            
            u_x = np.zeros((nx,ny))
            v_x = np.zeros((nx,ny))
            h_x = np.zeros((nx,ny))
            
            u_xx = np.zeros((nx,ny))
            v_xx = np.zeros((nx,ny))
            h_xx = np.zeros((nx,ny))
            
            u_xxx = np.zeros((nx,ny))
            v_xxx = np.zeros((nx,ny))
            h_xxx = np.zeros((nx,ny))
            
            u_xxxx = np.zeros((nx,ny))
            v_xxxx = np.zeros((nx,ny))
            h_xxxx = np.zeros((nx,ny))
            
            u_y = np.zeros((nx,ny))
            v_y = np.zeros((nx,ny))
            h_y = np.zeros((nx,ny))
            
            u_yy = np.zeros((nx,ny))
            v_yy = np.zeros((nx,ny))
            h_yy = np.zeros((nx,ny))
            
            u_yyy = np.zeros((nx,ny))
            v_yyy = np.zeros((nx,ny))
            h_yyy = np.zeros((nx,ny))
            
            u_yyyy = np.zeros((nx,ny))
            v_yyyy = np.zeros((nx,ny))
            h_yyyy = np.zeros((nx,ny))
            # u_xxx = np.zeros((nx,ny))#np.zeros((nx,1))
            # h_xxx =np.zeros((nx,ny))# np.zeros((nx,1))
            
            # u_xxxx = np.zeros((nx,ny))#np.zeros((nx,1))
            # h_xxxx = np.zeros((nx,ny))#np.zeros((nx,1))
            
            # u_xxxxx = np.zeros((nx,1))
            # h_xxxxx = np.zeros((nx,1))
            
            # u_xxxxxx = np.zeros((nx,1))
            # h_xxxxxx = np.zeros((nx,1))
                # if fd_type == 'DP':
            #     dxd_m = dxd_m_DP
            #     dxd_p = dxd_p_DP
                
            # if fd_type == 'DRP':
            #     dxd_m = dxd_m_DRP
            #     dxd_p = dxd_p_DRP

            # if fd_type == 'SBP':
            #     dxd_m = dxd_m_SBP
            #     dxd_p = dxd_m
            
            # if fd_type == 'DRP_periodic':
            #     dxd_m = dxd_m_DRP_periodic
            #     dxd_p = dxd_p_DRP_periodic
                
            # if fd_type == 'DP_periodic':
            #     dxd_m = dxd_m_DP_periodic
            #     dxd_p = dxd_p_DP_periodic
            
           # if fd_type == 'SBP_periodic':
            dxd_m =dxd_m_DP_periodic#dxd_m_DP_periodic#dxd_m_DRP_periodic##dxd_m_SBP_periodic#dxd_m_DRP_periodic#dxd_m_SBP_periodic#dxd_m_DP_periodic#dxd_m_SBP_periodic# dxd_m_DP_periodic#dxd_m_SBP_periodic#dxd_m_DP_periodic#dxd_m_SBP_periodic#dxd_m_SBP_periodic
            dxd_p =dxd_p_DP_periodic#dxd_p_DP_periodic#dxd_p_DRP_periodic#dxd_m_SBP_periodic#dxd_p_DRP_periodic#dxd_m_SBP_periodic#dxd_p_DP_periodic#dxd_m_SBP_periodic# dxd_p_DP_periodic#dxd_m_SBP_periodic#dxd_p_DP_periodic# dxd_p_DP_periodic#dxd_m_SBP_periodic #dxd_m
            dyd_m =dyd_m_DP_periodic#dyd_m_DP_periodic#dyd_m_DRP_periodic#dyd_m_SBP_periodic#dyd_m_DRP_periodic#dyd_m_SBP_periodic#dyd_m_DP_periodic#dyd_m_SBP_periodic#dyd_m_DP_periodic#dyd_m_SBP_periodic#dyd_m_DP_periodic#dyd_m_SBP_periodic#dyd_m_DP_periodic
            dyd_p =dyd_p_DP_periodic#dyd_p_DRP_periodic#dyd_m_SBP_periodic#dyd_p_DRP_periodic#dyd_m_SBP_periodic#dyd_p_DP_periodic#dyd_m_SBP_periodic# dyd_p_DP_periodic#dyd_m_SBP_periodic#dyd_p_DP_periodic#dyd_m_SBP_periodic
                        
            alpha = 0.#0.5 *np.max(np.max( np.sqrt(u**2+v**2)) + np.sqrt(g*h)  )
            
            # dxd_p(flux_hx_x_p, flux_hx - alpha * h , nx,dx, order)
            # dyd_p(flux_hy_y_p, flux_hy - alpha * h , ny,dy, order)
            # dxd_p(flux_u_x_p, flux_u - alpha *u , nx, dx, order)
            # dyd_p(flux_u_y_p, flux_u - alpha *v ,  ny,dy, order)            
            
            # dxd_m(flux_u_x_m, flux_u + alpha * u, nx, dx, order)
            # dyd_m(flux_u_y_m, flux_u + alpha * v,  ny,dy, order)
            # dxd_m(flux_hx_x_m, flux_hx + alpha * h, nx,dx, order)
            # dyd_m(flux_hy_y_m, flux_hy + alpha * h, ny,dy, order)
            
            dxd_p(flux_hx_x_p, flux_hx - alpha * flux_u , nx,dx, order)
            dyd_p(flux_hy_y_p, flux_hy - alpha * flux_u , ny,dy, order)
            dxd_p(flux_u_x_p, flux_u - alpha *flux_hx , nx, dx, order)
            dyd_p(flux_u_y_p, flux_u - alpha *flux_hy ,  ny,dy, order)            
            
            dxd_m(flux_u_x_m, flux_u + alpha * flux_hx, nx, dx, order)
            dyd_m(flux_u_y_m, flux_u + alpha * flux_hy,  ny,dy, order)
            dxd_m(flux_hx_x_m, flux_hx + alpha *flux_u , nx,dx, order)
            dyd_m(flux_hy_y_m, flux_hy + alpha * flux_u, ny,dy, order)
                        
            flux_hx_x = 0.5 * (2*flux_hx_x_p + 0*flux_hx_x_m)
            flux_hy_y = 0.5 * (2*flux_hy_y_p + 0*flux_hy_y_m)
            flux_u_x = 0.5 * (0*flux_u_x_p + 2*flux_u_x_m)
            flux_u_y = 0.5 * (0*flux_u_y_p + 2*flux_u_y_m)
            #flux_u_y = flux_u_y -           
                          
            # dxd_p(u_x, v, nx, dx, order)
            # dxd_m(h_x, h, nx, dx, order)
            # dxd_p(u_x, v, nx, dx, order)    
            
            # dxd_m(u_xx, u_x, nx, dx, order)
            # dxd_p(h_xx, h_x, nx, dx, order)
            
            # dxd_p(u_xxx, u_xx, nx, dx, order)
            # dxd_m(h_xxx, h_xx, nx, dx, order)
            
            # dxd_m(u_xxxx, u_xxx, nx, dx, order)
            # dxd_p(h_xxxx, h_xxx, nx, dx, order)   
              #10percent of the domain length 
    #   # Ubar*h + H*v 
    #     #forcing = 1
    #     if flux_type == 'linear':
    #         forcing_term_1 = Ubar * Ux + g * Px
    #         forcing_term_2 = Ubar * Px + H * Ux

    #     if flux_type == 'nonlinear':
    #         forcing_term_1 = U*Ux + g*Px
    #         forcing_term_2 =  Px * U + Ux* P

            dxd_p(u_x, u, nx, dx, order)
            dxd_p(v_x, v, nx, dx, order)
            dxd_m(h_x, h, nx, dx, order)    
            
            dxd_m(u_xx, u_x, nx, dx, order)
            dxd_m(v_xx, v_x, nx, dx,order)
            dxd_p(h_xx, h_x, nx, dx, order)
            
            dxd_p(u_xxx, u_xx, nx, dx, order)
            dxd_p(v_xxx, v_xx, nx, dx,order)
            dxd_m(h_xxx, h_xx, nx, dx, order)
            
            dxd_m(u_xxxx, u_xxx, nx, dx, order)
            dxd_m(v_xxxx, v_xxx, nx, dx,order)
            dxd_p(h_xxxx, h_xxx, nx, dx, order)
            
            
            dyd_p(u_y, u, ny, dy, order)
            dyd_p(v_y,v,ny,dy,order)
            dyd_m(h_y, h, ny, dy, order)    
            
            dyd_m(u_yy, u_y, ny, dy, order)
            dyd_m(v_yy, v_y, ny, dy, order)
            dyd_p(h_yy, h_y, ny, dy, order)
            
            dyd_p(u_yyy, u_yy, ny, dy, order)
            dyd_p(v_yyy, v_yy, ny, dy, order)
            dyd_m(h_yyy, h_yy, ny, dy, order)
            
            dyd_m(u_yyyy, u_yyy, ny, dy, order)
            dyd_m(v_yyyy, v_yyy, ny, dy, order)
            dyd_p(h_yyyy, h_yyy, ny, dy, order)
            
            
            # dxd_p(u_xxx, u_xx, nx, dx, order)
            # dxd_m(h_xxx, h_xx, nx, dx, order)
            
            # dxd_m(u_xxxx, u_xxx, nx, dx, order)
            # dxd_p(h_xxxx, h_xxx, nx, dx, order)   
            # #10percent of the domain length 
    #   # Ubar*h + H*v 
    #     #forcing = 1
    #     if flux_type == 'linear':
    #         forcing_term_1 = Ubar * Ux + g * Px
    #         forcing_term_2 = Ubar * Px + H * Ux

    #     if flux_type == 'nonlinear':
    #         forcing_term_1 = U*Ux + g*Px
    #         forcing_term_2 =  Px * U + Ux* P
        
        
        
        
    #   # Ubar*h + H*v 
    #     #forcing = 1
    #             if flux_type == 'linear':
    #                 forcing_term_1 = Ubar * Ux + g * Px
    #                 forcing_term_2 = Ubar * Px + H * Ux
    #                 w11, w12, w21, w22 = g , -Ubar, -Ubar, H
                    
    #                 Art = 1/(g*H - Ubar**2) 



            w11, w12, w13, w21, w22, w23, w31, w32, w33 = h**2/4, -h*u/4, -h*v/4, -h*u/4, g*h/2 - v**2/4, u*v/4, -h*v/4, u*v/4, g*h/2 - u**2/4 # 1 0 
            
            Art = 1/(g*h**2/4 - h* u **2/8 - h*v**2/8)
    #(g h^2)/4 - (h u^2)/8 - (h v^2)/8



            hyper_factor = 0.5#0.1#0.1# 1.0
        
        # hv[:,:] = -flux_1_x + forcing * (Ut + forcing_term_1) + hyper_factor * dx**5 * Art * (w11 * u_xxxxxx + w12 * h_xxxxxx)
        # hp[:,:] = -flux_2_x + forcing * ( Pt +  forcing_term_2 ) + hyper_factor*dx**5*Art*(w21*h_xxxxxx + w22*u_xxxxxx) 
        
        #hv[:,:] = -flux_1_x + forcing*(Ut + forcing_term_1)  - hyper_factor*dx**3*Art*(w11*u_xxxx + w12*h_xxxx) #- g * 0 * bx
        #hp[:,:] = -flux_2_x + forcing*( Pt +  forcing_term_2 ) - hyper_factor*dx**3*Art*(w21*h_xxxx + w22*u_xxxx) 
                                          #e
                                                        
            hu[:,:] = -flux_u_x - hyper_factor * dx**3 * Art * (w21 * (h_xxxx + h_yyyy) + w22 * (u_xxxx + u_yyyy) + w23 * (v_xxxx + v_yyyy))
            hv[:,:] = -flux_u_y -  hyper_factor * dx**3 * Art * (w31 * (h_xxxx + h_yyyy) + w32 * (u_xxxx + u_yyyy) + w33 * (v_xxxx + v_yyyy))
            hp[:,:] = -flux_hx_x - flux_hy_y - hyper_factor * dx**3 * Art * (w11 * (h_xxxx + h_yyyy) + w12 * (u_xxxx + u_yyyy) + w13 * (v_xxxx + v_yyyy))
            
                        
            if vorticity == 'true':
                f =  5 #2 * 7.292e-5
                
                dyd_p(uyp,u,ny,dy,order) 
                dxd_p(vxp,v, nx,dx,order)
                dyd_m(uym,u,ny,dy,order) 
                dxd_m(vxm,v, nx,dx,order)
                
                vx = 0.5* (0*vxp + 2*vxm)
                uy = 0.5 * (0*uyp + 2*uym)
                
                vort = (vx  -  uy  + f) 
                hu +=  vort * v
                hv -=  vort * u
                                        
    