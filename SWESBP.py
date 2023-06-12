'''Shallow water wave solver'''
"""Author: Justin Kin Jun Hew"""
class acoustic_sbp:
    
    
    def __init__(self):
        import numpy as np
        self.ndim     = 1 # the dimensionality (1,2)
        self.ncells   = np.array([500]) # number of cells (2D)
        self.scheme   = "sbp" # sbp
        self.boundary = "non-periodic" # boundary condition
        self.cfl      = 0.8 # CFL safety factor
        self.nend     = 100 # number of timesteps
        self.tmax     = 1e10 # end time
        self.dx       = 1.0 # cell size x
        self.dy       = 1.0 # cell size y
        
        
    def penaltyweight(h11, dx, order):
        if order==2:
            h11[:] = (0.25)*dx
            #h12[:] = (1.25)*dx
        if order== 4:
            h11[:] = (49/144)*dx
        if order ==5:
            h11[:] = (251./720) * dx
        
        if order==6:
            h11[:] = (13613.0/43200.0)*dx
        
    def impose_bc(self,hv, hp, v, p, y, rho, K, nx, dx, order, V, P, r0, r1, tau0_1, tau0_2, tauN_1, tauN_2, H, Ubar, g, type_0, fd_type,flux_type,bc_type_0, bc_type_N,forcing):
        # impose boundary conditions                                                                                                                         
        import numpy as np
        
        #from RK4 import penaltyweight


        # penalty weights                                                                                                                                    
        h11 = np.zeros((1, 1))
        #h12 = np.zeros((1,1))
        #penaltyweight(h11, dx, order)
        
        if fd_type == 'DP':
            if order==2:
                h11[:] = (0.25)*dx
                #h12[:] = (1.25)*dx
            elif order== 4:
                h11[:] = (49./144.)*dx
            
            elif order==5:
                h11[:] = (251./720) * dx
        
            elif order==6:
                h11[:] = (13613.0/43200.0)*dx
                
                
        if fd_type == 'DRP':
            if order == 4:
                h11[:] = 0.315586328095031*dx
            elif order == 5:
                h11[:] =0.318079048635254*dx
            elif order == 6:
                h11[:] =0.294425496425380*dx
            elif order == 7:
                h11[:] =0.294900239556150*dx

        if fd_type == 'SBP':  
            if order==2:
                h11[:] = 0.5*dx
            elif order== 4:
                h11[:] = (17.0/48.0)*dx

            elif order==6:
                h11[:] = (13649.0/43200.0)*dx

        mv = np.zeros((1,1))
        mp = np.zeros((1,1))

        pv = np.zeros((1,1))
        pp = np.zeros((1,1))

        v0 = v[0,:]
        p0 = p[0,:]

        vn = v[nx-1,:]
        pn = p[nx-1,:]

        # boundary forcing                                                                                                                                   
        V0 = V[0,:] 
        P0 = P[0,:]

        Vn = V[nx-1,:]
        Pn = P[nx-1,:]
        #H = 1
        #g = 9.8
        #ubar= 0.5*np.sqrt(g*H)
        U_data, P_data, U_t, V_t, U_x, V_x = 0*P, 0*P, 0*P, 0*P, 0*P,0*P
        self.mms(U_data, P_data, U_t, V_t, U_x, V_x, y, 0, Ubar,type_0 ,nx)
        if flux_type == 'nonlinear':
         
            flux_1_0 = v0**2/2 + g*p0 - (g*P0 + V0**2/2) - (1-forcing)*(U_data[0]**2/2 + g*P_data[0]) #  du/dt
            flux_2_0 =  v0 * p0  - ( V0 * P0) - (1-forcing)*(U_data[0]*P_data[0]) #Ubar*v0 + g*p0 #- (Ubar*V0 + g*P0) #dh/dt
            
            if bc_type_0 == 'mass flux': 
                mv = flux_1_0 * 0
                mp = flux_2_0 #*0
                
            if bc_type_0 == 'energy flux': 
                mv = flux_1_0 #* 0
                mp = flux_2_0 * 0
                
            if bc_type_0 == 'transmissive': 
                alpha_0 = 2/(g*p0 - 0.5*v0**2)
            
                f2_data_0 =  -1/(g - v0*g/np.sqrt(g*p0))*((np.sqrt(g*p0) - v0/2)*flux_1_0)
                mv = flux_1_0 * 0
                mp = flux_2_0-f2_data_0
            
            
           
            flux_1_n =   vn**2/2 + g*pn - (g*Pn + Vn**2/2) - (1-forcing)*(U_data[-1]**2/2 + g*P_data[-1])
            flux_2_n =   vn * pn  - (Vn * Pn) - (1-forcing)*(U_data[-1]*P_data[-1]) #Ubar*vn + g*pn #- (Ubar*Vn + g*Pn)
            
            if bc_type_N == 'mass flux':
                pv = flux_1_n  * 0
                pp = flux_2_n #*0
                
            if bc_type_N == 'energy flux':
                pv = flux_1_n # * 0
                pp = flux_2_n *0
                
            if bc_type_N == 'transmissive':
                alpha_n = 2/(g*pn - 0.5*vn**2)
                wn = U_data[-1]-2*np.sqrt(g*P_data[-1])
                f2_data_n =  1/(g + vn*g/np.sqrt(g*pn))*((np.sqrt(g*pn) + vn/2)*flux_1_n)
                f1_data_n =  1/(np.sqrt(g*pn) + vn/2)*((g + vn*g/np.sqrt(g*pn))*flux_2_n)
            
                pv = (flux_1_n - f1_data_n ) * 0
                pp = (flux_2_n - f2_data_n) #*0

        if flux_type == 'linear':
            flux_1_0 = Ubar*v0 + g * p0 - (Ubar*V0 + g*P0) #  du/dt
            flux_2_0 =  Ubar*p0 + H * v0  - ( Ubar*P0 + H*V0) #Ubar*v0 + g*p0 #- (Ubar*V0 + g*P0) #dh/dt
            
            mv = flux_1_0  * 0
            mp = flux_2_0 #* 0 
            
          # Ubar*h + H*v 
            flux_1_n =   Ubar*vn + g * pn - (Ubar*Vn + g*Pn)
            flux_2_n =   Ubar*pn + H * vn  - ( Ubar*Pn + H*Vn)#Ubar*vn + g*pn #- (Ubar*Vn + g*Pn)
            
            pv = flux_1_n * 0
            pp = flux_2_n #* 0 #* 0 
             


        # compute SAT terms                                                                                                                                  
        #self.bcm(mv, mp, v0, p0, V0, P0, rho, K, r0)
        #self.bcp(pv, pp, vn, pn, Vn, Pn, rho, K, r1)

        # penalize boundaries with the SAT terms                                                                                                             
        hv[0,:] =  hv[0,:] - tau0_1/(h11) * mv #dh_dt #tau21 in paper
        hp[0,:] =  hp[0,:] -  tau0_2/h11*mp #du_dt #tau_11 in paper
        
        #hv[1,:] =  hv[0,:] + tau0_1/(5./4 * rho) * mv 
        #hp[1,:] =  hp[0,:] - K * tau0_2/((5./4)*mp) 
        
        #hv[1,:] =  hv[1,:] + tau0_1/(h12 * rho) * mv 
        #hp[1,:] =  hp[1,:] - K * tau0_2/h12*mp

        hv[nx-1,:] = hv[nx-1,:] + tauN_1/(h11 ) * pv #tau2n in paper
        hp[nx-1,:] = hp[nx-1,:] + tauN_2/h11*pp     #tau1n in paper
        #hv[nx-2,:] = hv[nx-1,:] - tauN_1/( * rho) * pv
        #hp[nx-2, : ] = hp[nx-1,:] + K * tauN_2/h11*pp
        #hv[nx-2,:] = hv[nx-1,:] - tauN_1/(h12 * rho) * pv
        #hp[nx-2, : ] = hp[nx-1,:] + K * tauN_2/h12*pp
    
    
    def mms(self,V, P, V_t, P_t, V_x, P_x, y, t, Ubar,type_0,nx):

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

            x0 = 0.5*(y[-1, 0]-y[0, 0])

            V[:,:]=  (np.exp(-(y -x0 -  cs*t)**2)/1)  #*p.cos(2*np.pi*y) #1/np.sqrt(2.0*np.pi*delta**2)*0.5*(np.exp(-(y+cs*(t)-x0)**2/(2.0*delta**2))\
                                               #   + np.exp(-(y-cs*(t)-x0)**2/(2.0*delta**2)))

            P[:,:] =   V + 10 #(np.exp(-(y - cs*(t))**2/(0.3 * x0)))#*np.cos(2*np.pi*y) #0 #1/np.sqrt(2.0*np.pi*delta**2)*0.5*Zs*(np.exp(-(y+cs*(t)-x0)**2/(2.0*delta**2))\
                                                 #   - np.exp(-(y-cs*(t)-x0)**2/(2.0*delta**2)))

            V_t[:,:] =   2* cs* V*(y -x0 -  cs*t)/1# 10/x0* (-cs/0.3) * 2 * t  * (np.exp(-( cs*(t))**2/(0.3)))*np.cos(2*np.pi*y)
            P_t[:,:] =  V_t # 5/x0 *(-cs/0.3) * 2 * t  * (np.exp(-(cs*(t))**2/(0.3)))*np.cos(2*np.pi*y)
            V_x[:,:] =  -V*2*(y -x0 -  cs*t)/1 #10 * (-2*np.pi) * (np.exp(-( cs*(t))**2/(0.3)))*np.sin(2*np.pi*y)
            P_x[:,:] =  V_x# -5  * 2*np.pi*  (np.exp(-(cs*(t))**2/(0.3)))*np.sin(2*np.pi*y)


        if type_0 in ('Sinusoidal'):
           # L = 10 

            delta = (y[-1,0]-y[0,0])
            


            V[:,:]=  np.cos(2*np.pi* t) * np.sin(5. * np.pi *y/delta)#np.cos(ntv * t)*np.sin(nyv *y + fs)

            P[:,:]= np.sin(3*np.pi *t) * np.sin(10. * np.pi *y/delta) + 10

            V_t[:,:] = -2.*np.pi*np.sin(2*np.pi*t)*np.sin(5*np.pi*y/delta)
            P_t[:,:] = 3.*np.pi*np.cos(3*np.pi*t)*np.sin(10*np.pi*y/delta)

            V_x[:,:] = (5.*np.pi/delta) *np.cos(2 * np.pi * t)*np.cos( 5*np.pi*y/delta)
            P_x[:,:] = (10.*np.pi/delta) *np.sin(3*np.pi * t)*np.cos(10*np.pi*y/delta)
            
        if type_0 in ('Dam_Break'):
            
            h = nx/(y[-1,0] - y[0,0]) #get number of cells per 1 L of domain
            position_break =   int(5.0 * h)    #then use it to find index for x = 5.0 of the domain
            #rint(position_break)
            P[:,:] = 1.0000000  #
            V[:,:] = 0.0
            P[position_break:,:] = 0.5#0.0000000000000001
            #V_t[:,:] = 0.0
            #P_t[:,:] = 0.0
            
            #V_x[:,:] = 0.0
            #P_x[:,:] = 0.0
        if type_0 in ('Lake_At_Rest'):
            b = 0 * y
            V = 0 * y
            
            mask_b = np.logical_and(8 < y, y < 12)
            b[mask_b] = 0.2 - 0.05*(y[mask_b] - 10 )**2 
            #P[:,:] = 0.5
            
            P[:,:] =  0.5 - b  # h + b
            #V[:,:] = 0.0000000000000000 
            
        # if type_0 in ('Dam_Break_Dry'):
        #     h = nx/(y[-1,0] - y[0,0]) #get number of cells per 1 L of domain
        #     position_break =   int(5.0 * h)    #then use it to find index for x = 5.0 of the domain
        #     #rint(position_bxsreak)
        #     P[:,:] = 1.0000000  #
        #     V[:,:] = 0.000000000
        #     P[position_break:,:] = 0.01#0.0000000000000001
        #     #V_t[:,:] = 0.0
        #     #P_t[:,:] = 0.0
        #      #this does not work
        #     #V_x[:,:] = 0.0 
            






    def bcm(self,mv, mp, v, p, V, P , rho, K , r):

        
        zs = np.sqrt(rho * K ) #rho*cs

        a = 0.5*(zs*v - p)
        b = 0.5*(zs*v + p)

        A = 0.5*(zs*V - P)
        B = 0.5*(zs*V + P) 

        g = A - r*B
        B1 = p - P #  0 

        mv[:] = B1
        mp[:] =  B1  #-K/zs*(a - r*b - (A - r*B))

    def bcp(self, pv, pp, v, p, V, P, rho, K, r):

        
        zs = np.sqrt( rho * K )#rho*cs

        a = 0.5*(zs*v + p)
        b = 0.5*(zs*v - p)

        A = 0.5*(zs*V + P)
        B = 0.5*(zs*V - P)

        g = A - r*B
        Bn = 0.5  * (zs *v - p ) - 0.5*(zs*V-P)

        pv[:] = Bn
        pp[:] = Bn
        
    def f_v(x,t,v):
        return np.cos(2* np.pi *t) * np.sin(5 * np.pi *y/10 + 5)#np.cos(ntv * t)*np.sin(nyv *y + fs)

    def p_v(x,t,v):#np.cos(ntv * t)*np.sin(nyv *y + fs)
        return np.sin(3*np.pi *t) * np.sin(10 * np.pi *y / 10  + 5)
    
    

        
        
    
    def acoustic_rate(self,hv, hp, v, h, rho, K, nx, dx, order, t, y, r0, r1, tau0_1,tau0_2,tauN_1,tauN_2, type_0, forcing, H, Ubar, g,bc_type_0, bc_type_N, fd_type,flux_type,topography,hyperviscosity):
# we compute rates that will be used for Runge-Kutta time-stepping
# 
        from dpsbp_operators_correct import dxd_m_DP, dxd_p_DP
        from DRP_FD import dxd_m_DRP, dxd_p_DRP
        from DRP_FD_periodic import dxd_m_DRP_periodic, dxd_p_DRP_periodic
        from dpsbp_operators_periodic import dxd_m_DP_periodic, dxd_p_DP_periodic
        from sbp_operators_periodic import dxd_m_SBP_periodic
        from sbp_operators import dxd_m_SBP

        import numpy as np
    
        #delta = (y[-1,0]-y[0,0])
       # L = 10
        #ubar = 0
        #H = 1
        #g = 9.8
        #ubar= 0.5*np.sqrt(g*H)


        U = np.zeros((nx, 1))
        P = np.zeros((nx, 1))
        Ut = np.zeros((nx, 1))
        Pt = np.zeros((nx, 1))
        Ux = np.zeros((nx, 1))
        Px = np.zeros((nx, 1))
        
        bx = np.zeros((nx,1))
        b = 0 * y
        
        if topography == 'yes':
            bx = 0 * y
            mask_b = np.logical_and(8 < y, y < 12)
            bx[mask_b] = - 0.10 * (y[mask_b] - 10 )
            #mask_b = np.logical_and(8 < y, y < 12)
            b[mask_b] = 0.2 - 0.05*(y[mask_b] - 10 )**2 
            #mask_a = np.logical_and(12 <= y, y <= 8) 
            
        else:
            bx = 0.0
            b = 0
        
        self.mms(U, P, Ut,Pt, Ux, Px, y, t,Ubar, type_0,nx)

        if flux_type == 'linear':
            flux_1 = Ubar*v + g*h
            flux_2 = Ubar*h + H*v

        
        
        if flux_type == 'nonlinear':
            #b = 0
            flux_1 =  v**2/2 + g*(h+b) ##Ubar*v + g*h
            flux_2 = v*(h+b)
        # initialize arrays for computing derivatives
        
        flux_1_x = np.zeros((nx,1))
        
        
        flux_2_x = np.zeros((nx,1))
        
        if fd_type == 'DP':
            dxd_m = dxd_m_DP
            dxd_p = dxd_p_DP
            
        if fd_type == 'DRP':
            dxd_m = dxd_m_DRP
            dxd_p = dxd_p_DRP

        if fd_type == 'SBP':
            dxd_m = dxd_m_SBP
            dxd_p = dxd_m
        
        if fd_type == 'DRP_periodic':
            dxd_m = dxd_m_DRP_periodic
            dxd_p = dxd_p_DRP_periodic
            
        if fd_type == 'DP_periodic':
            dxd_m = dxd_m_DP_periodic
            dxd_p = dxd_p_DP_periodic
        
        if fd_type == 'SBP_periodic':
            dxd_m = dxd_m_SBP_periodic
            dxd_p = dxd_m
            
        dxd_m(flux_1_x, flux_1, nx, dx, order)
        dxd_p(flux_2_x, flux_2, nx, dx, order)
        
        u_x = np.zeros((nx,1))
        h_x = np.zeros((nx,1))
        
        u_xx = np.zeros((nx,1))
        h_xx = np.zeros((nx,1))
        
        u_xxx = np.zeros((nx,1))
        h_xxx = np.zeros((nx,1))
        
        u_xxxx = np.zeros((nx,1))
        h_xxxx = np.zeros((nx,1))
        
        u_xxxxx = np.zeros((nx,1))
        h_xxxxx = np.zeros((nx,1))
        
        u_xxxxxx = np.zeros((nx,1))
        h_xxxxxx = np.zeros((nx,1))
        
    
        
        #vx = np.zeros((nx, 1))
        #px = np.zeros((nx, 1))
        


        # compute first derivatives for velocity and stress fields
        # dxd_m(flux_1_x, flux_1, nx, dx, order)
        # dxd_p(flux_2_x, flux_2, nx, dx, order)
        # dxd_p(u_x, v, nx, dx, order)
        # dxd_m(h_x, h, nx, dx, order)
        
        # dxd_m(u_xx, u_x, nx, dx, order)
        # dxd_p(h_xx, h_x, nx, dx, order)
        
        # dxd_p(u_xxx, u_xx, nx, dx, order)
        # dxd_m(h_xxx, h_xx, nx, dx, order)
        
        # dxd_m(u_xxxx, u_xxx, nx, dx, order)
        # dxd_p(h_xxxx, h_xxx, nx, dx, order)    
        
        h11 = np.zeros((1, 1))
        #h12 = np.zeros((1,1))
        #penaltyweight(h11, dx, order)
        
        if fd_type == 'DP':
            if order==2:
                h11[:] = (0.25)*dx
                #h12[:] = (1.25)*dx
            elif order== 4:
                h11[:] = (49./144.)*dx
                
            elif order==5:
                h11[:] = (251./720)*dx
        
            elif order==6:
                h11[:] = (13613.0/43200.0)*dx
                
                
        if fd_type == 'DRP':
            if order == 4:
                h11[:] = 0.315586328095031*dx
            elif order == 5:
                h11[:] =0.318079048635254*dx
            elif order == 6:
                h11[:] =0.294425496425380*dx
            elif order == 7:
                h11[:] =0.294900239556150*dx

        if fd_type == 'SBP':  
            if order==2:
                h11[:] = 0.5*dx
            elif order== 4:
                h11[:] = (17.0/48.0)*dx

            elif order==6:
                h11[:] = (13649.0/43200.0)*dx

        # compute first derivatives for velocity and stress fields
        # def filter_func(y):
    
        #     L = 10
        #     x1 = 0.2 * L  
        #     x2 = 0.8 * L 
        #     maskx =np.logical_and( (x1 <= y), (y<= x2))
        #     maskx2 = np.logical_and((x2<= y),(y<= L))
        #     maskx1 = np.logical_and((0<= y),(y<= x1))
        #     b = np.zeros_like(y) 
        #     b[maskx] = 1
        #     b[maskx2] = 1 - ((y[maskx2] -x2)/(L - x2))**4
        #     b[maskx1] = 1 -((x1- y[maskx1])/(x1))**4
            
        #     return b
        
        
        def smooth_boxcar(x, a, b, c):
            """
            Smooth boxcar function with zero derivative at the boundaries.

            Arguments:
            x -- input values
            a -- start of the boxcar region
            b -- end of the boxcar region
            c -- smoothness parameter

            Returns:
            y -- smoothed boxcar function values
            """

            y = 1 / (1 + np.exp(-c * (x - a))) - 1 / (1 + np.exp(-c * (x - b)))
            
            return y
           
        if hyperviscosity == 'on':
            L = 10
            
        
            
            tophat = smooth_boxcar(y,10*dx,L-10*dx,20)
            
            dxd_p(u_x, v, nx, dx, order)
            dxd_m(h_x, h, nx, dx, order)
            
            #h_x[0,0] =  h_x[0,0] + 0.5/h11[:]*h[0,0]
            #h_x[-1,0] =  h_x[-1,0] - 0.5/h11[:]*h[-1,0]
            
            #u_x[0,0] =  u_x[0,0] + 0.5/h11[:]*v[0,0]
            #u_x[-1,0] =  u_x[-1,0] - 0.5/h11[:]*v[-1,0]
            dxd_m(u_xx, u_x, nx, dx, order)
            dxd_p(h_xx, h_x, nx, dx, order)
            
            u_xx = tophat * u_xx
            h_xx = tophat * h_xx
            h_xx[0,0] = h_xx[0,0] + 0.0/h11[:]*h_x[0,0] * tophat[0]
            h_xx[-1,0] =  h_xx[-1,0] - 0.0/h11[:]*h_x[-1,0] * tophat[-1]
            
            u_xx[0,0] =  u_xx[0,0] + 0.0/h11[:]*u_x[0,0]* tophat[0]
            u_xx[-1,0] =  u_xx[-1,0] - 0.0/h11[:]*u_x[-1,0]* tophat[-1]
            
            dxd_p(u_xxx, u_xx, nx, dx, order)
            dxd_m(h_xxx, h_xx, nx, dx, order)
            
            #h_xxx[0,0] =  h_xxx[0,0] - 0.5/h11[:]*h_xx[0,0]
            #h_xxx[-1,0] =  h_xxx[-1,0] + 0.5/h11[:]*h_xx[-1,0]
            
            #u_xxx[0,0] =  u_xxx[0,0] - 0.5/h11[:]*u_xx[0,0]
            #u_xxx[-1,0] =  u_xxx[-1,0] + 0.5/h11[:]*u_xx[-1,0]
            
            dxd_m(u_xxxx, u_xxx, nx, dx, order)
            dxd_p(h_xxxx, h_xxx, nx, dx, order)
            
            h_xxxx[0,0] =  h_xxxx[0,0] + 0.0/h11[:]*h_xxx[0,0]
            h_xxxx[-1,0] =  h_xxxx[-1,0] - 0.0/h11[:]*h_xxx[-1,0]
            
            u_xxxx[0,0] =  u_xxxx[0,0] + 0.0/h11[:]*u_xxx[0,0]
            u_xxxx[-1,0] =  u_xxxx[-1,0] - 0.0/h11[:]*u_xxx[-1,0]
            
            dxd_p(u_xxxxx, u_xxxx, nx, dx, order)
            dxd_m(h_xxxxx, h_xxxx, nx, dx, order)
            
            dxd_m(u_xxxxxx, u_xxxxx, nx, dx, order)
            dxd_p(h_xxxxxx, h_xxxxx, nx, dx, order)
            
            h_xxxxxx[0,0] =  h_xxxxxx[0,0] + 0.5/h11[:]*h_xxxxx[0,0]
            h_xxxxxx[-1,0] =  h_xxxxxx[-1,0] - 0.5/h11[:]*h_xxxxx[-1,0]
            
            u_xxxxxx[0,0] =  u_xxxxxx[0,0] + 0.5/h11[:]*u_xxxxx[0,0]
            u_xxxxxx[-1,0] =  u_xxxxxx[-1,0] - 0.5/h11[:]*u_xxxxx[-1,0]
            
        if hyperviscosity == 'on_periodic':
            dxd_p(u_x, v, nx, dx, order)
            dxd_m(h_x, h, nx, dx, order)    
            
            dxd_m(u_xx, u_x, nx, dx, order)
            dxd_p(h_xx, h_x, nx, dx, order)
            
            dxd_p(u_xxx, u_xx, nx, dx, order)
            dxd_m(h_xxx, h_xx, nx, dx, order)
            
            dxd_m(u_xxxx, u_xxx, nx, dx, order)
            dxd_p(h_xxxx, h_xxx, nx, dx, order)   
              #10percent of the domain length 
    #   # Ubar*h + H*v 
    #     #forcing = 1
    #     if flux_type == 'linear':
    #         forcing_term_1 = Ubar * Ux + g * Px
    #         forcing_term_2 = Ubar * Px + H * Ux

    #     if flux_type == 'nonlinear':
    #         forcing_term_1 = U*Ux + g*Px
    #         forcing_term_2 =  Px * U + Ux* P
        
        
        
        
      # Ubar*h + H*v 
        #forcing = 1
        if flux_type == 'linear':
            forcing_term_1 = Ubar * Ux + g * Px
            forcing_term_2 = Ubar * Px + H * Ux
            w11, w12, w21, w22 = g , -Ubar, -Ubar, H
            
            Art = 1/(g*H - Ubar**2) 

        if flux_type == 'nonlinear':
            forcing_term_1 = U*Ux + g*Px
            forcing_term_2 =  Px * U + Ux* P
            w11, w12, w21, w22 = g, -0.5*v, h, -0.5*v # 1 0 
            
            Art = 1/(g*h/2 - 1/4*v**2)
            



        hyper_factor = 0.1#.1#0.1#0.1# 1.0
                
        
        hv[:,:] = -flux_1_x + forcing*(Ut + forcing_term_1)  - hyper_factor*dx**3*Art*(w11*u_xxxx + w12*h_xxxx) #- g * 0 * bx
        hp[:,:] = -flux_2_x + forcing*( Pt +  forcing_term_2 ) - hyper_factor*dx**3*Art*(w21*h_xxxx + w22*u_xxxx) 
               
        # hv[:,:] = -flux_1_x + forcing*(Ut + forcing_term_1) - hyper_const*dx**3*u_xxxx
        # hp[:,:] = -flux_2_x + forcing*( Pt +  forcing_term_2 ) - hyper_const*dx**3*h_xxxx
        
        if fd_type == 'DRP' or fd_type == 'DP' or fd_type == 'SBP':
            # impose boundary conditions using penalty: SAT
            self.impose_bc(hv, hp, v, h, y, rho, K, nx, dx, order, forcing*U, forcing*P, r0, r1, tau0_1, tau0_2, tauN_1, tauN_2, H, Ubar, g, type_0, fd_type,flux_type,bc_type_0, bc_type_N,forcing)
              
        

