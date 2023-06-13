import numpy as np 
    
def acoustic_RK4(self,rv, rp, v, p, rho, K, nx, dx, order, y, t, dt, r0, r1, tau0_1, tau0_2, tauN_1, tauN_2, type_0, forcing, H, Ubar, g,bc_type_0, bc_type_N, fd_type,flux_type, topography,hyperviscosity):

        # fourth order Runge-Kutta time-stepping



        # intialize arrays for Runge-Kutta stages
        k1v = np.zeros((nx, 1))
        k1p = np.zeros((nx, 1))
        k2v = np.zeros((nx, 1))
        k2p = np.zeros((nx, 1))
        k3v = np.zeros((nx, 1))
        k3p = np.zeros((nx, 1))
        k4v = np.zeros((nx, 1))
        k4p = np.zeros((nx, 1))
        
        


        self.acoustic_rate(k1v, k1p, v, p, rho, K, nx, dx, order, t, y, r0, r1, tau0_1, tau0_2, tauN_1, tauN_2, type_0, forcing, H, Ubar, g,bc_type_0, bc_type_N, fd_type,flux_type,topography,hyperviscosity)


        self.acoustic_rate(k2v, k2p, v+0.5*dt*k1v, p+0.5*dt*k1p, rho, K, nx, dx, order, t+0.5*dt, y, r0, r1, tau0_1, tau0_2, tauN_1, tauN_2, type_0, forcing, H, Ubar, g,bc_type_0, bc_type_N, fd_type,flux_type,topography,hyperviscosity)

        self.acoustic_rate(k3v, k3p, v+0.5*dt*k2v, p+0.5*dt*k2p, rho, K, nx, dx, order, t+0.5*dt, y, r0, r1, tau0_1, tau0_2, tauN_1, tauN_2, type_0, forcing, H, Ubar, g,bc_type_0, bc_type_N, fd_type,flux_type,topography,hyperviscosity)



        self.acoustic_rate(k4v, k4p, v+dt*k3v, p+dt*k3p, rho, K, nx, dx, order, t+dt, y,  r0, r1, tau0_1, tau0_2, tauN_1, tauN_2, type_0, forcing, H, Ubar, g,bc_type_0, bc_type_N, fd_type,flux_type,topography,hyperviscosity)

        # update fields
        rv[:,:] = v + (dt/6.0)*(k1v + 2.0*k2v + 2.0*k3v + k4v)  
        rp[:,:] = p + (dt/6.0)*(k1p + 2.0*k2p + 2.0*k3p + k4p)




            
