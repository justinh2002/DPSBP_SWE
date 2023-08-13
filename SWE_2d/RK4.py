import numpy as np 
import SWESBP_2d
    
def acoustic_RK4(self,ru, rv, rp,u, v, p, rho, K, nx,ny, dx,dy, order, x, y, t, dt, type_0, Ubar,Vbar, H, g,flux_type,vorticity):

        # fourth order Runge-Kutta time-stepping



        # intialize arrays for Runge-Kutta stages
        k1u = np.zeros((nx, ny))
        k1v = np.zeros((nx, ny))
        k1p = np.zeros((nx, ny))
        
        k2u = np.zeros((nx, ny))
        k2v = np.zeros((nx, ny))
        k2p = np.zeros((nx, ny))
        
        k3u = np.zeros((nx, ny))
        k3v = np.zeros((nx, ny))
        k3p = np.zeros((nx, ny))
        
        k4u = np.zeros((nx, ny))
        k4v = np.zeros((nx, ny))
        k4p = np.zeros((nx, ny))
        

        #print(energy_t)        
        


        self.acoustic_rate(k1u,k1v, k1p,u, v, p, rho, K, nx,ny, dx,dy, order, t,x, y,  type_0, Ubar,Vbar,H,g,flux_type,vorticity)
       # def acoustic_rate(self,hu,hv, hp,u, v, h, rho, K, nx, ny,dx,dy, order, t, x,y, type_0, Ubar,Vbar,H, g):
        energy =  ((p*(u**2+v**2) /2  + g * p**2).sum())
                
        energy_t = np.sum(u * p * k1u +  v * p * k1v +  (1/2 * (u ** 2+v**2)  +  g * p) * k1p)/energy
        #print(energy_t)       

        self.acoustic_rate(k2u,k2v, k2p,u + 0.5*dt*k1u, v+0.5*dt*k1v, p+0.5*dt*k1p, rho, K, nx,ny, dx,dy, order, t+0.5*dt,x, y,  type_0, Ubar,Vbar, H, g,flux_type,vorticity)

        self.acoustic_rate(k3u,k3v, k3p, u + 0.5*dt*k2u ,v+0.5*dt*k2v, p+0.5*dt*k2p, rho, K, nx,ny, dx, dy,order, t+0.5*dt, x, y, type_0, Ubar,Vbar, H,g,flux_type,vorticity)



        self.acoustic_rate(k4u,k4v, k4p,u+dt*k3u, v+dt*k3v, p+dt*k3p, rho, K, nx,ny, dx,dy, order, t+dt,x, y, type_0, Ubar,Vbar, H,g,flux_type,vorticity)


        # update fields
        ru[:,:] = u + (dt/6.0)*(k1u + 2.0*k2u + 2.0*k3u + k4u)  
        rv[:,:] = v + (dt/6.0)*(k1v + 2.0*k2v + 2.0*k3v + k4v)  
        rp[:,:] = p + (dt/6.0)*(k1p + 2.0*k2p + 2.0*k3p + k4p)
        





            
