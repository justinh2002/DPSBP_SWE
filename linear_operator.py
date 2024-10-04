import numpy as np
import matplotlib.pyplot as plt
from SWESBP import acoustic_sbp 
from scipy import linalg

from matplotlib import rcParams 
import matplotlib_rcParams
solver = acoustic_sbp()
solver.ncells = 101
nx = solver.ncells
x = np.zeros((nx,1))

    # Initial particle velocity perturbation and discretize the domain
for j in range(0, nx):
    x[j, :] = j*solver.dx
H = np.zeros((nx,1))
V = np.zeros((nx,1))
# Define a constant background state for height (H) and velocity (V)
H[:,:] = np.sin(x[:,:] + 0.7) + 10

V[:,:]=  np.cos(x[:,:] - 0.7) 

# Create the initial state as a constant background state
# h_background = H #* np.ones((nx, 1))

jacobian_massflux = solver.compute_jacobian(H[:,:], V[:,:],bc = 'mass flux',epsilon = 1e-8)
jacobian_energyflux = solver.compute_jacobian(H[:,:], V[:,:],bc = 'energy flux',epsilon = 1e-8)
jacobian_transmissive = solver.compute_jacobian(H[:,:], V[:,:],bc = 'transmissive',epsilon = 1e-8)


eigenvalues_1 = linalg.eigvals(jacobian_massflux)
eigenvalues_2 = linalg.eigvals(jacobian_energyflux)
eigenvalues_3 = linalg.eigvals(jacobian_transmissive)




plt.scatter(np.real(eigenvalues_1),np.imag(eigenvalues_1),marker = 'x',color = 'cornflowerblue',label = 'Mass flux BC',s = 10)
plt.scatter(np.real(eigenvalues_2),np.imag(eigenvalues_2),marker = 'o',color = 'lightcoral',label = 'Velocity flux BC',s = 3,alpha = 0.3)
plt.scatter(np.real(eigenvalues_3),np.imag(eigenvalues_3),marker = '^',color = 'green',label = 'Transmissive BC',s = 8)
plt.xlabel(r'Re')
plt.ylabel(r'Im')
# print(eigenvalues_1)
plt.legend()

plt.show()
