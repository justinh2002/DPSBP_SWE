import numpy as np 
import SWESBP_2d

def acoustic_SSPRK3(self, ru, rv, rp, u, v, p, rho, K, nx, ny, dx, dy, order, x, y, t, dt, type_0, Ubar, Vbar, H, g, flux_type, vorticity):

    # Initialize arrays for SSP RK3 stages
    k1u = np.zeros((nx, ny))
    k1v = np.zeros((nx, ny))
    k1p = np.zeros((nx, ny))

    k2u = np.zeros((nx, ny))
    k2v = np.zeros((nx, ny))
    k2p = np.zeros((nx, ny))

    k3u = np.zeros((nx, ny))
    k3v = np.zeros((nx, ny))
    k3p = np.zeros((nx, ny))

    # Stage 1
    self.acoustic_rate(k1u, k1v, k1p, u, v, p, rho, K, nx, ny, dx, dy, order, t, x, y, type_0, Ubar, Vbar, H, g, flux_type, vorticity)

    # Stage 2
    u_stage2 = u + dt * k1u
    v_stage2 = v + dt * k1v
    p_stage2 = p + dt * k1p
    self.acoustic_rate(k2u, k2v, k2p, u_stage2, v_stage2, p_stage2, rho, K, nx, ny, dx, dy, order, t + dt, x, y, type_0, Ubar, Vbar, H, g, flux_type, vorticity)

    # Stage 3
    u_stage3 = 0.25 * u + 0.75 * u_stage2 + 0.75 * dt * k2u
    v_stage3 = 0.25 * v + 0.75 * v_stage2 + 0.75 * dt * k2v
    p_stage3 = 0.25 * p + 0.75 * p_stage2 + 0.75 * dt * k2p
    self.acoustic_rate(k3u, k3v, k3p, u_stage3, v_stage3, p_stage3, rho, K, nx, ny, dx, dy, order, t + 0.5 * dt, x, y, type_0, Ubar, Vbar, H, g, flux_type, vorticity)

    # Update fields using SSP RK3 formula
    ru[:, :] = (1.0 / 3.0) * u + (2.0 / 3.0) * u_stage3 + (2.0 / 3.0) * dt * k3u
    rv[:, :] = (1.0 / 3.0) * v + (2.0 / 3.0) * v_stage3 + (2.0 / 3.0) * dt * k3v
    rp[:, :] = (1.0 / 3.0) * p + (2.0 / 3.0) * p_stage3 + (2.0 / 3.0) * dt * k3p

# Rest of your code...
