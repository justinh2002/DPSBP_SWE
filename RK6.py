import numpy as np

def acoustic_RK6(self, rv, rp, v, p, rho, K, nx, dx, order, y, t, dt, r0, r1, tau0_1, tau0_2, tauN_1, tauN_2, type_0, forcing, H, Ubar, g, fd_type, flux_type, topography,hyperviscosity):

    # initialize arrays for Runge-Kutta stages
    k1v = np.zeros((nx, 1))
    k1p = np.zeros((nx, 1))
    k2v = np.zeros((nx, 1))
    k2p = np.zeros((nx, 1))
    k3v = np.zeros((nx, 1))
    k3p = np.zeros((nx, 1))
    k4v = np.zeros((nx, 1))
    k4p = np.zeros((nx, 1))
    k5v = np.zeros((nx, 1))
    k5p = np.zeros((nx, 1))
    k6v = np.zeros((nx, 1))
    k6p = np.zeros((nx, 1))

    self.acoustic_rate(k1v, k1p, v, p, rho, K, nx, dx, order, t, y, r0, r1, tau0_1, tau0_2, tauN_1, tauN_2, type_0, forcing, H, Ubar, g, fd_type, flux_type, topography,hyperviscosity)

    self.acoustic_rate(k2v, k2p, v + (1 / 4) * dt * k1v, p + (1 / 4) * dt * k1p, rho, K, nx, dx, order, t + (1 / 4) * dt, y, r0, r1, tau0_1, tau0_2, tauN_1, tauN_2, type_0, forcing, H, Ubar, g, fd_type, flux_type, topography,hyperviscosity)

    self.acoustic_rate(k3v, k3p, v + (1 / 8) * dt * k1v + (1 / 8) * dt * k2v, p + (1 / 8) * dt * k1p + (1 / 8) * dt * k2p, rho, K, nx, dx, order, t + (1 / 8) * dt, y, r0, r1, tau0_1, tau0_2, tauN_1, tauN_2, type_0, forcing, H, Ubar, g, fd_type, flux_type, topography,hyperviscosity)

    self.acoustic_rate(k4v, k4p, v - (1 / 2) * dt * k2v + dt * k3v, p - (1 / 2) * dt * k2p + dt * k3p, rho, K, nx, dx, order, t + (1 / 2) * dt, y, r0, r1, tau0_1, tau0_2, tauN_1, tauN_2, type_0, forcing, H, Ubar, g, fd_type, flux_type, topography,hyperviscosity)

    self.acoustic_rate(k5v, k5p, v + (3 / 16) * dt * k1v + (9 / 16) * dt * k4v, p + (3 / 16) * dt * k1p + (9 / 16) * dt * k4p, rho, K, nx, dx, order, t + (3 / 4) * dt, y, r0, r1, tau0_1, tau0_2, tauN_1, tauN_2, type_0, forcing, H, Ubar, g, fd_type, flux_type, topography,hyperviscosity)

    self.acoustic_rate(k6v, k6p, v - (3 / 7) * dt * k1v + (2 / 7) * dt * k2v + (12 / 7) * dt * k3v - (12 / 7) * dt * k4v + (8 / 7) * dt * k5v, p - (3 / 7) * dt * k1p + (2 / 7) * dt * k2p + (12 / 7) * dt * k3p - (12 / 7) * dt * k4p + (8 / 7) * dt * k5p, rho, K, nx, dx, order, t + dt, y, r0, r1, tau0_1, tau0_2, tauN_1, tauN_2, type_0, forcing, H, Ubar, g, fd_type, flux_type, topography,hyperviscosity)

    # update fields
    rv[:, :] = v + (dt / 90) * ( 7 * k1v + 32 * k3v + 12 * k4v + 32 * k5v + 7 * k6v)
    rp[:, :] = p + (dt / 90) * (7 * k1p + 32 * k3p + 12 * k4p + 32 * k5p + 7 * k6p)
