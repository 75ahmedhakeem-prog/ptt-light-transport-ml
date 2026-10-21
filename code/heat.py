"""
heat.py - axisymmetric transient heat conduction (Pennes with zero perfusion) for the dataset-1 phantom.
Explicit finite volumes on an (r, z) grid; convective (h) losses at the top, side and bottom surfaces.
Inputs: volumetric heat source Q(r, z) in W/m^3 (on the same grid) switched on for t_on seconds.
"""
import numpy as np

class Phantom:
    def __init__(self, R=11e-3, H=20e-3, nr=22, nz=40, k=0.6, rhoc=4.18e6):
        self.R, self.H, self.nr, self.nz, self.k, self.rhoc = R, H, nr, nz, k, rhoc
        self.dr, self.dz = R / nr, H / nz
        self.r = (np.arange(nr) + 0.5) * self.dr; self.z = (np.arange(nz) + 0.5) * self.dz
        rf = np.arange(nr + 1) * self.dr                       # face radii
        self.Ar = 2 * np.pi * rf[:, None] * self.dz            # radial face areas (nr+1, 1)
        self.Az = np.pi * (rf[1:]**2 - rf[:-1]**2)[:, None]    # axial face areas (nr, 1)
        self.V = self.Az * self.dz

    def run(self, Q, h_top, h_side, h_bot, t_on=900.0, t_end=1800.0, dt=0.2, probes=None, save_every=15.0):
        """returns times (s) and temperature rise at the probe cells [(r, z) in m]"""
        T = np.zeros((self.nr, self.nz)); k, dr, dz = self.k, self.dr, self.dz
        probes = probes or []
        idx = [(min(int(pr / dr), self.nr - 1), min(int(pz / dz), self.nz - 1)) for pr, pz in probes]
        C = self.rhoc * self.V; out_t, out = [], []
        nsteps = int(round(t_end / dt)); every = int(round(save_every / dt))
        for n in range(nsteps + 1):
            t = n * dt
            if n % every == 0:
                out_t.append(t); out.append([T[i, j] for i, j in idx])
            if n == nsteps: break
            q = np.zeros_like(T)
            # radial conduction
            fr = k * self.Ar[1:-1] * (T[1:] - T[:-1]) / dr            # heat flowing from cell i+1 into cell i
            q[:-1] += fr; q[1:] -= fr
            # axial conduction
            fz = k * self.Az * (T[:, 1:] - T[:, :-1]) / dz
            q[:, :-1] += fz; q[:, 1:] -= fz
            # convective boundaries (half-cell conduction + h in series)
            ut = 1 / (1 / h_top + dz / 2 / k); ub = 1 / (1 / h_bot + dz / 2 / k); us = 1 / (1 / h_side + dr / 2 / k)
            q[:, 0] -= ut * self.Az[:, 0] * T[:, 0]
            q[:, -1] -= ub * self.Az[:, 0] * T[:, -1]
            q[-1, :] -= us * self.Ar[-1, 0] * T[-1, :]
            if t < t_on: q += Q * self.V
            T = T + dt * q / C
        return np.array(out_t), np.array(out)
