"""heat2.py - axisymmetric transient conduction with spatially varying k and rho*c (explicit finite volumes)."""
import numpy as np

def run(R, H, nr, nz, k, rhoc, Q, h_top, h_side, h_bot, t_on, t_end, dt=0.25, probes=(), save_every=10.0, perf=None):
    """k, rhoc, Q (W/m^3) on an (nr, nz) grid; lengths in m; returns times and temperature rise at probes.
    perf: optional (nr, nz) array of w_b*rho_b*c_b (W/m^3/K) for Pennes perfusion toward the initial temperature."""
    dr, dz = R / nr, H / nz; r = (np.arange(nr) + 0.5) * dr; rf = np.arange(nr + 1) * dr
    Ar = 2 * np.pi * rf[1:-1, None] * dz; Az = np.pi * (rf[1:]**2 - rf[:-1]**2)[:, None]; V = Az * dz
    kr = 2 * k[1:] * k[:-1] / (k[1:] + k[:-1]); kz = 2 * k[:, 1:] * k[:, :-1] / (k[:, 1:] + k[:, :-1])
    Gr = kr * Ar / dr; Gz = kz * Az / dz
    ut = Az[:, 0] / (1 / h_top + dz / 2 / k[:, 0]); ub = Az[:, 0] / (1 / h_bot + dz / 2 / k[:, -1])
    us = 2 * np.pi * R * dz / (1 / h_side + dr / 2 / k[-1, :])
    C = rhoc * V; T = np.zeros((nr, nz)); idx = [(min(int(pr / dr), nr - 1), min(int(pz / dz), nz - 1)) for pr, pz in probes]
    out_t, out = [], []; every = int(round(save_every / dt)); nsteps = int(round(t_end / dt)); QV = Q * V
    for n in range(nsteps + 1):
        if n % every == 0: out_t.append(n * dt); out.append([T[a, b] for a, b in idx])
        if n == nsteps: break
        q = np.zeros_like(T); fr = Gr * (T[1:] - T[:-1]); q[:-1] += fr; q[1:] -= fr
        fz = Gz * (T[:, 1:] - T[:, :-1]); q[:, :-1] += fz; q[:, 1:] -= fz
        q[:, 0] -= ut * T[:, 0]; q[:, -1] -= ub * T[:, -1]; q[-1, :] -= us * T[-1, :]
        if perf is not None: q -= perf * V * T
        if n * dt < t_on: q += QV
        T = T + dt * q / C
    return np.array(out_t), np.array(out)
