"""
design_sim.py - subcutaneous tumour model for the design dataset (delta-P1 light + Pennes bioheat).
Geometry (axisymmetric, mm): skin 0-1.5 mm, adipose soft tissue below, spherical tumour of diameter Dt whose top lies
d_t mm below the skin; laser beam of diameter S on the surface.  Temperatures are computed for a unit irradiance
(1 W/cm^2) and scaled linearly, which is exact for constant properties and perfusion.
Optical properties: mu_a from water/blood/fat fractions, mu_s' = a (lambda/500)^-b, g = 0.9 (representative values [Jacques 2013]).
"""
import numpy as np, dp1

R, H = 25.0, 25.0; NR, NZ = 100, 100; DR, DZ = R / NR, H / NZ           # optical grid 0.25 mm
TR, TZ = 50, 50                                                         # thermal grid 0.5 mm
rc = (np.arange(NR) + 0.5) * DR; zc = (np.arange(NZ) + 0.5) * DZ
rt = (np.arange(TR) + 0.5) * 0.5; zt = (np.arange(TZ) + 0.5) * 0.5
SKIN = 1.5
MUA = {808: {"skin": 0.010, "fat": 0.005, "tum": 0.0135}, 1064: {"skin": 0.014, "fat": 0.0094, "tum": 0.018}}   # mm^-1
SCAT = {"skin": (4.6, 1.42), "fat": (1.84, 0.672), "tum": (1.89, 1.286)}      # mu_s' = a (lam/500)^-b, a in mm^-1
THERM = {"skin": (0.37, 1109 * 3391), "fat": (0.21, 911 * 2348), "tum": (0.51, 1050 * 3600)}   # k (W/m/K), rho*c (J/m3/K)
RHOC_B = 1060 * 3617

def regions(r, z, depth, Dt):
    zc0 = SKIN + depth + Dt / 2
    tum = (r[:, None]**2 + (z[None, :] - zc0)**2) <= (Dt / 2)**2
    skin = np.broadcast_to(z[None, :] < SKIN, tum.shape) & ~tum
    return tum, skin

def optics(lam, depth, Dt, mua_gnr, g=0.9):
    tum, skin = regions(rc, zc, depth, Dt)
    mua = np.full((NR, NZ), MUA[lam]["fat"]); musr = np.full((NR, NZ), SCAT["fat"][0] * (lam / 500)**-SCAT["fat"][1])
    mua[skin] = MUA[lam]["skin"]; musr[skin] = SCAT["skin"][0] * (lam / 500)**-SCAT["skin"][1]
    mua[tum] = MUA[lam]["tum"] + mua_gnr; musr[tum] = SCAT["tum"][0] * (lam / 500)**-SCAT["tum"][1]
    return mua, musr / (1 - g), np.full((NR, NZ), g), tum

def light(lam, spot, depth, Dt, mua_gnr):
    mua, mus, g, _ = optics(lam, depth, Dt, mua_gnr)
    P = 0.01 * np.pi * (spot / 2)**2                                    # 1 W/cm^2 = 0.01 W/mm^2
    o = dp1.solve(R, H, NR, NZ, mua, mus, g, spot / 2, P=P, n_in=1.4)
    vol = 2 * np.pi * rc[:, None] * DR * DZ; E = (o["Q"] * vol).reshape(TR, 2, TZ, 2).sum((1, 3))
    return E / (2 * np.pi * rt[:, None] * 0.5 * 0.5) * 1e9                # W/m^3 on the thermal grid

def thermal_maps(depth, Dt, perf):
    tum, skin = regions(rt, zt, depth, Dt)
    k = np.full((TR, TZ), THERM["fat"][0]); rhoc = np.full((TR, TZ), THERM["fat"][1])
    k[skin], rhoc[skin] = THERM["skin"]; k[tum], rhoc[tum] = THERM["tum"]
    return k, rhoc, np.full((TR, TZ), perf * RHOC_B), tum, skin

def heat_fields(Q, k, rhoc, wrc, t_end=900.0, dt=0.4, every=10.0, h_top=10.0):
    """transient temperature rise fields for a unit source; deep boundaries held at body temperature"""
    Rm, Hm = R * 1e-3, H * 1e-3; dr, dz = Rm / TR, Hm / TZ; rf = np.arange(TR + 1) * dr
    Ar = 2 * np.pi * rf[1:-1, None] * dz; Az = np.pi * (rf[1:]**2 - rf[:-1]**2)[:, None]; V = Az * dz
    Gr = (2 * k[1:] * k[:-1] / (k[1:] + k[:-1])) * Ar / dr; Gz = (2 * k[:, 1:] * k[:, :-1] / (k[:, 1:] + k[:, :-1])) * Az / dz
    ut = Az[:, 0] / (1 / h_top + dz / 2 / k[:, 0]); ub = Az[:, 0] * k[:, -1] / (dz / 2); us = 2 * np.pi * Rm * dz * k[-1, :] / (dr / 2)
    C = rhoc * V; T = np.zeros((TR, TZ)); QV = Q * V; frames = []; n_every = int(round(every / dt))
    for n in range(int(round(t_end / dt)) + 1):
        if n % n_every == 0: frames.append(T.copy())
        q = QV - wrc * V * T
        fr = Gr * (T[1:] - T[:-1]); q[:-1] += fr; q[1:] -= fr
        fz = Gz * (T[:, 1:] - T[:, :-1]); q[:, :-1] += fz; q[:, 1:] -= fz
        q[:, 0] -= ut * T[:, 0]; q[:, -1] -= ub * T[:, -1]; q[-1, :] -= us * T[-1, :]
        T = T + dt * q / C
    return np.array(frames)                                             # (n_frames, TR, TZ), every 10 s

def outcomes(frames, tum, skin, I, times, every=10.0, T0=37.0):
    """coverage (CEM43 >= 240 min), max skin temperature, collateral volume (mm^3) at each heating time"""
    T = T0 + I * frames
    R_ = np.where(T >= 43, 0.5, 0.25); cem = np.cumsum(R_**(43 - T) * every / 60.0, axis=0)
    Vc = (2 * np.pi * rt[:, None] * 0.5 * 0.5) * np.ones((1, TZ)); Vt = Vc[tum].sum()
    out = []
    for t in times:
        i = int(round(t / every)); dmg = cem[i] >= 240
        out.append(dict(coverage=float(Vc[tum & dmg].sum() / Vt), T_skin_max=float(T[: i + 1][:, skin].max()),
                        T_tum_min=float(T[i][tum].min()),
                        T_tum_max=float(T[: i + 1][:, tum].max()), V_collateral=float(Vc[~tum & dmg].sum())))
    return out
