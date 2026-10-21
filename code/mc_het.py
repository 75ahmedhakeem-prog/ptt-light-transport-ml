"""
mc_het.py - Monte Carlo photon transport in a heterogeneous axisymmetric cylinder (delta / Woodcock tracking).
Optical properties are given on an (r, z) grid; the refractive index is uniform inside (n_in).
Outer boundaries: Fresnel with n_top (top), n_side (side), n_bot (bottom).  Units: mm, mm^-1, W.
"""
import numpy as np
from mc import hg_scatter, fresnel

def run(R, H, nr, nz, mua, mus, g, beam_radius, P=1.0, n_in=1.37, n_top=1.0, n_side=1.0, n_bot=1.0, N=200000, seed=1, batch=40000):
    rng = np.random.default_rng(seed); dr, dz = R / nr, H / nz
    mut = mua + mus; mmax = mut.max(); A = np.zeros((nr, nz)); esc = np.zeros(3); spec = 0.0
    for b0 in range(0, N, batch):
        n = min(batch, N - b0)
        rr = beam_radius * np.sqrt(rng.random(n)); ph = 2 * np.pi * rng.random(n)
        p = np.c_[rr * np.cos(ph), rr * np.sin(ph), np.zeros(n)]; u = np.tile([0, 0, 1.0], (n, 1))
        Rsp = ((n_top - n_in) / (n_top + n_in))**2; w = np.full(n, 1 - Rsp); spec += Rsp * n
        alive = np.ones(n, bool)
        while alive.any():
            i = np.where(alive)[0]; pi, ui = p[i], u[i]
            s = -np.log(rng.random(len(i))) / mmax
            # distance to the outer boundary
            big = 1e30; dzb = np.full(len(i), big); bid = np.full(len(i), -1)
            m = ui[:, 2] < 0; dzb[m] = -pi[m, 2] / ui[m, 2]; m2 = ui[:, 2] > 0; dzb[m2] = (H - pi[m2, 2]) / ui[m2, 2]
            bid = np.where(m, 0, np.where(m2, 1, -1)); db = dzb.copy()
            aa = ui[:, 0]**2 + ui[:, 1]**2; bq = 2 * (pi[:, 0] * ui[:, 0] + pi[:, 1] * ui[:, 1]); cc = pi[:, 0]**2 + pi[:, 1]**2 - R**2
            disc = bq**2 - 4 * aa * cc; ok = (aa > 1e-14) & (disc > 0)
            ds = np.full(len(i), big); ds[ok] = (-bq[ok] + np.sqrt(disc[ok])) / (2 * aa[ok])
            side = ds < db; db = np.where(side, ds, db); bid = np.where(side, 2, bid)
            hit = db <= s; d = np.minimum(s, db); pi = pi + ui * d[:, None]
            # boundary interaction
            if hit.any():
                h = np.where(hit)[0]; bb = bid[h]; nrm = np.zeros((len(h), 3)); nrm[bb == 0, 2] = -1; nrm[bb == 1, 2] = 1
                sd = bb == 2; rad = np.sqrt(pi[h][sd, 0]**2 + pi[h][sd, 1]**2); nrm[sd, 0] = pi[h][sd, 0] / rad; nrm[sd, 1] = pi[h][sd, 1] / rad
                nout = np.where(bb == 0, n_top, np.where(bb == 1, n_bot, n_side))
                cos_i = np.abs((ui[h] * nrm).sum(1)); refl = rng.random(len(h)) < fresnel(n_in, nout, cos_i)
                uu = ui[h]; uu[refl] -= 2 * (uu[refl] * nrm[refl]).sum(1)[:, None] * nrm[refl]; ui[h] = uu
                pp = pi[h]; pp[refl] -= 1e-7 * nrm[refl]; pi[h] = pp
                out = i[h[~refl]]; np.add.at(esc, bb[~refl], w[out]); w[out] = 0; alive[out] = False
            # collision (real or null)
            c = np.where(~hit)[0]
            if len(c):
                r = np.sqrt(pi[c, 0]**2 + pi[c, 1]**2); ir = np.minimum((r / dr).astype(int), nr - 1)
                iz = np.clip((pi[c, 2] / dz).astype(int), 0, nz - 1)
                real = rng.random(len(c)) < mut[ir, iz] / mmax
                cr = c[real]; irr, izr = ir[real], iz[real]; j = i[cr]
                dw = w[j] * mua[irr, izr] / mut[irr, izr]; np.add.at(A, (irr, izr), dw); w[j] -= dw
                ui[cr] = hg_scatter(ui[cr], g[irr, izr], rng)
            p[i], u[i] = pi, ui
            low = alive & (w < 1e-4)
            if low.any():
                jl = np.where(low)[0]; surv = rng.random(len(jl)) < 0.1
                w[jl[surv]] /= 0.1; w[jl[~surv]] = 0; alive[jl[~surv]] = False
    rc = (np.arange(nr) + 0.5) * dr; vol = 2 * np.pi * rc[:, None] * dr * dz
    return dict(Q=A / N * P / vol, r=rc, z=(np.arange(nz) + 0.5) * dz, R_top=esc[0] / N, T_bot=esc[1] / N, side=esc[2] / N,
                spec=spec / N, absorbed=A.sum() / N)
