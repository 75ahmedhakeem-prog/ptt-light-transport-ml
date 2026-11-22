"""Heterogeneous gold-nanorod distributions at equal total gold (reviewer comment): effect on tumour coverage
(CEM43 >= 240 min), on the treatability maps (Fig. 9 / 10 set-up: mu_a,GNR = 0.1 mm^-1, perfusion 1e-3 1/s) and on the
ANN-designed protocols of Table 10.
The GNR absorption in the tumour is mu_a,GNR * c(r, z), where the relative concentration c has a volume-weighted tumour
mean of 1.  Patterns: rim (e-fold length 0.75 mm from the surface), necrotic core (no gold for rho < R/2), central
(intratumoural injection, Gaussian with sigma = 0.35 R), gradients of +-80 % towards / away from the beam, and log-normal
random fields (sigma_ln = 1; correlation length 0.5 or 2 mm; axisymmetric, i.e. ring-shaped).  delta-P1 + Pennes."""
import os, sys, json, time, itertools; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool
import design_sim as ds, dp1

MPE = {808: 0.2 * 10**(0.002 * 108), 1064: 1.0}
SPOT = [10, 15, 20, 25]; HC = [10.0, 200.0, 1000.0]; IRR = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]; TIMES = [120, 300, 450, 600, 900]
DEPTH = [0, 1, 2, 3, 4, 5]; DT = [5.0, 7.5, 10.0, 12.5, 15.0]; GNR = 0.1; PERF = 1e-3
NREAL = 5

def pattern(kind, depth, Dt, seed=0, sig=1.0, ell=2.0):
    tum, _ = ds.regions(ds.rc, ds.zc, depth, Dt); zc0 = ds.SKIN + depth + Dt / 2; R = Dt / 2
    rho = np.sqrt(ds.rc[:, None]**2 + (ds.zc[None, :] - zc0)**2) / R; dz = (ds.zc[None, :] - zc0) / R * np.ones_like(rho)
    V = 2 * np.pi * ds.rc[:, None] * ds.DR * ds.DZ * np.ones_like(rho)
    if kind == "uniform": c = np.ones_like(rho)
    elif kind == "rim": c = np.exp(-(1 - rho) * R / 0.75)
    elif kind == "shell": c = (rho >= 0.5).astype(float)
    elif kind == "centre": c = np.exp(-rho**2 / (2 * 0.35**2))
    elif kind == "light-side": c = 1 - 0.8 * dz
    elif kind == "far-side": c = 1 + 0.8 * dz
    elif kind == "lognormal":
        rng = np.random.default_rng(seed); w = gaussian_filter(rng.standard_normal(rho.shape), ell / ds.DR, mode="nearest")
        w = (w - w[tum].mean()) / w[tum].std(); c = np.exp(sig * w)
    c = np.where(tum, np.maximum(c, 0), 0.0); return c / ((c * V)[tum].sum() / V[tum].sum())

def light(lam, spot, depth, Dt, gmap):
    mua, mus, g, _ = ds.optics(lam, depth, Dt, 0.0); mua = mua + gmap
    o = dp1.solve(ds.R, ds.H, ds.NR, ds.NZ, mua, mus, g, spot / 2, P=0.01 * np.pi * (spot / 2)**2, n_in=1.4)
    vol = 2 * np.pi * ds.rc[:, None] * ds.DR * ds.DZ; E = (o["Q"] * vol).reshape(ds.TR, 2, ds.TZ, 2).sum((1, 3))
    return E / (2 * np.pi * ds.rt[:, None] * 0.25) * 1e9

def frames(lam, spot, h, depth, Dt, gnr, perf, c):
    Q = light(lam, spot, depth, Dt, gnr * c); k, rhoc, wrc, tum, skin = ds.thermal_maps(depth, Dt, perf)
    return ds.heat_fields(Q, k, rhoc, wrc, h_top=h), tum, skin

def best_cov(args):
    """best coverage of one tumour over beam, cooling, irradiance and time (dataset grid), with and without the MPE"""
    lam, depth, Dt, kind, seed, ell = args
    c = pattern(kind, depth, Dt, seed=seed, ell=ell); best = {"skin45": 0.0, "skin45+MPE": 0.0}
    for S, h in itertools.product(SPOT, HC):
        F, tum, skin = frames(lam, S, h, depth, Dt, GNR, PERF, c)
        for I in IRR:
            for o in ds.outcomes(F, tum, skin, I, TIMES):
                if o["T_tum_max"] > 100 or o["T_skin_max"] > 45.0: continue
                best["skin45"] = max(best["skin45"], o["coverage"])
                if I <= MPE[lam] + 1e-9: best["skin45+MPE"] = max(best["skin45+MPE"], o["coverage"])
    return dict(lam=lam, depth=depth, Dt=Dt, kind=kind, seed=seed, ell=ell, **best)

def design_check(kind, seed=0, ell=2.0):
    """coverage of the Table 10 designs (d = 2.5 mm, Dt = 9 mm, mu_a,GNR = 0.15, perfusion 1e-3)"""
    D = json.load(open("../data/design_opt.json")); out = {}
    for key in ("808|skin45", "1064|skin45+MPE"):
        b = D[key]; lam = int(key.split("|")[0]); c = pattern(kind, 2.5, 9.0, seed=seed, ell=ell)
        F, tum, skin = frames(lam, b["spot_mm"], b["h_cool"], 2.5, 9.0, 0.15, 1e-3, c)
        o = ds.outcomes(F, tum, skin, b["I_Wcm2"], [b["t_s"]])[0]
        # irradiance needed to restore coverage >= 0.98 at the same beam, cooling and time (skin <= 45 C)
        Ireq = None
        for I in np.arange(0.25, 3.001, 0.025):
            oo = ds.outcomes(F, tum, skin, I, [b["t_s"]])[0]
            if oo["T_skin_max"] > 45: break
            if oo["coverage"] >= 0.98: Ireq = float(I); break
        out[key] = dict(coverage=o["coverage"], T_skin_max=o["T_skin_max"], T_tum_min=o["T_tum_min"], V_collateral=o["V_collateral"], I_for_098=Ireq)
    return out

KINDS = [("uniform", 0, 2.0), ("rim", 0, 2.0), ("shell", 0, 2.0), ("centre", 0, 2.0), ("light-side", 0, 2.0), ("far-side", 0, 2.0)] + \
        [("lognormal", s, ell) for ell in (0.5, 2.0) for s in range(NREAL)]
if __name__ == "__main__":
    t0 = time.time(); res = {"design": {}, "maps": []}
    for kind, s, ell in KINDS: res["design"][f"{kind}|{s}|{ell}"] = design_check(kind, s, ell)
    print("designs done", round(time.time() - t0), flush=True)
    jobs = [(lam, dep, Dt, kind, s, ell) for kind, s, ell in KINDS for lam in (808, 1064) for dep in DEPTH for Dt in DT]
    with Pool(int(os.environ.get("NPROC", 1))) as P:
        for i, r in enumerate(P.imap_unordered(best_cov, jobs, chunksize=4)):
            res["maps"].append(r)
            if i % 60 == 0: print(i, len(jobs), round(time.time() - t0), flush=True); json.dump(res, open("../data/het_ptt.json", "w"), indent=0)
    json.dump(res, open("../data/het_ptt.json", "w"), indent=0); print("done", round(time.time() - t0))
