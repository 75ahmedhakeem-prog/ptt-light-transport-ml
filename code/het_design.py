"""Heterogeneity-aware protocol design at 1064 nm under the MPE for the reference tumour of Table 11
(d = 2.5 mm, Dt = 9 mm, mu_a,GNR = 0.15 mm^-1, perfusion 1e-3 1/s), exhaustive delta-P1 + Pennes search.
For each of the 16 GNR distributions of het_ptt.py (equal total gold) the full grid of beam (10-25 mm), cooling
(6 levels), irradiance (<= MPE) and heating time (120-1200 s) is simulated.  Outputs:
  (1) the optimum for each pattern (coverage >= 0.98 with the smallest collateral volume, T_skin,max <= 45 C);
  (2) one protocol that is robust to all 16 patterns (highest minimum coverage, then smallest median collateral);
  (3) the heating time needed by the Table 11 design (21 mm, h = 1000, 1.0 W/cm2) to reach 0.98 for every pattern."""
import os, sys, json, time, itertools; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import design_sim as ds, het_ptt as H
REF = dict(depth=2.5, Dt=9.0, gnr=0.15, perf=1e-3); LAM = 1064
SPOT = np.arange(10, 25.01, 1.0); HC = np.array([10, 30, 100, 200, 500, 1000.0]); IRR = np.round(np.arange(0.25, 1.0001, 0.025), 3)
TS = np.arange(120, 1201, 30.0)

def outcomes_grid(F, tum, skin, every=10.0):
    Vc = (2 * np.pi * ds.rt[:, None] * 0.25) * np.ones((1, ds.TZ)); Vt = Vc[tum].sum(); out = np.zeros((len(IRR), len(TS), 3))
    for a, I in enumerate(IRR):
        T = 37 + I * F; R_ = np.where(T >= 43, 0.5, 0.25); cem = np.cumsum(R_**(43 - T) * every / 60.0, axis=0)
        tsk = np.maximum.accumulate(T[:, skin].max(1))
        for b, t in enumerate(TS):
            i = int(round(t / every)); dmg = cem[i] >= 240
            out[a, b] = (Vc[tum & dmg].sum() / Vt, tsk[i], Vc[~tum & dmg].sum())
    return out

def run(args):
    kind, seed, ell = args; c = H.pattern(kind, REF["depth"], REF["Dt"], seed=seed, ell=ell); res = np.zeros((len(SPOT), len(HC), len(IRR), len(TS), 3))
    for i, S in enumerate(SPOT):
        Q = H.light(LAM, S, REF["depth"], REF["Dt"], REF["gnr"] * c); k, rhoc, wrc, tum, skin = ds.thermal_maps(REF["depth"], REF["Dt"], REF["perf"])
        for j, h in enumerate(HC):
            F = ds.heat_fields(Q, k, rhoc, wrc, t_end=TS[-1], h_top=h); res[i, j] = outcomes_grid(F, tum, skin)
    return f"{kind}|{seed}|{ell}", res

def best(r):
    cov, ts, vc = r[..., 0], r[..., 1], r[..., 2]; ok = ts <= 45.0; full = ok & (cov >= 0.98)
    if full.any(): idx = np.unravel_index(np.argmin(np.where(full, vc, np.inf)), vc.shape)
    else: idx = np.unravel_index(np.argmax(np.where(ok, cov, -1)), vc.shape)
    return dict(spot=float(SPOT[idx[0]]), h=float(HC[idx[1]]), I=float(IRR[idx[2]]), t=float(TS[idx[3]]), coverage=float(cov[idx]), T_skin_max=float(ts[idx]), V_collateral=float(vc[idx]))

if __name__ == "__main__":
    t0 = time.time()
    with Pool(int(os.environ.get("NPROC", 1))) as P: R = dict(P.map(run, H.KINDS))
    out = {"per_pattern": {k: best(v) for k, v in R.items()}}
    A = np.stack(list(R.values())); cov, ts, vc = A[..., 0], A[..., 1], A[..., 2]
    ok = (ts <= 45.0).all(0); mincov = np.where(ok, cov.min(0), -1); medvc = np.median(vc, 0)
    top = mincov.max(); cand = ok & (mincov >= min(0.98, top) - 1e-9)
    idx = np.unravel_index(np.argmin(np.where(cand, medvc, np.inf)), medvc.shape)
    out["robust_all_patterns"] = dict(spot=float(SPOT[idx[0]]), h=float(HC[idx[1]]), I=float(IRR[idx[2]]), t=float(TS[idx[3]]), min_coverage=float(mincov[idx]),
        median_coverage=float(np.median(cov[(slice(None),) + idx])), median_V_collateral=float(medvc[idx]), max_T_skin=float(ts[(slice(None),) + idx].max()))
    i, j, a = list(SPOT).index(21.0), list(HC).index(1000.0), list(IRR).index(1.0)
    need = {}
    for k, v in R.items():
        c = v[i, j, a, :, 0]; ok_t = np.where(c >= 0.98)[0]; need[k] = float(TS[ok_t[0]]) if len(ok_t) else None
    out["table11_design_time_needed"] = need
    uni = R["uniform|0|2.0"]; out["uniform_check"] = best(uni[:, :, :, TS <= 900])
    json.dump(out, open("../data/het_design.json", "w"), indent=1); print(json.dumps(out["robust_all_patterns"]), "\n", need, "\n", round(time.time() - t0), "s")
