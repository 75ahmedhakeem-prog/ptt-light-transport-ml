"""Stage C: calibrate h and the background source on the plain-gel curves (dataset 1)."""
import numpy as np, pandas as pd, json, mc, heat
from scipy.optimize import least_squares
water = {"filter1_720-870": 0.003, "filter2_800-1000": 0.020}
Rb, P = 6.5, 0.013 * np.pi * 6.5**2
ph = heat.Phantom(); probes = [(0.25e-3, z * 1e-3) for z in (1, 5, 10, 15)]
def mc_source(layers, seed=5):
    geo = mc.Cylinder(11.0, 20.0, layers, n_out=1.0)
    o = mc.run_cylinder(geo, Rb, P=P, N=400000, nr=44, nz=80, seed=seed)
    Q = o["Q"]; Q = Q.reshape(22, 2, 40, 2); vol = (2 * np.pi * o["r"][:, None] * 0.25 * 0.25).reshape(22, 2, 1, 1)
    Qc = (Q * vol).sum((1, 3)) / ((2 * np.pi * ph.r[:, None] * 1e3 * 0.5 * 0.5))   # W/mm^3 on coarse grid
    return Qc * 1e9                                                              # W/m^3
pg = pd.read_csv("../data/pratap_fig4_plaingel_dT.csv")
src = {f: mc_source([(0, 20, mu, 0.01, 0.9, 1.33)]) for f, mu in water.items()}
def sim(Qw, h, hb, qbg):
    t, T = ph.run(Qw + qbg, h, h, hb, probes=probes); return t, T
def resid(x, filters):
    h, hb, qbg = x; out = []
    for f in filters:
        t, T = sim(src[f], h, hb, qbg * 1e3)
        for j, z in enumerate((1, 5, 10, 15)):
            e = pg[(pg["filter"] == f) & (pg.depth_mm == z)]
            out.append(np.interp(t, e.t_min * 60, e.dT_C) - T[:, j])
    return np.concatenate(out)
fit = least_squares(resid, [10, 10, 5], args=(["filter2_800-1000"],), bounds=([1, 1, 0], [200, 2000, 100]))
h, hb, qbg = fit.x; r2 = resid(fit.x, ["filter2_800-1000"]); r1 = resid(fit.x, ["filter1_720-870"])
out = dict(h_W_m2K=h, h_bottom_W_m2K=hb, Q_bg_kW_m3=qbg, rmse_filter2=float(np.sqrt((r2**2).mean())), rmse_filter1_prediction=float(np.sqrt((r1**2).mean())),
           water_absorbed_W={f: float((s * ph.V).sum()) for f, s in src.items()}, bg_W=float(qbg * 1e3 * ph.V.sum() ))
print(json.dumps(out, indent=1)); json.dump(out, open("../data/stageC_plaingel_fit.json", "w"), indent=1)
