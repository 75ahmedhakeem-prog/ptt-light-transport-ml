"""Stages C-D on the excess temperature (GNR phantom minus plain gel, same filter): superposition removes the
background heating and the thermocouple artefact common to both. Calibrate h on NR1, predict the others blindly."""
import numpy as np, pandas as pd, json, mc, heat
from scipy.optimize import least_squares
water = {"720-870": 0.003, "800-1000": 0.020}; filt = {"720-870": "filter1_720-870", "800-1000": "filter2_800-1000"}
S = {"NR1": (0.025, 0.035, "720-870"), "NR1+4%BSA": (0.055, 0.065, "800-1000"), "NR2": (0.035, 0.205, "800-1000"), "NR2+4%BSA": (0.033, 0.375, "800-1000")}
Rb, P, g, n = 6.5, 0.013 * np.pi * 6.5**2, 0.8, 1.33
ph = heat.Phantom(); Z = (1, 5, 10, 15); probes = [(0.25e-3, z * 1e-3) for z in Z]
dT = pd.read_csv("../data/pratap_fig56_dT.csv"); pg = pd.read_csv("../data/pratap_fig4_plaingel_dT.csv")
def coarse(Q, r):   # MC grid 44x80 (0.25 mm) -> 22x40 (0.5 mm), W/mm^3 -> W/m^3
    vol = (2 * np.pi * r[:, None] * 0.25 * 0.25); E = (Q * vol).reshape(22, 2, 40, 2).sum((1, 3))
    return E / (2 * np.pi * ph.r[:, None] * 1e3 * 0.25) * 1e9
def sources(nm):
    mua, mus, band = S[nm]; w = water[band]; out = {}
    gl = mc.Cylinder(11.0, 20.0, [(0, 10, mua, mus, g, n), (10, 20, w, 0.01, 0.9, n)], n_out=1.0)
    o = mc.run_cylinder(gl, Rb, P=P, N=600000, nr=44, nz=80, seed=7)
    pl = mc.run_cylinder(mc.Cylinder(11.0, 20.0, [(0, 20, w, 0.01, 0.9, n)], n_out=1.0), Rb, P=P, N=600000, nr=44, nz=80, seed=8)
    out["MC"] = coarse(o["Q"] - pl["Q"], o["r"])
    # Beer-Lambert (corrected: mu_a only) and published form (mu_a + mu_s), same beam, minus plain-gel BL
    r = ph.r * 1e3; z = ph.z * 1e3; E0 = 0.013 * (1 - ((1 - n) / (1 + n))**2) * (r[:, None] <= Rb)
    def bl(ma, ms, heat_mu, ma2):
        att = np.where(z < 10, np.exp(-(ma + ms) * z), np.exp(-(ma + ms) * 10 - (ma2 + 0.01) * (z - 10)))
        return E0 * np.where(z < 10, heat_mu, ma2) * att * 1e9
    base = bl(w, 0.01, w, w)
    out["BL"] = bl(mua, mus, mua, w) - base
    out["BLpub"] = np.where(z < 10, bl(mua, mus, mua + mus, 0.0), 0) - base
    return out
def excess(nm):
    band = S[nm][2]; cols = []
    for z in Z:
        a = dT[(dT["sample"] == nm) & (dT.depth_mm == z)].sort_values("t_min"); b = pg[(pg["filter"] == filt[band]) & (pg.depth_mm == z)].sort_values("t_min")
        cols.append(a.dT_C.values - np.interp(a.t_min.values, b.t_min.values, b.dT_C.values))
    T = np.array(cols).T; ok = np.isfinite(T).all(1)
    return a.t_min.values[ok] * 60, T[ok]
SRC = {nm: sources(nm) for nm in S}; EXC = {nm: excess(nm) for nm in S}
def model(Q, h, hb):
    return ph.run(Q, h, h, hb, probes=probes)
def resid(x, nm, key="MC"):
    t, T = model(SRC[nm][key], *x); te, Te = EXC[nm]
    return np.concatenate([np.interp(te, t, T[:, j]) - Te[:, j] for j in range(4)])
fit = least_squares(resid, [20, 50], args=("NR1",), bounds=([1, 1], [500, 3000])); h, hb = fit.x
res = {"h": h, "h_bottom": hb, "rmse": {}, "end_of_heating": {}}
for nm in S:
    te, Te = EXC[nm]; i15 = np.argmin(abs(te - 900))
    res["end_of_heating"][nm] = {"measured_excess": Te[i15].round(2).tolist()}
    for key in ("MC", "BL", "BLpub"):
        r = resid((h, hb), nm, key); res["rmse"].setdefault(nm, {})[key] = float(np.sqrt((r**2).mean()))
        t, T = model(SRC[nm][key], h, hb); res["end_of_heating"][nm][key] = np.interp(900, t, T[:, 0]).round(2).tolist() if False else [round(float(np.interp(900, t, T[:, j])), 2) for j in range(4)]
print(json.dumps(res, indent=1)); json.dump(res, open("../data/stageCD_excess.json", "w"), indent=1)
import pickle; pickle.dump(dict(SRC=SRC, EXC=EXC, h=h, hb=hb), open("../data/stageCD.pkl", "wb"))
