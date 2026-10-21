"""Stage D (blind test 2): subsurface-tumour phantoms of Shaw et al. (2024), Case-1 (uniform) and Case-2 (peripheral ring).
No parameter is fitted to these data.  GNR absorption from Stage A (Gans, band 775-825 nm) bounds the prediction."""
import numpy as np, pandas as pd, json, mc_het, heat2
R, H, nr, nz = 20.0, 12.0, 80, 48                 # mm, MC grid 0.25 mm
dr, dz = R / nr, H / nz; rc = (np.arange(nr) + 0.5) * dr; zc = (np.arange(nz) + 0.5) * dz
g = 0.8; P = 0.25e-2 * np.pi * 10.0**2           # 0.25 W/cm^2 = 0.0025 W/mm^2 over a 20 mm spot
tis = (0.0035, 1.01); tum = (0.007, 0.46); gnr_mus_red = 0.0047
inT = (zc[None, :] >= 3) & (zc[None, :] <= 9) & (rc[:, None] <= 10)
ring = inT & (rc[:, None] > 7)
def maps(case, mua_gnr):
    mua = np.full((nr, nz), tis[0]); musr = np.full((nr, nz), tis[1])
    mua[inT] = tum[0]; musr[inT] = tum[1]
    G = inT if case == "Case-1" else ring
    mua[G] += mua_gnr; musr[G] += gnr_mus_red
    return mua, musr / (1 - g), np.full((nr, nz), g)
exp = pd.read_csv("../data/shaw_fig4_TE.csv"); probes_mm = {"(0,3)": (0, 3), "(10,3)": (10, 3), "(15,3)": (15, 3), "(0,6)": (0, 6), "(0,9)": (0, 9), "(10,9)": (10, 9)}
# thermal grid 0.5 mm
tr, tz = 40, 24; Rm, Hm = R * 1e-3, H * 1e-3
rr = (np.arange(tr) + 0.5) * 0.5; zz = (np.arange(tz) + 0.5) * 0.5
inT2 = (zz[None, :] >= 3) & (zz[None, :] <= 9) & (rr[:, None] <= 10)
k = np.where(inT2, 0.55, 0.5); rhoc = np.where(inT2, 1100 * 4200, 1000 * 4200.0)
def coarse(Q):
    vol = 2 * np.pi * rc[:, None] * dr * dz; E = (Q * vol).reshape(tr, 2, tz, 2).sum((1, 3))
    return E / (2 * np.pi * rr[:, None] * 0.5 * 0.5) * 1e9
probes = [(max(p[0], 0.25) * 1e-3, p[1] * 1e-3) for p in probes_mm.values()]
res = {}
for case in ("Case-1", "Case-2"):
    for label, mg in (("damped", 0.40), ("undamped", 0.71)):
        mua, mus, gg = maps(case, mg)
        o = mc_het.run(R, H, nr, nz, mua, mus, gg, 10.0, P=P, n_in=1.37, n_top=1.0, n_side=1.5, n_bot=1.5, N=300000, seed=11)
        Q = coarse(o["Q"])
        for h in (5.0, 10.0, 20.0):
            t, T = heat2.run(Rm, Hm, tr, tz, k, rhoc, Q, h, h, h, t_on=1200, t_end=1200, probes=probes)
            sim = {p: (t, 25 + T[:, j]) for j, p in enumerate(probes_mm)}
            err = []
            for p in probes_mm:
                e = exp[(exp["case"] == case) & (exp.probe_r_z_mm == p)].dropna()
                err.append(np.interp(e.t_s, *sim[p]) - e.T_C)
            err = np.concatenate(err)
            res[f"{case}|{label}|h={h:g}"] = dict(rmse=float(np.sqrt((err**2).mean())), bias=float(err.mean()), absorbed=float(o["absorbed"]),
                T1200={p: round(float(sim[p][1][-1]), 2) for p in probes_mm})
        print(case, label, {kk: round(v["rmse"], 2) for kk, v in res.items() if kk.startswith(f"{case}|{label}")})
meas = {case: {p: round(float(exp[(exp["case"] == case) & (exp.probe_r_z_mm == p)].dropna().T_C.values[-1]), 2) for p in probes_mm} for case in ("Case-1", "Case-2")}
res["measured_T1200"] = meas
# Shaw's own simulation (dashed lines) read at 1200 s from their Fig. 4, for comparison
json.dump(res, open("../data/stageD_shaw.json", "w"), indent=1)
print(json.dumps(meas)); print(json.dumps({k: v["T1200"] for k, v in res.items() if "h=10" in k}, indent=0))

# ---- Beer-Lambert comparison on the same thermal model (reduced attenuation mu_a + mu_s', heat = mu_a * E)
blres = {}
for case in ("Case-1", "Case-2"):
    mua, mus, gg = maps(case, 0.40); musr = mus * (1 - gg)
    E0 = 0.0025 * (1 - ((1 - 1.37) / (2.37))**2)
    att = np.exp(-np.cumsum((mua + musr) * dz, axis=1) + (mua + musr) * dz / 2)
    Qbl = np.where(rc[:, None] <= 10, mua * E0 * att, 0.0)        # W/mm^3
    Q = coarse(Qbl)
    t, T = heat2.run(Rm, Hm, tr, tz, k, rhoc, Q, 10.0, 10.0, 10.0, t_on=1200, t_end=1200, probes=probes)
    err = []
    for j, p in enumerate(probes_mm):
        e = exp[(exp["case"] == case) & (exp.probe_r_z_mm == p)].dropna(); err.append(np.interp(e.t_s, t, 25 + T[:, j]) - e.T_C)
    err = np.concatenate(err); blres[case] = dict(rmse=float(np.sqrt((err**2).mean())), T1200={p: round(25 + float(T[-1, j]), 2) for j, p in enumerate(probes_mm)})
print("Beer-Lambert (h=10, damped):", json.dumps(blres))
res["BeerLambert_h10_damped"] = blres; json.dump(res, open("../data/stageD_shaw.json", "w"), indent=1)
