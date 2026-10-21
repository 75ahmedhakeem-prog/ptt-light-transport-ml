import numpy as np, json, pickle, dp1, heat, heat2
res = {}
# ---------- dataset 1: excess temperature with the NR1-calibrated h (same as MC)
src = open("stageCD_excess.py").read().split("SRC = {nm")[0]; exec(src)
d = pickle.load(open("../data/stageCD.pkl", "rb")); h, hb, EXC, SRC = d["h"], d["hb"], d["EXC"], d["SRC"]
def maps1(mua1, mus1, w):
    nr, nz = 44, 80; z = (np.arange(nz) + 0.5) * 0.25
    mua = np.where(z[None, :] < 10, mua1, w) * np.ones((nr, 1)); mus = np.where(z[None, :] < 10, mus1, 0.01) * np.ones((nr, 1))
    g_ = np.where(z[None, :] < 10, g, 0.9) * np.ones((nr, 1)); return mua, mus, g_
r1 = {}
for nm, (mua1, mus1, band) in S.items():
    w = water[band]
    a = dp1.solve(11.0, 20.0, 44, 80, *maps1(mua1, mus1, w), Rb, P=P, n_in=n)
    b = dp1.solve(11.0, 20.0, 44, 80, *maps1(w, 0.01, w), Rb, P=P, n_in=n)
    Q = coarse(a["Q"] - b["Q"], a["r"]); t, T = ph.run(Q, h, h, hb, probes=probes); te, Te = EXC[nm]
    rr = np.concatenate([np.interp(te, t, T[:, j]) - Te[:, j] for j in range(4)])
    vol = 2 * np.pi * a["r"][:, None] * 0.25 * 0.25; zz = a["z"]
    QMC = SRC[nm]["MC"]
    r1[nm] = dict(rmse_dp1=round(float(np.sqrt((rr**2).mean())), 2), tumour_W_dp1=round(float((a["Q"] * vol)[:, zz < 10].sum()), 3))
    print("dataset1", nm, r1[nm])
res["dataset1"] = r1
# ---------- dataset 3
src3 = open("stageD_shaw.py").read().split("res = {}")[0]; exec(src3)
r3 = {}
for case in ("Case-1", "Case-2"):
    for label, mg in (("damped", 0.40), ("undamped", 0.71)):
        mua, mus, gg = maps(case, mg)
        o = dp1.solve(R, H, nr, nz, mua, mus, gg, 10.0, P=P, n_in=1.37)
        t, T = heat2.run(Rm, Hm, tr, tz, k, rhoc, coarse(o["Q"]), 10.0, 10.0, 10.0, t_on=1200, t_end=1200, probes=probes)
        err = []
        for j, p in enumerate(probes_mm):
            e = exp[(exp["case"] == case) & (exp.probe_r_z_mm == p)].dropna(); err.append(np.interp(e.t_s, t, 25 + T[:, j]) - e.T_C)
        err = np.concatenate(err); r3[f"{case}|{label}"] = dict(rmse=round(float(np.sqrt((err**2).mean())), 2), absorbed=round(o["absorbed"], 4),
                                                              T1200={p: round(25 + float(T[-1, j]), 2) for j, p in enumerate(probes_mm)})
        print("dataset3", case, label, r3[f"{case}|{label}"]["rmse"], r3[f"{case}|{label}"]["absorbed"])
res["dataset3"] = r3; json.dump(res, open("../data/stage_dp1_compare.json", "w"), indent=1)
