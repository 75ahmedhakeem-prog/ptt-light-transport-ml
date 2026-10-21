"""ANN design of the laser protocol for an off-grid reference tumour at 808 and 1064 nm, verified with delta-P1 and MC."""
import numpy as np, pandas as pd, json, itertools, time
import ml_design as M, design_sim as ds, mc_het
d = pd.read_csv("../data/design_dataset_valid.csv"); X = M.prep(d); rng = np.random.default_rng(1)
idx = rng.choice(len(d), 120000, replace=False)
models = {tg: M.ann().fit(X.iloc[idx], d[tg].values[idx]) for tg in ("coverage", "T_skin_max", "T_tum_min")}
models["logVc"] = M.ann().fit(X.iloc[idx], np.log1p(d.V_collateral.values[idx]))
REF = dict(depth_mm=2.5, Dt_mm=9.0, mua_gnr=0.15, perf=1e-3)          # off-grid reference tumour
MPE = {808: 0.2 * 10**(0.002 * 108), 1064: 1.0}
grid = dict(spot_mm=np.arange(10, 25.01, 1.0), h_cool=np.array([10, 30, 100, 200, 500, 1000.0]), I_Wcm2=np.arange(0.25, 2.001, 0.025), t_s=np.arange(120, 901, 30.0))
res = {}
for lam in (808, 1064):
    for cons in ("skin45", "skin45+MPE"):
        G = pd.DataFrame(list(itertools.product(grid["spot_mm"], grid["h_cool"], grid["I_Wcm2"], grid["t_s"])), columns=["spot_mm", "h_cool", "I_Wcm2", "t_s"])
        if cons == "skin45+MPE": G = G[G.I_Wcm2 <= MPE[lam] + 1e-9]
        for k, v in REF.items(): G[k] = v
        G["lam_nm"] = lam; XG = M.prep(G)
        cov = np.clip(models["coverage"].predict(XG), 0, 1); ts = models["T_skin_max"].predict(XG)
        ok = ts <= 45.0
        if not ok.any(): res[f"{lam}|{cons}"] = dict(feasible=False); continue
        # best coverage; ties broken by lower energy dose I*t
        vc = np.expm1(models["logVc"].predict(XG))
        full = ok & (cov >= 0.98)
        score = np.where(full, 10 - vc / 1e4, np.where(ok, cov - 1e-4 * G.I_Wcm2.values * G.t_s.values / 100, -1))
        i = int(np.argmax(score)); best = G.iloc[i].to_dict(); best["pred_coverage"] = float(cov[i]); best["pred_T_skin_max"] = float(ts[i]); best["pred_V_collateral"] = float(vc[i])
        # verification with delta-P1 (exact simulator) at the selected protocol
        Q = ds.light(lam, best["spot_mm"], REF["depth_mm"], REF["Dt_mm"], REF["mua_gnr"])
        k_, rhoc, wrc, tum, skin = ds.thermal_maps(REF["depth_mm"], REF["Dt_mm"], REF["perf"])
        F = ds.heat_fields(Q, k_, rhoc, wrc, h_top=best["h_cool"]); o = ds.outcomes(F, tum, skin, best["I_Wcm2"], [best["t_s"]])[0]
        best["dP1_coverage"], best["dP1_T_skin_max"], best["dP1_V_collateral"] = o["coverage"], o["T_skin_max"], o["V_collateral"]
        # verification with Monte Carlo light transport + the same heat model
        mua, mus, g, _ = ds.optics(lam, REF["depth_mm"], REF["Dt_mm"], REF["mua_gnr"]); P = 0.01 * np.pi * (best["spot_mm"] / 2)**2
        m = mc_het.run(ds.R, ds.H, ds.NR, ds.NZ, mua, mus, g, best["spot_mm"] / 2, P=P, n_in=1.4, n_top=1.0, n_side=1.4, n_bot=1.4, N=40000, seed=9)
        vol = 2 * np.pi * ds.rc[:, None] * ds.DR * ds.DZ; E = (m["Q"] * vol).reshape(ds.TR, 2, ds.TZ, 2).sum((1, 3))
        Qmc = E / (2 * np.pi * ds.rt[:, None] * 0.25) * 1e9
        F2 = ds.heat_fields(Qmc, k_, rhoc, wrc, h_top=best["h_cool"]); o2 = ds.outcomes(F2, tum, skin, best["I_Wcm2"], [best["t_s"]])[0]
        best["MC_coverage"], best["MC_T_skin_max"] = o2["coverage"], o2["T_skin_max"]
        res[f"{lam}|{cons}"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()}
        print(lam, cons, res[f"{lam}|{cons}"], flush=True)
json.dump(res, open("../data/design_opt.json", "w"), indent=1)
