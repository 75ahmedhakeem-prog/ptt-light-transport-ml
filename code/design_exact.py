"""Cost evidence for the surrogate (reviewer comment).
(1) Exhaustive exact (delta-P1 + Pennes) search of the design grid of design_opt.py for the off-grid reference tumour:
    irradiance and time are post-processing, so the grid needs one simulation per (beam, cooling) = 96 per wavelength.
(2) Robust design at 1064 nm under the MPE: perfusion x0.5-2 (log) and mu_a,GNR x0.67-1.33 around the reference tumour,
    NS scrambled-Sobol samples.  A protocol is feasible when T_skin,max <= 45 C in >= 90 % of the samples; among feasible
    protocols the one with the highest median coverage is chosen (ties: smallest median collateral volume).
    Exact search: 96 x NS simulations.  ANN search: separate networks for coverage, T_skin,max and log collateral volume.
Usage: python design_exact.py exact | robust NS"""
import os, sys, json, time, itertools; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import qmc
import design_sim as ds, ml_design as M
REF = dict(depth_mm=2.5, Dt_mm=9.0, mua_gnr=0.15, perf=1e-3)
MPE = {808: 0.2 * 10**(0.002 * 108), 1064: 1.0}
SPOT = np.arange(10, 25.01, 1.0); HC = np.array([10, 30, 100, 200, 500, 1000.0]); IRR = np.round(np.arange(0.25, 2.001, 0.025), 3); TS = np.arange(120, 901, 30.0)

def sim_table(lam, S, h, gnr, perf, irr):
    Q = ds.light(lam, S, REF["depth_mm"], REF["Dt_mm"], gnr); k, rhoc, wrc, tum, skin = ds.thermal_maps(REF["depth_mm"], REF["Dt_mm"], perf)
    F = ds.heat_fields(Q, k, rhoc, wrc, h_top=h); rows = []
    for I in irr:
        for t, o in zip(TS, ds.outcomes(F, tum, skin, I, list(TS))):
            rows.append((S, h, I, t, o["coverage"], o["T_skin_max"], o["V_collateral"]))
    return rows

def select(df):
    """design_opt.py rule: among T_skin,max <= 45, coverage >= 0.98 with the smallest collateral volume; else highest coverage"""
    ok = df.T_skin_max <= 45.0
    if not ok.any(): return None
    full = ok & (df.coverage >= 0.98)
    s = df[full].sort_values("V_collateral") if full.any() else df[ok].assign(sc=df.coverage - 1e-4 * df.I * df.t / 100).sort_values("sc", ascending=False)
    return s.iloc[0].to_dict()

if __name__ == "__main__":
    mode = sys.argv[1]; cols = ["spot_mm", "h_cool", "I", "t", "coverage", "T_skin_max", "V_collateral"]
    if mode == "exact":
        res = {}; D = json.load(open("../data/design_opt.json"))
        for lam in (808, 1064):
            t0 = time.perf_counter(); rows = [r for S, h in itertools.product(SPOT, HC) for r in sim_table(lam, S, h, REF["mua_gnr"], REF["perf"], IRR)]
            dt = time.perf_counter() - t0; E = pd.DataFrame(rows, columns=cols)
            for cons in ("skin45", "skin45+MPE"):
                e = E if cons == "skin45" else E[E.I <= MPE[lam] + 1e-9]; b = select(e); a = D.get(f"{lam}|{cons}", {})
                res[f"{lam}|{cons}"] = dict(exact_best=b, ann_choice=({k: a[k] for k in ("spot_mm", "h_cool", "I_Wcm2", "t_s", "dP1_coverage", "dP1_V_collateral")} if a.get("spot_mm") else None))
                print(lam, cons, "exact", {k: round(v, 3) for k, v in b.items()} if b else None, "| ANN", res[f"{lam}|{cons}"]["ann_choice"], flush=True)
            res[f"{lam}|cost"] = dict(n_sim=len(SPOT) * len(HC), t_total_s=dt, n_grid=len(E))
        json.dump(res, open("../data/design_exact.json", "w"), indent=1, default=float); print(res["808|cost"], res["1064|cost"])
    else:
        NS = int(sys.argv[2]); U = qmc.Sobol(2, scramble=True, seed=11).random(NS)
        kp = np.exp(np.log(0.5) + U[:, 0] * np.log(4.0)); kg = 0.667 + U[:, 1] * 0.666; lam = 1064; irr = IRR[IRR <= MPE[lam] + 1e-9]
        d = pd.read_csv("../data/design_dataset_valid.csv"); X = M.prep(d); rng = np.random.default_rng(1); idx = rng.choice(len(d), 120000, replace=False)
        t0 = time.perf_counter()
        mods = {tg: M.ann().fit(X.iloc[idx], d[tg].values[idx]) for tg in ("coverage", "T_skin_max")}; mods["logVc"] = M.ann().fit(X.iloc[idx], np.log1p(d.V_collateral.values[idx]))
        t_train = time.perf_counter() - t0
        G = pd.DataFrame(list(itertools.product(SPOT, HC, irr, TS)), columns=["spot_mm", "h_cool", "I_Wcm2", "t_s"])
        t0 = time.perf_counter(); P = []
        for a, b in zip(kp, kg):
            g = G.assign(depth_mm=REF["depth_mm"], Dt_mm=REF["Dt_mm"], mua_gnr=REF["mua_gnr"] * b, perf=REF["perf"] * a, lam_nm=lam); XG = M.prep(g)
            P.append(np.c_[np.clip(mods["coverage"].predict(XG), 0, 1), mods["T_skin_max"].predict(XG), np.expm1(mods["logVc"].predict(XG))])
        P = np.array(P); t_ann = time.perf_counter() - t0
        def pick(cov, ts, vc):
            pok = (ts <= 45.0).mean(0); med = np.median(cov, 0); mvc = np.median(vc, 0); feas = pok >= 0.9
            sc = np.where(feas, med - 1e-6 * mvc, -1); i = int(np.argmax(sc)); return i, dict(p_ok=float(pok[i]), cov_med=float(med[i]), cov_p05=float(np.percentile(cov[:, i], 5)), vc_med=float(mvc[i]))
        ia, sa = pick(P[:, :, 0], P[:, :, 1], P[:, :, 2]); print("ANN robust", G.iloc[ia].to_dict(), sa, round(t_ann, 1), "s", flush=True)
        t0 = time.perf_counter(); EX = []
        for a, b in zip(kp, kg):
            rows = [r for S, h in itertools.product(SPOT, HC) for r in sim_table(lam, S, h, REF["mua_gnr"] * b, REF["perf"] * a, irr)]
            EX.append(np.array([[r[4], r[5], r[6]] for r in rows]))
        t_ex = time.perf_counter() - t0; EX = np.array(EX)            # (NS, n_grid, 3), same ordering as G
        ie, se = pick(EX[:, :, 0], EX[:, :, 1], EX[:, :, 2]); _, sae = pick(EX[:, [ia], 0], EX[:, [ia], 1], EX[:, [ia], 2])
        res = dict(NS=NS, t_train_ann_s=t_train, t_ann_search_s=t_ann, n_ann_pred=int(P.shape[0] * P.shape[1] * 3), t_exact_search_s=t_ex, n_exact_sim=NS * len(SPOT) * len(HC),
                   ann=dict(G.iloc[ia].to_dict(), **sa), exact=dict(G.iloc[ie].to_dict(), **se), ann_choice_exact=sae)
        print("exact robust", res["exact"], "\nANN choice evaluated exactly", sae, round(t_ex), "s", flush=True)
        json.dump(res, open("../data/robust_design.json", "w"), indent=1, default=float)
