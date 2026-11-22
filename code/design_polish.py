"""Surrogate-assisted design with exact refinement: the ANN ranks the full design grid, and only the K best distinct
(beam, cooling) pairs are simulated exactly with delta-P1 (each simulation covers all irradiances and times).
(1) single-tumour design (design_opt.py set-up, Table 11) and (2) robust design at 1064 nm under the MPE
(design_exact.py robust set-up, same 64 Sobol samples).  Results are compared with the exhaustive exact optima."""
import os, sys, json, time, itertools; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import qmc
import design_sim as ds, ml_design as M, design_exact as E
REF, MPE, SPOT, HC, IRR, TS = E.REF, E.MPE, E.SPOT, E.HC, E.IRR, E.TS
COLS = ["spot_mm", "h_cool", "I", "t", "coverage", "T_skin_max", "V_collateral"]

def topk_pairs(G, order, K):
    pairs = []
    for i in order:
        p = (G.spot_mm.iat[i], G.h_cool.iat[i])
        if p not in pairs: pairs.append(p)
        if len(pairs) == K: break
    return pairs

if __name__ == "__main__":
    d = pd.read_csv("../data/design_dataset_valid.csv"); X = M.prep(d); rng = np.random.default_rng(1); idx = rng.choice(len(d), 120000, replace=False)
    mods = {tg: M.ann().fit(X.iloc[idx], d[tg].values[idx]) for tg in ("coverage", "T_skin_max")}; mods["logVc"] = M.ann().fit(X.iloc[idx], np.log1p(d.V_collateral.values[idx]))
    EX = json.load(open("../data/design_exact.json")); RB = json.load(open("../data/robust_design.json")); out = {"single": {}, "robust": {}}
    # (1) single tumour
    for lam, cons in ((808, "skin45"), (1064, "skin45+MPE")):
        irr = IRR if cons == "skin45" else IRR[IRR <= MPE[lam] + 1e-9]
        G = pd.DataFrame(list(itertools.product(SPOT, HC, irr, TS)), columns=["spot_mm", "h_cool", "I_Wcm2", "t_s"])
        XG = M.prep(G.assign(depth_mm=REF["depth_mm"], Dt_mm=REF["Dt_mm"], mua_gnr=REF["mua_gnr"], perf=REF["perf"], lam_nm=lam))
        cov = np.clip(mods["coverage"].predict(XG), 0, 1); ts = mods["T_skin_max"].predict(XG); vc = np.expm1(mods["logVc"].predict(XG))
        ok = ts <= 45.0; full = ok & (cov >= 0.98); score = np.where(full, 10 - vc / 1e4, np.where(ok, cov, -1)); order = np.argsort(-score)
        for K in (1, 3, 5):
            pairs = topk_pairs(G, order, K); t0 = time.perf_counter()
            rows = [r for S, h in pairs for r in E.sim_table(lam, S, h, REF["mua_gnr"], REF["perf"], irr)]
            b = E.select(pd.DataFrame(rows, columns=COLS)); ex = EX[f"{lam}|{cons}"]["exact_best"]
            out["single"][f"{lam}|{cons}|K{K}"] = dict(pairs=[list(map(float, p)) for p in pairs], n_sim=len(pairs), t_s=time.perf_counter() - t0, best=b,
                                                       exhaustive=ex, gap_Vc=(b["V_collateral"] / ex["V_collateral"] - 1) if b["coverage"] >= 0.98 else None)
            print(lam, cons, "K", K, {k: round(v, 3) for k, v in b.items() if k in COLS}, "exhaustive Vc", round(ex["V_collateral"], 1), flush=True)
    # (2) robust design, 1064 nm under the MPE
    NS = RB["NS"]; U = qmc.Sobol(2, scramble=True, seed=11).random(NS); kp = np.exp(np.log(0.5) + U[:, 0] * np.log(4.0)); kg = 0.667 + U[:, 1] * 0.666
    lam = 1064; irr = IRR[IRR <= MPE[lam] + 1e-9]
    G = pd.DataFrame(list(itertools.product(SPOT, HC, irr, TS)), columns=["spot_mm", "h_cool", "I_Wcm2", "t_s"]); P = []
    for a, b in zip(kp, kg):
        XG = M.prep(G.assign(depth_mm=REF["depth_mm"], Dt_mm=REF["Dt_mm"], mua_gnr=REF["mua_gnr"] * b, perf=REF["perf"] * a, lam_nm=lam))
        P.append(np.c_[np.clip(mods["coverage"].predict(XG), 0, 1), mods["T_skin_max"].predict(XG), np.expm1(mods["logVc"].predict(XG))])
    P = np.array(P); pok = (P[:, :, 1] <= 45.0).mean(0); med = np.median(P[:, :, 0], 0); mvc = np.median(P[:, :, 2], 0)
    sc = np.where(pok >= 0.9, med - 1e-6 * mvc, -1); order = np.argsort(-sc)
    for K in (1, 3, 5):
        pairs = topk_pairs(G, order, K); t0 = time.perf_counter(); EXs = []
        for a, b in zip(kp, kg):
            rows = [r for S, h in pairs for r in E.sim_table(lam, S, h, REF["mua_gnr"] * b, REF["perf"] * a, irr)]; EXs.append(np.array([[r[4], r[5], r[6]] for r in rows]))
        EXs = np.array(EXs); grid = pd.DataFrame([(S, h, I, t) for S, h in pairs for I in irr for t in TS], columns=["spot_mm", "h_cool", "I_Wcm2", "t_s"])
        po = (EXs[:, :, 1] <= 45.0).mean(0); md = np.median(EXs[:, :, 0], 0); mv = np.median(EXs[:, :, 2], 0)
        s2 = np.where(po >= 0.9, md - 1e-6 * mv, -1); i = int(np.argmax(s2))
        out["robust"][f"K{K}"] = dict(pairs=[list(map(float, p)) for p in pairs], n_sim=NS * len(pairs), t_s=time.perf_counter() - t0,
                                       best=dict(grid.iloc[i].to_dict(), p_ok=float(po[i]), cov_med=float(md[i]), vc_med=float(mv[i])), exhaustive=RB["exact"])
        print("robust K", K, out["robust"][f"K{K}"]["best"], "exhaustive", {k: RB["exact"][k] for k in ("spot_mm", "I_Wcm2", "t_s", "p_ok", "vc_med")}, flush=True)
    json.dump(out, open("../data/design_polish.json", "w"), indent=1, default=float)
