import pandas as pd, numpy as np, json
d = pd.read_csv("../data/design_dataset_valid.csv")
MPE = {808: 0.2 * 10**(0.002 * (808 - 700)), 1064: 1.0}
print("MPE", {k: round(v, 3) for k, v in MPE.items()})
out = {}
for perf in (1e-3,):
    s = d[np.isclose(d.perf, perf)]
    for cons in ("skin45", "skin45+MPE"):
        ok = s.T_skin_max <= 45.0
        if cons == "skin45+MPE": ok &= s.I_Wcm2 <= s.lam_nm.map(MPE)
        best = s[ok].groupby(["lam_nm", "mua_gnr", "depth_mm", "Dt_mm"]).coverage.max().unstack("lam_nm").fillna(0)
        for g in sorted(s.mua_gnr.unique()):
            b = best.xs(g, level="mua_gnr")
            frac = {lam: float((b[lam] >= 0.9).mean()) for lam in (808, 1064)}
            out[f"{cons}|gnr={g}"] = dict(treatable_fraction=frac, mean_best_coverage={lam: round(float(b[lam].mean()), 3) for lam in (808, 1064)})
            print(cons, "mua_gnr", g, "fraction of (depth,Dt) cases with coverage>=0.9:", {k: round(v, 2) for k, v in frac.items()}, " mean best coverage:", out[f"{cons}|gnr={g}"]["mean_best_coverage"])
json.dump(out, open("../data/treatability_grid.json", "w"), indent=1)
