"""Sensitivity of the NIR-II advantage to the nanorod absorption at 1064 nm: extra delta-P1 runs at low mua_GNR (perf 1e-3)."""
import itertools, numpy as np, pandas as pd, json
from multiprocessing import Pool
import gen_dataset as G
EXTRA = [0.01, 0.02]
if __name__ == "__main__":
    combos = list(itertools.product([1064], G.SPOT, EXTRA, G.DEPTH, G.DT, [1e-3], G.HCOOL))
    out = []
    with Pool(2) as p:
        for rows in p.imap_unordered(G.one, combos, chunksize=10): out += rows
    ex = pd.DataFrame(out); ex.to_csv("../data/sens_nir2_extra.csv", index=False)
    d = pd.read_csv("../data/design_dataset_valid.csv"); d = d[np.isclose(d.perf, 1e-3)]
    d = pd.concat([d, ex[ex[["T_skin_max"]].max(axis=1) <= 100]])
    MPE = {808: 0.2 * 10**(0.002 * 108), 1064: 1.0}; res = {}
    for lam in (808, 1064):
        for g in sorted(d[d.lam_nm == lam].mua_gnr.unique()):
            s = d[(d.lam_nm == lam) & np.isclose(d.mua_gnr, g)]
            for cons in ("skin45", "skin45+MPE"):
                ok = s.T_skin_max <= 45
                if cons.endswith("MPE"): ok &= s.I_Wcm2 <= MPE[lam] + 1e-9
                best = s[ok].groupby(["depth_mm", "Dt_mm"]).coverage.max().reindex(pd.MultiIndex.from_product([sorted(d.depth_mm.unique()), sorted(d.Dt_mm.unique())])).fillna(0)
                res[f"{lam}|{g}|{cons}"] = dict(frac=float((best >= 0.9).mean()), mean_cov=float(best.mean()))
    json.dump(res, open("../data/sens_nir2.json", "w"), indent=1)
    for k, v in res.items(): print(k, v)
