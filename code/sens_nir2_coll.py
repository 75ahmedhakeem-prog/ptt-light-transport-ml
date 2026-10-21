import numpy as np, pandas as pd, json
d = pd.read_csv("../data/design_dataset_valid.csv"); d = d[np.isclose(d.perf, 1e-3)]
ex = pd.read_csv("../data/sens_nir2_extra.csv"); d = pd.concat([d, ex])
MPE = {808: 0.2 * 10**(0.002 * 108), 1064: 1.0}
d = d[(d.T_skin_max <= 45) & (d.I_Wcm2 <= d.lam_nm.map(MPE) + 1e-9) & (d.coverage >= 0.9)]
d["Vt"] = np.pi / 6 * d.Dt_mm**3
out = {}
for (lam, g), s in d.groupby(["lam_nm", "mua_gnr"]):
    b = s.groupby(["depth_mm", "Dt_mm"]).apply(lambda x: (x.V_collateral / x.Vt).min())
    out[f"{lam}|{g}"] = dict(n=int(len(b)), median_ratio=round(float(b.median()), 2), min=round(float(b.min()), 2), max=round(float(b.max()), 2))
    print(lam, g, out[f"{lam}|{g}"])
json.dump(out, open("../data/sens_nir2_collateral.json", "w"), indent=1)
