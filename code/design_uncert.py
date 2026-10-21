"""5-seed ANN ensemble spread at the selected design protocols (Table 10)."""
import numpy as np, pandas as pd, json
from joblib import Parallel, delayed
import ml_design as M
d = pd.read_csv("../data/design_dataset_valid.csv"); X = M.prep(d); rng = np.random.default_rng(1)
idx = rng.choice(len(d), 120000, replace=False)
opt = json.load(open("../data/design_opt.json"))
P = pd.DataFrame([{k: opt[key][k] for k in ("lam_nm", "spot_mm", "mua_gnr", "depth_mm", "Dt_mm", "perf", "h_cool", "I_Wcm2", "t_s")} for key in ("808|skin45", "1064|skin45+MPE")])
XP = M.prep(P)
def fit(tg, seed):
    m = M.ann(); m.regressor.set_params(mlpregressor__random_state=seed)
    return tg, seed, m.fit(X.iloc[idx], d[tg].values[idx]).predict(XP)
res = Parallel(n_jobs=2)(delayed(fit)(tg, s) for tg in ("coverage", "T_skin_max") for s in (42, 7, 123, 2024, 99))
out = {}
for tg in ("coverage", "T_skin_max"):
    a = np.array([r[2] for r in res if r[0] == tg])
    if tg == "coverage": a = np.clip(a, 0, 1)
    out[tg] = dict(mean=a.mean(0).round(4).tolist(), std=a.std(0, ddof=1).round(4).tolist())
json.dump(dict(designs=["808|skin45", "1064|skin45+MPE"], **out), open("../data/design_uncert.json", "w"), indent=1); print(out)
