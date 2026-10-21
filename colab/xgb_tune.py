"""Fair tuning of XGBoost for the level hold-out test.
Random search (24 configs incl. the original) scored on 4 validation hold-out levels per target;
the best config is then evaluated on all 22 hold-out levels of both targets.
Usage: python xgb_tune.py --data design_dataset.csv --out results_tuned"""
import argparse, json, os, time, numpy as np, pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import ml_design as M
ap = argparse.ArgumentParser(); ap.add_argument("--data", default="design_dataset.csv"); ap.add_argument("--out", default="results_tuned")
ap.add_argument("--n-configs", type=int, default=24); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
d = pd.read_csv(a.data); d = d[d[["T_skin_max", "T_tum_max"]].max(axis=1) <= 100].reset_index(drop=True) if "T_tum_max" in d else d
X = M.prep(d)
try:
    XGBRegressor(n_estimators=2, device="cuda").fit(X.iloc[:100], d.T_tum_min.values[:100]); DEV = "cuda"
except Exception:
    DEV = "cpu"
print("device:", DEV, " samples:", len(d), flush=True)
VAL = [("spot_mm", 15), ("depth_mm", 2), ("Dt_mm", 10.0), ("I_Wcm2", 0.75)]
rng = np.random.default_rng(0)
space = dict(n_estimators=[300, 800, 1500, 3000], max_depth=[3, 4, 6, 8, 10], learning_rate=[0.02, 0.05, 0.1],
             subsample=[0.7, 0.85, 1.0], colsample_bytree=[0.7, 0.9, 1.0], min_child_weight=[1, 5, 20], reg_lambda=[0.0, 1.0, 10.0])
cfgs = [dict(n_estimators=1500, max_depth=8, learning_rate=0.05, subsample=0.85, colsample_bytree=0.9, min_child_weight=1, reg_lambda=1.0)]
while len(cfgs) < a.n_configs:
    c = {k: (v[rng.integers(len(v))]) for k, v in space.items()}; c = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in c.items()}
    if c not in cfgs: cfgs.append(c)
def split(tg, var, lv, seed=M.SEED):
    r = np.random.default_rng(seed); y = d[tg].values; mask = np.isclose(d[var].values, lv)
    trp = r.choice(np.where(~mask)[0], min(60000, (~mask).sum()), replace=False); tep = r.choice(np.where(mask)[0], min(20000, mask.sum()), replace=False)
    return trp, tep, y
def fit_eval(cfg, tg, var, lv):
    trp, tep, y = split(tg, var, lv)
    m = XGBRegressor(**cfg, tree_method="hist", device=DEV, random_state=M.SEED, n_jobs=-1).fit(X.iloc[trp], y[trp])
    p = m.predict(X.iloc[tep]); return float(r2_score(y[tep], p)), float(np.sqrt(mean_squared_error(y[tep], p)))
log = []; t0 = time.time()
for i, c in enumerate(cfgs):
    sc = [fit_eval(c, tg, v, lv)[0] for tg in ("T_tum_min", "T_skin_max") for v, lv in VAL]
    log.append(dict(cfg=c, val_R2_mean=float(np.mean(sc)), val_R2_min=float(np.min(sc))))
    print(i, round(np.mean(sc), 4), c, round(time.time() - t0), flush=True)
    json.dump(log, open(os.path.join(a.out, "tuning_log.json"), "w"), indent=1)
best = max(log, key=lambda r: r["val_R2_mean"])["cfg"]; print("best:", best, flush=True)
rows = []
for tg in ("T_tum_min", "T_skin_max"):
    for v, lvs in M.HOLD.items():
        for lv in lvs:
            r2, rm = fit_eval(best, tg, v, lv)
            rows.append(dict(target=tg, variable=v, level=lv, XGB_tuned_R2=r2, XGB_tuned_RMSE=rm, used_for_tuning=(v, lv) in VAL)); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(a.out, "xgb_tuned_holdout.csv"), index=False)
json.dump(dict(best=best, device=DEV, log=log), open(os.path.join(a.out, "tuning_log.json"), "w"), indent=1); print("done")
