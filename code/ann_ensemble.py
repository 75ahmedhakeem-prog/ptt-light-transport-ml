"""5-seed ANN ensemble on all 44 level hold-outs: mean/std of single-model R2 and R2 of the ensemble mean."""
import numpy as np, pandas as pd, json, time
from joblib import Parallel, delayed
import ml_design as M
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error
d = pd.read_csv("../data/design_dataset_valid.csv"); X = M.prep(d)
SEEDS = [42, 7, 123, 2024, 99]
def ann(seed):
    m = M.ann(); m.regressor.set_params(mlpregressor__random_state=seed); return m
def job(tg, var, lv):
    rng = np.random.default_rng(M.SEED); y = d[tg].values
    mask = np.isclose(d[var].values, lv); trp = np.where(~mask)[0]; tep = np.where(mask)[0]
    trp = rng.choice(trp, min(60000, len(trp)), replace=False); tep = rng.choice(tep, min(20000, len(tep)), replace=False)
    P = np.array([ann(s).fit(X.iloc[trp], y[trp]).predict(X.iloc[tep]) for s in SEEDS])
    r2s = [r2_score(y[tep], p) for p in P]; pm = P.mean(0)
    return dict(target=tg, variable=var, level=lv, R2_mean=float(np.mean(r2s)), R2_std=float(np.std(r2s, ddof=1)), R2_min=float(np.min(r2s)),
                R2_ens=float(r2_score(y[tep], pm)), RMSE_ens=float(np.sqrt(mean_squared_error(y[tep], pm))), spread_mean=float(P.std(0, ddof=1).mean()))
jobs = [(tg, v, lv) for tg in ("T_tum_min", "T_skin_max") for v, lvs in M.HOLD.items() for lv in lvs]
t0 = time.time()
res = Parallel(n_jobs=2, verbose=5)(delayed(job)(*j) for j in jobs)
pd.DataFrame(res).to_csv("../data/ann_ensemble_holdout.csv", index=False); print("done", round(time.time() - t0))
