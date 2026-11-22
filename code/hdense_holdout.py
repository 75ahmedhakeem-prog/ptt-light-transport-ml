"""Level hold-out at h = 200 W m^-2 K^-1 after adding three cooling levels (30, 100, 500) to the design dataset.
Same sampling as ann_ensemble.py (fresh generator, seed 42; 60,000 training and 20,000 test samples); models: five-seed
ANN ensemble, RBF interpolation, Gaussian process and the quartic polynomial."""
import os, sys, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import ml_design as M, ml_baselines as B
SEEDS = [42, 7, 123, 2024, 99]
if __name__ == "__main__":
    d = pd.concat([pd.read_csv("../data/design_dataset_valid.csv"), pd.read_csv("../data/design_dataset_hdense.csv.gz")], ignore_index=True)
    X = M.prep(d).values; out = {"n_samples": len(d), "h_levels": sorted(d.h_cool.unique().tolist())}
    for tg in ("T_tum_min", "T_skin_max"):
        y = d[tg].values; rng = np.random.default_rng(M.SEED); mask = np.isclose(d.h_cool.values, 200.0)
        trp = rng.choice(np.where(~mask)[0], 60000, replace=False); tep = rng.choice(np.where(mask)[0], 20000, replace=False); r = {}
        P = []
        for s in SEEDS:
            m = M.ann(); m.regressor.set_params(mlpregressor__random_state=s); P.append(m.fit(X[trp], y[trp]).predict(X[tep]))
        P = np.array(P); r["ANN_single_mean"] = float(np.mean([r2_score(y[tep], p) for p in P])); r["ANN_ens"] = float(r2_score(y[tep], P.mean(0)))
        r["ANN_ens_RMSE"] = float(np.sqrt(mean_squared_error(y[tep], P.mean(0))))
        for nm in ("RBF", "GP", "Poly4"):
            f = B.fit_eval(nm, X[trp], y[trp], X[tep], y[tep]); r[nm] = f["R2"]; r[nm + "_RMSE"] = f["RMSE"]
        out[tg] = r; print(tg, {k: round(v, 4) for k, v in r.items()}, flush=True)
    json.dump(out, open("../data/hdense_holdout.json", "w"), indent=1)
