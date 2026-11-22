"""Fit and prediction times of the ANN and the smooth baselines (T_tum_min, 60,000 training and 20,000 test samples of the
first level hold-out), on one core of an otherwise idle machine; median of three repetitions."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import json, time, numpy as np, pandas as pd, ml_design as M, ml_baselines as B
d = pd.read_csv("../data/design_dataset_valid.csv"); X = M.prep(d).values; y = d.T_tum_min.values
rng = np.random.default_rng(M.SEED); mask = np.isclose(d.spot_mm.values, 15)
tr = rng.choice(np.where(~mask)[0], 60000, replace=False); te = rng.choice(np.where(mask)[0], 20000, replace=False)
MK = {"ANN": M.ann, **B.MODELS}; out = {}
for nm, mk in MK.items():
    tf, tp = [], []
    for rep in range(3):
        t0 = time.perf_counter(); m = mk().fit(X[tr], y[tr]); t1 = time.perf_counter(); m.predict(X[te]); t2 = time.perf_counter()
        tf.append(t1 - t0); tp.append((t2 - t1) / len(te) * 1e6)
    out[nm] = dict(t_fit=float(np.median(tf)), t_pred_us=float(np.median(tp)), t_fit_all=tf, t_pred_all=tp); print(nm, round(out[nm]["t_fit"], 2), round(out[nm]["t_pred_us"], 2), flush=True)
json.dump(out, open("../data/time_all.json", "w"), indent=1)
