"""
ml_design.py - surrogate models for the photothermal design dataset.
Targets: T_tum_min (minimum tumour temperature) and T_skin_max (maximum skin temperature), both in degC.
Models: ANN (scikit-learn MLP) and a gradient-boosted tree ensemble (XGBoost if installed, otherwise
scikit-learn HistGradientBoosting as a stand-in).  Tests: random 80/20 split and level hold-out.
Usage:  python ml_design.py --data design_dataset.csv --out results_dir [--n-train 60000]
"""
import argparse, json, os, time, numpy as np, pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_squared_error
try:
    from xgboost import XGBRegressor; HAVE_XGB = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingRegressor; HAVE_XGB = False
SEED = 42
FEAT = ["lam", "spot_mm", "log_gnr", "depth_mm", "Dt_mm", "perf", "log_h", "I_Wcm2", "t_s"]
HOLD = {"spot_mm": [15, 20], "mua_gnr": [0.1, 0.2, 0.5], "depth_mm": [1, 2, 3, 4], "Dt_mm": [7.5, 10.0, 12.5],
        "perf": [5e-4, 1e-3], "h_cool": [200.0], "I_Wcm2": [0.5, 0.75, 1.0, 1.5], "t_s": [300, 450, 600]}
def prep(d):
    X = pd.DataFrame({"lam": (d.lam_nm == 1064).astype(float), "spot_mm": d.spot_mm, "log_gnr": np.log10(d.mua_gnr), "depth_mm": d.depth_mm,
                      "Dt_mm": d.Dt_mm, "perf": d.perf * 1e3, "log_h": np.log10(d.h_cool), "I_Wcm2": d.I_Wcm2, "t_s": d.t_s / 100})
    return X[FEAT]
def ann():
    return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 128, 64), activation="tanh",
        alpha=1e-5, learning_rate_init=2e-3, batch_size=512, max_iter=200, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        random_state=SEED)), transformer=StandardScaler())
def gbt():
    if HAVE_XGB:
        return XGBRegressor(n_estimators=1500, max_depth=8, learning_rate=0.05, subsample=0.85, colsample_bytree=0.9, random_state=SEED, n_jobs=-1)
    return HistGradientBoostingRegressor(max_iter=600, max_depth=None, max_leaf_nodes=127, learning_rate=0.05, random_state=SEED)
def metr(y, p): return dict(R2=float(r2_score(y, p)), RMSE=float(np.sqrt(mean_squared_error(y, p))))
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="design_dataset.csv"); ap.add_argument("--out", default="ml_out")
    ap.add_argument("--n-train", type=int, default=60000); ap.add_argument("--targets", default="T_tum_min,T_skin_max"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); d = pd.read_csv(a.data); X = prep(d); rng = np.random.default_rng(SEED)
    tree_name = "XGBoost" if HAVE_XGB else "HistGB (stand-in)"; res = {"tree_model": tree_name, "random": {}, "holdout": []}
    for tg in a.targets.split(","):
        y = d[tg].values; idx = rng.permutation(len(d)); ntr = int(0.8 * len(d)); tr, te = idx[:ntr], idx[ntr:]
        tr = tr[: a.n_train]; te = te[: 20000]
        for name, mk in (("ANN", ann), (tree_name, gbt)):
            t0 = time.time(); m = mk().fit(X.iloc[tr], y[tr]); res["random"].setdefault(tg, {})[name] = {**metr(y[te], m.predict(X.iloc[te])), "time_s": round(time.time() - t0, 1)}
            print(tg, name, res["random"][tg][name], flush=True)
        for var, levels in HOLD.items():
            for lv in levels:
                mask = np.isclose(d[var].values, lv); trp = np.where(~mask)[0]; tep = np.where(mask)[0]
                trp = rng.choice(trp, min(a.n_train, len(trp)), replace=False); tep = rng.choice(tep, min(20000, len(tep)), replace=False)
                row = dict(target=tg, variable=var, level=lv)
                for name, mk in (("ANN", ann), (tree_name, gbt)):
                    m = mk().fit(X.iloc[trp], y[trp]); mm = metr(y[tep], m.predict(X.iloc[tep])); row[f"{name}_R2"] = mm["R2"]; row[f"{name}_RMSE"] = mm["RMSE"]
                res["holdout"].append(row); print(row, flush=True)
                json.dump(res, open(os.path.join(a.out, "ml_results.json"), "w"), indent=1)
    pd.DataFrame(res["holdout"]).to_csv(os.path.join(a.out, "level_holdout.csv"), index=False)
    json.dump(res, open(os.path.join(a.out, "ml_results.json"), "w"), indent=1); print("done")
