"""Smooth-function baselines for the surrogate comparison (reviewer comment): polynomial response surfaces, additive +
pairwise-interaction cubic splines, local multiquadric RBF interpolation and Gaussian-process regression (ARD Matern 5/2).
The 44 level hold-outs use exactly the samples of ann_ensemble.py (a fresh generator with seed 42 for each hold-out),
so every baseline is trained and tested on the same samples as the five-network ANN ensemble of Table 9."""
import os, sys, json, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, SplineTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error
from scipy.interpolate import RBFInterpolator
import ml_design as M
N_GP_OPT, N_GP = 1500, 6000          # GP: hyperparameters optimised on 1500 training points, posterior on 6000

def poly(deg): return make_pipeline(StandardScaler(), PolynomialFeatures(deg), Ridge(alpha=1e-6, solver="cholesky"))
def spline():
    cont = [i for i, f in enumerate(M.FEAT) if f != "lam"]
    ct = ColumnTransformer([("s", SplineTransformer(n_knots=5, degree=3, knots="quantile", extrapolation="linear"), cont)], remainder="passthrough")
    return make_pipeline(ct, PolynomialFeatures(2, interaction_only=True, include_bias=False), StandardScaler(), Ridge(alpha=1e-3, solver="cholesky"))
class RBF:
    def fit(self, X, y):
        self.s = StandardScaler().fit(X); self.m = y.mean()
        self.f = RBFInterpolator(self.s.transform(X), y - self.m, kernel="multiquadric", epsilon=1.0, neighbors=64, smoothing=1e-6, degree=0); return self
    def predict(self, X): return self.f(self.s.transform(X)) + self.m
class GP:
    def __init__(self, seed=0): self.seed = seed
    def fit(self, X, y):
        rng = np.random.default_rng(self.seed); X = np.asarray(X, float)
        self.sx = StandardScaler().fit(X); self.my, self.sy = y.mean(), y.std(); Z = self.sx.transform(X); t = (y - self.my) / self.sy
        k = ConstantKernel(1.0) * Matern(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e3), nu=2.5) + WhiteKernel(1e-4, (1e-8, 1e-1))
        i = rng.choice(len(Z), min(N_GP_OPT, len(Z)), replace=False)
        g = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=0, random_state=self.seed).fit(Z[i], t[i])
        j = rng.choice(len(Z), min(N_GP, len(Z)), replace=False)
        self.g = GaussianProcessRegressor(g.kernel_, optimizer=None).fit(Z[j], t[j]); return self
    def predict(self, X):
        Z = self.sx.transform(np.asarray(X, float))
        return np.concatenate([self.g.predict(Z[i:i + 1000]) for i in range(0, len(Z), 1000)]) * self.sy + self.my
MODELS = {"Poly2": lambda: poly(2), "Poly3": lambda: poly(3), "Poly4": lambda: poly(4), "Poly5": lambda: poly(5), "Spline": spline, "RBF": lambda: RBF(), "GP": lambda: GP()}

def fit_eval(nm, Xtr, ytr, Xte, yte):
    t0 = time.time(); m = MODELS[nm]().fit(Xtr, ytr); t1 = time.time(); p = m.predict(Xte); t2 = time.time()
    return dict(M.metr(yte, p), t_fit=t1 - t0, t_pred_us=(t2 - t1) / len(Xte) * 1e6)


def metr(y, p): return dict(R2=float(r2_score(y, p)), RMSE=float(np.sqrt(mean_squared_error(y, p))))
def fit_eval(nm, Xtr, ytr, Xte, yte):
    t0 = time.time(); m = MODELS[nm]().fit(Xtr, ytr); t1 = time.time(); p = m.predict(Xte); t2 = time.time()
    return dict(metr(yte, p), t_fit=t1 - t0, t_pred_us=(t2 - t1) / len(Xte) * 1e6)

if __name__ == "__main__":
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else list(MODELS)
    d = pd.read_csv("../data/design_dataset_valid.csv"); X = M.prep(d).values; rows = []
    for tg in ("T_tum_min", "T_skin_max"):
        y = d[tg].values
        for var, lvs in M.HOLD.items():
            for lv in lvs:
                rng = np.random.default_rng(M.SEED)                     # as in ann_ensemble.py
                mask = np.isclose(d[var].values, lv); trp = rng.choice(np.where(~mask)[0], min(60000, (~mask).sum()), replace=False)
                tep = rng.choice(np.where(mask)[0], min(20000, mask.sum()), replace=False); row = dict(target=tg, variable=var, level=lv)
                for nm in only:
                    r = fit_eval(nm, X[trp], y[trp], X[tep], y[tep]); row[f"{nm}_R2"] = r["R2"]; row[f"{nm}_RMSE"] = r["RMSE"]
                rows.append(row); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
                pd.DataFrame(rows).to_csv(f"../data/ml_baselines_holdout_{'_'.join(only)}.csv", index=False)
    print("done")
