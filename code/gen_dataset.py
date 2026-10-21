"""Generate the full-factorial design dataset with delta-P1 + Pennes (unit-irradiance runs, scaled in post-processing)."""
import itertools, numpy as np, pandas as pd, time, sys
from multiprocessing import Pool
import design_sim as ds
LAM = [808, 1064]; SPOT = [10, 15, 20, 25]; GNR = [0.05, 0.1, 0.2, 0.5, 1.0]; DEPTH = [0, 1, 2, 3, 4, 5]
DT = [5.0, 7.5, 10.0, 12.5, 15.0]; PERF = [0.0, 5e-4, 1e-3, 2e-3]; HCOOL = [10.0, 200.0, 1000.0]
IRR = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]; TIMES = [120, 300, 450, 600, 900]
def one(args):
    lam, spot, gnr, depth, Dt, perf, hc = args
    Q = ds.light(lam, spot, depth, Dt, gnr); k, rhoc, wrc, tum, skin = ds.thermal_maps(depth, Dt, perf)
    F = ds.heat_fields(Q, k, rhoc, wrc, h_top=hc); rows = []
    for I in IRR:
        for t, o in zip(TIMES, ds.outcomes(F, tum, skin, I, TIMES)):
            rows.append(dict(lam_nm=lam, spot_mm=spot, mua_gnr=gnr, depth_mm=depth, Dt_mm=Dt, perf=perf, h_cool=hc, I_Wcm2=I, t_s=t, **o))
    return rows
if __name__ == "__main__":
    combos = list(itertools.product(LAM, SPOT, GNR, DEPTH, DT, PERF, HCOOL)); t0 = time.time()
    out = []
    with Pool(2) as p:
        for i, rows in enumerate(p.imap_unordered(one, combos, chunksize=20)):
            out += rows
            if i % 1000 == 0: print(i, len(combos), round(time.time() - t0), flush=True)
    pd.DataFrame(out).to_csv("../data/design_dataset.csv", index=False); print("done", len(out), round(time.time() - t0))
