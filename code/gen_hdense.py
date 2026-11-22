"""Extra design samples at three intermediate cooling levels (h = 30, 100, 500 W m^-2 K^-1), all other factors as in
gen_dataset.py; samples above 100 C are removed as in design_dataset_valid.csv."""
import itertools, time, os, numpy as np, pandas as pd
from multiprocessing import Pool
import gen_dataset as G
H_NEW = [30.0, 100.0, 500.0]
if __name__ == "__main__":
    combos = list(itertools.product(G.LAM, G.SPOT, G.GNR, G.DEPTH, G.DT, G.PERF, H_NEW)); t0 = time.time(); out = []
    with Pool(int(os.environ.get("NPROC", 2))) as p:
        for i, rows in enumerate(p.imap_unordered(G.one, combos, chunksize=20)):
            out += rows
            if i % 1000 == 0: print(i, len(combos), round(time.time() - t0), flush=True)
    d = pd.DataFrame(out); d = d[(d.T_tum_max <= 100) & (d.T_skin_max <= 100)]
    d.to_csv("../data/design_dataset_hdense.csv.gz", index=False); print("done", len(d), round(time.time() - t0))
