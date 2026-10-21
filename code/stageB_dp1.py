"""delta-P1 vs MC: (1) tissue slab benchmark, (2) dataset-1 phantoms, (3) dataset-3 phantoms (absorbed power and temperatures)."""
import numpy as np, json, dp1, mc_het
out = {}
# (1) turbid tissue: R=30, H=30 mm, mua 0.01, mus 10, g 0.9, n 1.4, beam radius 10 mm
nr, nz, R, H = 60, 120, 30.0, 30.0
for name, mua_v, mus_v, g_v in [("tissue mua0.01 mus10 g0.9", 0.01, 10.0, 0.9), ("tissue mua0.1 mus10 g0.9", 0.1, 10.0, 0.9)]:
    mua = np.full((nr, nz), mua_v); mus = np.full((nr, nz), mus_v); g = np.full((nr, nz), g_v)
    d = dp1.solve(R, H, nr, nz, mua, mus, g, 10.0, n_in=1.4); m = mc_het.run(R, H, nr, nz, mua, mus, g, 10.0, n_in=1.4, n_top=1, n_side=1, n_bot=1, N=200000, seed=2)
    zc = d["z"]; ax_d = d["Q"][:4].mean(0); ax_m = m["Q"][:4].mean(0)
    sel = lambda z: int(z / (H / nz))
    out[name] = dict(absorbed_dp1=d["absorbed"], absorbed_mc=float(m["absorbed"]),
                     axial_ratio_dp1_mc={f"{z} mm": round(float(ax_d[sel(z)] / ax_m[sel(z)]), 3) for z in (0.5, 2, 5, 10, 20)})
    print(name, out[name])
json.dump(out, open("../data/stageB_dp1_benchmark.json", "w"), indent=1)
