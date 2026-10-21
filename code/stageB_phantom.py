"""Stage B: light transport in the dataset-1 phantoms: MC vs Beer-Lambert (corrected and published forms)."""
import numpy as np, pandas as pd, json, mc
# band-averaged optical properties (mm^-1) digitised from the IAD figure; water from Hale & Querry (approx.)
water = {"720-870": 0.003, "800-1000": 0.020}
samples = {"NR1": (0.025, 0.035, "720-870"), "NR1+4%BSA": (0.055, 0.065, "800-1000"),
           "NR2": (0.035, 0.205, "800-1000"), "NR2+4%BSA": (0.033, 0.375, "800-1000")}
g, n_gel, Rb, R, H = 0.8, 1.33, 6.5, 11.0, 20.0
P = 1.3e-2 * np.pi * Rb**2          # W (1.3 W/cm2 = 0.013 W/mm2)
res = {}; prof = []
for nm, (mua, mus, band) in samples.items():
    geo = mc.Cylinder(R, H, [(0, 10, mua, mus, g, n_gel), (10, 20, water[band], 0.01, 0.9, n_gel)], n_out=1.0)
    o = mc.run_cylinder(geo, Rb, P=P, N=400000, nr=44, nz=80, seed=3)
    dz = H / 80; dr = R / 44; rc = o["r"]; vol = 2 * np.pi * rc[:, None] * dr * dz
    Pz = (o["Q"] * vol).sum(0) / dz                 # W/mm absorbed per unit depth
    z = o["z"]; E0 = 0.013 * (1 - ((1 - n_gel) / (1 + n_gel))**2)
    area = np.pi * Rb**2
    mut1, mut2 = mua + mus, water[band] + 0.01
    att = np.where(z < 10, np.exp(-mut1 * z), np.exp(-mut1 * 10 - mut2 * (z - 10)))
    bl = np.where(z < 10, mua, water[band]) * E0 * att * area
    blpub = np.where(z < 10, mua + mus, 0.0) * E0 * att * area   # published: total attenuation as heat, plain gel 0
    A_t = {k: float(v[z < 10].sum() * dz) for k, v in [("MC", Pz), ("BL", bl), ("BLpub", blpub)]}
    A_h = {k: float(v[z >= 10].sum() * dz) for k, v in [("MC", Pz), ("BL", bl), ("BLpub", blpub)]}
    res[nm] = dict(albedo=mus / (mua + mus), P_in=P, R_top=o["R_top"], T_bot=o["T_bot"], side=o["side"],
                   absorbed=o["absorbed"], tumour_W=A_t, healthy_W=A_h)
    for zi, a, b, c in zip(z, Pz, bl, blpub): prof.append((nm, zi, a, b, c))
    print(nm, json.dumps({k: (round(v, 4) if isinstance(v, float) else {kk: round(vv, 4) for kk, vv in v.items()}) for k, v in res[nm].items()}))
pd.DataFrame(prof, columns=["sample", "z_mm", "MC_W_per_mm", "BL_W_per_mm", "BLpub_W_per_mm"]).to_csv("../data/stageB_depth_profiles.csv", index=False)
json.dump(res, open("../data/stageB_summary.json", "w"), indent=1)
