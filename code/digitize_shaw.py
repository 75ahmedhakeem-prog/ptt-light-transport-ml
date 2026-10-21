from PIL import Image; import numpy as np, pandas as pd
a = np.array(Image.open('../refs/new/shawfig4-000.png').convert('RGB')).astype(int); R, G, B = a[..., 0], a[..., 1], a[..., 2]
masks = {"(0,3)": (B > 150) & (R < 90) & (G < 90), "(10,3)": (G > 150) & (R < 110) & (B < 110), "(15,3)": (R > 150) & (G < 90) & (B < 90),
         "(0,6)": (G > 150) & (B > 150) & (R < 110), "(0,9)": (R > 150) & (B > 150) & (G < 110), "(10,9)": (R + G + B) < 150}
panels = {"Case-1": (95.5, 686.5), "Case-2": (872, 1462.5)}; y24, y40 = 642.5, 155
SKIP, JUMP = 3.0, 1.0
rows = []
for case, (xa, xb) in panels.items():
    for key, m in masks.items():
        mm = m.copy(); mm[:158, :] = False; mm[640:, :] = False
        X = list(range(int(xa) + 2, int(xb) - 1)); C = []
        for x in X:
            ys = np.where(mm[:, x])[0]
            grp = np.split(ys, np.where(np.diff(ys) > 2)[0] + 1) if len(ys) else []
            C.append([g.mean() for g in grp if 0 < len(g) <= 14])
        # DP over (column, candidate); transitions allowed from up to 25 columns back
        best = [dict() for _ in X]
        for i, cands in enumerate(C):
            for k, y in enumerate(cands):
                bc, bp = SKIP * i, None                       # start new path (penalised by skipped columns)
                for j in range(max(0, i - 25), i):
                    for kk, (cost, _) in best[j].items():
                        yp = C[j][kk]; c = cost + JUMP * abs(y - yp) / 1.0 + SKIP * (i - j - 1) + (50 if abs(y - yp) > 4 * (i - j) + 3 else 0)
                        if c < bc: bc, bp = c, (j, kk)
                best[i][k] = (bc, bp)
        # end: account for remaining skipped columns
        fin = [(cost + SKIP * (len(X) - 1 - i), i, k) for i in range(len(X)) for k, (cost, _) in best[i].items()]
        cost, i, k = min(fin); pts = []
        while True:
            pts.append((X[i], C[i][k])); prev = best[i][k][1]
            if prev is None: break
            i, k = prev
        pts = np.array(pts[::-1]); tsec = (pts[:, 0] - xa) / (xb - xa) * 1200; T = 24 + (y24 - pts[:, 1]) / (y24 - y40) * 16
        grid = np.arange(0, 1200.1, 10); Tg = np.interp(grid, tsec, T, left=np.nan, right=np.nan)
        Tg[0] = 25.0 if np.isnan(Tg[0]) else Tg[0]
        for u, v in zip(grid, Tg): rows.append((case, key, u, v))
        print(case, key, "pts %d" % len(pts), "T(100)=%.2f T(600)=%.2f T(1190)=%.2f" % (Tg[10], Tg[60], Tg[-2]))
pd.DataFrame(rows, columns=["case", "probe_r_z_mm", "t_s", "T_C"]).to_csv("../data/shaw_fig4_TE.csv", index=False)
