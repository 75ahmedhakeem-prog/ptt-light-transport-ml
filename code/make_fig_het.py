"""Figure: heterogeneous GNR distribution at equal total gold (het_ptt.py)."""
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.gridspec as gs
import het_ptt as H, design_sim as ds
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.linewidth": 1.4, "xtick.direction": "in", "ytick.direction": "in",
                     "savefig.dpi": 600, "savefig.bbox": "tight"})
R = json.load(open("../data/het_ptt.json")); m = pd.DataFrame(R["maps"])
PAT = [("uniform", "Uniform"), ("rim", "Rim"), ("shell", "Necrotic core"), ("centre", "Central"), ("light-side", "Light side"), ("far-side", "Far side"),
       ("lognormal|0.5", "Random, 0.5 mm"), ("lognormal|2.0", "Random, 2 mm")]
fig = plt.figure(figsize=(15, 10)); G = gs.GridSpec(2, 7, height_ratios=[1, 1.25], hspace=0.55, wspace=0.12)
maps = [("uniform", "Uniform"), ("rim", "Rim"), ("shell", "Necrotic core"), ("centre", "Central"), ("light-side", "Light side"), ("far-side", "Far side"), ("lognormal", "Random")]
ir = ds.rc <= 6.0; iz = (ds.zc >= 3.0) & (ds.zc <= 14.0)
for j, (k, t) in enumerate(maps):
    a = fig.add_subplot(G[0, j]); c = H.pattern(k, 2.5, 9.0, seed=0, ell=0.5)
    mm = np.ma.masked_where(c[np.ix_(ir, iz)].T <= 0, c[np.ix_(ir, iz)].T)
    im = a.imshow(mm, extent=[0, 6, 14, 3], vmin=0, vmax=3, cmap="viridis", aspect="auto")
    a.add_patch(matplotlib.patches.Arc((0, 1.5 + 2.5 + 4.5), 9, 9, theta1=-90, theta2=90, color="k", lw=1.3))
    a.set_title(t, fontsize=12); a.set_xticks([0, 3, 6]); a.tick_params(labelsize=11); a.set_xlabel("r (mm)", fontsize=11)
    if j: a.set_yticklabels([])
    else: a.set_ylabel("z (mm)"); a.text(-0.8, 1.22, "(a)", transform=a.transAxes, fontsize=16, fontweight="bold")
cb = fig.colorbar(im, ax=fig.axes[:7], fraction=0.015, pad=0.01); cb.set_label("c / c$_{mean}$")
def frac(kind, lam, col):
    k, *e = kind.split("|"); g = m[(m.kind == k) & (m.lam == lam)]
    if e: g = g[np.isclose(g.ell, float(e[0]))]
    f = g.groupby("seed")[col].apply(lambda s: (s >= 0.9).mean()); return f.median(), f.min(), f.max()
B = gs.GridSpecFromSubplotSpec(1, 2, subplot_spec=G[1, :], width_ratios=[1.25, 1], wspace=0.22)
a = fig.add_subplot(B[0]); x = np.arange(len(PAT))
for i, (lam, col, lab, mk, cl) in enumerate(((808, "skin45", "808 nm, T$_{skin}$ ≤ 45 °C", "s", "#1F4FB4"), (1064, "skin45+MPE", "1064 nm, T$_{skin}$ ≤ 45 °C and MPE", "o", "#D62728"),
                                               (808, "skin45+MPE", "808 nm, T$_{skin}$ ≤ 45 °C and MPE", "D", "#7F7F7F"))):
    v = np.array([frac(k, lam, col) for k, _ in PAT]) * 100
    a.errorbar(x + (i - 1) * 0.18, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=mk, color=cl, ms=8, capsize=3, label=lab)
a.set_xticks(x); a.set_xticklabels([t for _, t in PAT], rotation=35, ha="right"); a.set_ylabel("Treatable tumors (%)"); a.set_ylim(-3, 70)
a.legend(loc="upper left", fontsize=10.5, frameon=False); a.text(0.01, 1.04, "(b)", transform=a.transAxes, fontsize=16, fontweight="bold")
a = fig.add_subplot(B[1])
for i, (key, lab, mk, cl) in enumerate((("808|skin45", "808 nm design", "s", "#1F4FB4"), ("1064|skin45+MPE", "1064 nm design (MPE)", "o", "#D62728"))):
    v = []
    for k, _ in PAT:
        kk, *e = k.split("|")
        vals = [R["design"][d][key]["coverage"] for d in R["design"] if d.split("|")[0] == kk and (not e or np.isclose(float(d.split("|")[2]), float(e[0])))]
        v.append((np.median(vals), min(vals), max(vals)))
    v = np.array(v); a.errorbar(x + (i - 0.5) * 0.25, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=mk, color=cl, ms=8, capsize=3, label=lab)
a.axhline(0.98, color="grey", ls="--", lw=1.2); a.set_xticks(x); a.set_xticklabels([t for _, t in PAT], rotation=35, ha="right")
a.set_ylabel("Tumor coverage"); a.set_ylim(0.915, 1.0); a.legend(loc="lower left", ncol=1, fontsize=10.5, frameon=False); a.text(0.01, 1.04, "(c)", transform=a.transAxes, fontsize=16, fontweight="bold")
fig.savefig("../figs/fig11_heterogeneity.png"); print("ok")
