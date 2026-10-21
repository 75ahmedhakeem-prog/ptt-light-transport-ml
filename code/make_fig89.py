import sys; sys.path.insert(0, "../code")
exec(open("../code/make_fig_ptt.py").read().split("D = \"../data/\"")[0])
D = "../data/"
h = pd.read_csv(D + "colab_res/results/level_holdout.csv")
tu = pd.read_csv(D + "xgb_tuned/results_tuned/xgb_tuned_holdout.csv"); en = pd.read_csv(D + "ann_ensemble_holdout.csv")
h = h.merge(tu, on=["target", "variable", "level"]).merge(en, on=["target", "variable", "level"])
names = {"spot_mm": "Spot", "mua_gnr": "μ$_{a,GNR}$", "depth_mm": "Depth", "Dt_mm": "Diameter", "perf": "Perfusion", "h_cool": "Cooling h", "I_Wcm2": "Irradiance", "t_s": "Time"}
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.0))
for a, tg, l, ttl in ((axs[0], "T_tum_min", "(a)", "Minimum tumour temperature"), (axs[1], "T_skin_max", "(b)", "Maximum skin temperature")):
    s = h[h.target == tg].groupby("variable")[["XGBoost_R2", "XGB_tuned_R2", "R2_ens", "R2_std"]].mean().reindex(list(names))
    x = np.arange(len(s)); w = 0.27
    for off, col, c, lb in ((-w, "XGBoost_R2", GREEN, "XGBoost (default)"), (0, "XGB_tuned_R2", ORANGE, "XGBoost (tuned)")):
        a.bar(x + off, s[col].clip(lower=0), w, color=c, edgecolor=BLACK, lw=1.1, label=lb)
        for xi, v in zip(x, s[col]):
            if v < 0: a.text(xi + off, 0.03, f"{v:.2f}", rotation=90, ha="center", va="bottom", fontsize=11)
    a.bar(x + w, s["R2_ens"], w, color=RED, edgecolor=BLACK, lw=1.1, yerr=s["R2_std"], capsize=3, label="ANN (5-seed ensemble)")
    a.set_xticks(x); a.set_xticklabels([names[v] for v in s.index], rotation=35, ha="right", fontsize=13)
    a.tick_params(axis="x", which="both", top=False, length=0); a.set_ylim(0, 1.32); a.set_ylabel("Level hold-out R$^2$")
    a.legend(loc="upper left", ncol=2, fontsize=11.5); lab(a, l, x=0.97, ha="right", y=0.95); a.set_title(ttl, fontsize=14.5, fontweight="bold")
    a.yaxis.set_minor_locator(AutoMinorLocator(2))
fig.tight_layout(); fig.savefig("../figs/ptt_fig8_ml.png"); plt.close(fig)
# ---- treatability maps
d = pd.read_csv(D + "design_dataset_valid.csv"); MPE = {808: 0.329, 1064: 1.0}
s = d[np.isclose(d.perf, 1e-3) & np.isclose(d.mua_gnr, 0.1)]
fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.6), sharey=True)
from matplotlib.colors import ListedColormap
for a, (lam, mpe, ttl), l in zip(axs, [(808, False, "808 nm, skin ≤ 45 °C"), (1064, True, "1064 nm, skin ≤ 45 °C + MPE"), (808, True, "808 nm, skin ≤ 45 °C + MPE")], ("(a)", "(b)", "(c)")):
    q = s[(s.lam_nm == lam) & (s.T_skin_max <= 45)]
    if mpe: q = q[q.I_Wcm2 <= MPE[lam]]
    b = q.groupby(["depth_mm", "Dt_mm"]).coverage.max().unstack().reindex(index=sorted(s.depth_mm.unique()), columns=sorted(s.Dt_mm.unique())).fillna(0)
    im = a.imshow(b.values, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto",
                  extent=[-0.5, len(b.columns) - 0.5, -0.5, len(b.index) - 0.5])
    for i in range(len(b.index)):
        for j in range(len(b.columns)):
            v = b.values[i, j]; a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11, color="white" if v < 0.6 else "black")
    a.set_xticks(range(len(b.columns))); a.set_xticklabels([f"{c:g}" for c in b.columns]); a.set_yticks(range(len(b.index))); a.set_yticklabels([f"{c:g}" for c in b.index])
    a.set_xlabel("Tumour diameter (mm)"); a.set_title(l + " " + ttl, fontsize=14.5, fontweight="bold"); a.tick_params(which="both", top=False, right=False)
axs[0].set_ylabel("Depth below skin (mm)")
cb = fig.colorbar(im, ax=axs, shrink=0.9, pad=0.015); cb.set_label("Best tumour coverage", fontweight="bold")
fig.savefig("../figs/ptt_fig9_treat.png"); plt.close(fig); print("ok")
