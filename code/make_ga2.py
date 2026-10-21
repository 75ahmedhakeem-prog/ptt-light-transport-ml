exec(open("../code/make_fig_ptt.py").read().split("D = \"../data/\"")[0])
import json
h = pd.read_csv("../data/colab_res/results/level_holdout.csv")
fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw=dict(wspace=0.38))
# (a) blind-test RMSE
a = axs[0]; x = np.arange(2); w = 0.26
mc = [np.mean([2.24, 3.44, 2.87]), np.mean([1.24, 0.88])]; bl = [np.mean([4.73, 8.36, 9.02]), np.mean([4.15, 3.04])]
a.bar(x - w, mc, w, color=BLUE, ec=BLACK, lw=1.2, label="Monte Carlo")
a.bar(x, bl, w, color=ORANGE, ec=BLACK, lw=1.2, label="Beer–Lambert")
a.bar([0 + w], [np.mean([5.41, 12.66, 16.18])], w, color=GREY, ec=BLACK, lw=1.2, label="Published model")
a.set_xticks(x); a.set_xticklabels(["GNR\nphantoms", "Subsurface\nphantoms"]); a.set_ylabel("Blind-test RMSE (°C)"); a.set_ylim(0, 15)
a.tick_params(axis="x", which="both", top=False, length=0); a.legend(loc="upper right", fontsize=12.5); lab(a, "(a)")
a.set_title("Light model vs. measured", fontsize=16, fontweight="bold"); a.yaxis.set_minor_locator(AutoMinorLocator(2))
# (b) level hold-out
a = axs[1]; s = h.groupby("target")[["ANN_R2", "XGBoost_R2"]].mean().reindex(["T_tum_min", "T_skin_max"])
a.bar(x - w / 2, s.XGBoost_R2, w, color=GREEN, ec=BLACK, lw=1.2, label="XGBoost")
a.bar(x + w / 2, s.ANN_R2, w, color=RED, ec=BLACK, lw=1.2, label="ANN")
a.set_xticks(x); a.set_xticklabels(["T$_{tumour,min}$", "T$_{skin,max}$"]); a.set_ylabel("Level hold-out R$^2$"); a.set_ylim(0, 1.25)
a.tick_params(axis="x", which="both", top=False, length=0); a.legend(loc="upper right", ncol=2, fontsize=12.5); lab(a, "(b)")
a.set_title("Surrogate on unseen levels", fontsize=16, fontweight="bold"); a.yaxis.set_minor_locator(AutoMinorLocator(2))
# (c) treatability
a = axs[2]; t = json.load(open("../data/treatability_grid.json"))
vals = [t["skin45|gnr=0.1"]["treatable_fraction"]["808"], t["skin45+MPE|gnr=0.1"]["treatable_fraction"]["808"], t["skin45+MPE|gnr=0.1"]["treatable_fraction"]["1064"]]
xx = np.arange(3); a.bar(xx, np.array(vals) * 100, 0.55, color=[BLUE, BLUE, PURPLE], ec=BLACK, lw=1.2)
for xi, v in zip(xx, vals): a.text(xi, v * 100 + 1.5, f"{v*100:.0f}%", ha="center", va="bottom", fontsize=15)
a.set_xticks(xx); a.set_xticklabels(["808 nm\nskin limit", "808 nm\n+ MPE", "1064 nm\n+ MPE"]); a.set_ylabel("Treatable tumours (%)"); a.set_ylim(0, 70)
a.tick_params(axis="x", which="both", top=False, length=0); lab(a, "(c)", x=0.96, ha="right")
a.set_title("Safe design: NIR-I vs. NIR-II", fontsize=16, fontweight="bold"); a.yaxis.set_minor_locator(AutoMinorLocator(2))
fig.savefig("../figs/ptt_ga2.png"); plt.close(fig)
