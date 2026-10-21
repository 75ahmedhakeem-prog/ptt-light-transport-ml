import sys; sys.path.insert(0, "../code")
exec(open("../code/make_fig_ptt.py").read().split("D = \"../data/\"")[0])
import json, numpy as np, dp1, mc_het
src3 = open("../code/stageD_shaw.py").read().split("res = {}")[0]
import os; os.chdir("../code"); exec(src3)
mua, mus, gg = maps("Case-1", 0.40)
m = mc_het.run(R, H, nr, nz, mua, mus, gg, 10.0, P=P, n_in=1.37, n_top=1.0, n_side=1.5, n_bot=1.5, N=400000, seed=21)
d = dp1.solve(R, H, nr, nz, mua, mus, gg, 10.0, P=P, n_in=1.37)
musr = mus * (1 - gg); E0 = 0.0025 * (1 - (0.37 / 2.37)**2); att = np.exp(-np.cumsum((mua + musr) * dz, axis=1) + (mua + musr) * dz / 2)
bl = np.where(rc[:, None] <= 10, mua * E0 * att, 0)
ax = slice(0, 8)                                       # r < 2 mm
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8)); a = axs[0]
a.semilogy(zc, m["Q"][ax].mean(0) * 1e3, "-", color=BLUE, label="Monte Carlo")
a.semilogy(zc, d["Q"][ax].mean(0) * 1e3, "--", color=ORANGE, label="δP1")
a.semilogy(zc, np.maximum(bl[ax].mean(0) * 1e3, 1e-6), ":", color=GREEN, lw=2.8, label="Beer–Lambert")
a.axvspan(3, 9, color="#F8CFA0", alpha=0.45, lw=0); a.text(6, 3e-3, "tumour", ha="center", fontsize=14)
a.set_xlabel("Depth (mm)"); a.set_ylabel("Absorbed power density (mW mm$^{-3}$)"); a.set_xlim(0, 12); a.set_ylim(1e-3, 30)
a.legend(loc="upper right", fontsize=13); lab(a, "(a)"); a.xaxis.set_minor_locator(AutoMinorLocator(2))
b = axs[1]
D = "../data/"; ex = json.load(open(D + "stageCD_excess.json")); dp = json.load(open(D + "stage_dp1_compare.json")); sd = json.load(open(D + "stageD_shaw.json"))
names = ["NR1", "NR1+4%BSA", "NR2", "NR2+4%BSA", "Case-1", "Case-2"]
mcv = [ex["rmse"][n]["MC"] for n in names[:4]] + [sd["Case-1|damped|h=10"]["rmse"], sd["Case-2|damped|h=10"]["rmse"]]
dpv = [dp["dataset1"][n]["rmse_dp1"] for n in names[:4]] + [dp["dataset3"]["Case-1|damped"]["rmse"], dp["dataset3"]["Case-2|damped"]["rmse"]]
blv = [ex["rmse"][n]["BL"] for n in names[:4]] + [sd["BeerLambert_h10_damped"]["Case-1"]["rmse"], sd["BeerLambert_h10_damped"]["Case-2"]["rmse"]]
x = np.arange(6); w = 0.27
b.bar(x - w, mcv, w, color=BLUE, edgecolor=BLACK, lw=1.2, label="Monte Carlo"); b.bar(x, dpv, w, color=ORANGE, edgecolor=BLACK, lw=1.2, label="δP1")
b.bar(x + w, blv, w, color=GREEN, edgecolor=BLACK, lw=1.2, label="Beer–Lambert"); b.axhline(2, color=GREY, ls="--", lw=1.5)
b.set_xticks(x); b.set_xticklabels(["NR1", "NR1\n+BSA", "NR2", "NR2\n+BSA", "Sub-\nsurf. 1", "Sub-\nsurf. 2"], fontsize=12.5)
b.tick_params(axis="x", which="both", top=False, length=0); b.set_ylabel("RMSE (°C)"); b.set_ylim(0, 10.5)
b.axvline(3.5, color=BLACK, lw=1); b.text(1.5, 9.8, "Dataset 1", ha="center", fontsize=13); b.text(4.5, 9.8, "Dataset 3", ha="center", fontsize=13)
b.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), fontsize=12.5); lab(b, "(b)", x=0.97, ha="right", y=0.88)
fig.tight_layout(); fig.savefig("../figs/ptt_fig7_dp1.png"); plt.close(fig)
print("MC absorbed", m["absorbed"], "dP1", d["absorbed"])
