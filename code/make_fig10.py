import sys; sys.path.insert(0, "../code")
exec(open("../code/make_fig_ptt.py").read().split("D = \"../data/\"")[0])
import json
D = "../data/"; s = json.load(open(D + "sens_nir2.json")); c = json.load(open(D + "sens_nir2_collateral.json"))
def ser(lam, cons):
    k = sorted([(float(x.split("|")[1]), v["frac"]) for x, v in s.items() if x.startswith(f"{lam}|") and x.endswith("|" + cons)])
    return np.array(k).T
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
a = axs[0]
x, y = ser(808, "skin45"); a.plot(x, y * 100, "o-", color=BLUE, ms=9, label="808 nm, skin ≤ 45 °C")
x, y = ser(808, "skin45+MPE"); a.plot(x, y * 100, "s--", color=BLUE, mfc="white", ms=9, label="808 nm, + MPE")
x, y = ser(1064, "skin45+MPE"); a.plot(x, y * 100, "^-", color=RED, ms=10, label="1064 nm, + MPE")
a.set_xscale("log"); a.set_xlabel("GNR absorption μ$_{a,GNR}$ (mm$^{-1}$)"); a.set_ylabel("Treatable tumours (%)"); a.set_ylim(-3, 75)
a.legend(loc="upper left", fontsize=12.5); lab(a, "(a)", x=0.96, ha="right"); a.yaxis.set_minor_locator(AutoMinorLocator(2))
b = axs[1]
k = sorted([(float(x.split("|")[1]), v["median_ratio"]) for x, v in c.items() if x.startswith("1064|")]); k = np.array(k).T
b.plot(k[0], k[1], "^-", color=RED, ms=10, label="1064 nm, + MPE"); b.set_xscale("log")
b.set_xlabel("GNR absorption μ$_{a,GNR}$ (mm$^{-1}$)"); b.set_ylabel("Collateral / tumour volume (median)"); b.set_ylim(0, 2.6)
b.legend(loc="upper right", fontsize=12.5); lab(b, "(b)"); b.yaxis.set_minor_locator(AutoMinorLocator(2))
fig.tight_layout(); fig.savefig("../figs/ptt_fig10_sens.png"); plt.close(fig); print("ok")
