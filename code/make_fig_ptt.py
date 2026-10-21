"""Figures for the photothermal protocol (Origin style, 600 dpi, large fonts, panel labels inside)."""
import sys; sys.path.insert(0, "../code")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import AutoMinorLocator
import gans
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 16.5, "font.weight": "bold", "axes.labelweight": "bold",
    "axes.labelsize": 18, "axes.linewidth": 1.8, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
    "ytick.right": True, "xtick.major.size": 7, "ytick.major.size": 7, "xtick.minor.size": 3.5, "ytick.minor.size": 3.5,
    "xtick.major.width": 1.5, "ytick.major.width": 1.5, "legend.frameon": False, "legend.fontsize": 13.5,
    "lines.linewidth": 2.2, "savefig.dpi": 600, "savefig.bbox": "tight"})
BLUE, RED, GREEN, ORANGE, PURPLE, BLACK, GREY = "#1F4FB4", "#D62728", "#2CA02C", "#E67E22", "#6A3D9A", "#000000", "#7F7F7F"
def lab(ax, s, x=0.04, y=0.95, ha="left", **k): ax.text(x, y, s, transform=ax.transAxes, ha=ha, va="top", fontsize=19, fontweight="bold", **k)
D = "../data/"

# ---------------- Fig 1: phantom and computational domain
fig, ax = plt.subplots(figsize=(7.2, 6.0)); ax.set_xlim(-1, 15); ax.set_ylim(-3.2, 14); ax.axis("off")
ax.add_patch(Rectangle((2, 0), 11, 1.0, fc="#DDE8F0", ec=BLACK, lw=1.5)); ax.text(7.5, 0.5, "Glass plate", ha="center", va="center", fontsize=13)
ax.add_patch(Rectangle((2.5, 1.0), 10, 5, fc="#EAF2F8", ec=BLACK, lw=1.8)); ax.text(7.5, 3.5, "Plain gel\n(healthy tissue)", ha="center", va="center", fontsize=14)
ax.add_patch(Rectangle((2.5, 6.0), 10, 5, fc="#F8CFA0", ec=BLACK, lw=1.8)); ax.text(7.5, 9.6, "Agarose gel + GNRs\n(tumour)", ha="center", va="center", fontsize=14)
for x in np.linspace(4.6, 10.4, 5):
    ax.add_patch(FancyArrowPatch((x, 13.6), (x, 11.15), arrowstyle="-|>", mutation_scale=18, color=RED, lw=2.6))
ax.text(7.5, 13.75, "NIR beam, Ø 13 mm, 1.3 W cm$^{-2}$", ha="center", va="bottom", fontsize=13.5, color=RED)
for z in [1, 5, 10, 15]:
    y = 11 - z / 2; ax.plot([12.5, 14.6], [y, y], color=BLACK, lw=1.4); ax.plot(12.5, y, "o", color=BLACK, ms=5)
    ax.text(14.7, y, f"{z} mm", va="center", fontsize=13)
ax.annotate("", xy=(1.8, 11), xytext=(1.8, 6), arrowprops=dict(arrowstyle="<->", lw=1.5)); ax.text(1.5, 8.5, "10 mm", rotation=90, ha="right", va="center", fontsize=13)
ax.annotate("", xy=(1.8, 6), xytext=(1.8, 1), arrowprops=dict(arrowstyle="<->", lw=1.5)); ax.text(1.5, 3.5, "10 mm", rotation=90, ha="right", va="center", fontsize=13)
ax.annotate("", xy=(2.5, -0.6), xytext=(12.5, -0.6), arrowprops=dict(arrowstyle="<->", lw=1.5)); ax.text(7.5, -1.0, "Ø 22 mm", ha="center", va="top", fontsize=13)
ax.text(13.6, 12.0, "Thermocouples", ha="center", fontsize=13)
fig.savefig("figs/ptt_fig1_phantom.png"); plt.close(fig)

# ---------------- Fig 2: Stage A
ab = pd.read_csv(D + "pratap_fig2_absorbance.csv"); lam = np.arange(500, 1100.1, 1.0)
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
a = axs[0]
for nm, L, Dm, col in [("NR1", 24.6, 5.9, BLUE), ("NR2", 42.4, 9.4, RED)]:
    s = ab[(ab["sample"] == nm) & (ab.lam_nm >= 600)]
    base = s.absorbance.min(); y = (s.absorbance - base) / (s.absorbance.max() - base)
    a.plot(s.lam_nm, y, "-", color=col, label=f"{nm} measured")
    c, _ = gans.cabs(lam, L, Dm); a.plot(lam, c / c.max(), "--", color=col, label=f"{nm} Gans model")
a.set_xlim(600, 1100); a.set_ylim(0, 1.12); a.set_xlabel("Wavelength (nm)"); a.set_ylabel("Normalised absorption")
a.legend(loc="upper right", fontsize=12.5); a.xaxis.set_minor_locator(AutoMinorLocator(2)); a.yaxis.set_minor_locator(AutoMinorLocator(2)); lab(a, "(a)")
b = axs[1]
pred = {}
for nm, L, Dm, conc in [("NR1", 24.6, 5.9, 15.9 / 20), ("NR2", 42.4, 9.4, 663 / 20)]:
    V = np.pi / 6 * L * Dm**2; N = conc * 1e-3 / (V * 1e-21 * 19.3) / 1e3
    ar = L / Dm; e2 = 1 - 1 / ar**2; ecc = np.sqrt(e2); S = 2 * np.pi * (Dm / 2)**2 * (1 + (L / 2) / ((Dm / 2) * ecc) * np.arcsin(ecc))
    lo = gans.cabs(lam, L, Dm, eps=gans.eps_au(lam, width_nm=2 * 4 * V / S))[0].max(); hi = gans.cabs(lam, L, Dm, eps=gans.eps_au(lam))[0].max()
    pred[nm] = (N * lo * 1e-15, N * hi * 1e-15)          # mm^-1
meas = {"NR1": 0.027, "NR2": 0.065}
x = np.arange(2)
for i, nm in enumerate(["NR1", "NR2"]):
    lo, hi = pred[nm]; b.bar(i - 0.18, np.sqrt(lo * hi), 0.32, color=ORANGE, edgecolor=BLACK, lw=1.4, label="Gans + ICP-MS" if i == 0 else None)
    b.errorbar(i - 0.18, np.sqrt(lo * hi), yerr=[[np.sqrt(lo * hi) - lo], [hi - np.sqrt(lo * hi)]], color=BLACK, capsize=6, lw=1.6)
    b.bar(i + 0.18, meas[nm], 0.32, color=BLUE, edgecolor=BLACK, lw=1.4, label="IAD measurement" if i == 0 else None)
b.set_yscale("log"); b.set_ylim(3e-3, 5); b.set_xticks(x); b.set_xticklabels(["NR1", "NR2"]); b.tick_params(axis="x", which="both", top=False, length=0)
b.set_ylabel(r"$\mu_a$ at LSPR (mm$^{-1}$)"); b.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88)); lab(b, "(b)")
b.text(1, 2.6, "×10–30", ha="center", fontsize=15, color=RED)
fig.tight_layout(); fig.savefig("figs/ptt_fig2_stageA.png"); plt.close(fig)
pd.DataFrame([(k, *v, meas[k]) for k, v in pred.items()], columns=["sample", "mua_pred_low_mm-1", "mua_pred_high_mm-1", "mua_IAD_mm-1"]).to_csv(D + "stageA_mua.csv", index=False)

# ---------------- Fig 3: re-analysis of the published data
dT = pd.read_csv(D + "pratap_fig56_dT.csv"); pg = pd.read_csv(D + "pratap_fig4_plaingel_dT.csv")
model = {"NR1": [18.33, 15.51, 8.91, 3.39], "NR1+4%BSA": [25.16, 21.29, 12.22, 4.64], "NR2": [14.29, 11.06, 5.78, 2.19], "NR2+4%BSA": [10.43, 7.18, 3.40, 1.29]}
z = [1, 5, 10, 15]
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
a = axs[0]
for (nm, col, mk) in [("NR1", BLUE, "o"), ("NR1+4%BSA", GREEN, "s"), ("NR2", RED, "^"), ("NR2+4%BSA", PURPLE, "D")]:
    m = dT[(dT["sample"] == nm) & (dT.t_min == 15)].sort_values("depth_mm")
    a.plot(m.depth_mm, m.dT_C, "-" + mk, color=col, mfc=col, ms=8, label=nm)
    a.plot(z, model[nm], "--" + mk, color=col, mfc="white", mew=1.6, ms=8)
m = pg[(pg["filter"] == "filter2_800-1000") & (pg.t_min == 15)].sort_values("depth_mm")
a.plot(m.depth_mm, m.dT_C, "-x", color=GREY, ms=9, mew=2, label="Plain gel")
a.plot([], [], "k-", label="Measured"); a.plot([], [], "k--", label="Published model")
a.set_xlabel("Depth (mm)"); a.set_ylabel(r"$\Delta$T at 15 min (°C)"); a.set_xlim(0, 16); a.set_ylim(0, 36); a.set_xticks([0, 5, 10, 15])
a.legend(loc="upper right", ncol=2, fontsize=11.5, columnspacing=0.8, handlelength=1.8); lab(a, "(a)", y=0.08, x=0.04, )
a.xaxis.set_minor_locator(AutoMinorLocator(2)); a.yaxis.set_minor_locator(AutoMinorLocator(2))
b = axs[1]
for zz, col in zip(z, [BLACK, BLUE, RED, GREEN]):
    s = pg[(pg["filter"] == "filter2_800-1000") & (pg.depth_mm == zz)]
    b.plot(s.t_min, s.dT_C, color=col, label=f"z = {zz} mm")
    s = dT[(dT["sample"] == "NR1+4%BSA") & (dT.depth_mm == zz)]
    b.plot(s.t_min, s.dT_C, ":", color=col, lw=1.8)
b.set_xlabel("Time (min)"); b.set_ylabel(r"$\Delta$T (°C)"); b.set_xlim(0, 30); b.set_ylim(0, 26)
b.legend(loc="upper right", title="Plain gel", title_fontsize=13.5); lab(b, "(b)")
b.plot([], [], ":", color=GREY, lw=1.8, label="NR1+4%BSA"); b.legend(loc="upper right", title="Plain gel", title_fontsize=13.5)
b.xaxis.set_minor_locator(AutoMinorLocator(2)); b.yaxis.set_minor_locator(AutoMinorLocator(2))
fig.tight_layout(); fig.savefig("figs/ptt_fig3_reanalysis.png"); plt.close(fig)

# ---------------- Graphical abstract
fig = plt.figure(figsize=(13.0, 5.2))
g = fig.add_axes([0.06, 0.17, 0.27, 0.62])
for nm, col in [("NR1", BLUE), ("NR2", RED)]:
    m = dT[(dT["sample"] == nm) & (dT.t_min == 15)].sort_values("depth_mm")
    g.plot(m.depth_mm, m.dT_C, "-o", color=col, ms=8, label=f"{nm} measured"); g.plot(z, model[nm], "--o", color=col, mfc="white", ms=8, mew=1.6)
g.plot([], [], "k--", label="Beer–Lambert model")
g.set_xlabel("Depth (mm)", fontsize=17); g.set_ylabel(r"$\Delta$T (°C)", fontsize=17); g.set_ylim(0, 25); g.legend(fontsize=12, loc="upper right")
fig.text(0.195, 0.955, "The gap: error grows with depth", ha="center", va="top", fontsize=19)
ax = fig.add_axes([0.37, 0.0, 0.63, 1.0]); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
boxes = [("Light\ntransport", "Monte Carlo\nδP1\nBeer–Lambert", "#DCE6F2"), ("Bioheat\n+ dose", "Pennes\nCEM43\ncalibrated", "#FBE5C8"),
         ("ML\nsurrogate", "ANN vs\nXGBoost\nhold-out", "#FFF2CC"), ("Design", "808 vs\n1064 nm\nverified", "#E2EFDA")]
for i, (t, s, c) in enumerate(boxes):
    x = 0.15 + i * 2.48
    ax.add_patch(FancyBboxPatch((x, 1.0), 2.1, 4.1, boxstyle="round,pad=0.02,rounding_size=0.12", fc=c, ec=BLACK, lw=1.8))
    ax.text(x + 1.05, 4.85, t, ha="center", va="top", fontsize=17); ax.text(x + 1.05, 3.25, s, ha="center", va="top", fontsize=15, linespacing=1.5)
    if i < 3: ax.add_patch(FancyArrowPatch((x + 2.12, 3.05), (x + 2.46, 3.05), arrowstyle="-|>", mutation_scale=22, color=BLACK, lw=2.5))
fig.savefig("../figs/ptt_ga.png"); plt.close(fig)
print(pred)
