import sys; sys.path.insert(0, "../code")
exec(open("../code/make_fig_ptt.py").read().split("D = \"../data/\"")[0])
import json, pickle
D = "../data/"
# ---------- Fig 4: Stage B
pr = pd.read_csv(D + "stageB_depth_profiles.csv"); sm = json.load(open(D + "stageB_summary.json"))
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8)); a = axs[0]
for nm, col in [("NR1", BLUE), ("NR2+4%BSA", RED)]:
    s = pr[pr["sample"] == nm]
    a.plot(s.z_mm, s.MC_W_per_mm * 1e3, "-", color=col, label=f"{nm}: MC")
    a.plot(s.z_mm, s.BL_W_per_mm * 1e3, "--", color=col, label=f"{nm}: BL")
    a.plot(s.z_mm, s.BLpub_W_per_mm * 1e3, ":", color=col, lw=2.6, label=f"{nm}: published")
a.axvline(10, color=GREY, ls="-.", lw=1.4); a.text(15, 40, "plain gel", fontsize=13, color=GREY, ha="center")
a.set_xlabel("Depth (mm)"); a.set_ylabel("Absorbed power (mW mm$^{-1}$)"); a.set_xlim(0, 20); a.set_ylim(0, 190)
a.legend(loc="upper right", fontsize=11.5, ncol=1); lab(a, "(a)", x=0.04, y=0.95)
a.xaxis.set_minor_locator(AutoMinorLocator(2)); a.yaxis.set_minor_locator(AutoMinorLocator(2))
b = axs[1]
ks = sorted(sm, key=lambda k: sm[k]["albedo"]); alb = [sm[k]["albedo"] for k in ks]; e1 = [100 * (sm[k]["tumour_W"]["BL"] / sm[k]["tumour_W"]["MC"] - 1) for k in ks]
e2 = [100 * (sm[k]["tumour_W"]["BLpub"] / sm[k]["tumour_W"]["MC"] - 1) for k in ks]
b.plot(alb, e1, "o-", color=BLUE, ms=9, label="Beer–Lambert (μ$_a$ heat)"); b.plot(alb, e2, "s-", color=RED, ms=9, label="Published (μ$_a$+μ$_s$ heat)")
for x, y, k, dy in zip(alb, e2, ks, (-22, 10, 10, 10)): b.annotate(k, (x, y), textcoords="offset points", xytext=(0, dy), fontsize=11.5, ha="center")
b.axhline(0, color=BLACK, lw=1.2); b.set_xlabel(r"Albedo $\mu_s/(\mu_a+\mu_s)$"); b.set_ylabel("Error in tumour heat vs MC (%)")
b.set_xlim(0.5, 0.97); b.set_ylim(-100, 260); b.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), fontsize=12.5); lab(b, "(b)")
b.xaxis.set_minor_locator(AutoMinorLocator(2)); b.yaxis.set_minor_locator(AutoMinorLocator(2))
fig.tight_layout(); fig.savefig("figs/ptt_fig4_stageB.png"); plt.close(fig)
# ---------- Fig 5: dataset 1 (calibration + blind) and uncertainty
ex = json.load(open(D + "stageCD_excess.json")); un = json.load(open(D + "stageC_uncertainty.json"))
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8)); a = axs[0]
names = list(ex["rmse"]); x = np.arange(len(names)); w = 0.26
for i, (key, col, l) in enumerate([("MC", BLUE, "Monte Carlo"), ("BL", GREEN, "Beer–Lambert"), ("BLpub", RED, "Published model")]):
    a.bar(x + (i - 1) * w, [ex["rmse"][n][key] for n in names], w, color=col, edgecolor=BLACK, lw=1.2, label=l)
a.axhline(2, color=GREY, ls="--", lw=1.5, label="Criterion (2 °C)")
a.set_xticks(x); a.set_xticklabels(["NR1\n(calib.)", "NR1+BSA\n(blind)", "NR2\n(blind)", "NR2+BSA\n(blind)"], fontsize=13.5)
a.tick_params(axis="x", which="both", top=False, length=0); a.set_ylabel("RMSE of excess ΔT (°C)"); a.set_ylim(0, 18.5)
a.legend(loc="upper left", fontsize=12.5); lab(a, "(a)", x=0.97, ha="right"); a.yaxis.set_minor_locator(AutoMinorLocator(2))
b = axs[1]
for (nm, col, mk) in [("NR1", BLUE, "o"), ("NR1+4%BSA", GREEN, "s"), ("NR2", RED, "^"), ("NR2+4%BSA", PURPLE, "D")]:
    f = sorted(un[nm], key=float); b.plot([float(v) for v in f], [un[nm][v]["rmse"] for v in f], "-" + mk, color=col, ms=8, label=nm)
b.axhline(2, color=GREY, ls="--", lw=1.5); b.set_xlabel("Scaling of the IAD nanorod absorption, f"); b.set_ylabel("RMSE of excess ΔT (°C)")
b.set_ylim(0, 9); b.legend(loc="upper left", fontsize=12.5); lab(b, "(b)", x=0.97, y=0.12, ha="right"); b.xaxis.set_minor_locator(AutoMinorLocator(2))
fig.tight_layout(); fig.savefig("figs/ptt_fig5_dataset1.png"); plt.close(fig)
# ---------- Fig 6: blind test on dataset 3 (Shaw)
import importlib.util
exp = pd.read_csv(D + "shaw_fig4_TE.csv"); sd = json.load(open(D + "stageD_shaw.json"))
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
cols = {"(0,3)": BLUE, "(10,3)": GREEN, "(15,3)": RED, "(0,6)": "#17BECF", "(0,9)": "#C51B8A", "(10,9)": BLACK}
sim = pickle.load(open(D + "stageD_curves.pkl", "rb"))
for a, case, l in zip(axs, ("Case-1", "Case-2"), ("(a)", "(b)")):
    for p, c in cols.items():
        e = exp[(exp["case"] == case) & (exp.probe_r_z_mm == p)].dropna(); a.plot(e.t_s, e.T_C, "-", color=c, label=p)
        t, T = sim[case][p]; a.plot(t, T, "--", color=c, lw=1.9)
    a.plot([], [], "k-", label="Measured"); a.plot([], [], "k--", label="MC model (blind)")
    a.set_xlabel("Time (s)"); a.set_ylabel("Temperature (°C)"); a.set_xlim(0, 1200); a.set_ylim(24, 40)
    a.text(0.04, 0.83, f"{case}\nRMSE {sd[f'{case}|damped|h=10']['rmse']:.2f} °C", transform=a.transAxes, fontsize=14, va="top")
    a.legend(loc="upper right", ncol=2, fontsize=11, columnspacing=0.8, handlelength=1.6); lab(a, l)
    a.xaxis.set_minor_locator(AutoMinorLocator(2)); a.yaxis.set_minor_locator(AutoMinorLocator(2))
fig.tight_layout(); fig.savefig("figs/ptt_fig6_blind_shaw.png"); plt.close(fig)
print("ok")
