"""Stage A: absorption of gold nanorods with the Gans (spheroid) model and an analytic gold dielectric function."""
import numpy as np
def eps_au(lam_nm, width_nm=None):
    # Etchegoin, Le Ru, Meyer, J. Chem. Phys. 125 (2006) 164705 (analytic model, 400-1000 nm)
    einf, lp, gp = 1.53, 145.0, 17000.0
    terms = [(0.94, -np.pi/4, 468.0, 2300.0), (1.36, -np.pi/4, 331.0, 940.0)]
    l = np.asarray(lam_nm, float)
    if width_nm:  # surface-scattering correction of the Drude damping (A=1, vF=1.4e6 m/s)
        hbar_g_bulk = 1239.84 / gp                       # eV
        dg = 6.582e-16 * 1.4e6 / (width_nm * 1e-9 / 2) * 1.0  # eV (A vF / Leff, Leff = radius)
        gp = 1239.84 / (hbar_g_bulk + dg)
    e = einf - 1.0 / (lp**2 * (1 / l**2 + 1j / (gp * l)))
    for A, phi, li, gi in terms:
        e += A / li * (np.exp(1j*phi) / (1/li - 1/l - 1j/gi) + np.exp(-1j*phi) / (1/li + 1/l + 1j/gi))
    return e
def depol(ar):
    e2 = 1 - 1/ar**2; e = np.sqrt(e2)
    La = (1 - e2) / e2 * (np.log((1 + e) / (1 - e)) / (2 * e) - 1)
    return La, (1 - La) / 2
def cabs(lam_nm, L_nm, D_nm, n_m=1.33, eps=None):
    """orientation-averaged absorption cross-section (nm^2) of a prolate spheroid L x D"""
    ar = L_nm / D_nm; La, Lb = depol(ar)
    V = np.pi / 6 * L_nm * D_nm**2
    em = n_m**2; e = eps_au(lam_nm, D_nm) if eps is None else eps
    k = 2 * np.pi * n_m / np.asarray(lam_nm)
    a = lambda Lj: V * (e - em) / (em + Lj * (e - em))
    return k / 3 * (np.imag(a(La)) + 2 * np.imag(a(Lb))), V
if __name__ == "__main__":
    e = eps_au(np.array([600, 700, 800, 900]))
    print("eps_Au:", np.round(e, 2))
    lam = np.arange(500, 1100.1, 1.0)
    for name, L, D, lspr, conc in [("NR1", 24.6, 5.9, 797, 15.9 / 20), ("NR2", 42.4, 9.4, 812, 663 / 20)]:
        c, V = cabs(lam, L, D); i = c.argmax()
        mass = V * 1e-21 * 19.3          # g per rod
        N = conc * 1e-3 / mass / 1e3     # rods per cm^3  (mg/L -> g/cm^3)
        mua = N * c * 1e-14              # cm^-1
        print(f"{name}: AR={L/D:.2f}  Gans LSPR={lam[i]:.0f} nm (measured {lspr})  Cabs_max={c.max():.0f} nm^2  "
              f"N={N:.2e} cm^-3  mua@LSPR={mua.max():.3f} cm^-1 = {mua.max()/10:.4f} mm^-1  mua@800={mua[lam==800][0]/10:.4f} mm^-1")
