"""
dp1.py - delta-P1 approximation (Carp, Prahl, Venugopalan 2004) for a heterogeneous axisymmetric cylinder.
Collimated irradiance E_c (delta-Eddington scaled) + diffuse fluence phi_d from a finite-volume solution of
    div j_d + mu_a phi_d = mu_s* E_c,     j_d = -D grad(phi_d) + S1 z_hat,   S1 = (g* mu_s* / mu_tr) E_c,  D = 1/(3 mu_tr)
mu_s* = mu_s (1 - g^2), g* = g/(1+g), mu_tr = mu_a + mu_s*(1 - g*), mu_t* = mu_a + mu_s*.
Partial-current (Robin) condition on every outer surface: j_d . n = phi_d / (2A), A = (1+R_eff)/(1-R_eff).
Absorbed power Q = mu_a (phi_d + E_c).  Units: mm, mm^-1, W.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla

def r_eff(n):
    return -1.440 / n**2 + 0.710 / n + 0.668 + 0.0636 * n

def solve(R, H, nr, nz, mua, mus, g, beam_radius, P=1.0, n_in=1.37, n_out=1.0):
    dr, dz = R / nr, H / nz; rc = (np.arange(nr) + 0.5) * dr; rf = np.arange(nr + 1) * dr
    muss = mus * (1 - g**2); gs = g / (1 + g); mutr = mua + muss * (1 - gs); muts = mua + muss; D = 1 / (3 * mutr)
    c = 1 / (2 * (1 + r_eff(n_in / n_out)) / (1 - r_eff(n_in / n_out)))
    E0 = P / (np.pi * beam_radius**2) * (1 - ((n_out - n_in) / (n_out + n_in))**2)
    beam = (rc <= beam_radius)[:, None]
    tau_f = np.concatenate([np.zeros((nr, 1)), np.cumsum(muts * dz, axis=1)], axis=1)
    Ecf = np.where(beam, E0 * np.exp(-tau_f), 0.0)                       # at z faces (nr, nz+1)
    Ec = np.where(beam, E0 * np.exp(-(tau_f[:, :-1] + muts * dz / 2)), 0.0)
    s1 = gs * muss / mutr
    s1f = np.empty((nr, nz + 1)); s1f[:, 1:-1] = 0.5 * (s1[:, 1:] + s1[:, :-1]); s1f[:, 0] = s1[:, 0]; s1f[:, -1] = s1[:, -1]
    S1 = s1f * Ecf
    Az = np.pi * (rf[1:]**2 - rf[:-1]**2)[:, None]; Ar = 2 * np.pi * rf[1:-1, None] * dz; V = Az * dz
    idx = np.arange(nr * nz).reshape(nr, nz)
    diag = mua * V; rhs = muss * Ec * V
    rows, cols, vals = [], [], []
    Gr = (2 * D[1:] * D[:-1] / (D[1:] + D[:-1])) * Ar / dr               # radial faces (nr-1, nz)
    rows += [idx[:-1].ravel(), idx[1:].ravel()]; cols += [idx[1:].ravel(), idx[:-1].ravel()]; vals += [-Gr.ravel(), -Gr.ravel()]
    diag[:-1] += Gr; diag[1:] += Gr
    Gz = (2 * D[:, 1:] * D[:, :-1] / (D[:, 1:] + D[:, :-1])) * Az / dz   # axial faces (nr, nz-1)
    rows += [idx[:, :-1].ravel(), idx[:, 1:].ravel()]; cols += [idx[:, 1:].ravel(), idx[:, :-1].ravel()]; vals += [-Gz.ravel(), -Gz.ravel()]
    diag[:, :-1] += Gz; diag[:, 1:] += Gz
    rhs[:, :-1] -= S1[:, 1:-1] * Az; rhs[:, 1:] += S1[:, 1:-1] * Az        # source flux through internal z faces
    h = D[:, 0] / (dz / 2); diag[:, 0] += c * h / (h + c) * Az[:, 0]; rhs[:, 0] += c * S1[:, 0] / (h + c) * Az[:, 0]
    h = D[:, -1] / (dz / 2); diag[:, -1] += c * h / (h + c) * Az[:, 0]; rhs[:, -1] -= c * S1[:, -1] / (h + c) * Az[:, 0]
    h = D[-1, :] / (dr / 2); diag[-1, :] += c * h / (h + c) * 2 * np.pi * R * dz
    M = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nr * nz,) * 2) + sp.diags(diag.ravel())
    phi = spla.spsolve(M.tocsc(), rhs.ravel()).reshape(nr, nz)
    Q = mua * (phi + Ec)
    return dict(Q=Q, phi=phi, Ec=Ec, r=rc, z=(np.arange(nz) + 0.5) * dz, absorbed=float((Q * V).sum() / P))
