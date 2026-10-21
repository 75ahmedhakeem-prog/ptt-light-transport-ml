"""
mc.py - vectorised Monte Carlo photon transport (MCML-type rules) for axisymmetric problems.

Geometry: z points down into the medium (z = 0 is the irradiated surface).
  * Cylinder(R, H, layers=[(z_top, z_bot, mua, mus, g, n), ...], n_out=1.0)
  * Sphere(R, mua, mus, g)      (centre at z = R; index-matched, no internal reflection)
Units: mm and mm^-1.  Absorbed power is tallied on an (r, z) grid -> W mm^-3 for a beam of power P.
Rules: step s = -ln(xi)/mu_t, absorbed fraction mu_a/mu_t, Henyey-Greenstein scattering,
Fresnel reflection at the outer boundaries of the cylinder, Russian roulette (w < 1e-4, survival 0.1).
"""
import numpy as np

def hg_scatter(u, g, rng):
    n = len(u)
    xi = rng.random(n)
    g = np.broadcast_to(np.asarray(g, float), (n,))
    ct = np.where(np.abs(g) > 1e-6,
                  (1 + g**2 - ((1 - g**2) / (1 - g + 2 * g * xi))**2) / (2 * np.where(np.abs(g) > 1e-6, g, 1)),
                  2 * xi - 1)
    ct = np.clip(ct, -1, 1); st = np.sqrt(1 - ct**2)
    phi = 2 * np.pi * rng.random(n); cp, sp = np.cos(phi), np.sin(phi)
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
    den = np.sqrt(np.maximum(1 - uz**2, 1e-12))
    big = np.abs(uz) > 0.99999
    nx = np.where(big, st * cp, st * (ux * uz * cp - uy * sp) / den + ux * ct)
    ny = np.where(big, st * sp, st * (uy * uz * cp + ux * sp) / den + uy * ct)
    nz = np.where(big, np.sign(uz) * ct, -st * cp * den + uz * ct)
    v = np.c_[nx, ny, nz]; return v / np.linalg.norm(v, axis=1)[:, None]

def fresnel(n1, n2, cos_i):
    """unpolarised reflectance; cos_i >= 0"""
    n1 = np.broadcast_to(n1, cos_i.shape).astype(float); n2 = np.broadcast_to(n2, cos_i.shape).astype(float)
    sin_t = n1 / n2 * np.sqrt(np.maximum(1 - cos_i**2, 0))
    R = np.ones_like(cos_i); ok = sin_t < 1
    ct = np.sqrt(1 - sin_t[ok]**2); ci = cos_i[ok]; a, b = n1[ok], n2[ok]
    rs = ((a * ci - b * ct) / (a * ci + b * ct))**2; rp = ((a * ct - b * ci) / (a * ct + b * ci))**2
    R[ok] = 0.5 * (rs + rp); return R

class Cylinder:
    def __init__(self, R, H, layers, n_out=1.0, n_bottom_out=None):
        self.R, self.H, self.layers, self.n_out = R, H, layers, n_out
        self.n_bot = n_out if n_bottom_out is None else n_bottom_out
        self.zb = np.array([l[0] for l in layers] + [layers[-1][1]])
    def props(self, z):
        k = np.clip(np.searchsorted(self.zb, z, side="right") - 1, 0, len(self.layers) - 1)
        L = np.array(self.layers, float)
        return k, L[k, 2], L[k, 3], L[k, 4], L[k, 5]
    def dist(self, p, u):
        """distance to the outer boundary and the boundary id (0 top, 1 bottom, 2 side)"""
        big = 1e30; dz = np.full(len(p), big); bid = np.full(len(p), -1)
        m = u[:, 2] < 0; dz[m] = -p[m, 2] / u[m, 2]
        m2 = u[:, 2] > 0; dz[m2] = (self.H - p[m2, 2]) / u[m2, 2]
        bid = np.where(m, 0, np.where(m2, 1, -1)); d = dz.copy()
        a = u[:, 0]**2 + u[:, 1]**2; bq = 2 * (p[:, 0] * u[:, 0] + p[:, 1] * u[:, 1]); c = p[:, 0]**2 + p[:, 1]**2 - self.R**2
        disc = bq**2 - 4 * a * c; ok = (a > 1e-14) & (disc > 0)
        ds = np.full(len(p), big); ds[ok] = (-bq[ok] + np.sqrt(disc[ok])) / (2 * a[ok])
        side = ds < d; d = np.where(side, ds, d); bid = np.where(side, 2, bid)
        return d, bid
    def dist_layer(self, p, u, k):
        """distance to the next internal layer interface along u"""
        big = 1e30; d = np.full(len(p), big); zb = self.zb
        dn = u[:, 2] > 0; up = u[:, 2] < 0
        zt = zb[k + 1]; m = dn & (k + 1 < len(self.layers)); d[m] = (zt[m] - p[m, 2]) / u[m, 2]
        zt2 = zb[k]; m = up & (k > 0); d[m] = (zt2[m] - p[m, 2]) / u[m, 2]
        return np.maximum(d, 0)
    def normal(self, p, bid):
        nrm = np.zeros_like(p); nrm[bid == 0, 2] = -1; nrm[bid == 1, 2] = 1
        s = bid == 2; r = np.sqrt(p[s, 0]**2 + p[s, 1]**2); nrm[s, 0] = p[s, 0] / r; nrm[s, 1] = p[s, 1] / r
        return nrm
    def n_outside(self, bid):
        return np.where(bid == 1, self.n_bot, self.n_out)

def run_cylinder(geo, beam_radius, P=1.0, N=200000, nr=110, nz=200, seed=1, batch=50000, rmax=None):
    rng = np.random.default_rng(seed); rmax = rmax or geo.R
    A = np.zeros((nr, nz)); esc = np.zeros(3); spec = 0.0; absorbed_total = 0.0
    dr, dz = rmax / nr, geo.H / nz
    for b0 in range(0, N, batch):
        n = min(batch, N - b0)
        rr = beam_radius * np.sqrt(rng.random(n)); ph = 2 * np.pi * rng.random(n)
        p = np.c_[rr * np.cos(ph), rr * np.sin(ph), np.zeros(n)]; u = np.tile([0, 0, 1.0], (n, 1))
        n1 = geo.layers[0][5]; Rsp = ((geo.n_out - n1) / (geo.n_out + n1))**2
        w = np.full(n, 1.0 - Rsp); spec += Rsp * n
        s = np.zeros(n); alive = np.ones(n, bool)
        while alive.any():
            i = np.where(alive)[0]; pi, ui = p[i], u[i]
            need = s[i] <= 0; s[i[need]] = -np.log(rng.random(need.sum()))
            k, mua, mus, g, nl = geo.props(pi[:, 2] + 1e-9 * ui[:, 2])
            mut = mua + mus; step = s[i] / mut
            db, bid = geo.dist(pi, ui); dl = geo.dist_layer(pi, ui, k)
            d = np.minimum(np.minimum(step, db), dl)
            pi = pi + ui * d[:, None]; s[i] -= d * mut
            hitb = (db <= step) & (db <= dl); hitl = (dl < step) & (dl < db) & ~hitb
            inter = ~hitb & ~hitl
            # interaction: absorb and scatter
            j = i[inter]
            if len(j):
                dw = w[j] * mua[inter] / mut[inter]
                r = np.sqrt(pi[inter, 0]**2 + pi[inter, 1]**2); ir = np.minimum((r / dr).astype(int), nr - 1)
                iz = np.minimum((pi[inter, 2] / dz).astype(int), nz - 1)
                np.add.at(A, (ir, iz), dw); absorbed_total += dw.sum()
                w[j] -= dw; ui[inter] = hg_scatter(ui[inter], g[inter], rng); s[j] = 0
            # boundary: Fresnel
            if hitb.any():
                jb = i[hitb]; nrm = geo.normal(pi[hitb], bid[hitb]); cos_i = np.abs((ui[hitb] * nrm).sum(1))
                Rf = fresnel(nl[hitb], geo.n_outside(bid[hitb]), cos_i)
                refl = rng.random(len(jb)) < Rf
                uu = ui[hitb]; uu[refl] = uu[refl] - 2 * (uu[refl] * nrm[refl]).sum(1)[:, None] * nrm[refl]
                ui[hitb] = uu
                out = jb[~refl]; np.add.at(esc, bid[hitb][~refl], w[out]); w[out] = 0; alive[out] = False
            p[i], u[i] = pi, ui
            # roulette
            low = alive & (w < 1e-4)
            if low.any():
                jl = np.where(low)[0]; surv = rng.random(len(jl)) < 0.1
                w[jl[surv]] /= 0.1; w[jl[~surv]] = 0; alive[jl[~surv]] = False
    rc = (np.arange(nr) + 0.5) * dr; vol = 2 * np.pi * rc[:, None] * dr * dz
    Q = A / N * P / vol            # W mm^-3
    return dict(Q=Q, r=rc, z=(np.arange(nz) + 0.5) * dz, R_top=esc[0] / N, T_bot=esc[1] / N, side=esc[2] / N,
                spec=spec / N, absorbed=absorbed_total / N)

def run_sphere(Rs, mua, mus, g, beam_radius, Rspec=0.0298, N=200000, seed=1, batch=50000, n_in=1.0):
    """index-matched sphere (centre z = Rs, top at z = 0) irradiated by a collimated disk from above"""
    rng = np.random.default_rng(seed); absd = 0.0; esc_top = esc_bot = 0.0; mut = mua + mus
    for b0 in range(0, N, batch):
        n = min(batch, N - b0)
        rr = beam_radius * np.sqrt(rng.random(n)); ph = 2 * np.pi * rng.random(n)
        x, y = rr * np.cos(ph), rr * np.sin(ph); z0 = Rs - np.sqrt(Rs**2 - rr**2)
        p = np.c_[x, y, z0]; u = np.tile([0, 0, 1.0], (n, 1)); w = np.full(n, 1 - Rspec)
        alive = np.ones(n, bool)
        while alive.any():
            i = np.where(alive)[0]; pi, ui = p[i], u[i]
            step = -np.log(rng.random(len(i))) / mut
            c = pi - [0, 0, Rs]; bq = (c * ui).sum(1); cc = (c * c).sum(1) - Rs**2
            db = -bq + np.sqrt(np.maximum(bq**2 - cc, 0))
            hit = db <= step; d = np.minimum(step, db); pi = pi + ui * d[:, None]
            if n_in != 1.0 and hit.any():
                nrm = (pi[hit] - [0, 0, Rs]) / Rs; cos_i = np.abs((ui[hit] * nrm).sum(1))
                refl = rng.random(hit.sum()) < fresnel(n_in, 1.0, cos_i)
                uu = ui[hit]; uu[refl] -= 2 * (uu[refl] * nrm[refl]).sum(1)[:, None] * nrm[refl]; ui[hit] = uu
                pi[np.where(hit)[0][refl]] -= 1e-7 * nrm[refl]
                hit2 = hit.copy(); hit2[np.where(hit)[0][refl]] = False
            else:
                hit2 = hit
            jo = i[hit2]; top = pi[hit2, 2] < Rs
            esc_top += w[jo][top].sum(); esc_bot += w[jo][~top].sum(); w[jo] = 0; alive[jo] = False
            ji = i[~hit]; dw = w[ji] * mua / mut; absd += dw.sum(); w[ji] -= dw
            ui[~hit] = hg_scatter(ui[~hit], g, rng); p[i], u[i] = pi, ui
            low = alive & (w < 1e-4)
            if low.any():
                jl = np.where(low)[0]; surv = rng.random(len(jl)) < 0.1
                w[jl[surv]] /= 0.1; w[jl[~surv]] = 0; alive[jl[~surv]] = False
    return dict(absorbed=absd / N, esc_top=esc_top / N, esc_bottom=esc_bot / N, spec=Rspec)
