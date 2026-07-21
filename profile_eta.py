"""Profile likelihood for eta_0 from x-only bounded viscoelastic data.

Asks: with delta0 free and only the parallel displacement observed, how well is
the total viscosity eta_0 = eta_s + eta_p determined? For each trial eta_0 the
remaining parameters (delta0, lambda, and the eta_s/eta_p split) are re-fitted,
so this measures what survives after the model absorbs what it can.
"""

import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

from scipy.optimize import minimize
from packages import InferenceProcedure

ETA_S, ETA_P, LAM, D0 = 1.0, 1.0, 0.5, 0.1
ETA0 = ETA_S + ETA_P


def make(theta, F, tu):
    """One reusable model instance; the Brenner cache is built once."""
    return InferenceProcedure(force=F, a=1.0, theta=theta,
                              material_model="viscoelastic", boundary_model="bounded",
                              delta0=D0, t_unload=tu)


def profile(tag, theta, F, T, N=60, noise_pct=2.0):
    tu = 0.4 * T
    m = make(theta, F, tu)
    t = np.linspace(0.0, T, N)

    def x_of(eta_s, eta_p, lam, d0):
        out = m.model_viscoelastic(eta_s, eta_p, lam, t, F, delta0=d0, component="x")
        return None if out is None else np.asarray(out, float)

    x_true = x_of(ETA_S, ETA_P, LAM, D0)
    y_true = np.asarray(m.model_viscoelastic(ETA_S, ETA_P, LAM, t, F,
                                             delta0=D0, component="y"), float)
    sigma = noise_pct / 100.0 * np.max(np.abs(x_true))
    print(f"\n=== {tag}: theta={theta}, F={F/np.pi:.0f}pi, T={T}, N={N}, "
          f"noise={noise_pct}%, max|y|/delta0={np.max(np.abs(y_true))/D0:.2f}",
          flush=True)

    def chi2_at(eta0):
        def obj(u):
            d0, lam = 10.0**u[0], 10.0**u[1]
            phi = 1.0 / (1.0 + np.exp(-u[2]))
            if not (1e-4 < d0 < 1e3 and 1e-3 < lam < 1e2):
                return 1e12
            x = x_of(phi * eta0, (1.0 - phi) * eta0, lam, d0)
            if x is None or not np.all(np.isfinite(x)):
                return 1e12
            return float(np.sum((x_true - x) ** 2) / sigma**2)

        best = np.inf
        for s in ([np.log10(D0), np.log10(LAM), 0.0],
                  [np.log10(D0) + 0.6, np.log10(LAM) + 0.3, 0.6],
                  [np.log10(D0) - 0.6, np.log10(LAM) - 0.3, -0.6]):
            r = minimize(obj, s, method="Nelder-Mead",
                         options=dict(maxiter=800, fatol=1e-4, xatol=1e-5))
            best = min(best, r.fun)
        return best

    cs = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    vals = np.array([chi2_at(c * ETA0) for c in cs])
    vals -= vals.min()
    for c, v in zip(cs, vals):
        s = np.sqrt(max(v, 0.0))
        verdict = "EXCLUDED" if s > 3 else ("marginal" if s > 1 else "indistinguishable")
        print(f"   eta_0 x {c:<5.2f} -> delta-chi2 {v:9.2f}   {s:5.1f} sigma   {verdict}",
              flush=True)


if __name__ == "__main__":
    profile("standard",   45.0, 12 * np.pi, 0.5)
    profile("long trace", 45.0, 12 * np.pi, 2.0)
    profile("aggressive", 80.0, 60 * np.pi, 2.0)
