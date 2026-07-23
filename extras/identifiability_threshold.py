
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

def _find_root():
    starts = []
    try:
        starts.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:          # e.g. a Jupyter cell, where __file__ is undefined
        pass
    starts.append(os.getcwd())

    for start in starts:
        d = start
        for _ in range(8):
            if os.path.isdir(os.path.join(d, "data", "newtonian", "angled_delta")):
                return d
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    raise FileNotFoundError(
        "Could not locate the repository root (no data/newtonian/angled_delta "
        "found above the script location or the current directory). "
        "Run from within the project, or set ROOT manually."
    )


ROOT = _find_root()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from packages import InferenceProcedure  # noqa: E402

FORCE = 12 * np.pi
DATA = os.path.join(ROOT, "data", "newtonian", "angled_delta", "angle-{}.txt")
OUTDIR = os.path.join(ROOT, "plots", "angle_study")

SNR_THRESHOLD = 1.0

TRUE_VALUES = {"eta_s": 1.0, "delta0": 0.1}  
RHAT_MAX = 1.05                  
REL_TOL = 0.50             

def bend_rms(t, x):
    basis = np.vstack([np.ones_like(t), t]).T
    coef, *_ = np.linalg.lstsq(basis, x, rcond=None)
    return float(np.sqrt(np.mean((x - basis @ coef) ** 2)))

def curvature_to_noise(t, x_clean, sigma_noise_percent):
    sigma = (sigma_noise_percent / 100.0) * np.max(np.abs(x_clean))
    return bend_rms(t, x_clean) / sigma if sigma > 0 else np.inf

def sweep(angles, noises, with_mcmc=True, nsteps=2000):
    """Grid-scan identifiability over (angle, noise), skipping runs by monotonicity.

    Identifiability is monotonic in the signal-to-noise ratio, which gives two
    short-circuits:

    * within one angle, once a noise level fails, every higher noise fails too;
    * across angles (processed in order of increasing curvature), any noise at or
      below the previous angle's highest identifiable noise is guaranteed
      identifiable for the next, higher-curvature angle -- so it is not sampled.

    Skipped runs carry ``skipped=True`` and ``nan`` for the MCMC columns, with
    ``skip_reason`` recording which rule fired. Records are returned sorted by
    (theta, noise) regardless of the processing order.

    Returns
    -------
    list of dict
        One record per (angle, noise).
    """
    noises = sorted(noises)
    # Curvature per angle sets how much noise it tolerates; process weakest first.
    grid, curv = {}, {}
    for theta in angles:
        d = np.loadtxt(DATA.format(theta))
        t, x = d[:, 0], d[:, 1] - d[0, 1]
        grid[theta] = (t, x)
        curv[theta] = 100.0 * bend_rms(t, x) / np.max(np.abs(x))

    records = []
    guaranteed = -np.inf   # every noise <= this is identifiable for the next angle
    for theta in sorted(angles, key=lambda a: curv[a]):
        t, x = grid[theta]
        cpct = curv[theta]
        failed = False        # within this angle a lower noise already failed
        last_ident = guaranteed
        nan = {p: np.nan for p in TRUE_VALUES}
        for npct in noises:
            snr = curvature_to_noise(t, x, npct)
            rec = dict(theta=theta, noise=npct, curvature_percent=cpct, snr=snr,
                       predicted=snr > SNR_THRESHOLD, skipped=False, skip_reason="")
            if with_mcmc:
                if npct <= guaranteed:
                    rec.update(converged=True, accurate=True, identifiable=True,
                               mean=dict(nan), rhat=dict(nan), rel_err=dict(nan),
                               skipped=True, skip_reason="lower angle already identifiable")
                    last_ident = npct
                elif failed:
                    rec.update(converged=False, accurate=False, identifiable=False,
                               mean=dict(nan), rhat=dict(nan), rel_err=dict(nan),
                               skipped=True, skip_reason="lower noise already failed")
                else:
                    rec.update(mcmc_identifiable(theta, npct, nsteps=nsteps))
                    if rec["identifiable"]:
                        last_ident = npct
                    else:
                        failed = True
                rec["correct"] = (rec["predicted"] == rec["identifiable"])
            records.append(rec)

            if with_mcmc and rec["skipped"]:
                verdict = (f"{'IDENTIFIABLE' if rec['identifiable'] else 'NOT IDENTIFIABLE'}"
                           f" (skipped: {rec['skip_reason']})")
            elif with_mcmc:
                verdict = "IDENTIFIABLE" if rec["identifiable"] else "NOT IDENTIFIABLE"
                reasons = [r for r, ok in (("not converged", rec["converged"]),
                                           ("not accurate", rec["accurate"])) if not ok]
                if reasons:
                    verdict += " (" + ", ".join(reasons) + ")"
                rh = ", ".join(f"{k} {v:.3f}" for k, v in rec["rhat"].items())
                re = ", ".join(f"{k} {100*v:.1f}%" for k, v in rec["rel_err"].items())
                verdict += f"  [Rhat: {rh}] [err: {re}]"
            else:
                verdict = ""
            print(f"theta={theta:>3} noise={npct:>4}%  SNR={snr:>6.2f}  "
                  f"predicted={'ident' if rec['predicted'] else 'not'}  ->  {verdict}",
                  flush=True)
        guaranteed = max(guaranteed, last_ident)   # non-decreasing with curvature

    records.sort(key=lambda r: (r["theta"], r["noise"]))
    return records


def save_records(records, path=None):
    """Write every (theta, noise) result to a whitespace-delimited text table.

    One row per grid cell, with the curvature, SNR, inferred parameter means and
    stds, R-hat, relative errors, and the boolean verdicts -- ready to load with
    ``numpy.genfromtxt(path, names=True)`` for later plotting.

    Parameters
    ----------
    records : list of dict
        Output of :func:`sweep` (run with ``with_mcmc=True``).
    path : str, optional
        Destination file. Defaults to ``plots/angle_study/identifiability_results.txt``.

    Returns
    -------
    str
        The path written.
    """
    if path is None:
        os.makedirs(OUTDIR, exist_ok=True)
        path = os.path.join(OUTDIR, "identifiability_results.txt")

    params = list(TRUE_VALUES)
    cols = ["theta", "noise_percent", "curvature_percent", "snr"]
    for p in params:
        cols += [f"{p}_mean", f"{p}_rhat", f"{p}_rel_err"]
    cols += ["converged", "accurate", "identifiable", "predicted", "skipped"]

    def cell(v):
        if isinstance(v, bool):
            return "T" if v else "F"
        return f"{v:.6g}"

    def row_values(r):
        vals = [r["theta"], r["noise"], r["curvature_percent"], r["snr"]]
        for p in params:
            vals += [r["mean"][p], r["rhat"][p], r["rel_err"][p]]
        return vals + [r["converged"], r["accurate"], r["identifiable"],
                       r["predicted"], r["skipped"]]

    rows = [[cell(v) for v in row_values(r)] for r in records]
    widths = [max(len(cols[i]), *(len(row[i]) for row in rows)) for i in range(len(cols))]

    with open(path, "w") as fh:
        fh.write("  ".join(c.rjust(w) for c, w in zip(cols, widths)) + "\n")
        for row in rows:
            fh.write("  ".join(v.rjust(w) for v, w in zip(row, widths)) + "\n")
    print(f"Saved results table: {path}")
    return path


def mcmc_identifiable(theta, sigma_noise_percent, nsteps, seed=42):
    m = InferenceProcedure(
        force=FORCE, a=1.0, theta=theta, material_model="newtonian",
        boundary_model="bounded", delta0=0.1,
        sigma_noise_percent=sigma_noise_percent, seed=seed,
        nsteps=nsteps, thin_factor=4, nwalkers = 20, sigma_bias = None
    )
    m.load_data(DATA.format(theta))
    m.run_mcmc(warmup=True, progress=True)
    results = m.print_results()

    mean = {p: results[p]["mean"] for p in TRUE_VALUES}
    std = {p: results[p]["std"] for p in TRUE_VALUES}
    rhat = {p: results[p]["Rhat"] for p in TRUE_VALUES}
    rel_err = {p: abs(mean[p] - t) / t for p, t in TRUE_VALUES.items()}
    converged = all(v <= RHAT_MAX for v in rhat.values())
    accurate = all(v <= REL_TOL for v in rel_err.values())
    return dict(converged=converged, accurate=accurate,
                identifiable=converged and accurate,
                mean=mean, std=std, rhat=rhat, rel_err=rel_err)


def main():
    angles = [ 5, 10, 20, 30, 45, 75]
    noises = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 9.0, 10.0]
    # angles = [ 5, 75]
    # noises = [0.1, 10.0]
    records = sweep(angles, noises, with_mcmc=True, nsteps = 15000)

    correct = [r for r in records if r.get("correct")]
    print(f"\nClassifier agreement with MCMC: {len(correct)}/{len(records)} "
          f"at SNR threshold {SNR_THRESHOLD}")
    save_records(records)
    return records


if __name__ == "__main__":
    main()
