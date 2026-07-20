"""MCMC sampling and Gelman-Rubin diagnostics."""

from __future__ import annotations

import warnings
from typing import Optional

import emcee
import numpy as np

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None


class Sampler:
    """MCMC sampling, diagnostics, and walker initialisation."""

    _EPS = 1e-15
    _KM_NINIT = 10
    _KM_MAXITER = 200

    # Derived quantities reported alongside the sampled parameters.
    # Each entry: (name, numerator_label, denominator_label, human formula).
    # Labels refer to the non-latex parameter names from _get_parameter_labels.
    _DERIVED_RATIOS = (
        ("G", "eta_p_eff", "lambda", "G = eta_p_eff / lambda"),
    )

    @staticmethod
    def _gelman_rubin_diag(chain: np.ndarray) -> np.ndarray:
        """Gelman-Rubin R-hat for a physical-space chain shaped (p, walkers, steps)."""
        p, M, N = chain.shape
        if M < 2 or N < 2:
            return np.full(p, np.nan)

        W = np.mean(np.var(chain, axis=2, ddof=1), axis=1)
        B = N * np.var(np.mean(chain, axis=2), axis=1, ddof=1)
        var_plus = ((N - 1) / N) * W + B / N
        return np.sqrt((var_plus + Sampler._EPS) / (W + Sampler._EPS))

    def compute_derived_quantities(self) -> dict:
        """Return posterior summaries for simple derived quantities such as G."""
        if getattr(self, "samples", None) is None or len(self.samples) == 0:
            return {}

        labels = self._get_parameter_labels(latex=False)
        samples = np.asarray(self.samples, dtype=float)
        out = {}

        for name, num, den, formula in self._DERIVED_RATIOS:
            if num not in labels or den not in labels:
                continue
            i_num, i_den = labels.index(num), labels.index(den)

            with np.errstate(divide="ignore", invalid="ignore"):
                g = samples[:, i_num] / samples[:, i_den]
            g = g[np.isfinite(g)]
            if g.size == 0:
                continue

            stats = {
                "formula": formula,
                "mean": float(np.mean(g)),
                "median": float(np.median(g)),
                "std": float(np.std(g)),
                "ci95_lo": float(np.percentile(g, 2.5)),
                "ci95_hi": float(np.percentile(g, 97.5)),
            }

            if self.sampler is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    raw = self.sampler.get_chain()
                    burn = min(int(self.burn_fraction * raw.shape[0]), raw.shape[0] - 2)
                    chain = self._to_physical(raw[burn:]).transpose(2, 1, 0)
                    g_chain = (chain[i_num] / chain[i_den])[None, :, :]
                stats["Rhat"] = float(self._gelman_rubin_diag(g_chain)[0])

            out[name] = stats

        return out

    def print_diagnostics(self) -> dict:
        """Compute and print mean, std, and Gelman-Rubin R-hat for every parameter."""
        if self.sampler is None:
            raise RuntimeError("Run MCMC before computing diagnostics.")

        raw = self.sampler.get_chain()
        burn = min(int(self.burn_fraction * raw.shape[0]), raw.shape[0] - 2)
        M = raw.shape[1]
        N = raw.shape[0] - burn

        chain = self._to_physical(raw[burn:]).transpose(2, 1, 0)
        if chain.shape[2] < 3:
            raise RuntimeError("Not enough post burn-in samples for diagnostics.")

        names = self._get_parameter_labels(latex=False)
        means = np.mean(chain, axis=(1, 2))
        stds = np.std(chain, axis=(1, 2), ddof=0)
        rhat = self._gelman_rubin_diag(chain)

        d = {
            name: {
                "mean": float(means[i]),
                "std": float(stds[i]),
                "Rhat": float(rhat[i]),
            }
            for i, name in enumerate(names)
        }

        print("\n" + "=" * 64)
        print("MCMC Diagnostics Summary")
        print("=" * 64)
        print(f"Chains: {M}  |  Samples/chain after burn-in: {N}  |  Total: {M * N}")
        print("-" * 64)
        print(f"{'Parameter':<14s} {'Mean':>14s} {'Std':>14s} {'Rhat':>10s}")
        print("-" * 64)

        for name in names:
            dd = d[name]
            print(f"{name:<14s} {dd['mean']:14.4e} {dd['std']:14.4e} {dd['Rhat']:10.4f}")

        if len(names) > 1:
            vals = list(d.values())
            mean_rhat = np.mean([v["Rhat"] for v in vals])
            std_rhat = np.std([v["Rhat"] for v in vals])
            print("-" * 64)
            print(f"{'Mean Rhat':<14s} {'':>14s} {'':>14s} {mean_rhat:10.4f}")
            print(f"{'Std Rhat':<14s} {'':>14s} {'':>14s} {std_rhat:10.4f}")

        derived = self.compute_derived_quantities()
        if derived:
            print("-" * 64)
            print("Derived quantities")
            for name, s in derived.items():
                print(f"{name:<14s} {s['mean']:14.4e} {s['std']:14.4e} {s.get('Rhat', np.nan):10.4f}")
            for name, s in derived.items():
                print(f"  {s['formula']}")

        print("=" * 64)

        bad = [name for name in names if d[name]["Rhat"] > 1.1]
        if bad:
            print("\nConvergence warnings (Rhat > 1.1):")
            for name in bad:
                print(f"  - {name}: Rhat = {d[name]['Rhat']:.4f}")
        else:
            print("\nAll parameters show good convergence (Rhat <= 1.1)")

        print()
        return d

    @staticmethod
    def _run_sampler(sampler, p0, nsteps: int, label: str, progress: bool = True):
        """Run emcee with a visible tqdm progress bar."""
        if not progress:
            return sampler.run_mcmc(p0, nsteps, progress=False)

        try:
            from tqdm.auto import tqdm
            import sys
        except ImportError:
            return sampler.run_mcmc(p0, nsteps, progress=True)

        state = None
        with tqdm(total=nsteps, desc=label, dynamic_ncols=True, file=sys.stdout) as bar:
            for state in sampler.sample(p0, iterations=nsteps, progress=False):
                bar.update(1)
        return state

    def _warmup_starting_points(self, p0, moves, random_state, progress):
        """Run a short chain and restart near the best warm-up samples."""
        n_warmup = int(0.1 * self.nsteps)

        print(f"\nWarm-up: {n_warmup} steps x {self.nwalkers} walkers")
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_posterior, moves=moves)
        self._run_sampler(sampler, p0, n_warmup, "Warm-up", progress)

        chain = sampler.get_chain(flat=True)
        log_prob = sampler.get_log_prob(flat=True)
        keep = np.isfinite(log_prob) & np.all(np.isfinite(chain), axis=1)
        if np.count_nonzero(keep) < self.nwalkers:
            warnings.warn("Warm-up did not find enough finite samples; using original walkers.")
            return p0

        chain = chain[keep]
        log_prob = log_prob[keep]
        best = chain[np.argsort(log_prob)[-self.nwalkers:]]
        scale = np.maximum(0.1 * np.std(chain, axis=0), 1e-8)
        rng = np.random.default_rng(None if random_state is None else random_state + 1)
        p0 = best + rng.normal(0.0, scale, size=best.shape)

        # Noise parameters are sampled directly in physical units, so keep them positive.
        for j in np.where(~self._get_log10_mask())[0]:
            p0[:, j] = np.maximum(np.abs(p0[:, j]), 1e-300)
        return p0

    def run_mcmc(
        self,
        init_method: str = "kmeans",
        warmup: bool = True,
        random_state: Optional[int] = 42,
        progress: bool = True,
    ) -> np.ndarray:
        """
        Execute MCMC sampling using emcee ensemble sampler.

        Parameters
        ----------
        init_method : str
            Walker initialization method ("random" or "kmeans").
        warmup : bool
            If True, run a short preliminary chain and reinitialize walkers
            around the highest-posterior positions before the main run.
        random_state : int or None
            Seed for reproducibility.

        Returns
        -------
        samples : np.ndarray, shape (nsamples, ndim) in physical space
        """
        self.init_method = init_method
        self.ndim = self._get_ndim()
        print(
            f"\nPreparing MCMC: ndim={self.ndim}, walkers={self.nwalkers}, "
            f"steps={self.nsteps}, init={init_method}, warmup={warmup}",
            flush=True,
        )
        p0 = self.sample_starting_points(method=init_method, random_state=random_state)

        print("\nInitial walkers (physical):")
        for i in range(min(10, self.nwalkers)):
            print(f"  {i:02d}: {self._to_physical(p0[i])}")

        moves = [
            (emcee.moves.StretchMove(a=2.0), 0.6),
            (emcee.moves.DEMove(), 0.25),
            (emcee.moves.GaussianMove(0.01), 0.15),
        ]

        if warmup:
            p0 = self._warmup_starting_points(p0, moves, random_state, progress)
            print("\nWalkers after warm-up (physical):")
            for i in range(min(10, self.nwalkers)):
                print(f"  {i:02d}: {self._to_physical(p0[i])}")

        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_posterior, moves=moves)
        self._run_sampler(self.sampler, p0, self.nsteps, "MCMC", progress)

        burn = min(int(self.burn_fraction * self.nsteps), self.nsteps - 2)
        self.samples = self._to_physical(self.sampler.get_chain(discard=burn, flat=True))
        self.print_diagnostics()

        return self.samples

    def sample_starting_points(
        self,
        n_candidates: Optional[int] = None,
        method: str = "kmeans",
        random_state: Optional[int] = 42,
    ) -> np.ndarray:
        """
        Generate starting points for MCMC walkers.

        Samples bounded parameters in log10 space for log-uniform priors,
        samples physical-space nuisance parameters from their configured
        positive priors, then applies KMeans (or random) selection.

        Returns
        -------
        p0 : np.ndarray, shape (nwalkers, ndim)
            Walker positions in sampler space, with all sampled positive
            parameters log10-transformed.
        """
        method = str(method).lower()
        if method not in ("random", "kmeans"):
            raise ValueError("method must be 'random' or 'kmeans'")
        if method == "kmeans" and KMeans is None:
            warnings.warn(
                "scikit-learn is not installed; falling back to random initialization.",
                RuntimeWarning, stacklevel=2)
            method = "random"

        ndim = self._get_ndim()
        phys_bounds = self._get_parameter_bounds()
        rng = np.random.default_rng(random_state)

        n_draw = self.nwalkers if method == "random" else (
            int(n_candidates) if n_candidates is not None else 100 * self.nwalkers
        )

        phi = np.zeros((n_draw, ndim), dtype=float)

        # Bounded material parameters are drawn uniformly in log10 space.
        mask = self._get_log10_mask()
        for j, (lo, hi) in enumerate(phys_bounds):
            if lo <= 0 or hi <= 0 or hi <= lo:
                raise ValueError(f"Invalid bounds for parameter {j}: ({lo}, {hi})")
            if mask[j]:
                phi[:, j] = rng.uniform(np.log10(lo), np.log10(hi), size=n_draw)
            else:
                phi[:, j] = rng.uniform(lo, hi, size=n_draw)

        idx_noise = len(phys_bounds)
        idx_bias = idx_noise + 1 if self._bias_is_inferred() else None

        if not (np.isfinite(self.sigma_noise_prior) and self.sigma_noise_prior > 0):
            raise ValueError("sigma_noise_prior must be positive.")
        phi[:, idx_noise] = rng.exponential(
            scale=1.0 / self.sigma_noise_prior,
            size=n_draw,
        )

        # sigma_bias prior (physical, linear coordinate: phi = sigma)
        if idx_bias is not None:
            if not (np.isfinite(self.sigma_bias_prior) and self.sigma_bias_prior > 0):
                raise ValueError("sigma_bias_prior must be positive.")
            phi[:, idx_bias] = rng.exponential(
                scale=1.0 / self.sigma_bias_prior,
                size=n_draw,
            )

        if method == "random":
            return phi

        km = KMeans(
            n_clusters=self.nwalkers,
            n_init=self._KM_NINIT,
            max_iter=self._KM_MAXITER,
            random_state=random_state,
        )
        km.fit(phi)
        return km.cluster_centers_
