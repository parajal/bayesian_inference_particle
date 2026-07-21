"""MCMC sampling and Gelman-Rubin diagnostics."""

import warnings
from typing import Optional
import emcee
import numpy as np
from sklearn.cluster import KMeans
import arviz as az

class Sampler:
    """MCMC sampling, diagnostics, and walker initialisation."""

    @staticmethod
    def _gelman_rubin_diag(chain: np.ndarray) -> np.ndarray:
        """Gelman-Rubin R-hat for a physical-space chain shaped (p, chains, length).
        """
        p, m, n = chain.shape

        W = np.mean(np.var(chain, axis=2, ddof=1), axis=1)
        B = n * np.var(np.mean(chain, axis=2), axis=1, ddof=1)
        var_hat = ((n - 1) / n) * W + B / n
        return np.sqrt(var_hat / W)

    def print_results(self) -> dict:
        """Compute and print mean, std, and Gelman-Rubin R-hat for every parameter."""

        raw = self.sampler.get_chain()
        burn = min(int(self.burn_fraction * raw.shape[0]), raw.shape[0] - 2)
        M = raw.shape[1]
        N = raw.shape[0] - burn

        chain = self._to_physical(raw[burn:]).transpose(2, 1, 0)

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

    def _warmup_starting_points(self, p0, moves, random_state, progress):
        """Run a short chain and restart near the best warm-up samples."""
        n_warmup = int(0.1 * self.nsteps)

        print(f"\nWarm-up: {n_warmup} steps x {self.nwalkers} walkers")
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_posterior, moves=moves)
        sampler.run_mcmc(p0, n_warmup, progress=True)

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
            (emcee.moves.DEMove(), 0.20),
            (emcee.moves.GaussianMove(0.01), 0.20),
        ]

        if warmup:
            p0 = self._warmup_starting_points(p0, moves, random_state, progress)
            print("\nWalkers after warm-up (physical):")
            for i in range(min(10, self.nwalkers)):
                print(f"  {i:02d}: {self._to_physical(p0[i])}")

        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_posterior, moves=moves)
        self.sampler.run_mcmc(p0, self.nsteps, progress=True)
        burn = min(int(self.burn_fraction * self.nsteps), self.nsteps - 2)
        self.samples = self._to_physical(self.sampler.get_chain(discard=burn, flat=True))
        self.print_results()

        return self.samples

    def sample_starting_points(
        self, method: str = "kmeans", random_state: Optional[int] = 42) -> np.ndarray:
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

        phys_bounds = self._get_parameter_bounds()
        rng = np.random.default_rng(random_state)
        phi = np.zeros((self.nwalkers, self._get_ndim()), dtype=float)

        mask = self._get_log10_mask()
        for j, (lo, hi) in enumerate(phys_bounds):
            if lo <= 0 or hi <= 0 or hi <= lo:
                raise ValueError(f"Invalid bounds for parameter {j}: ({lo}, {hi})")
            if mask[j]:
                phi[:, j] = rng.uniform(np.log10(lo), np.log10(hi), size=self.nwalkers)
            else:
                phi[:, j] = rng.uniform(lo, hi, size=self.nwalkers)

        idx_noise = len(phys_bounds)
        idx_bias = idx_noise + 1 if self._bias_is_inferred() else None

        phi[:, idx_noise] = rng.exponential(scale=1.0 / self.sigma_noise_prior, size=self.nwalkers,)

        if idx_bias is not None:
            phi[:, idx_bias] = rng.exponential(scale=1.0 / self.sigma_bias_prior, size=self.nwalkers,)
        
        if method == "random":
            return phi

        km = KMeans(n_clusters=self.nwalkers, random_state=random_state)
        km.fit(phi)
        return km.cluster_centers_
