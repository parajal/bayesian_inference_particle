"""MCMC sampling and Gelman-Rubin diagnostics."""

import warnings

import emcee
import numpy as np
from sklearn.cluster import KMeans


class Sampler:
    """MCMC sampling, diagnostics, and walker initialisation."""

    _KM_NINIT = 10
    _KM_MAXITER = 300

    # ---------------------------------------------------------------- diagnostics
    @staticmethod
    def _gelman_rubin_diag(chain: np.ndarray) -> np.ndarray:
        """Gelman-Rubin R-hat statistic for each parameter.

        Parameters
        ----------
        chain : numpy.ndarray
            Chain shaped ``(n_params, n_chains, n_samples)``.

        Returns
        -------
        numpy.ndarray
            One R-hat per parameter; values near 1 indicate convergence.
        """
        n_samples = chain.shape[2]

        within = np.mean(np.var(chain, axis=2, ddof=1), axis=1)
        between = n_samples * np.var(np.mean(chain, axis=2), axis=1, ddof=1)
        var_hat = ((n_samples - 1) / n_samples) * within + between / n_samples
        return np.sqrt(var_hat / within)

    def print_results(self, cred_level: float = 0.95) -> dict:
        """Print and return posterior summaries per parameter.

        Parameters
        ----------
        cred_level : float, optional
            Central credible level for the reported interval (default 0.95).

        Returns
        -------
        dict
            Maps each parameter name to a dict with keys ``"mean"``, ``"std"``,
            ``"Rhat"``, ``"ci_low"`` and ``"ci_high"`` (the equal-tailed
            ``cred_level`` credible interval).
        """
        raw = self.sampler.get_chain()
        n_chains = raw.shape[1]
        burn = min(int(self.burn_fraction * raw.shape[0]), raw.shape[0] - 2)
        n_kept = raw.shape[0] - burn

        # Physical-space chain shaped (n_params, n_chains, n_samples).
        chain = self._to_physical(raw[burn:]).transpose(2, 1, 0)

        names = self._get_parameter_labels(latex=False)
        flat = chain.reshape(len(names), -1)
        means = np.mean(flat, axis=1)
        stds = np.std(flat, axis=1, ddof=0)
        rhat = self._gelman_rubin_diag(chain)
        q_lo, q_hi = 50.0 * (1.0 - cred_level), 50.0 * (1.0 + cred_level)
        ci_low, ci_high = np.percentile(flat, [q_lo, q_hi], axis=1)

        results = {
            name: {"mean": float(means[i]), "std": float(stds[i]),
                   "Rhat": float(rhat[i]),
                   "ci_low": float(ci_low[i]), "ci_high": float(ci_high[i])}
            for i, name in enumerate(names)
        }

        pct = int(round(100 * cred_level))
        print("\n" + "=" * 78)
        print("MCMC Diagnostics Summary")
        print("=" * 78)
        print(f"Chains: {n_chains}  |  Samples/chain after burn-in: {n_kept}  "
              f"|  Total: {n_chains * n_kept}")
        print("-" * 78)
        print(f"{'Parameter':<14s} {'Mean':>12s} {'Std':>12s} {'Rhat':>8s} "
              f"{f'{pct}% CI low':>13s} {f'{pct}% CI high':>13s}")
        print("-" * 78)
        for name in names:
            r = results[name]
            print(f"{name:<14s} {r['mean']:12.4e} {r['std']:12.4e} {r['Rhat']:8.4f} "
                  f"{r['ci_low']:13.4e} {r['ci_high']:13.4e}")

        if len(names) > 1:
            all_rhat = [r["Rhat"] for r in results.values()]
            print("-" * 78)
            print(f"{'Mean Rhat':<14s} {np.mean(all_rhat):>47.4f}")
            print(f"{'Std Rhat':<14s} {np.std(all_rhat):>47.4f}")

        return results

    def _warmup_starting_points(self, p0, moves, random_state, progress):
        """Run a short chain and restart walkers near the best warm-up samples.

        Parameters
        ----------
        p0 : numpy.ndarray
            Initial walker positions, shape ``(nwalkers, ndim)``.
        moves : list
            emcee move mixture.
        random_state : int or None
            Seed for the jitter applied to the restarted walkers.
        progress : bool
            Show the emcee progress bar.

        Returns
        -------
        numpy.ndarray
            Restarted walker positions, or ``p0`` unchanged if the warm-up did
            not produce enough finite samples.
        """
        n_warmup = int(0.1 * self.nsteps)
        print(f"\nWarm-up: {n_warmup} steps x {self.nwalkers} walkers")

        sampler = emcee.EnsembleSampler(
            self.nwalkers, self.ndim, self.log_posterior, moves=moves
        )
        sampler.run_mcmc(p0, n_warmup, progress=progress)

        chain = sampler.get_chain(flat=True)
        log_prob = sampler.get_log_prob(flat=True)

        keep = np.isfinite(log_prob) & np.all(np.isfinite(chain), axis=1)
        if np.count_nonzero(keep) < self.nwalkers:
            warnings.warn("Warm-up did not find enough finite samples; using original walkers.")
            return p0
        chain, log_prob = chain[keep], log_prob[keep]

        # Restart from the highest-posterior samples, jittered by each parameter's spread.
        best = chain[np.argsort(log_prob)[-self.nwalkers:]]
        scale = np.maximum(0.1 * np.std(chain, axis=0), 1e-8)
        rng = np.random.default_rng(None if random_state is None else random_state + 1)
        p0 = best + rng.normal(0.0, scale, size=best.shape)

        # Noise parameters live in physical units, so keep them strictly positive.
        for j in np.where(~self._get_log10_mask())[0]:
            p0[:, j] = np.maximum(np.abs(p0[:, j]), 1e-300)
        return p0

    # ----------------------------------------------------------------- main driver
    def run_mcmc(
        self,
        warmup: bool = True,
        random_state: int | None = 42,
        progress: bool = True,
    ) -> np.ndarray:
        """Run the emcee ensemble sampler and return samples in physical space.

        Walkers are seeded at the k-means centres of a prior pool; see
        :meth:`sample_starting_points`.

        Parameters
        ----------
        warmup : bool, optional
            If ``True``, run a short preliminary chain and reinitialise walkers
            around the highest-posterior positions before the main run.
        random_state : int or None, optional
            Seed for reproducibility.
        progress : bool, optional
            Show the emcee progress bar.

        Returns
        -------
        numpy.ndarray
            Post-burn-in samples in physical space, shape ``(nsamples, ndim)``.
        """
        print(
            f"\nPreparing MCMC: ndim={self.ndim}, walkers={self.nwalkers}, "
            f"steps={self.nsteps}, warmup={warmup}",
            flush=True,
        )

        p0 = self.sample_starting_points(random_state=random_state)
        self._print_walkers("Initial walkers (physical):", p0)

        moves = [
            (emcee.moves.StretchMove(a=2.0), 0.6),
            (emcee.moves.DEMove(), 0.20),
            (emcee.moves.GaussianMove(0.01), 0.20),
        ]

        if warmup:
            p0 = self._warmup_starting_points(p0, moves, random_state, progress)
            self._print_walkers("Walkers after warm-up (physical):", p0)

        self.sampler = emcee.EnsembleSampler(
            self.nwalkers, self.ndim, self.log_posterior, moves=moves
        )
        self.sampler.run_mcmc(p0, self.nsteps, progress=progress)

        burn = min(int(self.burn_fraction * self.nsteps), self.nsteps - 2)
        self.samples = self._to_physical(
            self.sampler.get_chain(discard=burn, flat=True)
        )
        self.print_results()
        return self.samples

    def _print_walkers(self, header, p0, limit=10):
        """Print the first few walker positions in physical space.

        Parameters
        ----------
        header : str
            Line printed above the positions.
        p0 : numpy.ndarray
            Walker positions in sampler space.
        limit : int, optional
            Maximum number of walkers shown.

        Returns
        -------
        None
        """
        print(f"\n{header}")
        for i in range(min(limit, self.nwalkers)):
            print(f"  {i:02d}: {self._to_physical(p0[i])}")

    # ---------------------------------------------------------- walker initialisation
    def _prior_draws(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``n`` samples from the priors in sampler space.

        Material parameters are drawn log-uniformly (so uniformly in the log10
        sampler coordinate); the nuisance parameters (noise and optional bias)
        are drawn from their exponential priors in physical units.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random generator.
        n : int
            Number of samples.

        Returns
        -------
        numpy.ndarray
            Prior samples, shape ``(n, ndim)``.
        """
        bounds = self._get_parameter_bounds()
        phi = np.zeros((n, self.ndim), dtype=float)
        for j, (lo, hi) in enumerate(bounds):
            phi[:, j] = rng.uniform(np.log10(lo), np.log10(hi), size=n)
        idx = len(bounds)
        phi[:, idx] = rng.exponential(1.0 / self.sigma_noise_prior, n)
        if self._bias_is_inferred():
            phi[:, idx + 1] = rng.exponential(1.0 / self.sigma_bias_prior, n)
        return phi

    def sample_starting_points(self, random_state: int | None = 42,
                               pool_factor: int = 20) -> np.ndarray:
        """Seed the walkers at the k-means centres of a prior pool.

        A pool of ``pool_factor * nwalkers`` prior draws is grouped into
        ``nwalkers`` k-means clusters, and the cluster centres are used as the
        starting positions. This spreads the walkers evenly across the prior
        support rather than scattering them at random. Any centre that the
        posterior rejects is replaced by the nearest pool point, so every walker
        starts with a finite log-posterior.

        Parameters
        ----------
        random_state : int or None, optional
            Seed for the pool draw and the k-means initialisation.
        pool_factor : int, optional
            Pool size as a multiple of ``nwalkers``.

        Returns
        -------
        numpy.ndarray
            Walker positions in sampler space, shape ``(nwalkers, ndim)``.
        """
        rng = np.random.default_rng(random_state)
        phi = self._prior_draws(rng, pool_factor * self.nwalkers)

        km = KMeans(
            n_clusters=self.nwalkers,
            n_init=self._KM_NINIT,
            max_iter=self._KM_MAXITER,
            random_state=random_state,
        )
        km.fit(phi)
        centers = km.cluster_centers_

        # Replace any centre the posterior rejects with the nearest finite pool point.
        bad = [k for k, c in enumerate(centers) if not np.isfinite(self.log_posterior(c))]
        if bad:
            finite = np.array([np.isfinite(self.log_posterior(p)) for p in phi])
            if not finite.any():
                raise RuntimeError("No prior draw has a finite posterior; check priors and data.")
            good = phi[finite]
            for k in bad:
                centers[k] = good[np.argmin(np.linalg.norm(good - centers[k], axis=1))]
        return centers