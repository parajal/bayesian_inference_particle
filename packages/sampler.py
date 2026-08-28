import warnings

import emcee
import numpy as np
from sklearn.cluster import KMeans

class Sampler:

    @staticmethod
    def _gelman_rubin_diag(chain):
        n = chain.shape[2]
        W = np.mean(np.var(chain, axis=2, ddof=1), axis=1)
        B = n * np.var(np.mean(chain, axis=2), axis=1, ddof=1)
        V = ((n - 1) / n) * W + B / n
        return np.sqrt(V / W)

    def print_results(self):
        raw = self.sampler.get_chain()
        burn = min(int(self.burn_fraction * raw.shape[0]), raw.shape[0] - 2)
        chain = self._to_physical(raw[burn:]).transpose(2, 1, 0)
        names = self._get_parameter_labels(latex=False)
        flat = chain.reshape(len(names), -1)

        mean = flat.mean(axis=1)
        std = flat.std(axis=1)
        rhat = self._gelman_rubin_diag(chain)
        lo, hi = np.percentile(flat, [2.5, 97.5], axis=1)

        results = {
            n: {
                "mean": float(mean[i]), "std": float(std[i]),
                "Rhat": float(rhat[i]), "ci_low": float(lo[i]), "ci_high": float(hi[i]),
            }
            for i, n in enumerate(names)
        }

        print("\nMCMC summary\n")
        for n in names:
            r = results[n]
            print(f"{n:12s} {r['mean']:.4e} ± {r['std']:.4e} (Rhat={r['Rhat']:.4f})")

        return results

    def _warmup_starting_points(self, p0, moves, progress, warmup_frac, jitter_frac):
        n_warmup = int(warmup_frac * self.nsteps)

        sampler = emcee.EnsembleSampler(
            self.nwalkers, self.ndim, self.log_posterior, moves=moves,
        )
        sampler.run_mcmc(p0, n_warmup, progress=progress)

        chain = sampler.get_chain(flat=True)
        logp = sampler.get_log_prob(flat=True)
        best = chain[np.argsort(logp)[-self.nwalkers:]]

        rng = np.random.default_rng(self.seed)
        scale = jitter_frac * chain.std(axis=0)
        p0 = best + rng.normal(0, scale, size=best.shape)

        print("\nFirst 5 warmup-final walkers, physical coordinates:")
        print(self._to_physical(p0[:5]))

        return p0
    
    def run_mcmc(self, warmup=True, random_state=42, progress=True,
                 warmup_frac=0.1, jitter_frac = 0.1, n_init=10, max_iter=300):
        p0 = self.sample_starting_points(random_state, n_init, max_iter)
        moves = [(emcee.moves.StretchMove(a=2), 1.0)]

        if warmup:
            p0 = self._warmup_starting_points(
                p0, moves, progress, warmup_frac, jitter_frac)

        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_posterior, moves=moves)
        self.sampler.run_mcmc(p0, self.nsteps, progress=progress)

        burn = min(int(self.burn_fraction * self.nsteps), self.nsteps - 2)
        self.samples = self._to_physical(self.sampler.get_chain(discard=burn, flat=True))

        self.print_results()
        return self.samples

    def _prior_draws(self, rng, n):
        bounds = self._get_parameter_bounds()
        phi = np.zeros((n, self.ndim))

        for j, (lo, hi) in enumerate(bounds):
            phi[:, j] = rng.uniform(np.log10(lo), np.log10(hi), n)

        k = len(bounds)
        phi[:, k] = rng.exponential(1 / self.sigma_noise_prior, n)
        if self._bias_is_inferred():
            phi[:, k + 1] = rng.exponential(1 / self.sigma_bias_prior, n)

        return phi

    def sample_starting_points(self, random_state, n_init, max_iter):
        rng = np.random.default_rng(random_state)
        pool = self._prior_draws(rng, self.nwalkers)
        km = KMeans(
            n_clusters=self.nwalkers, n_init=n_init,
            max_iter=max_iter, random_state=random_state,
        ).fit(pool)
        return km.cluster_centers_