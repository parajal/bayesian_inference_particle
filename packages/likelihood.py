"""
The observation error is the sum of independent measurement noise and a
correlated model discrepancy, so the combined error is zero-mean
Gaussian with covariance Sigma. The residual is taken against the
forward model evaluated at the physical parameters 10^mu_hat, and the
log-likelihood is the Gaussian density at that residual.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

class Likelihood:
    """Implements the likelihood."""

    @staticmethod
    def _covariance_matrix(t, sigma_noise, sigma_bias, l_bias):
        """Sigma = sigma_noise^2 I + sigma_bias^2 K."""
        K = np.exp(-np.abs(np.subtract.outer(t, t)) / l_bias)
        return sigma_noise**2 * np.eye(len(t)) + sigma_bias**2 * K

    @staticmethod
    def _cholesky(covariance):
        """Factorise Sigma = L L^T, or return None if it is not positive definite."""
        if not np.all(np.isfinite(covariance)):
            return None
        try:
            return cho_factor(covariance, lower=True)
        except np.linalg.LinAlgError:
            return None

    @staticmethod
    def _log_likelihood(residual, cholesky=None, sigma_noise=None):
        """-1/2 r^T Sigma^-1 r - 1/2 log|Sigma| - N/2 log(2*pi)."""
        n = len(residual)

        if cholesky is not None:
            r_sq = residual @ cho_solve(cholesky, residual)
            log_sigma = 2.0 * np.sum(np.log(np.diag(cholesky[0])))
        else:
            r_sq = float(residual @ residual) / sigma_noise**2
            log_sigma = 2.0 * n * np.log(sigma_noise)

        return -r_sq/2.0 - log_sigma/2.0 - (n/2.0) * np.log(2.0 * np.pi)

    def _residuals(self, d, theta):
        """r = x_obs - d(10^mu_hat), one entry per fitted component."""
        t = d["t"]
        residuals = []

        for component in self._fit_components(d):
            observed = d.get(component)
            model = self._model_component_at(theta, t, d, component=component)
            residual = np.asarray(observed, dtype=float) - np.asarray(model, dtype=float)
            residuals.append(residual)

        return residuals

    def log_likelihood(self, phi):
        """the loaded trajectory, summed over its fitted components."""
        d = getattr(self, "data", None)

        theta = self._to_physical(phi)
        sigma_noise, sigma_bias = self._extract_noise_bias(theta)
        residuals = self._residuals(d, theta)

        if sigma_bias is None:
            values = [self._log_likelihood(r, sigma_noise=sigma_noise) for r in residuals]
        else:
            covariance = self._covariance_matrix(d["t"], sigma_noise, float(sigma_bias), self.l_bias)
            cholesky = self._cholesky(covariance)
            values = [self._log_likelihood(r, cholesky=cholesky) for r in residuals]

        total = 0.0
        for value in values:
            if not np.isfinite(value):
                return -np.inf
            total += value
        return float(total)

    def log_posterior(self, phi):
        """ log p(theta | x_obs) = log L + log prior, up to a constant."""
        log_prior = self.log_prior(phi)
        if not np.isfinite(log_prior):
            return -np.inf

        log_likelihood = self.log_likelihood(phi)
        if not np.isfinite(log_likelihood):
            return -np.inf

        return float(log_prior + log_likelihood)
