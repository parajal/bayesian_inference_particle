"""Likelihood and posterior (manuscript Section 5.2).

The observation error is the sum of independent measurement noise and a
correlated model discrepancy, Eq. (27), so the combined error is zero-mean
Gaussian with covariance Sigma, Eq. (30).  The residual is taken against the
forward model evaluated at the physical parameters 10^mu_hat, Eq. (31), and the
log-likelihood is the Gaussian density at that residual, Eq. (33).

Manuscript symbols map to the code as: sigma_exp -> sigma_noise,
sigma_bias -> sigma_bias, l_bias -> self.l_bias, x_obs - d(10^mu_hat) -> residual.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

#: Largest standard deviation accepted before Sigma is treated as degenerate.
_MAX_SIGMA = 1e150


class Likelihood:
    """Implements the manuscript likelihood in direct matrix form."""

    def _bias_correlation_matrix(self, t_left, l_bias, t_right=None):
        """K_ij = exp(-|t_i - t_j| / l_bias)."""
        l_bias = float(l_bias)
        t_left = np.asarray(t_left, dtype=float)
        t_right = t_left if t_right is None else np.asarray(t_right, dtype=float)
        return np.exp(-np.abs(np.subtract.outer(t_left, t_right)) / l_bias)

    def _covariance_matrix(self, t, sigma_noise, sigma_bias):
        """Sigma = sigma_noise^2 I + sigma_bias^2 K."""
        K = self._bias_correlation_matrix(t, self.l_bias)
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
    def _gaussian_log_likelihood(residual, cholesky):
        """-1/2 r^T Sigma^-1 r - 1/2 log|Sigma| - N/2 log(2*pi)."""
        quad = residual @ cho_solve(cholesky, residual)
        if not np.isfinite(quad):
            return -np.inf
        logdet = 2.0 * np.sum(np.log(np.diag(cholesky[0])))
        if not np.isfinite(logdet):
            return -np.inf

        return -0.5 * quad - 0.5 * logdet - 0.5 * len(residual) * np.log(2.0 * np.pi)

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

        sigma_noise = 0.0 if sigma_noise is None else float(sigma_noise)
        sigma_bias = 0.0 if sigma_bias is None else float(sigma_bias)

        residuals = self._residuals(d, theta)

        # Sigma depends only on t, so factorise once and reuse for every component.
        covariance = self._covariance_matrix(d["t"], sigma_noise, sigma_bias)
        cholesky = self._cholesky(covariance)

        total = 0.0
        for residual in residuals:
            value = self._gaussian_log_likelihood(residual, cholesky)
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
