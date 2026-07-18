"""Prior density for MCMC sampler coordinates."""

import math
import numpy as np


class Priors:
    """Log-uniform material priors and exponential noise/bias priors."""

    def log_prior(self, phi: np.ndarray) -> float:
        phi = np.asarray(phi, dtype=float)
        if not np.all(np.isfinite(phi)):
            return -np.inf

        theta = self._to_physical(phi)
        bounds = np.asarray(self._get_parameter_bounds(), dtype=float)
        values = theta[: len(bounds)]

        if np.any(
            (bounds[:, 0] <= 0.0)
            | (bounds[:, 1] <= bounds[:, 0])
            | (values <= bounds[:, 0])
            | (values >= bounds[:, 1])
        ):
            return -np.inf

        logp = -sum(math.log(math.log10(hi) - math.log10(lo)) for lo, hi in bounds)
        sigma_noise, sigma_bias = self._extract_noise_bias(theta)
        logp += self._log_exponential_prior(sigma_noise, self.sigma_noise_prior)
        if self._bias_is_inferred():
            logp += self._log_exponential_prior(sigma_bias, self.sigma_bias_prior)
        return float(logp)

    @staticmethod
    def _log_exponential_prior(x: float, rate: float) -> float:
        x = float(x)
        rate = float(rate)
        if x <= 0.0 or rate <= 0.0 or not np.isfinite(x + rate):
            return -np.inf
        return math.log(rate) - rate * x
