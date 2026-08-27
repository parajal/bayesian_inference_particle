import numpy as np


class Priors:
    """Log-uniform material priors and exponential noise/bias priors."""

    @staticmethod
    def _log_exponential_prior(x, rate):
        if x > 0 and rate > 0 and np.isfinite(x):
            return np.log(rate) - rate * x
        return -np.inf

    def log_prior(self, phi):
        phi = np.asarray(phi, float)
        if not np.isfinite(phi).all():
            return -np.inf

        theta = self._to_physical(phi)
        bounds = np.asarray(self._get_parameter_bounds())
        values = theta[:len(bounds)]
        if np.any((values <= bounds[:, 0]) | (values > bounds[:, 1])):
            return -np.inf

        logp = -np.sum(np.log(np.log10(bounds[:, 1] / bounds[:, 0])))
        sigma_noise, sigma_bias = self._extract_noise_bias(theta)
        logp += self._log_exponential_prior(sigma_noise, self.sigma_noise_prior)

        if self._bias_is_inferred():
            logp += self._log_exponential_prior(sigma_bias, self.sigma_bias_prior)

        return float(logp)