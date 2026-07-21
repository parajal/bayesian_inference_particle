"""Prior density for MCMC sampler coordinates."""

import numpy as np


class Priors:
    """Log-uniform material priors and exponential noise/bias priors."""

    def log_prior(self, phi: np.ndarray) -> float:
        """Log prior density of a sampler-space position.

        Material parameters are log-uniform on their prior bounds; the noise and
        (optional) bias scales are exponential. Bounds are validated once in
        ``InferenceProcedure.__init__``, so only the sample is checked here.

        Parameters
        ----------
        phi : numpy.ndarray
            Sampler-space coordinates of length ``self.ndim``.

        Returns
        -------
        float
            Log prior density, or ``-inf`` outside the support.
        """
        phi = np.asarray(phi, dtype=float)
        if not np.all(np.isfinite(phi)):
            return -np.inf

        theta = self._to_physical(phi)
        bounds = np.asarray(self._get_parameter_bounds(), dtype=float)
        lo, hi = bounds[:, 0], bounds[:, 1]
        values = theta[: len(bounds)]
        if np.any((values <= lo) | (values >= hi)):
            return -np.inf

        logp = -np.sum(np.log(np.log10(hi / lo)))
        sigma_noise, sigma_bias = self._extract_noise_bias(theta)
        logp += self._log_exponential_prior(sigma_noise, self.sigma_noise_prior)
        if self._bias_is_inferred():
            logp += self._log_exponential_prior(sigma_bias, self.sigma_bias_prior)
        return float(logp)

    @staticmethod
    def _log_exponential_prior(x: float, rate: float) -> float:
        """Log density of an exponential prior with the given rate.

        Parameters
        ----------
        x : float
            Point at which the density is evaluated.
        rate : float
            Rate parameter, so the mean is ``1 / rate``.

        Returns
        -------
        float
            ``log(rate) - rate * x``, or ``-inf`` for non-positive or
            non-finite arguments.
        """
        if not (x > 0.0 and rate > 0.0 and np.isfinite(x)):
            return -np.inf
        return float(np.log(rate) - rate * x)
