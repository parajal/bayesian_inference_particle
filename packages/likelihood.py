"""Gaussian likelihood for the particle-trajectory inference.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


class Likelihood:
    """Gaussian log-likelihood and log-posterior for one loaded trajectory."""

    def _fit_components(self, d):
        """Return the tuple of components to fit for dataset ``d``.

        Parameters
        ----------
        d : dict
            Dataset dict; ``d["fit_components"]`` is used when present.

        Returns
        -------
        tuple of str
            The fitted components, defaulting to the pull geometry.
        """
        default = "y" if self.is_perpendicular() else "x"
        return tuple(d.get("fit_components") or (default,))

    @staticmethod
    def _bias_correlation_matrix(t, l_bias):
        """Exponential (Ornstein-Uhlenbeck) correlation matrix of the discrepancy.

        Parameters
        ----------
        t : array_like
            Time grid.
        l_bias : float
            Correlation length.

        Returns
        -------
        numpy.ndarray
            The ``(len(t), len(t))`` correlation matrix ``exp(-|t - t'| / l_bias)``.
        """
        return np.exp(-np.abs(np.subtract.outer(t, t)) / l_bias)

    def _covariance_matrix(self, t, sigma_noise, sigma_bias, l_bias):
        """Build ``Sigma = sigma_noise**2 I + sigma_bias**2 K``.

        Parameters
        ----------
        t : array_like
            Time grid.
        sigma_noise : float
            Measurement-noise standard deviation.
        sigma_bias : float
            Model-discrepancy standard deviation.
        l_bias : float
            Correlation length of the discrepancy.

        Returns
        -------
        numpy.ndarray
            The ``(len(t), len(t))`` covariance matrix.
        """
        K = self._bias_correlation_matrix(t, l_bias)
        return sigma_noise**2 * np.eye(len(t)) + sigma_bias**2 * K

    @staticmethod
    def _cholesky(covariance):
        """Cholesky-factorise ``Sigma``, returning ``None`` if it is not usable.

        Parameters
        ----------
        covariance : numpy.ndarray
            Covariance matrix.

        Returns
        -------
        tuple or None
            The ``(c, lower)`` factor from :func:`scipy.linalg.cho_factor`, or
            ``None`` if the matrix is non-finite or not positive definite.
        """
        if not np.all(np.isfinite(covariance)):
            return None
        try:
            return cho_factor(covariance, lower=True)
        except np.linalg.LinAlgError:
            return None

    @staticmethod
    def _log_likelihood(residual, cholesky=None, sigma_noise=None):
        """Gaussian log-density of a residual vector.

        Computes ``-r^T Sigma^-1 r / 2 - log|Sigma| / 2 - (N/2) log(2*pi)``,
        using the Cholesky factor when given, otherwise a scalar-diagonal
        ``Sigma = sigma_noise**2 I``.

        Parameters
        ----------
        residual : numpy.ndarray
            Residual vector ``r``.
        cholesky : tuple, optional
            Factor from :func:`scipy.linalg.cho_factor` for the full ``Sigma``.
        sigma_noise : float, optional
            Noise standard deviation, used only when ``cholesky`` is ``None``.

        Returns
        -------
        float
            The log-likelihood.
        """
        n = len(residual)
        if cholesky is not None:
            r_sq = residual @ cho_solve(cholesky, residual)
            log_sigma = 2.0 * np.sum(np.log(np.diag(cholesky[0])))
        else:
            r_sq = float(residual @ residual) / sigma_noise**2
            log_sigma = 2.0 * n * np.log(sigma_noise)
        return -r_sq / 2.0 - log_sigma / 2.0 - (n / 2.0) * np.log(2.0 * np.pi)

    def _residuals(self, d, theta):
        """Residuals ``x_obs - model(10**mu_hat)`` for each fitted component.

        Parameters
        ----------
        d : dict
            Dataset providing the time grid and observed components.
        theta : sequence of float
            Physical model parameters.

        Returns
        -------
        list of numpy.ndarray
            One residual vector per fitted component.
        """
        t = d["t"]
        residuals = []
        for component in self._fit_components(d):
            observed = np.asarray(d.get(component), dtype=float)
            model = np.asarray(
                self._model_component(theta, t, d, component=component), dtype=float
            )
            residuals.append(observed - model)
        return residuals

    def log_likelihood(self, phi):
        """Log-likelihood of the loaded trajectory, summed over components.

        Parameters
        ----------
        phi : numpy.ndarray
            Parameters in sampler space.

        Returns
        -------
        float
            The total log-likelihood, or ``-inf`` if any component is non-finite
            (including a failed covariance factorisation).
        """
        d = self.data
        theta = self._to_physical(phi)
        sigma_noise, sigma_bias = self._extract_noise_bias(theta)
        residuals = self._residuals(d, theta)

        if sigma_bias is None:
            terms = (self._log_likelihood(r, sigma_noise=sigma_noise) for r in residuals)
        else:
            cholesky = self._cholesky(
                self._covariance_matrix(d["t"], sigma_noise, float(sigma_bias), self.l_bias)
            )
            if cholesky is None:
                return -np.inf
            terms = (self._log_likelihood(r, cholesky=cholesky) for r in residuals)

        total = 0.0
        for value in terms:
            if not np.isfinite(value):
                return -np.inf
            total += value
        return float(total)

    def log_posterior(self, phi):
        """Log-posterior ``log L + log prior``, up to an additive constant.

        Parameters
        ----------
        phi : numpy.ndarray
            Parameters in sampler space.

        Returns
        -------
        float
            The log-posterior, or ``-inf`` if the prior or likelihood is
            non-finite.
        """
        log_prior = self.log_prior(phi)
        if not np.isfinite(log_prior):
            return -np.inf

        log_likelihood = self.log_likelihood(phi)
        if not np.isfinite(log_likelihood):
            return -np.inf

        return float(log_prior + log_likelihood)