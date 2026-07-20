"""Likelihood and posterior."""

import numpy as np


class Likelihood:
    """Implements the manuscript likelihood in direct matrix form."""

    def _bias_correlation_matrix(self, t_left, l_bias, t_right=None):
        """K_ij = exp(-|t_i - t_j| / l_bias)."""
        l_bias = float(l_bias)
        if l_bias <= 0.0 or not np.isfinite(l_bias):
            raise ValueError("l_bias must be positive and finite.")

        t_left = np.asarray(t_left, dtype=float)
        t_right = t_left if t_right is None else np.asarray(t_right, dtype=float)
        return np.exp(-np.abs(np.subtract.outer(t_left, t_right)) / l_bias)

    def _covariance_matrix(self, t, sigma_noise, sigma_bias):
        """Sigma = sigma_noise^2 I + sigma_bias^2 K."""
        K = self._bias_correlation_matrix(t, self.l_bias)
        return sigma_noise**2 * np.eye(len(t)) + sigma_bias**2 * K

    # def _use_y_displacement(self, d):
    #     """Return True when a dataset is fitted only through y."""
    #     return self._fit_components(d) == ("y",)

    @staticmethod
    def _gaussian_log_likelihood(residual, covariance):
        """-1/2 r^T Sigma^-1 r - 1/2 log|Sigma| - N/2 log(2*pi)."""
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0.0 or not np.isfinite(logdet):
            return -np.inf

        try:
            solve = np.linalg.solve(covariance, residual)
        except np.linalg.LinAlgError:
            return -np.inf

        quad = residual @ solve
        if not np.isfinite(quad):
            return -np.inf

        return -0.5 * quad - 0.5 * logdet - 0.5 * len(residual) * np.log(2.0 * np.pi)

    def _dataset_residuals(self, d, theta):
        """Return fitted residuals rx and, when requested, ry for one dataset."""
        t = d["t"]
        residuals = []

        for component in self._fit_components(d):
            observed = d.get(component)
            model = self._model_component_at(theta, t, d, component=component)
            if observed is None or model is None or not np.all(np.isfinite(model)):
                return None

            residual = np.asarray(observed, dtype=float) - np.asarray(model, dtype=float)
            if not np.all(np.isfinite(residual)):
                return None
            residuals.append((component, residual))

        return residuals

    def _dataset_residual(self, d, theta):
        """Return all fitted residual components concatenated for diagnostics."""
        residuals = self._dataset_residuals(d, theta)
        if residuals is None:
            return None
        return np.concatenate([residual for _, residual in residuals])

    @staticmethod
    def _valid_sigma(value):
        return value is None or (0.0 <= float(value) <= 1e150)

    def log_likelihood(self, phi):
        theta = self._to_physical(phi)
        sigma_noise, sigma_bias = self._extract_noise_bias(theta)

        if not self._valid_sigma(sigma_noise) or not self._valid_sigma(sigma_bias):
            return -np.inf

        sigma_noise = 0.0 if sigma_noise is None else float(sigma_noise)
        sigma_bias = 0.0 if sigma_bias is None else float(sigma_bias)

        total = 0.0
        for d in self.datasets:
            residuals = self._dataset_residuals(d, theta)
            if residuals is None:
                return -np.inf

            covariance = self._covariance_matrix(d["t"], sigma_noise, sigma_bias)
            for _, residual in residuals:
                value = self._gaussian_log_likelihood(residual, covariance)
                if not np.isfinite(value):
                    return -np.inf
                total += value

        return float(total)

    def log_posterior(self, phi):
        log_prior = self.log_prior(phi)
        if not np.isfinite(log_prior):
            return -np.inf

        log_likelihood = self.log_likelihood(phi)
        if not np.isfinite(log_likelihood):
            return -np.inf

        return float(log_prior + log_likelihood)
