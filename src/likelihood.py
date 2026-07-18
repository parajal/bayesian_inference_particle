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

    def _use_y_displacement(self, d):
        """Use y for wall-normal/perpendicular trajectories (theta ~ 90)."""
        return self._is_perpendicular()

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

    def _dataset_residual(self, d, theta):
        """r = x_obs - d(10^mu_hat), using x except for perpendicular cases."""
        t = d["t"]
        use_y = self._use_y_displacement(d)

        if self.material_model == "newtonian":
            model = self.model_newtonian(
                theta[0],
                t,
                d["F"],
                self._get_delta0(theta),
                L=d.get("L"),
            )
        elif self.material_model == "viscoelastic":
            model = self.model_viscoelastic(
                theta[0],
                theta[1],
                theta[2],
                t,
                d["F"],
                self._get_delta0(theta),
                t_unload=d.get("t_unload"),
            )
        else:
            raise ValueError(f"Invalid model in log_likelihood: {self.model!r}")

        observed = d["y"] if use_y else d["x"]
        if observed is None or not np.all(np.isfinite(model)):
            return None

        residual = np.asarray(observed, dtype=float) - np.asarray(model, dtype=float)
        return residual if np.all(np.isfinite(residual)) else None

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
            residual = self._dataset_residual(d, theta)
            if residual is None:
                return -np.inf

            covariance = self._covariance_matrix(d["t"], sigma_noise, sigma_bias)
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
