import numpy as np
from scipy.linalg import cho_factor, cho_solve

class Likelihood:
    """Gaussian likelihood and posterior."""

    @staticmethod
    def _bias_correlation_matrix(t, l_bias):
        return np.exp(-np.abs(np.subtract.outer(t, t)) / l_bias)

    @staticmethod
    def _loglikelihood(residual, sigma=None, chol=None):
        n = len(residual)
        if chol is None:
            r2 = residual @ residual / sigma**2
            logdet = 2 * n * np.log(sigma)
        else:
            r2 = residual @ cho_solve(chol, residual)
            logdet = 2 * np.sum(np.log(np.diag(chol[0])))
        return -0.5 * (r2 + logdet + n * np.log(2 * np.pi))

    def _residuals(self, theta):
        d = self.data
        comps = d.get("fit_components", ("y",) if self.is_perpendicular() else ("x",))
        return [
            np.asarray(d[c]) - np.asarray(self._model_component(theta, d["t"], d, component=c))
            for c in comps
        ]

    def log_likelihood(self, phi):
        d = self.data
        theta = self._to_physical(phi)
    
        sigma_noise, sigma_bias = self._extract_noise_bias(theta)
    
        # Validate noise/bias parameters
        if sigma_noise is None or sigma_noise <= 0:
            return -np.inf
    
        if sigma_bias is not None and sigma_bias < 0:
            return -np.inf
    
        if sigma_bias not in (None, 0.0) and self.l_bias <= 0:
            return -np.inf
    
        residuals = self._residuals(theta)
    
        chol = None
        if sigma_bias not in (None, 0.0):
            K = self._bias_correlation_matrix(d["t"], self.l_bias)
            cov = (
                sigma_noise**2 * np.eye(len(d["t"]))
                + sigma_bias**2 * K
            )
    
            try:
                chol = cho_factor(cov, lower=True)
            except (np.linalg.LinAlgError, ValueError):
                return -np.inf
    
        logL = 0.0
        for r in residuals:
            value = self._loglikelihood(
                r,
                sigma=sigma_noise,
                chol=chol
            )
    
            if not np.isfinite(value):
                return -np.inf
    
            logL += value
    
        return float(logL)

    def log_posterior(self, phi):
        lp = self.log_prior(phi)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(phi)
        if not np.isfinite(ll):
            return -np.inf
        return float(lp + ll)