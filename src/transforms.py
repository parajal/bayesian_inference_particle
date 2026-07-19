"""Transforms between sampler coordinates and physical parameters."""

import numpy as np


class BoundedTransform:
    """Logit transform from a finite interval to the real line."""

    def __init__(self, lower: float, upper: float):
        if not lower < upper:
            raise ValueError("lower must be less than upper.")
        self.lower = float(lower)
        self.upper = float(upper)

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return np.log((x - self.lower) / (self.upper - x))

    def inverse(self, y):
        y = np.asarray(y, dtype=float)
        sigmoid = 1.0 / (1.0 + np.exp(-y))
        return self.lower + (self.upper - self.lower) * sigmoid

    def jacobian(self, x):
        x = np.asarray(x, dtype=float)
        return (self.upper - self.lower) / ((x - self.lower) * (self.upper - x))


class LogTransform:
    """Log transform from positive values to the real line."""

    def __call__(self, x):
        return np.log(x)

    def inverse(self, y):
        return np.exp(y)

    def jacobian(self, x):
        return 1.0 / np.asarray(x, dtype=float)


class Transforms:
    """Material parameters use log10; noise/bias use physical units."""

    def _get_log10_mask(self) -> np.ndarray:
        """True for coordinates sampled in log10 space."""
        mask = np.zeros(self._get_ndim(), dtype=bool)
        mask[: len(self._get_parameter_bounds())] = True
        return mask

    def _to_physical(self, phi: np.ndarray) -> np.ndarray:
        """Sampler -> physical: theta = 10**phi for log10 coordinates."""
        theta = np.array(phi, dtype=float, copy=True)
        mask = self._get_log10_mask()
        theta[..., mask] = 10.0 ** (theta[..., mask])
        return theta

    def _to_logtransform(self, theta: np.ndarray) -> np.ndarray:
        """Physical -> sampler: phi = log10(theta) for log10 coordinates."""
        phi = np.array(theta, dtype=float, copy=True)
        mask = self._get_log10_mask()
        phi[..., mask] = np.log10(np.maximum(phi[..., mask], np.finfo(float).tiny))
        return phi
