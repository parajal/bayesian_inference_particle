"""Transforms between sampler coordinates and physical parameters."""

import numpy as np


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
