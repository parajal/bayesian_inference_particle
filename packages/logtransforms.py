"""Transforms between sampler coordinates and physical parameters."""

import numpy as np


class Transforms:
    """Coordinate maps: material parameters use log10, noise/bias stay physical."""

    def _get_log10_mask(self) -> np.ndarray:
        """Boolean mask marking coordinates that are sampled in log10 space.

        The leading coordinates (the bounded material parameters) are sampled in
        log10 space; the trailing nuisance parameters (noise, optional bias) are
        sampled directly in physical units.

        Returns
        -------
        numpy.ndarray
            Boolean array of length ``self._get_ndim()``, ``True`` where the
            coordinate is log10-transformed.
        """
        mask = np.zeros(self._get_ndim(), dtype=bool)
        mask[: len(self._get_parameter_bounds())] = True
        return mask

    def _to_physical(self, phi: np.ndarray) -> np.ndarray:
        """Map sampler coordinates to physical parameters.

        Applies ``theta = 10**phi`` on the log10 coordinates and leaves the rest
        unchanged.

        Parameters
        ----------
        phi : numpy.ndarray
            Sampler-space coordinates. The last axis indexes parameters, so
            batched inputs of shape ``(..., ndim)`` are supported.

        Returns
        -------
        numpy.ndarray
            Physical parameters, same shape as ``phi``.
        """
        theta = np.array(phi, dtype=float, copy=True)
        mask = self._get_log10_mask()
        theta[..., mask] = 10.0 ** theta[..., mask]
        return theta

    def _to_logtransform(self, theta: np.ndarray) -> np.ndarray:
        """Map physical parameters to sampler coordinates.

        Applies ``phi = log10(theta)`` on the log10 coordinates (clipped at the
        smallest positive float to avoid ``-inf``) and leaves the rest unchanged.

        Parameters
        ----------
        theta : numpy.ndarray
            Physical parameters. The last axis indexes parameters, so batched
            inputs of shape ``(..., ndim)`` are supported.

        Returns
        -------
        numpy.ndarray
            Sampler-space coordinates, same shape as ``theta``.
        """
        phi = np.array(theta, dtype=float, copy=True)
        mask = self._get_log10_mask()
        phi[..., mask] = np.log10(np.maximum(phi[..., mask], np.finfo(float).tiny))
        return phi