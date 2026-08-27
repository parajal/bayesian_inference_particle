import numpy as np

class Transforms:
    """Transforms between sampler and physical parameters."""

    def _to_physical(self, phi):
        theta = np.array(phi, dtype=float, copy=True)
        n = len(self._get_parameter_bounds())
        theta[..., :n] = 10**theta[..., :n]
        return theta

    def _to_logtransform(self, theta):
        phi = np.array(theta, dtype=float, copy=True)
        n = len(self._get_parameter_bounds())
        phi[..., :n] = np.log10(np.maximum(phi[..., :n], np.finfo(float).tiny))
        return phi