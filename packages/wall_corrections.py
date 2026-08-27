import numpy as np

class WallCorrections:
    """Wall-correction factors for a sphere near a wall."""

    def _wall_factors(self, delta):
        delta = max(float(delta), self.bounds["delta0"][0])
        return float(self.zeng_parallel(delta)), float(self.brenner_perpendicular(delta))

    def zeng_parallel(self, delta):
        d = np.asarray(delta, float)
        a = self.a
        return (1.028 - 0.07 * a**2 / (a**2 + d**2)
                - (8 / 15) * np.log(135 * d / (135 * a + 128 * d)))

    def _brenner_series(self, delta):
        d = np.asarray(delta, float)
        beta = np.arccosh((d + self.a) / self.a)[..., None]
        n = np.arange(1, self.n_brenner_terms + 1)

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            term = (
                (2 * np.sinh((2 * n + 1) * beta) + (2 * n + 1) * np.sinh(2 * beta))
                / (4 * np.sinh((n + 0.5) * beta)**2 - (2 * n + 1)**2 * np.sinh(beta)**2)
                - 1
            )
            weight = n * (n + 1) / ((2 * n - 1) * (2 * n + 3))
            total = np.sum(weight * np.where(np.isfinite(term), term, 0), axis=-1)

        return (4 / 3) * np.sinh(beta[..., 0]) * total

    def brenner_perpendicular(self, delta):
        key = (self.a, self.n_brenner_terms)
        if getattr(self, "_brenner_cache", None) is None or self._brenner_cache[0] != key:
            grid = np.geomspace(self.bounds["delta0"][0], self.bounds["delta0"][1], 3000)
            self._brenner_cache = (key, grid, self._brenner_series(grid))

        _, grid, values = self._brenner_cache
        return np.interp(np.asarray(delta, float), grid, values)