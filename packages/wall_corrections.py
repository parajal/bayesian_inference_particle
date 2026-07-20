"""Wall correction factors used in the manuscript.

Parallel motion uses the Zeng interpolant. Perpendicular motion uses
Brenner's exact series with 100 terms. Both functions use the physical
wall gap delta and the particle radius self.a.
"""

import numpy as np


class WallCorrections:
    """Wall-correction formulas."""

    _MIN_GAP_FACTOR = 1e-4

    def _radius(self):
        """Particle radius; defaults to 1.0 when used standalone (no InferenceProcedure)."""
        return float(getattr(self, "a", 1.0))

    def _min_gap(self):
        return self._MIN_GAP_FACTOR * self._radius()

    def _wall_factors(self, delta):
        delta = max(float(delta), self._min_gap())
        return float(self.zeng_parallel(delta)), float(self.brenner_perpendicular(delta))

    def zeng_parallel(self, delta):
        """Zeng interpolant for parallel motion near a wall."""
        delta = np.asarray(delta, dtype=float)
        a = self._radius()

        return (
            1.028
            - 0.07 * a**2 / (a**2 + delta**2)
            - (8.0 / 15.0) * np.log(135.0 * delta / (135.0 * a + 128.0 * delta))
        )

    def _brenner_perpendicular_direct(self, delta, n_terms=100):
        """Brenner series."""
        delta = np.asarray(delta, dtype=float)
        a = self._radius()
        beta = np.arccosh((delta + a) / a)
        n = np.arange(1, n_terms + 1, dtype=float)
        beta_n = np.expand_dims(beta, axis=-1)

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            term = (
                (
                    2.0 * np.sinh((2 * n + 1) * beta_n)
                    + (2 * n + 1) * np.sinh(2.0 * beta_n)
                )
                / (
                    4.0 * np.sinh((n + 0.5) * beta_n) ** 2
                    - (2 * n + 1) ** 2 * np.sinh(beta_n) ** 2
                )
                - 1.0
            )
            term = np.where(np.isfinite(term), term, 0.0)
            total = np.sum(
                n * (n + 1) / ((2 * n - 1) * (2 * n + 3)) * term,
                axis=-1,
            )

        return (4.0 / 3.0) * np.sinh(beta) * total

    def brenner_perpendicular(self, delta, n_terms=100):
        """Brenner series, cached for fast MCMC calls."""
        cache = getattr(self, "_brenner_cache", None)
        if cache is None or cache["a"] != self._radius() or cache["n_terms"] != n_terms:
            grid = np.geomspace(1e-3 * self._radius(), 1e2 * self._radius(), 3000)
            cache = {
                "a": self._radius(),
                "n_terms": n_terms,
                "delta": grid,
                "value": self._brenner_perpendicular_direct(grid, n_terms=n_terms),
            }
            self._brenner_cache = cache

        return np.interp(np.asarray(delta, dtype=float), cache["delta"], cache["value"])
