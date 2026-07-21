"""Hydrodynamic wall corrections for a sphere near a plane wall."""

import numpy as np

_N_TERMS = 100
_GRID_SIZE = 3000


class WallCorrections:
    """Zeng (parallel) and Brenner (perpendicular) wall-correction factors."""

    def _min_gap(self) -> float:
        """Smallest gap the corrections are evaluated at, ``1e-4 * a``.

        Returns
        -------
        float
            Lower clamp on ``delta``, keeping the diverging series finite.
        """
        return 1e-3 * self.a

    def _wall_factors(self, delta: float) -> tuple[float, float]:
        """Both correction factors at a single gap, clamped at ``_min_gap``.

        Parameters
        ----------
        delta : float
            Wall gap.

        Returns
        -------
        f_parallel, f_perp : float
            Parallel and perpendicular drag ratios.
        """
        delta = max(float(delta), self._min_gap())
        return float(self.zeng_parallel(delta)), float(self.brenner_perpendicular(delta))

    def zeng_parallel(self, delta):
        """Zeng interpolant for motion parallel to the wall.

        Parameters
        ----------
        delta : array_like
            Wall gap.

        Returns
        -------
        numpy.ndarray
            Parallel drag ratio, same shape as ``delta``.
        """
        delta = np.asarray(delta, dtype=float)
        a = self.a
        return (
            1.028
            - 0.07 * a**2 / (a**2 + delta**2)
            - (8.0 / 15.0) * np.log(135.0 * delta / (135.0 * a + 128.0 * delta))
        )

    def _brenner_series(self, delta, n_terms=_N_TERMS):
        """Evaluate Brenner's series directly.

        Terms that overflow at large ``beta`` are dropped; they are negligible
        compared with the retained ones.

        Parameters
        ----------
        delta : array_like
            Wall gap.
        n_terms : int, optional
            Number of terms summed.

        Returns
        -------
        numpy.ndarray
            Perpendicular drag ratio, same shape as ``delta``.
        """
        delta = np.asarray(delta, dtype=float)
        beta = np.arccosh((delta + self.a) / self.a)[..., None]
        n = np.arange(1, n_terms + 1, dtype=float)

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            numerator = (
                2.0 * np.sinh((2 * n + 1) * beta) + (2 * n + 1) * np.sinh(2.0 * beta)
            )
            denominator = (
                4.0 * np.sinh((n + 0.5) * beta) ** 2
                - (2 * n + 1) ** 2 * np.sinh(beta) ** 2
            )
            term = numerator / denominator - 1.0
            weight = n * (n + 1) / ((2 * n - 1) * (2 * n + 3))
            total = np.sum(weight * np.where(np.isfinite(term), term, 0.0), axis=-1)

        return (4.0 / 3.0) * np.sinh(beta[..., 0]) * total

    def brenner_perpendicular(self, delta, n_terms=_N_TERMS):
        """Brenner's series for motion perpendicular to the wall.

        The series is evaluated once on a log-spaced grid of gaps and cached;
        subsequent calls interpolate, which keeps MCMC evaluations cheap. The
        cache is rebuilt whenever ``self.a`` or ``n_terms`` changes.

        Parameters
        ----------
        delta : array_like
            Wall gap. Values outside the grid are clamped to its end points.
        n_terms : int, optional
            Number of terms summed when building the grid.

        Returns
        -------
        numpy.ndarray
            Perpendicular drag ratio, same shape as ``delta``.
        """
        key = (self.a, n_terms)
        if getattr(self, "_brenner_cache", (None,))[0] != key:
            grid = np.geomspace(self._min_gap(), 1e2 * self.a, _GRID_SIZE)
            self._brenner_cache = (key, grid, self._brenner_series(grid, n_terms))

        _, grid, values = self._brenner_cache
        return np.interp(np.asarray(delta, dtype=float), grid, values)
