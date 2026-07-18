from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


class ForwardModels:
    """Two forward models: Newtonian and viscoelastic."""

    @staticmethod
    def _hasimoto_Q(L):
        L = float(L)
        return 1.0 - 2.8373 / L + (4.0 * np.pi / 3.0) / L**3

    def _is_perpendicular(self):
        return abs(float(self.theta) % 180.0 - 90.0) < 1e-6

    @staticmethod
    def _force(time, force, t_unload):
        return float(force) if time <= t_unload else 0.0

    def _solve_ode(self, ode_eqn, y0, t):
        sol = solve_ivp(
            ode_eqn,
            (float(t[0]), float(t[-1])),
            np.asarray(y0, dtype=float),
            t_eval=t,
            rtol=1e-7,
            atol=1e-9,
            max_step=float(np.min(np.diff(t))),
        )
        return sol.y if sol.success and sol.y.shape[1] == t.size else None

    def _cache_get(self, key):
        sol = getattr(self, "_forward_solution_cache", {}).get(key)
        return None if sol is None else sol.copy()

    def _cache_set(self, key, sol):
        cache = self.__dict__.setdefault("_forward_solution_cache", {})
        if len(cache) > 256:
            cache.clear()
        cache[key] = sol.copy()

    def _time_key(self, t):
        return np.asarray(t, dtype=float).tobytes()

    def model_newtonian(self, eta, t, F, delta0=None, L=None):
        """Newtonian displacement using boundary_model and theta."""
        t = np.asarray(t, dtype=float)
        x = np.zeros_like(t)
        y = np.zeros_like(t)
        drag = 6.0 * np.pi * self.a * eta
        theta = np.deg2rad(float(self.theta))
        Fx = F * np.cos(theta)
        Fy = F * np.sin(theta)

        if self.boundary_model == "unbounded":
            # Isotropic Stokes drag; use the force component along the observed
            # direction (Fy for perpendicular / theta=90, Fx otherwise).
            F_dir = Fy if self._is_perpendicular() else Fx
            return F_dir * (t - t[0]) / drag

        if self.boundary_model == "particle_particle":
            L = self.L if L is None else L
            if L is None or not np.isfinite(L) or L <= 0.0:
                raise ValueError("particle_particle requires a positive L.")
            hasimoto = self._hasimoto_Q(L)
            return Fx * ((t - t[0]) / drag) * hasimoto

        delta0 = self.delta0 if delta0 is None else delta0
        if delta0 is None or not np.isfinite(delta0) or delta0 <= 0.0:
            raise ValueError("bounded Newtonian model requires a positive delta0.")

        # Forward-Euler step through the wall-corrected drag (delta depends on y).
        for i in range(t.size - 1):
            f_parallel, f_perp = self._wall_factors(delta0 + y[i])
            dxdt = Fx / (drag * f_parallel)
            dydt = Fy / (drag * f_perp)
            x[i + 1] = x[i] + (t[i + 1] - t[i]) * dxdt
            y[i + 1] = y[i] + (t[i + 1] - t[i]) * dydt

        return y if self._is_perpendicular() else x

    def model_viscoelastic(self, eta_s, eta_p, lambda_, t, F, delta0=None, t_unload=None):
        """Viscoelastic displacement selected by boundary_model and theta."""
        t = np.asarray(t, dtype=float)

        t_unload = self._t_unload_eff if t_unload is None else t_unload
        solvent_drag = 6.0 * np.pi * self.a * eta_s
        polymer_drag = 6.0 * np.pi * self.a * eta_p
        theta = np.deg2rad(float(self.theta))
        Fx = F * np.cos(theta)
        Fy = F * np.sin(theta)

        # Reuse ODE solutions when MCMC uses the same parameters.
        def solve(key, y0, ode_eqn):
            sol = self._cache_get(key)
            if sol is None:
                sol = self._solve_ode(ode_eqn, y0, t)
                if sol is not None:
                    self._cache_set(key, sol)
            return sol

        if self.boundary_model == "unbounded":
            eta_0 = eta_s + eta_p
            eff_tau = lambda_ * eta_s / eta_0
            v_steady = Fx / (6.0 * np.pi * self.a * eta_0)
            prefactor = Fx * lambda_ * eta_p / (6.0 * np.pi * self.a * eta_0**2)

            load_time = t - t[0]                    
            end_load = max(0.0, t_unload - t[0])     
            recovery_time = np.maximum(t - t_unload, 0.0)

            x_load = v_steady * load_time + prefactor * (1.0 - np.exp(-load_time / eff_tau))
            x_at_unload = v_steady * end_load + prefactor * (1.0 - np.exp(-end_load / eff_tau))
            recoil = prefactor * (1.0 - np.exp(-end_load / eff_tau))
            x_recovery = x_at_unload - recoil * (1.0 - np.exp(-recovery_time / eff_tau))

            return np.where(t <= t_unload, x_load, x_recovery)

        delta0 = self.delta0 if delta0 is None else delta0
        if delta0 is None or not np.isfinite(delta0) or delta0 <= 0.0:
            raise ValueError("bounded viscoelastic model requires a positive delta0.")

        if self._is_perpendicular():
            key = ("viscoelastic_bounded_perp", self.a, eta_s, eta_p, lambda_, delta0, Fy, t_unload, self._time_key(t))

            def ode_eqn(time, state):
                y, Fpy = state
                f_perp = float(self.brenner_perpendicular(delta0 + y))
                dydt = (self._force(time, Fy, t_unload) / f_perp - Fpy) / solvent_drag
                dFpydt = -Fpy / lambda_ + polymer_drag * dydt / lambda_
                return [dydt, dFpydt]

            return solve(key, [0.0, 0.0], ode_eqn)[0]

        key = ("viscoelastic_bounded", self.a, eta_s, eta_p, lambda_, delta0, Fx, Fy, t_unload, self._time_key(t))

        def ode_eqn(time, state):
            x, y, Fpx, Fpy = state
            f_parallel, f_perp = self._wall_factors(delta0 + y)
            dxdt = (self._force(time, Fx, t_unload) / f_parallel - Fpx) / solvent_drag
            dydt = (self._force(time, Fy, t_unload) / f_perp - Fpy) / solvent_drag
            dFpxdt = -Fpx / lambda_ + polymer_drag * dxdt / lambda_
            dFpydt = -Fpy / lambda_ + polymer_drag * dydt / lambda_
            return [dxdt, dydt, dFpxdt, dFpydt]

        return solve(key, [0.0, 0.0, 0.0, 0.0], ode_eqn)[0]
