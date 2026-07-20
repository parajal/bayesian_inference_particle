import numpy as np
from scipy.integrate import solve_ivp

class ForwardModels:
    """Two forward models: Newtonian and viscoelastic."""

    @staticmethod
    def _solve_creep_recovery(
        ode_eqn, y0, t, t_unload, load_args, recovery_args
    ):
        t = np.asarray(t, dtype=float)
        y0 = np.asarray(y0, dtype=float)
        load = solve_ivp(
            ode_eqn,
            (t[0], t_unload),
            y0,
            t_eval=np.append(t[t < t_unload], t_unload),
            rtol=1e-7,
            atol=1e-9,
            args=load_args,
        )
        if not load.success:
            return None

        recovery = solve_ivp(
            ode_eqn,
            (t_unload, t[-1]),
            y0=load.y[:, -1],
            t_eval=t[t >= t_unload],
            rtol=1e-7,
            atol=1e-9,
            args=recovery_args,
        )
        if not recovery.success:
            return None

        result = np.empty((y0.size, t.size))
        result[:, t < t_unload] = load.y[:, :-1]
        result[:, t >= t_unload] = recovery.y
        return result

    def model_newtonian(self, eta, t, F, delta0=None, L=None, component=None, angle=None):
        """Newtonian displacement using boundary_model and theta."""
        angle = float(self.theta if angle is None else angle) % 180.0
        component = ( "y" if abs(angle - 90.0) < 1e-6 else "x" )
        x = np.zeros_like(t) ; y = np.zeros_like(t)
        drag = 6.0 * np.pi * self.a * eta
        theta = np.deg2rad(angle)
        Fx = F * np.cos(theta)
        Fy = F * np.sin(theta)

        if self.boundary_model == "unbounded":
            F_dir = Fy if component == "y" else Fx
            return F_dir * (t - t[0]) / drag

        delta0 = self.delta0 if delta0 is None else delta0
        # Forward-Euler step through the wall-corrected drag (delta depends on y).
        for i in range(t.size - 1):
            f_parallel, f_perp = self._wall_factors(delta0 + y[i])
            dxdt = Fx / (drag * f_parallel)
            dydt = Fy / (drag * f_perp)
            x[i + 1] = x[i] + (t[i + 1] - t[i]) * dxdt
            y[i + 1] = y[i] + (t[i + 1] - t[i]) * dydt

        return y if component == "y" else x

    def model_viscoelastic(self, eta_s, eta_p, lambda_, t, F, delta0=None, t_unload=None, component=None, angle=None):
        """Viscoelastic displacement selected by boundary_model and theta."""
        t = np.asarray(t, dtype=float)
        angle = float(self.theta if angle is None else angle) % 180.0
        component = (
            "y" if abs(angle - 90.0) < 1e-6 else "x"
        ) if component is None else str(component).lower()
        if component not in ("x", "y"):
            raise ValueError("component must be 'x' or 'y'.")

        t_unload = self._t_unload_eff if t_unload is None else t_unload
        if t_unload is None or not np.isfinite(float(t_unload)):
            raise ValueError("t_unload must be finite for viscoelastic models.")
        t_unload = float(t_unload)
        solvent_drag = 6.0 * np.pi * self.a * eta_s
        polymer_drag = 6.0 * np.pi * self.a * eta_p
        theta = np.deg2rad(angle)
        Fx = F * np.cos(theta)
        Fy = F * np.sin(theta)

        if self.boundary_model == "unbounded":
            eta_0 = eta_s + eta_p
            eff_tau = lambda_ * eta_s / eta_0
            F_dir = Fy if component == "y" else Fx
            v_steady = F_dir / (6.0 * np.pi * self.a * eta_0)
            prefactor = F_dir * lambda_ * eta_p / (6.0 * np.pi * self.a * eta_0**2)

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

        if abs(angle - 90.0) < 1e-6:
            if component == "x":
                return np.zeros_like(t)

            def ode_eqn(time, state, force_y):
                y, Fpy = state
                f_perp = float(self.brenner_perpendicular(delta0 + y))
                dydt = (force_y / f_perp - Fpy) / solvent_drag
                dFpydt = -Fpy / lambda_ + polymer_drag * dydt / lambda_
                return [dydt, dFpydt]

            sol = self._solve_creep_recovery(ode_eqn,
                [0.0, 0.0], t, t_unload, (Fy,), (0.0,),)
            return None if sol is None else sol[0]

        def ode_eqn(time, state, force_x, force_y):
            x, y, Fpx, Fpy = state
            f_parallel, f_perp = self._wall_factors(delta0 + y)
            dxdt = (force_x / f_parallel - Fpx) / solvent_drag
            dydt = (force_y / f_perp - Fpy) / solvent_drag
            dFpxdt = -Fpx / lambda_ + polymer_drag * dxdt / lambda_
            dFpydt = -Fpy / lambda_ + polymer_drag * dydt / lambda_
            return [dxdt, dydt, dFpxdt, dFpydt]

        sol = self._solve_creep_recovery(
            ode_eqn, [0.0, 0.0, 0.0, 0.0],
            t, t_unload, (Fx, Fy), (0.0, 0.0),)
        if sol is None:
            return None
        return sol[1] if component == "y" else sol[0]
