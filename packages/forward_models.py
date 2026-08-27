import numpy as np
from scipy.integrate import solve_ivp

class ForwardModels:

    def _drag(self, eta):
        return 6 * np.pi * self.a * eta

    def _component(self, component=None):
        return component or ("y" if self.is_perpendicular() else "x")

    def _force(self, F):
        th = np.deg2rad(self.theta)
        return F * np.cos(th), F * np.sin(th)

    def _solve_ode(self, ode, y0, t, t_unload, load_args, unload_args):
        load = solve_ivp(ode, (t[0], t_unload), y0,
            t_eval=np.append(t[t < t_unload], t_unload),
            args=load_args, rtol=self.rtol, atol=self.atol)
        if not load.success:
            return None

        recovery = solve_ivp(ode, (t_unload, t[-1]), load.y[:, -1],
            t_eval=t[t >= t_unload], args=unload_args,
            rtol=self.rtol, atol=self.atol)
        if not recovery.success:
            return None

        sol = np.empty((len(y0), len(t)))
        sol[:, t < t_unload] = load.y[:, :-1]
        sol[:, t >= t_unload] = recovery.y
        return sol

    def model_newtonian(self, eta, t, F, delta0=None):
        comp = self._component()
        drag = self._drag(eta)
        Fx, Fy = self._force(F)

        if self.boundary_model == "unbounded":
            return (Fy if comp == "y" else Fx) * (t - t[0]) / drag

        delta0 = self.delta0 if delta0 is None else delta0
        x, y = np.zeros_like(t), np.zeros_like(t)

        for i in range(len(t) - 1):
            fp, fn = self._wall_factors(delta0 + y[i])
            dt = t[i + 1] - t[i]
            x[i + 1] = x[i] + dt * Fx / (drag * fp)
            y[i + 1] = y[i] + dt * Fy / (drag * fn)

        return y if comp == "y" else x

    def model_viscoelastic(self, eta_s, eta_p, lam, t, F,
            t_unload=None, component=None):
        comp = self._component(component)
        Fx, Fy = self._force(F)
        t_unload = self._t_unload_eff if t_unload is None else t_unload

        if self.boundary_model == "unbounded":
            eta0 = eta_s + eta_p
            tau = lam * eta_s / eta0
            drag = self._drag(eta0)
            Fdir = Fy if comp == "y" else Fx
            v = Fdir / drag
            A = Fdir * lam * eta_p / (drag * eta0)

            t_load = t - t[0]
            t_loading = max(0, t_unload - t[0])
            t_recovery = np.maximum(t - t_unload, 0)

            creep = v * t_load + A * (1 - np.exp(-t_load / tau))
            x0 = v * t_loading + A * (1 - np.exp(-t_loading / tau))
            recovery = x0 - A * (1 - np.exp(-t_loading / tau)) * (1 - np.exp(-t_recovery / tau))
            return np.where(t <= t_unload, creep, recovery)

        if self.is_perpendicular():

            def ode_perp(_, s, Fy):
                y, Fpy = s
                f = self.brenner_perpendicular(self.delta0 + y)
                dydt = (Fy / f - Fpy) / self._drag(eta_s)
                return [dydt, (-Fpy + self._drag(eta_p) * dydt) / lam]

            sol = self._solve_ode(ode_perp, [0, 0], t, t_unload, (Fy,), (0,))
            return None if sol is None else (np.zeros_like(t) if comp == "x" else sol[0])

        def ode_angled(_, s, Fx, Fy):
            _, y, Fpx, Fpy = s
            fp, fn = self._wall_factors(self.delta0 + y)
            dxdt = (Fx - Fpx) / (fp * self._drag(eta_s))
            dydt = (Fy - Fpy) / (fn * self._drag(eta_s))
            return [dxdt, dydt,
                (-Fpx + fp * self._drag(eta_p) * dxdt) / lam,
                (-Fpy + fn * self._drag(eta_p) * dydt) / lam]

        sol = self._solve_ode(ode_angled, [0, 0, 0, 0], t, t_unload, (Fx, Fy), (0, 0))
        return None if sol is None else (sol[1] if comp == "y" else sol[0])

    def _model_component(self, theta, t, d, component=None):
        if self.material_model == "newtonian":
            return self.model_newtonian(theta[0], t, d["F"], self._get_delta0(theta))

        return self.model_viscoelastic(theta[0], theta[1], theta[2], t, d["F"],
            t_unload=d.get("t_unload"), component=component)