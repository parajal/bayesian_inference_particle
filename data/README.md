# Data folder — column descriptions

This folder holds particle trajectory data. 

Notation:

- `t` — time
- `X(t)`, `Y(t)`, `Z(t)` — particle position components
- `U_x(t)`, `U_y(t)`, `U_z(t)` — particle (translational) velocity components
- `Omega_x(t)`, `Omega_y(t)`, `Omega_z(t)` — particle angular velocity components

## Column layout by folder

| Folder | Files | Cols | Columns (in order) |
|---|---|:--:|---|
| `linear_viscoelastic/angle0`, `angle45`, `angle90` | `*.out` | 7 | `t`, `X(t)`, `Y(t)`, `Z(t)`, `U_x(t)`, `U_y(t)`, `U_z(t)` |
| `linear_viscoelastic/particle-particle` | `lx-*.out` | 10 | `t`, `X(t)`, `Y(t)`, `Z(t)`, `U_x(t)`, `U_y(t)`, `U_z(t)`, `Omega_x(t)`, `Omega_y(t)`, `Omega_z(t)` |
| `linear_viscoelastic/unbounded` | `*.out` | 3 | `t`, `X(t)`, `U_x(t)` |
| `newtonian/angle-0`, `angle-45`, `angle-90` | `*.txt` | 7 | `t`, `X(t)`, `Y(t)`, `Z(t)`, `U_x(t)`, `U_y(t)`, `U_z(t)` |
| `newtonian/unbounded` | `*.txt` | 2 | `t`, `X(t)` |
| `nonlinear_viscoelastic` | `*.out` | 3 | `t`, `X(t)`, `U_x(t)` |