# Data folder — column descriptions

This folder holds particle trajectory data. All files are plain whitespace-delimited
text with no header row (one exception, noted below). The number of columns depends on
the sub-folder. Units follow the simulation's own non-dimensionalization.

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

## Notes

- **`linear_viscoelastic/particle-particle/analytical_model.txt`** is the exception:
  it is generated separately and *does* have a header. Lines starting with `#` are
  metadata (mode, dataset index, source file, true parameters, and a `# columns:` line);
  the data rows have 3 columns: `t`, `X(t)`, `U_x(t)`.
- File names generally encode the run parameters, e.g. `delta-<value>-<angle>` (the
  gap/offset `delta` and the inclination angle), `lx-<value>` (box length for the
  particle–particle cases), and `<N>pi` (forcing frequency for the nonlinear cases).
