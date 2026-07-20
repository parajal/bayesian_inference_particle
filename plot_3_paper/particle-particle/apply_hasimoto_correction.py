"""
Apply the Hasimoto simple-cubic-array correction to the triperiodic
particle-particle runs and write corrected copies.

For each box file  lx_<name>.out  a corrected file  lx_<name>_corr.out  is
written. The Hasimoto drag factor for a simple-cubic array (sphere radius a),

    Q(L) = 1 - 2.8373*(a/L) + (4*pi/3)*(a/L)**3

is constant for a given box, so it simply rescales the whole time series.
Dividing the finite-box translational motion by Q(L) recovers the unbounded
behaviour. The translational displacement (x,y,z = cols 1-3) and translational
velocity (Ux,Uy,Uz = cols 4-6) are divided by Q(L); the time (col 0) and the
angular velocity (Wx,Wy,Wz = cols 7-9) are left unchanged.

Data file columns:
    0:t  1:x  2:y  3:z  4:Ux  5:Uy  6:Uz  7:Wx  8:Wy  9:Wz
"""

import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent

A_RADIUS = 1.0  # sphere radius

# Columns to correct (translational position + velocity).
TRANS_COLS = [1, 2, 3, 4, 5, 6]

LX_NAME_TO_VALUE = {
    "five": 5,
    "ten": 10,
    "twenty": 20,
    "fourty": 40, 
    "sixty": 60,
}


def hasimoto_Q(L, a=A_RADIUS):
    """Hasimoto drag factor for a simple-cubic array (leading terms)."""
    return 1.0 - 2.8373 * (a / L) + (4.0 * math.pi / 3.0) * (a / L) ** 3


def discover_boxes():
    """Return sorted {L: path} for lx_<name>.out files (skip *_corr.out)."""
    found = {}
    for p in BASE.glob("lx_*.out"):
        if p.stem.endswith("_corr"):
            continue
        token = p.stem.split("_", 1)[1] if "_" in p.stem else ""
        try:
            L = int(token)
        except ValueError:
            L = LX_NAME_TO_VALUE.get(token.lower())
        if L is not None:
            found[L] = p
    return dict(sorted(found.items()))


def main():
    boxes = discover_boxes()
    if not boxes:
        raise SystemExit("No lx_*.out files found.")

    print(f"{'L':>4}  {'Q(L)':>10}  {'1/Q(L)':>10}  file")
    for L, path in boxes.items():
        data = np.loadtxt(path, ndmin=2)
        q = hasimoto_Q(L)

        corrected = data.copy()
        corrected[:, TRANS_COLS] = data[:, TRANS_COLS] / q

        out = path.with_name(f"{path.stem}_corr.out")
        header = (
            f"Hasimoto simple-cubic correction, a={A_RADIUS}, L={L}\n"
            f"Q(L) = 1 - 2.8373*(a/L) + (4*pi/3)*(a/L)**3 = {q:.8f}\n"
            f"columns 1-6 (x,y,z,Ux,Uy,Uz) divided by Q(L); t and W unchanged\n"
            "t  x  y  z  Ux  Uy  Uz  Wx  Wy  Wz"
        )
        np.savetxt(out, corrected, fmt="% .8E", header=header)
        print(f"{L:>4}  {q:>10.6f}  {1.0/q:>10.6f}  {out.name}")


if __name__ == "__main__":
    main()
