import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

data_dir = Path(r"C:\Users\paraj\Documents\particle_inference_test\data\newtonian\angled_delta")
out_path = Path(r"C:\Users\paraj\Documents\particle_inference_test\plots\angled_delta_t_vs_x.png")

files = sorted(data_dir.glob("*.txt"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))

fig, ax = plt.subplots(figsize=(7, 5))
for f in files:
    angle = int(re.search(r"(\d+)", f.stem).group(1))
    d = np.loadtxt(f)
    ax.plot(d[:, 0], d[:, 1], label=f"angle = {angle}")

ax.set_xlabel("t")
ax.set_ylabel("x(t)")
ax.set_title("t vs x(t)")
ax.legend()
fig.tight_layout()
fig.savefig(out_path, dpi=150)
print(f"wrote {out_path}")
