"""Contact sheet: odometry path and GNSS fixes (local ENU metres) for every bag."""
import numpy as np, glob
from pathlib import Path
from pyproj import Transformer
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
fs = sorted(glob.glob(str(ROOT/"data/*.npz")))
fig, ax = plt.subplots(4, 5, figsize=(20, 16)); ax = ax.ravel()
for a, f in zip(ax, fs):
    d = np.load(f); o, g = d["odom"], d["gps"]
    a.plot(o[:,1], o[:,2], c="gray", lw=1, label="odometry (own frame)")
    a.set_title(Path(f).stem.replace("__","\n"), fontsize=8)
    if len(g):
        e = Transformer.from_crs(4326, f"+proj=aeqd +lat_0={g[0,1]} +lon_0={g[0,2]}", always_xy=True)
        x, y = e.transform(g[:,2], g[:,1])
        rtk = g[:,4] == 2
        a.scatter(x[~rtk], y[~rtk], s=6, c="tab:red", label="GNSS plain")
        a.scatter(x[rtk], y[rtk], s=6, c="tab:green", label="GNSS RTK")
    a.set_aspect("equal"); a.grid(alpha=.3)
ax[0].legend(fontsize=7)
for a in ax[len(fs):]: a.axis("off")
fig.tight_layout(); fig.savefig(ROOT/"out/overview.png", dpi=70)
