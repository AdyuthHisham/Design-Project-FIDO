"""Re-creation of the report's Fig. 5 (EKF test results: straight, circular, L-shaped): EKF estimate, RTK trajectory, odometry trajectory.
RE-IMPLEMENTATION (EKF with PAPER_EKF extensions) on REPLAY. Odometry is left in its own frame, as in the report; GNSS is aligned by the heading regression."""
import sys, json; sys.path.insert(0, __file__.rsplit("/", 1)[0])
import numpy as np
from fido.data import *; from fido.ekf import replay, PAPER_EKF; from fido.heading import align_gnss; from fido.plot import *

PANELS = [("dec04_line10", "(a) Straight"), ("nov24_circ_a", "(b) Circular"), ("dec04_L", "(c) L-shaped")]
fig, axs = plt.subplots(1, 3, figsize=(16, 5.4)); out = {}
for a, (k, ttl) in zip(axs, PANELS):
    o, g = load(k); r = replay(o, g, **PAPER_EKF); al = align_gnss(o, g, still_below=.05)
    ok = ~np.isnan(al["xy"]).any(1); T = (g[:, 0] - o[0, 0]) / 1e9
    xe = np.c_[np.interp(T, r["t"], r["x"][:, 0]), np.interp(T, r["t"], r["x"][:, 1])]
    a.plot(o[::10, 1], o[::10, 2], "r--", lw=1.2, label="Odometry Trajectory")
    a.plot(*al["xy"][ok].T, "-o", color="#f59e0b", ms=2.5, lw=1.2, label="RTK Trajectory" if (g[:, 4] == 2).mean() > .5 else "GNSS Trajectory")
    a.plot(r["x"][::10, 0], r["x"][::10, 1], color="#2563eb", lw=1.2, label="EKF Estimate"); a.plot(*xe[ok].T, ".", color="#2563eb", ms=4)
    a.set_xlabel("px [m]"); a.set_ylabel("py [m]"); a.set_title(f"{ttl}: {k}", fontsize=9); a.axis("equal"); a.legend(fontsize=7, loc="upper right")
    d = lambda p: float(np.sqrt(np.mean(np.sum(np.diff(p, 3, axis=0) ** 2, 1))))
    e10 = np.c_[np.interp(np.arange(0, r["t"][-1], .1), r["t"], r["x"][:, 0]), np.interp(np.arange(0, r["t"][-1], .1), r["t"], r["x"][:, 1])]
    out[k] = dict(rms_ekf_to_gnss_m=float(np.sqrt(np.mean(np.sum((xe[ok] - al["xy"][ok]) ** 2, 1)))), jerk10_ekf_m=d(e10), jerk_gnss_1hz_m=d(al["xy"][ok]),
                  odom_path_m=float(np.hypot(*np.diff(o[::10, 1:3], axis=0).T).sum()), ekf_path_m=float(np.hypot(*np.diff(e10, axis=0).T).sum()))
fig.suptitle("EKF test results for various trajectories (re-creation of report Fig. 5); RE-IMPLEMENTATION (EKF with extensions) on REPLAY", fontsize=10)
fig.tight_layout(); tag(fig, "RE-IMPLEMENTATION on REPLAY"); fig.savefig(OUT / "fig_ekf_paper.png"); json.dump(out, open(OUT / "ekf_paper.json", "w"), indent=1); print(json.dumps(out, indent=1))
