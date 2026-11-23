"""Test: does raising the odometry cost R after each GNSS fix and decaying it (instead of scaling the gain) smooth the EKF?
Odometry replayed at 50 Hz (the on-robot rate). Smoothness measured on the EKF output at 10 Hz. Compared with odometry only. Not accuracy."""
import sys, json; sys.path.insert(0, __file__.rsplit("/", 1)[0])
import numpy as np
from fido.data import *; from fido.ekf import replay; from fido.plot import *

RUNS_T = ["nov24_circ_a", "nov24_circ_b", "dec04_L", "dec04_line10", "nov25_circ5", "nov25_line15"]
V = {"odometry only": None, "original (gain scale)": dict(), "R schedule": dict(odom_mode="R"),
     "R schedule + stationary gate (ext.)": dict(odom_mode="R", still_below=.05),
     "R schedule + odometry v,w (ext.)": dict(odom_mode="R", odom_dyn=True, still_below=.05),
     "  ... decay 10x slower": dict(odom_mode="R", odom_dyn=True, still_below=.05, odom_rec=0.00025),
     "  ... decay 100x slower": dict(odom_mode="R", odom_dyn=True, still_below=.05, odom_rec=0.000025)}

def smooth(xy):
    j = np.diff(xy, 3, axis=0)                                  # 10 Hz third difference [m]
    return dict(len=float(np.hypot(*np.diff(xy, axis=0).T).sum()), jerk=float(np.sqrt(np.mean(np.sum(j ** 2, 1)))),
                max_step=float(np.hypot(*np.diff(xy, axis=0).T).max()))

res, paths = {}, {}
for k in RUNS_T:
    o, g = load(k); ts = (o[:, 0] - o[0, 0]) / 1e9; T = np.arange(0, ts[-1], .1); res[k] = {}; paths[k] = {}
    for nm, kw in V.items():
        if kw is None: xy = np.c_[np.interp(T, ts, o[:, 1]), np.interp(T, ts, o[:, 2])]
        else:
            r = replay(o, g, hz=50, **kw); xy = np.c_[np.interp(T, r["t"], r["x"][:, 0]), np.interp(T, r["t"], r["x"][:, 1])]
        paths[k][nm] = xy; res[k][nm] = smooth(xy)
    L0 = res[k]["odometry only"]["len"]
    for nm in V: res[k][nm]["len_ratio"] = res[k][nm]["len"] / L0
mean = {nm: {m: float(np.mean([res[k][nm][m] for k in RUNS_T])) for m in ("len_ratio", "jerk", "max_step")} for nm in V}
json.dump(dict(per_run=res, mean=mean), open(OUT / "odom_schedule.json", "w"), indent=1)
for nm, v in mean.items(): print(nm.ljust(38), {a: round(b, 3) for a, b in v.items()})
col = ["#6b7280", "#dc2626", "#2563eb", "#0891b2", "#16a34a", "#7c3aed", "#db2777"]
fig, ax = plt.subplots(2, 3, figsize=(14, 8))
for a, k in zip(ax[0], ["nov24_circ_a", "dec04_L", "nov25_circ5"]):
    for (nm, xy), c in zip(paths[k].items(), col): a.plot(*xy.T, color=c, lw=2 if nm == "odometry only" else 1, label=nm)
    a.set_title(f"{k}: {RUNS[k][1]}", fontsize=9); a.axis("equal")
ax[0, 0].legend(fontsize=6.5)
names = list(V); x = np.arange(len(names))
for a, m, t in zip(ax[1], ("jerk", "len_ratio", "max_step"), ("jerk RMS at 10 Hz [m] (lower = smoother)", "path length / odometry path length", "largest 0.1 s step [m]")):
    a.bar(x, [mean[n][m] for n in names], color=col); a.set_xticks(x); a.set_xticklabels([n.strip(" .") for n in names], fontsize=7, rotation=35, ha="right"); a.set_title(t + ", mean of 6 runs", fontsize=9)
fig.suptitle("Odometry-cost schedule test (R scaled by 1/odom_red after each GNSS fix) vs original gain schedule; 50 Hz odometry", fontsize=10)
fig.tight_layout(); tag(fig, "RE-IMPLEMENTATION on REPLAY"); fig.savefig(OUT / "fig_odom_schedule.png")
