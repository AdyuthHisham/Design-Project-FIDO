"""WP4: GNSS noise from the static RTK run vs the hand-set R in ekf_Controller.py, and EKF sensitivity to scaling R and Q.
REPLAY (static bag) + RE-IMPLEMENTATION (EKF). Sensitivity metrics are consistency/smoothness vs odometry, NOT accuracy."""
import sys, json; sys.path.insert(0, __file__.rsplit("/", 1)[0])
import numpy as np
from fido.data import *; from fido.ekf import replay, R_GPS, Q; from fido.geo import enu_aeqd; from fido.plot import *
from fig_ekf import metrics

# --- static noise
o, g = load("dec04_static"); e, n = enu_aeqd(g[:, 1], g[:, 2], g[0, 1], g[0, 2])
cov = np.cov(np.c_[e, n].T)
hand = np.diag(R_GPS)[:2] / 10                      # ekf_Controller.py R_gps (before the x10 inflation): 0.5, 0.5 m^2
res = dict(n_fix=len(g), std_e=float(e.std()), std_n=float(n.std()), cov_en=cov.tolist(), cov00_reported=float(g[:, 5].mean()),
           handset_R_gps_var_m2=hand.tolist(), handset_inflated_var_m2=np.diag(R_GPS)[:2].tolist())

# --- sensitivity
S = [0.01, 0.1, 1, 10, 100]; runs = ["nov24_circ_a", "dec04_L"]; grid = {}
for k in runs:
    o, g = load(k); gt = (g[:, 0] - o[0, 0]) / 1e9; ts = (o[:, 0] - o[0, 0]) / 1e9
    od = np.c_[np.interp(gt, ts, o[:, 1]), np.interp(gt, ts, o[:, 2])]
    J, D = np.zeros((5, 5)), np.zeros((5, 5))
    for i, sr in enumerate(S):
        for j, sq in enumerate(S):
            r = replay(o, g, hz=50, r_gps=R_GPS * sr, q={a: b * sq for a, b in Q.items()}, still_below=.05)
            xy = np.c_[np.interp(gt, r["t"], r["x"][:, 0]), np.interp(gt, r["t"], r["x"][:, 1])]
            J[i, j] = metrics(xy)["jerk_rms"]; D[i, j] = np.sqrt(np.mean(np.sum((xy - od) ** 2, 1)))
    grid[k] = dict(jerk=J.tolist(), rms_to_odom=D.tolist())

fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
a = ax[0, 0]; a.scatter(e - e.mean(), n - n.mean(), s=10, c=C["rtk"]); a.set_aspect("equal"); a.set_xlabel("E [m]"); a.set_ylabel("N [m]")
a.set_title(f"static RTK run (Dec04 18_29_15), {res['n_fix']} fixes\nstd E {res['std_e']*100:.1f} cm, N {res['std_n']*100:.1f} cm")
a = ax[0, 1]; a.bar(["static RTK\nmeasured", "hand-set R_gps\n(before x10)", "hand-set\n(x10, used)"], [np.sqrt(cov[0, 0]), np.sqrt(hand[0]), np.sqrt(np.diag(R_GPS)[0])], color=[C["rtk"], C["odom"], C["gnss"]])
a.set_yscale("log"); a.set_ylabel("position sigma [m] (log)"); a.set_title("GNSS position sigma: measured vs hand-set")
ax[0, 2].axis("off"); ax[0, 2].text(0, .9, "Reading the sensitivity maps\n\nRows: R_gps scale, columns: Q scale\n(1 = on-robot values).\nMetrics are smoothness / agreement with\nodometry, not accuracy.\nOdometry replay at 50 Hz, stationary gate on (ext.).", va="top", fontsize=9)
for c_, (m, ttl) in enumerate((("jerk", "jerk RMS [m]"), ("rms_to_odom", "RMS to odometry [m]"))):
    a = ax[1, c_]; M = np.mean([grid[k][m] for k in runs], 0)
    im = a.imshow(M, cmap="viridis", origin="lower"); a.set_xticks(range(5)); a.set_xticklabels(S); a.set_yticks(range(5)); a.set_yticklabels(S)
    a.set_xlabel("Q scale"); a.set_ylabel("R_gps scale"); a.set_title(f"{ttl}, mean of {len(runs)} runs"); a.grid(False); fig.colorbar(im, ax=a)
    for i in range(5):
        for j in range(5): a.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", color="w", fontsize=7)
ax[1, 2].axis("off")
fig.suptitle("GNSS noise calibration and EKF sensitivity to R and Q", fontsize=11); fig.tight_layout(); tag(fig, "REPLAY + RE-IMPLEMENTATION")
fig.savefig(OUT / "fig_noise.png"); json.dump(dict(static=res, sensitivity=dict(scales=S, runs=runs, grid=grid)), open(OUT / "noise.json", "w"), indent=1)
print(res)
