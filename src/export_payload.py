"""WP7: browser replay payload for the planned fido-ekf demo. REPLAY + RE-IMPLEMENTATION. Downsampled to ~5 Hz, positions in cm-rounded metres."""
import sys, json; sys.path.insert(0, __file__.rsplit("/", 1)[0])
import numpy as np
from fido.data import *; from fido.ekf import replay, PAPER_EKF; from fido.heading import align_gnss

KEYS = {"nov24_circ_a": "circle r~5 m, RTK", "dec04_L": "10 m + 90 deg turn + 5 m, RTK", "nov25_circ5": "circle r=5 m, plain GNSS"}
out = dict(note="REPLAY of bag data + re-implemented CT-EKF (EKF + stationary gate + odometry v,omega: extensions). Odometry is a relative baseline, not ground truth. Positions in metres, odometry frame.", runs={})
for k, lab in KEYS.items():
    o, g = load(k); r = replay(o, g, **PAPER_EKF); al = align_gnss(o, g, still_below=.05)
    ts = (o[:, 0] - o[0, 0]) / 1e9; T = np.arange(0, ts[-1], 0.2); R = lambda a: np.round(a, 2).tolist()
    xo = np.c_[np.interp(T, ts, o[:, 1]), np.interp(T, ts, o[:, 2])]; xe = np.c_[np.interp(T, r["t"], r["x"][:, 0]), np.interp(T, r["t"], r["x"][:, 1])]
    ell = []
    for i in range(0, len(T), 5):                      # 1 Hz ellipses: 2-sigma semi-axes and angle from P[:2,:2]
        j = min(np.searchsorted(r["t"], T[i]), len(r["t"]) - 1); w, V = np.linalg.eigh(r["P"][j][:2, :2])
        ell.append([round(T[i], 1), *R(xe[i]), round(float(2 * np.sqrt(w[1])), 2), round(float(2 * np.sqrt(w[0])), 2), round(float(np.degrees(np.arctan2(V[1, 1], V[0, 1]))), 1)])
    ok = ~np.isnan(al["xy"]).any(1)
    out["runs"][k] = dict(label=lab, rtk=bool((g[:, 4] == 2).mean() > .5), t=R(T), odom=R(xo), ekf=R(xe),
                          gnss=[[round(float(t), 1), *R(p)] for t, p in zip(al["t"][ok], al["xy"][ok])], ellipse_t_x_y_a_b_deg=ell)
s = json.dumps(out, separators=(",", ":")); open(OUT / "fido_ekf_replay.json", "w").write(s); print(len(s) / 1e3, "kB")
