"""GIFs: ekf_replay.gif (REPLAY + re-implemented EKF) and control_sim.gif (SIMULATION)."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse
from fido.data import *; from fido.ekf import replay, PAPER_EKF; from fido.heading import align_gnss; from fido.plot import *
from fido.sim import simulate, noise_pools; from fido.control import ref_path

# ---- EKF replay
k = "nov24_circ_a"; o, g = load(k); r = replay(o, g, **PAPER_EKF); al = align_gnss(o, g, still_below=.05)
ts = (o[:, 0] - o[0, 0]) / 1e9; N = 90; T = np.linspace(0, ts[-1], N)
fig, a = plt.subplots(figsize=(6.4, 6.4)); a.axis("equal")
a.set_xlim(r["x"][:, 0].min() - 4, r["x"][:, 0].max() + 4); a.set_ylim(r["x"][:, 1].min() - 4, r["x"][:, 1].max() + 4)
ln_o, = a.plot([], [], color=C["odom"], lw=2, label="odometry"); pts, = a.plot([], [], ".", color=C["gnss"], ms=5, label="GNSS (aligned)"); ln_e, = a.plot([], [], color=C["ekf"], lw=1.8, label="EKF (paper-like, ext.)")
el = Ellipse((0, 0), 0, 0, fill=False, ec=C["ekf"]); a.add_patch(el); ttl = a.set_title(""); a.legend(loc="upper left", fontsize=8)
tag(fig, "REPLAY + RE-IMPLEMENTATION", k)
def upd(i):
    t = T[i]; m = ts <= t; ln_o.set_data(o[m, 1][::10], o[m, 2][::10]); gm = (al["t"] <= t) & ~np.isnan(al["xy"][:, 0]); pts.set_data(al["xy"][gm, 0], al["xy"][gm, 1])
    j = np.searchsorted(r["t"], t) - 1; ln_e.set_data(r["x"][:j, 0][::10], r["x"][:j, 1][::10])
    P = r["P"][j][:2, :2]; w, V = np.linalg.eigh(P); el.set_center(r["x"][j, :2]); el.width, el.height = 4 * np.sqrt(w[1]), 4 * np.sqrt(w[0]); el.angle = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    ttl.set_text(f"t = {t:5.1f} s   (ellipse = 2σ position from P)"); return ln_o, pts, ln_e, el
FuncAnimation(fig, upd, frames=N).save(OUT / "ekf_replay.gif", writer=PillowWriter(fps=12)); plt.close(fig)

# ---- control sim
pools, scale = noise_pools(); ref = ref_path("circle")
runs = {m: simulate(ref, m, "holonomic", "plain" if m not in ("truth", "odom") else "rtk", 0, pools, scale) for m in ("odom", "gnss", "ekf_ext")}
col = dict(odom=C["odom"], gnss=C["gnss"], ekf_ext=C["ekf"]); lab = dict(odom="odometry only", gnss="raw GNSS + odometry increments", ekf_ext="EKF + odometry v,ω (ext.)")
fig, a = plt.subplots(figsize=(6.4, 6.4)); a.set_aspect("equal"); a.set_xlim(-8, 8); a.set_ylim(-3, 13); a.plot(ref[:, 0], ref[:, 1], "k--", lw=1, label="reference (5 m circle)")
L = {m: a.plot([], [], color=col[m], lw=1.5, label=lab[m])[0] for m in runs}; D = {m: a.plot([], [], "o", color=col[m])[0] for m in runs}
a.legend(loc="lower center", fontsize=7); ttl = a.set_title(""); tag(fig, "SIMULATION", "plain-GNSS noise magnitudes from bags")
frames = np.linspace(2, len(ref) - 1, 100).astype(int)
def upd2(f):
    for m, rr in runs.items(): L[m].set_data(rr["true"][:f, 0], rr["true"][:f, 1]); D[m].set_data([rr["true"][f, 0]], [rr["true"][f, 1]])
    ttl.set_text(f"LQR (Q=diag(10,10,60), R=I) closed loop, t = {f * .02:5.1f} s"); return list(L.values()) + list(D.values())
FuncAnimation(fig, upd2, frames=frames).save(OUT / "control_sim.gif", writer=PillowWriter(fps=15)); plt.close(fig)
print("ok")
