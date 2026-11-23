"""WP6: parse the two pasted traj_server terminal logs (per 20 ms cycle: vx, vy[printed as 'vx'], omega) and compare with the LQR command
the controller would give. REPLAY of logged commands vs SIMULATION of the LQR. Only the start of log B can be compared; see json/caption."""
import sys, json, os; sys.path.insert(0, __file__.rsplit("/", 1)[0])
import numpy as np
from fido.control import lqr_gain, DT; from fido.data import OUT, ROOT; from fido.plot import *

def parse(path):
    L = open(path).read().splitlines(); wp = L[0].split("[", 2)[-1]
    vals = []
    for l in L:
        p = l.replace("[traj_server-2]", "").split()
        if len(p) == 2 and p[0] in ("vx", "omega"):
            vals.append((p[0], float(p[1])))
    vx, vy, om = [], [], []; k = 0
    for name, v in vals:
        if name == "vx": (vx if len(vx) == len(vy) else vy).append(v)
        else: om.append(v)
    n = min(len(vx), len(vy), len(om)); return wp, np.array(vx[:n]), np.array(vy[:n]), np.array(om[:n])

D = ROOT / "_extract"; files = {"A": [f for f in os.listdir(D) if f.startswith("[traj_server")][0], "B": [f for f in os.listdir(D) if f.startswith("traj_server")][0]}
K = lqr_gain(); res = {}; fig, ax = plt.subplots(2, 3, figsize=(14, 6.5))
for r, (nm, f) in enumerate(files.items()):
    wp, vx, vy, om = parse(D / f); t = np.arange(len(vx)) * DT
    res[nm] = dict(file=f, first_line=wp, cycles=len(vx), vx_range=[float(vx.min()), float(vx.max())], vy_range=[float(vy.min()), float(vy.max())], omega_range=[float(om.min()), float(om.max())])
    for c, (y, n, u) in enumerate(((vx, "vx [m/s]", ""), (vy, "vy [m/s]", ""), (om, "omega [rad/s]", ""))):
        ax[r, c].plot(t, y, color=C["rtk"], lw=1.2, label="logged"); ax[r, c].set_xlabel("t [s] (cycle x 0.02)"); ax[r, c].set_ylabel(n)
    ax[r, 0].set_title(f"log {nm}: {len(vx)} cycles, first line {wp}", fontsize=8)
# overlay: robot not yet moving, reference advancing at v=0.3 m/s: vx = Kx * 0.3 * t (log B start). Heading error implied by omega0 / Kpsi.
_, vx, vy, om = parse(D / files["B"]); t = np.arange(len(vx)) * DT; n = 40
sim = K[0, 0] * 0.3 * t[:n]; ax[1, 0].plot(t[:n], sim, "--", color=C["bad"], label="LQR sim: ref moves 0.3 m/s, robot still"); ax[1, 0].legend(fontsize=7)
slope_log = float(np.polyfit(t[:20], vx[:20], 1)[0]); slope_sim = float(K[0, 0] * 0.3)
res["B"]["start_slope_logged_mps2"] = slope_log; res["B"]["start_slope_sim_mps2"] = slope_sim
res["B"]["implied_initial_heading_error_rad"] = float(om[0] / K[2, 2]); res["B"]["omega_decay_per_cycle_logged"] = float(om[20] / om[0]) ** (1 / 20)
res["B"]["omega_decay_per_cycle_if_robot_followed_LQR"] = 1 - DT * K[2, 2]
res["A"]["note"] = "vx/vy reach +-14 m/s and omega is stuck at one value: the controller diverged/saturated; not a usable trace of a planned path"
fig.suptitle("traj_server control logs: logged commands (REPLAY of pasted terminal output) vs LQR sim overlay (SIMULATION)", fontsize=10)
fig.tight_layout(); tag(fig, "REPLAY (logs) + SIMULATION overlay"); fig.savefig(OUT / "fig_logs.png"); json.dump(res, open(OUT / "logs.json", "w"), indent=1); print(json.dumps(res, indent=1))
