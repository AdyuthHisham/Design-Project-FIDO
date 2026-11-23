"""WP5 / paper IV-B: closed-loop LQR (Q=diag(10,10,60), R=I, dt=0.02, DARE gain) on four estimators. SIMULATION.
Plant: holonomic velocity command (what A=I, B=dt*I in ekf_Controller.py implies; Go2 can strafe) and a unicycle variant.
Noise: GNSS error magnitudes resampled from real bag residuals; odometry scale error 1.21 from RTK-circle-radius / odometry-radius (assumes RTK ~ truth)."""
import sys, json; sys.path.insert(0, __file__.rsplit("/", 1)[0])
import numpy as np
from fido.sim import *; from fido.control import ref_path; from fido.plot import *; from fido.data import OUT

if __name__ == "__main__":
    pools, scale = noise_pools()
    PATHS = ["circle", "line_turn", "L", "line10"]; MODES = ["truth", "odom", "gnss", "gnss_zoh", "ekf", "ekf_ext"]
    PLANT = "holonomic"; NS = 6; table = {}; runs = {}
    for pn in PATHS:
        ref = ref_path(pn)
        for q in ("rtk", "plain"):
            for m in MODES:
                if q == "plain" and m in ("truth", "odom"): continue
                n = 1 if m in ("truth", "odom", "gnss_zoh") else NS
                ms = []
                for sd in range(n):
                    r = simulate(ref, m, PLANT, q, sd, pools, scale); ms.append(metrics(ref, r)); runs[(pn, q, m)] = r if sd == 0 else runs.get((pn, q, m), r)
                table[f"{pn}|{q}|{m}"] = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]} | {"xtrack_rms_sd": float(np.std([x["xtrack_rms"] for x in ms])), "n": n}
        print(pn, flush=True)
    # unicycle variant
    for q in ("rtk", "plain"):
        ref = ref_path("circle")
        for m in ("truth", "odom", "gnss", "ekf_ext"):
            if q == "plain" and m in ("truth", "odom"): continue
            n = 1 if m in ("truth", "odom") else 4
            ms = [metrics(ref, simulate(ref, m, "unicycle", q, sd, pools, scale)) for sd in range(n)]
            table[f"circle_unicycle|{q}|{m}"] = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]} | {"xtrack_rms_sd": float(np.std([x["xtrack_rms"] for x in ms])), "n": n}
    json.dump(dict(scale=scale, noise_std=dict(rtk=float(pools["rtk"].std()), plain=float(pools["plain"].std())), table=table), open(OUT / "control.json", "w"), indent=1)
    # --- figure
    col = dict(truth=C["ref"], odom=C["odom"], gnss=C["gnss"], gnss_zoh="#f59e0b", ekf=C["bad"], ekf_ext=C["ekf"])
    lab = dict(truth="true state (oracle)", odom="odometry only", gnss="raw GNSS + odometry increments", gnss_zoh="raw GNSS, held 1 Hz", ekf="recovered EKF", ekf_ext="EKF + odometry v,ω (ext.)")
    fig, axs = plt.subplots(2, 4, figsize=(16, 8.4))
    for j, pn in enumerate(PATHS):
        a = axs[0, j]; ref = ref_path(pn); a.plot(ref[:, 0], ref[:, 1], "k--", lw=1, label="reference")
        for m in ("odom", "gnss", "ekf_ext", "ekf"):
            T = runs[(pn, "plain", m)]["true"] if (pn, "plain", m) in runs else runs[(pn, "rtk", m)]["true"]
            a.plot(T[:, 0], T[:, 1], color=col[m], lw=1.1, label=lab[m])
        a.set_title(f"{pn} (plain-GNSS noise, seed 0)"); a.set_aspect("equal", adjustable="box"); a.set_xlim(ref[:, 0].min() - 3, ref[:, 0].max() + 3); a.set_ylim(ref[:, 1].min() - 3, ref[:, 1].max() + 3)
        a.text(.02, .02, "recovered EKF: diverges, off-scale", transform=a.transAxes, fontsize=6.5, color=col["ekf"])
        if j == 0: a.legend(fontsize=6.5)
        a = axs[1, j]; ms = [m for m in MODES if m != "gnss_zoh"]; x = np.arange(len(ms))
        for off, q, hat in ((-.2, "rtk", ""), (.2, "plain", "//")):
            v = [table.get(f"{pn}|{q}|{m}", {}).get("xtrack_rms", np.nan) for m in ms]
            a.bar(x + off, np.minimum(v, 6), .38, color=[col[m] for m in ms], hatch=hat, alpha=.9 if q == "rtk" else .6, edgecolor="w")
            for xi, vv in zip(x + off, v):
                if vv > 6: a.text(xi, 5.5, f"{vv:.0f}", ha="center", fontsize=7, rotation=90, color="w")
        a.set_xticks(x); a.set_xticklabels(["truth", "odom", "GNSS+odo", "EKF rec.", "EKF ext."], fontsize=7); a.set_ylabel("cross-track RMS [m]"); a.set_ylim(0, 6)
        a.set_title("solid RTK, hatched plain (>6 clipped)", fontsize=8)
    fig.suptitle(f"Closed-loop LQR tracking, holonomic plant, odometry scale {scale:.2f}; SIMULATION (noise magnitudes resampled from bags)", fontsize=11)
    fig.tight_layout(); tag(fig, "SIMULATION"); fig.savefig(OUT / "fig_control.png"); plt.close(fig)
    for k, v in table.items(): print(k.ljust(26), {a: round(b, 2) for a, b in v.items() if a in ("xtrack_rms", "xtrack_max", "heading_rms_deg", "effort_rms")})
