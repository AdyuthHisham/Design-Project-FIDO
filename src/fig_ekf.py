"""WP3 / paper Fig 5: CT-EKF replay on every GPS bag + ablation. RE-IMPLEMENTATION (ported recovered EKF) fed with REPLAYED bag data.
Metrics are consistency/smoothness against the odometry baseline, NOT accuracy (no ground truth exists)."""
import sys, json; sys.path.insert(0, __file__.rsplit("/", 1)[0])
import numpy as np
from matplotlib.patches import Ellipse
from fido.data import *; from fido.heading import *; from fido.ekf import *; from fido.plot import *

R_RTK = 40 * np.diag([2.7649e-3, 4.5062e-4, 1.9383])   # the offline harness's own calibrated-looking set (coordTurnClass.py:575-576)
CFG = {  # name -> (replay kwargs, status-switched R?)
    "EKF original": ({}, False),
    "EKF + stationary gate (ext.)": (dict(still_below=.05), False),
    "+ RTK/plain R switch (ext.)": (dict(still_below=.05), True),
    "+ odometry v, omega (ext.)": (dict(still_below=.05, odom_dyn=True), False),
    "EKF paper-like (ext.)": (PAPER_EKF, False),
}
RUNS_USED = [k for k in GPS_RUNS if k != "dec04_static"]


def m10(t, xy):
    """Smoothness at 10 Hz (sees the 1 Hz sawtooth that fix-time sampling hides): path length and jerk RMS (third difference)."""
    T = np.arange(0, t[-1], .1); p = np.c_[np.interp(T, t, xy[:, 0]), np.interp(T, t, xy[:, 1])]
    return float(np.hypot(*np.diff(p, axis=0).T).sum()), float(np.sqrt(np.mean(np.sum(np.diff(p, 3, axis=0) ** 2, 1))))


def metrics(xy, odom_len=None):
    """xy sampled at GNSS fix times (~1 Hz, same for every method): path length, jerk RMS (third difference), step roughness."""
    d = np.diff(xy, axis=0)
    jerk = np.diff(xy, 3, axis=0)
    return dict(path_len=float(np.hypot(*d.T).sum()), jerk_rms=float(np.sqrt(np.mean(np.sum(jerk ** 2, 1)))) if len(jerk) else np.nan)


def run_all():
    out = {}
    for k in RUNS_USED:
        o, g = load(k); gt = (g[:, 0] - o[0, 0]) / 1e9; ts = (o[:, 0] - o[0, 0]) / 1e9
        odom_at = np.c_[np.interp(gt, ts, o[:, 1]), np.interp(gt, ts, o[:, 2])]
        r = dict(odom=odom_at, gt=gt)
        al = align_gnss(o, g, still_below=.05)
        r["t10"] = {"odometry only": (ts, np.c_[o[:, 1], o[:, 2]])}
        r["methods"] = {"odometry only": odom_at, "GNSS only (aligned, gated)": al["xy"]}
        r["runs"] = {}
        for nm, (kw, sw) in CFG.items():
            rr = replay(o, g, status_R=(R_RTK, R_GPS) if sw else None, **kw)
            xs = np.c_[np.interp(gt, rr["t"], rr["x"][:, 0]), np.interp(gt, rr["t"], rr["x"][:, 1])]
            r["methods"][nm] = xs; r["runs"][nm] = rr; r["t10"][nm] = (rr["t"], rr["x"][:, :2])
        out[k] = r
    return out


if __name__ == "__main__":
    R = run_all(); table = {}
    for k, r in R.items():
        table[k] = {}
        L0 = metrics(r["odom"])["path_len"]
        for nm, xy in r["methods"].items():
            ok = ~np.isnan(xy).any(1)
            m = metrics(xy[ok]); m["len_ratio_vs_odom"] = m["path_len"] / metrics(r["odom"][ok])["path_len"] if ok.sum() > 3 else np.nan
            m["rms_to_odom"] = float(np.sqrt(np.mean(np.sum((xy[ok] - r["odom"][ok]) ** 2, 1)))) if ok.any() else np.nan
            if nm in r["t10"]: m["len10"], m["jerk10"] = m10(*r["t10"][nm])
            table[k][nm] = m
        for nm in table[k]:
            if "len10" in table[k][nm]: table[k][nm]["len10_ratio"] = table[k][nm]["len10"] / table[k]["odometry only"]["len10"]
        for nm, rr in r["runs"].items():
            gl = rr["gps"]
            if len(gl):
                yy, ss = gl[:, 4:6], gl[:, 7:9]
                table[k][nm].update(n_gnss_updates=len(gl), innov_xy_rms=float(np.sqrt(np.mean(yy ** 2))), nis_xy_mean=float(np.mean(yy ** 2 / ss)),
                                    innov_yaw_rms=float(np.sqrt(np.mean(gl[:, 6] ** 2))))
    json.dump(table, open(OUT / "ekf.json", "w"), indent=1)

    # --- overview grid, gated EKF
    ks = RUNS_USED; fig, axs = plt.subplots(3, 4, figsize=(15, 11)); nm = "EKF paper-like (ext.)"
    for a, k in zip(axs.flat, ks):
        r = R[k]; rr = r["runs"][nm]; o, g = load(k); ts = (o[:, 0] - o[0, 0]) / 1e9
        a.plot(o[::10, 1], o[::10, 2], color=C["odom"], lw=1.5, label="odometry")
        xy = r["methods"]["GNSS only (aligned, gated)"]; a.plot(*xy.T, ".", ms=3, color=C["gnss"], label="GNSS (aligned)")
        a.plot(rr["x"][::10, 0], rr["x"][::10, 1], color=C["ekf"], lw=1.4, label="EKF")
        a.set_title(f"{k}\n{RUNS[k][1]}", fontsize=8); a.axis("equal")
    axs.flat[0].legend(fontsize=7)
    for a in list(axs.flat)[len(ks):]: a.axis("off")
    fig.suptitle("EKF replay on all GPS bags (paper-like EKF, ext.); RE-IMPLEMENTED EKF on REPLAYED data", fontsize=11)
    fig.tight_layout(); tag(fig, "RE-IMPLEMENTATION on REPLAY"); fig.savefig(OUT / "fig_ekf_all.png"); plt.close(fig)

    # --- detail: 3 runs: path + 2-sigma ellipses + innovations
    sel = ["nov24_circ_a", "dec04_L", "nov25_circ5"]; fig, axs = plt.subplots(3, 3, figsize=(14, 11.5))
    for j, k in enumerate(sel):
        r = R[k]; rr = r["runs"][nm]; o, g = load(k); a = axs[0, j]
        a.plot(o[::10, 1], o[::10, 2], color=C["odom"], lw=1.5, label="odometry")
        a.plot(*r["methods"]["GNSS only (aligned, gated)"].T, ".-", ms=4, lw=.5, color=C["gnss"], label="GNSS aligned")
        a.plot(rr["x"][::10, 0], rr["x"][::10, 1], color=C["ekf"], lw=1.6, label="EKF")
        for i in range(0, len(rr["t"]), max(1, len(rr["t"]) // 25)):
            P = rr["P"][i][:2, :2]; w, V = np.linalg.eigh(P); ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
            a.add_patch(Ellipse(rr["x"][i, :2], 4 * np.sqrt(w[1]), 4 * np.sqrt(w[0]), angle=ang, fill=False, ec=C["ekf"], lw=.6, alpha=.6))
        a.set_title(f"{k}: {RUNS[k][1]}\n2σ position ellipses from P", fontsize=9); a.axis("equal")
        if j == 0: a.legend(fontsize=7)
        gl = rr["gps"]; a = axs[1, j]
        if len(gl):
            a.plot(gl[:, 0], gl[:, 4], ".-", ms=3, lw=.5, label="x"); a.plot(gl[:, 0], gl[:, 5], ".-", ms=3, lw=.5, label="y")
            a.axhline(0, color="k", lw=.5)
        a.set(title="GNSS innovation (position) [m]", xlabel="t [s]"); a.legend(fontsize=7)
        a = axs[2, j]
        if len(gl): a.plot(gl[:, 0], np.degrees(gl[:, 6]), ".-", ms=3, lw=.5, color=C["bad"])
        a.plot(rr["odom_inn"][::100, 0], np.degrees(rr["odom_inn"][::100, 3]), lw=.6, color=C["odom"], label="odometry-pose innovation (yaw)")
        a.set(title="yaw innovation [deg] (red = GNSS update)", xlabel="t [s]"); a.legend(fontsize=7)
    fig.suptitle("EKF detail; RE-IMPLEMENTATION on REPLAY. paper-like EKF (ext.): odometry cost schedule + odometry v, omega", fontsize=11)
    fig.tight_layout(); tag(fig, "RE-IMPLEMENTATION on REPLAY"); fig.savefig(OUT / "fig_ekf.png"); plt.close(fig)

    # --- ablation bars (moving circle/line runs): length ratio and jerk
    meths = list(R[ks[0]]["methods"]); fig, axs = plt.subplots(1, 3, figsize=(15, 4.4)); x = np.arange(len(ks))
    for a, (m, lab) in zip(axs, (("len10_ratio", "path length / odometry path length (10 Hz)"), ("jerk10", "jerk RMS at 10 Hz [m] (third difference)"), ("rms_to_odom", "RMS distance to odometry [m] (consistency, not accuracy)"))):
        w = .8 / len(meths)
        for i, mm in enumerate(meths):
            a.bar(x + (i - len(meths) / 2 + .5) * w, [table[k][mm].get(m, np.nan) for k in ks], w, label=mm)
        a.set_xticks(x); a.set_xticklabels([k.replace("_", "\n", 1) for k in ks], fontsize=6); a.set_title(lab, fontsize=9)
        if m == "len10_ratio": a.set_ylim(0, 9)
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, fontsize=7, ncol=6, loc="lower center")
    fig.suptitle("Ablation: odometry only / GNSS only / EKF variants; RE-IMPLEMENTATION on REPLAY", fontsize=11)
    fig.tight_layout(rect=(0, .07, 1, 1)); tag(fig, "RE-IMPLEMENTATION on REPLAY"); fig.savefig(OUT / "fig_ekf_ablation.png"); plt.close(fig)
    # summary means
    summ = {}
    for mm in meths:
        summ[mm] = {m: float(np.nanmean([table[k][mm].get(m, np.nan) for k in ks])) for m in ("len_ratio_vs_odom", "jerk_rms", "rms_to_odom", "len10_ratio", "jerk10")}
    json.dump(summ, open(OUT / "ekf_summary.json", "w"), indent=1)
    for mm, v in summ.items(): print(mm.ljust(34), {a: round(b, 3) for a, b in v.items()})
