"""WP2 / paper Fig 3: plain vs RTK GNSS on circle runs. REPLAY. Metrics: (1) GNSS vs odometry after paper-style alignment,
(2) after best rigid fit (alignment-independent lower bound), (3) circle-fit residual (does not use odometry at all)."""
import sys, json; sys.path.insert(0, __file__.rsplit("/", 1)[0])
import numpy as np
from fido.data import *; from fido.heading import *; from fido.plot import *

CIRC = {"rtk": ["nov24_circ_a", "nov24_circ_b"], "plain": ["nov25_circ5", "nov25_circ7", "nov18_circ_a", "nov18_circ_b", "jan12_plain"]}
rm = lambda a, b: float(np.sqrt(np.mean(np.sum((a - b) ** 2, 1))))
res = {}
for grp, ks in CIRC.items():
    for k in ks:
        o, g = load(k)
        r0 = align_gnss(o, g); r1 = align_gnss(o, g, still_below=0.05)
        th, rig, pa = rigid_fit(r0["en"], r0["odom"])
        # keep only fixes where robot moves for the circle fit
        mv = np.hypot(np.gradient(r0["odom"][:, 0]), np.gradient(r0["odom"][:, 1])) > 0.2
        cg = circle_fit(r0["en"][mv]); co = circle_fit(r0["odom"][mv])
        res[k] = dict(group=grp, status=sorted(int(s) for s in set(g[:, 4])), n_fix=len(g), rmse_paper_style=rm(r0["xy"], r0["odom"]),
                      rmse_gated=rm(r1["xy"], r1["odom"]), rmse_rigid=rig, circ_r_gnss=cg[2], circ_rms_gnss=cg[3],
                      circ_r_odom=co[2], circ_rms_odom=co[3], pa=pa, odom=r0["odom"], en=r0["en"])
# static noise floor
o, g = load("dec04_static"); from fido.geo import enu_aeqd
e, n = enu_aeqd(g[:, 1], g[:, 2], g[0, 1], g[0, 2])
floor = dict(std_e=float(e.std()), std_n=float(n.std()), max_jump=float(np.max(np.hypot(np.diff(e), np.diff(n)))), n=len(g))
# dropout run
o, g = load("jan12_mixed"); r0 = align_gnss(o, g); th, rig, pa = rigid_fit(r0["en"], r0["odom"])
st = g[:, 4]; err = np.hypot(*(pa - r0["odom"]).T)
chg = np.flatnonzero(np.diff(st) != 0) + 1
drop = dict(status_changes_at_s=[float(r0["t"][i]) for i in chg], frac_rtk=float((st == 2).mean()),
            rms_err_rtk=float(np.sqrt(np.mean(err[st == 2] ** 2))), rms_err_plain=float(np.sqrt(np.mean(err[st == 0] ** 2))))

summ = {}
for grp in CIRC:
    v = [res[k] for k in CIRC[grp]]
    summ[grp] = {m: float(np.mean([x[m] for x in v])) for m in ("rmse_paper_style", "rmse_gated", "rmse_rigid", "circ_rms_gnss", "circ_r_gnss", "circ_r_odom")}
    summ[grp]["n_runs"] = len(v)

fig = plt.figure(figsize=(12, 9.5)); gs = fig.add_gridspec(3, 4)
for i, k in enumerate(["nov24_circ_a", "nov25_circ5"]):
    a = fig.add_subplot(gs[0, 2 * i:2 * i + 2]); r = res[k]
    a.plot(*r["odom"].T - r["odom"].mean(0)[:, None], color=C["odom"], lw=2, label="odometry")
    a.plot(*(r["pa"] - r["odom"].mean(0)).T, "o-", ms=3, lw=.7, color=C["rtk"] if r["group"] == "rtk" else C["gnss"], label="GNSS (best rigid fit onto odometry)")
    a.set(title=f"({'ab'[i]}) {k}: {RUNS[k][1]}\nrigid RMSE {r['rmse_rigid']:.2f} m, circle-fit r={r['circ_r_gnss']:.1f} m, radial RMS {r['circ_rms_gnss']:.2f} m", xlabel="x [m]", ylabel="y [m]"); a.axis("equal"); a.legend(fontsize=7)
allk = CIRC["plain"] + CIRC["rtk"]; x = np.arange(len(allk)); w = .27
def panel_c(a):
    for j, (m, lab, c) in enumerate((("rmse_paper_style", "paper-style alignment (orig rule)", C["bad"]), ("rmse_gated", "paper-style + stationary gate (ext.)", C["rtk"]), ("rmse_rigid", "best rigid fit (lower bound)", C["ref"]))):
        a.bar(x + (j - 1) * w, [res[k][m] for k in allk], w, color=c, label=lab)
    a.axvline(len(CIRC["plain"]) - .5, color="k", lw=.8); a.text(1.5, 10.2, "plain GNSS", ha="center"); a.text(5.5, 10.2, "RTK", ha="center")
    a.set_xticks(x); a.set_xticklabels([k.replace("_", "\n", 1) for k in allk], fontsize=6.5); a.set(ylabel="RMSE vs odometry [m]", title="(a) GNSS-vs-odometry RMSE per circle run", ylim=(0, 11)); a.legend(fontsize=6.5, loc="upper right", ncol=1, bbox_to_anchor=(1, .93))
def panel_d(a):
    a.bar(x, [res[k]["circ_rms_gnss"] for k in allk], color=[C["gnss"]] * len(CIRC["plain"]) + [C["rtk"]] * len(CIRC["rtk"]))
    a.set_xticks(x); a.set_xticklabels([k.replace("_", "\n", 1) for k in allk], fontsize=6.5)
    for i, k in enumerate(allk): a.text(i, res[k]["circ_rms_gnss"] + .03, f"r={res[k]['circ_r_gnss']:.1f}", ha="center", fontsize=7)
    a.set(ylabel="RMS radial residual [m]", title="(b) circle-fit residual of GNSS (no odometry used; label = fitted radius)")
panel_c(fig.add_subplot(gs[1, :2])); panel_d(fig.add_subplot(gs[1, 2:]))
a = fig.add_subplot(gs[2, :2]); t = r0["t"]
a.plot(t, err, color=C["ref"], lw=.8, label="|GNSS - odometry| after rigid fit")
a.scatter(t, st / 2 * err.max() * 1.05, c=np.where(st == 2, C["rtk"], C["gnss"]), s=6, label="fix status (top: RTK=2, bottom: plain=0)")
for c_ in drop["status_changes_at_s"]: a.axvline(c_, color="k", ls=":", lw=.8)
a.set(title=f"(e) jan12_mixed: RTK status over time (RMS err RTK {drop['rms_err_rtk']:.2f} m, plain {drop['rms_err_plain']:.2f} m)", xlabel="t [s]", ylabel="error [m]"); a.legend(fontsize=7)
a = fig.add_subplot(gs[2, 2:]); o, g = load("dec04_static")
a.plot(e - e.mean(), n - n.mean(), "o", ms=3, color=C["rtk"]); a.axis("equal")
a.set(title=f"(f) Dec04 static run, RTK: std E {floor['std_e']*100:.1f} cm, N {floor['std_n']*100:.1f} cm (n={floor['n']})", xlabel="east [m]", ylabel="north [m]")
fig.suptitle("Plain vs RTK GNSS, circle runs; REPLAY (reference = odometry baseline, not ground truth)", fontsize=11)
fig.tight_layout(); tag(fig, "REPLAY"); fig.savefig(OUT / "fig_rtk_full.png"); plt.close(fig)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4)); panel_c(ax[0]); panel_d(ax[1]); fig.suptitle("Plain vs RTK GNSS, circle runs; REPLAY (reference = odometry baseline, not ground truth)", fontsize=11)
fig.tight_layout(); tag(fig, "REPLAY"); fig.savefig(OUT / "fig_rtk.png"); plt.close(fig)
out = dict(per_run={k: {m: v for m, v in d.items() if m not in ("pa", "odom", "en")} for k, d in res.items()}, summary=summ, static_floor=floor, dropout=drop,
           paper=dict(plain_rmse=6.7, rtk_rmse=1.7))
json.dump(out, open(OUT / "rtk.json", "w"), indent=1)
print(json.dumps(dict(summary=summ, floor=floor, drop=drop), indent=1))
for k, d in out["per_run"].items(): print(k, d["group"], d["status"], *(f"{d[m]:.2f}" for m in ("rmse_paper_style", "rmse_gated", "rmse_rigid", "circ_r_gnss", "circ_rms_gnss", "circ_r_odom")))
