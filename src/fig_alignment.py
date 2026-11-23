"""WP1 / paper Fig 4: GNSS->odometry frame alignment, yaw-correction events, R^2/buffer sweep. REPLAY of real bags + RE-IMPLEMENTED heading regression."""
import sys, json; sys.path.insert(0, __file__.rsplit("/", 1)[0])
import numpy as np
from fido.data import *; from fido.heading import *; from fido.plot import *
from fido.geo import wrap

KEY = "nov24_circ_a"
o, g = load(KEY)
raw = align_gnss(o, g)                      # original rule: first valid window locks the offset
gate = align_gnss(o, g, still_below=0.05)   # extension: median-speed stationary gate
oxy = raw["odom"]
fig, ax = plt.subplots(2, 2, figsize=(11, 9))
a = ax[0, 0]
a.plot(*(oxy - oxy[0]).T, color=C["odom"], lw=2, label="odometry (baseline)")
a.plot(*raw["en"].T, "o-", ms=3, lw=.7, color=C["gnss"], label="GNSS, east/north about first fix")
a.set(title="(a) Before alignment: different frames", xlabel="x / east [m]", ylabel="y / north [m]"); a.axis("equal"); a.legend(fontsize=7)
a = ax[0, 1]
a.plot(*oxy.T, color=C["odom"], lw=2, label="odometry")
a.plot(*raw["xy"].T, "o-", ms=3, lw=.7, color=C["bad"], label=f"aligned, original rule (offset {np.degrees(raw['bias_main']):.0f}°)")
a.plot(*gate["xy"].T, "o-", ms=3, lw=.7, color=C["rtk"], label=f"aligned, stationary gate (offset {np.degrees(gate['bias_main']):.0f}°)")
rm = lambda r: np.sqrt(np.mean(np.sum((r["xy"] - r["odom"]) ** 2, 1)))
a.set(title=f"(b) After alignment: RMSE vs odometry {rm(raw):.2f} m (orig) / {rm(gate):.2f} m (gate)", xlabel="x [m]", ylabel="y [m]"); a.axis("equal"); a.legend(fontsize=7)
a = ax[1, 0]
t = raw["t"]
a.plot(t, np.degrees(wrap(raw["oyaw"])), color=C["odom"], lw=1.5, label="odometry yaw")
for r, c, nm in ((raw, C["bad"], "orig"), (gate, C["rtk"], "gate")):
    if r["bias_main"] is None: continue
    h = np.degrees(wrap(r["est"] + r["bias_main"]))
    a.plot(t, h, ".", ms=3, color=c, alpha=.35)
    v = r["valid"]
    a.plot(t[v], h[v], "o", ms=5, mfc="none", color=c, label=f"valid GNSS-heading windows ({nm}): {v.sum()}")
s = np.hypot(o[:, 4], o[:, 5]); ts = (o[:, 0] - o[0, 0]) / 1e9
mv = np.interp(t, ts, np.convolve(s, np.ones(300) / 300, "same")) > .05
a.fill_between(t, -180, 180, where=~mv, color="#e5e7eb", zorder=0, label="robot standing (median speed < 0.05 m/s)")
a.set(title="(c) Yaw: odometry vs regression heading + offset; circles = R² gate passed", xlabel="t [s]", ylabel="yaw [deg]", ylim=(-185, 185)); a.legend(fontsize=7, loc="lower right")
a = ax[1, 1]
ks = [k for k in GPS_RUNS if k not in ("dec04_static", "dec16_short")]
x = np.arange(len(ks)); w = .27; ref = []
rows = {}
for k in ks:
    oo, gg = load(k); r0 = align_gnss(oo, gg); r1 = align_gnss(oo, gg, still_below=0.05)
    th, res, _ = rigid_fit(r0["en"], r0["odom"])
    rows[k] = dict(orig=float(np.degrees(r0["bias_main"])), gate=float(np.degrees(r1["bias_main"])), rigid=float(np.degrees(th)),
                   rmse_orig=float(rm(r0)), rmse_gate=float(rm(r1)), rmse_rigid=res)
for i, (nm, c) in enumerate((("orig", C["bad"]), ("gate", C["rtk"]), ("rigid", C["ref"]))):
    a.bar(x + (i - 1) * w, [abs(wrap(np.radians(rows[k][nm] - rows[k]["rigid"]))) * 180 / np.pi for k in ks], w, color=c,
          label={"orig": "original rule", "gate": "stationary gate", "rigid": "(rigid fit = reference)"}[nm]) if nm != "rigid" else None
a.set_xticks(x); a.set_xticklabels([k.replace("_", "\n", 1) for k in ks], fontsize=6.5); a.legend(fontsize=7)
a.set(title="(d) Offset error vs least-squares rotation of GNSS onto odometry", ylabel="|offset - reference| [deg]")
fig.suptitle(f"GNSS-to-odometry alignment, {KEY} ({RUNS[KEY][1]}); REPLAY + re-implemented heading regression", fontsize=11)
fig.tight_layout(); tag(fig, "REPLAY + RE-IMPLEMENTATION", "odometry is a baseline, not ground truth"); fig.savefig(OUT / "fig_alignment.png"); plt.close(fig)

# ---- sweep: R^2 threshold x buffer size, mean alignment RMSE over moving runs
thr = [0.80, 0.87, 0.93, 0.97, 0.99]; buf = [3, 4, 6, 8, 10]
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
sweep = {}
for j, (nm, kw) in enumerate((("original rule", {}), ("stationary gate (ext.)", dict(still_below=.05)))):
    M = np.full((len(buf), len(thr)), np.nan)
    for bi, b in enumerate(buf):
        for ti, tr in enumerate(thr):
            v = []
            for k in ks:
                oo, gg = load(k); r = align_gnss(oo, gg, buffer_size=b, r2_threshold=tr, **kw)
                if r["bias_main"] is not None: v.append(rm(r))
            if len(v) == len(ks): M[bi, ti] = np.mean(v)
    sweep[nm] = M.tolist()
    a = ax[j]; im = a.imshow(M, origin="lower", cmap="viridis_r", vmin=1, vmax=7)
    a.set_xticks(range(len(thr))); a.set_xticklabels(thr); a.set_yticks(range(len(buf))); a.set_yticklabels(buf)
    for (bi, ti), val in np.ndenumerate(M): a.text(ti, bi, "-" if np.isnan(val) else f"{val:.1f}", ha="center", va="center", color="w", fontsize=8)
    a.set(title=f"{nm}: mean RMSE [m] over {len(ks)} runs", xlabel="R² threshold", ylabel="buffer size [fixes]"); a.grid(False)
    if j == 0: plt.colorbar(im, ax=a, fraction=.046)
a = ax[2]
for k in ks[:0]: pass
nm = "nov24_circ_a"; oo, gg = load(nm); x = []
for b in buf:
    x.append([np.degrees(align_gnss(oo, gg, buffer_size=b, r2_threshold=t_, still_below=.05)["bias_main"] or np.nan) for t_ in thr])
for b, row in zip(buf, x): a.plot(thr, row, "o-", ms=3, label=f"buffer {b}")
a.axhline(rows[nm]["rigid"], color="k", ls="--", lw=1, label="rigid fit"); a.set(title=f"offset vs threshold, {nm} (gated)", xlabel="R² threshold", ylabel="offset [deg]"); a.legend(fontsize=7)
fig.suptitle("Heading-regression sweep (empty cell = window never passed the gate); REPLAY", fontsize=11)
fig.tight_layout(); tag(fig, "REPLAY + RE-IMPLEMENTATION"); fig.savefig(OUT / "fig_alignment_sweep.png"); plt.close(fig)
json.dump(dict(per_run=rows, sweep=dict(thr=thr, buf=buf, mean_rmse=sweep)), open(OUT / "alignment.json", "w"), indent=1)
print(json.dumps({k: {m: round(v, 2) for m, v in d.items()} for k, d in rows.items()}, indent=0))
