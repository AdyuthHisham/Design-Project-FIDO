"""Closed-loop LQR simulation. SIMULATION, not data: plant, reference and the odometry scale error are modelled; GNSS noise magnitudes are
resampled from real bag residuals. Estimators: truth | odom | gnss (fix + odometry increments) | gnss_zoh (held fix) | ekf (recovered CT-EKF) | ekf_ext (EKF + odometry v, omega)."""
import numpy as np
from pyproj import Proj, Transformer
from .control import lqr_gain, control, DT
from .data import load
from .heading import circle_fit
from .ekf import CTEKF
from .geo import wrap

LAT0, LON0 = 57.689, 11.975              # Gothenburg; only used to build lat/lon for the EKF (its input format)
BIAS_TRUE = np.radians(-127.0)          # hidden GNSS->odometry yaw offset (typical value measured in the bags, alignment.json)
RTK_RUNS = ["nov24_circ_a", "nov24_circ_b"]; PLAIN_RUNS = ["nov25_circ5", "nov25_circ7", "nov18_circ_a", "nov18_circ_b", "jan12_plain"]
_tf = Transformer.from_proj(Proj(proj="aeqd", lat_0=LAT0, lon_0=LON0, datum="WGS84"), "epsg:4326", always_xy=True)


def noise_pools():
    """Signed radial residuals (m) of real GNSS fixes about their own circle fit, for RTK and plain runs, plus odometry scale s =
    (RTK circle radius) / (odometry circle radius). ASSUMPTION: the RTK radius is closer to truth than odometry (no ground truth exists)."""
    from .geo import enu_aeqd
    pools, ratios = {}, []
    for name, ks in (("rtk", RTK_RUNS), ("plain", PLAIN_RUNS)):
        res = []
        for k in ks:
            o, g = load(k); e, n = enu_aeqd(g[:, 1], g[:, 2], g[0, 1], g[0, 2])
            ox = np.interp(g[:, 0], o[:, 0], o[:, 1]); oy = np.interp(g[:, 0], o[:, 0], o[:, 2])
            mv = np.hypot(np.gradient(ox), np.gradient(oy)) > .2
            cx, cy, r, _ = circle_fit(np.c_[e, n][mv]); res.append(np.hypot(e[mv] - cx, n[mv] - cy) - r)
            if name == "rtk": ratios.append(r / circle_fit(np.c_[ox, oy][mv])[2])
        pools[name] = np.concatenate(res)
    return pools, float(np.mean(ratios))


def simulate(ref, mode, plant="holonomic", gnss="rtk", seed=0, pools=None, scale=1.2, dt=DT, t_stand=0.0):
    """-> dict(t, true[N,3], est[N,3], u[N,3]). Robot frame == reference frame; truth starts at ref[0]."""
    rng = np.random.default_rng(seed); K = lqr_gain(dt); N = len(ref)
    pools = pools or noise_pools()[0]
    x = ref[0].astype(float).copy(); ekf = CTEKF(odom_dyn=(mode == "ekf_ext"), still_below=.05 if mode == "ekf_ext" else None) if mode.startswith("ekf") else None
    pool = pools[gnss]; start = rng.integers(len(pool)); gnss_hold = x[:2].copy(); gnss0 = None
    TRUE, EST, U = np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3))
    u = np.zeros(3); c0, s0 = np.cos(BIAS_TRUE), np.sin(BIAS_TRUE)
    for k in range(N):
        t = k * dt
        odom = np.array([x[0] / scale, x[1] / scale, x[2]])      # odometry under-reports travel by `scale` (modelled)
        if k % int(round(1 / dt)) == 0:                          # 1 Hz GNSS: truth + resampled magnitude, random direction
            mag = pool[(start + k // 50) % len(pool)]; ang = rng.uniform(0, 2 * np.pi)
            gnss_hold = x[:2] + mag * np.array([np.cos(ang), np.sin(ang)]); new_fix = True
        else:
            new_fix = False
        if mode == "truth": est = x
        elif mode == "odom": est = odom
        elif mode == "gnss_zoh": est = np.array([gnss_hold[0], gnss_hold[1], x[2]])   # held 1 Hz fix, yaw from odometry, frame known (oracle alignment)
        elif mode == "gnss":                                                          # fix + odometry displacement since the fix
            if new_fix: odom_at_fix = odom[:2].copy()
            est = np.array([*(gnss_hold + odom[:2] - odom_at_fix), x[2]])
        else:
            vb = (u[0] * np.cos(x[2]) + u[1] * np.sin(x[2])) / scale     # body-frame forward speed as odometry reports it
            om = np.array([0, odom[0], odom[1], odom[2], vb, 0.0, u[2]])
            fix = None
            if new_fix:   # robot->ENU: ENU = R(-bias) p
                en = np.array([c0 * gnss_hold[0] + s0 * gnss_hold[1], -s0 * gnss_hold[0] + c0 * gnss_hold[1]])
                lon, lat = _tf.transform(en[0], en[1]); fix = (lat, lon)
            ekf.step(t, om, fix)
            est = ekf.x[[0, 1, 3]] if ekf.x is not None else odom
        u = control(K, est, ref[k])
        TRUE[k], EST[k], U[k] = x, est, u
        if plant == "holonomic":
            x = x + dt * u
        else:                                                    # unicycle: only the forward component and yaw rate are realisable
            v = u[0] * np.cos(x[2]) + u[1] * np.sin(x[2]); x = x + dt * np.array([v * np.cos(x[2]), v * np.sin(x[2]), u[2]])
        x[2] = wrap(x[2]) if abs(x[2]) > 10 else x[2]
    return dict(t=np.arange(N) * dt, true=TRUE, est=EST, u=U)


def metrics(ref, run):
    """cross-track error (distance of true position to the reference polyline), heading error vs time-indexed reference, control effort."""
    P = ref[:, :2]; T = run["true"]
    d = np.sqrt(((T[:, None, :2] - P[None, ::5, :]) ** 2).sum(2)).min(1)
    he = np.abs(wrap(T[:, 2] - ref[:, 2]))
    return dict(xtrack_rms=float(np.sqrt(np.mean(d ** 2))), xtrack_max=float(d.max()), heading_rms_deg=float(np.degrees(np.sqrt(np.mean(he ** 2)))),
                effort_rms=float(np.sqrt(np.mean(np.sum(run["u"][:, :2] ** 2, 1)))), effort_omega_rms=float(np.sqrt(np.mean(run["u"][:, 2] ** 2))))
