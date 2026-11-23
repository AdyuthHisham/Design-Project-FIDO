"""Port of recovered/code/coordTurnClass.py CT_EKF_Filter (5-state coordinated-turn EKF) for offline replay.
State [px, py, v, psi, omega]. Faithful where it matters (it is the code that ran), fixed where it is broken:
see docs/deviations.md. Everything runs on real seconds."""
import numpy as np
from .geo import wrap
from .heading import HeadingCorrector

# On-robot parameter set (ekf_Controller.py:36-48)
R_GPS = 10 * np.diag([0.5, 0.5, 1.903270742219267])
R_ODOM = 40 * np.diag([9.0, 9.0, 0.25])
Q = dict(pos=25.0, vel=25.0, psi=(10 * np.pi / 180) ** 2, omega=0.20 ** 2)


# Configuration that reproduces the behaviour described for the report's Fig. 5 (EKF follows the GNSS trend but smooths it with odometry):
# odometry cost raised after each GNSS fix and decayed slowly, odometry v and omega in the prediction, stationary gate on the heading window.
# ALL EXTENSIONS (docs/deviations.md). The decay rate is my choice from src/test_odom_schedule.py, not a recovered value.
PAPER_EKF = dict(odom_mode="R", odom_rec=0.000025, odom_dyn=True, still_below=0.05)


class CTEKF:
    def __init__(self, r_gps=R_GPS, r_odom=R_ODOM, q=Q, buffer_size=6, r2=0.93, geo="aeqd",
                 odom_dyn=False, p0=10.0, flush_below=None, still_below=None, odom_mode="gain", odom_rec=0.0025):
        """odom_dyn=True is an EXTENSION: v and omega are re-set from odometry twist every step
        (the original never corrects them). flush_below: stationary pop threshold (m/s) or None (original rule). still_below: EXTENSION, clear regression window when 1 s median speed < this."""
        self.r_gps, self.r_odom = np.asarray(r_gps, float), np.asarray(r_odom, float)
        self.Qb = np.diag([q["pos"], q["pos"], q["vel"], q["psi"], q["omega"]])
        self.hc = HeadingCorrector(buffer_size, r2, geo)
        self.odom_rec = odom_rec     # per-pass (20 ms) recovery of odom_red; original 0.0025
        self.odom_mode = odom_mode   # "gain": original (scale the update gain by odom_red). "R": scale the odometry cost R_odom by 1/odom_red instead (the schedule as described by the owner)
        self.odom_dyn, self.flush_below, self.still_below = odom_dyn, flush_below, still_below
        self.sp = []   # (t, speed) last 1 s, for the median stationary gate
        self.x, self.P = None, np.eye(5) * p0
        self.t_old, self.odom_red, self.offset = None, 1.0, None
        self.H = np.zeros((3, 5)); self.H[0, 0] = self.H[1, 1] = self.H[2, 3] = 1.0
        self.Hc = np.vstack([self.H, self.H])
        self.gps_log = []   # (t, e_r, n_r, yaw, innov x,y,psi)
        self.odom_inn = []

    def _predict(self, dt, om):
        px, py, v, psi, w = self.x
        if self.odom_dyn:
            v, w = om[4], om[6]            # body-frame forward speed, yaw rate from odometry
        F = np.eye(5)
        if abs(w) < 0.02:
            px += v * np.cos(psi) * dt; py += v * np.sin(psi) * dt; psi += w * dt
            F[0, 3] = -v * np.sin(psi) * dt; F[1, 3] = v * np.cos(psi) * dt
            F[0, 2] = np.cos(psi) * dt; F[1, 2] = np.sin(psi) * dt
            F[0, 4] = -0.5 * v * np.sin(psi) * dt ** 2; F[1, 4] = 0.5 * v * np.cos(psi) * dt ** 2
        else:
            th = w * dt
            px += v / w * (np.sin(psi + th) - np.sin(psi)); py += v / w * (np.cos(psi) - np.cos(psi + th)); psi += th
            F[0, 2] = (np.sin(psi) - np.sin(psi - th)) / w; F[1, 2] = (np.cos(psi - th) - np.cos(psi)) / w
            F[0, 3] = v / w * (np.cos(psi) - np.cos(psi - th)); F[1, 3] = v / w * (np.sin(psi) - np.sin(psi - th))
            F[0, 4] = v * (th * np.cos(psi) - (np.sin(psi) - np.sin(psi - th))) / w ** 2
            F[1, 4] = v * (th * np.sin(psi) + (np.cos(psi) - np.cos(psi - th))) / w ** 2
        self.x = np.array([px, py, v, wrap(psi), w])
        self.P = F @ self.P @ F.T + self.Qb * dt

    def _upd(self, H, z, R, gain=1.0):
        y = z - H @ self.x
        y[2::3] = wrap(y[2::3])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + gain * (K @ y)
        self.x[3] = wrap(self.x[3])
        self.P = (np.eye(5) - K @ H) @ self.P
        return y, S

    def step(self, t, om, fix=None, r_gps=None):
        """om = odom row [t_ns, x, y, yaw, vx, vy, wz] (vx,vy body frame); fix = (lat, lon) or None. t in seconds."""
        if self.x is None:
            self.x = np.array([om[1], om[2], om[4], om[3], om[6]]); self.t_old = t; return
        dt = t - self.t_old; self.t_old = t
        if dt <= 0:
            return
        self._predict(dt, om)
        if self.flush_below and abs(om[4]) < self.flush_below and abs(om[5]) < self.flush_below:
            self.hc.flush()
        self.sp = [a for a in self.sp if t - a[0] < 1.0] + [(t, np.hypot(om[4], om[5]))]
        if fix is not None:
            if self.still_below and np.median([a[1] for a in self.sp]) < self.still_below:
                self.hc.buf.clear()
            if self.offset is None:
                self.offset = (om[1], om[2])
            yaw_c, e, n, valid, _ = self.hc.update(fix[0], fix[1], self.x[3])
            if self.hc.bias_main is None:
                return
            c, s = np.cos(self.hc.bias_main), np.sin(self.hc.bias_main)
            er, nr = c * e - s * n + self.offset[0], s * e + c * n + self.offset[1]   # rotate by +bias_main (see deviations)
            z = np.array([er, nr, yaw_c, om[1], om[2], om[3]])
            Rg = self.r_gps if r_gps is None else r_gps
            R = np.block([[Rg, np.zeros((3, 3))], [np.zeros((3, 3)), self.r_odom]])
            y, S = self._upd(self.Hc, z, R)
            self.gps_log.append((t, er, nr, yaw_c, *y[:3], *np.diag(S)[:3]))
            self.odom_red = 0.0002
        else:
            if self.odom_mode == "R":
                y, S = self._upd(self.H, np.array([om[1], om[2], om[3]]), self.r_odom / self.odom_red)
            else:
                y, S = self._upd(self.H, np.array([om[1], om[2], om[3]]), self.r_odom, self.odom_red)
            self.odom_inn.append((t, *y))
            self.odom_red = min(self.odom_red + self.odom_rec * dt / 0.02, 1.0)   # rate-independent recovery (robot ran at 50 Hz)


def replay(odom, gps, status_R=None, hz=None, **kw):
    """Run CTEKF over a bag. odom/gps arrays as in data/*.npz. hz: subsample odometry (None = native).
    status_R: (R_rtk, R_plain) to switch GNSS covariance on fix status (EXTENSION, not in the original).
    -> dict(t, x[N,5], P[N,5,5], gps_log[M,10])"""
    f = CTEKF(**kw)
    t0 = odom[0, 0]
    idx = np.arange(len(odom))
    if hz:
        tt = (odom[:, 0] - t0) / 1e9
        idx = np.unique(np.searchsorted(tt, np.arange(0, tt[-1], 1 / hz)))
    gi, T, X, P = 0, [], [], []
    for i in idx:
        t = (odom[i, 0] - t0) / 1e9
        fix, rg = None, None
        while gi < len(gps) and gps[gi, 0] <= odom[i, 0]:   # latest fix not later than this sample
            fix = (gps[gi, 1], gps[gi, 2])
            if status_R:
                rg = status_R[0] if gps[gi, 4] == 2 else status_R[1]
            gi += 1
        f.step(t, odom[i], fix, rg)
        if f.x is not None:
            T.append(t); X.append(f.x.copy()); P.append(f.P.copy())
    return dict(t=np.array(T), x=np.array(X), P=np.array(P), gps=np.array(f.gps_log), odom_inn=np.array(f.odom_inn), filt=f)
