"""Port of recovered/code/headingCorrector.py: GNSS-regression heading vs odometry yaw (sliding window + R^2 gate).
Fixes vs original: geo conversion is pluggable (AEQD or Haversine), no module-level growing lists, no prints,
heading is in radians throughout, consistent return shape."""
from collections import deque
import numpy as np
from .geo import wrap, enu_aeqd, enu_haversine


def regress_heading(e, n):
    """Heading (rad, ENU math convention, east=0 CCW) of a point window and the R^2 of the line fit.
    Regress the dominant axis (as the original), sign from end-to-start displacement."""
    de, dn = e[-1] - e[0], n[-1] - n[0]
    x, y = (e, n) if abs(de) >= abs(dn) else (n, e)
    slope, icpt = np.polyfit(x, y, 1)
    ss_res = np.sum((y - (slope * x + icpt)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-8 else 0.0
    if abs(de) >= abs(dn):
        pe, pn = de, slope * de
    else:
        pe, pn = slope * dn, dn
    return np.arctan2(pn, pe), r2


class HeadingCorrector:
    """Same logic as GPSHeadingBiasCorrector.update, with angles wrapped to [-pi, pi)."""

    def __init__(self, buffer_size=6, r2_threshold=0.93, geo="aeqd"):
        self.buf = deque(maxlen=buffer_size)
        self.r2_thr = r2_threshold
        self.geo = enu_aeqd if geo == "aeqd" else enu_haversine
        self.origin = None
        self.bias_main = None   # fixed GNSS->odometry yaw offset, set once by the first valid window
        self.bias = 0.0         # running residual
        self.events = []        # (index of fix, r2, estimated heading) for every valid window

    def update(self, lat, lon, measured_yaw):
        """-> corrected_yaw, east, north, valid (valid = regression window passed the R^2 gate)."""
        if self.origin is None:
            self.origin = (lat, lon)
        e, n = (float(v) for v in self.geo(lat, lon, *self.origin))
        self.buf.append((e, n))
        valid, est = False, None
        if len(self.buf) == self.buf.maxlen:
            pts = np.array(self.buf)
            est, r2 = regress_heading(pts[:, 0], pts[:, 1])
            valid = r2 > self.r2_thr
            if valid:
                if self.bias_main is None:
                    self.bias_main = float(wrap(measured_yaw - est))
                self.bias = float(wrap(measured_yaw - (est + self.bias_main)))
        return float(wrap(measured_yaw - self.bias)), e, n, valid, est

    def flush(self):
        """The stationary 'pop' of the original harness: drop the newest regression point."""
        if self.buf:
            self.buf.pop()


def align_gnss(odom, gps, buffer_size=6, r2_threshold=0.93, geo="aeqd", flush_below=None, still_below=None):
    """Paper III-A2 alignment, offline. Origin subtraction + rotation by bias_main from the first R^2-valid window.
    flush_below: m/s; odometry rows slower than this pop the newest regression point (original stationary flush).
    still_below: m/s; EXTENSION: clear the window if the median odometry speed since the previous fix is below this.
    -> dict(t[s], en[N,2] (GNSS metres about first fix), xy[N,2] aligned GNSS in odom frame, odom[N,2],
            bias_main, valid[N] bool, est[N] window heading (nan if window not full), r2[N], k0 first valid idx)"""
    t0 = odom[0, 0]
    hc = HeadingCorrector(buffer_size, r2_threshold, geo)
    N = len(gps)
    spd = np.hypot(odom[:, 4], odom[:, 5])
    ox = np.interp(gps[:, 0], odom[:, 0], odom[:, 1]); oy = np.interp(gps[:, 0], odom[:, 0], odom[:, 2])
    oyaw = np.interp(gps[:, 0], odom[:, 0], np.unwrap(odom[:, 3]))
    en, valid, est, r2s = np.zeros((N, 2)), np.zeros(N, bool), np.full(N, np.nan), np.full(N, np.nan)
    prev, k0 = odom[0, 0], None
    for k in range(N):
        if flush_below:
            a, b = np.searchsorted(odom[:, 0], [prev, gps[k, 0]], side="right")
            for _ in range(int(np.sum(spd[a:b] < flush_below))):
                hc.flush()
        if still_below:
            a, b = np.searchsorted(odom[:, 0], [prev, gps[k, 0]], side="right")
            if b > a and np.median(spd[a:b]) < still_below:
                hc.buf.clear()
        prev = gps[k, 0]
        _, e, n, v, h = hc.update(gps[k, 1], gps[k, 2], oyaw[k])
        en[k] = e, n; valid[k] = v
        if len(hc.buf) == hc.buf.maxlen:
            pts = np.array(hc.buf); est[k], r2s[k] = regress_heading(pts[:, 0], pts[:, 1])
        if v and k0 is None:
            k0 = k
    b = hc.bias_main
    if b is None:
        xy = np.full((N, 2), np.nan)
    else:
        c, s = np.cos(b), np.sin(b)
        xy = np.c_[c * en[:, 0] - s * en[:, 1] + ox[0], s * en[:, 0] + c * en[:, 1] + oy[0]]
    return dict(t=(gps[:, 0] - t0) / 1e9, en=en, xy=xy, odom=np.c_[ox, oy], oyaw=oyaw, bias_main=b, valid=valid,
                est=est, r2=r2s, k0=k0)


def rigid_fit(p, q):
    """Least-squares rotation+translation taking p onto q (both [N,2]). -> theta, rms residual, aligned p."""
    pc, qc = p - p.mean(0), q - q.mean(0)
    H = pc.T @ qc
    th = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    c, s = np.cos(th), np.sin(th)
    pa = pc @ np.array([[c, s], [-s, c]]) + q.mean(0)
    return th, float(np.sqrt(np.mean(np.sum((pa - q) ** 2, 1)))), pa


def circle_fit(xy):
    """Algebraic (Kasa) circle fit -> cx, cy, r, rms radial residual."""
    A = np.c_[2 * xy[:, 0], 2 * xy[:, 1], np.ones(len(xy))]
    b = (xy ** 2).sum(1)
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    r = np.sqrt(c + cx ** 2 + cy ** 2)
    return cx, cy, r, float(np.sqrt(np.mean((np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r) ** 2)))
