"""LQR trajectory controller as in recovered ekf_Controller.py + reference paths. u = -K (x - x_ref), x=[px,py,psi]."""
import numpy as np
from scipy.linalg import solve_discrete_are
from .geo import wrap

DT = 0.02
Q, Rm = np.diag([10.0, 10.0, 60.0]), np.eye(3)


def lqr_gain(dt=DT, Q=Q, R=Rm):
    A, B = np.eye(3), dt * np.eye(3)
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.inv(R + B.T @ P @ B) @ B.T @ P


def control(K, est, ref):
    e = np.asarray(est, float) - np.asarray(ref, float)
    e[2] = wrap(e[2])
    return -K @ e


def ref_path(name, v=0.3, w_turn=0.5, dt=DT):
    """Time-parameterised reference [N,3] (x, y, psi) at speed v (planner linear_v=0.3 in controller_manager_new.py).
    Turns happen in place at w_turn rad/s. Start at origin, heading +x."""
    if name == "circle":                    # r = 5 m, CCW, one lap, centre (0, 5)
        r = 5.0; n = int(2 * np.pi * r / v / dt); th = np.linspace(0, 2 * np.pi, n)
        return np.c_[r * np.sin(th), r * (1 - np.cos(th)), th]
    segs = {"line10": [("L", 10)], "line_turn": [("L", 10), ("T", -np.pi / 2), ("L", 5)],
            "L": [("L", 8), ("T", np.pi / 2), ("L", 8)]}[name]
    p = [(0.0, 0.0, 0.0)]; x, y, psi = p[0]
    for kind, val in segs:
        if kind == "L":
            for _ in range(int(abs(val) / v / dt)):
                x += v * dt * np.cos(psi); y += v * dt * np.sin(psi); p.append((x, y, psi))
        else:
            for _ in range(int(abs(val) / w_turn / dt)):
                psi += np.sign(val) * w_turn * dt; p.append((x, y, psi))
    return np.array(p)
