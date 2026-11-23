"""Run: .venv/bin/python -I tests/test_core.py   (asserts only, no framework)"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from fido.geo import wrap, enu_aeqd, enu_haversine
from fido.ekf import CTEKF
from fido.control import lqr_gain, DT


def test_haversine_vs_pyproj():
    lat0, lon0 = 57.689, 11.975          # Gothenburg
    rng = np.random.default_rng(0)
    dla, dlo = rng.uniform(-3e-4, 3e-4, 50), rng.uniform(-3e-4, 3e-4, 50)   # ~ +-30 m
    a = np.array(enu_aeqd(lat0 + dla, lon0 + dlo, lat0, lon0)); b = np.array(enu_haversine(lat0 + dla, lon0 + dlo, lat0, lon0))
    assert np.abs(a - b).max() < 0.10, np.abs(a - b).max()   # sphere vs ellipsoid is a ~0.2% scale effect: ~6 cm at 30 m


def test_wrap():
    a = np.array([-4 * np.pi, -np.pi - .1, np.pi + .1, 3 * np.pi + .2, 0.3])
    w = wrap(a)
    assert np.all(w >= -np.pi) and np.all(w < np.pi)
    assert np.allclose(np.cos(w), np.cos(a)) and np.allclose(np.sin(w), np.sin(a))


def test_ekf_reduces_to_odometry_when_gnss_noise_huge():
    # straight walk east at 0.5 m/s; GNSS says something wild but R is huge -> estimate must follow the odometry-driven prediction
    f = CTEKF(r_gps=np.eye(3) * 1e12, buffer_size=3, r2=0.5)
    lat0, lon0 = 57.689, 11.975; rng = np.random.default_rng(1)
    for k in range(3000):
        t = k * 0.02; om = np.array([0, 0.5 * t, 0, 0, 0.5, 0, 0.0])
        fix = (lat0 + rng.normal(0, 5e-5), lon0 + rng.normal(0, 5e-5)) if k % 50 == 0 else None
        f.step(t, om, fix)
    assert abs(f.x[0] - 0.5 * 59.98) < 0.5 and abs(f.x[1]) < 0.5, f.x


def test_dare_gain_stabilises():
    K = lqr_gain(); A_cl = np.eye(3) - DT * np.eye(3) @ K
    assert np.max(np.abs(np.linalg.eigvals(A_cl))) < 1


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
