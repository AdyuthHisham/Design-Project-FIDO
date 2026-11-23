"""Geodetic helpers. Local metres from lat/lon via pyproj AEQD (what the recovered code used) or Haversine (what the paper says)."""
from functools import lru_cache
import numpy as np
from pyproj import Proj, Transformer

R_EARTH = 6371008.8  # mean earth radius, m


def wrap(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


@lru_cache(maxsize=64)
def _aeqd(lat0, lon0):
    return Transformer.from_proj("epsg:4326", Proj(proj="aeqd", lat_0=lat0, lon_0=lon0, datum="WGS84"), always_xy=True)


def enu_aeqd(lat, lon, lat0, lon0):
    """East/north metres about (lat0, lon0), azimuthal equidistant on WGS84 (as headingCorrector.py)."""
    t = _aeqd(float(lat0), float(lon0))
    e, n = t.transform(np.asarray(lon), np.asarray(lat))
    return np.asarray(e), np.asarray(n)


def enu_haversine(lat, lon, lat0, lon0):
    """East/north metres about (lat0, lon0): Haversine distance + initial bearing (paper III-A2)."""
    p1, p2 = np.radians(lat0), np.radians(np.asarray(lat))
    dl = np.radians(np.asarray(lon) - lon0)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    d = 2 * R_EARTH * np.arcsin(np.sqrt(h))
    brg = np.arctan2(np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))
    return d * np.sin(brg), d * np.cos(brg)
