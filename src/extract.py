"""Dump every bag in _extract/ to data/<bag>.npz + data/index.csv (odom, imu, gps, heading, speed)."""
import sys, csv, numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
TS = get_typestore(Stores.ROS2_HUMBLE)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"; OUT.mkdir(exist_ok=True)

def yaw(q):  # quaternion -> yaw
    return np.arctan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))

rows = []
for meta in sorted((ROOT/"_extract").rglob("metadata.yaml")):
    d = meta.parent
    if not list(d.glob("*.db3")): continue
    name = (next(d.glob("*.db3")).stem[:-2] if d.name == "_extract" else d.name).replace("rosbag2_", "")
    # group folder (Circle/StraightLine/Nov25_Test..) for context
    grp = d.parent.name if d.name != "_extract" else "root"
    o, i, g, h, s = [], [], [], [], []
    with AnyReader([d], default_typestore=TS) as r:
        for c, t, raw in r.messages():
            m = r.deserialize(raw, c.msgtype); tp = c.topic
            if tp.endswith("/odometry"):
                p, v = m.pose.pose, m.twist.twist
                o.append((t, p.position.x, p.position.y, yaw(p.orientation), v.linear.x, v.linear.y, v.angular.z))
            elif tp.endswith("/imu_state"):
                a, w = m.linear_acceleration, m.angular_velocity
                i.append((t, a.x, a.y, a.z, w.x, w.y, w.z))
            elif tp.endswith("/fix"):
                g.append((t, m.latitude, m.longitude, m.altitude, m.status.status, m.position_covariance[0]))
            elif tp.endswith("/heading_deg"): h.append((t, m.data))
            elif tp.endswith("/speed"): s.append((t, m.x, m.y, m.z))
    f = lambda a, n: np.array(a, float).reshape(-1, n)
    np.savez_compressed(OUT/f"{grp}__{name}.npz", odom=f(o,7), imu=f(i,7), gps=f(g,6), heading=f(h,2), speed=f(s,4))
    st = np.array(g)[:,4] if g else np.array([])
    rows.append(dict(group=grp, bag=name, dur_s=round((max(x[0] for x in o)-min(x[0] for x in o))/1e9,1),
        odom=len(o), imu=len(i), gps=len(g), gps_rtk_status2=int((st==2).sum()), gps_status0=int((st==0).sum())))
    print(rows[-1], flush=True)
with open(OUT/"index.csv","w",newline="") as fh:
    w = csv.DictWriter(fh, rows[0].keys()); w.writeheader(); w.writerows(rows)
