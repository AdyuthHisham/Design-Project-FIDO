"""Run registry + loader. Labels come from the correspondence records in the archive (rec) or are inferred from odometry extent (inf)."""
import glob
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out"; OUT.mkdir(exist_ok=True)

# key -> (file stem, label, label source)
RUNS = {
    "dec04_static":   ("Dec04_Test__2025_12_04-18_29_15", "static, RTK", "rec"),
    "dec04_line10":   ("Dec04_Test__2025_12_04-18_36_09", "10 m line, RTK", "rec"),
    "dec04_L":        ("Dec04_Test__2025_12_04-18_37_53", "10 m + 90deg turn + 5 m, RTK", "rec"),
    "nov25_line15":   ("Nov25_Test__2025_11_25-18_20_45", "15 m line, plain", "rec"),
    "nov25_circ5":    ("Nov25_Test__2025_11_25-18_24_31", "circle r=5 m, plain", "rec"),
    "nov25_circ7":    ("Nov25_Test__2025_11_25-18_29_08", "circle r=7 m, plain", "rec"),
    "nov24_circ_a":   ("Nov24_Test__2025_11_25-02_21_09", "circle ~5 m, RTK", "inf"),
    "nov24_circ_b":   ("Nov24_Test__2025_11_25-02_50_49", "circle ~5 m, RTK", "inf"),
    "nov18_circ_a":   ("Nov18_Test__2025_11_18-18_14_55", "circle ~5 m, plain (cov unset)", "inf"),
    "nov18_circ_b":   ("Nov18_Test__2025_11_18-18_41_46", "circle ~5 m, plain (cov unset)", "inf"),
    "jan12_mixed":    ("0112Test__2026_01_12-21_46_33", "~circle, RTK drops mid-run", "inf"),
    "jan12_plain":    ("0112Test__2026_01_12-21_51_14", "~circle, plain", "inf"),
    "dec16_short":    ("root__2025_12_16-22_28_31", "short run, RTK", "inf"),
}
GPS_RUNS = list(RUNS)


def load(key):
    d = np.load(ROOT / "data" / (RUNS[key][0] + ".npz"))
    return d["odom"], d["gps"]


def tsec(a, t0):
    return (a - t0) / 1e9
