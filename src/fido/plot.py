import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10})
C = dict(odom="#6b7280", gnss="#d97706", rtk="#2563eb", ekf="#16a34a", bad="#dc2626", ref="#111827")


def tag(fig, kind, note=""):
    """Provenance label in the corner: REPLAY / RE-IMPLEMENTATION / SIMULATION."""
    fig.text(0.995, 0.005, f"{kind}{' - ' + note if note else ''}", ha="right", va="bottom", fontsize=7.5, color="#444",
             bbox=dict(fc="#f3f4f6", ec="#9ca3af", boxstyle="round,pad=0.25"))
