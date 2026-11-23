"""Pipeline schematic (diagram, no data)."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from matplotlib.patches import FancyBboxPatch
from fido.plot import *; from fido.data import OUT

fig, a = plt.subplots(figsize=(13, 4.6)); a.set_xlim(0, 13); a.set_ylim(-.8, 4.6); a.axis("off"); a.grid(False)
def box(x, y, w, h, t, fc, sub=""):
    a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12", fc=fc, ec="#374151", lw=1.2))
    a.text(x + w / 2, y + h / 2 + (.2 if sub else 0), t, ha="center", va="center", fontsize=10, weight="bold")
    if sub: a.text(x + w / 2, y + h / 2 - .35, sub, ha="center", va="center", fontsize=7.5, color="#374151")
def arrow(x0, y0, x1, y1, t="", ls="-", dx=0, dy=.15):
    a.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#111827", ls=ls))
    if t: a.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, t, ha="center", fontsize=7.5, color="#374151")
box(.2, 3.0, 2.2, 1.2, "RTK-GNSS", "#dbeafe", "lat/lon, ~1 Hz\nstatus 2 = RTK fix")
box(3.0, 3.0, 2.6, 1.2, "Frame alignment", "#dbeafe", "metres about 1st fix,\nregression heading (buf 6, R²>0.93)\nbias_main, rotate + shift")
box(.2, 1.0, 2.2, 1.2, "Odometry + IMU", "#e5e7eb", "x, y, yaw, ~300 Hz\n(own frame)")
box(6.2, 1.8, 2.6, 1.6, "Coordinated-turn EKF", "#dcfce7", "x = [px, py, v, ψ, ω]\npredict: odometry rate\nupdate: pose, GNSS async 1 Hz")
box(9.4, 1.8, 2.0, 1.6, "LQR", "#fef3c7", "x = [px, py, ψ]\nu = −K(x̂ − x_ref)\nK = diag(3.06, 3.06, 7.17)")
box(9.4, .1, 2.0, .9, "Waypoint reference", "#f3f4f6", "0.3 m/s, 50 Hz")
box(11.8, 1.8, 1.1, 1.6, "Go2", "#fee2e2", "Move(vx, vy, ω)")
arrow(2.4, 3.6, 3.0, 3.6); arrow(5.65, 3.5, 6.2, 3.1); a.text(5.8, 3.75, "aligned GNSS (x, y, ψ)", fontsize=7.5, color="#374151", ha="left")
arrow(2.4, 1.6, 6.2, 2.2, "pose, yaw (prediction + update)")
arrow(8.8, 2.6, 9.4, 2.6, "x̂"); arrow(10.4, 1.0, 10.4, 1.8, "x_ref", dx=.4, dy=-.1); arrow(11.4, 2.6, 11.8, 2.6)
a.plot([12.35, 12.35, 1.3], [1.8, -.25, -.25], color="#6b7280", ls="--", lw=1.1); arrow(1.3, -.25, 1.3, 1.0)
a.text(6.8, -.5, "robot motion feeds back into odometry (and the GNSS fixes)", fontsize=8, color="#6b7280", ha="center")
fig.suptitle("FIDO localization and control pipeline", fontsize=11); fig.tight_layout(); tag(fig, "DIAGRAM"); fig.savefig(OUT / "fig_pipeline.png")
