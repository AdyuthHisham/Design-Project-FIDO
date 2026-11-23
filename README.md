# FIDO localization and control, regenerated from the recorded bags

## The project
FIDO (Friendly and Interactive robot DOg) was a Chalmers masters course project (SSY226): a Unitree Go2 quadruped that guides visitors around campus, talks to them through a language model and follows a planned route outdoors. The project was ongoing across course years, and this year's team (Xiaoyang Chen, David Mörck, Andreas Särnholm, Shijie Zhou and me) improved the existing localization and trajectory control, and implemented a new LiDAR SLAM and new dialogue/voice modules. This repo covers only the localization and trajectory-control part, rebuilt offline from the recorded ROS 2 bags.

The robot has three position sources, none good enough alone:
- **Leg odometry** (~300 Hz): smooth but drifts, and lives in its own frame that starts at an arbitrary pose.
- **IMU**: its quaternion gives the yaw the controller and filter use.
- **GNSS** (~1 Hz, optionally network-RTK): absolute but noisy (metres without RTK, centimetres with it) and in a different frame from odometry.

![Pipeline](out/fig_pipeline.png)
*The pipeline is: RTK-GNSS → align GNSS to the odometry frame → coordinated-turn EKF fusing both → LQR tracking a waypoint trajectory on the EKF estimate.*

The plots below (Fig. 3 to 5) are taken from the project report. The numbers in the text come from re-running the recovered code on the recorded bags (**replay**), a port of that code (**re-implementation**) or a synthetic closed loop (**simulation**), and are labelled as such. Odometry is a *relative baseline, not ground truth*; no surveyed reference exists, so nothing here is a filter accuracy claim.

## 1. RTK-GNSS
RTK (real-time kinematic) uses correction data from a reference network to turn a metre-level GNSS fix into a centimetre-level one. The receiver reports the fix type in the NavSatFix status: `2` = RTK-corrected, `0` = plain. Fixes arrive at ~1 Hz as latitude/longitude. Here RTK is a property of the input; the EKF is unchanged by it in the original (this repo adds an optional status-dependent R, labelled "ext.").

![Report Fig. 3](report_figs/fig3_rtk_disabled_vs_enabled.png)
*Report Fig. 3: GNSS trajectory tracking for a 5-meter radius circle, RTK disabled (a) vs enabled (b). Odometry is the local reference baseline (from the project report).*

| | paper (Sec. IV-A) | here, recovered alignment | here, + stationary gate (ext.) | circle-fit RMS (no odometry) |
|---|---|---|---|---|
| plain GNSS, 5 circle runs | 6.7 m | 5.7 m | 5.4 m | 1.29 m |
| RTK, 2 circle runs | 1.7 m | 6.8 m | **1.8 m** | **0.32 m** |

## 2. Frame alignment
GNSS gives lat/lon, odometry gives metres in an unaligned frame. Lat/lon are converted to east/north metres about the first fix (Haversine in the paper, pyproj azimuthal-equidistant in the recovered code; both implemented, difference < 0.1 m over ±30 m). A sliding window of the last **6** GNSS points is fitted with a line; if the fit has **R² > 0.93** its direction is a valid GNSS heading. The first valid window fixes a constant offset `bias_main = odom_yaw − gnss_heading`; later windows update a running residual `bias`. GNSS points are rotated by `bias_main` and shifted by the odometry position at the first fix, so both sources share one frame.

![Report Fig. 4](report_figs/fig4_global_frame_yaw_corrections.png)
*Report Fig. 4: robot movement tracked in the global frame and yaw corrections (from the project report).*

GNSS metres about the first fix are rotated into the odometry frame by the yaw offset from a sliding-window regression (buffer 6, R² gate 0.93). The recovered rule picks a wrong offset on most runs (up to ~80° off the least-squares rotation) because the window also fills while the robot stands still; clearing the window when stationary (extension) fixes most of that.

## 3. Coordinated-turn EKF
**State** `x = [p_x, p_y, v, ψ, ω]ᵀ`: planar position, forward speed, heading, turn rate (coordinated turn / CTRV, 2-D only).

**Prediction** (every odometry sample, real `dt`), exact arc integration with `θ = ω·dt`:
- `ω ≠ 0`: `p_x += (v/ω)(sin(ψ+θ) − sin ψ)`, `p_y += (v/ω)(cos ψ − cos(ψ+θ))`, `ψ += θ`
- `|ω| < 0.02 rad/s`: straight-line limit `p += v·dt·(cos ψ, sin ψ)`, `ψ += ω·dt`
- `v` and `ω` follow a random walk. The Jacobian `F` is the analytic 5×5 one of the above, `P ← F P Fᵀ + Q·dt`.

**Measurements** (both observe `[p_x, p_y, ψ]`, so `H = [I₃ | 0]` on `(p_x, p_y, ψ)`; innovations on ψ are wrapped to [−π, π]):
- *Odometry pose* at every sample, with the correction scaled by a factor `odom_reduction`. It is set to 0.0002 right after each GNSS fix and recovers by +0.0025 per 20 ms step up to 1, so odometry only slowly pulls the estimate back after a GNSS correction.
- *GNSS pose* at ~1 Hz, asynchronously: the aligned GNSS position plus the corrected yaw, stacked with the odometry pose into one 6-dimensional update (`H_c = [H; H]`).

Nothing measures `v` or `ω`, so the recovered filter never corrects them and its prediction barely moves the robot between fixes.

**The EKF used in my replays (all changes labelled "ext.")** is the recovered filter with three changes, so the output follows the GNSS trend but smooths it with the higher-rate odometry. None is recovered code: (1) after each GNSS fix the odometry *cost* is raised, `R_odom / odom_red`, with `odom_red` reset to 0.0002 and recovering by 0.000025 per 20 ms (100x slower than the recovered gain schedule), so GNSS is trusted after a fix and odometry only gradually; (2) `v` and `ω` are set from the odometry twist at every step; (3) the heading-regression window is cleared while the robot stands still. The decay rate is my choice from a small test (`src/test_odom_schedule.py`), not the on-robot value. The unmodified recovered filter stays in the ablation as "EKF (recovered, unchanged)".

**Parameters** (on-robot set, `ekf_Controller.py`):

| Quantity | Base | Factor | Value used |
|---|---|---|---|
| GNSS noise `R_gps` (x, y, ψ) | diag(0.5, 0.5, 1.903) | ×10 | diag(5, 5, 19.03) |
| Odometry pose noise `R_odom` (x, y, ψ) | diag(9, 9, 0.25) | ×40 | diag(360, 360, 10) |
| Process noise `Q` (p, v) | 25 | | 25 each (times dt) |
| Process noise `Q` (ψ) | (10°)² | | 0.0305 rad² |
| Process noise `Q` (ω) | 0.20² | | 0.04 |
| Initial covariance `P₀` | | | 10·I (my port; not recorded in the original) |
| Heading regression | buffer 6, R² 0.93 | | |
| Odometry cost schedule (ext.) | `R_odom / odom_red` | `odom_red`: 0.0002 after a fix, +0.000025 / 20 ms | not the on-robot value |

The odometry-rate noise `diag(0.001, 0.004)` is passed in but never used by the recovered update. The static-run RTK covariance measured here (σ ≈ 5 cm / 2 cm) is far below the hand-set 5 m² (σ 2.2 m), so the robot ran a very conservative GNSS weight; see "Noise calibration and sensitivity" below.

### Replay on the bags
![Report Fig. 5](report_figs/fig5_ekf_test_results.png)
*Report Fig. 5: EKF test results for straight, circular and L-shaped trajectories (from the project report).*

**What the plots show.** The EKF follows the general trend of the GNSS and smooths it with odometry. On the straight and circular runs it stays within 0.11 m and 0.16 m RMS of the RTK trajectory, and its 10 Hz jerk (0.12 m and 0.09 m) is a third or less of the GNSS track's (0.36 m and 0.40 m, at 1 Hz). The straight run's GNSS track has a roughly 2 m hook at one end that is in the data, not in the filter. On the L-shaped run the EKF cuts across the second leg and is 1.3 m RMS from the GNSS track, because the GNSS-to-odometry alignment is off by about 16° on that run, so odometry heading and GNSS direction disagree. Averaging the first several heading windows or weighting GNSS yaw more did not fix it, and the alignment is unchanged. Each run has its own odometry frame, so absolute axes differ between runs.

Odometry prediction runs at its native rate (~300 Hz), GNSS updates asynchronously at ~1 Hz, real seconds. Smoothness is measured on the EKF output at 10 Hz (a 1 Hz measure hides the sawtooth the recovered filter produces after each fix). Means over 12 GNSS bags (`out/ekf_summary.json`):

| | path length / odometry | jerk RMS at 10 Hz | RMS distance to odometry |
|---|---|---|---|
| odometry only | 1.0 | 0.007 m | 0 |
| EKF (recovered, unchanged) | 9.4 | 0.82 m | 3.7 m |
| **EKF (ext.)** | **1.9** | **0.21 m** | 3.4 m |

The recovered filter produces a 1 Hz sawtooth: each fix pulls the estimate to GNSS, then the odometry pose update pulls it back, because GNSS and odometry disagree by metres. Raising the odometry cost with the recovered decay rate made this worse; slowing the decay and feeding odometry velocity into the prediction is what smooths it. The EKF (ext.) sits farther from odometry because it trusts GNSS more. None of these metrics measure accuracy.

### Noise calibration and sensitivity
Static RTK covariance (std 5.2 cm / 2.1 cm) gives sigma ≈ 0.05 m, versus 0.71 m for the hand-set `R_gps` (0.5 m²) and 2.2 m after the ×10 inflation (5 m²) actually used on the robot. The hand-set values are conservative by more than an order of magnitude for RTK; plain-GNSS circle residuals (1.3 m RMS) sit closer to them. A 5×5 scaling sweep of R and Q is in the figure. Low jerk or low distance to odometry trivially favours trusting odometry, so the heatmaps show trade-offs, not an optimum.

## 4. LQR trajectory control
**State** `x = [p_x, p_y, ψ]ᵀ`, taken from the EKF estimate (or odometry when the EKF is off). **Control** `u = [u_x, u_y, ω]ᵀ`: world-frame velocities and a yaw rate (the code docstring says `[v, ω]`, but the gain and the command mapping are 3×3).

**Plant** (the one the code implies): discrete single integrator `x_{k+1} = A x_k + B u_k` with `A = I₃`, `B = dt·I₃`, `dt = 0.02 s` (50 Hz). It is holonomic, which suits the Go2 (it can strafe). The paper calls the controller LQG and the recovered class is plain LQR; here "LQG-style" means LQR on the EKF estimate.

**Cost** `Σ xᵀQx + uᵀRu` with `Q = diag(10, 10, 60)` (heading weighted most), `R = I₃`. The infinite-horizon gain comes from the discrete ARE `P = AᵀPA − AᵀPB(R + BᵀPB)⁻¹BᵀPA + Q`, `K = (R + BᵀPB)⁻¹BᵀPA`, which gives `K = diag(3.064, 3.064, 7.169)`. The closed loop `A − BK` has eigenvalues 0.939, 0.939, 0.857, all inside the unit circle.

**Law** `u = −K (x̂ − x_ref)`, with ψ error wrapped. The world-frame `(u_x, u_y)` is rotated into the body frame by the yaw, `v_x = u_x cos ψ + u_y sin ψ`, `v_y = −u_x sin ψ + u_y cos ψ`, and sent to the robot as `Move(v_x, v_y, ω)`.

**Reference** `x_ref(t)` comes from a waypoint planner at a linear speed of 0.3 m/s sampled every 0.02 s, offset by the robot's start pose.

### Closed-loop simulation
**Simulation setup.** Plant as above (a unicycle variant is in `out/control.json`), 5 m circle, 10 m line + turn, L-shape. GNSS error magnitudes are resampled from the bags; odometry scale error 1.21 (assumes RTK radius is nearer truth than odometry). Cross-track RMS on the 5 m circle, plain-GNSS noise: GNSS+odometry increments 0.68 m, EKF with gain schedule + odometry v,ω (ext.) 1.17 m, **recovered EKF diverges** (~440 m). The closed loop uses that earlier gain-schedule extension, not the EKF (ext.), which diverged on this holonomic plant (it cannot see sideways motion between 1 Hz fixes). Odometry-only drifts by its scale error (1.33 m). These are simulation outcomes tied to my noise model, not robot results.

### Comparison with the recorded control logs
Two pasted `traj_server` terminal logs (per 20 ms cycle: vx, vy, omega). Log A shows a diverged/saturated controller (vx up to 14 m/s, omega stuck). Log B starts as expected for an LQR whose reference moves while the robot has not yet: vx rises 0.75 m/s² (sim 0.92 m/s² with K_x=3.06), then departs. The reference path behind either log is not recorded, so no matching simulation path could be identified.

## Reproduce
```
python -m venv .venv && .venv/bin/pip install numpy scipy pandas matplotlib pyproj rosbags pillow   # once
.venv/bin/python -I src/extract.py     # bags -> data/*.npz (needs _extract/, not in git)
make all                               # tests, every figure, out/results.json, out/fido_ekf_replay.json
```
`make all` works from `data/` alone (about 12 min; the EKF ablation dominates).

## Limits (carried over from the portfolio write-up)
The 6.7 to 1.7 m figure compares GNSS to odometry, and odometry is not ground truth. The EKF was judged visually on the robot; there is no surveyed reference, so no accuracy claim is made for the filter. 2-D only; indoor/outdoor operation is not fused. The final on-robot code is lost; the recovered code differs in detail from the robot version (class is LQR, paper says LQG; pyproj instead of Haversine). Deviations from the recovered code: `docs/deviations.md`. Trial counts and the scripts behind the paper's numbers are unrecoverable.

## Layout
`src/fido/` shared code (`geo`, `heading`, `ekf`, `control`, `sim`), `src/fig_*.py` one script per figure, `out/` figures + JSON (`results.json` lists every headline number with script and bag), `out/fido_ekf_replay.json` replay payload for the browser demo (~70 kB), `data/` extracted arrays (27 MB), `tests/test_core.py`. `_extract/` and the zip stay out of git.
