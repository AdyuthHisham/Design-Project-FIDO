# Deviations from the recovered code

Original = `recovered/code/{coordTurnClass,headingCorrector,readCSV}.py` (the final on-robot code is lost). Port = `src/fido/`.

## Fixes (needed to make the code run or the maths consistent)
| # | Original | Port | Where |
|---|----------|------|-------|
| 1 | `readCSV.simplify_columns` rescales timestamps to [0,1], so EKF `dt` is wrong | Real seconds from the bag: `(t_ns - t0)/1e9`; EKF `dt` = difference of consecutive odometry stamps | `fido/ekf.py` `replay`, `step` |
| 2 | `run` uses undefined `ts` | Replaced by the real-seconds array built in `replay` | `fido/ekf.py` |
| 3 | pyproj AEQD for lat/lon to metres (paper says Haversine) | Both implemented, `geo="aeqd"` default (as the code) and `"haversine"` (as the paper). Difference tested to be < 0.1 m over +-30 m | `fido/geo.py`, `tests/test_core.py` |
| 4 | The helper at `coordTurnClass.py` ~l.146 rotates by `-bias_main`, while `bias_main = odom_yaw - gnss_heading` (sign convention unclear; not verified which path ran on the robot) | Rotate by `+bias_main` (the sign that maps GNSS heading onto odometry yaw). Chosen because it gives the low RMSE after alignment (`out/alignment.json`) | `fido/ekf.py`, `fido/heading.py` |
| 5 | Module-level growing lists, prints, degrees/radians mixed | Instance state, no prints, radians everywhere, angles wrapped to [-pi, pi) | `fido/heading.py` |
| 6 | Odometry-pose update recovery `+0.0025` per step assumes 50 Hz | Scaled by `dt/0.02` so it does not depend on the bag rate (odometry here is ~300 Hz) | `fido/ekf.py` |

## Extensions (not in the original; every figure labels them `ext.`)
- `still_below`: clear the regression window when median speed < 0.05 m/s (the original popped one point per slow odometry sample).
- `odom_dyn`: set `v`, `omega` from odometry twist each step. **In the original no measurement ever corrects `v`.**
- `status_R`: switch GNSS covariance on fix status (RTK = 2). The original used one fixed R.
- `odom_mode="R"`, `odom_rec`: scale the odometry cost R_odom by 1/odom_red after each GNSS fix (instead of scaling the update gain, as the recovered code does) and set the decay rate. Off by default; used only by the standalone experiment `src/test_odom_schedule.py` (`out/odom_schedule.json`), not by any figure or the README.
- `PAPER_EKF` (`fido/ekf.py`): R schedule with `odom_rec=0.000025` + odometry v,omega + stationary gate. Used by the replay figures, ablation row "EKF paper-like", GIF and payload; not by the closed-loop simulation (it diverges on the holonomic plant). Decay rate is my choice, not recovered.
- Closed loop: `ekf_ext` estimator = the above; `ekf` = recovered filter unchanged.

## Not changed on purpose
- Hand-set noise values (`R_GPS`, `R_ODOM`, `Q`) are the on-robot ones from `ekf_Controller.py`.
- Class remains LQR-style (A=I, B=dt*I); the paper calls it LQG. The "LQG-style" loop is LQR on EKF estimates.
- Odometry is a relative baseline, not ground truth.

## Known behaviour that is a property of the recovered filter, not of the port
The recovered EKF diverges in closed loop (`out/control.json`, `ekf` rows) and its position path is much longer/rougher than odometry (`out/ekf_summary.json`, "EKF original"). The port reproduces this; it was not tuned away.
