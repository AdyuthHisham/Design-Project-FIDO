"""Collect headline numbers from the per-script JSONs into out/results.json, each with the script and bags that produced it."""
import sys, json; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fido.data import OUT, RUNS
J = lambda n: json.load(open(OUT / n)); rtk, al, ek, ns, ct, lg = J("rtk.json"), J("alignment.json"), J("ekf_summary.json"), J("noise.json"), J("control.json"), J("logs.json")
b = lambda *ks: [RUNS[k][0] for k in ks]
rtk_bags, plain_bags = b("nov24_circ_a", "nov24_circ_b"), b("nov25_circ5", "nov25_circ7", "nov18_circ_a", "nov18_circ_b", "jan12_plain")
H = lambda v, script, bags, note="": dict(value=v, script=script, bags=bags, note=note)
res = {
 "paper_reported_rmse_plain_m": H(6.7, "paper Sec. IV-A / Fig 3", [], "cited, not computed"),
 "paper_reported_rmse_rtk_m": H(1.7, "paper Sec. IV-A / Fig 3", [], "cited, not computed"),
 "rmse_rtk_paper_style_m": H(rtk["summary"]["rtk"]["rmse_paper_style"], "src/fig_rtk.py", rtk_bags, "aligned GNSS vs odometry, recovered alignment rule, mean of 2 runs"),
 "rmse_plain_paper_style_m": H(rtk["summary"]["plain"]["rmse_paper_style"], "src/fig_rtk.py", plain_bags, "same, mean of 5 runs"),
 "rmse_rtk_with_stationary_gate_m": H(rtk["summary"]["rtk"]["rmse_gated"], "src/fig_rtk.py", rtk_bags, "extension: regression window cleared when robot is still"),
 "rmse_plain_with_stationary_gate_m": H(rtk["summary"]["plain"]["rmse_gated"], "src/fig_rtk.py", plain_bags),
 "rmse_rtk_best_rigid_fit_m": H(rtk["summary"]["rtk"]["rmse_rigid"], "src/fig_rtk.py", rtk_bags, "alignment-independent lower bound"),
 "rmse_plain_best_rigid_fit_m": H(rtk["summary"]["plain"]["rmse_rigid"], "src/fig_rtk.py", plain_bags),
 "circle_fit_rms_rtk_m": H(rtk["summary"]["rtk"]["circ_rms_gnss"], "src/fig_rtk.py", rtk_bags, "RMS radial residual about fitted circle; no odometry used"),
 "circle_fit_rms_plain_m": H(rtk["summary"]["plain"]["circ_rms_gnss"], "src/fig_rtk.py", plain_bags),
 "static_rtk_std_e_m": H(rtk["static_floor"]["std_e"], "src/fig_rtk.py", b("dec04_static")),
 "static_rtk_std_n_m": H(rtk["static_floor"]["std_n"], "src/fig_rtk.py", b("dec04_static")),
 "mixed_run_rtk_dropout_at_s": H(rtk["dropout"]["status_changes_at_s"], "src/fig_rtk.py", b("jan12_mixed"), f"RMS err RTK part {rtk['dropout']['rms_err_rtk']:.2f} m, plain part {rtk['dropout']['rms_err_plain']:.2f} m (rigid-fit residual vs odometry)"),
 "ekf_ablation_mean_over_runs": H(ek, "src/fig_ekf.py", [RUNS[k][0] for k in RUNS if k != "dec04_static"], "len_ratio_vs_odom, jerk_rms, rms_to_odom per method; consistency, not accuracy"),
 "static_gnss_cov_en_m2": H(ns["static"]["cov_en"], "src/fig_noise.py", b("dec04_static"), "hand-set R_gps in ekf_Controller.py: 0.5 m^2 (x10 inflation = 5 m^2)"),
 "sim_closed_loop_circle_xtrack_rms_m": H({k: round(v["xtrack_rms"], 3) for k, v in ct["table"].items() if k.startswith("circle|")}, "src/fig_control.py", [], "SIMULATION; noise magnitudes resampled from bags; recovered EKF diverges closed-loop"),
 "control_log_B_start_slope_logged_vs_sim_mps2": H([lg["B"]["start_slope_logged_mps2"], lg["B"]["start_slope_sim_mps2"]], "src/fig_logs.py", ["_extract/traj_server logs"], "partial agreement only"),
}
json.dump(res, open(OUT / "results.json", "w"), indent=1, default=float); print(len(res), "entries")
