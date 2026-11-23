import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Proj, Transformer
import math

import sys
import os

import pdb

# sys.path.append(os.path.abspath("../Scripts")) # To import Ady scripts

from readCSV import simplify_columns  # Uncomment if needed
from headingCorrector import ContinuousYawExtractor,GPSHeadingBiasCorrector, custom_arctan2
"""
Notes: 
    1. Integrate LQG
    2. Signal Processing
    3. Add IMU
"""

class CT_EKF_Filter:
    """
    Constant Turn Rate Extended Kalman Filter (CT-EKF) for GPS trajectory smoothing.
    Implements a coordinated turn motion model with position measurements.
    """

    def __init__(
        self,
        init_pos_uncertainty: float = 10.0,
        init_omega_uncertainty: float = 10.0,
        odom_vel_noise: float = 3.0,
        odom_pose_noise: float = 3.0,
        gps_noise: float = 3.0,
        process_noise_pos: float = 0.1,
        process_noise_vel: float = 1.0,
        process_noise_omega: float = 0.1,
        process_noise_psi: float = 1.0,
        
    ):
        self.yaw_extractor = ContinuousYawExtractor()
        self.corrector = GPSHeadingBiasCorrector(buffer_size=6, r2_threshold=0.93) # Need to vary based on GPS/RTK
        self.pos_flag = 0
        self.pos_offset = [0.0,0.0]

        self.first_update = True

        self.odom_vel_noise = odom_vel_noise
        self.gps_noise = gps_noise
        self.odom_pose_noise = odom_pose_noise

        self.process_noise_pos = process_noise_pos
        self.process_noise_vel = process_noise_vel
        self.process_noise_omega = process_noise_omega
        self.process_noise_psi = process_noise_psi

        self.global_pos = []
        self.robot_pos = []

        self.inov_gps = []
        self.inov_odom = []

        self.gps_px = []
        self.gps_py = []
        self.gps_yaw = []

        self.odom_vx = []
        self.odom_vy = []
        self.odom_yaw = []
        self.odom_omega = []

        self.predicted_states = []  # stores x_pred, P_pred, F
        self.updated_states   = []  # stores x_post, P_post
        self.odom_reduction = 1.0        
        self.R = None
        self.Q_base = np.diag([process_noise_pos, process_noise_pos, process_noise_vel,process_noise_psi,
                               process_noise_omega])       # Base process noise

        # Initial state and covariance
        self.reset()

        # Initial covariance
        self.P = np.eye(5) * init_pos_uncertainty
        self.P[3, 3] = init_omega_uncertainty
        self.P[4, 4] = init_omega_uncertainty

        self.H_gps = np.zeros((3, 5))
        self.H_gps[0, 0] = 1.0
        self.H_gps[1, 1] = 1.0
        self.H_gps[2, 3] = 1.0

        # For dt
        self.dt = 0.0
        self.old_ts = 0.0

        self.update_pinged=False

        self.old_odom_meas = None

    def reset(self):
        """Reset state and covariance to initial values."""
        self.x = np.zeros(5)        # [px, py, v, psi, omega]
        self.P = np.eye(5) * 10.0
        # self.P[4, 4] = 0.1

    # def _latlon_to_xy(self, lat: float, lon: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    #     """Convert lat/lon to local Cartesian using AEQD projection centered at reference."""
        
    #     # Initialize projection only once
    #     if self.first_update:
    #         proj_enu = Proj(proj="aeqd", lat_0=ref_lat, lon_0=ref_lon, datum="WGS84")
    #         self.ll_to_xy = Transformer.from_proj("epsg:4326", proj_enu, always_xy=True)
    #         self.xy_to_ll = Transformer.from_proj(proj_enu, "epsg:4326", always_xy=True)
    #         self.first_update = False

    #     # Convert
    #     east, north = self.ll_to_xy.transform(lon, lat)
    #     return east, north

    # def _xy_to_latlon(self, x: float, y: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    #     """Convert local Cartesian back to lat/lon using inverse AEQD projection."""
        
    #     lon, lat = self.xy_to_ll.transform(x, y)
    #     return lat, lon


    def global_to_robot(self,dx_g, dy_g):
        """
        Convert displacement from global frame to robot frame.

        Parameters:
            dx_g  : displacement in global X
            dy_g  : displacement in global Y
            theta : robot orientation (radians)

        Returns:
            (dx_r, dy_r) in robot frame

            
        """

        # Convert degrees to radians
        theta_rad = self.corrector.bias_main

        dx_r = dx_g * math.cos(theta_rad) + dy_g * math.sin(theta_rad)
        dy_r = -dx_g * math.sin(theta_rad) + dy_g * math.cos(theta_rad)

        return dx_r, dy_r

    def _predict(self, odom_meas):
        """
        EKF prediction step using odometry velocity and angular rate as control inputs.
        odom_meas: tuple or list (v_odom, omega_odom)
        """

        if self.first_update:
            odom_yaw = self.yaw_extractor.quat_to_yaw_rad(odom_meas[4],odom_meas[5],odom_meas[6],odom_meas[7])
            v_x_odom,v_y_odom = odom_meas[2],odom_meas[3]
            self.x = [odom_meas[0],odom_meas[1],v_x_odom,
                odom_yaw,odom_meas[8]]
            self.first_update=False
            return

        if self.dt <= 0:
            return

        px, py, v, psi, omega = self.x
        # v_x_odom,v_y_odom, omega_odom = odom_meas[2],odom_meas[3],odom_meas[8]



        # Initialize Jacobian
        F = np.eye(5)

        # Small-omega threshold to avoid division by zero
        eps = 0.02

        if abs(omega) < eps:
            # Straight line motion
            px += v * np.cos(psi) * self.dt
            py += v * np.sin(psi) * self.dt
            psi += omega * self.dt  # tiny, but keep for completeness

            # Jacobian
            F[0, 3] = -v * np.sin(psi) * self.dt
            F[1, 3] =  v * np.cos(psi) * self.dt
            F[0, 2] = np.cos(psi) * self.dt
            F[1, 2] = np.sin(psi) * self.dt
            F[0, 4] = -0.5 * v * np.sin(psi) * (self.dt**2)
            F[1, 4] =  0.5 * v * np.cos(psi) * (self.dt**2)
        else:
            # Turning motion (CTRV)
            theta = omega * self.dt
            px += (v / omega) * (np.sin(psi + theta) - np.sin(psi))
            py += (v / omega) * (-np.cos(psi + theta) + np.cos(psi))
            psi += theta

            # Jacobian entries
            F[0, 2] = (np.sin(psi) - np.sin(psi - theta)) / omega
            F[1, 2] = (np.cos(psi - theta) - np.cos(psi)) / omega
            F[0, 3] = (v / omega) * (np.cos(psi) - np.cos(psi - theta))
            F[1, 3] = (v / omega) * (np.sin(psi) - np.sin(psi - theta))
            F[0, 4] = v * (theta * np.cos(psi) - (np.sin(psi) - np.sin(psi - theta))) / (omega**2)
            F[1, 4] = v * (theta * np.sin(psi) + (np.cos(psi) - np.cos(psi - theta))) / (omega**2)

        # Write back updated state
        self.x[0] = px
        self.x[1] = py
        self.x[3] = self.corrector.wrap_angle_2pi(psi)  # keep yaw in [-pi, pi)
        self.x[2] = v
        self.x[4] = omega

        # Store Jacobian for covariance propagation
        self.F = F

        # Propagate covariance
        Q_dt = self.Q_base * self.dt
        self.P = F @ self.P @ F.T + Q_dt

        self.old_odom_meas = odom_meas

        # # Copy the current predicted state and covariance
        # self.x_pred = self.x.copy()
        # self.P_pred = self.P.copy()
        # # self.F already computed in _predict

        # # Append to predicted_states list
        # self.predicted_states.append({
        #     'x_pred': self.x_pred.copy(),
        #     'P_pred': self.P_pred.copy(),
        #     'F': self.F.copy()
        # })

        # # Optional: keep only the last L + 1 elements to save memory
        # if len(self.predicted_states) > self.L + 1:
        #     self.predicted_states.pop(0)


    def _update_odom(self, odom_meas):
        """
        EKF Update Step: Odometry-based correction using only position and heading measurements.

        Measured states (z):
            z = [px, py, psi] (3x1)

        State vector (x):
            x = [px, py, v, psi, omega] (5x1)
        """

        # ------------------------------------------------------------------
        # 1. Measurement Vector (z) and Measurement Jacobian (H)
        # ------------------------------------------------------------------

        # Extract odometry measurements for position and heading
        px_odom    = odom_meas[0]
        py_odom    = odom_meas[1]
        psi_odom = self.yaw_extractor.quat_to_yaw_rad(odom_meas[4],odom_meas[5],odom_meas[6],odom_meas[7])

        # Measurement vector (3x1)
        z_odom = np.array([
            px_odom,
            py_odom,
            psi_odom
        ])

        # Measurement Jacobian (H): Maps the 5-state vector (x) to the 3-measurement vector (z)
        # [px, py, v, psi, omega] -> [px, py, psi]
        H_odom = np.array([
            [1, 0, 0, 0, 0],  # px = 1*px + 0*py + ...
            [0, 1, 0, 0, 0],  # py = 0*px + 1*py + ...
            [0, 0, 0, 1, 0]   # psi = 0*px + ... + 1*psi
        ])

        # Expected measurement from current state (h(x) = H * x) (3x1)
        h_x = H_odom @ self.x

        # ------------------------------------------------------------------
        # 2. Measurement Noise and Innovation
        # ------------------------------------------------------------------

        # NOTE: R_odom must now be a 3x3 matrix corresponding to [px, py, psi]
        # You need to define a new covariance matrix for this reduced measurement set.
        # For example, if your old R_odom was a 5x5 diagonal matrix, use the first, second, and fourth diagonal elements.
        # Let's assume you have a new variable for this: self.odom_pos_noise (3x3)
        R_odom = self.odom_pose_noise # Must be 3x3

        # Innovation (y = z - h(x)) (3x1)
        y = z_odom - h_x

        # Wrap heading innovation to [-pi, pi] (The heading is the 3rd element, index 2)
        # y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi

        # ------------------------------------------------------------------
        # 3. Kalman Gain and State/Covariance Update
        # ------------------------------------------------------------------

        # Innovation covariance (S = H*P*H^T + R) (3x3)
        S = H_odom @ self.P @ H_odom.T + R_odom

        # Kalman gain (K = P*H^T*S^-1) (5x3)
        K = self.P @ H_odom.T @ np.linalg.inv(S)

        delta_max = np.array([
            0.02,   # x max per update (meters)
            0.02,   # y max per update
            0,
            0.005,  # yaw max per update (rad)
            0
        ])
        # delta = (K @ y)
        # delta = np.clip(delta, -delta_max, delta_max)

        # State update (self.x is updated) (5x1)
        self.x = self.x + self.odom_reduction*(K @ y)
        # self.x = self.x + delta

        # Store innovation (optional logging)
        self.inov_odom.append((self.old_ts, *y)) # Note: y is now 3 elements long

        # Covariance update (P = (I - K*H)*P) (5x5)
        self.P = (np.eye(5) - K @ H_odom) @ self.P

        self.odom_reduction = np.minimum(self.odom_reduction + 0.0025,1.0)
        # self.P = (np.eye(5) - K @ H_odom) @ self.P @ (np.eye(5) - K @ H_odom).T + K @ R_odom @ K.T



    # def _update_gps(self, gps_meas: np.ndarray):
    #     """
    #     EKF update step.
    #     z_meas:       measurement vector [px_meas, py_meas, psi_meas]  (shape 3,)
    #     sensor_flag:  [odom_active, gps_active]  (list or array of 0/1)
    #     """
    #     # ------------------------------------------------------------------
    #     # 2. GPS update: also observes [px, py, ψ]
    #     # ------------------------------------------------------------------

    #     R_gps = self.gps_noise   # GPS yaw (from velocity) is usually quite good
    #     # Meas is [lat,lon]
    #     temp_meas = [0.0,0.0,0.0]
    #     # print(gps_meas[0], gps_meas[1])
    #     temp_meas[2], _, _, _, temp_meas[0], temp_meas[1] = self.corrector.update(gps_meas[0], gps_meas[1], self.x[3])
    #     # self.gps_px.append(temp_meas[0])
    #     # self.gps_py.append(temp_meas[1])
    #     # self.gps_yaw.append(temp_meas[2])
    #     self.global_pos.append((temp_meas[0], temp_meas[1]))
    #     if self.corrector.bias_main != None:
    #         temp_meas[0], temp_meas[1] = self.global_to_robot(temp_meas[0], temp_meas[1])
            
    #         # Assume that odom is init first
    #         z_meas = [temp_meas[0] + self.pos_offset[0],
    #                 temp_meas[1] + self.pos_offset[1],
    #                 temp_meas[2]]
    #         self.robot_pos.append((temp_meas[0]+ self.pos_offset[0], temp_meas[1]+ self.pos_offset[1]))
    #         y = z_meas - self.H_gps @ self.x
    #         y[2] =  self.corrector.wrap_angle_2pi(y[2]) 
    #         self.inov_gps.append((self.old_ts,y[0],y[1],y[2]))
    #         S = self.H_gps @ self.P @ self.H_gps.T + R_gps
    #         K = self.P @ self.H_gps.T @ np.linalg.inv(S)

    #         self.x = self.x + K @ y
    #         self.P = (np.eye(5) - K @ self.H_gps) @ self.P

    #         # Normalize yaw again after GPS correction
    #         self.x[3] = custom_arctan2(np.sin(self.x[3]), np.cos(self.x[3]))
    #     else:
    #         pass

    def _update_gps_combined_pose(self, gps_meas):
        """
        EKF Update Step (Combined Pose): Fuses GPS pose [px, py, psi] and 
        Odometry pose [px, py, psi] into a single measurement update.

        Args:
            gps_meas (np.ndarray): GPS measurement [lat, lon, psi] (assuming pre-processed)
            odom_pose (np.ndarray): Odometry pose [px, py, psi]
            R_gps (np.ndarray): 3x3 noise matrix for GPS [px, py, psi]
            R_odom_pose (np.ndarray): 3x3 noise matrix for Odom Pose [px, py, psi]
        """
        if (self.pos_flag==0):
                self.pos_offset = [self.old_odom_meas[0],self.old_odom_meas[1]]
                self.pos_flag = 1
                print(self.pos_offset)
        
        # ------------------------------------------------------------------
        # 1. GPS Lat/Lon to Local Frame (Preserving your existing conversion logic)
        # ------------------------------------------------------------------
        
        # temp_meas is the final, local-frame GPS measurement [px, py, psi]
        temp_meas = [0.0,0.0,0.0]
        
        # Assuming self.corrector handles the lat/lon to local frame conversion
        temp_meas[2], _, _, _, temp_meas[0], temp_meas[1] = self.corrector.update(gps_meas[0], gps_meas[1], self.x[3])
        self.global_pos.append((temp_meas[0], temp_meas[1]))
        gps_local_pose = np.array([0.0,0.0,0.0])
        # Apply offset for the robot's local frame
        if self.corrector.bias_main is not None:
            temp_meas[0], temp_meas[1] = self.global_to_robot(temp_meas[0], temp_meas[1])
            gps_local_pose = np.array([
                temp_meas[0] + self.pos_offset[0],
                temp_meas[1] + self.pos_offset[1],
                temp_meas[2]
            ])
            self.robot_pos.append((temp_meas[0]+ self.pos_offset[0], temp_meas[1]+ self.pos_offset[1]))

        else:
            # Cannot update if not initialized
            return
        
        # ------------------------------------------------------------------
        # 2. Define Combined Measurement Vector (z) and Jacobian (H)
        # ------------------------------------------------------------------

        # Combined Measurement Vector (6 elements): [GPS_px, GPS_py, GPS_psi, ODOM_px, ODOM_py, ODOM_psi]
        # NOTE: Assuming odom_pose is already in the same local coordinate frame as gps_local_pose
        odom_yaw = self.yaw_extractor.quat_to_yaw_rad(self.old_odom_meas[4],self.old_odom_meas[5],self.old_odom_meas[6],self.old_odom_meas[7])
        odom_px, odom_py, odom_psi = self.old_odom_meas[0], self.old_odom_meas[1], odom_yaw
        odom_pose = [odom_px, odom_py, odom_psi]
        z_combined = np.concatenate((gps_local_pose, odom_pose))  # Shape (6,)

        # H_combined: Jacobian for the combined measurement. It must select 
        # [px, py, psi, px, py, psi] from the state vector [px, py, v, psi, omega]
        # H must be 6 rows (measurements) by 5 columns (states)
        H_combined = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],  # GPS px
            [0.0, 1.0, 0.0, 0.0, 0.0],  # GPS py
            [0.0, 0.0, 0.0, 1.0, 0.0],  # GPS psi
            [1.0, 0.0, 0.0, 0.0, 0.0],  # ODOM px
            [0.0, 1.0, 0.0, 0.0, 0.0],  # ODOM py
            [0.0, 0.0, 0.0, 1.0, 0.0]   # ODOM psi
        ])
        
        # ------------------------------------------------------------------
        # 3. Define Combined Noise Matrix (R_combined)
        # ------------------------------------------------------------------

        # R_combined: Block-diagonal matrix combining R_gps and R_odom_pose. (6x6)
        # R_odom_pose MUST be significantly LARGER than R_gps to model the difference 
        # in confidence (GPS is the anchor).
        R_combined = np.block([
            [self.gps_noise, np.zeros((3, 3))],
            [np.zeros((3, 3)), self.odom_pose_noise]
        ])
        # ------------------------------------------------------------------
        # 4. Kalman Filter Update Logic
        # ------------------------------------------------------------------

        # Innovation: The difference between the measurement and the prediction
        h_x = H_combined @ self.x 
        y = z_combined - h_x
        
        # Yaw angle wrapping for the psi states (3rd and 6th elements in y)
        # y[2] = self.corrector.wrap_angle_2pi(y[2]) 
        # y[5] = self.corrector.wrap_angle_2pi(y[5])
        
        # Innovation Covariance (S)
        S = H_combined @ self.P @ H_combined.T + R_combined
        
        # Kalman Gain (K)
        K = self.P @ H_combined.T @ np.linalg.inv(S)

        # State Update
        self.x = self.x + K @ y
        
        # Covariance Update
        self.P = (np.eye(5) - K @ H_combined) @ self.P
        # self.P = (np.eye(5) - K @ H_combined) @ self.P @ (np.eye(5) - K @ H_combined).T + K @ R_combined @ K.T

        self.inov_gps.append((self.old_ts,y[0],y[1],y[2],y[0],y[4],y[5]))

        self.odom_reduction = 0.00020
        # self.odom_reduction = 1.0

        # Normalize yaw again after correction
        # self.x[3] = self.corrector.wrap_angle_2pi(self.x[3]) # Assuming wrap_angle_2pi handles the custom_arctan2 logic

    # def rts_smooth(self):
    #     """
    #     Fixed-lag RTS smoothing for the last self.L states.
    #     Requires that self.updated_states and self.predicted_states are populated
    #     with:
    #         - x_post, P_post for updated states
    #         - x_pred, P_pred, F for predicted states
    #     Adds x_s and P_s to self.updated_states for the smoothed estimates.
    #     """
    #     N = len(self.updated_states)
    #     if N < 2:
    #         return  # Not enough states to smooth
    #     # pdb.set_trace()
    #     # Determine start index for fixed-lag smoothing
    #     start_idx = max(0, N - self.L)

    #     # Initialize the last smoothed state (most recent)
    #     self.updated_states[-1]['x_s'] = self.updated_states[-1]['x_post'].copy()
    #     self.updated_states[-1]['P_s'] = self.updated_states[-1]['P_post'].copy()

    #     # Backward pass for smoothing
    #     for k in range(N - 2, start_idx - 1, -1):
    #         # Extract Jacobian and covariances
    #         F = self.predicted_states[k + 1]['F']
    #         P_post = self.updated_states[k]['P_post']
    #         P_pred_next = self.predicted_states[k + 1]['P_pred']

    #         # Compute RTS smoothing gain
    #         G = P_post @ F.T @ np.linalg.inv(P_pred_next)

    #         # Smoothed state
    #         x_post = self.updated_states[k]['x_post']
    #         x_pred_next = self.predicted_states[k + 1]['x_pred']
    #         x_s_next = self.updated_states[k + 1]['x_s']

    #         # Apply smoothing
    #         x_s = x_post + G @ (x_s_next - x_pred_next)

    #         # Wrap yaw angle properly (3rd index = psi)
    #         x_s[3] = self.corrector.wrap_angle_2pi(x_s[3])

    #         self.updated_states[k]['x_s'] = x_s

    #         # Smoothed covariance
    #         P_s_next = self.updated_states[k + 1]['P_s']
    #         P_s = P_post + G @ (P_s_next - P_pred_next) @ G.T
    #         self.updated_states[k]['P_s'] = P_s

    def run(
        self,
        dt,
        odom_meas,
        gps_meas=None # Sends out Odo: [Px, Py, Vx, Vy , Quats ] GPS: [Lat, Lon]
    ):
        self.dt = ts - self.old_ts
        self.old_ts = ts
        self._predict(odom_meas)
        # self.predicted_states.append([self.old_ts,self.x])
        
        if gps_meas is not None and np.any(gps_meas != 0):
            self.update_pinged = True
            # self._update_gps(gps_meas)
            self._update_gps_combined_pose(gps_meas)
            # self.updated_states.append([self.old_ts,self.x])
        else:
            self._update_odom(odom_meas)
            self.update_pinged = False
            # if len(self.predicted_states) >= self.L:
            #     self.rts_smooth()
            #     # Rebase current state
            #     self.x = self.updated_states[-1]['x_s'].copy()
            #     self.P = self.updated_states[-1]['P_s'].copy()

        # return self.x


# ================================
# Example Usage
# ================================
if __name__ == "__main__":
    variance_flag = False
    # R_gps = np.array([[ 2.73638351e-03, -7.22910176e-05, 0],
    #                   [-7.22910176e-05,  4.45971932e-04, 0],
    #                   [0, 0, 1.903270742219267]])

    R_gps = np.diag([0.5,0.5, 1.903270742219267])

    R_odom_rates = np.diag([
    45.0,    # px  (meters^2)
    45.0,    # py  (meters^2)
    4.0,     # v   ((m/s)^2)
    1.0,     # psi (rad^2)
    2.0      # omega ((rad/s)^2)
])

    R_odom_pose = np.diag([50.0,50.0,0.25])

    # R_odom = 40*np.diag([6.16185278385507e-08, 2.1641375691748348e-08, 1.83829414057374e-06]),
    # R_gps = 40*np.diag([0.002764887506999583, 0.0004506174733640733,1.9383101293182874]),
    ekf = CT_EKF_Filter(
        odom_vel_noise = 1000 * R_odom_rates,
        odom_pose_noise = 100 * R_odom_pose, 
        gps_noise = 1 * R_gps,
        # process_noise_pos=0.001,
        # process_noise_vel=0.05845593931085183,
        # process_noise_psi=0.1,
        # process_noise_omega=0.03799619779043127,
        process_noise_pos=5.0**2,
        process_noise_vel=2.5**2,
        process_noise_psi=(10.0 * math.pi / 180.0) ** 2,
        process_noise_omega=(0.20) ** 2,
    )

    # df = simplify_columns(pd.read_csv('/home/adhish/Productivity/FIDO/SSY226SensorFusion/programmeringstestsensorfusion/TestData/files4dec/rosbag2_2025_12_04-18_37_53/merged_data.csv'))
    df = simplify_columns(pd.read_csv('/home/adhish/Productivity/FIDO/SSY226SensorFusion/programmeringstestsensorfusion/TestData/Bag_Nov_25/Nov25_Test/rosbag2_2025_11_25-18_20_45/merged_data.csv'))
    # df = pd.read_csv("noisy_circle_data.csv")
    # ekf.corrector.bias_main = 171.97421602608037 # If known
    ekf.corrector.bias_main = 6.01847137441311
    df = df.sort_values('ros_timestamp_ns').reset_index(drop=True)

    # average = abs(df['odom_vel_angular_z'].mean())
    # max_val = abs(df['odom_vel_angular_z'].max())
    # min_val = abs(df['odom_vel_angular_z'].min())

    # print(f"Average: {average}")
    # print(f"Max: {max_val}")
    # print(f"Min: {min_val}")
    # exit()

    #--------------------------

#    # Define the columns needed for plotting
#     plot_cols = ['ros_timestamp_ns', 'odom_vel_angular_z']

#     # --- 1. Data Cleaning and Time Calculation ---
    
#     # Drop NaN values in the relevant columns and create a copy
#     df_clean = df.dropna(subset=plot_cols).copy()
    
#     if df_clean.empty:
#         print("Error: DataFrame is empty after dropping NaN values.")

#     # Convert nanoseconds to seconds, starting time at t=0
#     df_clean['time_sec'] = (df_clean['ros_timestamp_ns'] - df_clean['ros_timestamp_ns'].iloc[0]) / 1e9
    
#     # Calculate the time step (dt) for each row
#     df_clean['dt'] = df_clean['time_sec'].diff().fillna(0)
    
#     # Replace any dt=0 with a very small number to avoid division by zero in acceleration
#     df_clean['dt'] = df_clean['dt'].replace(0, 1e-9)

#     # --- 2. Calculate Angular Displacement (Integration) ---
    
#     # Calculate the angular change at each step: d_psi = omega_z * dt
#     df_clean['d_psi'] = df_clean['odom_vel_angular_z'] * df_clean['dt']
    
#     # Calculate the cumulative angular displacement
#     df_clean['cumulative_psi'] = df_clean['d_psi'].cumsum()

#     # --- 3. Calculate Angular Acceleration (Differentiation) ---

#     # Calculate Angular Acceleration: d(omega)/dt
#     # Note: Use the raw omega_z for differentiation, as smoothing can mask peaks, 
#     # but be aware the result will be very noisy.
#     df_clean['angular_acceleration'] = df_clean['odom_vel_angular_z'].diff() / df_clean['dt']
    
#     # --- 4. Plotting Subplots ---
    
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
#     fig.suptitle('Analysis of Robot Turning Dynamics', fontsize=16)
    
#     # --- Subplot 1: Angular Displacement ---
    
#     ax1.plot(df_clean['time_sec'], df_clean['cumulative_psi'], 
#              label='Cumulative Yaw Change', color='darkblue', linewidth=2)

#     # Highlight the 90-degree target
#     ninety_deg_rad = np.pi / 2
#     ax1.axhline(ninety_deg_rad, color='green', linestyle=':', 
#                 linewidth=1.5, label=f'Target 90° ({ninety_deg_rad:.2f} rad)')

#     ax1.set_title('1. Angular Displacement (Total Rotation)', fontsize=14)
#     ax1.set_ylabel('Yaw Change (Radians)', fontsize=12)
#     ax1.grid(True, linestyle=':', alpha=0.7)
#     ax1.legend()

#     # --- Subplot 2: Angular Acceleration ---
    
#     ax2.plot(df_clean['time_sec'], df_clean['angular_acceleration'], 
#              label='Angular Acceleration', color='red', linewidth=1)

#     # Highlight zero acceleration
#     ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)

#     ax2.set_title('2. Angular Acceleration (Rate of Change of $\omega_z$)', fontsize=14)
#     ax2.set_xlabel('Time (seconds)', fontsize=12)
#     ax2.set_ylabel('Acceleration (rad/s²)', fontsize=12)
#     ax2.grid(True, linestyle=':', alpha=0.7)
#     ax2.legend()
    
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
#     plt.show()
#     exit()

    results = []

    print("Processing...")
    # Iterate row-by-row
    time_arr = []
    odom_meas = np.zeros(9)
    gps_meas = np.zeros(2)
    for _, row in df.iterrows():
        ts = row['ros_timestamp_ns']
        time_arr.append(ts)
        # ODOM measurement
        if pd.notna(row['odom_pos_x']):
            # print("Sup")
            
            odom_meas = [
                row['odom_pos_x'],
                row['odom_pos_y'],
                row['odom_vel_linear_x'],
                row['odom_vel_linear_y'],
                row['odom_quat_x'],
                row['odom_quat_y'],
                row['odom_quat_z'],
                row['odom_quat_w'],
                row['odom_vel_angular_z'],
            ]
        # GPS measurement
        if pd.notna(row["gps_fix_lat"]) and pd.notna(row["gps_fix_lon"]):
            gps_meas = [
                row["gps_fix_lat"],
                row["gps_fix_lon"],
            ]
        # if pd.notna(row["imu_orient_y"]):
            # print(f"sup: {abs(np.sum(meas))}")
      
        # Run EKF
        if odom_meas is not None and np.any(odom_meas != 0):
            ekf.run(ts,odom_meas,gps_meas)
        else:
            continue

        
        odom_meas = None
        gps_meas = None

        
    
        # Store EKF full state vector [px, py, v, yaw, yaw_rate]
        results.append([ts, *ekf.x, *ekf.P.flatten(),ekf.update_pinged])
    if variance_flag:
        # Odom
        #Delete
        qx, qy, qz, qw = odom_meas[4:8]
        if np.allclose([qx, qy, qz, qw], 0):
            print("Warning: zero quaternion, skipping yaw calculation")
            odom_yaw = 0.0   # or np.nan, depending on your application
        else:
            odom_yaw = ekf.yaw_extractor.quat_to_yaw_rad(qx, qy, qz, qw)
        ekf.odom_yaw.append(odom_yaw)
        ekf.odom_vx.append(odom_meas[2])
        ekf.odom_vy.append(odom_meas[3])
        ekf.odom_omega.append(odom_meas[8])
        odom_vx = np.array(ekf.odom_vx)
        odom_vy = np.array(ekf.odom_vy)
        odom_yaw = np.array(ekf.odom_yaw) 
        omega = np.array(ekf.odom_omega) 
        # velocity along heading
        v_along_yaw = odom_vx * np.cos(odom_yaw) + odom_vy * np.sin(odom_yaw)
        print("-----------ODOM--------------")
        v_process_noise = np.var(v_along_yaw, ddof=0)   # variance of forward velocity
        omega_process_noise = np.var(omega, ddof=0)     # variance of angular velocity

        print("Process noise for v (along yaw):", v_process_noise)
        print("Process noise for omega:", omega_process_noise)
        print("-----------GPS--------------")
        GPS_pos = np.cov(np.vstack([ekf.gps_px, ekf.gps_py]),bias=True)
        # print(f"var_px: {np.var(ekf.gps_px, ddof=1)}")
        # print(f"var_py: {np.var(ekf.gps_py, ddof=1)}")
        print(f"cov_pos: {GPS_pos}")
        print(f"var_yaw: {np.var(ekf.gps_yaw, ddof=0)}")
        exit()
    
    

    # --- 1. Define the Yaw Offset (Directly in Radians) ---
    # Replace the placeholder value with your actual offset in radians
    # Example: If the offset is 0.26 radians (approx 15 degrees)
    yaw_offset_rad = ekf.corrector.bias_main + np.pi # Example value
    df_clean = df.dropna(subset=['odom_pos_x', 'odom_pos_y'])

    odom_x = df_clean['odom_pos_x'].values
    odom_y = df_clean['odom_pos_y'].values


    # Find the starting point (assumed to be the offset reference point)
    start_x = odom_x[0]
    start_y = odom_y[0]

    # --- 3. Create the Rotation Matrix ---
    # Use the radian value directly
    cos_t = np.cos(yaw_offset_rad)
    sin_t = np.sin(yaw_offset_rad)
    R = np.array([
        [cos_t, -sin_t],
        [sin_t, cos_t]
    ])

    # --- 4. Apply the Transformation ---

    # A. Translate to Origin (Center the path at (0,0))
    translated_x = odom_x - start_x
    translated_y = odom_y - start_y

    # Combine into a 2xN array for matrix multiplication
    translated_points = np.vstack((translated_x, translated_y))

    # B. Rotate the points
    rotated_points = R @ translated_points

    # C. Translate Back (Shift the origin back)
    corrected_odom_x = rotated_points[0, :] + start_x
    corrected_odom_y = rotated_points[1, :] + start_y
    # Convert results to a NumPy array for easier plotting
    results = np.array(results)
    ts = results[:, 0]  # timestamps
    states = results[:, 1:6]  # EKF states: px, py, v, yaw, yaw_rate

    # Convert timestamps to seconds relative to the first timestamp
    ts = (ts - ts[0])  # assuming ros_timestamp_ns is in nanoseconds

    state_labels = ['px', 'py', 'v', 'yaw', 'yaw_rate']

    plt.figure(figsize=(12, 8))
    for i in range(states.shape[1]):
        plt.subplot(states.shape[1], 1, i+1)
        plt.plot(ts, states[:, i])
        plt.ylabel(state_labels[i])
        plt.grid(True)
        if i == states.shape[1] - 1:
            plt.xlabel('Time [s]')

    plt.suptitle('EKF State Estimates Over Time')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    robot_pos = np.array(ekf.robot_pos)
    global_pos = np.array(ekf.global_pos)
    # Plot trajectory: px vs py
    plt.figure(figsize=(8, 6))
    plt.plot(states[:, 0], states[:, 1], label='EKF Estimate', marker='o', markersize=2)
    plt.plot(robot_pos[:, 0], robot_pos[:, 1], label='Robot True Path (w/o offset)', marker='o', markersize=2)
    plt.plot(global_pos[:, 0], global_pos[:, 1], label='Global True Path (w/o offset)', marker='o', markersize=2)
    # 3. Corrected Odometry Path
    plt.plot(
        corrected_odom_x,
        corrected_odom_y,
        label=f'Corrected Odometry (Offset: {yaw_offset_rad*180/np.pi:.1f}°)',
        linestyle='--',
        linewidth=1.5
    )
    plt.xlabel('px [m]')
    plt.ylabel('py [m]')
    plt.title('EKF Estimated Trajectory')
    plt.grid(True)
    plt.axis('equal')  # Keep aspect ratio 1:1
    plt.legend()
    plt.show()


    inov_gps = np.array(ekf.inov_gps)   # shape (N,4)

    t_gps  = inov_gps[:,0]
    y0_gps = inov_gps[:,1]
    y1_gps = inov_gps[:,2]
    y2_gps = inov_gps[:,3]
    y3_gps = inov_gps[:,4]
    y4_gps = inov_gps[:,5]
    y5_gps = inov_gps[:,6]

    inov_odom = np.array(ekf.inov_odom)   # shape (N,4)

    t_odom  = inov_odom[:,0]
    y0_odom = inov_odom[:,1]
    y1_odom = inov_odom[:,2]


    plt.figure(figsize=(10, 8))

    # --- GPS subplot ---
    plt.subplot(2, 1, 1)
    plt.plot(t_gps, y0_gps, label="GPS innovation Px")
    plt.plot(t_gps, y1_gps, label="GPS innovation Py")
    plt.plot(t_gps, y2_gps, label="GPS innovation Psi")
    plt.plot(t_gps, y3_gps, label="Odom innovation Px")
    plt.plot(t_gps, y4_gps, label="Odom innovation Py")
    plt.plot(t_gps, y5_gps, label="Odom innovation Psi")
    plt.title("GPS Innovation vs Time")
    plt.ylabel("Innovation")
    plt.grid(True)
    plt.legend()

        # --- GPS subplot ---
    plt.subplot(2, 1, 2)
    plt.plot(t_odom, y0_odom, label="GPS innovation V")
    plt.plot(t_odom, y1_odom, label="GPS innovation Omega")
    # plt.plot(t_odom, y2_gps, label="GPS innovation Psi")
    plt.title("GPS Innovation vs Time")
    plt.ylabel("Innovation")
    plt.grid(True)
    plt.legend()

    # Extract time stamps
    pred_t = [s[0] for s in ekf.predicted_states]
    upd_t = [s[0] for s in ekf.updated_states]

    # Convert predicted/updated states to arrays for easier indexing
    pred_states = np.array([s[1] for s in ekf.predicted_states])  # shape (N,5)
    upd_states = np.array([s[1] for s in ekf.updated_states])

    state_labels = ['px (m)', 'py (m)', 'v (m/s)', 'psi (rad)', 'omega (rad/s)']

    # plt.figure(figsize=(12,10))
    # for i in range(5):
    #     plt.subplot(5,1,i+1)
    #     plt.plot(pred_t, pred_states[:,i], 'r--', label='Predicted')
    #     plt.plot(upd_t, upd_states[:,i], 'b-', label='Updated')
    #     plt.ylabel(state_labels[i])
    #     if i == 0:
    #         plt.title('Predicted vs Updated States Over Time')
    #     if i == 4:
    #         plt.xlabel('Time (s)')
    #     plt.legend()
    #     plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # Extract columns
    timestamps = results[:, 0]
    states = results[:, 1:6]                    # px, py, theta, v, omega
    P_flat = results[:, 6:31]                   # 5x5 = 25 elements flattened
    update_pinged = results[:, 31].astype(bool) # Last column: boolean for GPS update

    # Unflatten P
    n_timesteps = len(results)
    P = P_flat.reshape(n_timesteps, 5, 5)

    # States
    px = states[:, 0]
    py = states[:, 1]
    theta = states[:, 2]
    v = states[:, 3]
    omega = states[:, 4]

    # Variances and std devs
    var_px = P[:, 0, 0]
    var_py = P[:, 1, 1]
    var_theta = P[:, 2, 2]
    var_v = P[:, 3, 3]
    var_omega = P[:, 4, 4]

    std_px = np.sqrt(var_px)
    std_py = np.sqrt(var_py)

    # Correlations
    corr_px_theta = P[:, 0, 2]
    corr_py_theta = P[:, 1, 2]

    # GPS update times (where update_pinged is True)
    gps_update_times = timestamps[update_pinged]

    # -----------------------------
    # Plotting
    # -----------------------------
    plt.figure(figsize=(12, 10))

    # 1. Position Estimate with Uncertainty (±1σ envelope)
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, px, label='Estimated px', color='tab:blue')
    plt.fill_between(timestamps, px - std_px, px + std_px, alpha=0.3, color='tab:blue', label='±1σ px')
    plt.plot(timestamps, py, label='Estimated py', color='tab:orange')
    plt.fill_between(timestamps, py - std_py, py + std_py, alpha=0.3, color='tab:orange', label='±1σ py')

    # Add vertical lines for GPS updates
    for t in gps_update_times:
        plt.axvline(x=t, color='red', linestyle='--', alpha=0.6, linewidth=1)

    plt.xlabel('Time')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.title('Position Estimate with Uncertainty (±1σ) and GPS Updates (red dashed)')
    plt.grid(True)

    # 2. State Variances
    plt.subplot(3, 1, 2)
    plt.plot(timestamps, var_px, label='Variance px')
    plt.plot(timestamps, var_py, label='Variance py')
    plt.plot(timestamps, var_theta, label='Variance θ (yaw)')
    plt.plot(timestamps, var_v, label='Variance v')
    plt.plot(timestamps, var_omega, label='Variance ω')

    # GPS update lines
    for t in gps_update_times:
        plt.axvline(x=t, color='red', linestyle='--', alpha=0.6, linewidth=1)

    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel('Variance')
    plt.legend()
    plt.title('State Variances (diagonal of P) with GPS Updates')
    plt.grid(True)

    # 3. Position-Yaw Correlations
    plt.subplot(3, 1, 3)
    plt.plot(timestamps, corr_px_theta, label='Cov(px, θ)')
    plt.plot(timestamps, corr_py_theta, label='Cov(py, θ)')

    # GPS update lines
    for t in gps_update_times:
        plt.axvline(x=t, color='red', linestyle='--', alpha=0.6, linewidth=1)

    plt.xlabel('Time')
    plt.ylabel('Covariance')
    plt.legend()
    plt.title('Position-Yaw Correlations with GPS Updates')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def transform_robot_gps(x_r, y_r):
    """
    Convert robot-frame coordinates to GPS coordinates using ENU projection
    stored in self.corrector and yaw from self.corrector.bias_main.
    """
    theta = math.radians(ekf.corrector.bias_main)

    # Rotate robot frame to global ENU
    x_g = x_r * math.cos(theta) - y_r * math.sin(theta)
    y_g = x_r * math.sin(theta) + y_r * math.cos(theta)

    # Transform ENU -> GPS
    
    # print("Heyy")
    lon_out, lat_out = enu2gps.transform(x_g, y_g)
    return lat_out, lon_out


# # Convert EKF results to GPS and save to CSV
# csv_data = []
# enu2gps = Transformer.from_proj(ekf.corrector.proj_enu, "epsg:4326", always_xy=True)
# for row in results:
#     ts = row[0]
#     px, py = row[1], row[2]  # robot frame positions
#     lat, lon = transform_robot_gps(px - ekf.pos_offset[0], py - ekf.pos_offset[1])
#     csv_data.append([ts, lat, lon])

# # Save GPS trajectory
# csv_filename = '../TestData/Testing/ekf_gps_trajectory_nov25.csv'
# pd.DataFrame(csv_data, columns=["timestamp", "latitude", "longitude"]).to_csv(csv_filename, index=False)
# print(f"\nDone! {len(csv_data)} GPS positions saved to {csv_filename}")



    
