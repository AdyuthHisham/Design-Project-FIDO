'''
Function to correct heading of FIDO while in straight line
Notes:
1. FIDO IMU is 180 deg inverted / I converted from Global to Robot frame
ToDo:
1. Need to check cases like 
1.1. Is Dawg moving? - Done
1.2. Is Dawg turning?
2. Put helper functions into class
3. Optimize polyfit
4. Profile code
5. Can tinker INIT
'''
import pandas as pd
from readCSV import simplify_columns
from pyproj import Proj, Transformer
from scipy.spatial.transform import Rotation as R
import numpy as np
from collections import deque
from typing import Tuple, Optional, List

import pdb


del_North = []
del_East = []

def global_to_body(self, x_global: float, y_global: float) -> tuple[float, float]:
    """
    Convert a point from global (map/ENU) frame → current robot body frame.
    Returns: (x_body, y_body) where:
        x_body > 0 : point is in front of robot
        y_body > 0 : point is to the right of robot
    """
    px = self.x[0]      # robot x in global
    py = self.x[1]      # robot y in global
    psi = self.bias_main     # robot yaw (global → body rotation angle)

    dx = x_global - px
    dy = y_global - py

    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    x_body =  dx * cos_psi + dy * sin_psi
    y_body = -dx * sin_psi + dy * cos_psi

    return x_body, y_body

class ContinuousYawExtractor:
    """
    Converts quaternion → yaw in degrees [0, 360) with perfect continuity.
    Eliminates ±180° jumps forever — even during full 360° turns.
    """
    def __init__(self):
        self.prev_yaw_rad = None  # Keeps track of previous unwrapped yaw in radians

    def quat_to_yaw_rad(self, qx: float, qy: float, qz: float, qw: float) -> float:
        """
        Input: quaternion as (x, y, z, w) — any order as long as consistent
        Output: continuous yaw in Radians [0, 2pi) — no jumps!
        """
        # Convert to Rotation object (order: [x, y, z, w])
        rot = R.from_quat([qx, qy, qz, qw])
        
        # Get yaw in radians, in [-pi, pi]
        yaw_rad = rot.as_euler('xyz', degrees=False)[2]  # yaw = z-axis rotation

        # First call: initialize
        if self.prev_yaw_rad is None:
            self.prev_yaw_rad = yaw_rad
            # return (np.degrees(yaw_rad) + 360) % 360
            return yaw_rad

        # Unwrap: detect and correct jumps larger than 180° (pi radians)
        # This handles crossing the ±180° boundary smoothly
        diff = yaw_rad - self.prev_yaw_rad
        if diff > np.pi:
            yaw_rad -= 2 * np.pi
        elif diff < -np.pi:
            yaw_rad += 2 * np.pi

        # Update history
        self.prev_yaw_rad = yaw_rad
        return yaw_rad

    def reset(self):
        """Call this if robot is repositioned or filter restarts"""
        self.prev_yaw_rad = None

def custom_arctan2(dy: float, dx: float, north_zero: bool = False) -> float:
    """
    Compute the angle of a vector (dx, dy) in degrees in the range [0, 360).

    Parameters
    ----------
    dy : float
        Vertical component (northing)
    dx : float
        Horizontal component (easting)
    north_zero : bool, default False
        If True, returns compass bearing: 0° = North, clockwise.
        If False, returns angle from positive x-axis (East) counter-clockwise.

    Returns
    -------
    angle : float
        Angle in degrees in range [0, 360)
    """
    if dx == 0 and dy == 0:
        return 0.0

    # theta = np.degrees(np.arctan2(dy, dx))  # angle from x-axis
    # angle = theta % 360.0                   # wrap to [0, 360)
    theta = np.arctan2(dy, dx)      # radians, from x-axis
    angle = theta % (2 * np.pi)     # wrap to [0, 2*pi)

    if north_zero:
        # Convert to compass bearing: 0° = North, clockwise
        angle = (90.0 - angle) % 360.0

    return angle

def compute_r_squared(points: List[Tuple[float, float]]) -> float:
    """
    Compute coefficient of determination R² for linear regression on points.
    """
    if len(points) < 2:
        return 0.0
    
    x = np.array([p[0] for p in points])  # East
    y = np.array([p[1] for p in points])  # North
    
    # Linear regression: y = mx + c  (or x = my + c depending on direction)
    # We'll fit in the dominant direction later, but for R² we need consistent model
    # Standard R² for best-fit line (we use total least squares approximation via numpy)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    y_pred = m * x + c
    
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

def estimate_heading_from_gps_trajectory(
    bias: float,
    gps_buffer: deque,
    min_points: int = 10,
    r2_threshold: float = 0.9 
) -> Tuple[Optional[float], Optional[float], bool]:
    """
    Estimate true heading from recent GPS points using linear regression.
    
    Returns:
        estimated_heading_deg (0-360), 
        r2_value, 
        is_valid_correction (whether criteria are met)
    """
    if len(gps_buffer) < min_points:
        return None, None, False

    
    points = list(gps_buffer)
    easts = np.array([p[0] for p in points])
    norths = np.array([p[1] for p in points])

    delta_e = easts[-1] - easts[0]
    delta_n = norths[-1] - norths[0]


    abs_de = abs(delta_e)
    abs_dn = abs(delta_n)
    if abs_de >= abs_dn:
        # Fit North = f(East)
        slope, intercept = np.polyfit(easts, norths, 1)
        north_pred = slope * easts + intercept
        delta_n_pred = slope * delta_e
        delta_e_pred = delta_e
    else:
        # Fit East = f(North)
        slope, intercept = np.polyfit(norths, easts, 1)
        east_pred = slope * norths + intercept
        delta_e_pred = slope * delta_n
        delta_n_pred = delta_n

    # Compute R²
    if abs_de >= abs_dn:
        ss_res = np.sum((norths - (slope * easts + intercept)) ** 2)
        ss_tot = np.sum((norths - np.mean(norths)) ** 2)
    else:
        ss_res = np.sum((easts - (slope * norths + intercept)) ** 2)
        ss_tot = np.sum((easts - np.mean(easts)) ** 2)
    
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-8 else 0.0

    is_valid = r2 > r2_threshold

    heading_act = (custom_arctan2(delta_n_pred, delta_e_pred))
    
    if not is_valid:
        return heading_act, r2, delta_e, delta_n, False
    # Estimate actual heading using predicted deltas

    return heading_act, r2, delta_e, delta_n, True

class GPSHeadingBiasCorrector:
    def __init__(
        self,
        buffer_size: int = 10,
        r2_threshold: float = 0.87
    ):
        self.buffer = deque(maxlen=buffer_size)
        self.bias = 0.0                    # current estimated bias b_k
        self.bias_main = None
        self.first_update = True           # force first bias update
        self.r2_threshold = r2_threshold
        self.origin_transformer = None
        self.proj_enu = None

    def wrap_angle_2pi(self,angle):
        """Wrap angle in radians to [0, 2π]."""
        return angle % (2 * np.pi)

    def update(
        self,
        lat: float,
        lon: float,
        measured_heading_deg: float
    ) -> Tuple[float, float, bool]:
        """
        Update bias and return corrected heading.

        Args:
            latitude, longitude: current GPS position (meters)
            measured_heading_deg: raw heading from IMU/GPS (0-360)

        Returns:
            corrected_heading_deg, current_bias, bias_updated (bool)
        """
        # Normalize heading to [0, 360)
        # measured_heading_deg = measured_heading_deg % 360 # Redundant IG

        if self.first_update:
            # ENU projection centered at origin
            self.proj_enu = Proj(proj="aeqd", lat_0=lat, lon_0=lon, datum="WGS84")
            self.origin_transformer = Transformer.from_proj("epsg:4326", self.proj_enu, always_xy=True)
            self.first_update = False

        # Add new point
        east, north = self.origin_transformer.transform(lon, lat)  # Prone to problem
        self.buffer.append((east, north))


        bias_updated = False
        estimated_act_heading = None
        delta_e = None
        delta_n = None
        if len(self.buffer) >= self.buffer.maxlen:
            estimated_act_heading, r2,  delta_e, delta_n, is_valid, = estimate_heading_from_gps_trajectory(
                self.bias,
                self.buffer,
                min_points=self.buffer.maxlen,
                r2_threshold=self.r2_threshold
            )

            if (is_valid):
                if self.bias_main is None:
                    self.bias_main = self.wrap_angle_2pi(measured_heading_deg - estimated_act_heading)
                    # self.bias = self.bias_main
                    print(self.bias_main)
                    bias_updated = True
                    # pdb.set_trace()
                if estimated_act_heading is not None: # Why did I put this; I mean if is_valid then works??

                    # Update bias: b_{k+1} = b_k + (measured - actual)
                    self.bias = self.wrap_angle_2pi(measured_heading_deg - (estimated_act_heading + self.bias_main))
                    # print(self.bias)
                    # self.bias = self.bias %  360  # keep in reasonable range
                    bias_updated = True
                
            # self.buffer.clear()

        # Corrected heading
        # corrected_heading = (measured_heading_deg - self.bias) % 360
        # Has self.bias_main earlier to move
        corrected_heading = self.wrap_angle_2pi(measured_heading_deg - self.bias)

        global del_East, del_North
        del_East.append(delta_e)

        del_North.append(delta_n)

        return corrected_heading, self.bias, bias_updated, estimated_act_heading, east, north


if __name__ == "__main__":
    df = simplify_columns(pd.read_csv('/home/adhish/Productivity/FIDO/SSY226SensorFusion/programmeringstestsensorfusion/TestData/Bag_Nov_25/Nov25_Test/rosbag2_2025_11_25-18_20_45/merged_data.csv'))
    df = df.sort_values('ros_timestamp_ns').reset_index(drop=True)

    # Buffers of size 1
    imu_buffer = deque(maxlen=1)    # (qx, qy, qz, qw, timestamp)
    gps_buffer = deque(maxlen=1)    # (lat, lon, timestamp)

    yaw_extractor = ContinuousYawExtractor()
    corrector = GPSHeadingBiasCorrector(buffer_size=3, r2_threshold=0.83)

    results = []

    print("Processing...")
    # New list to store yaw for each row (None when yaw cannot be computed)
    yaw_list = []

    # Iterate row-by-row
    for _, row in df.iterrows():
        ts = row['ros_timestamp_ns']

        # 1. Store IMU quaternion if valid
        if pd.notna(row['imu_orient_x']):
            imu_buffer.append((
                row['imu_orient_x'],
                row['imu_orient_y'],
                row['imu_orient_z'],
                row['imu_orient_w'],
                ts
            ))

        # 2. Store GPS fix if valid
        if pd.notna(row['gps_fix_lat']) and pd.notna(row['gps_fix_lon']):
            gps_buffer.append((row['gps_fix_lat'], row['gps_fix_lon'], ts))

        # 3. If velocity is too low, pop from corrector buffer
        threshold = 1e-4
        if abs(row['odom_vel_linear_x']) < threshold and abs(row['odom_vel_linear_y']) < threshold:
            if corrector.buffer:
                corrector.buffer.pop()

        # Default yaw per row (None unless computed below)
        current_yaw = None

        # 4. Only compute yaw when BOTH buffers have data
        if imu_buffer and gps_buffer:
            # FIFO pop oldest IMU quaternion
            qx, qy, qz, qw, _ts = imu_buffer.pop()

            # Pop current GPS (you may want FIFO here too)
            lat, lon, _ = gps_buffer.pop()

            # Compute yaw in degrees
            current_yaw = yaw_extractor.quat_to_yaw_rad(qx, qy, qz, qw)

        # Append current yaw (None if not computed this row)
        yaw_list.append(current_yaw)


    for _, row in df.iterrows():
        ts = row['ros_timestamp_ns']

        # === 1. Store latest IMU if valid ===
        if pd.notna(row['imu_orient_x']):
            imu_buffer.append((
                row['imu_orient_x'],
                row['imu_orient_y'],
                row['imu_orient_z'],
                row['imu_orient_w'],
                ts
            ))

        # === 2. Store latest GPS if valid ===
        if pd.notna(row['gps_fix_lat']) and pd.notna(row['gps_fix_lon']):
            gps_buffer.append((row['gps_fix_lat'], row['gps_fix_lon'], ts))

        # === 0. Dont bother if Odo velocity is less than 0 ===
        threshold = 1e-4  # adjust based on sensor noise
        if abs(row['odom_vel_linear_x']) < threshold and abs(row['odom_vel_linear_y']) < threshold:
            # Process this point — velocity is high enough
            n = 1  # number of pops
            for _ in range(n):
                if corrector.buffer:  # check deque is not empty
                    corrector.buffer.pop()

        # === 3. Call function only when both buffers full ===
        if imu_buffer and gps_buffer:
            # POP THE OLDEST IMU (FIFO) — this keeps buffers synced over time
            qx, qy, qz, qw, ts = imu_buffer.pop()  # <-- CORRECT pop

            # POP THE CURRENT GPS
            lat, lon, _ = gps_buffer.pop()  # <-- CORRECT pop
            # Get yaw from IMU quaternion
            yaw_deg = yaw_extractor.quat_to_yaw_rad(qx, qy, qz, qw)

            # Correct heading
            corrected, bias, updated, est_heading, east, north = corrector.update(lat, lon, yaw_deg)
            delta_e = del_East.pop()

            delta_n = del_North.pop()
            results.append({
                'timestamp': ts,
                'raw_yaw': yaw_deg,
                'corrected_yaw': corrected,
                'estimated_yaw' : est_heading,
                'bias': bias,
                'bias_updated': updated,
                'gps_lat': lat,
                'gps_lon': lon,
                'east': east,
                'north': north,
                'del_e':delta_e,
                'del_n':delta_n,
            })


    # ================== CONVERT TO DATAFRAME ==================
    result_df = pd.DataFrame(results)
    print(result_df.head)
    import matplotlib.pyplot as plt
    if result_df.empty:
        print("No valid data processed!")
    else:
        print(f"Processed {len(result_df)} points. Plotting...")

        timestamp = result_df['timestamp']

        # ================== PLOT 1: Yaw comparison ==================
        plt.figure(figsize=(12, 6))
        plt.plot(timestamp, result_df['raw_yaw'], 'r-', label='Raw IMU Yaw', alpha=0.7)
        plt.plot(timestamp, result_df['corrected_yaw'], 'b-', label='Corrected Yaw', linewidth=2)
        plt.xlabel('timestamp [s]')
        plt.ylabel('Heading [°]')
        plt.title('Raw vs Corrected Heading (GPS Bias Correction)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # ================== PLOT 2: Bias evolution ==================
        plt.figure(figsize=(12, 4))
        plt.plot(timestamp, result_df['bias'], 'm.-', markersize=4)
        updates = result_df[result_df['bias_updated']]
        plt.plot(updates['timestamp'] , updates['bias'], 'co', label='Bias Update')
        plt.xlabel('timestamp [s]')
        plt.ylabel('Bias [°]')
        plt.title('Learned Heading Bias Over timestamp')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(timestamp, result_df['raw_yaw'], 'r-', label='Raw IMU', alpha=0.6)
        plt.plot(timestamp, result_df['corrected_yaw'], 'b-', linewidth=2, label='Corrected (Your Algo)')
        plt.plot(timestamp, result_df['estimated_yaw'], 'g--', linewidth=3, label='GPS Ground Truth')
        plt.legend()
        plt.title("Visualized Trajectory and Heading")
        plt.ylim(0, 360)
        plt.show()

        # 1. Extract timestamp-synchronized data
        # ---------------------------------------
        timestamps = result_df["timestamp"].values
        east       = result_df["east"].values
        north      = result_df["north"].values
        raw        = result_df["raw_yaw"].values
        corrected  = result_df["corrected_yaw"].values
        bias = result_df["bias"].values

        # 2. Choose arrow timestamps every N samples (or use fixed time spacing)
        # ----------------------------------------------------------------------
        N = 1  # sample every N timestamps

        arrow_ts = timestamps[::N]  # timestamps where arrows should be drawn

        # 3. Align ALL columns to arrow timestamps (nearest match)
        # --------------------------------------------------------
        # Use pandas.merge_asof for perfect nearest-timestamp alignment
        arrow_df = (
            result_df[["timestamp", "east", "north", "raw_yaw", "corrected_yaw","bias"]]
            .sort_values("timestamp")
        )

        # Build a mini DataFrame of desired timestamps
        arrow_times_df = (
            result_df[["timestamp"]].iloc[::N]
        )

        # Merge to nearest timestamp
        aligned = pd.merge_asof(
            arrow_times_df.sort_values("timestamp"),
            arrow_df,
            on="timestamp",
            direction="nearest"
        )

        # Extract aligned columns
        arrow_east      = aligned["east"].values
        arrow_north     = aligned["north"].values
        arrow_raw       = aligned["raw_yaw"].values
        arrow_corrected = aligned["corrected_yaw"].values

        # 4. Convert heading degrees → radians (and flip 180° if needed)
        # --------------------------------------------------------------
        arrow_raw_rad = arrow_raw - corrector.bias_main
        arrow_cor_rad = arrow_corrected - corrector.bias_main

        # 5. Plot trajectory with time-aligned arrows
        # --------------------------------------------
        plt.figure(figsize=(10, 10))

        # Trajectory
        plt.plot(east, north, "k-", linewidth=2, label="Trajectory")

        # Start & end markers
        plt.scatter(east[0], north[0], color="green", s=120, marker="o", label="Start")
        plt.scatter(east[-1], north[-1], color="red", s=120, marker="X", label="End")


        # Filter only the timestamps where bias was updated
        updates = result_df[result_df['bias_updated']]

        # Plot bias updates as points (timestamp vs bias)
        plt.scatter(updates['east'], updates['north'],  # or use timestamps if plotting vs time
                    color="orange", s=75, marker="o", label="Bias Update")
        # Straight dotted black line from start to end
        # plt.plot([east[0], east[-1]], [north[0], north[-1]], 'k--', linewidth=2, label="Direct Line")

        # Raw IMU arrows
        plt.quiver(
            arrow_east, arrow_north,
            np.cos(arrow_raw_rad), np.sin(arrow_raw_rad),
            color="red", scale=22, width=0.005, alpha=0.6,
            label="Raw Heading"
        )

        # # Corrected Yaw arrows
        plt.quiver(
            arrow_east, arrow_north,
            np.cos(arrow_cor_rad), np.sin(arrow_cor_rad),
            color="green", scale=22, width=0.008, alpha=0.8,
            label="Corrected Yaw"
        )

        # Formatting
        plt.xlabel("East (m)")
        plt.ylabel("North (m)")
        plt.title("Trajectory with Timestamp-Aligned Raw vs Corrected Heading in GLOBAL FRAME")
        plt.legend()
        plt.axis("equal")
        plt.grid(True)

        plt.savefig("my_plot.png")
        plt.show()

        del_east       = result_df["del_e"].values
        del_north      = result_df["del_n"].values
        # Create a figure with 3 subplots (vertically stacked)
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))

        # 1️⃣ East displacement over time
        axs[0].plot(timestamps, del_east, color='blue', label='East Δ')
        axs[0].set_title('East Displacement over Time')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Δ East [m]')
        axs[0].legend()
        axs[0].grid(True)

        # 2️⃣ North displacement over time
        axs[1].plot(timestamps, del_north, color='green', label='North Δ')
        axs[1].set_title('North Displacement over Time')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Δ North [m]')
        axs[1].legend()
        axs[1].grid(True)

        # 3️⃣ East vs North (trajectory)
        axs[2].plot(del_east, del_north, color='red', label='Trajectory')
        axs[2].set_title('Trajectory: East vs North')
        axs[2].set_xlabel('Δ East [m]')
        axs[2].set_ylabel('Δ North [m]')
        axs[2].legend()
        axs[2].grid(True)
        axs[2].axis('equal')  # keep aspect ratio square

        plt.tight_layout()
        plt.show()




    # Save results
    pd.DataFrame(results).to_csv('/home/adhish/Productivity/FIDO/SSY226SensorFusion/programmeringstestsensorfusion/TestData/Testing/corrected_heading_nov25.csv', index=False)
    print(f"\nDone! {len(results)} corrections saved to corrected_heading_nov25.csv")
