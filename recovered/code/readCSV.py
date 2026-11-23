# Visualize Data

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

## Makes life easier
study_plots = True
dwag_path = False
gmaps = False
plot_lat = True

def simplify_columns(df):
    """
    Simplify column names in a pandas DataFrame by removing common prefixes.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with long column names.
    
    Returns:
        pd.DataFrame: DataFrame with simplified column names.
    """
    def simplify_col(name):
        # Replace known prefixes with shorter versions
        name = name.replace('test_robot_Group20_imu_state_', 'imu_')
        name = name.replace('test_robot_Group20_odometry_', 'odom_')
        name = name.replace('waveshare_gps_', 'gps_')
        return name

    # Apply simplification to all columns
    df = df.copy()  # avoid modifying original DataFrame
    df.columns = [simplify_col(col) for col in df.columns]
    # Normalize to start at 0
    # df["ros_timestamp_ns"] = df["ros_timestamp_ns"] - df["ros_timestamp_ns"].iloc[0] * 1e-9

    # Normalize to [0, 1]
    df["ros_timestamp_ns"] = (df["ros_timestamp_ns"] - df["ros_timestamp_ns"].min()) / \
                            (df["ros_timestamp_ns"].max() - df["ros_timestamp_ns"].min())

    # Print new column names
    print("New column names:")
    for col in df.columns:
        print(col)
    return df

if __name__ == "__main__":
    # Replace 'your_file.csv' with the path to your CSV file
    # file_path = '../TestData/Bag_14_55/Bag_14_55.csv'
    file_path = '../TestData/Bag_Nov_25/Nov25_Test/rosbag2_2025_11_25-18_24_31/merged_data.csv'

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Display the DataFrame
    # print(df)

    # print(df.columns)

    # Optional: display first 5 rows neatly
    print(df.shape)


    # # Check number of NaN values in a column, e.g., 'Age'
    IMU_nan_count = df['test_robot_Group20_imu_state_orient_x'].isna().sum()

    print(f"Number of NaN values in 'IMU': {IMU_nan_count}")

    # # Check number of NaN values in a column, e.g., 'Age'
    gps_nan_count = df['waveshare_gps_fix_lon'].isna().sum()

    print(f"Number of NaN values in 'gps': {gps_nan_count}")

    # # Check number of NaN values in a column, e.g., 'Age'
    odo_nan_count = df['test_robot_Group20_odometry_pos_x'].isna().sum()

    print(f"Number of NaN values in 'Odometry': {odo_nan_count}")
    # print(df.iloc[1])

    df_simplified = simplify_columns(df)

    # Example: assume df_simplified is already your DataFrame with simplified column names

    # 1. Rows where both GPS and Odometry are available
    gps_odo_rows = df_simplified.dropna(subset=['gps_fix_lat', 'gps_fix_lon', 'odom_pos_x', 'odom_pos_y']).index
    print("Rows with GPS and Odometry:", gps_odo_rows.tolist())
    print("Count:", len(gps_odo_rows))

    # 2. Rows where both Odometry and IMU are available
    odo_imu_rows = df_simplified.dropna(subset=['odom_pos_x', 'odom_pos_y', 
                                                'imu_orient_x', 'imu_orient_y', 'imu_orient_z', 'imu_orient_w']).index
    print("Rows with Odometry and IMU:", odo_imu_rows.tolist())
    print("Count:", len(odo_imu_rows))

    # 3. Rows where both GPS and IMU are available
    gps_imu_rows = df_simplified.dropna(subset=['gps_fix_lat', 'gps_fix_lon', 
                                                'imu_orient_x', 'imu_orient_y', 'imu_orient_z', 'imu_orient_w']).index
    print("Rows with GPS and IMU:", gps_imu_rows.tolist())
    print("Count:", len(gps_imu_rows))




    if study_plots:
        # Remove NaNs for all IMU + GPS + Odom columns you plot
        df_clean_imu = df_simplified.dropna(
            subset=[
                'ros_timestamp_ns',
                # IMU
                'imu_orient_x','imu_orient_y','imu_orient_z','imu_orient_w',
                'imu_ang_vel_x','imu_ang_vel_y','imu_ang_vel_z',
                'imu_lin_acc_x','imu_lin_acc_y','imu_lin_acc_z'
            ]
        )

        df_clean_gps = df_simplified.dropna(
        subset=[
            'ros_timestamp_ns',
            # GPS
            'gps_fix_lat','gps_fix_lon','gps_fix_alt','gps_fix_status',
            'gps_speed_x','gps_speed_y','gps_speed_z',
            'gps_heading_deg_data'
        ]
    )
        df_clean_odo = df_simplified.dropna(
        subset=[
            'ros_timestamp_ns',
            # Odometry
            'odom_pos_x','odom_pos_y','odom_pos_z',
            'odom_vel_linear_x','odom_vel_linear_y','odom_vel_linear_z',
            'odom_vel_angular_x','odom_vel_angular_y','odom_vel_angular_z',
            'odom_quat_x','odom_quat_y','odom_quat_z','odom_quat_w'
        ]
    )
        from scipy.spatial.transform import Rotation as R

        # Convert quaternion columns to roll, pitch, yaw
        quats = df_clean_imu[['imu_orient_x', 'imu_orient_y', 'imu_orient_z', 'imu_orient_w']].values
        rpy = R.from_quat(quats).as_euler('xyz', degrees=True)  # roll=x, pitch=y, yaw=z

        df_clean_imu['roll']  = rpy[:, 0]
        df_clean_imu['pitch'] = rpy[:, 1]
        yaw_raw = rpy[:, 2]
        yaw_unwrapped = np.unwrap(np.deg2rad(yaw_raw))  # removes ±180 jumps
        df_clean_imu['yaw'] = np.mod(np.rad2deg(yaw_unwrapped), 360)

        # Convert quaternion columns to roll, pitch, yaw
        quats = df_clean_odo[['odom_quat_x', 'odom_quat_y', 'odom_quat_z', 'odom_quat_w']].values
        rpy = R.from_quat(quats).as_euler('xyz', degrees=True)  # roll=x, pitch=y, yaw=z

        df_clean_odo['roll']  = rpy[:, 0]
        df_clean_odo['pitch'] = rpy[:, 1]
        yaw_raw = rpy[:, 2]
        yaw_unwrapped = np.unwrap(np.deg2rad(yaw_raw))  # removes ±180 jumps
        df_clean_odo['yaw'] = np.mod(np.rad2deg(yaw_unwrapped), 360)

        # -----------------------------
        # 1. IMU Plots + RPY
        # -----------------------------

        fig, axes = plt.subplots(4, 1, figsize=(10, 14))
        fig.suptitle("IMU Data", fontsize=16)

        # 1. Roll/Pitch/Yaw
        axes[0].plot(df_clean_imu['ros_timestamp_ns'], df_clean_imu['roll'], label='roll')
        axes[0].plot(df_clean_imu['ros_timestamp_ns'], df_clean_imu['pitch'], label='pitch')
        axes[0].plot(df_clean_imu['ros_timestamp_ns'], df_clean_imu['yaw'], label='yaw')
        axes[0].set_ylabel('RPY (rad)')
        axes[0].legend()

        # 2. Angular velocity
        axes[1].plot(df_clean_imu['ros_timestamp_ns'], df_clean_imu['imu_ang_vel_x'], label='x')
        axes[1].plot(df_clean_imu['ros_timestamp_ns'], df_clean_imu['imu_ang_vel_y'], label='y')
        axes[1].plot(df_clean_imu['ros_timestamp_ns'], df_clean_imu['imu_ang_vel_z'], label='z')
        axes[1].set_ylabel('Angular Velocity')
        axes[1].legend()

        # 3. Linear acceleration
        axes[2].plot(df_clean_imu['ros_timestamp_ns'], df_clean_imu['imu_lin_acc_x'], label='x')
        axes[2].plot(df_clean_imu['ros_timestamp_ns'], df_clean_imu['imu_lin_acc_y'], label='y')
        axes[2].plot(df_clean_imu['ros_timestamp_ns'], df_clean_imu['imu_lin_acc_z'], label='z')
        axes[2].set_ylabel('Linear Acceleration')
        axes[2].legend()

        # 4. Original quaternion components (optional)
        axes[3].plot(df_clean_imu['ros_timestamp_ns'], df_clean_imu['imu_orient_x'], label='qx')
        axes[3].plot(df_clean_imu['ros_timestamp_ns'], df_clean_imu['imu_orient_y'], label='qy')
        axes[3].plot(df_clean_imu['ros_timestamp_ns'], df_clean_imu['imu_orient_z'], label='qz')
        axes[3].plot(df_clean_imu['ros_timestamp_ns'], df_clean_imu['imu_orient_w'], label='qw')
        axes[3].set_ylabel('Quat')
        axes[3].set_xlabel('Time')
        axes[3].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()



        # -----------------------------
        # 2. GPS Plots 
        # -----------------------------

        fig, axes = plt.subplots(4, 1, figsize=(10, 12))
        fig.suptitle("GPS Fix Data", fontsize=16)

        axes[0].plot(df_clean_gps['ros_timestamp_ns'], df_clean_gps['gps_fix_lat'], label='lat', linestyle='--')
        axes[0].set_ylabel('Latitude')

        axes[1].plot(df_clean_gps['ros_timestamp_ns'], df_clean_gps['gps_fix_lon'], label='lon')
        axes[1].set_ylabel('Longitude')

        axes[2].plot(df_clean_gps['ros_timestamp_ns'], df_clean_gps['gps_fix_alt'], label='alt')
        axes[2].set_ylabel('Altitude')

        axes[3].plot(df_clean_gps['ros_timestamp_ns'], df_clean_gps['gps_fix_status'], label='status')
        axes[3].set_ylabel('Status')
        axes[3].set_xlabel('Time')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        # GPS Speed
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        fig.suptitle("GPS Speed", fontsize=16)
        axes[0].plot(df_clean_gps['ros_timestamp_ns'], df_clean_gps['gps_speed_x'], label='x')
        axes[0].set_ylabel('Speed X')

        axes[1].plot(df_clean_gps['ros_timestamp_ns'], df_clean_gps['gps_speed_y'], label='y')
        axes[1].set_ylabel('Speed Y')

        axes[2].plot(df_clean_gps['ros_timestamp_ns'], df_clean_gps['gps_speed_z'], label='z')
        axes[2].set_ylabel('Speed Z')
        axes[2].set_xlabel('Time')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        # GPS heading
        plt.figure(figsize=(10, 4))
        plt.plot(df_clean_gps['ros_timestamp_ns'], df_clean_gps['gps_heading_deg_data'])
        plt.title('GPS Heading (deg)')
        plt.xlabel('Time')
        plt.ylabel('Heading')
        plt.tight_layout()
        plt.show()


        # -----------------------------
        # 3. Odometry Plots
        # -----------------------------

        # Position
        fig, axes = plt.subplots(4, 1, figsize=(10, 10))
        # 1. Roll/Pitch/Yaw
        axes[0].plot(df_clean_odo['ros_timestamp_ns'], df_clean_odo['roll'], label='roll')
        axes[0].plot(df_clean_odo['ros_timestamp_ns'], df_clean_odo['pitch'], label='pitch')
        axes[0].plot(df_clean_odo['ros_timestamp_ns'], df_clean_odo['yaw'], label='yaw')
        axes[0].set_ylabel('RPY (rad)')
        axes[0].legend()
        fig.suptitle("Odometry Position", fontsize=16)
        axes[0].plot(df_clean_odo['ros_timestamp_ns'], df_clean_odo['odom_pos_x'], label='x')
        axes[0].set_ylabel('Pos X')

        axes[1].plot(df_clean_odo['ros_timestamp_ns'], df_clean_odo['odom_pos_y'], label='y')
        axes[1].set_ylabel('Pos Y')

        axes[2].plot(df_clean_odo['ros_timestamp_ns'], df_clean_odo['odom_pos_z'], label='z')
        axes[2].set_ylabel('Pos Z')
        axes[2].set_xlabel('Time')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        # Linear velocity
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        fig.suptitle("Odometry Linear Velocity", fontsize=16)

        axes[0].plot(df_clean_odo['ros_timestamp_ns'], df_clean_odo['odom_vel_linear_x'], label='x')
        axes[0].set_ylabel('Linear Vel X')

        axes[1].plot(df_clean_odo['ros_timestamp_ns'], df_clean_odo['odom_vel_linear_y'], label='y')
        axes[1].set_ylabel('Linear Vel Y')

        axes[2].plot(df_clean_odo['ros_timestamp_ns'], df_clean_odo['odom_vel_linear_z'], label='z')
        axes[2].set_ylabel('Linear Vel Z')
        axes[2].set_xlabel('Time')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        # Angular velocity
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        fig.suptitle("Odometry Angular Velocity", fontsize=16)

        axes[0].plot(df_clean_odo['ros_timestamp_ns'], df_clean_odo['odom_vel_angular_x'], label='x')
        axes[0].set_ylabel('Angular Vel X')

        axes[1].plot(df_clean_odo['ros_timestamp_ns'], df_clean_odo['odom_vel_angular_y'], label='y')
        axes[1].set_ylabel('Angular Vel Y')

        axes[2].plot(df_clean_odo['ros_timestamp_ns'], df_clean_odo['odom_vel_angular_z'], label='z')
        axes[2].set_ylabel('Angular Vel Z')
        axes[2].set_xlabel('Time')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        # Quaternions
        fig, axes = plt.subplots(4, 1, figsize=(10, 12))
        fig.suptitle("Odometry Quaternions", fontsize=16)

        axes[0].plot(df_clean_odo['ros_timestamp_ns'], df_clean_odo['odom_quat_x'], label='x')
        axes[0].set_ylabel('Quat X')

        axes[1].plot(df_clean_odo['ros_timestamp_ns'], df_clean_odo['odom_quat_y'], label='y')
        axes[1].set_ylabel('Quat Y')

        axes[2].plot(df_clean_odo['ros_timestamp_ns'], df_clean_odo['odom_quat_z'], label='z')
        axes[2].set_ylabel('Quat Z')

        axes[3].plot(df_clean_odo['ros_timestamp_ns'], df_clean_odo['odom_quat_w'], label='w')
        axes[3].set_ylabel('Quat W')
        axes[3].set_xlabel('Time')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


    if dwag_path:
        ## Path taken by dawg based on odometry
        # Plot X vs Y to see the path
        plt.figure(figsize=(8, 8))
        plt.plot(df_simplified['odom_pos_x'], df_simplified['odom_pos_y'], linestyle='-', marker=None, color='blue')
        plt.scatter(df_simplified['odom_pos_x'].iloc[0], df_simplified['odom_pos_y'].iloc[0], color='green', label='Start')
        plt.scatter(df_simplified['odom_pos_x'].iloc[-1], df_simplified['odom_pos_y'].iloc[-1], color='red', label='End')
        plt.title('Odometry Path')
        plt.xlabel('Position X')
        plt.ylabel('Position Y')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')  # Keeps X and Y scales equal for accurate trajectory
        plt.show()

        x = df_simplified['odom_pos_x'].values
        y = df_simplified['odom_pos_y'].values

        # Mask out NaNs
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_clean = x[mask]
        y_clean = y[mask]

        # Estimate center
        x_center = np.mean(x_clean)
        y_center = np.mean(y_clean)

        # Compute distances to the center using cleaned arrays
        distances = np.sqrt((x_clean - x_center)**2 + (y_clean - y_center)**2)

        # Estimate radius as average distance
        radius_est = np.mean(distances)
        print(f"Estimated radius: {radius_est:.3f}")



        df_clean = df_simplified.dropna(subset=['gps_fix_lon', 'gps_fix_lat']).copy()
        # --- Corrected Plotting Using Cleaned (Raw) Data ---

        plt.figure(figsize=(8, 8)) # Create a new figure instance

        # Plot 1: The entire raw path
        plt.plot(
            df_clean['gps_fix_lon'],
            df_clean['gps_fix_lat'],
            label='GPS Path',
            color='gray',
            linestyle='-',
            linewidth=5.5
        )

        # 2. Mark the starting point (first row of the CLEANED data)
        plt.scatter(
            df_clean['gps_fix_lon'].iloc[0],
            df_clean['gps_fix_lat'].iloc[0],
            color='green',
            marker='o',
            s=100,
            label='Start Point'
        )

        # 3. Mark the ending point (last row of the CLEANED data)
        plt.scatter(
            df_clean['gps_fix_lon'].iloc[-1],
            df_clean['gps_fix_lat'].iloc[-1],
            color='red',
            marker='X',
            s=150,
            label='End Point'
        )

        # Set labels for the raw data
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('GPS Trajectory (Non-NaN Values Only)')

        plt.legend()
        plt.grid(True)
        plt.axis('equal') # Highly recommended for raw geographic coordinates

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        if gmaps:
            # Rename latitude/longitude columns for Google Maps
            # Replace 'lat' and 'lon' with your actual column names if different
            df_gmaps = df_clean.rename(columns={
                "gps_fix_lat": "latitude",
                "gps_fix_lon": "longitude"
            })

            # Keep only required columns
            df_gmaps = df_gmaps[["latitude", "longitude"]]

            # Add label column
            df_gmaps["label"] = ""

            # Set first and last labels
            if len(df_gmaps) >= 1:
                df_gmaps.loc[df_gmaps.index[0], "label"] = "start"
                df_gmaps.loc[df_gmaps.index[-1], "label"] = "end"

            # Create new filename based on original file
            base_dir = os.path.dirname(file_path)
            base_name = os.path.basename(file_path)
            name_part = os.path.splitext(base_name)[0]  # e.g., Bag_41_46
            new_name = f"gmap_{name_part}.csv"
            save_path = os.path.join(base_dir, new_name)

            # Save CSV
            df_gmaps.to_csv(save_path, index=False)
            print(f"Saved Google Maps CSV at: {save_path}")
        if plot_lat:
            # Take first GPS as reference
            lat0 = df_clean['gps_fix_lat'].iloc[0]
            lon0 = df_clean['gps_fix_lon'].iloc[0]
            lat0_rad = np.radians(lat0)
            # print(df_clean.head)
            # Convert odom (x, y in meters) to latitude/longitude
            df_clean['lat_odom'] = lat0 + (df_clean['odom_pos_y'] / 111132.92)
            df_clean['lon_odom'] = lon0 + (df_clean['odom_pos_x'] / (111319.5 * np.cos(lat0_rad)))
            print(df_clean[['odom_pos_y', 'odom_pos_x']].head(10))
            plt.figure(figsize=(10,5))

            # Plot latitude
            plt.subplot(2,1,1)
            plt.plot(df_clean['ros_timestamp_ns'], df_clean['gps_fix_lat'], label='GPS Latitude', marker='o', markersize=3)
            plt.plot(df_clean['ros_timestamp_ns'], df_clean['lat_odom'], label='Odom-derived Latitude', marker='x', markersize=8)
            plt.ylabel('Latitude (deg)')
            plt.legend()
            plt.grid(True)

            # Plot longitude
            plt.subplot(2,1,2)
            plt.plot(df_clean['ros_timestamp_ns'], df_clean['gps_fix_lon'], label='GPS Longitude', marker='o', markersize=8)
            plt.plot(df_clean['ros_timestamp_ns'], df_clean['lon_odom'], label='Odom-derived Longitude', marker='x', markersize=3)
            plt.ylabel('Longitude (deg)')
            plt.xlabel('Time (s)')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()

