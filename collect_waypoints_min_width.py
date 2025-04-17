#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import atexit
from os.path import expanduser
from time import gmtime, strftime
from numpy import linalg as LA
from tf_transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import pandas as pd
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# Constants
MIN_SPACING = 0.01  # Minimum distance between consecutive waypoints in meters
home = expanduser('~')
filename = strftime(home + '/mo_ws/src/wp-%Y-%m-%d-%H-%M', gmtime()) + '.csv'
file = open(filename, 'w')

# Add the header row to the file
file.write('# x_m, y_m, w_tr_right_m, w_tr_left_m\n')

class WaypointsLogger(Node):
    def __init__(self):
        super().__init__('waypoints_logger')

        # Subscriptions
        self.subscription_odom = self.create_subscription(
            Odometry, '/pf/pose/odom', self.process_odometry, 10)
        self.subscription_scan = self.create_subscription(
            LaserScan, '/scan', self.process_scan, 10)

        # Initialize variables
        self.latest_scan = None
        self.latest_odometry = None
        self.left_width = float('inf')
        self.right_width = float('inf')
        self.previous_point = None
        self.waypoints = []

    def process_scan(self, scan_data):
        """Process LiDAR scan data to compute right and left track widths."""
        ranges = np.array(scan_data.ranges)
        angles = np.linspace(scan_data.angle_min, scan_data.angle_max, len(ranges))

        # Filter valid ranges (exclude inf/NaN values)
        valid_indices = ~np.isinf(ranges) & ~np.isnan(ranges)
        ranges = ranges[valid_indices]
        angles = angles[valid_indices]

        if len(ranges) == 0:
            self.get_logger().warn('No valid ranges in scan data.')
            return

        # Split ranges into left and right halves based on angles
        left_indices = angles > 0
        right_indices = angles <= 0

        self.left_width = np.min(ranges[left_indices]) if np.any(left_indices) else float('inf')
        self.right_width = np.min(ranges[right_indices]) if np.any(right_indices) else float('inf')

        self.get_logger().info(f'Left width: {self.left_width}, Right width: {self.right_width}')

        # Save the scan data
        self.latest_scan = True
        self.save_waypoint()

    def process_odometry(self, odometry_data):
        """Process odometry data and cache it."""
        self.latest_odometry = odometry_data
        self.save_waypoint()

    def save_waypoint(self):
        """Save waypoint data if both odometry and LiDAR data are available."""
        if self.latest_scan and self.latest_odometry:
            # Extract odometry data
            data = self.latest_odometry
            quaternion = np.array([data.pose.pose.orientation.x, 
                                   data.pose.pose.orientation.y, 
                                   data.pose.pose.orientation.z, 
                                   data.pose.pose.orientation.w])
            euler = euler_from_quaternion(quaternion)
            x, y = data.pose.pose.position.x, data.pose.pose.position.y

            # Ensure minimum spacing between waypoints
            if self.previous_point is None or LA.norm([x - self.previous_point[0], y - self.previous_point[1]]) >= MIN_SPACING:
                self.get_logger().info(f'Saving waypoint: x={x}, y={y}, left_width={self.left_width}, right_width={self.right_width}')
                self.waypoints.append((x, y, self.left_width, self.right_width))
                self.previous_point = (x, y)

            # Reset the flags to wait for new data
            self.latest_scan = None
            self.latest_odometry = None

    def filter_outliers(self, points, threshold=1.5):
        """Filter out waypoints that are too far from their neighbors."""
        filtered = [points[0]]
        for i in range(1, len(points)):
            if euclidean(points[i], filtered[-1]) < threshold:
                filtered.append(points[i])
        return np.array(filtered)

    def save_and_interpolate(self):
        """Save waypoints to CSV, perform spline interpolation, and plot the trajectory."""
        if len(self.waypoints) > 1:
            waypoints_array = np.array(self.waypoints)
            x, y = waypoints_array[:, 0], waypoints_array[:, 1]

            # Filter outliers
            waypoints_filtered = self.filter_outliers(waypoints_array[:, :2], threshold=2.0)
            x, y = waypoints_filtered[:, 0], waypoints_filtered[:, 1]

            # Perform spline interpolation with smoothing
            tck, u = splprep([x, y], s=0.2)
            unew = np.linspace(0, 1, 5000)
            x_new, y_new = splev(unew, tck)

            

            # Save interpolated waypoints
            for i in range(len(x_new)):
                file.write(f'{x_new[i]}, {y_new[i]}, {self.left_width}, {self.right_width}\n')

            self.get_logger().info('Waypoints filtered, smoothed, and saved.')
            
            # Log the length of x_new
            self.get_logger().info(f'Length of x_new: {len(x_new)}')

            # Plot the original, filtered, and interpolated trajectory
            plt.figure()
            plt.plot(waypoints_array[:, 0], waypoints_array[:, 1], 'o', label='Original Waypoints')
            plt.plot(waypoints_filtered[:, 0], waypoints_filtered[:, 1], 'x', label='Filtered Waypoints')
            plt.plot(x_new, y_new, '-', label='Smoothed Trajectory')
            plt.xlabel('X-coordinate (m)')
            plt.ylabel('Y-coordinate (m)')
            plt.title('Trajectory Smoothing and Filtering')
            plt.legend()
            plt.grid()
            plt.show()

            # warn if the number of points is insufficient
            if len(x_new) < 10:
                self.get_logger().warn('Generated trajectory has fewer points than the required horizon (10).')
def shutdown():
    file.close()
    print('Goodbye')

def main(args=None):
    atexit.register(shutdown)
    print('Saving waypoints...')
    rclpy.init(args=args)
    node = WaypointsLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_and_interpolate()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
