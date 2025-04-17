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

        self.subscription_odom = self.create_subscription(
            Odometry, '/pf/pose/odom', self.process_odometry, 10)
        self.subscription_scan = self.create_subscription(
            LaserScan, '/scan', self.process_scan, 10)

        self.latest_scan = None
        self.latest_odometry = None
        self.left_width = float('inf')
        self.right_width = float('inf')
        self.previous_point = None
        self.waypoints = []

    def process_scan(self, scan_data):
        """Compute distance to left/right boundaries using fixed LiDAR angles."""
        ranges = np.array(scan_data.ranges)
        angles = np.linspace(scan_data.angle_min, scan_data.angle_max, len(ranges))
        angles_deg = np.degrees(angles)

        # Find indices closest to -90° (right) and +90° (left)
        idx_right = np.argmin(np.abs(angles_deg + 90))
        idx_left = np.argmin(np.abs(angles_deg - 90))

        # Use those beams to get boundary distances
        if 0 <= idx_right < len(ranges):
            right_distance = ranges[idx_right]
            self.right_width = min(right_distance, 1.0) if not np.isinf(right_distance) and not np.isnan(right_distance) else 1.0
        if 0 <= idx_left < len(ranges):
            left_distance = ranges[idx_left]
            self.left_width = min(left_distance, 1.0) if not np.isinf(left_distance) and not np.isnan(left_distance) else 1.0

        self.get_logger().info(f'Left width: {self.left_width:.2f} m, Right width: {self.right_width:.2f} m')

        self.latest_scan = True
        self.save_waypoint()

    def process_odometry(self, odometry_data):
        self.latest_odometry = odometry_data
        self.save_waypoint()

    def save_waypoint(self):
        if self.latest_scan and self.latest_odometry:
            data = self.latest_odometry
            quaternion = np.array([
                data.pose.pose.orientation.x, 
                data.pose.pose.orientation.y, 
                data.pose.pose.orientation.z, 
                data.pose.pose.orientation.w])
            euler = euler_from_quaternion(quaternion)
            x, y = data.pose.pose.position.x, data.pose.pose.position.y

            if self.previous_point is None or LA.norm([x - self.previous_point[0], y - self.previous_point[1]]) >= MIN_SPACING:
                self.get_logger().info(f'Saving waypoint: x={x}, y={y}, left_width={self.left_width}, right_width={self.right_width}')
                self.waypoints.append((x, y, self.left_width, self.right_width))
                self.previous_point = (x, y)

            self.latest_scan = None
            self.latest_odometry = None

    def filter_outliers(self, points, threshold=1.5):
        filtered = [points[0]]
        for i in range(1, len(points)):
            if euclidean(points[i], filtered[-1]) < threshold:
                filtered.append(points[i])
        return np.array(filtered)

    def save_and_interpolate(self):
        if len(self.waypoints) > 1:
            waypoints_array = np.array(self.waypoints)
            x, y = waypoints_array[:, 0], waypoints_array[:, 1]
            right_widths = waypoints_array[:, 2]
            left_widths = waypoints_array[:, 3]

            waypoints_filtered = self.filter_outliers(waypoints_array[:, :2], threshold=2.0)
            x, y = waypoints_filtered[:, 0], waypoints_filtered[:, 1]

            # Interpolate X-Y
            tck, u = splprep([x, y], s=0.2)
            unew = np.linspace(0, 1, 5000)
            x_new, y_new = splev(unew, tck)

            # Interpolate widths along u (parameterization)
            right_interp = np.clip(np.interp(unew, u, right_widths), 0.0, 1.0)
            left_interp = np.clip(np.interp(unew, u, left_widths), 0.0, 1.0)

            for i in range(len(x_new)):
                file.write(f'{x_new[i]}, {y_new[i]}, {right_interp[i]}, {left_interp[i]}\n')

            self.get_logger().info('Waypoints filtered, smoothed, and saved.')
            self.get_logger().info(f'Length of x_new: {len(x_new)}')

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
