#!/usr/bin/env python3
"""Publish Isaac BlueROV telemetry UDP packets as ROS2 topics."""

from __future__ import annotations

import argparse
import base64
import json
import math
import socket
import struct
import zlib

import rclpy
from geometry_msgs.msg import Point, PoseArray, PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, LaserScan, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, String
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


def _xyz(values) -> tuple[float, float, float]:
    values = values or [0.0, 0.0, 0.0]
    return float(values[0]), float(values[1]), float(values[2])


def _quat_xyzw_from_wxyz(values) -> tuple[float, float, float, float]:
    values = values or [1.0, 0.0, 0.0, 0.0]
    return float(values[1]), float(values[2]), float(values[3]), float(values[0])


def _yaw_from_quat_xyzw(q: tuple[float, float, float, float]) -> float:
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _pointcloud_xyz(stamp, frame_id: str, points: list[tuple[float, float, float]]) -> PointCloud2:
    cloud = PointCloud2()
    cloud.header.stamp = stamp
    cloud.header.frame_id = frame_id
    cloud.height = 1
    cloud.width = len(points)
    cloud.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    cloud.is_bigendian = False
    cloud.point_step = 12
    cloud.row_step = cloud.point_step * cloud.width
    cloud.is_dense = False
    cloud.data = b"".join(struct.pack("<fff", float(x), float(y), float(z)) for x, y, z in points)
    return cloud


def _camera_info(stamp, frame_id: str, width: int, height: int, fov_deg: float) -> CameraInfo:
    info = CameraInfo()
    info.header.stamp = stamp
    info.header.frame_id = frame_id
    info.width = int(width)
    info.height = int(height)
    fx = (0.5 * float(width)) / max(1e-6, math.tan(0.5 * math.radians(float(fov_deg))))
    fy = fx
    cx = 0.5 * (float(width) - 1.0)
    cy = 0.5 * (float(height) - 1.0)
    info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.distortion_model = "plumb_bob"
    info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    return info


class UdpTelemetryPublisher(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("bluerov_udp_telemetry_publisher")
        self.args = args
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.bind((args.udp_host, args.udp_port))

        qos = int(args.qos)
        self.odom_pub = self.create_publisher(Odometry, "bluerov/odom", qos)
        self.pose_pub = self.create_publisher(PoseStamped, "bluerov/pose", qos)
        self.goal_pub = self.create_publisher(PoseStamped, "bluerov/goal", qos)
        self.scan_pub = self.create_publisher(LaserScan, "bluerov/sonar/scan", qos)
        self.sonar_hits_pub = self.create_publisher(PointCloud2, "bluerov/sonar/hits", qos)
        self.sonar_hits_reconstructed_pub = self.create_publisher(
            PointCloud2, "bluerov/sonar/hits_reconstructed", qos
        )
        self.depth_pub = self.create_publisher(Image, "bluerov/depth/image", qos)
        self.depth_info_pub = self.create_publisher(CameraInfo, "bluerov/depth/camera_info", qos)
        self.rgb_pub = self.create_publisher(Image, "bluerov/camera/image_raw", qos)
        self.rgb_info_pub = self.create_publisher(CameraInfo, "bluerov/camera/camera_info", qos)
        self.obstacles_pub = self.create_publisher(PoseArray, "bluerov/obstacles", qos)
        self.obstacle_markers_pub = self.create_publisher(MarkerArray, "bluerov/obstacles/markers", qos)
        self.sonar_rays_pub = self.create_publisher(Marker, "bluerov/sonar/rays", qos)
        self.body_pub = self.create_publisher(Marker, "bluerov/body", qos)
        self.dynamic_estimates_pub = self.create_publisher(
            MarkerArray, "bluerov/dynamic_obstacles_estimated", qos
        )
        self.map_pub = self.create_publisher(OccupancyGrid, "bluerov/map", qos)
        self.map_occ_points_pub = self.create_publisher(PointCloud2, "bluerov/map_occ/points", qos)
        self.map_occ_static_points_pub = self.create_publisher(PointCloud2, "bluerov/map_occ/static_points", qos)
        self.map_occ_dynamic_points_pub = self.create_publisher(PointCloud2, "bluerov/map_occ/dynamic_points", qos)
        self.map_occ_hit_points_pub = self.create_publisher(PointCloud2, "bluerov/map_occ/hit_points", qos)
        self.map_occ_dynamic_hit_points_pub = self.create_publisher(
            PointCloud2, "bluerov/map_occ/dynamic_hit_points", qos
        )
        self.map_occ_scan_origin_pub = self.create_publisher(PointCloud2, "bluerov/map_occ/scan_origin", qos)
        self.map_occ_last_scan_origin_pub = self.create_publisher(
            PointCloud2, "bluerov/map_occ/last_scan_origin", qos
        )
        self.path_pub = self.create_publisher(Path, "bluerov/path", qos)
        self.trajectory_pub = self.create_publisher(Path, "bluerov/trajectory", qos)
        self.status_pub = self.create_publisher(String, "bluerov/status", qos)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.trajectory = Path()
        self.trajectory.header.frame_id = args.frame_id
        self.last_traj_position: tuple[float, float, float] | None = None

        self.create_timer(0.005, self._poll)
        self.create_timer(0.5, self._publish_heartbeat)
        self.get_logger().info(f"listening for Isaac telemetry on udp://{args.udp_host}:{args.udp_port}")

    def _publish_heartbeat(self) -> None:
        status = String()
        status.data = json.dumps({"bridge_alive": True}, separators=(",", ":"))
        self.status_pub.publish(status)

    def _poll(self) -> None:
        while True:
            try:
                data, _ = self.sock.recvfrom(65535)
            except BlockingIOError:
                return
            try:
                payload = json.loads(data.decode("utf-8"))
            except json.JSONDecodeError as exc:
                self.get_logger().warning(f"bad telemetry JSON: {exc}")
                continue
            self._publish_payload(payload)

    def _publish_payload(self, payload: dict) -> None:
        stamp = self.get_clock().now().to_msg()
        frame_id = self.args.frame_id or str(payload.get("frame_id", "map"))
        child_frame_id = self.args.child_frame_id or str(payload.get("child_frame_id", "bluerov_base_link"))
        pose = payload.get("pose", {}) or {}
        position = _xyz(pose.get("position"))
        orientation = _quat_xyzw_from_wxyz(pose.get("orientation_wxyz"))

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = frame_id
        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = position
        (
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w,
        ) = orientation
        self.pose_pub.publish(pose_msg)
        self._publish_trajectory(pose_msg)

        odom = Odometry()
        odom.header = pose_msg.header
        odom.child_frame_id = child_frame_id
        odom.pose.pose = pose_msg.pose
        twist = payload.get("twist", {}) or {}
        linear = _xyz(twist.get("linear_world"))
        angular = _xyz(twist.get("angular_world"))
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = linear
        odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z = angular
        self.odom_pub.publish(odom)

        tf_msg = TransformStamped()
        tf_msg.header = pose_msg.header
        tf_msg.child_frame_id = child_frame_id
        tf_msg.transform.translation.x, tf_msg.transform.translation.y, tf_msg.transform.translation.z = position
        tf_msg.transform.rotation = pose_msg.pose.orientation
        self.tf_broadcaster.sendTransform(tf_msg)
        self._publish_body(stamp, child_frame_id)

        goal = payload.get("goal")
        if goal and goal.get("position") is not None:
            goal_msg = PoseStamped()
            goal_msg.header.stamp = stamp
            goal_msg.header.frame_id = frame_id
            goal_msg.pose.position.x, goal_msg.pose.position.y, goal_msg.pose.position.z = _xyz(goal.get("position"))
            goal_msg.pose.orientation.w = 1.0
            self.goal_pub.publish(goal_msg)

        sonar = payload.get("sonar", {}) or {}
        ranges = sonar.get("ranges")
        if ranges:
            scan = LaserScan()
            scan.header.stamp = stamp
            scan.header.frame_id = child_frame_id
            scan.angle_min = float(sonar.get("angle_min", -3.14159))
            scan.angle_max = float(sonar.get("angle_max", 3.14159))
            scan.range_min = float(sonar.get("range_min", 0.2))
            scan.range_max = float(sonar.get("range_max", 8.0))
            count = max(1, len(ranges))
            scan.angle_increment = (scan.angle_max - scan.angle_min) / max(1, count - 1)
            scan.time_increment = 0.0
            scan.scan_time = 0.0
            scan.ranges = [
                float("inf") if float(v) >= (scan.range_max - 1e-6) else min(scan.range_max, max(scan.range_min, float(v)))
                for v in ranges
            ]
            self.scan_pub.publish(scan)

            yaw = _yaw_from_quat_xyzw(orientation)
            points = []
            px, py, pz = position
            for i, value in enumerate(scan.ranges):
                r = float(value)
                if not math.isfinite(r) or r < scan.range_min or r > scan.range_max:
                    continue
                angle = scan.angle_min + float(i) * scan.angle_increment
                world_angle = yaw + angle
                points.append((px + r * math.cos(world_angle), py + r * math.sin(world_angle), pz))
            self.sonar_hits_reconstructed_pub.publish(_pointcloud_xyz(stamp, frame_id, points))
            self._publish_sonar_rays(stamp, child_frame_id, scan)

        obstacles = payload.get("obstacles", []) or []
        pose_array = PoseArray()
        pose_array.header.stamp = stamp
        pose_array.header.frame_id = frame_id
        for item in obstacles:
            p = PoseStamped().pose
            p.position.x, p.position.y, p.position.z = _xyz(item.get("position"))
            q = _quat_xyzw_from_wxyz(item.get("orientation_wxyz"))
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
            pose_array.poses.append(p)
        self.obstacles_pub.publish(pose_array)
        self._publish_obstacle_markers(stamp, frame_id, pose_array)

        dynamic_estimates = payload.get("dynamic_obstacle_estimates", []) or []
        self._publish_dynamic_estimates(stamp, frame_id, dynamic_estimates)

        depth = payload.get("depth")
        if depth:
            try:
                raw = zlib.decompress(base64.b64decode(depth["zlib_base64"]))
                image = Image()
                image.header.stamp = stamp
                image.header.frame_id = f"{child_frame_id}_depth_camera"
                image.height = int(depth["height"])
                image.width = int(depth["width"])
                image.encoding = str(depth.get("encoding", "32FC1"))
                image.is_bigendian = int(depth.get("is_bigendian", 0))
                image.step = int(depth.get("step", image.width * 4))
                image.data = raw
                self.depth_pub.publish(image)
                self.depth_info_pub.publish(
                    _camera_info(stamp, image.header.frame_id, image.width, image.height, self.args.camera_fov_deg)
                )
            except Exception as exc:
                self.get_logger().warning(f"bad depth packet: {exc}")

        rgb = payload.get("rgb")
        if rgb:
            try:
                raw = zlib.decompress(base64.b64decode(rgb["zlib_base64"]))
                image = Image()
                image.header.stamp = stamp
                image.header.frame_id = f"{child_frame_id}_camera"
                image.height = int(rgb["height"])
                image.width = int(rgb["width"])
                image.encoding = str(rgb.get("encoding", "rgb8"))
                image.is_bigendian = int(rgb.get("is_bigendian", 0))
                image.step = int(rgb.get("step", image.width * 3))
                image.data = raw
                self.rgb_pub.publish(image)
                self.rgb_info_pub.publish(
                    _camera_info(stamp, image.header.frame_id, image.width, image.height, self.args.camera_fov_deg)
                )
            except Exception as exc:
                self.get_logger().warning(f"bad rgb packet: {exc}")

        occ_map = payload.get("map")
        if occ_map:
            try:
                raw = zlib.decompress(base64.b64decode(occ_map["zlib_base64"]))
                grid = OccupancyGrid()
                grid.header.stamp = stamp
                grid.header.frame_id = frame_id
                grid.info.width = int(occ_map["width"])
                grid.info.height = int(occ_map["height"])
                grid.info.resolution = float(occ_map.get("resolution", 0.1))
                origin = _xyz(occ_map.get("origin"))
                grid.info.origin.position.x = origin[0]
                grid.info.origin.position.y = origin[1]
                grid.info.origin.position.z = origin[2]
                grid.info.origin.orientation.w = 1.0
                expected = grid.info.width * grid.info.height
                values = list(raw[:expected])
                if len(values) < expected:
                    values.extend([-1] * (expected - len(values)))
                grid.data = [max(0, min(100, int(v))) for v in values]
                self.map_pub.publish(grid)
            except Exception as exc:
                self.get_logger().warning(f"bad map packet: {exc}")

        occ_debug = payload.get("map_occ_debug")
        if occ_debug:
            self._publish_map_occ_debug(stamp, frame_id, occ_debug)

        path = payload.get("path")
        if path and path.get("points"):
            try:
                path_msg = Path()
                path_msg.header.stamp = stamp
                path_msg.header.frame_id = frame_id
                for point in path.get("points", []):
                    p = PoseStamped()
                    p.header = path_msg.header
                    p.pose.position.x, p.pose.position.y, p.pose.position.z = _xyz(point)
                    p.pose.orientation.w = 1.0
                    path_msg.poses.append(p)
                self.path_pub.publish(path_msg)
            except Exception as exc:
                self.get_logger().warning(f"bad path packet: {exc}")

        status = String()
        status.data = json.dumps(
            {
                "metrics": payload.get("metrics", {}),
                "termination_reason_rates": payload.get("termination_reason_rates", {}),
            },
            separators=(",", ":"),
        )
        self.status_pub.publish(status)

    def _publish_body(self, stamp, frame_id: str) -> None:
        """BlueROV2 hull as a box (0.46 x 0.34 x 0.25 m, x forward), so rviz shows size and heading."""
        m = Marker()
        m.header.stamp, m.header.frame_id = stamp, frame_id
        m.ns, m.id, m.type, m.action = "body", 0, Marker.CUBE, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = 0.46, 0.34, 0.25
        m.color = ColorRGBA(r=0.2, g=0.5, b=1.0, a=0.6)
        self.body_pub.publish(m)

    def _publish_sonar_rays(self, stamp, frame_id: str, scan: LaserScan) -> None:
        """The simulated sonar as it is: one line per ray from the vehicle, in the body frame.
        Hits are opaque and end at the hit, misses are faint and end at range_max."""
        m = Marker()
        m.header.stamp, m.header.frame_id = stamp, frame_id
        m.ns, m.id, m.type, m.action = "sonar_rays", 0, Marker.LINE_LIST, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.008
        hit_color = ColorRGBA(r=1.0, g=0.85, b=0.1, a=0.18)
        miss_color = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.06)
        for i, value in enumerate(scan.ranges):
            hit = math.isfinite(value)
            r = value if hit else scan.range_max
            a = scan.angle_min + i * scan.angle_increment
            m.points += [Point(), Point(x=r * math.cos(a), y=r * math.sin(a))]
            m.colors += [hit_color if hit else miss_color] * 2
        self.sonar_rays_pub.publish(m)

    def _publish_obstacle_markers(self, stamp, frame_id: str, pose_array: PoseArray) -> None:
        """Ground-truth obstacles at their true size (cylinders, task.obstacles.radius/height)."""
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        d, h = 2.0 * float(self.args.obstacle_radius), float(self.args.obstacle_height)
        for i, pose in enumerate(pose_array.poses):
            m = Marker()
            m.header.stamp, m.header.frame_id = stamp, frame_id
            m.ns, m.id, m.type, m.action = "obstacles_gt", i, Marker.CYLINDER, Marker.ADD
            m.pose = pose
            m.scale.x, m.scale.y, m.scale.z = d, d, h
            m.color = ColorRGBA(r=0.9, g=0.25, b=0.25, a=0.35)
            markers.markers.append(m)
        self.obstacle_markers_pub.publish(markers)

    def _publish_dynamic_estimates(self, stamp, frame_id: str, estimates: list[dict]) -> None:
        markers = MarkerArray()
        clear = Marker()
        clear.header.stamp = stamp
        clear.header.frame_id = frame_id
        clear.ns = "bluerov_dynamic_estimates"
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        mid = 0
        for item in estimates:
            pos = _xyz(item.get("position"))
            vel = _xyz(item.get("velocity"))
            speed = math.sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2])
            is_dynamic = bool(item.get("is_dynamic", False))
            oid = int(item.get("id", mid))
            if (not is_dynamic) and bool(self.args.dynamic_markers_dynamic_only):
                continue

            sphere = Marker()
            sphere.header.stamp = stamp
            sphere.header.frame_id = frame_id
            sphere.ns = "bluerov_dynamic_estimates_pose"
            sphere.id = mid
            mid += 1
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x, sphere.pose.position.y, sphere.pose.position.z = pos
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.22
            sphere.scale.y = 0.22
            sphere.scale.z = 0.22
            sphere.color.a = float(self.args.dynamic_marker_alpha)
            if is_dynamic:
                sphere.color.r = 1.0
                sphere.color.g = 0.45
                sphere.color.b = 0.05
            else:
                sphere.color.r = 0.55
                sphere.color.g = 0.55
                sphere.color.b = 0.55
            markers.markers.append(sphere)

            arrow = Marker()
            arrow.header.stamp = stamp
            arrow.header.frame_id = frame_id
            arrow.ns = "bluerov_dynamic_estimates_velocity"
            arrow.id = mid
            mid += 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.scale.x = 0.05
            arrow.scale.y = 0.12
            arrow.scale.z = 0.12
            arrow.color.a = 0.95 if is_dynamic else 0.35
            arrow.color.r = 0.1
            arrow.color.g = 0.9 if is_dynamic else 0.55
            arrow.color.b = 1.0 if is_dynamic else 0.55
            p0 = Point()
            p0.x, p0.y, p0.z = pos
            p1 = Point()
            scale = float(self.args.dynamic_arrow_scale) if speed > 1e-4 else 0.0
            p1.x = pos[0] + vel[0] * scale
            p1.y = pos[1] + vel[1] * scale
            p1.z = pos[2] + vel[2] * scale
            arrow.points = [p0, p1]
            markers.markers.append(arrow)

            if not bool(self.args.dynamic_labels):
                continue
            label = Marker()
            label.header.stamp = stamp
            label.header.frame_id = frame_id
            label.ns = "bluerov_dynamic_estimates_label"
            label.id = mid
            mid += 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = pos[0]
            label.pose.position.y = pos[1]
            label.pose.position.z = pos[2] + float(self.args.dynamic_label_z_offset)
            label.pose.orientation.w = 1.0
            label.scale.z = float(self.args.dynamic_label_size)
            label.color.a = float(self.args.dynamic_label_alpha)
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            source = str(item.get("source", "estimate"))
            if str(self.args.dynamic_label_mode).strip().lower() == "none":
                continue
            if str(self.args.dynamic_label_mode).strip().lower() == "full":
                label.text = f"id={oid} v={float(item.get('speed', speed)):.2f} {source}"
            else:
                prefix = "D" if is_dynamic else "S"
                label.text = f"{prefix}{oid} {float(item.get('speed', speed)):.2f}"
            markers.markers.append(label)

        self.dynamic_estimates_pub.publish(markers)

    def _publish_map_occ_debug(self, stamp, frame_id: str, occ_debug: dict) -> None:
        try:
            layers = occ_debug.get("layers", {}) or {}
            threshold = int(occ_debug.get("threshold", 5))
            x_min = float(occ_debug.get("x_min", 0.0))
            x_max = float(occ_debug.get("x_max", 1.0))
            y_min = float(occ_debug.get("y_min", 0.0))
            y_max = float(occ_debug.get("y_max", 1.0))
            pubs = {
                "occ": self.map_occ_points_pub,
                "static": self.map_occ_static_points_pub,
                "dynamic": self.map_occ_dynamic_points_pub,
            }
            for name, pub in pubs.items():
                layer = layers.get(name)
                if not layer:
                    continue
                raw = zlib.decompress(base64.b64decode(layer["zlib_base64"]))
                width = int(layer["width"])
                height = int(layer["height"])
                expected = width * height
                if width <= 0 or height <= 0:
                    continue
                values = raw[:expected]
                dx = (x_max - x_min) / max(1, width - 1)
                dy = (y_max - y_min) / max(1, height - 1)
                points = []
                for iy in range(height):
                    row = iy * width
                    y = y_min + float(iy) * dy
                    for ix in range(width):
                        if int(values[row + ix]) < threshold:
                            continue
                        x = x_min + float(ix) * dx
                        points.append((x, y, 0.03))
                pub.publish(_pointcloud_xyz(stamp, frame_id, points))
            hit_points = [
                (float(p[0]), float(p[1]), 0.08)
                for p in occ_debug.get("hit_points", [])
                if p is not None and len(p) >= 2
            ]
            self.map_occ_hit_points_pub.publish(_pointcloud_xyz(stamp, frame_id, hit_points))
            if hit_points:
                self.sonar_hits_pub.publish(_pointcloud_xyz(stamp, frame_id, hit_points))
            dyn_hit_points = [
                (float(p[0]), float(p[1]), 0.12)
                for p in occ_debug.get("dynamic_hit_points", [])
                if p is not None and len(p) >= 2
            ]
            self.map_occ_dynamic_hit_points_pub.publish(_pointcloud_xyz(stamp, frame_id, dyn_hit_points))
            scan_origin = occ_debug.get("scan_origin")
            if scan_origin and len(scan_origin) >= 2:
                z = float(scan_origin[2]) if len(scan_origin) >= 3 else 0.0
                self.map_occ_scan_origin_pub.publish(
                    _pointcloud_xyz(stamp, frame_id, [(float(scan_origin[0]), float(scan_origin[1]), z)])
                )
            last_scan_origin = occ_debug.get("last_scan_origin")
            if last_scan_origin and len(last_scan_origin) >= 2:
                z = float(last_scan_origin[2]) if len(last_scan_origin) >= 3 else 0.0
                self.map_occ_last_scan_origin_pub.publish(
                    _pointcloud_xyz(
                        stamp,
                        frame_id,
                        [(float(last_scan_origin[0]), float(last_scan_origin[1]), z)],
                    )
                )
        except Exception as exc:
            self.get_logger().warning(f"bad map_occ_debug packet: {exc}")

    def _publish_trajectory(self, pose_msg: PoseStamped) -> None:
        p = pose_msg.pose.position
        pos = (float(p.x), float(p.y), float(p.z))
        if self.last_traj_position is not None:
            dx = pos[0] - self.last_traj_position[0]
            dy = pos[1] - self.last_traj_position[1]
            dz = pos[2] - self.last_traj_position[2]
            if math.sqrt(dx * dx + dy * dy + dz * dz) < self.args.trajectory_min_step:
                return
        traj_pose = PoseStamped()
        traj_pose.header = pose_msg.header
        traj_pose.pose = pose_msg.pose
        self.trajectory.header = pose_msg.header
        self.trajectory.poses.append(traj_pose)
        max_len = max(1, int(self.args.trajectory_max_points))
        if len(self.trajectory.poses) > max_len:
            self.trajectory.poses = self.trajectory.poses[-max_len:]
        self.last_traj_position = pos
        self.trajectory_pub.publish(self.trajectory)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=15010)
    parser.add_argument("--qos", type=int, default=10)
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--child-frame-id", default="bluerov_base_link")
    parser.add_argument("--camera-fov-deg", type=float, default=90.0)
    parser.add_argument("--trajectory-max-points", type=int, default=3000)
    parser.add_argument("--trajectory-min-step", type=float, default=0.03)
    parser.add_argument("--dynamic-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic-label-mode", choices=["compact", "full", "none"], default="compact")
    parser.add_argument("--dynamic-label-size", type=float, default=0.16)
    parser.add_argument("--dynamic-label-alpha", type=float, default=0.65)
    parser.add_argument("--dynamic-label-z-offset", type=float, default=0.22)
    parser.add_argument("--dynamic-marker-alpha", type=float, default=0.75)
    parser.add_argument("--dynamic-arrow-scale", type=float, default=0.70)
    parser.add_argument("--obstacle-radius", type=float, default=0.25, help="task.obstacles.radius")
    parser.add_argument("--obstacle-height", type=float, default=0.5, help="task.obstacles.height")
    parser.add_argument("--dynamic-markers-dynamic-only", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = UdpTelemetryPublisher(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
