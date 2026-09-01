#!/usr/bin/env python3
"""Forward RViz /goal_pose PoseStamped messages to Isaac over UDP."""

from __future__ import annotations

import argparse
import json
import socket
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class GoalToUdp(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("bluerov_goal_to_udp")
        self.args = args
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target = (args.udp_host, args.udp_port)
        self.create_subscription(PoseStamped, args.topic, self._on_goal, 10)
        self.get_logger().info(
            f"subscribing {args.topic}; sending goals to udp://{args.udp_host}:{args.udp_port}"
        )

    def _on_goal(self, msg: PoseStamped) -> None:
        z = float(msg.pose.position.z)
        if self.args.force_z is not None:
            z = float(self.args.force_z)
        payload = {
            "stamp": time.time(),
            "frame_id": msg.header.frame_id or "map",
            "position": [
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                z,
            ],
        }
        self.sock.sendto(json.dumps(payload).encode("utf-8"), self.target)
        self.get_logger().info(
            "goal=[{:.2f}, {:.2f}, {:.2f}]".format(*payload["position"])
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/goal_pose")
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=15020)
    parser.add_argument("--force-z", type=float, default=-0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = GoalToUdp(args)
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
