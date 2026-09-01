#!/usr/bin/env python3
"""Bridge ROS2 sensor_msgs/Joy to the Isaac debug launcher over UDP."""

from __future__ import annotations

import argparse
import json
import socket
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Joy


def _axis(msg: Joy, index: int) -> float:
    if index < 0 or index >= len(msg.axes):
        return 0.0
    return float(msg.axes[index])


def _button(msg: Joy, index: int) -> bool:
    if index < 0 or index >= len(msg.buttons):
        return False
    return bool(msg.buttons[index])


def _deadzone(value: float, deadzone: float) -> float:
    if abs(value) < deadzone:
        return 0.0
    return value


class JoyToUdp(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("bluerov_joy_to_udp")
        self.args = args
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target = (args.udp_host, args.udp_port)
        self.last_log_time = 0.0
        self.create_subscription(Joy, args.topic, self._on_joy, 10)
        self.get_logger().info(
            f"subscribing {args.topic}; sending actions to udp://{args.udp_host}:{args.udp_port}"
        )

    def _on_joy(self, msg: Joy) -> None:
        enabled = True
        if self.args.enable_button >= 0:
            enabled = _button(msg, self.args.enable_button)

        surge = _deadzone(_axis(msg, self.args.axis_surge), self.args.deadzone) * self.args.scale_surge
        sway = _deadzone(_axis(msg, self.args.axis_sway), self.args.deadzone) * self.args.scale_sway
        heave = _deadzone(_axis(msg, self.args.axis_heave), self.args.deadzone) * self.args.scale_heave
        yaw = _deadzone(_axis(msg, self.args.axis_yaw), self.args.deadzone) * self.args.scale_yaw

        if self.args.invert_surge:
            surge *= -1.0
        if self.args.invert_sway:
            sway *= -1.0
        if self.args.invert_heave:
            heave *= -1.0
        if self.args.invert_yaw:
            yaw *= -1.0
        if not enabled:
            surge = sway = heave = yaw = 0.0

        payload = {
            "stamp": time.time(),
            "action": [
                max(-1.0, min(1.0, surge)),
                max(-1.0, min(1.0, sway)),
                max(-1.0, min(1.0, heave)),
                max(-1.0, min(1.0, yaw)),
            ],
            "buttons": {
                "reset": _button(msg, self.args.reset_button),
                "stop": _button(msg, self.args.stop_button),
            },
        }
        self.sock.sendto(json.dumps(payload).encode("utf-8"), self.target)

        now = time.time()
        if now - self.last_log_time >= 1.0:
            self.last_log_time = now
            self.get_logger().info(
                "action=[{:.2f}, {:.2f}, {:.2f}, {:.2f}] enabled={} reset={} stop={}".format(
                    *payload["action"],
                    enabled,
                    payload["buttons"]["reset"],
                    payload["buttons"]["stop"],
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/joy")
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=15000)
    parser.add_argument("--deadzone", type=float, default=0.08)

    parser.add_argument("--axis-surge", type=int, default=1, help="forward/back axis")
    parser.add_argument("--axis-sway", type=int, default=0, help="left/right strafe axis")
    parser.add_argument("--axis-heave", type=int, default=4, help="up/down axis")
    parser.add_argument("--axis-yaw", type=int, default=3, help="yaw axis")

    parser.add_argument("--scale-surge", type=float, default=0.45)
    parser.add_argument("--scale-sway", type=float, default=0.35)
    parser.add_argument("--scale-heave", type=float, default=0.30)
    parser.add_argument("--scale-yaw", type=float, default=0.45)

    parser.add_argument("--invert-surge", action="store_true")
    parser.add_argument("--invert-sway", action="store_true")
    parser.add_argument("--invert-heave", action="store_true")
    parser.add_argument("--invert-yaw", action="store_true")

    parser.add_argument("--enable-button", type=int, default=-1, help="-1 means always enabled")
    parser.add_argument("--reset-button", type=int, default=7)
    parser.add_argument("--stop-button", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = JoyToUdp(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
