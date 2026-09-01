#!/usr/bin/env python3
"""Publish keyboard commands as sensor_msgs/Joy for BlueROV teleoperation."""

from __future__ import annotations

import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy


HELP = """
Keyboard Joy controls:
  w/s : surge forward/back
  a/d : sway left/right
  r/f : heave up/down
  q/e : yaw left/right
  x   : reset button pulse
  space: stop
  Ctrl-C: quit
"""


class KeyboardJoy(Node):
    def __init__(self) -> None:
        super().__init__("bluerov_keyboard_joy")
        self.pub = self.create_publisher(Joy, "/joy", 10)
        self.axes = [0.0] * 8
        self.buttons = [0] * 12
        self.reset_pulse = 0
        self.timer = self.create_timer(0.05, self._publish)
        self.get_logger().info(HELP)

    def handle_key(self, key: str) -> None:
        step = 1.0
        if key == "w":
            self.axes[1] = step
        elif key == "s":
            self.axes[1] = -step
        elif key == "a":
            self.axes[0] = step
        elif key == "d":
            self.axes[0] = -step
        elif key == "r":
            self.axes[4] = step
        elif key == "f":
            self.axes[4] = -step
        elif key == "q":
            self.axes[3] = step
        elif key == "e":
            self.axes[3] = -step
        elif key == "x":
            self.reset_pulse = 2
        elif key == " ":
            self.axes = [0.0] * 8
            self.buttons[1] = 1

    def release_stop(self) -> None:
        self.buttons[1] = 0

    def _publish(self) -> None:
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.axes = list(self.axes)
        msg.buttons = list(self.buttons)
        if self.reset_pulse > 0:
            msg.buttons[7] = 1
            self.reset_pulse -= 1
        self.pub.publish(msg)
        self.release_stop()


def _read_key(timeout: float = 0.02) -> str | None:
    readable, _, _ = select.select([sys.stdin], [], [], timeout)
    if not readable:
        return None
    return sys.stdin.read(1)


def main() -> None:
    old_settings = termios.tcgetattr(sys.stdin)
    rclpy.init()
    node = KeyboardJoy()
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            key = _read_key()
            if key:
                node.handle_key(key)
            rclpy.spin_once(node, timeout_sec=0.02)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
